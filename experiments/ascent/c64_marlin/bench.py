import hashlib, json, os, statistics, time
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
from torch import nn
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn
from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod, _build_fp8_nvfp4_decode_copy
# Idea 33 (branch 26): FP8-origin linears at decode rows 33-64 (c64) and 128 (c128). Brace-free so
# the same text runs in both trees via python -c. Synthetic FP8 per-tensor linears of the four real
# shapes with main's load-time copies (W4A16 Marlin decode copy + GEMV copy + W4A4 prefill copy;
# attn qkv with the Q+gate split), R rotated copies per shape (cold L2) and a 512 MB L2 flush
# before every timed replay. Arms per (shape, M), each one CUDA graph over the R modules:
#   tree   = module.quant_method.apply on a BF16 input (what the tree dispatches);
#   marlin = explicit marlin_nvfp4_gemm on the decode copy (workspace_mode 2 = FP32 reduce);
#   w4a4   = explicit prefill_nvfp4_apply (fp4_quantize + nvfp4_gemm [+ K/V FP8 GEMM + copy]);
#   gemm4  = nvfp4_gemm alone on a pre-quantized input (what (5120,6144) runs in the model, where
#            the producer - GDN gated norm / attn output gate - emits the NVFP4 activation).
# Last stdout line: pair JSON, batch = tree arm at M in BATCH (step-weighted ms over the 4 shapes).
T0 = time.time()
dev = "cuda"
WARM, ITERS = int(os.environ.get("BENCH_WARM", "10")), int(os.environ.get("BENCH_ITERS", "60"))
MS = [int(v) for v in os.environ.get("BENCH_MS", "33,48,56,64,128").split(",")]
ARMS = os.environ.get("BENCH_ARMS", "tree,marlin,w4a4,gemm4").split(",")
BATCH = [int(v) for v in os.environ.get("BENCH_BATCH", "64,128").split(",")]
ONLY = os.environ.get("BENCH_SHAPES", "")
# name, N, K, layer count per step, rotation copies, Q+gate split
SHAPES = [("qkvz", 16384, 5120, 48, 4, None),
          ("gdn_out", 5120, 6144, 48, 8, None),
          ("qkv", 14336, 5120, 16, 4, 12288),
          ("o_proj", 5120, 6144, 16, 8, None)]
if ONLY:
    SHAPES = [s for s in SHAPES if s[0] in ONLY.split(",")]
def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
gen = torch.Generator(device=dev)
gen.manual_seed(1234)
def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, device=dev, generator=gen) * scale).to(torch.bfloat16)
class Lin(nn.Module):
    def forward(self, x):
        return self.quant_method.apply(self, x, None)
def make_lin(n, k, in_scale, split):
    W = rnd(n, k, scale=0.02)
    ws = (W.float().abs().amax() / 448.0).reshape(())
    m = Lin()
    m.quant_method = FP8QDQLinearMethod()
    m.weight = nn.Parameter((W.float() / ws).to(torch.float8_e4m3fn), requires_grad=False)
    m.weight_scale = nn.Parameter(ws.float().clone(), requires_grad=False)
    m.input_scale = nn.Parameter(torch.tensor(in_scale, device=dev), requires_grad=False)
    m.force_dynamic_quantization = False
    m.enable_cuda_core = True
    m.all_reduce = None
    m.mapping = None
    m.dtype = torch.bfloat16
    m.out_features, m.in_features = n, k
    m.bias = None
    m._nvfp4_decode_copy = True
    _build_fp8_nvfp4_decode_copy(m)
    env = fpn.ENV_ATTN_QG if split else fpn.ENV_GDN_IN_PROJ_QKVZ
    assert fpn.maybe_build_prefill_nvfp4_copy(m, env)
    if split:
        fpn.get_prefill_nvfp4_copy(m).split = split
    del W
    return m
def marlin(m, x):
    return torch.ops.trtllm.marlin_nvfp4_gemm(
        x, m._nvfp4_decode_weight, scale_a=None, scale_b=m._nvfp4_decode_scale, alpha=None,
        weight_global_scale=m._nvfp4_decode_global_scale, bias=None,
        out_dtype=torch.bfloat16, size_n=m.out_features, size_k=m.in_features,
        workspace_mode=2 if x.shape[0] > 16 else 0)
def w4a4(m, x):
    return fpn.prefill_nvfp4_apply(fpn.get_prefill_nvfp4_copy(m), m, x, x.shape, None)
QX = dict()
def gemm4(m, x):
    c = fpn.get_prefill_nvfp4_copy(m)
    key = (id(c), x.shape[0])
    if key not in QX:
        QX[key] = torch.ops.trtllm.fp4_quantize(x, c.act_global, 16, False)
    a, sf = QX[key]
    return torch.ops.trtllm.nvfp4_gemm(a, c.w_fp4, sf, c.w_sf, c.alpha, torch.bfloat16,
                                       allowed_backends="cutlass,cublaslt,cuda_core")
def tree(m, x):
    return m(x)
FN = dict(tree=tree, marlin=marlin, w4a4=w4a4, gemm4=gemm4)
def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
log("gpu", torch.cuda.get_device_name(), "PDL", os.environ.get("TRTLLM_ENABLE_PDL"), "MS", MS, "arms", ARMS)
per_iter = dict((a, dict((M, [0.0] * ITERS) for M in MS)) for a in ARMS)
FLUSH = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)
detail = dict()
hashes = dict()
for sname, n, k, cnt, R, split in SHAPES:
    xs = dict((M, rnd(M, k, scale=1.5)) for M in MS)
    s_in = 1.1 * max(float(x.float().abs().amax()) for x in xs.values()) / 448
    mods = [make_lin(n, k, s_in, split) for _ in range(R)]
    log("shape %s (%d,%d) x%d built; marlin cap %s; mem %.1f GiB" % (
        sname, n, k, R, mods[0].__dict__.get("_nvfp4_decode_max_rows"), torch.cuda.memory_allocated() / 2**30))
    for M in MS:
        x = xs[M]
        with autotune():
            for a in ARMS:
                FN[a](mods[0], x)
        torch.cuda.synchronize()
        graphs, outs = dict(), dict()
        d = dict()
        detail["%s_M%d" % (sname, M)] = d
        for a in ARMS:
            f = FN[a]
            for m in mods:
                if a == "gemm4":
                    f(m, x)
            s0 = torch.cuda.Stream()
            s0.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s0):
                for _ in range(2):
                    for m in mods:
                        f(m, x)
            torch.cuda.current_stream().wait_stream(s0)
            torch.cuda.synchronize()
            c0 = fpn.prefill_nvfp4_stats()["calls"]
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                o = [f(m, x) for m in mods]
            d[a + "_w4a4_calls"] = fpn.prefill_nvfp4_stats()["calls"] - c0
            graphs[a], outs[a] = g, o
            for _ in range(WARM):
                g.replay()
        torch.cuda.synchronize()
        evs = dict((a, []) for a in ARMS)
        for it in range(ITERS):
            for a in ARMS:
                FLUSH.zero_()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[a].replay()
                e1.record()
                evs[a].append((e0, e1))
        torch.cuda.synchronize()
        for a in ARMS:
            ts = [e0.elapsed_time(e1) / R for e0, e1 in evs[a]]
            for i, t in enumerate(ts):
                per_iter[a][M][i] += cnt * t
            st = sorted(ts)
            d[a + "_us"] = round(statistics.median(st) * 1e3, 2)
            d[a + "_p90_us"] = round(st[int(0.9 * len(st)) - 1] * 1e3, 2)
            hashes["%s_M%d_%s" % (sname, M, a)] = h(torch.cat(outs[a]))
        if "tree" in ARMS and "marlin" in ARMS:
            d["tree_eq_marlin"] = bool(torch.equal(torch.cat(outs["tree"]), torch.cat(outs["marlin"])))
        if "tree" in ARMS and "w4a4" in ARMS:
            d["tree_eq_w4a4"] = bool(torch.equal(torch.cat(outs["tree"]), torch.cat(outs["w4a4"])))
        log("%s M=%d" % (sname, M), json.dumps(d))
        del graphs, outs
        QX.clear()
        torch.cuda.synchronize()
    del mods, xs
    torch.cuda.empty_cache()
summary = dict()
batch = dict()
for a in ARMS:
    for M in MS:
        st = sorted(per_iter[a][M])
        med = statistics.median(st)
        p10, p90 = st[int(0.1 * len(st))], st[int(0.9 * len(st)) - 1]
        summary["%s_M%d" % (a, M)] = dict(step_ms=round(med, 4), p90_ms=round(p90, 4), nf=round((p90 - p10) / med, 4))
        if a == "tree" and M in BATCH:
            batch[str(M)] = dict(kernel_ms=med, noise_floor=(p90 - p10) / med, p90_ms=p90)
log("step-weighted FP8-origin linear ms:", json.dumps(summary))
print(json.dumps(dict(batch=batch, output=dict(**dict(summary=summary, detail=detail, hashes=hashes),
                                               **dict([("class", "precision")])))), flush=True)
