import hashlib, json, os, statistics, time
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
from torch import nn
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn
from tensorrt_llm._torch.modules import linear as L
# Idea 55 (branch 36) pair bench: FP8-origin linears at decode rows, synthetic FP8 per-tensor
# linears of the real shapes with the tree's load-time copies, dispatched by
# module.quant_method.apply (what the tree runs). BRACE-FREE so the same text runs in both trees
# via python -c. Cold L2: R rotated modules per shape in one CUDA graph (PDL on), >= 400 MB of
# routed weights per graph. Arms (rotated order, one process): tree = default env; alt =
# TRTLLM_MARLIN_M32_TABLE=0 (branch 18's generic rows 17-32 launch in the candidate tree; an A/A
# copy in the base tree). Optional diagnostic arms via BENCH_ARMS: marlin (explicit
# marlin_nvfp4_gemm, workspace_mode as apply passes it at rows > 16), w4a4 (explicit
# prefill_nvfp4_apply). Per-step time = sum over shapes of calls/step x per-call time, per
# iteration index. Last stdout line: pair JSON, batch 8/16/32 = tree arm (step-weighted ms).
T0 = time.time()
dev = "cuda"
WARM, ITERS = 10, int(os.environ.get("BENCH_ITERS", "60"))
MS = [int(v) for v in os.environ.get("BENCH_MS", "8,16,17,24,32").split(",")]
ARMS = os.environ.get("BENCH_ARMS", "tree,alt").split(",")
ONLY = os.environ.get("BENCH_SHAPES", "")
# name, N, K, calls per step, rotated modules, Q+gate split
SHAPES = [("qkvz", 16384, 5120, 48, 9, None),
          ("out_oproj", 5120, 6144, 64, 24, None),
          ("qkv", 14336, 5120, 16, 10, 12288)]
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
    m.quant_method = L.FP8QDQLinearMethod()
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
    L._build_fp8_nvfp4_decode_copy(m)
    env = fpn.ENV_ATTN_QG if split else fpn.ENV_GDN_IN_PROJ_QKVZ
    assert fpn.maybe_build_prefill_nvfp4_copy(m, env)
    if split:
        fpn.get_prefill_nvfp4_copy(m).split = split
    del W
    return m
def marlin(m, x):
    mode = 2 if x.shape[0] > 16 else 0
    return torch.ops.trtllm.marlin_nvfp4_gemm(
        x, m._nvfp4_decode_weight, scale_a=None, scale_b=m._nvfp4_decode_scale, alpha=None,
        weight_global_scale=m._nvfp4_decode_global_scale, bias=None,
        out_dtype=torch.bfloat16, size_n=m.out_features, size_k=m.in_features, workspace_mode=mode)
def w4a4(m, x):
    return fpn.prefill_nvfp4_apply(fpn.get_prefill_nvfp4_copy(m), m, x, x.shape, None)
def tree(m, x):
    return m(x)
FN = dict(tree=tree, alt=tree, marlin=marlin, w4a4=w4a4)
def set_env(a):
    os.environ.pop("TRTLLM_MARLIN_M32_TABLE", None)
    if a == "alt":
        os.environ["TRTLLM_MARLIN_M32_TABLE"] = "0"
def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
p = torch.cuda.get_device_properties(0)
log("gpu", p.name, "sms", p.multi_processor_count, "PDL", os.environ.get("TRTLLM_ENABLE_PDL"), "MS", MS, "arms", ARMS)
log("tree caps", json.dumps(dict(("%d,%d" % kk, v) for kk, v in getattr(L, "_FP8_NVFP4_DECODE_ROW_CAPS", dict()).items())))
log("has M32 table src", os.path.exists("cpp/tensorrt_llm/kernels/marlin/marlin_nvfp4_gemm.cu")
    and "kM32Table" in open("cpp/tensorrt_llm/kernels/marlin/marlin_nvfp4_gemm.cu").read())
per_iter = dict((a, dict((M, [0.0] * ITERS) for M in MS)) for a in ARMS)
detail = dict()
hashes = dict()
for sname, n, k, cnt, R, split in SHAPES:
    if ONLY and sname not in ONLY.split(","):
        continue
    xs = dict((M, rnd(M, k, scale=1.5)) for M in MS)
    s_in = 1.1 * max(float(x.float().abs().amax()) for x in xs.values()) / 448
    mods = [make_lin(n, k, s_in, split) for _ in range(R)]
    log("shape %s (%d,%d) x%d built; cap %d; mem %.1f GiB" % (
        sname, n, k, R, L.fp8_nvfp4_decode_max_rows(mods[0]), torch.cuda.memory_allocated() / 2**30))
    for M in MS:
        x = xs[M]
        with autotune():
            for a in ARMS:
                set_env(a)
                FN[a](mods[0], x)
        set_env("tree")
        torch.cuda.synchronize()
        graphs, outs = dict(), dict()
        d = dict()
        detail["%s_M%d" % (sname, M)] = d
        for a in ARMS:
            set_env(a)
            f = FN[a]
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
        set_env("tree")
        for a in ARMS:
            for _ in range(WARM):
                graphs[a].replay()
        torch.cuda.synchronize()
        evs = dict((a, []) for a in ARMS)
        for it in range(ITERS):
            r = it % len(ARMS)
            order = ARMS[r:] + ARMS[:r]
            if it % 2:
                order = order[::-1]
            for a in order:
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
            d[a + "_p10_us"] = round(st[int(0.1 * len(st))] * 1e3, 2)
            d[a + "_p90_us"] = round(st[int(0.9 * len(st)) - 1] * 1e3, 2)
            hashes["%s_M%d_%s" % (sname, M, a)] = h(torch.cat(outs[a]))
        for a in ARMS:
            if a != "tree" and "tree" in ARMS:
                d["tree_eq_" + a] = bool(torch.equal(torch.cat(outs["tree"]), torch.cat(outs[a])))
        log("%s M=%d" % (sname, M), json.dumps(d))
        del graphs, outs
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
        if a == "tree" and M in (8, 16, 32):
            batch[str(M)] = dict(kernel_ms=med, noise_floor=(p90 - p10) / med, p90_ms=p90)
log("step-weighted FP8-origin linear ms:", json.dumps(summary))
print(json.dumps(dict(batch=batch, output=dict(summary=summary, detail=detail, hashes=hashes,
                                               **dict([("class", "precision")])))), flush=True)
