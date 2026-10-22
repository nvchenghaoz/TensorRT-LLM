"""Idea 10 (branch 18): FP8-origin linears at decode rows 17-32 (c32).

Synthetic FP8 per-tensor linears of the four real shapes, each with main's
load-time copies (W4A16 Marlin decode copy + GEMV copy + W4A4 prefill copy,
attn qkv with the Q+gate split), R rotated copies per shape (cold L2).
Arms per (shape, M), each one CUDA graph chaining the R modules:
  tree   module.quant_method.apply (what the tree dispatches: base -> W4A4 at
         M > 16, branch -> Marlin up to its per-shape cap);
  marlin explicit marlin_nvfp4_gemm on the decode copy;
  w4a4   explicit prefill_nvfp4_apply (fp4_quantize + nvfp4_gemm [+ K/V FP8]).
Per-step time = sum over shapes of layer_count x per-layer time, computed per
iteration index. Last stdout line: pair JSON ("batch" 8/16/32 = tree arm at
M = 8/16/32, step-weighted ms of the FP8-origin linears).
"""
import hashlib
import json
import os
import statistics
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402
from torch import nn  # noqa: E402

from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn  # noqa: E402
from tensorrt_llm._torch.modules.linear import (  # noqa: E402
    FP8QDQLinearMethod, _build_fp8_nvfp4_decode_copy)

T0 = time.time()
dev = "cuda"
WARM, ITERS = 10, 60
MS = [int(v) for v in os.environ.get("BENCH_MS", "8,16,17,24,32").split(",")]
ARMS = os.environ.get("BENCH_ARMS", "tree,marlin,w4a4").split(",")
# name, N, K, layer count per step, rotation copies, Q+gate split
SHAPES = [("qkvz", 16384, 5120, 48, 4, None),
          ("gdn_out", 5120, 6144, 48, 8, None),
          ("qkv", 14336, 5120, 16, 4, 12288),
          ("o_proj", 5120, 6144, 16, 8, None)]
ATTR = "_fp8_prefill_nvfp4"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


gen = torch.Generator(device=dev)
gen.manual_seed(1234)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, device=dev, generator=gen) * scale).to(
        torch.bfloat16)


class Lin(nn.Module):

    def forward(self, x):
        return self.quant_method.apply(self, x, None)


def make_lin(n, k, in_scale, split):
    W = rnd(n, k, scale=0.02)
    ws = (W.float().abs().amax() / 448.0).reshape(())
    m = Lin()
    m.quant_method = FP8QDQLinearMethod()
    m.weight = nn.Parameter((W.float() / ws).to(torch.float8_e4m3fn),
                            requires_grad=False)
    m.weight_scale = nn.Parameter(ws.float().clone(), requires_grad=False)
    m.input_scale = nn.Parameter(torch.tensor(in_scale, device=dev),
                                 requires_grad=False)
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
        x, m._nvfp4_decode_weight, scale_a=None,
        scale_b=m._nvfp4_decode_scale, alpha=None,
        weight_global_scale=m._nvfp4_decode_global_scale, bias=None,
        out_dtype=torch.bfloat16, size_n=m.out_features,
        size_k=m.in_features)


def w4a4(m, x):
    return fpn.prefill_nvfp4_apply(fpn.get_prefill_nvfp4_copy(m), m, x,
                                   x.shape, None)


FN = {"tree": lambda m, x: m(x), "marlin": marlin, "w4a4": w4a4}


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy()
                          .tobytes()).hexdigest()[:16]


log("gpu", torch.cuda.get_device_name(), "PDL",
    os.environ.get("TRTLLM_ENABLE_PDL"), "MS", MS, "arms", ARMS)
per_iter = {a: {M: [0.0] * ITERS for M in MS} for a in ARMS}
FLUSH = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)
detail = {}
hashes = {}
for sname, n, k, cnt, R, split in SHAPES:
    xs = {M: rnd(M, k, scale=1.5) for M in MS}
    s_in = 1.1 * max(float(x.float().abs().amax()) for x in xs.values()) / 448
    mods = [make_lin(n, k, s_in, split) for _ in range(R)]
    log(f"shape {sname} ({n},{k}) x{R} built; mem "
        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    for M in MS:
        x = xs[M]
        with autotune():
            for a in ARMS:
                FN[a](mods[0], x)
        torch.cuda.synchronize()
        graphs, outs = {}, {}
        for a in ARMS:
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
            dc = fpn.prefill_nvfp4_stats()["calls"] - c0
            graphs[a], outs[a] = g, o
            detail.setdefault(f"{sname}_M{M}", {})[a + "_w4a4_calls"] = dc
            for _ in range(WARM):
                g.replay()
        torch.cuda.synchronize()
        evs = {a: [] for a in ARMS}
        for it in range(ITERS):
            for a in ARMS:
                FLUSH.zero_()  # arms share weights: evict L2 between arms
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[a].replay()
                e1.record()
                evs[a].append((e0, e1))
        torch.cuda.synchronize()
        d = detail[f"{sname}_M{M}"]
        for a in ARMS:
            ts = [e0.elapsed_time(e1) / R for e0, e1 in evs[a]]
            for i, t in enumerate(ts):
                per_iter[a][M][i] += cnt * t
            st = sorted(ts)
            med = statistics.median(st)
            d[a + "_us"] = round(med * 1e3, 2)
            d[a + "_p90_us"] = round(st[int(0.9 * len(st)) - 1] * 1e3, 2)
            hashes[f"{sname}_M{M}_{a}"] = h(torch.cat(outs[a]))
        if "tree" in ARMS and "marlin" in ARMS:
            d["tree_eq_marlin"] = bool(
                torch.equal(torch.cat(outs["tree"]), torch.cat(outs["marlin"])))
        if "tree" in ARMS and "w4a4" in ARMS:
            d["tree_eq_w4a4"] = bool(
                torch.equal(torch.cat(outs["tree"]), torch.cat(outs["w4a4"])))
        log(f"{sname} M={M}", d)
        del graphs, outs
        torch.cuda.synchronize()
    del mods, xs
    torch.cuda.empty_cache()

summary = {}
batch = {}
for a in ARMS:
    for M in MS:
        st = sorted(per_iter[a][M])
        med = statistics.median(st)
        p10, p90 = st[int(0.1 * len(st))], st[int(0.9 * len(st)) - 1]
        summary[f"{a}_M{M}"] = {"step_ms": round(med, 4),
                                "p90_ms": round(p90, 4),
                                "nf": round((p90 - p10) / med, 4)}
        if a == "tree" and M in (8, 16, 32):
            batch[str(M)] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med,
                             "p90_ms": p90}
log("step-weighted FP8-origin linear ms:", summary)
print(json.dumps({"batch": batch,
                  "output": {"class": "precision", "summary": summary,
                             "detail": detail, "hashes": hashes}}),
      flush=True)
