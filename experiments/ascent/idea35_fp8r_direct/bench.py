"""Idea 35: image.0 FP8R QKV chain at M=16384 (both cases run image.0 per modality at M=16384).

A (main):   trtllm::flux3_sm120_fp8r_qkv_qknorm_rope_cg -> fp8_sm12x.quantize_qkv_e4m3 (K1/K2)
            [-> fmha_v2 via visual_gen_fp8_attention_sm120's call]
B (branch): trtllm::flux3_sm120_fp8r_qkv_rope_fp8_cg (same overlay GEMM, qknorm_rope_fp8)
            [-> visual_gen_fp8_attention_sm120_prequant]
Each chain is captured in a CUDA graph and replayed; arms interleaved, CUDA events, median of
>= 60 replays per round. In the base tree only A exists (the twin is the branch's code); the
candidate tree times A and B in the same process (base re-measured in the same job).
Mode 'check': untimed sequence with the same warmup/iterations: every replay's qkv8 / scales /
FMHA output vs eager A (main) bitwise and vs iteration 0, inputs unchanged before each reset.
Usage: python bench.py [bench|check]
"""

import json
import math
import sys

import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
M, K, N = 16384, 3072, 9216
W, HEADS, HD = N // 3, N // 3 // 128, 128
SM_SCALE = 1.0 / math.sqrt(HD)
WARMUP, ITERS, ROUNDS = 10, 60, 5


def log(*a):
    print(*a, flush=True)


with open("/proc/loadavg") as f:
    log("loadavg start", f.read().strip())

from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as g  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv  # noqa: E402,F401
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_large_m as lm  # noqa: E402

HAS_B = hasattr(g, "fp8r_qkv_rope_fp8_cg")
dev = torch.device("cuda:0")
torch.manual_seed(1234)


def rowwise_e4m3(x):
    amax = x.float().abs().amax(dim=1).clamp(min=1e-6)
    scale = amax / 448.0
    q = (x.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.float().contiguous()


x_bf = (torch.randn(M, K, device=dev) * 2.0).to(torch.bfloat16)
w_bf = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
act, act_s = rowwise_e4m3(x_bf)
wt, wt_s = rowwise_e4m3(w_bf)
q_norm = (1.0 + 0.1 * torch.randn(HD, device=dev)).to(torch.bfloat16)
k_norm = (1.0 + 0.1 * torch.randn(HD, device=dev)).to(torch.bfloat16)
pos = torch.arange(M, device=dev, dtype=torch.float32)[:, None]
freq = 1.0 / (10000.0 ** (torch.arange(0, 64, device=dev, dtype=torch.float32) / 64.0))
ang = pos * freq[None, :]
cos_sin = torch.cat((ang.cos(), ang.sin()), dim=-1).contiguous()
inputs = (act, wt, act_s, wt_s, q_norm, k_norm, cos_sin)
pristine = [t.clone() for t in inputs]
log(f"M={M} K={K} N={N} heads={HEADS}; large_m use={lm.use('fp8r', M)} "
    f"arm={lm.fp8r_arm(N) if lm.use('fp8r', M) else g.ARM}; has_twin={HAS_B}")

from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as _fi  # noqa: E402


def fmha(q8, k8, v8, scales):
    out = torch.empty((1, M, HEADS, HD), dtype=torch.bfloat16, device=dev)
    _fi.run_fmha_v2_fp8_sm12x(q8, k8, v8, out, HEADS, HD, M, scale_bmm1_d=scales[3:4],
                              scale_bmm2_d=scales[2:3], causal=False, return_lse=False)
    return out


def chain_a(with_fmha):
    out = torch.ops.trtllm.flux3_sm120_fp8r_qkv_qknorm_rope_cg(*inputs, False)
    q, k, v = (t.reshape(1, M, HEADS, HD) for t in out.split(W, -1))
    q8, k8, v8, sc = fp8_sm12x.quantize_qkv_e4m3(q, k, v, SM_SCALE)
    if with_fmha:
        return fmha(q8, k8, v8, sc), sc, q8, k8, v8
    return q8, k8, v8, sc


def chain_b(with_fmha, keep_v=False):
    qkv8, sc, gemm = torch.ops.trtllm.flux3_sm120_fp8r_qkv_rope_fp8_cg(*inputs, SM_SCALE, keep_v)
    if with_fmha:
        o = torch.ops.trtllm.visual_gen_fp8_attention_sm120_prequant(qkv8, sc, 1, HEADS)
        return o, sc, qkv8, gemm
    return qkv8, sc, gemm


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        res = fn()
    torch.cuda.synchronize()
    return graph, res


def time_graph(graph, iters):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(WARMUP):
        graph.replay()
    for i in range(iters):
        st[i].record()
        graph.replay()
        en[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(st, en)]


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    return med, (ts[int(0.9 * (n - 1))] - ts[int(0.1 * (n - 1))]) / med


def inputs_ok():
    return all(torch.equal(a, b) for a, b in zip(inputs, pristine))


if MODE == "bench":
    arms = {"A_qkv": capture(lambda: chain_a(False)), "A_qkv_fmha": capture(lambda: chain_a(True))}
    if HAS_B:
        arms["B_qkv"] = capture(lambda: chain_b(False))
        arms["B_qkv_fmha"] = capture(lambda: chain_b(True))
    order = list(arms)
    if HAS_B:
        order = ["A_qkv", "B_qkv", "A_qkv_fmha", "B_qkv_fmha"]
    times = {k: [] for k in arms}
    times["A_qkv_aa"] = []
    for r in range(ROUNDS):
        for k in order:
            times[k] += time_graph(arms[k][0], ITERS)
            if k == "A_qkv":
                times["A_qkv_aa"] += time_graph(arms[k][0], ITERS)
        log(f"round {r}: " + ", ".join(f"{k}={stats(v)[0]:.4f}" for k, v in times.items()))
    summary = {k: {"median_ms": round(stats(v)[0], 5), "noise": round(stats(v)[1], 5)}
               for k, v in times.items()}
    log("SUMMARY", json.dumps(summary))
    head = "B_qkv" if HAS_B else "A_qkv"
    med, noise = stats(times[head])
    with open("/proc/loadavg") as f:
        log("loadavg end", f.read().strip())
    print(json.dumps({
        "batch": {case: {"kernel_ms": round(med, 5), "noise_floor": round(noise, 5)}
                  for case in ("t2i-2048", "i2i-2048")},
        "arm_timed": head,
        "same_shape_note": "image.0 runs at M=16384 in both cases; one measurement serves both",
        "summary": summary,
        "output": {"class": "exact"},
    }))
    sys.exit(0)

# ---- check: untimed sequence, same warmup/iterations/reset points as the timed one ----
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


assert HAS_B, "check needs the branch tree"
ref_o, ref_sc, rq8, rk8, rv8 = chain_a(True)
torch.cuda.synchronize()
ref_qkv8 = torch.stack((rq8.reshape(M, W), rk8.reshape(M, W), rv8.reshape(M, W))).clone()
ref_o, ref_sc = ref_o.clone(), ref_sc.clone()
ref_gemm = g.gemm(act, wt, act_s, wt_s)
ref_v = ref_gemm[:, 2 * W:].clone()
torch.cuda.synchronize()
note(inputs_ok(), "inputs unchanged after eager A")
# fp32 sanity: scales near expectation
results = {}
for keep_v in (False, True):
    eo, esc, eqkv8, egemm = chain_b(True, keep_v)
    torch.cuda.synchronize()
    tag = f"eager_B_keep_v{int(keep_v)}"
    results[tag] = {
        "qkv8": bool(torch.equal(eqkv8.view(torch.uint8), ref_qkv8.view(torch.uint8))),
        "scales0_3": bool(torch.equal(esc[:4], ref_sc[:4])),
        "fmha": bool(torch.equal(eo, ref_o)),
        "v_view": bool(torch.equal(egemm[:, 2 * W:], ref_v)) if keep_v else None,
        "scales_b": esc.tolist(), "scales_a": ref_sc.tolist(),
    }
    note(results[tag]["qkv8"] and results[tag]["scales0_3"] and results[tag]["fmha"], tag)
    if keep_v:
        note(results[tag]["v_view"], tag + " v view")
note(inputs_ok(), "inputs unchanged after eager B")
for keep_v in (False, True):
    graph, res = capture(lambda kv=keep_v: chain_b(True, kv))
    for r in range(ROUNDS):
        for _ in range(WARMUP):
            graph.replay()
        bad = 0
        it0 = None
        for i in range(ITERS):
            graph.replay()
            torch.cuda.synchronize()
            o, sc, qkv8 = res[0], res[1], res[2]
            same = (torch.equal(qkv8.view(torch.uint8), ref_qkv8.view(torch.uint8))
                    and torch.equal(sc[:4], ref_sc[:4]) and torch.equal(o, ref_o))
            if keep_v:
                same = same and torch.equal(res[3][:, 2 * W:], ref_v)
            if i == 0:
                it0 = (o.clone(), sc.clone(), qkv8.clone())
            else:
                same = same and torch.equal(o, it0[0]) and torch.equal(sc, it0[1])
            bad += 0 if same else 1
        note(bad == 0, f"graph B keep_v={keep_v} round {r}: {bad}/{ITERS} replays differ")
        note(inputs_ok(), f"inputs unchanged before reset, keep_v={keep_v} round {r}")
        for a, b in zip(inputs, pristine):
            a.copy_(b)
    results[f"graph_B_keep_v{int(keep_v)}_replays"] = ROUNDS * ITERS
# Also the A graph (main) replays equal eager A.
graph_a, res_a = capture(lambda: chain_a(True))
bad = 0
for i in range(ITERS):
    graph_a.replay()
    torch.cuda.synchronize()
    bad += 0 if (torch.equal(res_a[0], ref_o) and torch.equal(res_a[1], ref_sc)) else 1
note(bad == 0, f"graph A replays differ {bad}")
print(json.dumps({
    "passed": not fails, "failures": fails,
    "compared_against": "main's chain (qknorm_rope_cg + fp8_sm12x K1/K2 + fmha_v2), eager, same "
    "inputs; qkv8 bytes, scales[0..3], FMHA output bitwise; graph replays vs eager and iter 0",
    "results": results,
}))
