"""Op-level FLUX.3 klein attention bench, in one process, interleaved A/B (+A/A):
VANILLA backend (main default) vs FLASHINFER per-tensor FP8 (this branch).

usage: python op_bench.py [reps]

Per denoise step the model runs 28 joint attentions (S=4096+T=4176), 5 image ModeBlock
attentions (S=4096) and 5 text ModeBlock attentions (S=T=80); B=1, H=24, d=128, BF16.
Q/K/V are [B,S,H,D] planes (the projection output layout); VANILLA receives the HND views
the model passes it, FP8 receives NHD. FP8 timing includes the three per-tensor
amax+quantize passes and the scale product. Timing: CUDA events, per rep the order
vanilla, fp8, vanilla2 (A/A) for every shape, median of `reps` after warmup.
step_ms = weighted per-step sum (28*t4176 + 5*t4096 + 5*t80).

Untimed checks for fp8 (same warmup/iteration count as timing): every iteration bitwise
equal to iteration 0; inputs unchanged; error vs an FP32 reference for randn and a peaky
(per-channel gain) Q/K, next to VANILLA's error; interleaved different-S calls equal fresh
calls; CUDA-graph capture and replay equal eager.
"""
import json
import math
import sys

import torch
import torch.nn.functional as F

print("importing", flush=True)
import tensorrt_llm  # noqa: F401,E402
from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention  # noqa: E402
from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention  # noqa: E402
from tensorrt_llm.visual_gen.args import QuantAttentionConfig  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
WARM = 10
H, D = 24, 128
SHAPES = {4176: 28, 4096: 5, 80: 5}

van = VanillaAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16)
fp8 = FlashInferAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16,
                          quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"))


def call_van(q, k, v):
    return van.forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def call_fp8(q, k, v):
    return fp8.forward(q, k, v)


ARMS = {"vanilla": call_van, "fp8": call_fp8, "vanilla_aa": call_van}
print(torch.cuda.get_device_name(), flush=True)


def make(S, peaky=False):
    g = torch.Generator(device="cuda").manual_seed(1234 + S)
    q, k, v = (torch.randn(1, S, H, D, device="cuda", generator=g) for _ in range(3))
    if peaky:
        gain = torch.rand(D, device="cuda", generator=g) * 3 + 0.25
        q, k = q * gain, k * gain
    return [t.bfloat16().contiguous() for t in (q, k, v)]


def ref32(q, k, v):
    return F.scaled_dot_product_attention(q.float().transpose(1, 2), k.float().transpose(1, 2),
                                          v.float().transpose(1, 2),
                                          scale=1 / math.sqrt(D)).transpose(1, 2)


def err(a, r):
    d = (a.float() - r).abs()
    return {"max_abs": round(d.max().item(), 6), "rel_l2": round((d.norm() / r.norm()).item(), 6)}


def stats(t):
    ts = sorted(t)
    med = ts[len(ts) // 2]
    return med, (ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / med


inputs = {S: make(S) for S in SHAPES}
pristine = {S: [t.clone() for t in v] for S, v in inputs.items()}

# --- timing ---
for S in SHAPES:
    for f in ARMS.values():
        for _ in range(WARM):
            f(*inputs[S])
torch.cuda.synchronize()
times = {a: {S: [] for S in SHAPES} for a in ARMS}
for i in range(REPS):
    for S in SHAPES:
        for a, f in ARMS.items():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            f(*inputs[S])
            e.record()
            e.synchronize()
            times[a][S].append(s.elapsed_time(e))
    if i % 25 == 0:
        print("rep", i, flush=True)
timing = {}
for a in ARMS:
    per = {}
    for S in SHAPES:
        m, n = stats(times[a][S])
        per[str(S)] = {"median_ms": round(m, 5), "noise": round(n, 4)}
    sm, sn = stats([sum(c * times[a][S][i] for S, c in SHAPES.items()) for i in range(REPS)])
    timing[a] = {"step_ms": round(sm, 4), "noise": round(sn, 4), "per_shape": per}
print("timing", json.dumps(timing), flush=True)

# --- untimed correctness (fp8), same warmup/iteration count ---
checks = {}
ok = True
for S in SHAPES:
    for t, p in zip(inputs[S], pristine[S]):
        t.copy_(p)
    for _ in range(WARM):
        call_fp8(*inputs[S])
    o0 = call_fp8(*inputs[S]).clone()
    rep_ok = True
    for _ in range(REPS):
        rep_ok &= torch.equal(call_fp8(*inputs[S]), o0)
    in_ok = all(torch.equal(t, p) for t, p in zip(inputs[S], pristine[S]))
    r = ref32(*inputs[S])
    qp, kp, vp = make(S, peaky=True)
    rp = ref32(qp, kp, vp)
    c = {"repeat_bitwise": bool(rep_ok), "inputs_unchanged": bool(in_ok),
         "fp8_err_randn": err(o0, r), "vanilla_err_randn": err(call_van(*inputs[S]), r),
         "fp8_err_peaky": err(call_fp8(qp, kp, vp), rp),
         "vanilla_err_peaky": err(call_van(qp, kp, vp), rp)}
    checks[str(S)] = c
    ok &= rep_ok and in_ok
    print(S, json.dumps(c), flush=True)

fresh = {S: call_fp8(*inputs[S]).clone() for S in SHAPES}
inter_ok = True
for _ in range(3):
    for S in (80, 4176, 4096, 4176, 80, 4096):
        inter_ok &= torch.equal(call_fp8(*inputs[S]), fresh[S])
checks["interleaved_matches_fresh"] = bool(inter_ok)
ok &= inter_ok

graph_ok = True
try:
    for S in (4176, 4096, 80):
        q, k, v = inputs[S]
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                call_fp8(q, k, v)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_g = call_fp8(q, k, v)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        graph_ok &= torch.equal(out_g, fresh[S])
except Exception as ex:  # noqa: BLE001
    graph_ok = False
    checks["graph_error"] = repr(ex)[:300]
checks["cuda_graph_replay_matches_eager"] = bool(graph_ok)
ok &= graph_ok

print(json.dumps({
    "passed": bool(ok),
    "compared_against": "FP32 torch SDPA reference of the same inputs (fp8 and VANILLA errors); "
                        "fp8 iteration 0; fresh calls; eager",
    "batch": {"1": {"step_ms": timing["fp8"]["step_ms"], "noise_floor": timing["fp8"]["noise"]}},
    "baseline_in_process": {"vanilla": timing["vanilla"]["step_ms"],
                            "vanilla_aa": timing["vanilla_aa"]["step_ms"]},
    "output": {"class": "mode_switch"},
    "timing": timing,
    "checks": checks,
}))
