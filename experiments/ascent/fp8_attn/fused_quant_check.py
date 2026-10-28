"""Bitwise check: fused strided Q/K/V E4M3 quantize (fp8_sm12x K1/K2) vs main's copying path.

Reference (main): FlashInferAttention._quantize_e4m3 = trtllm::quantize_e4m3_per_tensor on
x.contiguous(), descale.float(); scale_bmm1 = (q_d * k_d).mul_(sm_scale).
Compares E4M3 Q/K/V bytes and the 4 FP32 scalars bit for bit, for every stride case at
S=4176/4096, 300 random magnitudes, 100 repeated calls, interleaved S, CUDA-graph replay, the
full custom op (vs main quantize + run_fmha_v2_fp8_sm12x), and the op under torch.compile
(fullgraph, no emulate_precision_casts). Last stdout line is JSON.
"""

import json
import math
import random
import sys
import time

import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fi_backend
from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import (
    quantize_qkv_e4m3,
    visual_gen_fp8_attention_sm120,
)

H, D = 24, 128
N = H * D
SM_SCALE = 1.0 / math.sqrt(D)
dev = torch.device("cuda")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def ref_quant(q, k, v):
    outs = []
    descs = []
    for x in (q, k, v):
        x_q, d = torch.ops.trtllm.quantize_e4m3_per_tensor(x.contiguous())
        outs.append(x_q)
        descs.append(d.float().reshape(1))
    bmm1 = (descs[0] * descs[1]).mul_(SM_SCALE)
    scales = torch.cat([descs[0], descs[1], descs[2], bmm1])
    return outs[0], outs[1], outs[2], scales


def make_case(case, S, gen, mags=(1.0, 1.0, 1.0), B=1):
    def rnd(shape, mag):
        return (torch.randn(shape, generator=gen, device=dev, dtype=torch.float32) * mag).to(
            torch.bfloat16
        )

    if case == "packed":  # [B,S,9216] row stride 9216
        buf = torch.empty(B, S, 3 * N, device=dev, dtype=torch.bfloat16)
        for i in range(3):
            buf[..., i * N : (i + 1) * N] = rnd((B, S, N), mags[i])
        views = [buf[..., i * N : (i + 1) * N].view(B, S, H, D) for i in range(3)]
        return [buf], views
    if case == "contiguous":
        ts = [rnd((B, S, N), mags[i]).view(B, S, H, D) for i in range(3)]
        return ts, ts
    if case == "mixed":  # FP8R blocks: Q,K contiguous, V from packed 9216 rows
        q = rnd((B, S, N), mags[0]).view(B, S, H, D)
        k = rnd((B, S, N), mags[1]).view(B, S, H, D)
        buf = torch.empty(B, S, 3 * N, device=dev, dtype=torch.bfloat16)
        buf.copy_(rnd((B, S, 3 * N), 1.0))
        buf[..., 2 * N :] = rnd((B, S, N), mags[2])
        v = buf[..., 2 * N :].view(B, S, H, D)
        return [q, k, buf], [q, k, v]
    if case == "planar":  # [3,B,S,3072] planes, row stride 3072
        buf = torch.empty(3, B, S, N, device=dev, dtype=torch.bfloat16)
        for i in range(3):
            buf[i] = rnd((B, S, N), mags[i])
        return [buf], [buf[i].view(B, S, H, D) for i in range(3)]
    if case == "sliced":  # general-loop slice of a longer packed buffer
        buf = torch.empty(B, S + 80, 3 * N, device=dev, dtype=torch.bfloat16)
        buf.copy_(rnd((B, S + 80, 3 * N), 1.0))
        for i in range(3):
            buf[:, 80:, i * N : (i + 1) * N] = rnd((B, S, N), mags[i])
        return [buf], [buf[:, 80:, i * N : (i + 1) * N].view(B, S, H, D) for i in range(3)]
    raise ValueError(case)


def same(a, b):
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    elif a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    return a.shape == b.shape and bool(torch.equal(a, b))


def cmp(got, ref):
    return all(same(g, r) for g, r in zip(got, ref))


failures = []


def expect(ok, what):
    if not ok:
        failures.append(what)
        log(f"FAIL {what}")


gen = torch.Generator(device=dev)
gen.manual_seed(1234)
CASES = ["packed", "contiguous", "mixed", "planar", "sliced"]

# 1) every stride case at S=4176/4096 (and B=2 packed)
n_case = 0
for S in (4176, 4096):
    for case in CASES:
        storage, (q, k, v) = make_case(case, S, gen, mags=(3.0, 5.0, 0.7))
        pristine = [t.clone() for t in storage]
        got = quantize_qkv_e4m3(q, k, v, SM_SCALE)
        ref = ref_quant(q, k, v)
        expect(cmp(got, ref), f"stride case {case} S={S}")
        expect(all(torch.equal(a, b) for a, b in zip(storage, pristine)), f"input mutated {case}")
        n_case += 1
storage, (q, k, v) = make_case("packed", 2048, gen, B=2)
expect(cmp(quantize_qkv_e4m3(q, k, v, SM_SCALE), ref_quant(q, k, v)), "packed B=2")
log(f"stride cases done ({n_case}+1), failures={len(failures)}")

# 2) 300 random magnitudes (incl. min-scale clamp, outliers, zero tensor)
rng = random.Random(7)
n_rand = 0
for t in range(300):
    case = rng.choice(CASES)
    S = rng.choice((4176, 4096))
    mags = tuple(10 ** rng.uniform(-7, 4) for _ in range(3))
    storage, (q, k, v) = make_case(case, S, gen, mags=mags)
    if t % 10 == 3:  # outlier
        idx = rng.randrange(S)
        q[0, idx, rng.randrange(H), rng.randrange(D)] = rng.choice([-1, 1]) * mags[0] * 1e3
    if t == 5:
        v.zero_()
    got = quantize_qkv_e4m3(q, k, v, SM_SCALE)
    ref = ref_quant(q, k, v)
    expect(cmp(got, ref), f"random t={t} case={case} S={S} mags={mags}")
    n_rand += 1
log(f"random magnitudes done ({n_rand}), failures={len(failures)}")

# 3) 100 repeated calls on a fixed packed S=4176 input (pristine restore checks)
storage, (q, k, v) = make_case("packed", 4176, gen, mags=(2.0, 4.0, 1.0))
pristine = storage[0].clone()
ref = ref_quant(q, k, v)
first = [t.clone() for t in quantize_qkv_e4m3(q, k, v, SM_SCALE)]
for i in range(100):
    got = quantize_qkv_e4m3(q, k, v, SM_SCALE)
    expect(cmp(got, ref) and cmp(got, first), f"repeat {i}")
    expect(torch.equal(storage[0], pristine), f"repeat {i} input mutated")
log(f"repeat done, failures={len(failures)}")

# 4) interleaved S 4176 -> 4096 -> 80 -> 4176
inputs = {}
for S in (4176, 4096, 80):
    st, qkv = make_case("packed", S, gen, mags=(1.5, 2.5, 0.5))
    inputs[S] = (st, qkv, ref_quant(*qkv))
for rep in range(25):
    for S in (4176, 4096, 80, 4176):
        st, qkv, r = inputs[S]
        expect(cmp(quantize_qkv_e4m3(*qkv, SM_SCALE), r), f"interleave rep={rep} S={S}")
log(f"interleave done, failures={len(failures)}")

# 5) CUDA-graph capture + replay with new data copied into the static input
st, (q, k, v) = make_case("packed", 4176, gen)
static_buf = st[0]
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        quantize_qkv_e4m3(q, k, v, SM_SCALE)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    gout = quantize_qkv_e4m3(q, k, v, SM_SCALE)
for i in range(20):
    nst, _ = make_case("packed", 4176, gen, mags=(10 ** rng.uniform(-3, 3),) * 3)
    static_buf.copy_(nst[0])
    g.replay()
    expect(cmp(gout, ref_quant(q, k, v)), f"graph replay {i}")
log(f"graph replay done, failures={len(failures)}")

# 6) full op vs main path (ref quantize + run_fmha_v2_fp8_sm12x), every stride case
op_max = {}
for S in (4176, 4096):
    for case in CASES:
        st, (q, k, v) = make_case(case, S, gen)
        q8, k8, v8, sc = ref_quant(q, k, v)
        ref_out = torch.empty(q.shape, dtype=torch.bfloat16, device=dev)
        fi_backend.run_fmha_v2_fp8_sm12x(
            q8, k8, v8, ref_out, H, D, S,
            scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False, return_lse=False,
        )
        outs = [visual_gen_fp8_attention_sm120(q, k, v, SM_SCALE, False) for _ in range(3)]
        expect(all(torch.equal(o, ref_out) for o in outs), f"op vs main path {case} S={S}")
        op_max[f"{case}_{S}"] = float((outs[0].float() - ref_out.float()).abs().max())
log(f"op checks done, failures={len(failures)}")

# 7) backend.forward with strided views (accepts_strided_qkv) vs main path, and under
#    torch.compile fullgraph (no emulate_precision_casts): op is opaque -> bitwise.
from tensorrt_llm.visual_gen.args import QuantAttentionConfig  # noqa: E402

backend = fi_backend.FlashInferAttention(
    num_heads=H, head_dim=D,
    quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
)
expect(getattr(backend, "accepts_strided_qkv", False) is True, "accepts_strided_qkv")


def attend(q, k, v):
    return backend.forward(q, k, v)


compiled = torch.compile(attend, fullgraph=True, dynamic=False)
for S in (4176, 4096):
    for case in ("packed", "mixed", "planar"):
        st, (q, k, v) = make_case(case, S, gen)
        q8, k8, v8, sc = ref_quant(q, k, v)
        ref_out = torch.empty(q.shape, dtype=torch.bfloat16, device=dev)
        fi_backend.run_fmha_v2_fp8_sm12x(
            q8, k8, v8, ref_out, H, D, S,
            scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False, return_lse=False,
        )
        eager = attend(q, k, v)
        comp = [compiled(q, k, v) for _ in range(3)]
        expect(torch.equal(eager, ref_out), f"backend eager {case} S={S}")
        expect(all(torch.equal(c, ref_out) for c in comp), f"backend compiled {case} S={S}")
# short-S fallback still works (SDPA on contiguous copies)
st, (q, k, v) = make_case("packed", 80, gen)
short = attend(q, k, v)
short_ref = torch.nn.functional.scaled_dot_product_attention(
    q.contiguous().transpose(1, 2), k.contiguous().transpose(1, 2),
    v.contiguous().transpose(1, 2), scale=SM_SCALE,
).transpose(1, 2)
expect(torch.equal(short, short_ref), "short-S SDPA fallback eager")
# LSE path still works
out_lse, lse = backend.forward_with_lse(*make_case("packed", 4176, gen)[1])
expect(lse is not None and lse.shape == (1, H, 4176) and bool(torch.isfinite(lse).all()), "lse path")
log(f"backend checks done, failures={len(failures)}")

torch.cuda.synchronize()
result = {
    "passed": not failures,
    "compared": "fp8_sm12x K1/K2 E4M3 Q/K/V + 4 FP32 scalars vs trtllm::quantize_e4m3_per_tensor "
    "on contiguous copies (main _quantize_e4m3), bitwise; op and backend.forward (eager and "
    "torch.compile fullgraph) vs main quantize + run_fmha_v2_fp8_sm12x, bitwise",
    "stride_cases": n_case + 1,
    "random_trials": n_rand,
    "repeats": 100,
    "interleave_reps": 25,
    "graph_replays": 20,
    "op_max_abs_diff": op_max,
    "failures": failures[:20],
    "n_failures": len(failures),
}
print(json.dumps(result))
sys.exit(0 if not failures else 1)
