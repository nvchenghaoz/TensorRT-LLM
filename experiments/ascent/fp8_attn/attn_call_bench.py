"""Per-call FP8 attention time as Flux3Attention._attend + FlashInferAttention run it (eager).

Works on both trees: main copies Q/K/V (.contiguous()) and runs the 3x quantize_e4m3 path;
the branch skips the copies when the backend sets accepts_strided_qkv and runs the fused
custom op. Inputs: strided [1,S,3072] views of a packed [1,S,9216] BF16 buffer (single
blocks / image blocks) and the FP8R mixed layout. CUDA events, 30 warmup, median of 200
reps per case, cases interleaved. Last stdout line is the evidence JSON.
"""

import json
import math
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

H, D = 24, 128
N = H * D
dev = torch.device("cuda")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


backend = FlashInferAttention(
    num_heads=H, head_dim=D,
    quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
)
strided_ok = getattr(backend, "accepts_strided_qkv", False)
log(f"accepts_strided_qkv={strided_ok}")


def attend(q, k, v):
    # Flux3Attention._attend single-range fast path + _attn_impl's NHD view.
    if not strided_ok:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    S = q.shape[1]
    out = backend.forward(q.view(1, S, H, D), k.view(1, S, H, D), v.view(1, S, H, D))
    return out.flatten(2)


torch.manual_seed(0)
cases = {}
for S in (4176, 4096):
    buf = torch.randn(1, S, 3 * N, device=dev, dtype=torch.bfloat16) * 2
    cases[f"packed_{S}"] = (buf, (buf[..., :N], buf[..., N : 2 * N], buf[..., 2 * N :]))
qc = torch.randn(1, 4176, N, device=dev, dtype=torch.bfloat16)
kc = torch.randn(1, 4176, N, device=dev, dtype=torch.bfloat16)
vbuf = torch.randn(1, 4176, 3 * N, device=dev, dtype=torch.bfloat16)
cases["mixed_4176"] = (vbuf, (qc, kc, vbuf[..., 2 * N :]))
# Landing 18 (planar Q/K/V): q, k, v are contiguous [M, 3072] planes of a [3, M, 3072] buffer.
# This is the production layout of the NVFP4 blocks on main 0220e28c; it is the headline case.
for S in (4176, 4096):
    pbuf = torch.randn(3, 1, S, N, device=dev, dtype=torch.bfloat16) * 2
    cases[f"planar_{S}"] = (pbuf, (pbuf[0], pbuf[1], pbuf[2]))
pristine = {name: (c[0].clone(), [t.clone() for t in c[1]]) for name, c in cases.items()}

WARMUP, REPS = 30, 200
for name, (_, qkv) in cases.items():
    for _ in range(WARMUP):
        attend(*qkv)
torch.cuda.synchronize()
log("warmup done")
times = {name: [] for name in cases}
# L2 flush buffer so each call reads Q/K/V cold, as in the model (77 MB packed buffer).
flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
for r in range(REPS):
    for name, (_, qkv) in cases.items():
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        attend(*qkv)
        e.record()
        e.synchronize()
        times[name].append(s.elapsed_time(e))
    if r % 50 == 0:
        log(f"rep {r}")
stats = {}
for name, t in times.items():
    t = sorted(t)
    med = t[len(t) // 2]
    stats[name] = {
        "median_ms": med,
        "p10_ms": t[len(t) // 10],
        "p90_ms": t[(9 * len(t)) // 10],
        "noise": (t[(9 * len(t)) // 10] - t[len(t) // 10]) / med,
    }
    log(f"{name}: median {med * 1e3:.1f} us noise {stats[name]['noise']:.3f}")
unchanged = all(
    torch.equal(cases[n][0], pristine[n][0])
    and all(torch.equal(a, b) for a, b in zip(cases[n][1], pristine[n][1]))
    for n in cases
)
main = stats["planar_4176"]
print(json.dumps({
    "batch": {"1": {"kernel_ms": main["median_ms"], "noise_floor": main["noise"]}},
    "output": {"class": "reordered"},
    "cases": stats,
    "accepts_strided_qkv": strided_ok,
    "inputs_unchanged": unchanged,
}))
