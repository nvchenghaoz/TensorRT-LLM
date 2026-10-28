"""Idea 82 pair microbench, run unchanged in both trees (base main / candidate branch).

Ops (synthetic NVFP4 tensors at FLUX.3 klein shapes, K=3072, N=3x3072, per-plane alphas,
M in {4096, 4176, 4256}):
  fp8 (headline, the served route after landing 54): flux3_primsts.nvfp4_qkv_rope_fp8_impl =
      planar NVFP4 QKV GEMMs (_sm120_qkv_planar_gemms) -> qknorm_rope_fp8 (E4M3 q/k/v + scales).
  bf16: flux3_primsts._sm120_cutlass_qkv(planar=True) = planar GEMMs -> in-place qknorm_rope.
Base tree: 3 per-plane Pingpong GEMMs on 3 streams. Candidate (default
TRTLLM_FLUX3_SM120_QKV_SCHED=batched): one L=3 Pingpong launch.
Timing: CUDA graph of REPS calls per replay, ~WARM_S s sustained warm-up, SAMPLES replays,
median; noise = spread of 10 block medians / median. Untimed: 4 calls bitwise vs call 0,
inputs unchanged, output sha per M (compare across the two arms for bitwise equality).
Last line: {"batch": {"1": {"kernel_ms": <fp8 op M4176 us/call / 1000>, "noise_floor": ...}}}.
"""

import hashlib
import json
import os
import statistics
import time

import torch

REPS = 10
SAMPLES = int(os.environ.get("IDEA82_SAMPLES", "300"))
BLOCKS = 10
WARM_S = float(os.environ.get("IDEA82_WARM_S", "4"))
DEV = torch.device("cuda", 0)


def log(*a):
    print(*a, flush=True)


def rup(x, m):
    return (x + m - 1) // m * m


def bits(t):
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    return t.view(torch.int32)


def as_list(x):
    return list(x) if isinstance(x, (tuple, list)) else [x]


def measure(op, pristine, live):
    first = [t.clone() for t in as_list(op())]
    torch.cuda.synchronize()
    rep = True
    for _ in range(3):
        got = as_list(op())
        rep &= all(torch.equal(bits(x), bits(y)) for x, y in zip(got, first))
    inputs_ok = all(torch.equal(x, y) for x, y in zip(live, pristine))
    h = hashlib.sha256()
    for t in first:
        h.update(bits(t).cpu().numpy().tobytes())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            op()
    torch.cuda.synchronize()
    t_end = time.time() + WARM_S
    while time.time() < t_end:
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    xs = []
    for _ in range(SAMPLES):
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        xs.append(e0.elapsed_time(e1) * 1000.0 / REPS)
    bs = len(xs) // BLOCKS
    block_meds = [statistics.median(xs[i * bs:(i + 1) * bs]) for i in range(BLOCKS)]
    xs.sort()
    med = statistics.median(xs)
    p10, p90 = xs[int(0.1 * (len(xs) - 1))], xs[int(0.9 * (len(xs) - 1))]
    del g
    return {"us_med": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
            "raw_p10_p90_rel": round((p90 - p10) / med, 4),
            "block_medians": [round(b, 2) for b in block_meds],
            "noise": round((max(block_meds) - min(block_meds)) / med, 4),
            "repeat_bitwise": bool(rep), "inputs_unchanged": bool(inputs_ok),
            "out_sha": h.hexdigest()[:24]}


def main():
    torch.manual_seed(1234)
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    sched = strided.qkv_schedule()
    log("sched", sched, "fp8 op present", hasattr(fp, "nvfp4_qkv_rope_fp8_impl"))
    strided.preload(DEV)
    K = W = 3072
    wq = torch.randint(0, 256, (3 * W, K // 2), dtype=torch.uint8, device=DEV)
    wq_sf = torch.randint(0x30, 0x40, (3 * W * (K // 16),), dtype=torch.uint8, device=DEV)
    qkv_scale = torch.tensor([0.011, 0.013, 0.017], dtype=torch.float32, device=DEV)
    q_norm = (torch.rand(128, device=DEV) + 0.5).to(torch.bfloat16)
    k_norm = (torch.rand(128, device=DEV) + 0.5).to(torch.bfloat16)
    sm_scale = 1.0 / (128 ** 0.5)
    res = {"sched": sched, "fp8": {}, "bf16": {}}
    ok_all = True
    for M in (4096, 4176, 4256):
        a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=DEV)
        a_sf = torch.randint(0x30, 0x40, (rup(M, 128) * (K // 16),), dtype=torch.uint8, device=DEV)
        cos_sin = torch.randn(M, 128, device=DEV, dtype=torch.float32)
        positions = torch.randperm(M, device=DEV).to(torch.int64)
        live = (a, a_sf, cos_sin, positions, wq, wq_sf, qkv_scale, q_norm, k_norm)
        pristine = [t.clone() for t in live]

        def op_fp8():
            return fp.nvfp4_qkv_rope_fp8_impl(a, wq, a_sf, wq_sf, q_norm, k_norm, cos_sin,
                                              positions, qkv_scale, sm_scale)

        def op_bf16():
            return fp._sm120_cutlass_qkv(a, wq, a_sf, wq_sf, 1.0, 1.0, q_norm, k_norm, cos_sin,
                                         positions, qkv_scale, planar=True)

        for name, op in (("fp8", op_fp8), ("bf16", op_bf16)):
            r = measure(op, pristine, live)
            res[name][M] = r
            ok_all &= r["repeat_bitwise"] and r["inputs_unchanged"]
            log(json.dumps({name: {M: r}}))
        torch.cuda.empty_cache()
    m = res["fp8"][4176]
    res["batch"] = {"1": {"kernel_ms": m["us_med"] / 1000.0, "noise_floor": m["noise"]}}
    res["output"] = {"class": "exact"}
    res["passed"] = bool(ok_all)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
