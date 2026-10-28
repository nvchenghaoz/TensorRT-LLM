"""Idea 82 pair microbench: the composed SM120 QKV op flux3_primsts._sm120_cutlass_qkv(planar=True)
with per-plane alphas (the served NVFP4 QKV route), run unchanged in both trees.

Base tree: 3 per-plane Pingpong GEMMs on 3 streams + qknorm_rope. Candidate tree (default
TRTLLM_FLUX3_SM120_QKV_SCHED=batched): one L=3 Pingpong launch + qknorm_rope.
Synthetic NVFP4 tensors at FLUX.3 klein shapes, K=3072, N=3x3072, M in {4096, 4176, 4256}.
CUDA graph of REPS calls per replay, median of SAMPLES replays; prints
{"batch": {"1": {"kernel_ms": <M4176 us/call / 1000>, "noise_floor": ...}}, ...}.
Also an untimed repeat check (4 calls bitwise vs call 0, inputs unchanged) and the output
sha per M so the two arms can be compared bitwise.
"""

import hashlib
import json
import os
import statistics

import torch

REPS = 10
SAMPLES = int(os.environ.get("IDEA82_SAMPLES", "120"))
DEV = torch.device("cuda", 0)


def log(*a):
    print(*a, flush=True)


def rup(x, m):
    return (x + m - 1) // m * m


def main():
    torch.manual_seed(1234)
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    sched = strided.qkv_schedule()
    log("sched", sched)
    strided.preload(DEV)
    K = W = 3072
    wq = torch.randint(0, 256, (3 * W, K // 2), dtype=torch.uint8, device=DEV)
    wq_sf = torch.randint(0x30, 0x40, (3 * W * (K // 16),), dtype=torch.uint8, device=DEV)
    qkv_scale = torch.tensor([0.011, 0.013, 0.017], dtype=torch.float32, device=DEV)
    q_norm = (torch.rand(128, device=DEV) + 0.5).to(torch.bfloat16)
    k_norm = (torch.rand(128, device=DEV) + 0.5).to(torch.bfloat16)
    res = {"sched": sched, "M": {}}
    ok_all = True
    for M in (4096, 4176, 4256):
        a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=DEV)
        a_sf = torch.randint(0x30, 0x40, (rup(M, 128) * (K // 16),), dtype=torch.uint8, device=DEV)
        cos_sin = torch.randn(M, 128, device=DEV, dtype=torch.float32)
        positions = torch.randperm(M, device=DEV).to(torch.int64)
        pristine = [t.clone() for t in (a, a_sf, cos_sin, positions, wq, wq_sf, qkv_scale)]

        def op():
            return fp._sm120_cutlass_qkv(a, wq, a_sf, wq_sf, 1.0, 1.0, q_norm, k_norm, cos_sin,
                                         positions, qkv_scale, planar=True)

        first = op().clone()
        torch.cuda.synchronize()
        rep = True
        for _ in range(3):
            rep &= torch.equal(op().view(torch.int16), first.view(torch.int16))
        inputs_ok = all(torch.equal(x, y) for x, y in zip(
            (a, a_sf, cos_sin, positions, wq, wq_sf, qkv_scale), pristine))
        ok_all &= bool(rep and inputs_ok)
        sha = hashlib.sha256(first.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(REPS):
                op()
        torch.cuda.synchronize()
        for _ in range(10):
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
        xs.sort()
        med = statistics.median(xs)
        p10, p90 = xs[int(0.1 * (len(xs) - 1))], xs[int(0.9 * (len(xs) - 1))]
        res["M"][M] = {"us_med": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
                       "noise": round((p90 - p10) / med, 4), "repeat_bitwise": bool(rep),
                       "inputs_unchanged": bool(inputs_ok), "out_sha": sha[:24]}
        log(json.dumps({M: res["M"][M]}))
        del g
        torch.cuda.empty_cache()
    m = res["M"][4176]
    res["batch"] = {"1": {"kernel_ms": m["us_med"] / 1000.0, "noise_floor": m["noise"]}}
    res["output"] = {"class": "exact"}
    res["passed"] = bool(ok_all)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
