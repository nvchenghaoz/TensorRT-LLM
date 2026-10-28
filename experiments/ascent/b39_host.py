"""Branch 39 diagnostic: host (CPU) cost per call of each LINEAR GEMM path, GPU kept busy.

A long torch.cuda._sleep kernel is queued first, so every launch is asynchronous and the wall time
of N back-to-back calls is pure host cost (Python + binding + CUTLASS setup + launch).
Paths: TRT-LLM FP4GemmRunner t0 (main), FlashInfer stock fp4_gemm tactics 17 (DP) and 18 (StreamK),
and the full custom op torch.ops.trtllm.flux3_primsts_nvfp4_linear (branch route).
Also GPU time of each path for the same shapes (CUDA events, back-to-back, 50 calls).
"""

import json
import sys
import time

import torch

sys.path.insert(0, "experiments/ascent")
import tensorrt_llm  # noqa: E402,F401
from b39_linear_sweep import make_case  # noqa: E402
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import (  # noqa: E402
    _scalar,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402

N = 200


def log(*a):
    print(*a, flush=True)


def main():
    fp.sm120_linear_preload(torch.device("cuda"))
    mod = fp._sm120_fi_fp4_module()
    ws = fp._sm120_fi_workspace(torch.device("cuda"))
    res = {}
    for M in (4176,):
        case = make_case(M, seed=M)
        for name, args in case.items():
            a, w, sfa, sfw, gsa, gsw = args
            alpha = _scalar(a.device, gsa * gsw)
            out = torch.empty(a.shape[0], w.shape[0], device="cuda", dtype=torch.bfloat16)
            paths = {
                "t0_runner": lambda: fp._sm120_fp4_gemm_runner().run_gemm(
                    a, w, sfa, sfw, alpha, 0, 0, None, None),
                "fi17_DP": lambda: mod.fp4_gemm(a, w, sfa, sfw, alpha, out, ws, 17),
                "fi18_SK": lambda: mod.fp4_gemm(a, w, sfa, sfw, alpha, out, ws, 18),
                "branch_op": lambda: torch.ops.trtllm.flux3_primsts_nvfp4_linear(*args),
            }
            row = {}
            for k, f in paths.items():
                for _ in range(5):
                    f()
                torch.cuda.synchronize()
                host = []
                for rep in range(5):
                    torch.cuda._sleep(int(2.4e9 * 0.25))  # ~250 ms of GPU busy
                    t = time.perf_counter()
                    for _ in range(N // 5):
                        f()
                    host.append((time.perf_counter() - t) / (N // 5) * 1e6)
                    torch.cuda.synchronize()
                host.sort()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda._sleep(int(2.4e9 * 0.05))
                s.record()
                for _ in range(50):
                    f()
                e.record()
                torch.cuda.synchronize()
                row[k] = {"host_us_per_call": round(host[len(host) // 2], 2),
                          "gpu_us_per_call_b2b": round(s.elapsed_time(e) * 1e3 / 50, 2)}
                log(f"M{M} {name} {k}: {row[k]}")
            res[f"M{M}_{name}"] = row
    print(json.dumps({"passed": True, "res": res}))


if __name__ == "__main__":
    main()
