"""Branch 57 (idea 103) served check: branch 52's experiments/ascent/b52_fp8r/served_check.py
(trtllm-serve with the benchmark defaults + engine.json; repeat P1==P2, L1==L2 bitwise,
A/B/A, graph-on == graph-off bitwise, per-request graph captures and Dynamo recompiles), with
the per-request log patterns widened to also count FlashInfer JIT builds/loads, CuTe compiles and
autotuner lines, so 'steady_requests' == 0 shows no JIT build, recompile or autotune inside a
request.

usage: python served_check_case1.py <checkpoint> <worktree>
"""
import os
import sys

WT = sys.argv[2]
sys.path.insert(0, os.path.join(WT, "experiments/ascent/b52_fp8r"))
import served_check as sc  # noqa: E402

sc.RECOMP = sc.RECOMP + ("flashinfer.jit", "flux3_fp4_gemm_sm120_strided", "[Autotuner]",
                         "Autotuning", "cute.compile")

if __name__ == "__main__":
    print("per-request patterns (counted as recompile_msgs):", sc.RECOMP, flush=True)
    sc.main()
