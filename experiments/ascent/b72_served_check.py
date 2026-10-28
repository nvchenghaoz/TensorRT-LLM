"""Branch 72 (idea 124, softmax_trim): served stand-in for check.py (item 4).

Reuses experiments/ascent/b52_fp8r/served_check.py's run_server / cmp (benchmark defaults +
this tree's experiments/ascent-config/engine.json, graphs as in engine.json).

Server 1, TRTLLM_FMHA_V2_FLUX3_ST=1 (softmax_trim routed): the 12 check prompts twice (P1, P2).
  Require P1 == P2 bitwise (pixels and PNG bytes) -> repeated_execution_passed.
Server 2, TRTLLM_FMHA_V2_FLUX3_ST=0 (b2_ms_rq without st): P1 once.
  Require ST=1 == ST=0 bitwise on every prompt.
gate v2 against reference.json is not shipped to worker jobs and runs at landing.

usage: python b72_served_check.py <checkpoint> <worktree>
"""
import json
import os
import sys

CKPT, WT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(WT, "experiments/ascent/b52_fp8r"))
import served_check as sc  # noqa: E402


def allbit(rs):
    return all(r["pixels_bitwise"] and r["png_bytes_equal"] for r in rs)


def main():
    idx = list(range(len(sc.PROMPTS)))
    args, overlay = sc.server_args()
    os.environ["TRTLLM_FMHA_V2_FLUX3_ST"] = "1"
    st1, req1, start1 = sc.run_server("st1", args, [("P1", idx), ("P2", idx)])
    os.environ["TRTLLM_FMHA_V2_FLUX3_ST"] = "0"
    st0, req0, start0 = sc.run_server("st0", args, [("P1", idx)])

    r12 = sc.cmp(st1["P1"], st1["P2"])
    r10 = sc.cmp(st1["P1"], st0["P1"])
    rep, eq = allbit(r12), allbit(r10)
    distinct = len({st1["P1"][i][0] for i in idx})
    result = {
        "passed": rep and eq and distinct == len(idx),
        "repeated_execution_passed": rep,
        "st1_eq_st0_bitwise": eq,
        "distinct_images": distinct,
        "n_prompts": len(idx),
        "P1_vs_P2_st1": r12,
        "st1_vs_st0_P1": r10,
        "per_request_st1": req1,
        "per_request_st0": req0,
        "startup": {"st1": start1, "st0": start0},
        "engine_overlay": overlay,
        "note": "stand-in for check.py (served, graphs as in engine.json); gate v2 vs "
                "reference.json is not shipped to worker jobs and runs at landing",
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
