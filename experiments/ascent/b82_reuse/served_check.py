"""Branch 82: b44_served_check.py (served A(T=80)/B(T=160)/A, P1/P2/L1/L2/P3) plus per-request
CUDA-graph capture counts. Runs b44's exact protocol on a fresh server with TLLM_LOG_LEVEL=INFO
(pitfall 644) and counts 'Capturing graph for key' lines (pitfall 572) and Dynamo recompile
messages per request.

usage: python served_check.py <checkpoint> <worktree>
"""
import os
import sys

WT = sys.argv[2]
sys.path.insert(0, os.path.join(WT, "experiments/ascent"))
os.environ["TLLM_LOG_LEVEL"] = "INFO"
import b44_served_check as b44  # noqa: E402

CAPTURE = "Capturing graph for key"
RECOMP = b44.RECOMP_PATTERNS
# Count both kinds in the same per-request split; report them separately below.
b44.RECOMP_PATTERNS = RECOMP + (CAPTURE,)

_print = print
_out = {}


def _capture_print(*a, **k):
    if len(a) == 1 and isinstance(a[0], str) and a[0].startswith("{"):
        import json

        try:
            res = json.loads(a[0])
        except Exception:
            return _print(*a, **k)
        per = res.get("per_request", [])
        steady_caps, steady_recomp, caps_by_req = 0, 0, []
        for r in per:
            caps = sum(1 for s in r.get("sample", []) if CAPTURE in s)
            # sample holds at most 4 lines; recompute exactly from the count split below
            caps_by_req.append({"tag": r["tag"], "prompt": r["prompt"], "msgs": r["recompile_msgs"],
                                "capture_lines_in_sample": caps})
        first_t80 = per[0] if per else None
        first_t160 = next((r for r in per if r["tag"] == "L1"), None)
        steady = [r for r in per if r is not first_t80 and r is not first_t160]
        steady_msgs = sum(r["recompile_msgs"] for r in steady)
        res["steady_requests"] = len(steady)
        res["steady_capture_plus_recompile_msgs"] = steady_msgs
        res["steady_zero_captures_and_recompiles"] = steady_msgs == 0
        res["P3_eq_P1"] = res.get("T80_after_T160_bitwise_count") == len(
            [r for r in per if r["tag"] == "P1"])
        res["note_counts"] = ("recompile_msgs per request count Dynamo recompile lines AND "
                              "'Capturing graph for key' lines (TLLM_LOG_LEVEL=INFO)")
        res["per_request_counts"] = caps_by_req
        return _print(json.dumps(res), **k)
    return _print(*a, **k)


b44.print = _capture_print

if __name__ == "__main__":
    b44.main()
