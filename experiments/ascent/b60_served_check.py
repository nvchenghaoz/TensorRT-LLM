"""Branch 60 (idea 96): served-path checks for the step-batched flow-map image vec rows.

Reuses experiments/ascent/b52_fp8r/served_check.py's server driver (trtllm-serve VisualGen,
benchmark defaults + this tree's experiments/ascent-config/engine.json, TORCH_LOGS=recompiles,
same 12 prompts / seeds as attend_fastpath/e2e_repeat_check.py).

Server ON  (TRTLLM_FLUX3_IMAGE_VEC_STEP_BATCH=1, graphs as in engine.json):
  P1 (11 T=80 prompts), P2, L1/L2 (T=160 prompt twice), P3 (T=80 again).
  repeat: P1 == P2, L1 == L2 bitwise; A/B/A: P1 == P3 bitwise; captures / recompiles per request.
Server OFF (TRTLLM_FLUX3_IMAGE_VEC_STEP_BATCH=0 = main's per-step path, graphs on):
  P1, L1, P3: A/B/A bitwise in this mode; ON vs OFF per prompt PSNR / max pixel diff.
Server ON, cuda_graph_config.enable=False: P1 + L1: graph-on == graph-off bitwise.

usage: python b60_served_check.py <checkpoint> <worktree>
"""
import json
import math
import os
import sys

CKPT = sys.argv[1]
WT = sys.argv[2]
sys.path.insert(0, os.path.join(WT, "experiments/ascent/b52_fp8r"))
import served_check as sc  # noqa: E402

ENV = "TRTLLM_FLUX3_IMAGE_VEC_STEP_BATCH"


def psnr(a, b):
    import numpy as np

    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(255.0 * 255.0 / mse)


def steady_stats(reqs, first_tags=("P1", "L1")):
    seen, steady = set(), []
    for r in reqs:
        if r["tag"] in first_tags and r["tag"] not in seen:
            seen.add(r["tag"])
            continue
        steady.append(r)
    return (sum(r["graph_captures"] for r in steady), sum(r["recompile_msgs"] for r in steady))


def main():
    short = list(range(len(sc.PROMPTS) - 1))
    long_i = len(sc.PROMPTS) - 1

    def allbit(rs):
        return all(r["pixels_bitwise"] and r["png_bytes_equal"] for r in rs)

    args_on, overlay = sc.server_args()
    os.environ[ENV] = "1"
    on, req_on, start_on = sc.run_server(
        "switch_on", args_on,
        [("P1", short), ("P2", short), ("L1", [long_i]), ("L2", [long_i]), ("P3", short)])
    os.environ[ENV] = "0"
    off, req_off, start_off = sc.run_server(
        "switch_off", args_on, [("P1", short), ("L1", [long_i]), ("P3", short)])
    os.environ[ENV] = "1"
    args_ng, _ = sc.server_args(graphs_override=False)
    ng, req_ng, start_ng = sc.run_server("switch_on_graphs_off", args_ng,
                                         [("P1", short), ("L1", [long_i])])

    r12, rL, r13 = sc.cmp(on["P1"], on["P2"]), sc.cmp(on["L1"], on["L2"]), sc.cmp(on["P1"], on["P3"])
    off13 = sc.cmp(off["P1"], off["P3"])
    g80, g160 = sc.cmp(on["P1"], ng["P1"]), sc.cmp(on["L1"], ng["L1"])
    onoff = []
    for tag in ("P1", "L1"):
        for i in on[tag]:
            a, b = on[tag][i][1], off[tag][i][1]
            onoff.append({"prompt": i, "psnr_db": psnr(a, b),
                          "max_abs_px_diff": int(abs(a.astype(int) - b.astype(int)).max()),
                          "bitwise": bool((a == b).all())})
    cap_on, rec_on = steady_stats(req_on)
    cap_off, rec_off = steady_stats(req_off)
    rep = allbit(r12) and allbit(rL)
    result = {
        "passed": rep and allbit(r13) and allbit(off13) and allbit(g80) and allbit(g160)
        and cap_on == 0 and rec_on == 0,
        "repeated_execution_passed": rep,
        "ABA_T80_T160_T80_bitwise_switch_on": allbit(r13),
        "ABA_T80_T160_T80_bitwise_switch_off": allbit(off13),
        "graph_on_eq_graph_off_bitwise_switch_on": allbit(g80) and allbit(g160),
        "steady_graph_captures": {"on": cap_on, "off": cap_off},
        "steady_dynamo_recompile_msgs": {"on": rec_on, "off": rec_off},
        "startup": {"on": start_on, "off": start_off, "on_graphs_off": start_ng},
        "first_requests": {
            "on": [r for r in req_on if r["tag"] in ("P1", "L1")][:1]
            + [r for r in req_on if r["tag"] == "L1"][:1],
            "off": [r for r in req_off if r["tag"] in ("P1", "L1")][:1]
            + [r for r in req_off if r["tag"] == "L1"][:1],
        },
        "switch_on_vs_off": onoff,
        "switch_on_vs_off_min_psnr_db": min(x["psnr_db"] for x in onoff),
        "P1_vs_P2": r12, "L1_vs_L2": rL, "P1_vs_P3": r13, "off_P1_vs_P3": off13,
        "graphs_on_vs_off_T80": g80, "graphs_on_vs_off_T160": g160,
        "per_request": {"on": req_on, "off": req_off, "on_graphs_off": req_ng},
        "engine_overlay": overlay,
        "note": "stand-in for check.py repeated execution; gate v2 vs reference.json is not "
                "shipped to worker jobs and runs at landing",
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
