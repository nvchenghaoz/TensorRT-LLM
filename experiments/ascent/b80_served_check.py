"""Branch 80 (idea 128, v_ldsm_x4): served-path repeated-execution stand-in for check.py.

Reuses experiments/ascent/b52_fp8r/served_check.py's server driver (trtllm-serve VisualGen,
benchmark defaults + this tree's experiments/ascent-config/engine.json, same 12 prompts / seeds).

Server VX4=1 (default routing, b2_ms_rq_vx4): P1 (11 T=80 prompts), P2, L1/L2 (T=160 twice), P3.
  repeated_execution_passed = P1 == P2 and L1 == L2 bitwise; A/B/A: P1 == P3 bitwise.
Server VX4=0 (TRTLLM_FMHA_V2_FLUX3_VX4=0 -> b2_ms_rq, main's kernel): P1, L1.
  The change is exact, so VX4=1 must equal VX4=0 bitwise per prompt.

usage: python b80_served_check.py <checkpoint> <worktree>
"""
import json
import os
import sys

CKPT = sys.argv[1]
WT = sys.argv[2]
sys.path.insert(0, os.path.join(WT, "experiments/ascent/b52_fp8r"))
import served_check as sc  # noqa: E402

ENV = "TRTLLM_FMHA_V2_FLUX3_VX4"


def main():
    short = list(range(len(sc.PROMPTS) - 1))
    long_i = len(sc.PROMPTS) - 1

    def allbit(rs):
        return all(r["pixels_bitwise"] and r["png_bytes_equal"] for r in rs)

    args_on, overlay = sc.server_args()
    os.environ[ENV] = "1"
    on, req_on, start_on = sc.run_server(
        "vx4_on", args_on,
        [("P1", short), ("P2", short), ("L1", [long_i]), ("L2", [long_i]), ("P3", short)])
    os.environ[ENV] = "0"
    off, req_off, start_off = sc.run_server("vx4_off", args_on, [("P1", short), ("L1", [long_i])])
    os.environ[ENV] = "1"

    r12, rL, r13 = sc.cmp(on["P1"], on["P2"]), sc.cmp(on["L1"], on["L2"]), sc.cmp(on["P1"], on["P3"])
    oP, oL = sc.cmp(on["P1"], off["P1"]), sc.cmp(on["L1"], off["L1"])
    onoff = []
    for tag in ("P1", "L1"):
        for i in on[tag]:
            a, b = on[tag][i][1], off[tag][i][1]
            onoff.append({"prompt": i, "tag": tag,
                          "max_abs_px_diff": int(abs(a.astype(int) - b.astype(int)).max()),
                          "bitwise": bool((a == b).all())})
    rep = allbit(r12) and allbit(rL)
    result = {
        "passed": rep and allbit(r13) and allbit(oP) and allbit(oL),
        "repeated_execution_passed": rep,
        "ABA_T80_T160_T80_bitwise_vx4_on": allbit(r13),
        "vx4_on_eq_vx4_off_bitwise": allbit(oP) and allbit(oL),
        "vx4_on_vs_off": onoff,
        "max_abs_px_diff_on_vs_off": max(x["max_abs_px_diff"] for x in onoff),
        "startup": {"on": start_on, "off": start_off},
        "P1_vs_P2": r12, "L1_vs_L2": rL, "P1_vs_P3": r13,
        "on_vs_off_P1": oP, "on_vs_off_L1": oL,
        "per_request": {"on": req_on, "off": req_off},
        "engine_overlay": overlay,
        "note": "stand-in for check.py repeated execution (accepted pattern b60_served_check.py); "
                "gate v2 vs reference.json runs at landing",
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
