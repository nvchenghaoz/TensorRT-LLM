"""Idea 386: run the correctness suite in one GPU job, each part in its own subprocess; prints one JSON line.

argv: --ckpt <checkpoint> [--parts check sass layer chain]
  check : a_k1fold/check.py (kernel bitwise, graphs, skip-wait control, occupancy/regs/spills)
  sass  : a_k1fold/sass.py (per-instance SASS hashes of this build; compare across a base/head pair)
  layer : a_k1fold/layer_check.py --batches 1 2 4 (real decoder-layer chain, fold on/off/sleep/nopdl vs REF)
  chain : moe_ar_trigger/chain.py --mode check352 --batches 1 2 4 (arms tree/tree_fast0/ideal/fold/tree_k1 bitwise)
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name, cmd, timeout):
    t0 = time.time()
    print(f"[run_all] {name}: {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    for line in p.stdout:
        sys.stdout.write(f"[{name}] {line}" if len(line) < 4000 else f"[{name}] <{len(line)} chars>\n")
        sys.stdout.flush()
        if line.strip():
            last = line.strip()
    p.wait(timeout=timeout)
    try:
        res = json.loads(last)
    except Exception:  # noqa: BLE001
        res = {"passed": False, "parse_error": last[-800:]}
    res["rc"] = p.returncode
    res["seconds"] = round(time.time() - t0, 1)
    print(f"[run_all] {name}: passed={res.get('passed')} rc={p.returncode} in {res['seconds']} s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--parts", nargs="+", default=["check", "sass", "layer", "chain"])
    args = ap.parse_args()
    py = sys.executable
    out = {}
    if "check" in args.parts:
        out["check"] = run("check", [py, os.path.join(HERE, "check.py")], 1800)
    if "sass" in args.parts:
        out["sass"] = run("sass", [py, os.path.join(HERE, "sass.py")], 3000)
    if "layer" in args.parts:
        out["layer"] = run("layer", [py, os.path.join(HERE, "layer_check.py"), "--ckpt", args.ckpt, "--batches", "1",
                                     "2", "4"], 3000)
    if "chain" in args.parts:
        out["chain"] = run("chain", [py, os.path.join(HERE, "..", "moe_ar_trigger", "chain.py"), "--mode", "check352",
                                     "--ckpt", args.ckpt, "--batches", "1", "2", "4", "--ws", "2", "5", "--iters", "8"],
                           3000)
    if "bench" in args.parts:
        # Diagnostic timing in this tree only (the evidence is the base/head pair): tree / tree_k1 / ideal.
        out["bench"] = run("bench", [py, os.path.join(HERE, "..", "moe_ar_trigger", "chain.py"), "--mode", "bench",
                                     "--ckpt", args.ckpt, "--batches", "1", "2", "4", "--arms", "tree", "tree_k1",
                                     "ideal", "--ws", "2", "--reps", "20", "--inner", "5"], 3000)
        out["bench"]["passed"] = True
    passed = all(v.get("passed") for k, v in out.items() if k != "sass")
    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("stats", "hashes", "report", "summary", "bitwise")}
            for k, v in out.items()}
    print(json.dumps({"passed": bool(passed), "parts": out, "slim": slim}))


if __name__ == "__main__":
    main()
