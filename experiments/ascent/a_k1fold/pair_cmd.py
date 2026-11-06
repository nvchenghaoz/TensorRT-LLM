"""Idea 386 pair arm: SASS hashes of this tree's csa2_mxfp8_low_m_gemm_kernel instances (sass.py), then the chain.py
bench (36 boundaries, M = 1, 2, 4 at W_crit p10/p50 + W=2 us; arms tree / tree_k1 / fold / ideal). Prints the bench
evidence JSON last, with the SASS hashes added under "sass" (compare the K1FOLD=false keys across the two arms).
argv: --ckpt <checkpoint>
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def last_json(text):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:  # noqa: BLE001
                continue
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()
    py = sys.executable
    print("[pair_cmd] sass.py", flush=True)
    try:
        r = subprocess.run([py, os.path.join(HERE, "sass.py")], capture_output=True, text=True, timeout=2400)
        sass = last_json(r.stdout)
        print(f"[pair_cmd] sass: {sass.get('instances')} instances, libs {sass.get('libs')}", flush=True)
    except Exception as e:  # noqa: BLE001
        sass = {"error": str(e)}
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "sass.json"), "w") as f:
            json.dump(sass, f, indent=1)
    cmd = [py, os.path.join(HERE, "..", "moe_ar_trigger", "chain.py"), "--mode", "bench", "--ckpt", args.ckpt,
           "--batches", "1", "2", "4", "--arms", "tree", "tree_k1", "fold", "ideal", "--ws", "2", "--reps", "40",
           "--inner", "5"]
    print(f"[pair_cmd] {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if len(line) < 4000:
            sys.stdout.write(line)
            sys.stdout.flush()
    p.wait()
    res = last_json("".join(lines))
    res["sass"] = {"instances": sass.get("instances"), "libs": sass.get("libs"), "hashes": sass.get("hashes"),
                   "error": sass.get("error")}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
