"""Idea 211 pair arm (run_gpu_pair: identical command in the base tree and the branch).

(1) G1: k1fast_g1.py, 40-layer FFN-input chain (attention-AR stand-in -> hc_ffn K1 -> router -> MoE), fold off,
    M = 3/4/8, W = 2.9 us, cold L2.
(2) G2 (claim A): moe_ar_trigger/chain.py --mode bench, 36 hc_attn boundaries (MoE-AR stand-in -> K1 trigger-first
    per the tree's M >= 4 gate -> wq_a||wkv -> wq_b), arm 'tree', M = 4/8, W = 2.9 us (--wcrit 0 --pair-w 2.9).
(3) K1 alone: cold-L2 CUDA graph of 40 K1 launches on distinct inputs, us per launch at M = 3/4/8, P = 224/480.
Prints {"batch": {"G1_M3": ..., "G2_M4": ...}, "output": {"class": "exact"}, ...}.
Usage: k1fast_pair2.py <ckpt>
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


def run(cmd, timeout):
    log("=== " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        s = line.strip()
        if s.startswith("{"):
            try:
                last = json.loads(s)
            except json.JSONDecodeError:
                pass
    rc = p.wait(timeout=timeout)
    return rc, last


def k1_alone():
    import torch

    sys.path.insert(0, HERE)
    from k1fast_common import EPS, ColdGraph, make_set, stats

    from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import mhc_lagged_layer_input

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(99)
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    out = {}
    for M in (3, 4, 8):
        sets = [make_set(M, gen, dev) for _ in range(40)]
        for P in (224, 480):
            def call(st, P=P):
                x, r, po, c, pr, nw = st
                return mhc_lagged_layer_input(x, r, po, c, pr, nw, EPS, P, enable_pdl=True)
            g = ColdGraph(call, sets)
            row = stats([g.time_us(flush) for _ in range(30)])
            out[f"M{M}/P{P}"] = row
            log(f"K1 alone M={M} P={P}: {row}")
            del g
    return out


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    py = sys.executable
    rc1, g1 = run([py, os.path.join(HERE, "k1fast_g1.py")], 2400)
    rc2, g2 = run([py, os.path.join(HERE, "..", "moe_ar_trigger", "chain.py"), "--mode", "bench", "--ckpt", ckpt,
                   "--batches", "4", "8", "--ws", "2.9", "--wcrit", "0", "--pair-w", "2.9", "--pair-arm", "tree",
                   "--P", "480"], 2400)
    k1 = k1_alone()
    batch = {}
    for k, v in ((g1 or {}).get("batch") or {}).items():
        batch[k] = v
    for k, v in ((g2 or {}).get("batch") or {}).items():
        batch[f"G2_M{k}"] = {kk: v[kk] for kk in ("kernel_ms", "noise_floor", "us_per_boundary", "W_us",
                                                  "in_job_base_us_per_boundary", "in_job_gain_vs_base") if kk in v}
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "g1_rc": rc1, "g2_rc": rc2,
                      "g1_checks_passed": (g1 or {}).get("checks_passed"), "g1_tree": (g1 or {}).get("tree"),
                      "g2_gate": (g2 or {}).get("gate"), "g2_trigger_first_in_tree": (g2 or {}).get(
                          "trigger_first_in_tree"), "k1_alone_us": k1}))
    return 0 if rc1 == 0 and rc2 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
