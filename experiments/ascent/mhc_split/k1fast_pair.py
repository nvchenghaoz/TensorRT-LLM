"""Same-GPU pair arm for the fast K1: runs unchanged in the base tree (main's K1) and the branch
(fast K1 by default). (a) K1 cold-L2 graph us per launch at M=1/2/4/8 for P=480 and P=96 (40
launches on distinct input sets per graph, L2 flushed before every replay); (b) chain.py's
40-layer chain bench in a subprocess. The evidence JSON's kernel_ms is the chain time per M.
"""

import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from k1fast_common import EPS, HIDDEN, ColdGraph, log, make_set, stats  # noqa: E402

from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import mhc_lagged_layer_input  # noqa: E402


def k1_graph_us(rounds):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(99)
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    out = {}
    for M in (1, 2, 4, 8):
        sets = [make_set(M, gen, dev) for _ in range(40)]
        for P in (480, 96):
            def call(st, P=P):
                x, r, po, c, pr, nw = st
                return mhc_lagged_layer_input(x, r, po, c, pr, nw, EPS, P)
            g = ColdGraph(call, sets)
            row = stats([g.time_us(flush) for _ in range(rounds)])
            out[f"M{M}/P{P}"] = row
            log(f"K1 M={M} P={P}: {row}")
            del g
    del flush
    torch.cuda.synchronize()
    return out


def main():
    torch.cuda.set_device(0)
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    k1 = k1_graph_us(int(os.environ.get("K1_ROUNDS", "40")))
    cmd = [sys.executable, os.path.join(HERE, "chain.py"), "--mode", "bench", "--ckpt", ckpt,
           "--batches", "1", "2", "4", "8"]
    log("chain:", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(p.stdout[-6000:])
    sys.stderr.write(p.stderr[-4000:])
    last = [ln for ln in p.stdout.strip().splitlines() if ln.startswith("{")]
    chain = json.loads(last[-1]) if last else {"batch": {}}
    batch = {m: {"kernel_ms": v["kernel_ms"], "noise_floor": v["noise_floor"],
                 "per_layer_us": v["per_layer_us"]} for m, v in chain["batch"].items()}
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "k1_graph_us": k1,
                      "chain_rc": p.returncode}))


if __name__ == "__main__":
    main()
