"""run_gpu_pair benchmark for idea 372: the CSA2 FI shared-prefill call as each tree runs it.

Same production dispatch (CSA2FlashInfer._forward_selected) and harness as pair.py (branch
197). Batch labels 1/2/4 are one prefill step's 40 CSA2 attention calls per concurrency:
c1 = 10 x {2048x640, 2048x128} + 10 x {128x640, 128x128}; c2 = 4096/256 rows; c4 = 6144/384
rows. Eager, back to back, CUDA events around the whole 40-call sequence (host-bound calls
show their host time). Per-shape times and host us per call (perf_counter over 500
back-to-back calls, one sync at the end) go in separate fields.
Last stdout line: {"batch": {label: {"step_ms", "noise_floor", ...}}, "output": {"class": "exact"}, ...}.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pair import Case, timed  # noqa: E402
from stage0 import Scenario, log  # noqa: E402


def host_us(fn, n=500):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    h = (time.perf_counter() - t) / n * 1e6
    torch.cuda.synchronize()
    return h


def sm_clock():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


def main():
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import fi_sparse_prefill as fsp

    tree = "branch (direct execute)" if hasattr(fsp, "_launch_index_kernel") else "base (wrapper.run)"
    log("tree:", tree, "clock:", sm_clock())
    shapes = []
    for r in (1, 2, 3):
        for t in (2048, 128):
            for hm in (True, False):
                shapes.append((r, t, hm))
    cases = {}
    for i, (r, t, hm) in enumerate(shapes):
        s = Scenario(r, t, hm, seed=1000 + i)
        cases[s.name] = Case(s)
        cases[s.name]()
    torch.cuda.synchronize()
    per_shape, host = {}, {}
    for name, c in cases.items():
        per_shape[name] = timed(c, 30, 3)
        if name.startswith(("128x", "384x")):
            host[name] = host_us(c)
        log(f"{name}: {per_shape[name]['kernel_ms']:.4f} ms (noise {per_shape[name]['noise_floor']:.3f})"
            + (f" host {host[name]:.1f} us/call" if name in host else ""))
    seqs = {
        "1": ["2048x640", "2048x128"] * 10 + ["128x640", "128x128"] * 10,
        "2": ["4096x640", "4096x128"] * 10 + ["256x640", "256x128"] * 10,
        "4": ["6144x640", "6144x128"] * 10 + ["384x640", "384x128"] * 10,
    }
    batch = {}
    for rep in range(2):
        for label, names in seqs.items():
            def run(names=names):
                for nm in names:
                    cases[nm]()

            r = timed(run, 30, 3)
            if rep == 1:
                batch[label] = {"step_ms": r["kernel_ms"], "noise_floor": r["noise_floor"],
                                "p10_ms": r["p10_ms"], "p90_ms": r["p90_ms"]}
                log(f"step c{label} (40 calls): {r['kernel_ms']:.3f} ms p10 {r['p10_ms']:.3f} p90 {r['p90_ms']:.3f}")
    tail = {}
    for label, names in (("1", ["128x640", "128x128"] * 10), ("4", ["384x640", "384x128"] * 10)):
        def run(names=names):
            for nm in names:
                cases[nm]()

        r = timed(run, 30, 3)
        tail[label] = r
        log(f"L20-39 tail c{label} (20 calls): {r['kernel_ms']:.3f} ms")
    print(json.dumps({
        "batch": batch,
        "output": {"class": "exact"},
        "l20_39_tail_20_calls": tail,
        "per_shape": per_shape,
        "host_us_per_call_500_back_to_back": host,
        "sm_clock_end": sm_clock(),
        "tree": tree,
    }))


if __name__ == "__main__":
    sys.exit(main())
