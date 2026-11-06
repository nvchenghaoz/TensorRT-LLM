"""Idea 308 pair bench: the captured 8-layer SM120 CSA2 split-key decode indexer chain.

Runs in both trees (main: three ATen replica copies per layer; branch: one Triton
replicate launch per layer under capture). Wraps csa2_split_decode_bench.run_bench:
3 ratio-2 Full layers (18432 keys) + 1 ratio-1 Full layer (36864) + 4 Reindex
consumers (16384 candidates), 32 index heads (replicated per TP4 rank), captured in
one CUDA graph, 256 MB L2 flush before every replay, outputs and Top-K scratch reset
at every reset point, 10 warmup + 100 replays; median and (p90-p10)/median. NVML SM
clock is sampled during the run and reported.
"""

import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import csa2_split_decode_bench as sdb  # noqa: E402


class Clock:
    def __init__(self):
        self.samples = []
        self.stop = False
        try:
            import pynvml

            pynvml.nvmlInit()
            idx = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
            self.h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self.nv = pynvml
        except Exception as exc:  # noqa: BLE001
            print(f"nvml unavailable: {exc}", flush=True)
            self.nv = None

    def run(self):
        while not self.stop and self.nv is not None:
            try:
                self.samples.append(
                    (self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM),
                     self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.02)


def main():
    import io
    from contextlib import redirect_stdout

    class A:
        batches = os.environ.get("B308_BATCHES", "1,2,4,8")
        tokens_per_block = 128
        max_seq_len = 36864
        context = 34816
        warmup = 10
        iters = 100
        prefill = 0

    clock = Clock()
    t = threading.Thread(target=clock.run, daemon=True)
    t.start()
    real = sys.stdout
    buf = io.StringIO()

    class Tee:
        def write(self, s):
            buf.write(s)
            return real.write(s)

        def flush(self):
            real.flush()

    with redirect_stdout(Tee()):
        sdb.run_bench(A, torch.device("cuda"))
    clock.stop = True
    t.join(timeout=1)
    lines = buf.getvalue().strip().splitlines()
    result = json.loads(lines[-1])
    busy = [s for s, u in clock.samples if u >= 50]
    if busy:
        busy.sort()
        result["sm_clock_mhz"] = {"busy_samples": len(busy), "min": busy[0],
                                  "p10": busy[len(busy) // 10], "median": busy[len(busy) // 2]}
    else:
        result["sm_clock_mhz"] = {"busy_samples": 0,
                                  "all": sorted(s for s, _ in clock.samples)[-5:]}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
