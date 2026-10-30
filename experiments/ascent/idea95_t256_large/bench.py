"""Idea 95 (denoiser_gemm, branch 61): route i2i single 32848 (M >= 24576) to the t256 fused pair +
flag-wait residual (lpt_pair_residual_t256, landing 27's instance) instead of fused-128 AlongN r2.

Reuses experiments/ascent/idea55_t256/bench.py (synthetic NVFP4 shapes: down [3072 x 9216] K=9216 +
attn_out [3072 x 3072], i2i single 32848 form [1,3,D] table + u8 index).

modes:
  step0 : one process, 20-call graphs at 32848 u8x3, seeded random order, A/A twins: routed (the
          tree's served route), f128 r2 (+A/A), t256 r2 (+A/A), t256 r1. SM MHz / power sampled.
  bench : run_gpu_pair arm (both trees): per-call graphs at 32848 u8x3 (routed op), plus one i2i
          denoise step of fused tails (27 x 32848 u8x3 + 9 x 16384 row + 1 x 16384 u8s) and the t2i
          step (27 x 16464 select + 4 x 16384 row + 1 x 16384 sel).
  check : branch only, untimed, bitwise (see check()).
"""

import json
import os
import random
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "idea55_t256"))
import bench as b55  # noqa: E402

H = b55.H
log = b55.log
SETS = 3


class Sampler:
    """nvidia-smi SM clock / power / temperature samples while timing."""

    def __init__(self):
        try:
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            self.sel = ["--id=" + (uuid if uuid.startswith("GPU-") else "GPU-" + uuid)]
        except Exception:  # noqa: BLE001
            self.sel = []
        self.samples = []
        self.stop = False
        self.t = threading.Thread(target=self.run, daemon=True)

    def one(self):
        try:
            r = subprocess.run(["nvidia-smi", *self.sel, "--query-gpu=clocks.sm,power.draw,"
                                "temperature.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            v = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
            return v
        except Exception:  # noqa: BLE001
            return None

    def run(self):
        while not self.stop:
            v = self.one()
            if v:
                self.samples.append(v)
            time.sleep(0.5)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join(timeout=15)

    def summary(self):
        if not self.samples:
            return {"n": 0}
        out = {"n": len(self.samples)}
        for i, k in enumerate(("sm_mhz", "power_w", "temp_c")):
            v = sorted(s[i] for s in self.samples)
            out[k] = {"median": v[len(v) // 2], "min": v[0], "max": v[-1]}
        return out


def table_rows(res, base_key):
    base = res[base_key]
    row = {}
    for n, v in res.items():
        med, p10, p90 = b55.stat(v)
        diffs = sorted(b - a for a, b in zip(base, v))
        row[n] = {"us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
                  "noise_floor": round((p90 - p10) / med, 4),
                  f"med_paired_diff_vs_{base_key}_us": round(diffs[len(diffs) // 2], 2),
                  "ratio": round(med / b55.stat(base)[0], 4)}
    return row


def step0():
    lptr = b55.preload()
    M, form, calls = 32848, "u8x3", 20
    cs = [b55.make_case(M, 7 * M + i + len(form), form) for i in range(SETS)]
    s = torch.cuda.Stream()

    def seq(f):
        return [lambda c=cs[i % SETS]: f(c) for i in range(calls)]

    graphs = {}
    graphs["routed"], _ = b55.capture(seq(b55.run_fused), s)
    graphs["f128_r2"], _ = b55.capture(seq(lambda c: b55.run_forced(c, "128", 2)), s)
    graphs["f128_r2_AA"], _ = b55.capture(seq(lambda c: b55.run_forced(c, "128", 2)), s)
    graphs["t256_r2"], _ = b55.capture(seq(lambda c: b55.run_forced(c, "t256", 2)), s)
    graphs["t256_r2_AA"], _ = b55.capture(seq(lambda c: b55.run_forced(c, "t256", 2)), s)
    graphs["t256_r1"], _ = b55.capture(seq(lambda c: b55.run_forced(c, "t256", 1)), s)
    samples = int(os.environ.get("IDEA95_SAMPLES", "100"))
    with Sampler() as smp:
        res = b55.time_graphs(graphs, samples=samples, warm=10, calls={k: calls for k in graphs},
                              seed=95)
    row = table_rows(res, "f128_r2")
    log(f"M={M} {form}: {json.dumps(row, indent=1)}")
    log(f"clocks: {smp.summary()}")
    print(json.dumps({"step0_us_per_call": {f"M{M}_{form}": row}, "calls_per_replay": calls,
                      "samples": samples, "gpu": smp.summary(),
                      "route_32848": b55.route_of(lptr, M), "raster_for_32848": lptr.raster_for(M)}))


I2I = [(32848, "u8x3", 27), (16384, "row", 9), (16384, "u8s", 1)]
T2I = b55.CASES["t2i-2048"]


def bench():
    lptr = b55.preload()
    log(f"route_32848={b55.route_of(lptr, 32848)} raster={lptr.raster_for(32848)}")
    try:
        log(subprocess.run(["git", "log", "-3", "--oneline"], capture_output=True, text=True,
                           timeout=20).stdout)
    except Exception as e:  # noqa: BLE001
        log(f"git log unavailable: {e}")
    pool = {}
    for parts in (I2I, T2I):
        for M, form, _ in parts:
            if (M, form) not in pool:
                pool[(M, form)] = [b55.make_case(M, 1000 * M + 17 * i + len(form), form)
                                   for i in range(SETS)]
    pristine = {k: [[t.clone() for t in b55.inputs(c)] for c in cs] for k, cs in pool.items()}
    s = torch.cuda.Stream()
    graphs, calls = {}, {}
    cs = pool[(32848, "u8x3")]
    graphs["M32848_u8x3"], _ = b55.capture([lambda c=cs[i % SETS]: b55.run_fused(c)
                                            for i in range(20)], s)
    calls["M32848_u8x3"] = 20
    for case, parts in (("i2i-2048", I2I), ("t2i-2048", T2I)):
        seq = []
        for M, form, n in parts:
            cc = pool[(M, form)]
            seq += [lambda c=cc[i % SETS]: b55.run_fused(c) for i in range(n)]
        graphs[case], _ = b55.capture(seq, s)
        calls[case] = 1
    with Sampler() as smp:
        res = b55.time_graphs(graphs, samples=100, warm=10, calls=calls, seed=95)
    detail = {}
    for k, v in res.items():
        med, p10, p90 = b55.stat(v)
        detail[k] = {"us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
                     "noise_floor": round((p90 - p10) / med, 4)}
        log(f"{k}: median {med:.2f} us p10 {p10:.2f} p90 {p90:.2f}")
    for k, cc in pool.items():
        for p, c in zip(pristine[k], cc):
            assert all(torch.equal(x, y) for x, y in zip(p, b55.inputs(c))), "inputs changed"
    print(json.dumps({
        "batch": {k: {"kernel_ms": detail[k]["us"] / 1e3, "noise_floor": detail[k]["noise_floor"]}
                  for k in ("M32848_u8x3", "i2i-2048", "t2i-2048")},
        "detail_us": detail, "gpu": smp.summary(),
        "unit": "M32848_u8x3: ms per call (20-call graph, routed op); i2i-2048 / t2i-2048: one "
                "CUDA graph replay = one denoise step of the case's fused pair+residual tails",
        "output": {"class": "exact"},
    }))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    {"step0": step0, "bench": bench}[mode]()
