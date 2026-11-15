"""Idea 73 op pair (run_gpu_pair: same script in base and branch trees). Times the Gemma4 qkv
projection exactly as modeling_gemma4 routes it on this tree: a Linear with main's SM120 cublas_mm
token gate, plus (branch tree only) the sequential-K route via _route_sm120_seqk_gemm. Base tree:
main's dispatch. Synthetic BF16 tensors with the checkpoint's shapes, no model load.

Timing: cold L2 (4 weight copies per shape, 4 x >=176 MB), calls enqueued back to back with a CUDA
event between consecutive calls (GPU-side duration of each call), 8 warmup + 40 timed calls per
round, 3 rounds alternating case order; median with p10/p90 over all timed calls. SM clock,
temperature and throttle reasons sampled with NVML every 10 ms during the timed loops.
Derived: qkv ms per M=300 forward = 50 x sliding + 10 x global (layer counts from config).

usage: pair_bench.py <ckpt>"""
import json
import os
import statistics
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ckpt_dims, gpu_receipt, preheat  # noqa: E402


class Nvml:
    def __init__(self):
        self.samples = []
        self.on = False
        try:
            import pynvml
            pynvml.nvmlInit()
            idx = 0
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            n = pynvml.nvmlDeviceGetCount()
            self.h = None
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                uuid = pynvml.nvmlDeviceGetUUID(h)
                uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
                if vis and uuid in vis:
                    self.h = h
            if self.h is None:
                self.h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self.nv = pynvml
        except Exception as ex:  # noqa: BLE001
            self.nv = None
            print("pynvml unavailable:", ex, flush=True)

    def _loop(self):
        nv = self.nv
        while self.on:
            try:
                self.samples.append((nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM),
                                     nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU),
                                     nv.nvmlDeviceGetCurrentClocksThrottleReasons(self.h)))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.01)

    def start(self):
        if self.nv is None:
            return
        self.on = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self):
        if self.nv is None:
            return {}
        self.on = False
        self.t.join()
        if not self.samples:
            return {}
        clk = sorted(s[0] for s in self.samples)
        reasons = 0
        for s in self.samples:
            reasons |= s[2]
        out = {"sm_clock_mhz_median": clk[len(clk) // 2], "sm_clock_min": clk[0], "sm_clock_max": clk[-1],
               "temp_c_max": max(s[1] for s in self.samples), "throttle_reasons_or": hex(reasons),
               "n": len(self.samples)}
        self.samples = []
        return out


def make_linear(K, N, w, mg):
    from tensorrt_llm._torch.modules.linear import Linear
    lin = Linear(K, N, bias=False, dtype=torch.bfloat16)
    lin.weight.data = w
    route = mg._SM120_BF16_CUBLAS_MM_ROUTES.get((K, N))
    if route is not None:
        lin.use_custom_cublas_mm = True
        lin.custom_cublas_mm_tokens = mg._CublasMmTokenGate(route)
    routed = hasattr(mg, "_route_sm120_seqk_gemm")
    if routed:
        mg._route_sm120_seqk_gemm(lin)
    return lin, routed and type(lin).__name__.startswith("_Sm120SeqK")


def time_calls(fns, warm=8, iters=40):
    n = len(fns)
    for i in range(warm):
        fns[i % n]()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
    ev[0].record()
    for i in range(iters):
        fns[(warm + i) % n]()
        ev[i + 1].record()
    torch.cuda.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) for i in range(iters)]


def stats(ts):
    s = sorted(ts)
    med = statistics.median(s)
    p10 = s[int(0.1 * (len(s) - 1))]
    p90 = s[int(round(0.9 * (len(s) - 1)))]
    return med, p10, p90


def main():
    ckpt = sys.argv[1]
    dims, shapes = ckpt_dims(ckpt)
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    t = cfg.get("text_config", cfg)
    types = t.get("layer_types") or []
    n_sl = sum(1 for x in types if x == "sliding_attention") or 50
    n_gl = (len(types) - n_sl) if types else 10
    print("layers sliding", n_sl, "global", n_gl, flush=True)
    print("receipt", gpu_receipt(), flush=True)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    nvml = Nvml()
    cases = [("sl_qkv", 300), ("gl_qkv", 300), ("sl_qkv", 288), ("sl_qkv", 301), ("sl_qkv", 320),
             ("gl_qkv", 301), ("gl_qkv", 320), ("sl_qkv", 1200), ("gl_qkv", 1200)]
    torch.manual_seed(0)
    lins = {}
    routed_any = {}
    for sn in ("sl_qkv", "gl_qkv"):
        N, K = shapes[sn]
        lins[sn] = []
        for _ in range(4):
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
            lin, routed = make_linear(K, N, w, mg)
            lins[sn].append(lin)
            routed_any[sn] = routed
    print("seqk routed:", routed_any, flush=True)
    xs = {(sn, M): torch.randn(M, shapes[sn][1], device="cuda", dtype=torch.bfloat16) for sn, M in cases}
    preheat(2.0)
    times = {c: [] for c in cases}
    clocks = []
    for rnd in range(3):
        order = cases if rnd % 2 == 0 else list(reversed(cases))
        for sn, M in order:
            x = xs[(sn, M)]
            fns = [lambda lin=lin: lin(x) for lin in lins[sn]]
            nvml.start()
            times[(sn, M)] += time_calls(fns)
            c = nvml.stop()
            if c:
                clocks.append(c)
        print("round", rnd, "done", flush=True)
    batch = {}
    for (sn, M), ts in times.items():
        med, p10, p90 = stats(ts)
        batch[f"{sn}@{M}"] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med, "p10": p10, "p90": p90,
                              "n": len(ts)}
        print(f"{sn}@{M}: {med*1e3:.1f} us (p10 {p10*1e3:.1f} p90 {p90*1e3:.1f})", flush=True)
    sl, gl = batch["sl_qkv@300"], batch["gl_qkv@300"]
    batch["qkv_per_forward@300"] = {
        "kernel_ms": n_sl * sl["kernel_ms"] + n_gl * gl["kernel_ms"],
        "noise_floor": max(sl["noise_floor"], gl["noise_floor"]),
        "formula": f"{n_sl} x sl_qkv@300 + {n_gl} x gl_qkv@300"}
    clk = {}
    if clocks:
        clk = {"sm_clock_mhz_median_of_cases": sorted(c["sm_clock_mhz_median"] for c in clocks)[len(clocks) // 2],
               "sm_clock_min": min(c["sm_clock_min"] for c in clocks),
               "temp_c_max": max(c["temp_c_max"] for c in clocks),
               "throttle_reasons_or": hex(eval("|".join(c["throttle_reasons_or"] for c in clocks)))}
    print("receipt_end", gpu_receipt(), flush=True)
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "seqk_routed": routed_any,
                      "nvml": clk}))


if __name__ == "__main__":
    main()
