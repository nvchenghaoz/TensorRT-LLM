"""Branch 54 / idea 84: cost of build_rope_compact as the t2i / i2i prepare runs it (one table per
stream + the joint table), host time per call, GPU time (events), and whether it synchronizes
(torch.cuda.set_sync_debug_mode). Also times the model's own _packed_positions with the lever on
and off on a synthetic Flux3PosEmbed (no model load). Last stdout line: JSON."""

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rope import make_cos_sin  # noqa: E402


def main():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi

    dev = torch.device("cuda")
    tables = {"text80": make_cos_sin(16464, dev)[:80].contiguous(),
              "image16384": make_cos_sin(16384, dev),
              "joint16464": make_cos_sin(16464, dev),
              "cond16384": make_cos_sin(16384, dev, cond=True),
              "joint32848": make_cos_sin(32848, dev)}
    for t in tables.values():
        epi.build_rope_compact(t)
    torch.cuda.synchronize()
    out = {}
    for name, t in tables.items():
        host, gpu = [], []
        for _ in range(20):
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            h0 = time.perf_counter()
            epi.build_rope_compact(t)
            h1 = time.perf_counter()
            e1.record()
            torch.cuda.synchronize()
            host.append(1e3 * (h1 - h0))
            gpu.append(e0.elapsed_time(e1))
        host.sort()
        gpu.sort()
        out[name] = {"host_ms_med": round(host[10], 3), "gpu_ms_med": round(gpu[10], 3)}
        print(name, out[name], flush=True)
    torch.cuda.set_sync_debug_mode("error")
    syncs = "none"
    try:
        epi.build_rope_compact(tables["joint16464"])
    except RuntimeError as e:
        syncs = str(e)[:300]
    torch.cuda.set_sync_debug_mode(0)
    # Profiler op breakdown for one joint16464 build.
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        epi.build_rope_compact(tables["joint16464"])
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15), flush=True)
    print(json.dumps({"cost": out, "sync": syncs}))


if __name__ == "__main__":
    main()
