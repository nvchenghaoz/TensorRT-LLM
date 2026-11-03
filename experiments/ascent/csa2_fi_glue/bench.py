"""Time one decode step's CSA2FlashInfer adapter (40 layers) under CUDA graph replay.

Baseline tree: prepare_selected glue (cat + valid) + CSA2FlashInfer._forward + output.copy_.
Candidate tree: the fused _forward_pools path. Same synthetic inputs (fixed seed) in both.
Prints {"batch": {count: {"kernel_ms", "noise_floor"}}, "output": {"class": "exact"}}.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXTRA, adapter_call, has_fused, log, make_layer, new_adapter  # noqa: E402

LAYERS = 40
COUNTS = [int(c) for c in os.environ.get("CSA2_GLUE_COUNTS", "1,2,4,8").split(",")]
SAMPLES = 25
REPLAYS = 5


def main():
    torch.cuda.set_device(0)
    log("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    adapters = [new_adapter() for _ in range(LAYERS)]
    log("fused path:", has_fused(adapters[0]))
    results = {}
    for count in COUNTS:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1234 + count)
        layers = [make_layer(count, gen, EXTRA) for _ in range(LAYERS)]
        outs = [torch.empty(count, 16 * 512, dtype=torch.bfloat16, device="cuda") for _ in layers]

        def step():
            for inst, layer, out in zip(adapters, layers, outs):
                adapter_call(inst, layer, out)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(SAMPLES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPLAYS):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / REPLAYS)
        times.sort()
        median = times[len(times) // 2]
        p10 = times[int(0.1 * (len(times) - 1))]
        p90 = times[int(0.9 * (len(times) - 1))]
        results[str(count)] = {
            "kernel_ms": round(median, 5),
            "noise_floor": round((p90 - p10) / median, 5),
            "per_layer_us": round(median * 1000 / LAYERS, 3),
        }
        log(f"count {count}: step {median:.4f} ms ({LAYERS} layers), p10 {p10:.4f} p90 {p90:.4f}")
        del graph, layers, outs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(json.dumps({"batch": results, "output": {"class": "exact"}}))


if __name__ == "__main__":
    main()
