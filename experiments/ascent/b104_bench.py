"""40-layer wo_a chain timing (eager, CUDA events) at M = 16/64/512/2048, TP4 rank shapes."""
import json
import statistics

import torch
from b104_common import ITERS, LAYERS, WARMUP, chain, make_inputs, make_layer_weights, mla, sm_clock_mhz

from tensorrt_llm._utils import get_sm_version


def main():
    torch.manual_seed(0)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    sm = get_sm_version()
    direct = hasattr(mla, "_sm120_bmm_out_direct_ok")
    print(f"sm={sm} candidate_direct_helper={direct}", flush=True)
    weights = make_layer_weights(gen)
    result = {}
    for m in (16, 64, 512, 2048):
        inputs = make_inputs(gen, m)
        if direct:
            probe = torch.empty([m, 2, 1024], device="cuda", dtype=torch.bfloat16).transpose(0, 1)
            print(f"M={m} direct_ok={mla._sm120_bmm_out_direct_ok(probe)}", flush=True)

        def run_once():
            for i in range(LAYERS):
                chain(inputs[i], weights[i][0], weights[i][1])

        for _ in range(WARMUP):
            run_once()
        torch.cuda.synchronize()
        times = []
        for rep in range(3):  # 3 blocks of ITERS timed iterations
            for _ in range(ITERS):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                run_once()
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e))
        times.sort()
        med = statistics.median(times)
        p10 = times[int(0.1 * (len(times) - 1))]
        p90 = times[int(0.9 * (len(times) - 1))]
        noise = (p90 - p10) / med
        result[str(m)] = {"kernel_ms": med, "noise_floor": noise}
        print(
            f"M={m} 40-layer chain median={med:.4f} ms p10={p10:.4f} p90={p90:.4f} "
            f"noise={noise:.4f} clock={sm_clock_mhz()}",
            flush=True,
        )
        del inputs
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "batch": result,
                "output": {"class": "exact"},
                "timing": "eager, CUDA events, 40-layer chain per iteration, 45 iterations",
            }
        )
    )


if __name__ == "__main__":
    main()
