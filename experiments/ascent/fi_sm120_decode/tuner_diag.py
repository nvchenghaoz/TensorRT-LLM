# Idea 43 diag: what the AutoTuner's own profiler (_profile_single_kernel, cold-L2 input
# clones, CUDA graph of `repeat` calls) measures for main's top tactics vs the fi_cutlass
# tactics at M=64/128, repeated R times, interleaved. Shows whether the tuner can rank them.
import json
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import nvfp4_coldtune as ct  # noqa: E402

SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o_out": (5120, 6144),
          "down": (5120, 17408)}
MS = (64, 128)
R = 7


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    ct.MS = MS
    tuner = AutoTuner.get()
    cfg = NVFP4GemmUnifiedRunner.tuning_config
    r = ct.runner()
    out = {}
    for nm, (n, k) in SHAPES.items():
        s = ct.Shape(nm, n, k)
        for m in MS:
            inp = s.inputs(m)
            tacs = r.get_valid_tactics(inp, OptimizationProfile())
            times = {t: [] for t in tacs}
            for rep in range(R):
                for t in tacs:
                    times[t].append(tuner._profile_single_kernel(r, inp, t, cfg,
                                                                 use_cuda_graph=True) * 1e3)
            med = {"%s%d" % t: sorted(v)[len(v) // 2] for t, v in times.items()}
            spread = {"%s%d" % t: (max(v) - min(v)) for t, v in times.items()}
            top = sorted(med.items(), key=lambda kv: kv[1])[:8]
            key = "%s/%d" % (nm, m)
            out[key] = {"top8_median_us": top,
                        "spread_us_of_top": {k: round(spread[k], 2) for k, _ in top}}
            ct.log("%s top8 (tuner profiler, median of %d): %s" % (
                key, R, ", ".join("%s %.2f(+-%.2f)" % (k, v, spread[k] / 2) for k, v in top)))
        del s
        torch.cuda.empty_cache()
    print(json.dumps(out))


main()
