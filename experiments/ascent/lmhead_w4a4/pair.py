# Paired lm_head benchmark (run_gpu_pair: main tree vs branch tree), Qwen3.8-27B-NVFP4 SM120.
# Arm-agnostic: builds the tree's own LMHead from the checkpoint's {NVFP4, group 16} lm_head
# entry (lmhead_bench.build_head -> modeling_qwen3_5 routing + LMHead.get_quant_method):
# main = Marlin W4A16; branch = Marlin below TRTLLM_LMHEAD_W4A4_MIN_ROWS, W4A4 (fp4_quantize +
# nvfp4_gemm, FP32 out) at/above. Timed op = what LogitsProcessor runs: lm_head(x).float().
# Autotuner warmup emulated as in serving: one lm_head call with 1 row under autotune().
# Decode M in GRAPH_MS: one call per CUDA graph; mixed M in EAGER_MS: eager. 512 MiB read
# flush before every timed call (eager: plus a sync, so host launch overhead is timed);
# 10 warmup + ITERS timed; median/p10/p90.
# usage: python pair.py <checkpoint>
import json
import os
import sys

# run inline (python -c) from the tree root: the baseline tree has no copy of this file
sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import torch  # noqa: E402

import lmhead_bench as lb  # noqa: E402

K = 5120
GRAPH_MS = [int(v) for v in os.environ.get("GRAPH_MS", "1,16,32,64,128").split(",") if v]
EAGER_MS = [int(v) for v in os.environ.get(
    "EAGER_MS", "48,52,56,60,63,65,68,72,96,97,104,112,115,132").split(",") if v]
WARM, ITERS = 10, int(os.environ.get("ITERS", "100"))


def log(*a):
    print(*a, flush=True)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def main():
    from tensorrt_llm._torch.autotuner import autotune
    ckpt = sys.argv[1]
    t = lb.load_lm_head(ckpt)
    head, arm = lb.build_head(t)
    min_rows = head.__dict__.get("_w4a4_min_rows", None)
    log("arm %s w4a4_min_rows %s" % (arm, min_rows))
    sigma = float(t["input_scale"].float()) * 2688.0 / 6.0
    g = torch.Generator(device="cuda").manual_seed(4242)
    xs = {m: (torch.randn(m, K, generator=g, device="cuda") * sigma).to(torch.bfloat16)
          for m in sorted(set(GRAPH_MS + EAGER_MS))}
    with torch.inference_mode(), autotune():
        head(xs[min(xs)][:1])
    torch.cuda.synchronize()
    flush = torch.empty(512 << 18, dtype=torch.float32, device="cuda")
    fsum = torch.empty((), dtype=torch.float32, device="cuda")

    def fn(x):
        return head(x).float()

    res = {}
    for m in GRAPH_MS + EAGER_MS:
        x = xs[m]
        eager = m in EAGER_MS and m not in GRAPH_MS
        if not eager:
            gr, _ = lb.capture(fn, x)
        e0 = [torch.cuda.Event(enable_timing=True) for _ in range(WARM + ITERS)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(WARM + ITERS)]
        with torch.inference_mode():
            for i in range(WARM + ITERS):
                torch.sum(flush, dim=0, out=fsum)
                if eager:
                    # host overhead counts: the GPU is idle when e0 is recorded
                    torch.cuda.synchronize()
                e0[i].record()
                if eager:
                    fn(x)
                else:
                    gr.replay()
                e1[i].record()
        torch.cuda.synchronize()
        ts = [e0[i].elapsed_time(e1[i]) for i in range(WARM, WARM + ITERS)]
        med = pct(ts, 0.5)
        key = str(m) if not eager else "eager%d" % m
        res[key] = {"kernel_ms": round(med, 5),
                    "noise_floor": round((pct(ts, 0.9) - pct(ts, 0.1)) / med, 5),
                    "p10_ms": round(pct(ts, 0.1), 5), "p90_ms": round(pct(ts, 0.9), 5),
                    "mode": "eager" if eager else "graph", "n": ITERS}
        log("M=%d %s median %.4f p10 %.4f p90 %.4f" % (m, res[key]["mode"], med, pct(ts, 0.1),
                                                     pct(ts, 0.9)))
        if not eager:
            del gr
        torch.cuda.empty_cache()
    cls = "mode_switch" if min_rows else "exact"  # baseline arm = main's own lm_head
    print(json.dumps({"batch": res, "output": {
        "class": cls, "arm": arm, "w4a4_min_rows": min_rows,
        "note": "mode_switch: checkpoint lm_head quant entry NVFP4 group 16 with static "
                "input_scale (W4A4) at rows >= min_rows; error vs FP64 reference in check.py"}}))


main()
