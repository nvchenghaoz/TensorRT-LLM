# Step 0 for idea 3 (bf16-linear): Qwen3.8-27B-NVFP4 lm_head (N=248320, K=5120) on SM120.
# Arm A: main's Marlin W4A16 LMHead (bf16 out; "+f32" adds LogitsProcessor's logits.float()).
# Arm B: the checkpoint's own NVFP4 W4A4 mode: fp4_quantize(x, input_scale) + nvfp4_gemm
#        (cutlass,cublaslt,cuda_core autotuned), bf16 out and fp32 out.
# Real lm_head tensors from the checkpoint (only the lm_head tensors, no model load);
# synthetic activations scaled to the checkpoint's static input_scale.
# Decode M: one call per CUDA graph, 512 MiB read flush between replays, arms interleaved.
# Mixed M: eager, same flush.
# usage: python step0.py <checkpoint>
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

import lmhead_bench as lb  # noqa: E402

V, K = 248320, 5120
GRAPH_MS = [int(v) for v in os.environ.get("GRAPH_MS", "1,16,32,48,64,96,128").split(",")]
EAGER_MS = [int(v) for v in os.environ.get("EAGER_MS", "65,68,72,96,132").split(",")]
WARM, ITERS = 10, int(os.environ.get("ITERS", "60"))
BACKENDS = os.environ.get("W4A4_BACKENDS", "cutlass,cublaslt,cuda_core")
WBYTES = V * K // 2 + V * K // 16  # fp4 weight + e4m3 scales = 715 MB


def log(*a):
    print(*a, flush=True)


def build_w4a4(t):
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo
    with torch.device("cuda"):
        head = LMHead(V, K, dtype=torch.bfloat16, mapping=Mapping(),
                      tensor_parallel_mode=TensorParallelMode.COLUMN, gather_output=True,
                      reduce_output=False,
                      quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16))
    head.load_weights([{k: t[k] for k in ("weight", "weight_scale", "weight_scale_2",
                                          "input_scale")}])
    head.post_load_weights()
    head.nvfp4_allowed_backends = BACKENDS.split(",")
    log("  built W4A4 head: method=%s input_scale=%g alpha=%g" %
        (type(head.quant_method).__name__, float(head.input_scale), float(head.alpha)))
    return head


def w4a4_call(head, x, out_dtype):
    fp4, sf = torch.ops.trtllm.fp4_quantize(x, head.input_scale, 16, False)
    return torch.ops.trtllm.nvfp4_gemm(fp4, head.weight, sf, head.weight_scale, head.alpha,
                                       out_dtype, allowed_backends=BACKENDS)


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        out = fn()
    return g, out


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def stats(ts):
    med = pct(ts, 0.5)
    return {"kernel_ms": round(med, 5), "p10_ms": round(pct(ts, 0.1), 5),
            "p90_ms": round(pct(ts, 0.9), 5),
            "noise_floor": round((pct(ts, 0.9) - pct(ts, 0.1)) / med, 5),
            "TBps_weight": round(WBYTES / (med * 1e-3) / 1e12, 3)}


def main():
    from tensorrt_llm._torch.autotuner import autotune
    ckpt = sys.argv[1]
    dev = torch.cuda.get_device_properties(0)
    log("device %s sm%d%d %d SMs" % (dev.name, dev.major, dev.minor, dev.multi_processor_count))
    t = lb.load_lm_head(ckpt)
    s_in = float(t["input_scale"].float())
    marlin, arm = lb.build_head(t)
    w4 = build_w4a4(t)
    sigma = s_in * 2688.0 / 6.0
    log("  input_scale(ckpt)=%g -> amax_in=%g, synthetic sigma=%g" % (s_in, s_in * 2688, sigma))
    flush = torch.empty(512 << 18, dtype=torch.float32, device="cuda")
    fsum = torch.empty((), dtype=torch.float32, device="cuda")

    def do_flush():
        torch.sum(flush, dim=0, out=fsum)

    g = torch.Generator(device="cuda").manual_seed(4242)
    xs = {m: (torch.randn(m, K, generator=g, device="cuda") * sigma).to(torch.bfloat16)
          for m in sorted(set(GRAPH_MS + EAGER_MS))}
    # autotune W4A4 (both output dtypes) at every M
    with torch.inference_mode(), autotune():
        for m, x in xs.items():
            w4a4_call(w4, x, torch.bfloat16)
            w4a4_call(w4, x, torch.float32)
    torch.cuda.synchronize()
    log("  autotune done")
    arms = {
        "marlin_bf16": lambda x: marlin(x),
        "marlin_bf16+float": lambda x: marlin(x).float(),
        "w4a4_bf16": lambda x: w4a4_call(w4, x, torch.bfloat16),
        "w4a4_f32": lambda x: w4a4_call(w4, x, torch.float32),
    }
    res, errs = {}, {}
    for m in GRAPH_MS:
        x = xs[m]
        graphs = {k: capture(lambda f=f: f(x)) for k, f in arms.items()}
        ev = {k: ([torch.cuda.Event(enable_timing=True) for _ in range(WARM + ITERS)],
                  [torch.cuda.Event(enable_timing=True) for _ in range(WARM + ITERS)])
              for k in arms}
        for i in range(WARM + ITERS):
            for k, (gr, _) in graphs.items():
                do_flush()
                ev[k][0][i].record()
                gr.replay()
                ev[k][1][i].record()
        torch.cuda.synchronize()
        r = {}
        for k in arms:
            ts = [ev[k][0][i].elapsed_time(ev[k][1][i]) for i in range(WARM, WARM + ITERS)]
            r[k] = stats(ts)
        res["graph_M%d" % m] = r
        log("M=%d graph: %s" % (m, json.dumps({k: (v["kernel_ms"], v["p90_ms"]) for k, v in r.items()})))
        if m in (16, 64, 128):
            ref = lb.ref_fp64(t, x)
            e = {k: lb.err_stats(graphs[k][1], ref) for k in ("marlin_bf16", "w4a4_bf16", "w4a4_f32")}
            e["w4a4_vs_marlin_argmax_match"] = float(
                (graphs["w4a4_f32"][1].argmax(-1) == graphs["marlin_bf16"][1].float().argmax(-1)).float().mean())
            errs["M%d" % m] = e
            log("  M=%d err %s" % (m, json.dumps(e)))
            del ref
        del graphs
        torch.cuda.empty_cache()
    for m in EAGER_MS:
        x = xs[m]
        ev = {k: ([torch.cuda.Event(enable_timing=True) for _ in range(WARM + ITERS)],
                  [torch.cuda.Event(enable_timing=True) for _ in range(WARM + ITERS)])
              for k in arms}
        with torch.inference_mode():
            for i in range(WARM + ITERS):
                for k, f in arms.items():
                    do_flush()
                    ev[k][0][i].record()
                    f(x)
                    ev[k][1][i].record()
        torch.cuda.synchronize()
        r = {}
        for k in arms:
            ts = [ev[k][0][i].elapsed_time(ev[k][1][i]) for i in range(WARM, WARM + ITERS)]
            r[k] = stats(ts)
        res["eager_M%d" % m] = r
        log("M=%d eager: %s" % (m, json.dumps({k: (v["kernel_ms"], v["p90_ms"]) for k, v in r.items()})))
    # tactic picks
    try:
        from tensorrt_llm._torch.autotuner import AutoTuner
        cache = AutoTuner.get().profiling_cache
        picks = {str(k): str(v) for k, v in list(getattr(cache, "cache", {}).items())[:64]}
    except Exception as ex:  # noqa
        picks = {"err": repr(ex)}
    print(json.dumps({"step0": res, "errors_vs_fp64": errs, "tactics": picks,
                      "marlin_arm": arm}))


if __name__ == "__main__":
    main()
