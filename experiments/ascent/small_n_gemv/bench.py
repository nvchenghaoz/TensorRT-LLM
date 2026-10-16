# Paired microbenchmark: GDN in_proj_ba BF16 Linear [M,5120]x[5120,96] through the
# real tensorrt_llm Linear (UnquantizedLinearMethod.apply), main vs candidate.
# Self-contained and brace-free so it can be passed inline with `python -c` to
# both trees. Cold L2: each CUDA graph holds 48 calls (one per GDN layer) on 48
# distinct Linear modules/activations, and a 256 MiB read evicts L2 before every
# replay (outside the timed region). 10 warmup + 50 timed replays, CUDA events,
# per-call median / p10 / p90.
# Also times the two-stream GDN window: FP8 per-tensor in_proj_qkvz (static
# quant + trtllm::cuda_scaled_mm, N=16384, K=5120, 4 distinct 84 MB weights) on
# the main stream and in_proj_ba on an aux stream via maybe_execute_in_parallel,
# against qkvz alone.
import json

import torch

from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel, with_multi_stream)

try:
    from tensorrt_llm._torch.modules import small_n_gemv as sng
except ImportError:
    sng = None


def log(*a):
    print(*a, flush=True)


dev = torch.device("cuda")
LAYERS, WARMUP, ITERS = 48, 10, 50
N, K, NQ = 96, 5120, 16384
SINGLE_MS = [1, 2, 3, 4, 5, 8, 12, 16]
WINDOW_MS = [1, 2, 3, 4, 8]
REPORT = ["1", "2", "4"]
flush_buf = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def time_graph(body):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), with_multi_stream(True):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), with_multi_stream(True):
        body()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
           for _ in range(WARMUP + ITERS)]
    for e0, e1 in evs:
        flush_buf.sum()
        e0.record()
        g.replay()
        e1.record()
    torch.cuda.synchronize()
    us = sorted(e0.elapsed_time(e1) * 1e3 / LAYERS for e0, e1 in evs[WARMUP:])
    del g
    return (us[len(us) // 2], us[int(0.1 * len(us))], us[int(0.9 * len(us))])


def main():
    log("gpu", torch.cuda.get_device_name(), "small_n_gemv",
        sng is not None)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    lins = []
    for _ in range(LAYERS):
        lin = Linear(K, N, bias=False, dtype=torch.bfloat16,
                     tensor_parallel_mode=TensorParallelMode.COLUMN).to(dev)
        lin.weight.data.copy_(
            (torch.randn(N, K, device=dev, generator=gen) * 0.02).to(
                torch.bfloat16))
        lins.append(lin)
    log("method", type(lins[0].quant_method).__name__)
    single = dict()
    for M in SINGLE_MS:
        xs = [torch.randn(M, K, device=dev, generator=gen).to(torch.bfloat16)
              for _ in range(LAYERS)]
        routed = (sng is not None
                  and sng.maybe_small_n_gemv(xs[0], lins[0].weight) is not None)
        med, p10, p90 = time_graph(
            lambda: [lin(x) for lin, x in zip(lins, xs)])
        single[str(M)] = (med, p10, p90)
        log("in_proj_ba M=%d routed=%s per call: median %.3f us p10 %.3f "
            "p90 %.3f" % (M, routed, med, p10, p90))
        if sng is not None and M > 4:
            gm = time_graph(lambda: [
                sng.small_n_gemv(x, lin.weight) for lin, x in zip(lins, xs)])
            log("   (raw small_n_gemv at M=%d, not routed: median %.3f p90 "
                "%.3f us)" % (M, gm[0], gm[2]))
    # Two-stream GDN window.
    wq = [(torch.randn(NQ, K, device=dev, generator=gen) * 0.5).to(
        torch.float8_e4m3fn) for _ in range(4)]
    in_scale = torch.tensor([0.05], device=dev)
    w_scale = torch.tensor([0.01], device=dev)
    aux = torch.cuda.Stream()
    ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()

    def qkvz(x, i):
        qx, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x, in_scale)
        return torch.ops.trtllm.cuda_scaled_mm(qx, wq[i % 4].t(),
                                               scale_a=in_scale,
                                               scale_b=w_scale,
                                               bias=None,
                                               out_dtype=torch.bfloat16)

    window = dict()
    for M in WINDOW_MS:
        xs = [torch.randn(M, K, device=dev, generator=gen).to(torch.bfloat16)
              for _ in range(LAYERS)]
        q_only = time_graph(lambda: [qkvz(x, i) for i, x in enumerate(xs)])

        def win():
            for i, (lin, x) in enumerate(zip(lins, xs)):
                maybe_execute_in_parallel((lambda x=x, i=i: qkvz(x, i)),
                                          (lambda x=x, lin=lin: lin(x)), ev0,
                                          ev1, aux)

        w = time_graph(win)
        window[str(M)] = (w, q_only)
        log("window M=%d: qkvz alone median %.3f p90 %.3f us | qkvz||ba "
            "median %.3f p90 %.3f us | exposed %.3f us" %
            (M, q_only[0], q_only[2], w[0], w[2], w[0] - q_only[0]))
    batch = dict()
    for b in REPORT:
        med, p10, p90 = single[b]
        batch[b] = dict(kernel_ms=med * 1e-3,
                        noise_floor=(p90 - p10) / med,
                        p90_ms=p90 * 1e-3)
    extra = dict(
        single_us=dict((m, dict(median=v[0], p10=v[1], p90=v[2]))
                       for m, v in single.items()),
        window_us=dict((m, dict(window_median=v[0][0],
                                window_p90=v[0][2],
                                qkvz_alone_median=v[1][0],
                                qkvz_alone_p90=v[1][2],
                                exposed_median=v[0][0] - v[1][0]))
                       for m, v in window.items()),
        small_n_gemv=sng is not None,
    )
    out = dict(batch=batch,
               output=dict(cls="reordered" if sng is not None else "exact"),
               extra=extra)
    out["output"]["class"] = out["output"].pop("cls")
    print(json.dumps(out), flush=True)


main()
