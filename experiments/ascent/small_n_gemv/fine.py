# Fine config sweep for the small-N BF16 GEMV at in_proj_ba (N=96, K=5120),
# single-stream per call and in the two-stream GDN window: FP8 per-tensor
# in_proj_qkvz (static quant + trtllm::cuda_scaled_mm, N=16384, K=5120) on the
# main stream, in_proj_ba on an aux stream (maybe_execute_in_parallel).
# Cold L2: 48 distinct in_proj_ba weights, 4 distinct 84 MB qkvz weights and a
# 256 MiB read before every replay. 10 warmup + 50 timed replays.
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules import small_n_gemv as sng
from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel, with_multi_stream)


def log(*a):
    print(*a, flush=True)


dev = torch.device("cuda")
LAYERS, WARMUP, ITERS = 48, 10, 50
N, K, NQ = 96, 5120, 16384
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
    return us[len(us) // 2], us[int(0.9 * len(us))]


def main():
    gen = torch.Generator(device=dev)
    gen.manual_seed(11)
    ws = [(torch.randn(N, K, device=dev, generator=gen) * 0.02).to(
        torch.bfloat16) for _ in range(LAYERS)]
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

    cfgs = [(1, 1, 128, 4), (1, 1, 256, 4), (1, 1, 256, 8), (1, 1, 256, 16),
            (1, 1, 512, 8), (1, 1, 512, 16), (1, 2, 256, 8), (1, 2, 512, 8),
            (2, 1, 256, 8), (1, 2, 1024, 4)]
    for M in (1, 2, 3, 4, 5, 6, 8):
        xs = [torch.randn(M, K, device=dev, generator=gen).to(torch.bfloat16)
              for _ in range(LAYERS)]
        arms = [("cublas", lambda x, w: F.linear(x, w))]
        for c in cfgs:
            arms.append((str(c), (lambda c: lambda x, w: sng.small_n_gemv(
                x, w, c))(c)))
        q_only = time_graph(lambda: [qkvz(x, i) for i, x in enumerate(xs)])
        log("M=%d qkvz alone med %.2f p90 %.2f us" % (M, q_only[0], q_only[1]))
        for name, fn in arms:
            single = time_graph(
                lambda: [fn(x, w) for x, w in zip(xs, ws)])

            def win():
                for i, (x, w) in enumerate(zip(xs, ws)):
                    maybe_execute_in_parallel(
                        (lambda x=x, i=i: qkvz(x, i)),
                        (lambda x=x, w=w: fn(x, w)), ev0, ev1, aux)

            window = time_graph(win)
            log("M=%d %-18s single med %.2f p90 %.2f | window med %.2f p90 "
                "%.2f | window-qkvz %.2f us" %
                (M, name, single[0], single[1], window[0], window[1],
                 window[0] - q_only[0]))


main()
