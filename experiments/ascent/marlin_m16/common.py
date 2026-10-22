# Shared helpers for the dense Marlin NVFP4 rows 9-16 launch-shape sweep / pair / check.
# Builds the same load-time NVFP4 W4A16 Marlin copies that linear.py's
# _build_fp8_nvfp4_decode_copy builds from an FP8 per-tensor weight, plus an
# independent FP32 reference from the dequantized NVFP4 operands.
import os

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

from tensorrt_llm._torch.modules import linear as L  # noqa: E402

dev = torch.device("cuda")
LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                   dtype=torch.float32, device=dev)

# (N, K) -> distinct copies so that copies x bytes >= 400 MB (cold L2).
SHAPES = {
    (5120, 6144): 24,   # GDN out_proj x48 + attn o_proj x16, 17.7 MB each
    (16384, 5120): 9,   # GDN in_proj_qkvz x48, 47.2 MB each
    (14336, 5120): 10,  # attn qkv_proj x16, 41.3 MB each
}
CALLS = {(5120, 6144): 64, (16384, 5120): 48, (14336, 5120): 16}


def log(*a):
    print(*a, flush=True)


def nbytes(n, k):
    return n * k // 2 + n * k // 16


def make_copy(n, k, seed, want_ref=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=dev) * 0.02
    w = w * (torch.rand(n, 1, generator=g, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    wv = w8.to(torch.bfloat16)
    amax = wv.abs().amax().float() * s
    gg = torch.exp2(torch.ceil(torch.log2(amax / 2688.0)))
    fp4, sf = torch.ops.trtllm.fp4_quantize(wv, (s / gg).reshape(1), 16, False, True)
    del wv
    mw, ms_, mg = L._pack_nvfp4_weight_for_marlin(fp4, sf, gg.reshape(1).to(torch.float32), n, k, 16)
    ref = None
    if want_ref:
        sf2 = torch.ops.trtllm.block_scale_interleave_reverse(sf.view(n, -1))[:, :k // 16].contiguous()
        sb = sf2.view(torch.uint8)
        pk = fp4.view(torch.uint8)
        vals = torch.stack([LUT[(pk & 0xF).long()], LUT[(pk >> 4).long()]], -1).reshape(n, k)
        w_deq = (vals.view(n, k // 16, 16) * sf2.view(torch.float8_e4m3fn).float().unsqueeze(-1)).reshape(n, k) * gg
        # Marlin flushes blocks whose E4M3 scale is subnormal.
        sub_blk = (((sb & 0x78) == 0) & ((sb & 7) != 0))
        w_ref = (w_deq.view(n, k // 16, 16) * (~sub_blk).unsqueeze(-1)).reshape(n, k)
        ref = dict(w_ref=w_ref, n_sub=int(sub_blk.sum().item()))
        del w_deq, vals
    del fp4, sf, w8
    return (mw, ms_, mg), ref


def gemm(x, c, n, k):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, c[0], scale_a=None, scale_b=c[1], alpha=None,
                                              weight_global_scale=c[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=n, size_k=k)


def set_cfg(cfg):
    """None -> this tree's default; 'main' -> table off; else explicit TRTLLM_MARLIN_M16_CFG."""
    os.environ.pop("TRTLLM_MARLIN_M16_CFG", None)
    os.environ.pop("TRTLLM_MARLIN_M16_TABLE", None)
    if cfg == "main":
        os.environ["TRTLLM_MARLIN_M16_TABLE"] = "0"
    elif cfg is not None:
        os.environ["TRTLLM_MARLIN_M16_CFG"] = cfg


def err(out, ref):
    d = (out.float() - ref).abs()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                bf16_ulp=(ref.abs().max() * 2 ** -8).item(),
                finite=bool(torch.isfinite(out.float()).all().item()))


def eq(a, b):
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def capture(x, copies, n, k):
    s0 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for _ in range(2):
            [gemm(x, c, n, k) for c in copies]
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        o = [gemm(x, c, n, k) for c in copies]
    return gr, o


def time_graphs(graphs, warm, iters, ncalls):
    """Replay graphs alternately (rotated order); return per-call us stats per key."""
    keys = list(graphs)
    for key in keys:
        for _ in range(warm):
            graphs[key].replay()
    torch.cuda.synchronize()
    evs = dict((key, []) for key in keys)
    for it in range(iters):
        r = it % len(keys)
        order = keys[r:] + keys[:r]
        if it % 2:
            order = order[::-1]
        for key in order:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[key].replay()
            e1.record()
            evs[key].append((e0, e1))
    torch.cuda.synchronize()
    out = dict()
    for key in keys:
        t = sorted(e0.elapsed_time(e1) * 1e3 / ncalls for e0, e1 in evs[key])
        med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
        out[key] = dict(us=med, p10=p10, p90=p90, noise=(p90 - p10) / med)
    return out
