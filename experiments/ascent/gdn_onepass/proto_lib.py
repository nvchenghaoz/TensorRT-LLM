"""Shared helpers for the one-pass GDN prefill prototypes (synthetic real shapes)."""
import torch

from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
)
from tensorrt_llm._torch.modules.mamba.gdn_prefill_conv import (
    gdn_fused_transpose_conv_enabled,
    gdn_prefill_transpose_conv,
)

try:
    from tensorrt_llm._torch.modules.mamba.gdn_prefill_onepass import gdn_prefill_onepass
except ImportError:
    gdn_prefill_onepass = None

dev = torch.device("cuda")
DIM, QKVZ, NB = 10240, 16384, 12
BLK = 817152


def make(lens, ndec, seed=0, has_init=None):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    T = sum(lens)
    qkvz = (torch.randn(T, QKVZ, device=dev, generator=g) * 1.5).to(torch.bfloat16)
    dec = (torch.randn(ndec, DIM, device=dev, generator=g)).to(torch.bfloat16) if ndec else None
    w = (torch.randn(DIM, 4, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    pool = torch.zeros(NB * BLK, dtype=torch.bfloat16, device=dev)
    conv = pool.as_strided((NB, DIM, 3), (BLK, 3, 1))
    conv.copy_(torch.randn(NB, DIM, 3, device=dev, generator=g).to(torch.bfloat16))
    ba = torch.randn(T + ndec, 96, device=dev, generator=g).to(torch.bfloat16)
    A_log = torch.log(torch.rand(48, device=dev, generator=g) * 16)
    dt_bias = torch.rand(48, device=dev, generator=g) * 2 - 1
    n = len(lens)
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32,
                       device=dev)
    idx = torch.randperm(NB, device=dev, generator=g)[:n].to(torch.int32)
    hi = torch.ones(n, dtype=torch.bool, device=dev) if has_init is None else torch.tensor(
        has_init, dtype=torch.bool, device=dev)
    return dict(x=qkvz[:, :DIM], qkvz=qkvz, dec=dec, w=w, pool=pool, conv=conv,
                pool0=pool.clone(), a=ba[:, 48:], b=ba[:, :48], A_log=A_log, dt_bias=dt_bias,
                qsl=qsl, idx=idx, hi=hi, T=T)


def run_main(c):
    x = c["x"]
    T = x.shape[0]
    if gdn_fused_transpose_conv_enabled(T):
        xt = gdn_prefill_transpose_conv(x, c["w"], None, c["conv"], c["qsl"], c["idx"], c["hi"],
                                        activation="silu")
    else:
        xt = extract_transpose_prefill_slice(x, T, 0, x.shape[1])
        xt = causal_conv1d_fn(xt, c["w"], None, activation="silu", conv_states=c["conv"],
                              has_initial_state=c["hi"], cache_indices=c["idx"],
                              query_start_loc=c["qsl"])
    return fused_gdn_post_conv(xt, c["dec"], c["a"], c["b"], c["A_log"], c["dt_bias"], 16, 128,
                               48, 128)


def run_one(c, **kw):
    return gdn_prefill_onepass(c["x"], c["dec"], c["w"], None, c["conv"], c["qsl"], c["idx"],
                               c["hi"], c["a"], c["b"], c["A_log"], c["dt_bias"], 16, 128, 48,
                               128, **kw)
