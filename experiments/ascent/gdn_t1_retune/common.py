# Shared synthetic serving-shape inputs for the GDN T1 retune experiments.
import torch

from tensorrt_llm._torch.modules.fla import gdn_decode_t1 as t1mod

H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VAL_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VAL_DIM
QKVZ_DIM = CONV_DIM + VAL_DIM
SLOT_STRIDE = 817152
dev = torch.device("cuda")
flush_buf = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def make_layer(n, gen, pad=False):
    slots = n + 4
    qkvz = (torch.randn(n, QKVZ_DIM, device=dev, generator=gen) * 0.7).to(
        torch.bfloat16)
    ba = (torch.randn(n, 2 * HV, device=dev, generator=gen) * 1.5).to(
        torch.bfloat16)
    A = torch.exp(torch.rand(HV, device=dev, generator=gen) * 6.0 - 5.3)
    flat = (torch.randn(slots * SLOT_STRIDE, device=dev, generator=gen) *
            0.2).to(torch.bfloat16)
    pool = flat.as_strided((slots, HV, V, K), (SLOT_STRIDE, V * K, K, 1))
    idx = torch.randperm(slots, device=dev, generator=gen)[:n].to(torch.int32)
    if pad:
        idx[n // 2] = -1
    mixed = qkvz[:, :CONV_DIM]
    return dict(
        n=n,
        qkvz=qkvz,
        ba=ba,
        q=mixed[:, :KEY_DIM].view(1, n, H, K),
        k=mixed[:, KEY_DIM:2 * KEY_DIM].view(1, n, H, K),
        v=mixed[:, 2 * KEY_DIM:].view(1, n, HV, V),
        z=qkvz[:, CONV_DIM:].view(n, HV, V),
        b=ba[:, :HV],
        a=ba[:, HV:],
        A_log=torch.log(A).float(),
        dt_bias=(torch.randn(HV, device=dev, generator=gen) * 0.5 -
                 1.0).float(),
        norm_w=(1.0 + 0.1 * torch.randn(V, device=dev, generator=gen)).to(
            torch.bfloat16),
        flat=flat,
        flat0=flat.clone(),
        pool=pool,
        idx=idx,
        out=torch.zeros(n, HV, V, dtype=torch.bfloat16, device=dev),
    )


def make_call(bv, warps, fn=None):
    f = fn or t1mod.fused_sigmoid_gating_delta_rule_update_t1

    def call(lay):
        f(A_log=lay["A_log"], a=lay["a"], dt_bias=lay["dt_bias"],
          softplus_beta=1.0, softplus_threshold=20.0, q=lay["q"], k=lay["k"],
          v=lay["v"], b=lay["b"], initial_state_source=lay["pool"],
          initial_state_indices=lay["idx"], scale=K**-0.5,
          use_qk_l2norm_in_kernel=True, output=lay["out"], BV=bv,
          num_warps=warps)

    return call


def reset(layers):
    for lay in layers:
        lay["flat"].copy_(lay["flat0"])
        lay["out"].zero_()
    flush_buf.sum()
