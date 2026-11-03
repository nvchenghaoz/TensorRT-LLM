"""Shared setup for idea 73 (CSA2 decode WO projection on b12x).

Builds per-rank synthetic tensors (TP4) with a fixed seed, main's chain and the
candidate chains. No model load.
"""
import json
import math
import os

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm.quantization.utils import fp8_utils  # noqa: E402

DEV = torch.device("cuda")
TP = 4


def dims_from_ckpt(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    heads = tc["num_attention_heads"]
    head_dim = tc["head_dim"]
    rope = tc["qk_rope_head_dim"]
    groups = tc["o_groups"]
    d = dict(
        heads_tp=heads // TP,
        head_dim=head_dim,
        rope=rope,
        nope=head_dim - rope,
        groups_local=groups // TP,
        group_width=heads * head_dim // groups,
        heads_per_group=heads // groups,
        rank=tc["o_lora_rank"],
        hidden=tc["hidden_size"],
    )
    return d


# ---------------------------------------------------------------------------
# Exact quantizer: bf16 inverse RoPE (as mla_rope_inplace) + 1x128 UE8M0 quant
# (as fp8_utils._per_token_quant_and_transform_kernel), scales replicated into
# the four 1x32 MXFP8 slots of b12x's MMA scale layout.
# ---------------------------------------------------------------------------
@triton.jit
def _inv_rope_quant128_kernel(
    o, positions, cos_sin, values, scale_mma,
    o_stride_t, cs_stride_pos,
    v_stride_t, v_stride_d, v_stride_g,
    s0, s1, s2, s3, s4, s5,
    fp8_max, fp8_min,
    GROUP_WIDTH: tl.constexpr, HEAD_DIM: tl.constexpr, NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr, NB: tl.constexpr, ROPE_VARIANT: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    cb = tl.program_id(2)
    kc = cb * NB + tl.arange(0, NB)
    d = kc[:, None] * 128 + tl.arange(0, 128)[None, :]
    base = o + token * o_stride_t + group * GROUP_WIDTH
    src = tl.load(base + d).to(tl.float32)
    head_d = d % HEAD_DIM
    is_rope = head_d >= NOPE_DIM
    partner = tl.load(base + (d ^ 1), mask=is_rope, other=0.0).to(tl.float32)
    pos = tl.load(positions + token)
    ci = tl.maximum((head_d - NOPE_DIM) >> 1, 0)
    cbase = cos_sin + pos.to(tl.int64) * cs_stride_pos
    c = tl.load(cbase + ci, mask=is_rope, other=1.0)
    s = tl.load(cbase + HALF_ROPE + ci, mask=is_rope, other=0.0)
    ns = -s
    even = (head_d & 1) == 0
    # CUDA: coef=(cos, -sin); x' = coef.x*x - coef.y*y ; y' = coef.x*y + coef.y*x
    if ROPE_VARIANT == 0:
        r_even = tl.fma(-ns, partner, c * src)
        r_odd = tl.fma(ns, partner, c * src)
    elif ROPE_VARIANT == 1:
        r_even = tl.fma(c, src, -(ns * partner))
        r_odd = tl.fma(c, src, ns * partner)
    else:
        r_even = c * src - ns * partner
        r_odd = c * src + ns * partner
    rot = tl.where(even, r_even, r_odd)
    rot = rot.to(tl.bfloat16).to(tl.float32)
    v = tl.where(is_rope, rot, src)
    amax = tl.maximum(tl.max(tl.abs(v), axis=1), 1e-10)
    sc = amax / fp8_max
    sc = tl.exp2(tl.ceil(tl.log2(tl.abs(sc))))
    q = tl.clamp(v / sc[:, None], fp8_min, fp8_max).to(values.dtype.element_ty)
    tl.store(values + token * v_stride_t + d * v_stride_d + group * v_stride_g, q)
    byte = ((sc.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8)
    j = tl.arange(0, 4)
    row32 = token % 32
    row4 = (token // 32) % 4
    tile_m = token // 128
    addr = (row32 * s0 + row4 * s1 + tile_m * s2 + j[None, :] * s3 + kc[:, None] * s4
            + group * s5)
    tl.store(scale_mma + addr, tl.broadcast_to(byte[:, None], (NB, 4)))


def inv_rope_quant128(o3, positions, cos_sin, x_q, dims, rope_variant=0, nb=4):
    """o3: [M, heads_tp, head_dim] bf16 (not modified). Writes x_q (MXFP8Rows)."""
    M = o3.shape[0]
    G = dims["groups_local"]
    GW = dims["group_width"]
    sm = x_q.scale_mma.view(torch.uint8)
    vals = x_q.values
    _inv_rope_quant128_kernel[(M, G, GW // 128 // nb)](
        o3, positions, cos_sin, vals, sm,
        o3.stride(0), cos_sin.stride(0),
        vals.stride(0), vals.stride(1), vals.stride(2) if vals.ndim == 3 else 0,
        *sm.stride(),
        448.0, -448.0,
        GROUP_WIDTH=GW, HEAD_DIM=dims["head_dim"], NOPE_DIM=dims["nope"],
        HALF_ROPE=dims["rope"] // 2, NB=nb, ROPE_VARIANT=rope_variant,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Synthetic weights / inputs
# ---------------------------------------------------------------------------
class Weights:
    pass


def make_weights(dims, seed=1234):
    from b12x.gemm._shared import wo_mxfp8 as W
    g = torch.Generator(device="cuda").manual_seed(seed)
    G, R, GW, H = dims["groups_local"], dims["rank"], dims["group_width"], dims["hidden"]
    w = Weights()
    # "checkpoint" wo_a: e4m3 + arbitrary fp32 128x128 block scales
    wa = torch.randn(G, R, GW, device=DEV, generator=g) * 0.02
    blocks = wa.view(G, R // 128, 128, GW // 128, 128)
    amax = blocks.abs().amax(dim=(2, 4))
    s_ck = (amax / 448.0).float()  # [G, R/128, GW/128]
    w8 = (blocks / s_ck[:, :, None, :, None]).view(G, R, GW).to(torch.float8_e4m3fn)
    w.wa_ckpt = w8.clone()
    w.sa_ckpt = s_ck.clone()
    # main's SM120 load: resmooth to UE8M0, then transform the scale layout
    wr, sr = fp8_utils.resmooth_to_fp8_e8m0(w8.clone(), s_ck.clone())
    w.wa_main = wr.view(G, R, GW).contiguous()
    w.sa_pow2 = sr.view(G, R // 128, GW // 128).contiguous()
    w.sa_main = fp8_utils.transform_sf_into_required_layout(
        w.sa_pow2, mn=R, k=GW, recipe=(1, 128, 128), num_groups=G, is_sfa=False)
    # b12x wo_a: same e4m3 bytes, UE8M0 exponents expanded to 1x32 rows
    sa_u8 = (torch.log2(w.sa_pow2) + 127).round().to(torch.uint8)
    w.wa_b12x = W.pack_fp8_block_scaled_weight_mxfp8(
        w.wa_main.view(G * R, GW), sa_u8, m=R, k=GW, num_groups=G)
    # wo_b: native MXFP8 1x32 (as main's o_b_proj MXFP8 linear stores it)
    wb = torch.randn(H, G * R, device=DEV, generator=g, dtype=torch.float32).to(torch.bfloat16) * 0.02
    w.wb_q, w.wb_sf = torch.ops.trtllm.mxfp8_quantize(wb, True)
    _, wb_sf_lin = torch.ops.trtllm.mxfp8_quantize(wb, False)
    K = G * R
    w.wb_sf_lin = wb_sf_lin.view(torch.uint8).view(H, K // 32)
    sfb = w.wb_sf.view(torch.float8_e8m0fnu).view(1, H // 128, K // 128, 32, 4, 4).permute(
        3, 4, 1, 5, 2, 0)
    w.wb_b12x = W.MXFP8Rows(values=w.wb_q, scale_rows=w.wb_sf_lin.view(torch.float8_e8m0fnu).view(1, H, K // 32),
                            scale_mma=sfb)
    w.weights_b12x = W.WOProjectionMXFP8Weights(
        wo_a=w.wa_b12x, wo_b=w.wb_b12x, groups=G, group_width=GW, rank=R, hidden=H,
        sfb_k_replicated=False)
    return w


def clone_weights(w, dims):
    """A rotation copy (fresh storage) of every weight both arms read."""
    from b12x.gemm._shared import wo_mxfp8 as W
    c = Weights()
    G, R, GW, H = dims["groups_local"], dims["rank"], dims["group_width"], dims["hidden"]
    c.wa_main = w.wa_main.clone()
    c.sa_main = w.sa_main.clone()
    c.sa_pow2 = w.sa_pow2.clone()
    # b12x wo_a values view over a fresh [G,R,GW] copy (same bytes as wa_main)
    va = c.wa_main.permute(1, 2, 0)
    sm_phys = w.wa_b12x.scale_mma.permute(5, 2, 4, 0, 1, 3).contiguous()
    c.wa_b12x = W.MXFP8Rows(values=va, scale_rows=w.wa_b12x.scale_rows,
                            scale_mma=sm_phys.permute(3, 4, 1, 5, 2, 0))
    c.wb_q = w.wb_q.clone()
    c.wb_sf = w.wb_sf.clone()
    K = G * R
    sfb = c.wb_sf.view(torch.float8_e8m0fnu).view(1, H // 128, K // 128, 32, 4, 4).permute(
        3, 4, 1, 5, 2, 0)
    c.wb_b12x = W.MXFP8Rows(values=c.wb_q, scale_rows=w.wb_b12x.scale_rows, scale_mma=sfb)
    c.weights_b12x = W.WOProjectionMXFP8Weights(
        wo_a=c.wa_b12x, wo_b=c.wb_b12x, groups=G, group_width=GW, rank=R, hidden=H,
        sfb_k_replicated=False)
    return c


def make_inputs(dims, M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    o3 = torch.randn(M, dims["heads_tp"], dims["head_dim"], device=DEV, generator=g).to(torch.bfloat16)
    # an all-zero 128 block and a 30x-magnitude block
    o3[0, 0, :128] = 0
    if dims["heads_tp"] > 1:
        o3[0, 1, 128:256] *= 30
    pos = torch.randint(30000, 40000, (M,), device=DEV, generator=g, dtype=torch.int32)
    return o3, pos


def make_cos_sin(dims, max_pos=65536, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    half = dims["rope"] // 2
    theta = torch.rand(max_pos, half, device=DEV, generator=g, dtype=torch.float64) * 2 * math.pi
    cs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1).float().contiguous()
    return cs  # [max_pos, 2, half] fp32


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------
def main_chain(o3, pos, cs, w, dims):
    """Main's SM120 path (project_sparse_attn_output, fp8 block128). Mutates o3."""
    from tensorrt_llm._torch.attention.mla import fp8_block_scaling_bmm_out
    M = o3.shape[0]
    torch.ops.trtllm.mla_rope_inplace(o3, pos, cs, dims["heads_tp"], dims["nope"], dims["rope"],
                                      True, False)
    o_lora = torch.empty([M, dims["groups_local"], dims["rank"]], device=DEV, dtype=torch.bfloat16)
    fp8_block_scaling_bmm_out(o3.view(M, dims["groups_local"], -1), w.wa_main, w.sa_main,
                              o_lora.transpose(0, 1), None, False)
    act, act_sf = torch.ops.trtllm.mxfp8_quantize(o_lora.flatten(1), True)
    return torch.ops.trtllm.flashinfer_mm_mxfp8(act, act_sf, w.wb_q, w.wb_sf, torch.bfloat16)


class Scratch:
    def __init__(self, dims, M):
        from b12x.gemm._shared import wo_mxfp8 as W
        G, R, GW, H = dims["groups_local"], dims["rank"], dims["group_width"], dims["hidden"]
        self.x_q = W.empty_mxfp8_rows_for_dense_gemm(M, GW, num_groups=G, device=DEV)
        self.tmp = W.empty_dense_gemm_mnl_view(M, R, G, device=DEV, dtype=torch.bfloat16)
        self.out = torch.empty((M, H, 1), device=DEV, dtype=torch.bfloat16)
        self.alpha = W._cached_alpha_one(DEV)


def exact_chain(o3, pos, cs, w, dims, sc, rope_variant=0, wo_b="fused"):
    """E: exact quantizer -> b12x grouped wo_a -> wo_b (fused in-GEMM quant, or main's ops)."""
    from b12x.gemm._shared import wo_mxfp8 as W
    M = o3.shape[0]
    inv_rope_quant128(o3, pos, cs, sc.x_q, dims, rope_variant)
    W.wo_a_dense_gemm_mxfp8(sc.x_q, w.wa_b12x, out=sc.tmp, alpha=sc.alpha, expected_m=M,
                            sfb_k_replicated=True)
    if wo_b == "fused" and M <= 8:
        W.wo_b_dense_gemm_fused_quant_mxfp8(sc.tmp, w.wb_b12x, out=sc.out, expected_m=M,
                                            sfb_k_replicated=False)
        return sc.out[:, :, 0]
    o_lora = sc.tmp.permute(0, 2, 1).reshape(M, -1).contiguous()
    act, act_sf = torch.ops.trtllm.mxfp8_quantize(o_lora, True)
    return torch.ops.trtllm.flashinfer_mm_mxfp8(act, act_sf, w.wb_q, w.wb_sf, torch.bfloat16)


def stock_chain(o3, pos, cs, w, dims):
    """S: stock b12x inverse-RoPE WO projection (1x32 fp32-RoPE quant, fused wo_b)."""
    from b12x.gemm._shared import wo_mxfp8 as W
    return W.wo_projection_inv_rope_mxfp8(
        o3, pos, cs.view(cs.shape[0], -1), w.weights_b12x, heads_per_group=dims["heads_per_group"],
        nope_dim=dims["nope"], rope_dim=dims["rope"])


def fp64_reference(o3_orig, pos, cs, w, dims):
    """Original math in FP64: inverse RoPE, checkpoint wo_a (pre-resmooth), wo_b dequant."""
    M = o3_orig.shape[0]
    G, R, GW, H = dims["groups_local"], dims["rank"], dims["group_width"], dims["hidden"]
    x = o3_orig.double().clone()
    nope, rope = dims["nope"], dims["rope"]
    c = cs[pos.long(), 0].double()  # [M, half]
    s = cs[pos.long(), 1].double()
    r = x[:, :, nope:].view(M, -1, rope // 2, 2)
    xe, xo = r[..., 0].clone(), r[..., 1].clone()
    r[..., 0] = c[:, None] * xe + s[:, None] * xo
    r[..., 1] = c[:, None] * xo - s[:, None] * xe
    xg = x.view(M, G, GW)
    wa = w.wa_ckpt.double() * w.sa_ckpt.double().repeat_interleave(128, 1).repeat_interleave(128, 2)
    lora = torch.einsum("mgk,grk->mgr", xg, wa).reshape(M, G * R)
    K = G * R
    wsc = torch.exp2(w.wb_sf_lin.double() - 127).repeat_interleave(32, 1)
    wb = w.wb_q.double() * wsc
    return lora @ wb.T, lora


def err(a, ref):
    a = a.double()
    ref = ref.double()
    d = (a - ref).abs()
    return {"max_abs": float(d.max()), "max_rel": float(d.max() / ref.abs().max().clamp_min(1e-30)),
            "mean_abs": float(d.mean())}


def bit_cmp(a, b):
    if a.shape != b.shape:
        return {"equal": False, "shape": [list(a.shape), list(b.shape)]}
    ne = int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) if a.dtype.itemsize == 1 else int(
        (a != b).sum())
    return {"equal": bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8))) if a.dtype.itemsize == 1
            else bool(torch.equal(a, b)), "n_diff": ne}
