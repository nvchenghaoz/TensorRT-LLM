"""Low-M FP8 block-scale shared-expert MLP over a device-side 128-block range.

Decode-only (M <= MAX_TOKENS) replacement for GatedMLP's SM120 FP8 block-scale
chain (per_token_quant -> gate_up GEMM -> SwiGLU -> per_token_quant -> down GEMM)
that computes only the intermediate 128-column blocks [b0, b1) of this rank's
shared-expert shard. b0/b1 are read from a device int32[2] tensor, so the
range can change without re-capturing a CUDA graph.

Math (same as main's chain, in a different summation order):
* x is quantized per token and 128-column group exactly as
  per_token_quant_and_transform does (UE8M0 scale = 2^ceil(log2(amax/448))).
* gate/up: a tensor-core fp32 sum over K of exact products of the e4m3 values
  times their UE8M0 block scales (powers of two folded into bf16 operands, which
  is exact: e4m3 has 4 significant bits), rounded to bf16 like the GEMM output.
  The weight scales must be powers of two (block_scale_table checks it).
* down: per 128-K block, a tensor-core fp32 sum of exact e4m3 x e4m3 products
  (operands widened to fp16), scaled by both block scales, summed in fp32.
* SwiGLU exactly as silu_and_mul_kernel (gate upper clamp, symmetric up clamp),
  rounded to bf16; then the same per-token 1x128 requantization.
* down: in-range K blocks only, fp32, rounded to bf16. An empty range writes zeros.

Two Triton launches. Kernel 1 computes the bf16 gate and up pre-activations of the
in-range columns (one CTA per BN rows of gate or up, so a 5-block range runs on
2*640/BN CTAs); kernel 2 (PDL: waits for kernel 1 before reading them) applies
SwiGLU + requantization per 128-block while it streams the down weight. Tokens are
the MMA N dimension (padded to 16), so one compiled variant serves M = 1..16.
Small CTAs (no large shared-memory footprint) so they co-reside with the persistent
routed-MoE kernels running on the main stream.

That two-launch chain is v1 (shared_decode_mlp_v1). The default, v2
(shared_decode_mlp_v2), quantizes x once in its own launch and runs both GEMMs as
e4m3 x e4m3 tensor-core dots per 128-K block, scaling the fp32 partial sums by the
block scales (powers of two, exact); same math, another summation order.
"""

from typing import Optional

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
from ..switches import env_get  # run switches frozen at served values

MAX_TOKENS = 8
_TM = 16  # token tile (MMA N), >= MAX_TOKENS

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_FP8_MIN = -_FP8_MAX

# Launch configuration (tunable by env for microbenchmarks).
_GU_BN = int(env_get("TRTLLM_SHARED_DECODE_GU_BN", "16"))
_GU_BK = int(env_get("TRTLLM_SHARED_DECODE_GU_BK", "512"))
_GU_WARPS = int(env_get("TRTLLM_SHARED_DECODE_GU_WARPS", "4"))
_GU_STAGES = int(env_get("TRTLLM_SHARED_DECODE_GU_STAGES", "3"))
_DN_BN = int(env_get("TRTLLM_SHARED_DECODE_DN_BN", "64"))
_DN_WARPS = int(env_get("TRTLLM_SHARED_DECODE_DN_WARPS", "4"))
_DN_STAGES = int(env_get("TRTLLM_SHARED_DECODE_DN_STAGES", "3"))
_USE_PDL = env_get("TRTLLM_SHARED_DECODE_PDL", "1") == "1"

# v2 (default): x quantized once by its own kernel, e4m3 x e4m3 tensor-core dots per
# 128-K block with the fp32 accumulator scaled per block (less ALU per weight byte, so
# it contends less with the persistent routed-MoE kernels). TRTLLM_SHARED_DECODE_V2=0
# selects v1 (the two-kernel chain above).
_V2 = env_get("TRTLLM_SHARED_DECODE_V2", "1") == "1"
_V2_GU_BN = int(env_get("TRTLLM_SHARED_DECODE_V2_GU_BN", "16"))
_V2_GU_SUB = int(env_get("TRTLLM_SHARED_DECODE_V2_GU_SUB", "2"))
_V2_GU_WARPS = int(env_get("TRTLLM_SHARED_DECODE_V2_GU_WARPS", "4"))
_V2_GU_STAGES = int(env_get("TRTLLM_SHARED_DECODE_V2_GU_STAGES", "3"))
_V2_DN_BN = int(env_get("TRTLLM_SHARED_DECODE_V2_DN_BN", "64"))
_V2_DN_WARPS = int(env_get("TRTLLM_SHARED_DECODE_V2_DN_WARPS", "4"))
_V2_DN_STAGES = int(env_get("TRTLLM_SHARED_DECODE_V2_DN_STAGES", "2"))


@triton.jit
def _quant_rows_128(v, fp8_max, fp8_min):
    """v: [R, 128] fp32. Returns the e4m3 values and the UE8M0 scale per row, as
    _per_token_quant_and_transform_kernel computes them."""
    amax = tl.maximum(tl.max(tl.abs(v), axis=1), 1e-10)
    s = amax / fp8_max
    s = tl.exp2(tl.ceil(tl.log2(tl.abs(s))))
    q = tl.clamp(v / s[:, None], fp8_min, fp8_max).to(tl.float8e4nv)
    return q, s


@triton.jit(do_not_specialize=["M"])
def _shared_gate_up_kernel(
    x_ptr,
    w_ptr,
    ws_ptr,
    rng_ptr,
    g_ptr,
    M,
    stride_xm,
    stride_wn,
    stride_wsn,
    stride_gm,
    fp8_max,
    fp8_min,
    K: tl.constexpr,
    I: tl.constexpr,
    TM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    b0 = tl.load(rng_ptr)
    b1 = tl.load(rng_ptr + 1)
    span = (b1 - b0) * 128
    j = pid * BN
    if j < 2 * span:
        # rows [0, span) of the job are gate rows b0*128.., the rest are up rows.
        row0 = b0 * 128 + j
        if j >= span:
            row0 = I + b0 * 128 + (j - span)
        sblk = row0 // 128
        offs_n = row0 + tl.arange(0, BN)
        offs_k = tl.arange(0, 128)
        offs_m = tl.arange(0, TM)
        NKB: tl.constexpr = BK // 128
        offs_bk = tl.arange(0, BK)
        offs_kb = tl.arange(0, NKB)
        w_ptrs = w_ptr + offs_n.to(tl.int64)[:, None] * stride_wn + offs_bk[None, :]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_bk[None, :]
        xmask = offs_m[:, None] < M
        acc = tl.zeros([BN, TM], dtype=tl.float32)
        for k0 in tl.range(0, K, BK, num_stages=NUM_STAGES):
            w = tl.load(w_ptrs + k0)
            xv = tl.load(x_ptrs + k0, mask=xmask, other=0.0).to(tl.float32)
            x3 = tl.reshape(xv, [TM, NKB, 128])
            amax = tl.maximum(tl.max(tl.abs(x3), axis=2), 1e-10)
            s = amax / fp8_max
            s = tl.exp2(tl.ceil(tl.log2(tl.abs(s))))
            q = tl.clamp(x3 / s[:, :, None], fp8_min, fp8_max).to(tl.float8e4nv).to(tl.float32)
            # Power-of-two block scales folded into the operands: exact in bf16
            # (e4m3 has 4 significant bits), so each product is exact in fp32.
            qs = tl.reshape(q * s[:, :, None], [TM, BK]).to(tl.bfloat16)
            sw = tl.load(ws_ptr + sblk * stride_wsn + k0 // 128 + offs_kb)
            ww = tl.reshape(tl.reshape(w.to(tl.float32), [BN, NKB, 128]) * sw[None, :, None],
                            [BN, BK]).to(tl.bfloat16)
            acc += tl.dot(ww, tl.trans(qs))
        tl.store(g_ptr + offs_m[None, :] * stride_gm + offs_n[:, None],
                 acc.to(tl.bfloat16),
                 mask=offs_m[None, :] < M)


@triton.jit(do_not_specialize=["M"])
def _shared_down_kernel(
    g_ptr,
    w_ptr,
    ws_ptr,
    rng_ptr,
    out_ptr,
    M,
    stride_gm,
    stride_wn,
    stride_wsn,
    stride_om,
    fp8_max,
    fp8_min,
    I: tl.constexpr,
    TM: tl.constexpr,
    BN: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    LIMIT: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    n0 = pid * BN
    nblk = n0 // 128
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    offs_m = tl.arange(0, TM)
    b0 = tl.load(rng_ptr)
    b1 = tl.load(rng_ptr + 1)
    w_ptrs = w_ptr + offs_n.to(tl.int64)[:, None] * stride_wn + offs_k[None, :]
    g_ptrs = g_ptr + offs_m[:, None] * stride_gm + offs_k[None, :]
    gmask = offs_m[:, None] < M
    acc = tl.zeros([BN, TM], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    for kb in tl.range(b0, b1, num_stages=NUM_STAGES):
        w = tl.load(w_ptrs + kb * 128)
        a = tl.load(g_ptrs + kb * 128, mask=gmask, other=0.0).to(tl.float32)
        b = tl.load(g_ptrs + I + kb * 128, mask=gmask, other=0.0).to(tl.float32)
        if HAS_LIMIT:
            a = tl.minimum(a, LIMIT)
            b = tl.clamp(b, -LIMIT, LIMIT)
        hv = (a * tl.sigmoid(1.0 * a) * (b + 0.0)).to(tl.bfloat16).to(tl.float32)
        q, s = _quant_rows_128(hv, fp8_max, fp8_min)
        sw = tl.load(ws_ptr + nblk * stride_wsn + kb)
        p = tl.dot(w.to(tl.float16), tl.trans(q.to(tl.float16)))
        acc += p * (s * sw)[None, :]
    tl.store(out_ptr + offs_m[None, :] * stride_om + offs_n[:, None],
             acc.to(tl.bfloat16),
             mask=offs_m[None, :] < M)


@triton.jit(do_not_specialize=["M"])
def _v2_xq_kernel(
    x_ptr,
    xq_ptr,
    xs_ptr,
    M,
    stride_xm,
    fp8_max,
    fp8_min,
    K: tl.constexpr,
    TM: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """x [M, K] bf16 -> e4m3 xq [TM, K] (rows >= M zero) and fp32 xs [TM, K/128], per token
    and 128-column group as per_token_quant_and_transform."""
    kb = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    offs_m = tl.arange(0, TM)
    offs_k = tl.arange(0, 128)
    msk = offs_m[:, None] < M
    v = tl.load(x_ptr + offs_m[:, None] * stride_xm + kb * 128 + offs_k[None, :],
                mask=msk,
                other=0.0).to(tl.float32)
    q, s = _quant_rows_128(v, fp8_max, fp8_min)
    tl.store(xq_ptr + offs_m[:, None] * K + kb * 128 + offs_k[None, :], q)
    tl.store(xs_ptr + offs_m * (K // 128) + kb, s)


@triton.jit(do_not_specialize=["M"])
def _v2_gate_up_kernel(
    xq_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    rng_ptr,
    g_ptr,
    M,
    stride_wn,
    stride_wsn,
    stride_gm,
    K: tl.constexpr,
    I: tl.constexpr,
    TM: tl.constexpr,
    BN: tl.constexpr,
    SUB: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    b0 = tl.load(rng_ptr)
    b1 = tl.load(rng_ptr + 1)
    span = (b1 - b0) * 128
    j = pid * BN
    if j < 2 * span:
        row0 = b0 * 128 + j
        if j >= span:
            row0 = I + b0 * 128 + (j - span)
        sblk = row0 // 128
        offs_n = row0 + tl.arange(0, BN)
        offs_k = tl.arange(0, 128)
        offs_m = tl.arange(0, TM)
        w_ptrs = w_ptr + offs_n.to(tl.int64)[:, None] * stride_wn + offs_k[None, :]
        q_ptrs = xq_ptr + offs_m[None, :] * K + offs_k[:, None]  # [128, TM]
        acc = tl.zeros([BN, TM], dtype=tl.float32)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        for k0 in tl.range(0, K // 128, SUB, num_stages=NUM_STAGES):
            for u in tl.static_range(SUB):
                kb = k0 + u
                w = tl.load(w_ptrs + kb * 128)
                q = tl.load(q_ptrs + kb * 128)
                xs = tl.load(xs_ptr + offs_m * (K // 128) + kb)
                sw = tl.load(ws_ptr + sblk * stride_wsn + kb)
                p = tl.dot(w, q)
                acc += p * (xs * sw)[None, :]
        tl.store(g_ptr + offs_m[None, :] * stride_gm + offs_n[:, None],
                 acc.to(tl.bfloat16),
                 mask=offs_m[None, :] < M)


@triton.jit(do_not_specialize=["M"])
def _v2_down_kernel(
    g_ptr,
    w_ptr,
    ws_ptr,
    rng_ptr,
    out_ptr,
    M,
    stride_gm,
    stride_wn,
    stride_wsn,
    stride_om,
    fp8_max,
    fp8_min,
    I: tl.constexpr,
    TM: tl.constexpr,
    BN: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    LIMIT: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    n0 = pid * BN
    nblk = n0 // 128
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    offs_m = tl.arange(0, TM)
    b0 = tl.load(rng_ptr)
    b1 = tl.load(rng_ptr + 1)
    w_ptrs = w_ptr + offs_n.to(tl.int64)[:, None] * stride_wn + offs_k[None, :]
    g_ptrs = g_ptr + offs_m[:, None] * stride_gm + offs_k[None, :]
    gmask = offs_m[:, None] < M
    acc = tl.zeros([BN, TM], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    for kb in tl.range(b0, b1, num_stages=NUM_STAGES):
        w = tl.load(w_ptrs + kb * 128)
        a = tl.load(g_ptrs + kb * 128, mask=gmask, other=0.0).to(tl.float32)
        b = tl.load(g_ptrs + I + kb * 128, mask=gmask, other=0.0).to(tl.float32)
        if HAS_LIMIT:
            a = tl.minimum(a, LIMIT)
            b = tl.clamp(b, -LIMIT, LIMIT)
        hv = (a * tl.sigmoid(1.0 * a) * (b + 0.0)).to(tl.bfloat16).to(tl.float32)
        q, s = _quant_rows_128(hv, fp8_max, fp8_min)
        sw = tl.load(ws_ptr + nblk * stride_wsn + kb)
        p = tl.dot(w, tl.trans(q))
        acc += p * (s * sw)[None, :]
    tl.store(out_ptr + offs_m[None, :] * stride_om + offs_n[:, None],
             acc.to(tl.bfloat16),
             mask=offs_m[None, :] < M)


def block_scale_table(weight_scale: torch.Tensor, mn: int, k: int) -> torch.Tensor:
    """FP32 [ceil(mn/128), ceil(k/128)] block-scale grid of a FP8 block-scale Linear,
    from either the raw FP32 grid or SM100f/SM120's packed UE8M0 int32 layout."""
    if weight_scale.dtype == torch.float32 and weight_scale.dim() == 2 and tuple(
            weight_scale.shape) == ((mn + 127) // 128, (k + 127) // 128):
        table = weight_scale.contiguous()
    elif weight_scale.dtype == torch.int32:
        from tensorrt_llm.quantization.utils.fp8_utils import inverse_transform_sf
        table = inverse_transform_sf(weight_scale, mn, k).contiguous()
    else:
        raise NotImplementedError(
            f"unsupported FP8 block-scale layout {weight_scale.dtype} {tuple(weight_scale.shape)}")
    # The kernels fold the scales into bf16 operands, exact only for powers of two
    # (always true of SM120's UE8M0 layout).
    if not bool((table > 0).all().item()) or not bool(
            (torch.frexp(table).mantissa == 0.5).all().item()):
        raise NotImplementedError("FP8 block scales are not all positive powers of two")
    return table


def supported_shapes(hidden: int, inter: int) -> bool:
    """Shapes the kernels tile exactly (128-blocks in K and in the intermediate)."""
    if _V2:
        return (hidden % (128 * _V2_GU_SUB) == 0 and inter % 128 == 0
                and hidden % _V2_DN_BN == 0 and 128 % _V2_GU_BN == 0 and 128 % _V2_DN_BN == 0)
    return (hidden % _GU_BK == 0 and inter % 128 == 0 and hidden % _DN_BN == 0
            and _GU_BK % 128 == 0 and 128 % _GU_BN == 0 and 128 % _DN_BN == 0)


def shared_decode_mlp(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    block_range: torch.Tensor,
    swiglu_limit: Optional[float],
    out: Optional[torch.Tensor] = None,
    use_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Shared GatedMLP over intermediate blocks [block_range[0], block_range[1]).

    Dispatches to v2 (default) or v1 (TRTLLM_SHARED_DECODE_V2=0); same arguments and
    result contract as shared_decode_mlp_v1.
    """
    fn = shared_decode_mlp_v2 if _V2 else shared_decode_mlp_v1
    return fn(x, gate_up_weight, gate_up_scale, down_weight, down_scale, block_range,
              swiglu_limit, out=out, use_pdl=use_pdl)


def shared_decode_mlp_v2(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    block_range: torch.Tensor,
    swiglu_limit: Optional[float],
    out: Optional[torch.Tensor] = None,
    use_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """v2: three launches (x quant -> gate/up -> SwiGLU+requant+down), PDL-chained.

    gate/up: per 128-K block an fp32 tensor-core sum of exact e4m3 x e4m3 products,
    times the two power-of-two block scales (exact), summed over blocks in fp32 and
    rounded to bf16. Down as v1 with e4m3 operands. Same math as v1 in another
    summation order.
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
    m, k = x.shape
    assert 0 < m <= MAX_TOKENS
    two_i, k_w = gate_up_weight.shape
    i = two_i // 2
    n, i_w = down_weight.shape
    assert (k_w == k and i_w == i and n == k and k % (128 * _V2_GU_SUB) == 0
            and i % 128 == 0 and n % _V2_DN_BN == 0)
    pdl = _USE_PDL if use_pdl is None else use_pdl
    has_limit = swiglu_limit is not None and swiglu_limit > 0.0
    xq = torch.empty((_TM, k), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((_TM, k // 128), dtype=torch.float32, device=x.device)
    g = torch.empty((m, two_i), dtype=torch.bfloat16, device=x.device)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    launch = {"launch_pdl": True} if pdl else {}
    _v2_xq_kernel[(k // 128, )](
        x,
        xq,
        xs,
        m,
        x.stride(0),
        _FP8_MAX,
        _FP8_MIN,
        K=k,
        TM=_TM,
        USE_PDL=pdl,
        num_warps=4,
    )
    _v2_gate_up_kernel[(two_i // _V2_GU_BN, )](
        xq,
        xs,
        gate_up_weight,
        gate_up_scale,
        block_range,
        g,
        m,
        gate_up_weight.stride(0),
        gate_up_scale.stride(0),
        g.stride(0),
        K=k,
        I=i,
        TM=_TM,
        BN=_V2_GU_BN,
        SUB=_V2_GU_SUB,
        NUM_STAGES=_V2_GU_STAGES,
        USE_PDL=pdl,
        num_warps=_V2_GU_WARPS,
        **launch,
    )
    _v2_down_kernel[(n // _V2_DN_BN, )](
        g,
        down_weight,
        down_scale,
        block_range,
        out,
        m,
        g.stride(0),
        down_weight.stride(0),
        down_scale.stride(0),
        out.stride(0),
        _FP8_MAX,
        _FP8_MIN,
        I=i,
        TM=_TM,
        BN=_V2_DN_BN,
        HAS_LIMIT=has_limit,
        LIMIT=float(swiglu_limit) if has_limit else 0.0,
        NUM_STAGES=_V2_DN_STAGES,
        USE_PDL=pdl,
        num_warps=_V2_DN_WARPS,
        **launch,
    )
    return out


def shared_decode_mlp_v1(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    block_range: torch.Tensor,
    swiglu_limit: Optional[float],
    out: Optional[torch.Tensor] = None,
    use_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Shared GatedMLP over intermediate blocks [block_range[0], block_range[1]).

    x: [M, H] bf16, M <= MAX_TOKENS. gate_up_weight: [2I, H] e4m3 (gate rows then
    up rows); gate_up_scale: fp32 [2I/128, H/128]; down_weight: [H, I] e4m3;
    down_scale: fp32 [H/128, I/128]; block_range: device int32[2] with
    0 <= b0 <= b1 <= I/128.
    Returns [M, H] bf16 (zeros when the range is empty).
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
    m, k = x.shape
    assert 0 < m <= MAX_TOKENS
    two_i, k_w = gate_up_weight.shape
    i = two_i // 2
    n, i_w = down_weight.shape
    assert k_w == k and i_w == i and n == k and supported_shapes(k, i)
    pdl = _USE_PDL if use_pdl is None else use_pdl
    has_limit = swiglu_limit is not None and swiglu_limit > 0.0
    # bf16 gate (cols [0, I)) and up (cols [I, 2I)) pre-activations; only the
    # in-range columns are written and read.
    g = torch.empty((m, two_i), dtype=torch.bfloat16, device=x.device)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _shared_gate_up_kernel[(two_i // _GU_BN, )](
        x,
        gate_up_weight,
        gate_up_scale,
        block_range,
        g,
        m,
        x.stride(0),
        gate_up_weight.stride(0),
        gate_up_scale.stride(0),
        g.stride(0),
        _FP8_MAX,
        _FP8_MIN,
        K=k,
        I=i,
        TM=_TM,
        BN=_GU_BN,
        BK=_GU_BK,
        NUM_STAGES=_GU_STAGES,
        USE_PDL=pdl,
        num_warps=_GU_WARPS,
    )
    launch = {"launch_pdl": True} if pdl else {}
    _shared_down_kernel[(n // _DN_BN, )](
        g,
        down_weight,
        down_scale,
        block_range,
        out,
        m,
        g.stride(0),
        down_weight.stride(0),
        down_scale.stride(0),
        out.stride(0),
        _FP8_MAX,
        _FP8_MIN,
        I=i,
        TM=_TM,
        BN=_DN_BN,
        HAS_LIMIT=has_limit,
        LIMIT=float(swiglu_limit) if has_limit else 0.0,
        NUM_STAGES=_DN_STAGES,
        USE_PDL=pdl,
        num_warps=_DN_WARPS,
        **launch,
    )
    return out
