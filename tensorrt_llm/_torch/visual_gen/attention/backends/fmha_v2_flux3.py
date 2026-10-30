"""FLUX.3 d128 FP8 attention: an add-only FlashInfer fmha_v2 JIT variant.

FlashInfer's stock ``trtllm_fmha_v2`` module runs
``fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_128x128_output_bf16_sm120_nl_tiled``
for E4M3 Q/K/V with head_dim 128 and BF16 output. This module compiles only that
kernel (renamed, so it cannot alias the stock symbols) plus the binding, with
macro-gated exact levers in ``fused_multihead_flash_attention_kernel_noloop_tiled.h``:

- ``mask_skip``  (FMHA_FLUX3_MASK_SKIP): the padding mask runs only on the last partial
  KV step instead of on every step from the causal diagonal on (bitwise equal).
- ``resident_q`` (FMHA_FLUX3_RESIDENT_Q): Q is loaded into its two smem buffers once
  instead of being re-fetched every KV step (bitwise equal).
- ``min_blocks``: ``__launch_bounds__(128, min_blocks)`` on the kernel.

The stock module is untouched (macros default off), so both can be loaded in one
process. The module is built on first use; callers build it at warmup.

This file must not import tensorrt_llm at module level: microbenchmarks load it by path.
"""

from __future__ import annotations

import functools
import os
import re
from typing import Optional, Sequence

import torch

STOCK_KERNEL = "fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_128x128_output_bf16_sm120"

_LEVER_FLAGS = {
    "softmax_trim": ("st", "-DFMHA_FLUX3_SOFTMAX_TRIM=1"),
    "mask_skip": ("ms", "-DFMHA_FLUX3_MASK_SKIP=1"),
    "resident_q": ("rq", "-DFMHA_FLUX3_RESIDENT_Q=1"),
    # loop_trim (a) (FMHA_FLUX3_LOOP_TRIM, noloop_tiled.h, carried from branch 78): drops the
    #   WAR __syncthreads after the BMM1-main QMMA loop (bitwise equal).
    "loop_trim": ("lt", "-DFMHA_FLUX3_LOOP_TRIM=1"),
    # REORDERED levers (not bitwise vs stock):
    # base2 (FMHA_FLUX3_BASE2, softmax.h / fragment.h): raw QK accumulators, one
    #   exp2(fma(x, s*log2e, log2(256) - max*s*log2e)) per element and
    #   alpha = exp2((prev - cur)*s*log2e). No LSE / attention-sink support.
    "base2": ("b2", "-DFMHA_FLUX3_BASE2=1"),
    # v_ldsm_x4 (FMHA_FLUX3_V_LDSM_X4, smem_tile_v.h): one LDSM.MT88.4 per V fragment
    #   instead of two LDSM.MT88.2 (bitwise equal).
    "v_ldsm_x4": ("vx4", "-DFMHA_FLUX3_V_LDSM_X4=1"),
    # kv128: 64 q rows x 128-key step (the stock d128 E4M3 tile steps 64 keys). Padding mask
    #   only: the causal / sliding / custom-mask kernels are not compiled in.
    "kv128": ("k128", None),
    # unit_alpha (FMHA_FLUX3_SKIP_UNIT_ALPHA, fragment.h, base2 only): a warp whose 2x16 rows
    #   all keep their running max (alpha == 1.0f) skips the O rescale (bitwise vs base2).
    "unit_alpha": ("ua", "-DFMHA_FLUX3_SKIP_UNIT_ALPHA=1"),
    # load_trim (FMHA_FLUX3_LOAD_TRIM, gmem_tile_qkv_packed.h; bitwise): the K/V LDGSTS skip the
    #   always-true column predicate, compute row predicates only on the last partial KV step and
    #   use the compile-time row stride LOAD_TRIM_STRIDE_BYTES (H_kv 24 x D 128 x E4M3). Calls
    #   with another K/V stride run the module without it (load_trim_levers).
    "load_trim": ("ldt", "-DFMHA_FLUX3_LOAD_TRIM=1"),
}

LOAD_TRIM_STRIDE_BYTES = 3072

# The lever set main runs.
DEFAULT_LEVERS: tuple[str, ...] = ("mask_skip", "resident_q")
# The lever set for non-causal, no-LSE calls: base-2 softmax on main's ms_rq 64x64 tile.
# Job 727 (interleaved, S=4176/4096): ms_rq 427.7/408.4 us, +base2 415.2/397.0 (-2.9%/pass),
# base2+kv128 (branch 30's 64x128 rqms_b2) 415.5/397.1 (same speed, rel L2 1.0017x ms_rq's;
# base2 at 64x64 is 1.0000x), kv128 alone 437.9/415.6 (slower).
ROUTED_LEVERS: tuple[str, ...] = ("base2", "mask_skip", "resident_q", *(("unit_alpha",) if os.environ.get("TRTLLM_FMHA_V2_FLUX3_UA", "1") != "0" else ()), *(("loop_trim",) if os.environ.get("TRTLLM_FMHA_V2_FLUX3_LT", "1") != "0" else ()))  # noqa: E501
DEFAULT_MIN_BLOCKS = 0


def _tag(levers: Sequence[str], min_blocks: int) -> str:
    parts = [_LEVER_FLAGS[name][0] for name in sorted(set(levers))]
    tag = "_".join(parts) if parts else "base"
    if min_blocks:
        tag += f"_mb{min_blocks}"
    return tag


# v_ldsm_x4 (exact vs b2) joins the routed set; TRTLLM_FMHA_V2_FLUX3_VX4=0 drops it.
def _routed_vx4() -> tuple[str, ...]:
    import os

    return ("v_ldsm_x4",) if os.environ.get("TRTLLM_FMHA_V2_FLUX3_VX4", "1") != "0" else ()


ROUTED_LEVERS = ROUTED_LEVERS + _routed_vx4()

# load_trim (exact) joins the routed set; TRTLLM_FMHA_V2_FLUX3_LDT=0 drops it.
if os.environ.get("TRTLLM_FMHA_V2_FLUX3_LDT", "1") != "0":
    ROUTED_LEVERS = (*ROUTED_LEVERS, "load_trim")


def load_trim_levers(levers: tuple[str, ...], key: torch.Tensor, value: torch.Tensor):
    """The levers to run for these K/V: load_trim only when both row strides (contiguous
    [B, S, H_kv, D] E4M3) equal the stride the trimmed kernel is compiled for."""
    if "load_trim" not in levers:
        return levers
    kb = key.shape[-2] * key.shape[-1] * key.element_size()
    vb = value.shape[-2] * value.shape[-1] * value.element_size()
    if kb == LOAD_TRIM_STRIDE_BYTES and vb == LOAD_TRIM_STRIDE_BYTES:
        return levers
    return tuple(name for name in levers if name != "load_trim")


def gen_flux3_d128_spec(levers: Sequence[str] = DEFAULT_LEVERS, min_blocks: int = 0):
    """JitSpec of the d128 E4M3->BF16 variant with the given levers."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.attention.fmha_v2.generate_kernels import enumerate_kernels
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec
    from flashinfer.jit.utils import write_if_different

    tag = _tag(levers, min_blocks)
    uri = f"trtllm_fmha_v2_flux3_d128_{tag}"
    gen_dir = jit_env.FLASHINFER_JIT_DIR / uri
    gen_dir.mkdir(parents=True, exist_ok=True)
    fmha_v2_src_dir = jit_env.FLASHINFER_CSRC_DIR / "fmha_v2"

    enumerate_kernels(fmha_v2_src_dir, gen_dir)
    stock_path = gen_dir / "generated" / f"{STOCK_KERNEL}.cu"
    text = stock_path.read_text()
    kernel = STOCK_KERNEL.replace("fmha_v2_flash_attention_", f"fmha_v2_flux3_{tag}_")
    text = text.replace(STOCK_KERNEL, kernel)
    if min_blocks:
        text, count = re.subn(
            r"__global__\s*\nvoid (\w+_nl_tiled)\(",
            rf"__global__ __launch_bounds__(128, {int(min_blocks)})\nvoid \1(",
            text,
        )
        if count == 0:
            raise RuntimeError("could not place __launch_bounds__ in the generated kernel")
    if "kv128" in levers:
        # Kernel_traits<instruction_traits, kv_loop_step, head_size, head_size_v, noloop_step, ...>
        text, count = re.subn(
            r"(fmha::\w+_traits,\s*\n\s*)64(,\s*\n\s*128,\s*\n\s*128,\s*\n\s*64,)",
            r"\g<1>128\g<2>",
            text,
        )
        if count == 0:
            raise RuntimeError("could not set kv_loop_step=128 in the generated kernel")
        for mask in ("causal_mask", "sliding_or_chunked_causal_mask", "custom_mask"):
            text = text.replace(f"#if 1 // {mask}\n", f"#if 0 // {mask}\n")
    launcher_name = f"run_{kernel}_nl_tiled"
    if "g3" in levers:
        marker = "\n#endif // tiled\n"
        pos = text.rfind(marker)
        if pos < 0:
            raise RuntimeError("could not place the g3 kernel in the generated kernel file")
        snippet = _G3_SNIPPET.format(kname=f"{kernel}_kernel_nl_tiled", launcher=f"run_{kernel}")
        text = text[:pos] + "\n" + snippet + text[pos:]
        launcher_name = f"run_{kernel}_g3_nl_tiled"
    kernel_path = gen_dir / "generated" / f"{kernel}.cu"
    write_if_different(kernel_path, text)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    nvcc_flags.append(f"-I{fmha_v2_src_dir}")
    nvcc_flags.append("-Wno-deprecated-gpu-targets")
    nvcc_flags.append(f"-DFMHA_FLUX3_D128_LAUNCHER={launcher_name}")
    for name in sorted(set(levers)):
        if _LEVER_FLAGS[name][1]:
            nvcc_flags.append(_LEVER_FLAGS[name][1])
    binding = jit_env.FLASHINFER_CSRC_DIR / "trtllm_fmha_v2_binding.cu"
    return gen_jit_spec(uri, [kernel_path, binding], extra_cuda_cflags=nvcc_flags)


@functools.cache
def get_flux3_d128_module(levers: tuple[str, ...] = DEFAULT_LEVERS, min_blocks: int = 0):
    return gen_flux3_d128_spec(levers, min_blocks).build_and_load()


def flux3_d128_kernel_name(levers: Sequence[str] = DEFAULT_LEVERS, min_blocks: int = 0) -> str:
    tag = _tag(levers, min_blocks)
    return STOCK_KERNEL.replace("fmha_v2_flash_attention_", f"fmha_v2_flux3_{tag}_") + "_nl_tiled"


def fmha_v2_flux3_d128(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    *,
    scale_bmm1_d: Optional[torch.Tensor],
    scale_bmm2_d: Optional[torch.Tensor],
    causal: bool = False,
    lse: Optional[torch.Tensor] = None,
    module=None,
) -> torch.Tensor:
    """Same argument convention as flashinfer.prefill.fmha_v2_prefill_sm120 (E4M3, d128,
    BF16 out, contiguous BSHD, device scales, scale_softmax = scale_bmm1 = scale_bmm2 = 1)."""
    if head_dim != 128 or query.dtype != torch.float8_e4m3fn or out.dtype != torch.bfloat16:
        raise ValueError("fmha_v2_flux3_d128 supports E4M3 head_dim 128 with BF16 output only.")
    if module is None:
        module = get_flux3_d128_module()
    module.run(
        query,
        key,
        value,
        out,
        lse,
        scale_bmm1_d,
        scale_bmm2_d,
        num_heads,
        head_dim,
        seq_len,
        1.0,
        1.0,
        1.0,
        causal,
        True,
        True,
    )
    return out


# nvfp4_out (FMHA_FLUX3_NVFP4_OUT, fmha/flux3_nvfp4_out.h + noloop_tiled.h epilogue): the module
# also exports run_nvfp4_out, which writes FP4 [B*S, H*D/2] + 128x4-swizzled E4M3 SF (FlashInfer
# cute-dsl nvfp4_quantize of the BF16 O, bitwise) instead of BF16 O. Its plain run() still stores
# BF16 O (the same kernel, fp4 pointer null). Not in ROUTED_LEVERS: callers add it.
_LEVER_FLAGS["nvfp4_out"] = ("nfo", "-DFMHA_FLUX3_NVFP4_OUT=1")


def nvfp4_scale_bytes(rows: int, width: int) -> int:
    """Bytes of the flat 128x4-swizzled E4M3 scale buffer for an [rows, width] NVFP4 tensor."""
    return ((rows + 127) // 128 * 128) * ((width // 16 + 3) // 4 * 4)


def fmha_v2_flux3_d128_nvfp4_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fp4: torch.Tensor,
    sf: torch.Tensor,
    global_scale: torch.Tensor,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    *,
    scale_bmm1_d: Optional[torch.Tensor],
    scale_bmm2_d: Optional[torch.Tensor],
    module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fmha_v2_flux3_d128 (non-causal, no LSE) with the nvfp4_out epilogue: writes fp4 (uint8
    [B*S, H*D/2]) and sf (uint8, nvfp4_scale_bytes(B*S, H*D)) instead of BF16 O. Bitwise equal to
    FlashInfer nvfp4_quantize(O, global_scale, layout_128x4, cute-dsl) of the same module's BF16 O.
    global_scale is a float32 device tensor of one element. module must carry lever nvfp4_out."""
    if head_dim != 128 or query.dtype != torch.float8_e4m3fn:
        raise ValueError("fmha_v2_flux3_d128_nvfp4_out supports E4M3 head_dim 128 only.")
    module.run_nvfp4_out(
        query,
        key,
        value,
        fp4,
        sf,
        global_scale,
        scale_bmm1_d,
        scale_bmm2_d,
        num_heads,
        head_dim,
        seq_len,
    )
    return fp4, sf


# Q-row prefix (Launch_params.flux3_q_rows, G3 launcher only): run_q_rows / run_nvfp4_out_q_rows
# compute Q tiles [0, ceil(q_rows / 64)) rounded up to whole 3-group CTAs, over all seq_len K/V
# rows. q_rows == 0 or >= seq_len runs all rows. Rows [0, q_rows) are bitwise the full launch's
# (each Q tile is computed independently); rows past q_prefix_rows_written(q_rows) are not
# written. The 4-warp fallback (calls G3 does not support) ignores q_rows and runs all rows.
G3_GROUP_ROWS = 192


def q_prefix_rows_written(q_rows: int) -> int:
    """Rows a q_rows prefix launch writes: ceil(ceil(q_rows / 64) / 3) * 192 (16512 for 16384)."""
    return ((q_rows + 63) // 64 + 2) // 3 * G3_GROUP_ROWS


def fmha_v2_flux3_d128_q_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    q_rows: int,
    *,
    scale_bmm1_d: Optional[torch.Tensor],
    scale_bmm2_d: Optional[torch.Tensor],
    module,
) -> torch.Tensor:
    """fmha_v2_flux3_d128 (non-causal, no LSE) computing only the Q-row prefix q_rows."""
    if head_dim != 128 or query.dtype != torch.float8_e4m3fn or out.dtype != torch.bfloat16:
        raise ValueError("fmha_v2_flux3_d128_q_rows supports E4M3 head_dim 128 -> BF16 only.")
    module.run_q_rows(
        query,
        key,
        value,
        out,
        None,
        scale_bmm1_d,
        scale_bmm2_d,
        num_heads,
        head_dim,
        seq_len,
        1.0,
        1.0,
        1.0,
        False,
        True,
        True,
        int(q_rows),
    )
    return out


def fmha_v2_flux3_d128_nvfp4_out_q_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fp4: torch.Tensor,
    sf: torch.Tensor,
    global_scale: torch.Tensor,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    q_rows: int,
    *,
    scale_bmm1_d: Optional[torch.Tensor],
    scale_bmm2_d: Optional[torch.Tensor],
    module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fmha_v2_flux3_d128_nvfp4_out computing only the Q-row prefix q_rows (fp4 rows and SF row
    blocks of rows [0, q_rows) bitwise the full launch's)."""
    if head_dim != 128 or query.dtype != torch.float8_e4m3fn:
        raise ValueError("fmha_v2_flux3_d128_nvfp4_out_q_rows supports E4M3 head_dim 128 only.")
    module.run_nvfp4_out_q_rows(
        query,
        key,
        value,
        fp4,
        sf,
        global_scale,
        scale_bmm1_d,
        scale_bmm2_d,
        num_heads,
        head_dim,
        seq_len,
        int(q_rows),
    )
    return fp4, sf


# softmax_trim (FMHA_FLUX3_SOFTMAX_TRIM, softmax.h; bitwise vs base2): P packs to FP8 without the
# multiply by scale_softmax (requires scale_softmax == 1, as fmha_v2_flux3_d128 passes) and the
# row max reduces as a depth-4 fmaxf tree. TRTLLM_FMHA_V2_FLUX3_ST=0 drops it from the routed set.
import os as _os_st  # noqa: E402

if _os_st.environ.get("TRTLLM_FMHA_V2_FLUX3_ST", "1") != "0":
    ROUTED_LEVERS = (*ROUTED_LEVERS, "softmax_trim")

# g3 (FMHA_FLUX3_G3, fused_multihead_flash_attention_kernel_flux3_g3.h, padding mask only; bitwise
# vs the same lever set without g3): 1 CTA/SM, 384 threads = 3 groups x the unchanged 4-warp 64-row
# tile (192 Q rows per CTA) over one TMA/mbarrier K/V ring. Unsupported calls (causal, LSE, GQA,
# ...) fall back to the 4-warp launcher of the same module. TRTLLM_FMHA_V2_FLUX3_G3=0 drops it.
_LEVER_FLAGS["g3"] = ("g3", "-DFMHA_FLUX3_G3=1")

_G3_SNIPPET = r"""
// ---- FLUX.3 G3 shared-K/V variant (lever g3) ----
#include <fused_multihead_flash_attention_kernel_flux3_g3.h>

extern "C"
__global__ __launch_bounds__(384, 1)
void {kname}_g3(const __grid_constant__ bert::Fused_multihead_attention_params_v2 params,
                const __grid_constant__ CUtensorMap tm_k,
                const __grid_constant__ CUtensorMap tm_v){{
  fused_multihead_attention::device_flash_attention_nl_tiled_flux3_g3<Kernel_traits_nl_tiled>(
      params, &tm_k, &tm_v);
}}

#if defined(FMHA_FLUX3_V_CF) && FMHA_FLUX3_V_CF
// v_cf: the same kernel reading V from the swap23 stage layout (5D map, conflict-free V LDSM).
extern "C"
__global__ __launch_bounds__(384, 1)
void {kname}_g3_vcf(const __grid_constant__ bert::Fused_multihead_attention_params_v2 params,
                    const __grid_constant__ CUtensorMap tm_k,
                    const __grid_constant__ CUtensorMap tm_v){{
  fused_multihead_attention::device_flash_attention_nl_tiled_flux3_g3<Kernel_traits_nl_tiled,
                                                                      true>(params, &tm_k, &tm_v);
}}
#endif

void {launcher}_g3_nl_tiled(
    const bert::Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){{
  namespace g3 = fused_multihead_attention::flux3_g3;
  if( !g3::supported(params, launch_params) ) {{
    {launcher}_nl_tiled(params, launch_params, stream);
    return;
  }}
  CUtensorMap tm_k, tm_v;
  g3::make_map(&tm_k, params.k_ptr, (int)params.b, (int)params.s, (int)params.h_kv, 128,
               (int64_t)params.k_stride_in_bytes, 64, CU_TENSOR_MAP_SWIZZLE_64B);
#if defined(FMHA_FLUX3_V_CF) && FMHA_FLUX3_V_CF
  bool const v_cf = g3::v_swap23_ok(params);
#else
  bool const v_cf = false;
#endif
  if( v_cf ) {{
#if defined(FMHA_FLUX3_V_CF) && FMHA_FLUX3_V_CF
    g3::make_map_v_swap23(&tm_v, params.v_ptr, (int)params.b, (int)params.s, (int)params.h_kv,
                          128, (int64_t)params.v_stride_in_bytes);
#endif
  }} else {{
    g3::make_map(&tm_v, params.v_ptr, (int)params.b, (int)params.s, (int)params.h_kv, 128,
                 (int64_t)params.v_stride_in_bytes, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  }}
  constexpr int smem_size = g3::Layout<Kernel_traits_nl_tiled>::SMEM_BYTES;
  static bool attr_set = false;
  if( !attr_set ) {{
    FMHA_CHECK_CUDA(cudaFuncSetAttribute({kname}_g3,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
#if defined(FMHA_FLUX3_V_CF) && FMHA_FLUX3_V_CF
    FMHA_CHECK_CUDA(cudaFuncSetAttribute({kname}_g3_vcf,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
#endif
    attr_set = true;
  }}
  // Q-row prefix (run_q_rows / run_nvfp4_out_q_rows): 0 or >= params.s runs all rows. The kernel
  // derives its tile range from blockIdx.x and params.s only, so a smaller grid.x computes the
  // leading q_tiles * 64 rows (rounded up to whole CTAs) over every K/V row.
  int q_rows = (int)params.s;
  if( launch_params.flux3_q_rows > 0 && launch_params.flux3_q_rows < q_rows ) {{
    q_rows = launch_params.flux3_q_rows;
  }}
  int q_tiles = (q_rows + 63) / 64;
  dim3 grid((q_tiles + g3::GROUPS - 1) / g3::GROUPS, params.h, params.b);
#if defined(FMHA_FLUX3_V_CF) && FMHA_FLUX3_V_CF
  if( v_cf ) {{
    {kname}_g3_vcf<<<grid, g3::THREADS, smem_size, stream>>>(params, tm_k, tm_v);
    return;
  }}
#endif
  {kname}_g3<<<grid, g3::THREADS, smem_size, stream>>>(params, tm_k, tm_v);
}}
"""

if _os_st.environ.get("TRTLLM_FMHA_V2_FLUX3_G3", "1") != "0":
    ROUTED_LEVERS = (*ROUTED_LEVERS, "g3")

# q_regs (FMHA_FLUX3_Q_REGS, noloop_tiled.h + flux3_g3.h; bitwise vs the same levers without it):
# with resident Q, the 2 x MMAS_K Q A-fragments are read from smem once (4-warp: on the first KV
# step; G3: after the Q load) and stay in registers instead of an LDSM per KV step.
# TRTLLM_FMHA_V2_FLUX3_QR=0 drops it from the routed set.
_LEVER_FLAGS["q_regs"] = ("qr", "-DFMHA_FLUX3_Q_REGS=1")

if _os_st.environ.get("TRTLLM_FMHA_V2_FLUX3_QR", "1") != "0":
    ROUTED_LEVERS = (*ROUTED_LEVERS, "q_regs")

# v_b8t (FMHA_FLUX3_V_B8T, flux3_g3.h only; bitwise vs the same levers without it): the G3 BMM2
# loads each V B-fragment with one ldmatrix.m16n16.x2.trans.b8 from per-lane key-permuted rows
# instead of LDSM.MT88.4 + 4 PRMT, and moves acc_o back to main's d-map once per tile (quad
# shuffle) before the unchanged BF16 / NVFP4 epilogues. The 4-warp kernel is unchanged.
# TRTLLM_FMHA_V2_FLUX3_VB8=0 drops it from the routed set.
_LEVER_FLAGS["v_b8t"] = ("vb8", "-DFMHA_FLUX3_V_B8T=1")

if _os_st.environ.get("TRTLLM_FMHA_V2_FLUX3_VB8", "1") != "0" and "g3" in ROUTED_LEVERS:
    ROUTED_LEVERS = (*ROUTED_LEVERS, "v_b8t")

# v_cf (FMHA_FLUX3_V_CF, flux3_g3.h + the g3 launcher; needs v_b8t; bitwise vs the same levers
# without it): the V TMA uses a 5D map that writes each 16-key group with key bits 2 and 3
# swapped, and the v_b8t lane offsets address swap23(key), so the V
# ldmatrix.m16n16.x2.trans.b8 loads are bank-conflict free (4 wavefronts instead of 8). Calls
# with s % 16 != 0 (or b > 1 and s % 64 != 0) run the unchanged G3 kernel with the 4D map.
# TRTLLM_FMHA_V2_FLUX3_VCF=0 drops it from the routed set.
_LEVER_FLAGS["v_cf"] = ("vcf", "-DFMHA_FLUX3_V_CF=1")

if _os_st.environ.get("TRTLLM_FMHA_V2_FLUX3_VCF", "1") != "0" and "v_b8t" in ROUTED_LEVERS:
    ROUTED_LEVERS = (*ROUTED_LEVERS, "v_cf")
