"""Idea 39 JIT overlay: a private copy of FlashInfer's data/include + data/csrc with the
64-row head_dim-512 FP8-KV paged prefill kernel applied.

Microbenchmark tool only. The candidate arm imports this module BEFORE anything else
from flashinfer touches the JIT, with FLASHINFER_WORKSPACE_BASE pointing at a fresh
directory (all arms share one JIT URI). apply() points flashinfer.jit.env at the copy
and disables AOT artifacts so the module is really rebuilt from the overlay.

The edits below are the exact change stage 2 commits into the FlashInfer tree.
"""
import os
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
NEW_FILES = HERE / "fi"

MARKER = '#pragma message("idea39 d512 tile64 overlay active")\n'

EDITS = [
    # ---------------------------------------------------------------- prefill.cuh
    ("include/flashinfer/attention/prefill.cuh",
     '#include "variants.cuh"\nnamespace flashinfer {',
     '#include "variants.cuh"\n#include "prefill_d512_tile64.cuh"\nnamespace flashinfer {'),
    ("include/flashinfer/attention/prefill.cuh",
     "                                                   cudaStream_t stream) {\n"
     "  using DTypeQ = typename Params::DTypeQ;\n"
     "  using DTypeKV = typename Params::DTypeKV;\n"
     "  using DTypeO = typename Params::DTypeO;\n"
     "  const uint32_t padded_batch_size = params.padded_batch_size;\n"
     "  const uint32_t num_qo_heads = params.num_qo_heads;\n"
     "  const uint32_t num_kv_heads = params.paged_kv.num_heads;\n"
     "  // hd512 + 16-bit/NVFP4 KV uses",
     "                                                   cudaStream_t stream) {\n"
     "  using DTypeQ = typename Params::DTypeQ;\n"
     "  using DTypeKV = typename Params::DTypeKV;\n"
     "  using DTypeO = typename Params::DTypeO;\n"
     "  if constexpr (CTA_TILE_Q == 64 && HEAD_DIM_VO >= 512) {\n"
     "    // 64-row head_dim-512 tiles run on the dedicated FP8-KV kernel (prefill_d512_tile64.cuh);\n"
     "    // the planner selects them only for modules FA2D512Tile64Eligible accepts.\n"
     "    if constexpr (FA2D512Tile64Eligible<DTypeQ, DTypeKV, DTypeO, HEAD_DIM_QK, HEAD_DIM_VO,\n"
     "                                        POS_ENCODING_MODE, USE_FP16_QK_REDUCTION,\n"
     "                                        USE_TOKEN_HEAD_SF>()) {\n"
     "      return d512t64::BatchPrefillPagedD512Tile64Dispatched<SAME_KV_STRIDES, MASK_MODE,\n"
     "                                                            AttentionVariant>(\n"
     "          params, tmp_v, tmp_s, enable_pdl, stream);\n"
     "    } else {\n"
     "      FLASHINFER_ERROR(\"cta_tile_q=64 at head_dim 512 needs an FA2D512Tile64Eligible module\");\n"
     "      return cudaErrorNotSupported;\n"
     "    }\n"
     "  }\n"
     "  const uint32_t padded_batch_size = params.padded_batch_size;\n"
     "  const uint32_t num_qo_heads = params.num_qo_heads;\n"
     "  const uint32_t num_kv_heads = params.paged_kv.num_heads;\n"
     "  // hd512 + 16-bit/NVFP4 KV uses"),
    # ---------------------------------------------------------------- scheduler.cuh
    ("include/flashinfer/attention/scheduler.cuh",
     '#include "../utils.cuh"\n#include "heap.h"\n',
     '#include "../utils.cuh"\n#include "d512_tile64_config.cuh"\n#include "heap.h"\n'),
    ("include/flashinfer/attention/scheduler.cuh",
     "                                   uint32_t head_dim_qk = 0, uint32_t kv_dtype_bytes = 2) {\n"
     "  std::vector<IdType> request_indices,",
     "                                   uint32_t head_dim_qk = 0, uint32_t kv_dtype_bytes = 2,\n"
     "                                   bool d512_tile64_kv = false) {\n"
     "  std::vector<IdType> request_indices,"),
    ("include/flashinfer/attention/scheduler.cuh",
     "      cta_tile_q = FA2DetermineCtaTileQ(packed_uniform_len, head_dim, head_dim_qk, kv_dtype_bytes);\n",
     "      cta_tile_q = FA2MaybeD512Tile64(\n"
     "          FA2DetermineCtaTileQ(packed_uniform_len, head_dim, head_dim_qk, kv_dtype_bytes),\n"
     "          packed_uniform_len, head_dim, head_dim_qk, page_size, d512_tile64_kv);\n"),
    ("include/flashinfer/attention/scheduler.cuh",
     "      cta_tile_q = FA2DetermineCtaTileQ(max_qo_len, head_dim, head_dim_qk, kv_dtype_bytes);\n",
     "      cta_tile_q = FA2MaybeD512Tile64(\n"
     "          FA2DetermineCtaTileQ(max_qo_len, head_dim, head_dim_qk, kv_dtype_bytes), max_qo_len,\n"
     "          head_dim, head_dim_qk, page_size, d512_tile64_kv);\n"),
    ("include/flashinfer/attention/scheduler.cuh",
     "    cta_tile_q = FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim, head_dim_qk, kv_dtype_bytes);\n",
     "    cta_tile_q = FA2MaybeD512Tile64(\n"
     "        FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim, head_dim_qk, kv_dtype_bytes),\n"
     "        avg_packed_qo_len, head_dim, head_dim_qk, page_size, d512_tile64_kv);\n"),
    ("include/flashinfer/attention/scheduler.cuh",
     "    int64_t uniform_q_len, bool fixed_uniform_q_len, cudaStream_t stream,\n"
     "    uint32_t kv_dtype_bytes = 2) {\n"
     "  (void)sizeof_dtype_o;",
     "    int64_t uniform_q_len, bool fixed_uniform_q_len, cudaStream_t stream,\n"
     "    uint32_t kv_dtype_bytes = 2, bool d512_tile64_kv = false) {\n"
     "  (void)sizeof_dtype_o;"),
    ("include/flashinfer/attention/scheduler.cuh",
     "                             uniform_q_len, fixed_uniform_q_len, head_dim_qk, kv_dtype_bytes);\n",
     "                             uniform_q_len, fixed_uniform_q_len, head_dim_qk, kv_dtype_bytes,\n"
     "                             d512_tile64_kv);\n"),
    ("include/flashinfer/attention/scheduler.cuh",
     "                               int64_t uniform_q_len, bool fixed_uniform_q_len, cudaStream_t stream,\n"
     "                               uint32_t kv_dtype_bytes = 2) {\n"
     "  size_t used_float_workspace_size = 0;",
     "                               int64_t uniform_q_len, bool fixed_uniform_q_len, cudaStream_t stream,\n"
     "                               uint32_t kv_dtype_bytes = 2, bool d512_tile64_kv = false) {\n"
     "  size_t used_float_workspace_size = 0;"),
    ("include/flashinfer/attention/scheduler.cuh",
     "                               uniform_q_len, fixed_uniform_q_len, stream, kv_dtype_bytes);\n}",
     "                               uniform_q_len, fixed_uniform_q_len, stream, kv_dtype_bytes,\n"
     "                               d512_tile64_kv);\n}"),
    ("include/flashinfer/attention/scheduler.cuh",
     "    bool fixed_uniform_q_len, cudaStream_t stream, uint32_t kv_dtype_bytes = 2) {\n"
     "  PrefillPlanInfo plan_info;",
     "    bool fixed_uniform_q_len, cudaStream_t stream, uint32_t kv_dtype_bytes = 2,\n"
     "    bool d512_tile64_kv = false) {\n"
     "  PrefillPlanInfo plan_info;"),
    ("include/flashinfer/attention/scheduler.cuh",
     "      num_colocated_ctas, uniform_q_len, fixed_uniform_q_len, stream, kv_dtype_bytes);\n}",
     "      num_colocated_ctas, uniform_q_len, fixed_uniform_q_len, stream, kv_dtype_bytes,\n"
     "      d512_tile64_kv);\n}"),
    # ---------------------------------------------------------------- csrc
    ("csrc/batch_prefill.cu",
     "using namespace flashinfer;\n\nusing tvm::ffi::Array;",
     "using namespace flashinfer;\n\n"
     "// FP8-KV head_dim-512 paged modules plan 64-row tiles (attention/d512_tile64_config.cuh).\n"
     "constexpr bool kD512Tile64KV =\n"
     "    FA2D512Tile64Eligible<DTypeQ, DTypeKV, DTypeO, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,\n"
     "                          USE_FP16_QK_REDUCTION, USE_TOKEN_HEAD_SF>();\n\n"
     "using tvm::ffi::Array;"),
    ("csrc/batch_prefill.cu",
     "      uniform_q_len, fixed_uniform_q_len, stream, /*kv_dtype_bytes=*/sizeof(DTypeKV));",
     "      uniform_q_len, fixed_uniform_q_len, stream, /*kv_dtype_bytes=*/sizeof(DTypeKV),\n"
     "      /*d512_tile64_kv=*/kD512Tile64KV);"),
    ("csrc/batch_prefill.cu",
     "      num_colocated_ctas, uniform_q_len, fixed_uniform_q_len, stream,\n"
     "      /*kv_dtype_bytes=*/sizeof(DTypeKV));",
     "      num_colocated_ctas, uniform_q_len, fixed_uniform_q_len, stream,\n"
     "      /*kv_dtype_bytes=*/sizeof(DTypeKV), /*d512_tile64_kv=*/kD512Tile64KV);"),
    ("csrc/batch_prefill_paged_kernel_inst.jinja",
     "{% for cta_tile_q in ([16, 32] if head_dim_vo | int >= 512 else ([16] if head_dim_qk | int >= 512 else [16, 64, 128])) %}",
     "{% for cta_tile_q in (([16, 32, 64] if (head_dim_vo | int == 512 and head_dim_qk | int == 512 and dtype_kv in ['__nv_fp8_e4m3', '__nv_fp8_e5m2']) else [16, 32]) if head_dim_vo | int >= 512 else ([16] if head_dim_qk | int >= 512 else [16, 64, 128])) %}"),
]


def _package_data():
    import importlib.util
    spec = importlib.util.find_spec("flashinfer")
    root = pathlib.Path(spec.origin).resolve().parent
    return root / "data"


def build(dst: pathlib.Path, marker: bool = True) -> pathlib.Path:
    """Copy FlashInfer's include/csrc to dst and apply the idea-39 edits."""
    data = _package_data()
    dst = pathlib.Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ("include", "csrc"):
        shutil.copytree(data / sub, dst / sub, symlinks=False)
    for src in NEW_FILES.rglob("*"):
        if src.is_file():
            rel = src.relative_to(NEW_FILES)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
    for rel, old, new in EDITS:
        p = dst / rel
        text = p.read_text()
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"overlay edit anchor found {n}x in {rel}: {old[:80]!r}")
        p.write_text(text.replace(old, new))
    if marker:
        k = dst / "include/flashinfer/attention/prefill_d512_tile64.cuh"
        text = k.read_text()
        k.write_text(text.replace("namespace flashinfer {\nnamespace d512t64 {",
                                  MARKER + "namespace flashinfer {\nnamespace d512t64 {", 1))
    return dst


def apply(dst: pathlib.Path):
    """Point flashinfer's JIT at the overlay copy (call before any module is built)."""
    from flashinfer.jit import env as jit_env
    jit_env.FLASHINFER_INCLUDE_DIR = pathlib.Path(dst) / "include"
    jit_env.FLASHINFER_CSRC_DIR = pathlib.Path(dst) / "csrc"
    jit_env.get_aot_artifacts = lambda name: ()
    print(f"[overlay] include={jit_env.FLASHINFER_INCLUDE_DIR} csrc={jit_env.FLASHINFER_CSRC_DIR} "
          f"jit_dir={jit_env.FLASHINFER_JIT_DIR}", flush=True)
    return jit_env
