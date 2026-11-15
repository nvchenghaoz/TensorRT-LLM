"""Idea 38: text patches for FlashInfer include/flashinfer/attention/prefill.cuh (pin 775207de).

Variants (each a list of edits; an edit = (region, old, new), old must occur once in region)
  v1   launcher: d256 one-byte KV + 16-bit Q -> repack-free traits (in-register FP8->BF16
       dequant) when only they reach 2 CTAs/SM (SM120: NUM_MMA_KV 2, 49152 B, 8 warps/SM).
  v1b  diagnostic: repack-free traits at main's tiling (1 CTA/SM, NUM_MMA_KV 4).
  v2m  repack_fp8_tile_to_bf16: conflict-free thread->(row, col) map for 16-col FP8 rows
       (d256). Same values, same layout: exact.
  v2e  paged mainloop, FP8 repack path: issue K(i+1) right after K(i) is staged and V(i+1)
       right after V(i) is staged; 4 block syncs per KV tile instead of 6. Exact.
  v2   v2m + v2e.

make_overlay(variant, dst) copies FlashInfer's packaged include tree to dst/include and patches
prefill.cuh there; FLASHINFER_EXTRA_CUDAFLAGS=-I<dst>/include makes the JIT build use it.
'pin' rebuilds the pinned header from a tree that carries one of the variants (reverse patch).
"""
import os
import shutil

REGIONS = {
    "file": (None, None),
    "device": ("void BatchPrefillWithPagedKVCacheDevice(", "void BatchPrefillWithPagedKVCacheKernel("),
    "launcher": ("cudaError_t BatchPrefillWithPagedKVCacheDispatched(", None),
}

# ---------------------------------------------------------------- v1 (launcher)
L_OLD1 = """  const bool disable_repack = USE_TOKEN_HEAD_SF && !kSmemFitsRepackWithScale;
  const uint32_t kKVSmemPerMmaKVEffective =
      disable_repack ? kKVSmemPerMmaKVNoRepack : kKVSmemPerMmaKV;
  const int num_ctas_per_sm =
      max_smem_per_sm >= 2 * (kFixedSmem + kMinValidMmaKV * kKVSmemPerMmaKV) ? 2 : 1;
  const int max_smem_per_threadblock =
      min(max_smem_per_sm / num_ctas_per_sm, max_smem_per_block_optin);
"""

L_NEW1_V1 = """  // One-byte KV at head_dim 256 with 16-bit Q: the 16-bit repack staging doubles the per-mma_kv
  // K/V smem, so on ~100 KB-per-SM parts (SM86/89/120/121) the repack layout fits 1 CTA per SM
  // while the repack-free layout (one-byte -> 16-bit dequant in registers; the MMA sees the same
  // 16-bit operands) fits 2. Take the repack-free traits when only they reach 2 CTAs per SM.
  constexpr bool kNoRepackOccCandidate = kUseRepack && !is_fp4_type_v<DTypeKV> &&
                                         !USE_TOKEN_HEAD_SF && sizeof(DTypeQ) == 2 &&
                                         HEAD_DIM_QK == 256 && HEAD_DIM_VO == 256;
  // The hardware reserves 1 KiB per block on top of the dynamic smem.
  const bool norepack_occ =
      kNoRepackOccCandidate &&
      max_smem_per_sm < 2 * (kFixedSmem + kMinValidMmaKV * kKVSmemPerMmaKV) &&
      max_smem_per_sm >= 2 * (kFixedSmem + kMinValidMmaKV * kKVSmemPerMmaKVNoRepack + 1024);
  const bool disable_repack = (USE_TOKEN_HEAD_SF && !kSmemFitsRepackWithScale) || norepack_occ;
  const uint32_t kKVSmemPerMmaKVEffective =
      disable_repack ? kKVSmemPerMmaKVNoRepack : kKVSmemPerMmaKV;
  const int num_ctas_per_sm =
      (norepack_occ || max_smem_per_sm >= 2 * (kFixedSmem + kMinValidMmaKV * kKVSmemPerMmaKV))
          ? 2
          : 1;
  const int max_smem_per_threadblock =
      min(max_smem_per_sm / num_ctas_per_sm, max_smem_per_block_optin);
"""

L_NEW1_V1B = """  // idea38 v1b diagnostic: repack-free traits at main's tiling (1 CTA/SM, NUM_MMA_KV 4).
  constexpr bool kNoRepackOccCandidate = kUseRepack && !is_fp4_type_v<DTypeKV> &&
                                         !USE_TOKEN_HEAD_SF && sizeof(DTypeQ) == 2 &&
                                         HEAD_DIM_QK == 256 && HEAD_DIM_VO == 256;
  const bool norepack_occ = kNoRepackOccCandidate;
  const bool disable_repack = (USE_TOKEN_HEAD_SF && !kSmemFitsRepackWithScale) || norepack_occ;
  const uint32_t kKVSmemPerMmaKVEffective =
      (USE_TOKEN_HEAD_SF && !kSmemFitsRepackWithScale) ? kKVSmemPerMmaKVNoRepack : kKVSmemPerMmaKV;
  const int num_ctas_per_sm =
      max_smem_per_sm >= 2 * (kFixedSmem + kMinValidMmaKV * kKVSmemPerMmaKV) ? 2 : 1;
  const int max_smem_per_threadblock =
      min(max_smem_per_sm / num_ctas_per_sm, max_smem_per_block_optin);
"""

L_OLD2 = """          auto kernel = BatchPrefillWithPagedKVCacheKernel<SAME_KV_STRIDES, KTraits, Params>;
          // Exact final check: the analytic NUM_MMA_KV budget can slightly
"""

L_NEW2 = """          auto kernel = BatchPrefillWithPagedKVCacheKernel<SAME_KV_STRIDES, KTraits, Params>;
          if constexpr (kNoRepackOccCandidate) {
            if (norepack_occ) {
              // smem_size above is already the repack-free layout (disable_repack).
              kernel = BatchPrefillWithPagedKVCacheKernel<SAME_KV_STRIDES, KTraitsNoRepack, Params>;
              FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                  cudaSharedmemCarveoutMaxShared));
            }
          }
          // Exact final check: the analytic NUM_MMA_KV budget can slightly
"""

# ---------------------------------------------------------------- v2m (repack map)
M_OLD = """  for (uint32_t idx = thread_id; idx < NUM_B128; idx += NUM_THREADS) {
    uint32_t row = idx / FP8_COLS, col = idx % FP8_COLS;
"""

M_NEW = """  for (uint32_t idx = thread_id; idx < NUM_B128; idx += NUM_THREADS) {
    uint32_t row = idx / FP8_COLS, col = idx % FP8_COLS;
    if constexpr (FP8_COLS == 16 && SWIZZLE == SwizzleMode::k128B && NUM_THREADS % 32 == 0) {
      // Each warp converts 2 rows x 16 b128 columns. With the row-major map, the 8 lanes of a
      // quarter-warp write columns 2*col of one row, which land on 4 of the 8 16-byte bank groups
      // (2-way conflict on both staging stores). Here each quarter-warp takes 4 columns of the
      // even row and the paired 4 columns of the odd row: the k128B XOR (row % 8) then spreads
      // both the b128 reads and the two b128 writes over all 8 bank groups. Same data, same
      // layout, only the thread that moves each chunk changes.
      const uint32_t lane = idx % 32, quarter = lane / 8, half = (lane % 8) / 4;
      row = (idx / 32) * 2 + half;
      col = 8 * (quarter / 2) + 4 * ((quarter % 2) ^ half) + (lane % 4);
    }
"""

# ---------------------------------------------------------------- v2e (early K/V issue)
E_DECL_OLD = """      constexpr bool kRepackActive = KTraits::USE_KV_REPACK;
"""
E_DECL_NEW = """      constexpr bool kRepackActive = KTraits::USE_KV_REPACK;
      // FP8 repack path: once a K (V) tile has been staged as 16-bit, its one-byte buffer is free,
      // so the next tile's load is issued right after the staging sync instead of after QK (PV).
      // This drops two block syncs per KV tile and gives each load a longer lead.
      constexpr bool kEarlyKVIssue = kRepackActive && !is_fp4_type_v<DTypeKV> &&
                                     !KTraits::USE_TOKEN_HEAD_SF &&
                                     !KTraits::USE_KV_SHARED_SMEM && !KTraits::USE_VO_SPLIT;
"""

E_K_OLD = """          repack_kv_tile_to_16b<KTraits, KTraits::HEAD_DIM_QK>(
              smem_storage.k_smem, smem_storage.k_sf_smem_ptr(), smem_storage.kv_smem_repack_ptr(),
              warp_idx * WARP_SIZE + lane_idx);
          block.sync();
          compute_qk<KTraits, /*REPACK_BF16=*/true>(
"""
E_K_NEW = """          repack_kv_tile_to_16b<KTraits, KTraits::HEAD_DIM_QK>(
              smem_storage.k_smem, smem_storage.k_sf_smem_ptr(), smem_storage.kv_smem_repack_ptr(),
              warp_idx * WARP_SIZE + lane_idx);
          block.sync();
          if constexpr (kEarlyKVIssue) {
            page_produce_kv<false, KTraits>(&smem_storage, &k_smem_offset_w, paged_kv.k_data,
                                            (iter + 1) * CTA_TILE_KV, thr_local_kv_offset_k,
                                            chunk_size, warp_idx, lane_idx);
            cp_async::commit_group();
          }
          compute_qk<KTraits, /*REPACK_BF16=*/true>(
"""

E_MID_OLD = """        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {
          page_produce_kv_on_the_fly<true, KTraits>(
"""
E_MID_NEW = """        if constexpr (kEarlyKVIssue) {
          // K(iter + 1) is already in flight: wait for V(iter) only. The block.sync below also
          // orders every warp's QK reads of the staging buffer before the V repack.
          cp_async::wait_group<1>();
        } else {
        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {
          page_produce_kv_on_the_fly<true, KTraits>(
"""
E_MID2_OLD = """          cp_async::commit_group();
          cp_async::wait_group<1>();
        }
        block.sync();

        // compute sfm*v
"""
E_MID2_NEW = """          cp_async::commit_group();
          cp_async::wait_group<1>();
        }
        }  // kEarlyKVIssue
        block.sync();

        // compute sfm*v
"""

E_V_OLD = """          repack_kv_tile_to_16b<KTraits, KTraits::HEAD_DIM_VO>(
              smem_storage.v_smem, smem_storage.v_sf_smem_ptr(), smem_storage.kv_smem_repack_ptr(),
              warp_idx * WARP_SIZE + lane_idx);
          block.sync();
          compute_sfm_v<KTraits, /*REPACK_BF16=*/true>(
"""
E_V_NEW = """          repack_kv_tile_to_16b<KTraits, KTraits::HEAD_DIM_VO>(
              smem_storage.v_smem, smem_storage.v_sf_smem_ptr(), smem_storage.kv_smem_repack_ptr(),
              warp_idx * WARP_SIZE + lane_idx);
          block.sync();
          if constexpr (kEarlyKVIssue) {
            if constexpr (SAME_KV_STRIDES) {
              page_produce_kv<true, KTraits>(&smem_storage, &v_smem_offset_w, paged_kv.v_data,
                                             (iter + 1) * CTA_TILE_KV, thr_local_kv_offset_k,
                                             chunk_size, warp_idx, lane_idx);
            } else {
              page_produce_kv<true, KTraits>(&smem_storage, &v_smem_offset_w, paged_kv.v_data,
                                             (iter + 1) * CTA_TILE_KV, thr_local_kv_offset_v,
                                             chunk_size, warp_idx, lane_idx);
            }
            cp_async::commit_group();
          }
          compute_sfm_v<KTraits, /*REPACK_BF16=*/true>(
"""

E_END1_OLD = """        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {
          page_produce_kv_on_the_fly<false, KTraits>(
"""
E_END1_NEW = """        if constexpr (!kEarlyKVIssue) {
        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {
          page_produce_kv_on_the_fly<false, KTraits>(
"""
E_END2_OLD = """          cp_async::commit_group();
        }
      }
      cp_async::wait_group<0>();
      block.sync();

      finalize_m<KTraits>(variant, m);
"""
E_END2_NEW = """          cp_async::commit_group();
        }
        }  // !kEarlyKVIssue
      }
      cp_async::wait_group<0>();
      block.sync();

      finalize_m<KTraits>(variant, m);
"""

V1_EDITS = [("launcher", L_OLD1, L_NEW1_V1), ("launcher", L_OLD2, L_NEW2)]
V1B_EDITS = [("launcher", L_OLD1, L_NEW1_V1B), ("launcher", L_OLD2, L_NEW2)]
M_EDITS = [("file", M_OLD, M_NEW)]
E_EDITS = [("device", E_DECL_OLD, E_DECL_NEW), ("device", E_K_OLD, E_K_NEW),
           ("device", E_MID_OLD, E_MID_NEW), ("device", E_MID2_OLD, E_MID2_NEW),
           ("device", E_V_OLD, E_V_NEW), ("device", E_END1_OLD, E_END1_NEW),
           ("device", E_END2_OLD, E_END2_NEW)]

VARIANTS = {
    "v1": V1_EDITS,
    "v1b": V1B_EDITS,
    "v2m": M_EDITS,
    "v2e": E_EDITS,
    "v2": M_EDITS + E_EDITS,
    "v12": V1_EDITS + M_EDITS + E_EDITS,
}


def _span(text, region):
    a, b = REGIONS[region]
    i = 0 if a is None else text.index(a)
    j = len(text) if b is None else text.index(b, i)
    return i, j


def apply_edits(text, edits):
    for region, old, new in edits:
        i, j = _span(text, region)
        seg = text[i:j]
        n = seg.count(old)
        if n != 1:
            raise RuntimeError(f"edit anchor occurs {n} times in {region}: {old[:90]!r}")
        text = text[:i] + seg.replace(old, new) + text[j:]
    return text


def reverse_edits(edits):
    return [(r, new, old) for r, old, new in reversed(edits)]


def tree_variant(text):
    """Which variant the given (tree) text already carries, or 'pin'."""
    for name in ("v12", "v2", "v1", "v2e", "v2m", "v1b"):
        if all(new in text for _, _, new in VARIANTS[name]):
            return name
    return "pin"


def to_pin(text):
    tv = tree_variant(text)
    if tv == "pin":
        return text
    return apply_edits(text, reverse_edits(VARIANTS[tv]))


def patch_text(text, variant):
    """Return prefill.cuh text for `variant`, starting from whatever the tree carries."""
    pin = to_pin(text)
    out = pin if variant == "pin" else apply_edits(pin, VARIANTS[variant])
    return f'#pragma message("idea38 overlay: {variant}")\n' + out


def include_dir():
    # Locate flashinfer/data/include without importing flashinfer (its import reads
    # FLASHINFER_WORKSPACE_BASE, which the caller sets afterwards).
    import importlib.util
    spec = importlib.util.find_spec("flashinfer")
    root = os.path.dirname(os.path.realpath(spec.origin))
    return os.path.realpath(os.path.join(root, "data", "include"))


def make_overlay(variant, dst, src_include=None):
    """Copy the include tree to dst/include, patch prefill.cuh; returns (dst/include, tree var)."""
    src = src_include or include_dir()
    out = os.path.join(dst, "include")
    shutil.copytree(src, out, symlinks=False)
    p = os.path.join(out, "flashinfer", "attention", "prefill.cuh")
    with open(p) as f:
        text = f.read()
    new = patch_text(text, variant)
    with open(p, "w") as f:
        f.write(new)
    return out, tree_variant(text)
