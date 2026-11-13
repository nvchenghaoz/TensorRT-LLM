"""SM120 BF16 gate/up GEMM (CuTe DSL, TMA + mma.sync m16n8k16) with an optional SwiGLU epilogue.

Numerics contract (bitwise to F.linear -> swiglu on SM120):
- every output element is ONE in-order chain of mma.sync m16n8k16 (BF16 in, FP32 accumulate)
  over K = 0..K-1 in steps of 16, no split-K, the same chain cuBLASLt's nvjet_sm120 and a
  plain Triton tl.dot GEMM run (fact 662);
- the FP32 accumulator is rounded once to BF16 (cvt.rn);
- the fused epilogue then replays tensorrt_llm/_torch/modules/swiglu.py silu_and_mul_kernel
  (alpha 1, beta 0, no limit) instruction for instruction on the BF16-rounded gate/up values.

The CTA's B tile is two TMA boxes: gate rows [j*BN/2, (j+1)*BN/2) and the matching up rows
I + [j*BN/2, ...). The weight is used in place (no copy or interleave). With the tiled MMA's
identity permutation a thread that owns gate column c also owns up column c + BN/2, so the
epilogue is register-local.

One compiled kernel per (mode, N, K, tile config); M (rows) is dynamic.
"""

from typing import Dict, Optional, Tuple

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as cutlass_utils
    import cutlass.utils.hopper_helpers as sm90_utils_basic
    from cutlass import Float32, Int32, Int64, Uint32, const_expr
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.nvgpu import cpasync, warp, warpgroup
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils import LayoutEnum

    _CUTE_OK = True
except Exception:  # pragma: no cover - no CuTe DSL in this environment
    _CUTE_OK = False

# PTX replay of silu_and_mul_kernel's per-element math. Inputs: $1..$4 = fp32 accumulators
# (gate0, gate1, up0, up1); output $0 = packed bf16x2 (gate col c in the low half).
# Validated against the Triton kernel's PTX and exhaustively over all BF16 gate values.
_SWIGLU_X2_PTX = """{
.reg .b16 hg0, hg1, hu0, hu1, ho0, ho1;
.reg .f32 a0, a1, b0, b1, t0, t1, s0, s1, r0, r1;
cvt.rn.bf16.f32 hg0, $1;
cvt.rn.bf16.f32 hg1, $2;
cvt.rn.bf16.f32 hu0, $3;
cvt.rn.bf16.f32 hu1, $4;
cvt.f32.bf16 a0, hg0;
cvt.f32.bf16 a1, hg1;
cvt.f32.bf16 b0, hu0;
cvt.f32.bf16 b1, hu1;
mul.f32 t0, a0, 0fBFB8AA3B;
mul.f32 t1, a1, 0fBFB8AA3B;
ex2.approx.f32 t0, t0;
ex2.approx.f32 t1, t1;
add.f32 t0, t0, 0f3F800000;
add.f32 t1, t1, 0f3F800000;
div.full.f32 s0, 0f3F800000, t0;
div.full.f32 s1, 0f3F800000, t1;
mul.f32 r0, a0, s0;
mul.f32 r1, a1, s1;
add.f32 b0, b0, 0f00000000;
add.f32 b1, b1, 0f00000000;
mul.f32 r0, r0, b0;
mul.f32 r1, r1, b1;
cvt.rn.bf16.f32 ho0, r0;
cvt.rn.bf16.f32 ho1, r1;
mov.b32 $0, {ho0, ho1};
}"""

_PACK_X2_PTX = "cvt.rn.bf16x2.f32 $0, $2, $1;"

if _CUTE_OK:

    @dsl_user_op
    def _swiglu_bf16x2(g0, g1, u0, u1, *, loc=None, ip=None):
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [
                    Float32(g0).ir_value(loc=loc, ip=ip),
                    Float32(g1).ir_value(loc=loc, ip=ip),
                    Float32(u0).ir_value(loc=loc, ip=ip),
                    Float32(u1).ir_value(loc=loc, ip=ip),
                ],
                _SWIGLU_X2_PTX,
                "=r,f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            ))

    @dsl_user_op
    def _pack_bf16x2(v0, v1, *, loc=None, ip=None):
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [
                    Float32(v0).ir_value(loc=loc, ip=ip),
                    Float32(v1).ir_value(loc=loc, ip=ip),
                ],
                _PACK_X2_PTX,
                "=r,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            ))

    @dsl_user_op
    def _st_global_u32(addr, value, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "st.global.u32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _ptr_i64(tensor, *, loc=None, ip=None):
        return Int64(llvm.ptrtoint(T.i64(), tensor.iterator.llvm_ptr, loc=loc, ip=ip))

    def _assume_tma_aligned(tensor):
        row_stride = tensor.stride[0]
        if not isinstance(row_stride, int):
            row_stride = cute.assume(row_stride, divby=8)
        return cute.make_tensor(tensor.iterator,
                                cute.make_layout(tensor.shape, stride=(row_stride, 1)))

    class GateUpSm120Kernel:
        """C[M, N_out] = epilogue(x[M, K] @ W[N, K]^T); fused: N_out = N / 2 (SwiGLU)."""

        def __init__(self, n: int, k: int, fused: bool, tile_m: int = 128, tile_n: int = 256,
                     tile_k: int = 64, stages: int = 2, atom_mn: Tuple[int, int] = (2, 4),
                     group_m: int = 8, use_setmaxnreg: bool = False, persistent: bool = False,
                     prefetch_next: bool = False, max_ctas: int = 188,
                     inline_producer: bool = False):
            self.inline_producer = bool(inline_producer)
            self.persistent = bool(persistent)
            self.prefetch_next = bool(prefetch_next)
            self.max_ctas = int(max_ctas)
            self.n, self.k, self.fused = int(n), int(k), bool(fused)
            self.tile_m, self.tile_n, self.tile_k = tile_m, tile_n, tile_k
            self.stages = stages
            self.atom_mn = tuple(atom_mn)
            self.group_m = group_m
            self.half_n = tile_n // 2
            self.num_mma_warps = self.atom_mn[0] * self.atom_mn[1]
            self.producer_warp = self.num_mma_warps
            self.num_threads = 32 * (self.num_mma_warps + (0 if self.inline_producer else 1))
            self.k_tiles = self.k // tile_k
            if self.k % tile_k:
                raise ValueError("K must be a multiple of tile_k")
            if fused:
                self.n_out = self.n // 2
                if self.n_out % self.half_n:
                    raise ValueError("I must be a multiple of tile_n/2")
                self.n_tiles = self.n_out // self.half_n
                self.up_offset_tiles = self.n_out // self.half_n
            else:
                self.n_out = self.n
                if self.n % tile_n:
                    raise ValueError("N must be a multiple of tile_n")
                self.n_tiles = self.n // tile_n
                self.up_offset_tiles = 0
            # gate/up pairing needs both halves in the same thread: N rep period divides BN/2
            if self.half_n % (8 * self.atom_mn[1]):
                raise ValueError("tile_n/2 must be a multiple of 8 * warps_n")
            self.buffer_align_bytes = 1024
            self.use_setmaxnreg = bool(use_setmaxnreg)
            self.mma_regs = 232
            self.load_regs = 40

        def _tile_mn(self, tile, m_tiles):
            # grouped raster: GROUP_M m-tiles share each sweep over the n-tiles
            group_m = Int32(self.group_m)
            per_group = group_m * Int32(self.n_tiles)
            group_id = tile // per_group
            first_m = group_id * group_m
            group_size = cutlass.min(m_tiles - first_m, group_m)
            local = tile - group_id * per_group
            return first_m + local % group_size, local // group_size

        def _tiled_mma(self):
            return cute.make_tiled_mma(
                warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
                (self.atom_mn[0], self.atom_mn[1], 1),
                permutation_mnk=(self.tile_m, self.tile_n, 16),
            )

        def _smem_layouts(self):
            atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16,
                                                      self.tile_k),
                cutlass.BFloat16,
            )
            sA = cute.tile_to_shape(atom, (self.tile_m, self.tile_k, self.stages), order=(0, 1, 2))
            sB = cute.tile_to_shape(atom, (self.tile_n, self.tile_k, self.stages), order=(0, 1, 2))
            sBh = cute.tile_to_shape(atom, (self.half_n, self.tile_k, 2 * self.stages),
                                     order=(0, 1, 2))
            return sA, sB, sBh

        def _storage(self, sA, sB):
            class SharedStorage:
                pass

            SharedStorage.__annotations__ = {
                "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.stages * 2],
                "sA": cute.struct.Align[
                    cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sA)], self.buffer_align_bytes],
                "sB": cute.struct.Align[
                    cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sB)], self.buffer_align_bytes],
            }
            return cute.struct(SharedStorage)

        @cute.jit
        def __call__(self, x: cute.Tensor, w: cute.Tensor, out: cute.Tensor, m_rows: Int32,
                     stream: cuda.CUstream):
            sA_layout, sB_layout, sBh_layout = self._smem_layouts()
            tiled_mma = self._tiled_mma()
            storage_cls = self._storage(sA_layout, sB_layout)
            tma_atom_A, tma_A = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), _assume_tma_aligned(x),
                cute.slice_(sA_layout, (None, None, 0)), (self.tile_m, self.tile_k),
                num_multicast=1)
            tma_atom_B, tma_B = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), w,
                cute.slice_(sBh_layout, (None, None, 0)), (self.half_n, self.tile_k),
                num_multicast=1)
            m_tiles = (m_rows + Int32(self.tile_m - 1)) // Int32(self.tile_m)
            grid_x = m_tiles * Int32(self.n_tiles)
            if const_expr(self.inline_producer):
                if const_expr(self.persistent):
                    grid_x = cutlass.min(grid_x, Int32(self.max_ctas))
                self.kernel_inline(tma_A, tma_B, out, tma_atom_A, tma_atom_B, sA_layout,
                                   sB_layout, sBh_layout, tiled_mma, storage_cls, m_rows,
                                   m_tiles).launch(
                    grid=(grid_x, 1, 1),
                    block=[self.num_threads, 1, 1],
                    cluster=[1, 1, 1],
                    stream=stream,
                )
            else:
                if const_expr(self.persistent):
                    grid_x = cutlass.min(grid_x, Int32(self.max_ctas))
                self.kernel(tma_A, tma_B, out, tma_atom_A, tma_atom_B, sA_layout, sB_layout,
                            sBh_layout, tiled_mma, storage_cls, m_rows, m_tiles).launch(
                    grid=(grid_x, 1, 1),
                    block=[self.num_threads, 1, 1],
                    cluster=[1, 1, 1],
                    stream=stream,
                )

        @cute.jit
        def _epilogue(self, acc: cute.Tensor, coords: cute.Tensor, base: Int64, ld_out: Int64,
                      m_tile: Int32, n_tile: Int32, m_rows: Int32):
            row0 = m_tile * Int32(self.tile_m)
            n_vals = cute.size(acc)
            if const_expr(self.fused):
                col0 = n_tile * Int32(self.half_n)
                for i in cutlass.range_constexpr(0, n_vals // 2, 2):
                    crd = coords[i]
                    row = row0 + crd[0]
                    if row < m_rows:
                        packed = _swiglu_bf16x2(acc[i], acc[i + 1], acc[i + n_vals // 2],
                                                acc[i + n_vals // 2 + 1])
                        addr = base + (Int64(row) * ld_out + Int64(col0 + crd[1])) * Int64(2)
                        _st_global_u32(addr, packed)
            else:
                col0 = n_tile * Int32(self.tile_n)
                for i in cutlass.range_constexpr(0, n_vals, 2):
                    crd = coords[i]
                    row = row0 + crd[0]
                    if row < m_rows:
                        packed = _pack_bf16x2(acc[i], acc[i + 1])
                        addr = base + (Int64(row) * ld_out + Int64(col0 + crd[1])) * Int64(2)
                        _st_global_u32(addr, packed)

        @cute.kernel
        def kernel_inline(self, mA: cute.Tensor, mB: cute.Tensor, out: cute.Tensor,
                          tma_atom_A: cute.CopyAtom, tma_atom_B: cute.CopyAtom,
                          sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
                          sBh_layout: cute.ComposedLayout, tiled_mma: cute.TiledMma,
                          storage_cls: cutlass.Constexpr, m_rows: Int32, m_tiles: Int32):
            """8 MMA warps, no producer warp (256 threads keep the 255-register cap); warp 0
            issues the TMA loads, refilling the slot every warp released one k-tile earlier."""
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            nctas, _, _ = cute.arch.grid_dim()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            total_tiles = m_tiles * Int32(self.n_tiles)
            my_tiles = (total_tiles - Int32(bid) + Int32(nctas) - 1) // Int32(nctas)
            total_loads = my_tiles * Int32(self.k_tiles)

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_A)
                cpasync.prefetch_descriptor(tma_atom_B)

            smem = cutlass_utils.SmemAllocator()
            storage = smem.allocate(storage_cls)
            sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
            sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
            sBh = storage.sB.get_tensor(sBh_layout.outer, swizzle=sBh_layout.inner)

            tx_bytes = (self.tile_m + self.tile_n) * self.tile_k * 2
            load_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=self.stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                         self.num_mma_warps),
                tx_count=tx_bytes,
                barrier_storage=storage.mbar_ptr.data_ptr(),
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            cute.arch.sync_threads()

            gA = cute.local_tile(mA, (self.tile_m, self.tile_k), (None, None))
            gB = cute.local_tile(mB, (self.half_n, self.tile_k), (None, None))
            cta_layout = cute.make_layout(1)
            tAsA, tAgA = cpasync.tma_partition(tma_atom_A, 0, cta_layout,
                                               cute.group_modes(sA, 0, 2),
                                               cute.group_modes(gA, 0, 2))
            tBsB, tBgB = cpasync.tma_partition(tma_atom_B, 0, cta_layout,
                                               cute.group_modes(sBh, 0, 2),
                                               cute.group_modes(gB, 0, 2))
            up_off = Int32(self.up_offset_tiles)

            # prologue: linear load index j = tile_iter * k_tiles + kt
            if warp_idx == 0:
                for jj in cutlass.range_constexpr(self.stages):
                    j = Int32(jj)
                    if j < total_loads:
                        t_it = j // Int32(self.k_tiles)
                        lkt = j - t_it * Int32(self.k_tiles)
                        lm, ln = self._tile_mn(Int32(bid) + t_it * Int32(nctas), m_tiles)
                        if const_expr(self.fused):
                            g_nb = ln
                            u_nb = ln + up_off
                        else:
                            g_nb = ln * Int32(2)
                            u_nb = ln * Int32(2) + Int32(1)
                        ps = pipeline.PipelineState(self.stages, j, j, Int32(1))
                        load_pipeline.producer_acquire(ps)
                        bar = load_pipeline.producer_get_barrier(ps)
                        cute.copy(tma_atom_A, tAgA[(None, lm, lkt)], tAsA[(None, jj)],
                                  tma_bar_ptr=bar)
                        cute.copy(tma_atom_B, tBgB[(None, g_nb, lkt)], tBsB[(None, 2 * jj)],
                                  tma_bar_ptr=bar)
                        cute.copy(tma_atom_B, tBgB[(None, u_nb, lkt)],
                                  tBsB[(None, 2 * jj + 1)], tma_bar_ptr=bar)
                        load_pipeline.producer_commit(ps)

            state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.tile_m, self.tile_n)),
                                        Float32)
            cpA = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                                    cutlass.BFloat16), tiled_mma).get_slice(tidx)
            cpB = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                                    cutlass.BFloat16), tiled_mma).get_slice(tidx)
            tSsA = cpA.partition_S(sA)
            tSsB = cpB.partition_S(sB)
            tSrA = cpA.retile(tCrA)
            tSrB = cpB.retile(tCrB)
            num_kb = cute.size(tCrA, mode=[2])
            coords = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
            out_base = _ptr_i64(out)
            ld_out = Int64(out.stride[0])

            for t_iter in cutlass.range(my_tiles, unroll=1):
                m_tile, n_tile = self._tile_mn(Int32(bid) + t_iter * Int32(nctas), m_tiles)
                acc.fill(0.0)
                for kt in cutlass.range(self.k_tiles, unroll=1):
                    load_pipeline.consumer_wait(state)
                    if warp_idx == 0:
                        c = t_iter * Int32(self.k_tiles) + kt
                        j = c - 1 + self.stages
                        if c >= 1 and j < total_loads:
                            t_it = j // Int32(self.k_tiles)
                            lkt = j - t_it * Int32(self.k_tiles)
                            lm, ln = self._tile_mn(Int32(bid) + t_it * Int32(nctas), m_tiles)
                            if const_expr(self.fused):
                                g_nb = ln
                                u_nb = ln + up_off
                            else:
                                g_nb = ln * Int32(2)
                                u_nb = ln * Int32(2) + Int32(1)
                            slot = j % self.stages
                            ps = pipeline.PipelineState(self.stages, j, slot,
                                                        ((j // self.stages) & 1) ^ 1)
                            load_pipeline.producer_acquire(ps)
                            bar = load_pipeline.producer_get_barrier(ps)
                            cute.copy(tma_atom_A, tAgA[(None, lm, lkt)], tAsA[(None, slot)],
                                      tma_bar_ptr=bar)
                            cute.copy(tma_atom_B, tBgB[(None, g_nb, lkt)],
                                      tBsB[(None, slot * 2)], tma_bar_ptr=bar)
                            cute.copy(tma_atom_B, tBgB[(None, u_nb, lkt)],
                                      tBsB[(None, slot * 2 + 1)], tma_bar_ptr=bar)
                            load_pipeline.producer_commit(ps)
                    sAk = tSsA[None, None, None, state.index]
                    sBk = tSsB[None, None, None, state.index]
                    cute.copy(cpA, sAk[None, None, 0], tSrA[None, None, 0])
                    cute.copy(cpB, sBk[None, None, 0], tSrB[None, None, 0])
                    for kb in cutlass.range_constexpr(num_kb):
                        if const_expr(kb < num_kb - 1):
                            cute.copy(cpA, sAk[None, None, kb + 1], tSrA[None, None, kb + 1])
                            cute.copy(cpB, sBk[None, None, kb + 1], tSrB[None, None, kb + 1])
                        if const_expr(kb == num_kb - 1):
                            load_pipeline.consumer_release(state)
                        cute.gemm(thr_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                    state.advance()

                self._epilogue(acc, coords, out_base, ld_out, m_tile, n_tile, m_rows)

        @cute.kernel
        def kernel(self, mA: cute.Tensor, mB: cute.Tensor, out: cute.Tensor,
                   tma_atom_A: cute.CopyAtom, tma_atom_B: cute.CopyAtom,
                   sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
                   sBh_layout: cute.ComposedLayout, tiled_mma: cute.TiledMma,
                   storage_cls: cutlass.Constexpr, m_rows: Int32, m_tiles: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            nctas, _, _ = cute.arch.grid_dim()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            total_tiles = m_tiles * Int32(self.n_tiles)

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_A)
                cpasync.prefetch_descriptor(tma_atom_B)

            smem = cutlass_utils.SmemAllocator()
            storage = smem.allocate(storage_cls)
            sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
            sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
            sBh = storage.sB.get_tensor(sBh_layout.outer, swizzle=sBh_layout.inner)

            tx_bytes = (self.tile_m + self.tile_n) * self.tile_k * 2
            load_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=self.stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                         self.num_mma_warps),
                tx_count=tx_bytes,
                barrier_storage=storage.mbar_ptr.data_ptr(),
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            cute.arch.sync_threads()

            if warp_idx < self.num_mma_warps:
                if const_expr(self.use_setmaxnreg):
                    cute.arch.setmaxregister_increase(self.mma_regs)
                state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer,
                                                     self.stages)
                thr_mma = tiled_mma.get_slice(tidx)
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
                tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
                acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.tile_m, self.tile_n)),
                                            Float32)
                cpA = cute.make_tiled_copy_A(
                    cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                                        cutlass.BFloat16), tiled_mma).get_slice(tidx)
                cpB = cute.make_tiled_copy_B(
                    cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                                        cutlass.BFloat16), tiled_mma).get_slice(tidx)
                tSsA = cpA.partition_S(sA)
                tSsB = cpB.partition_S(sB)
                tSrA = cpA.retile(tCrA)
                tSrB = cpB.retile(tCrB)
                num_kb = cute.size(tCrA, mode=[2])
                coords = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
                base = _ptr_i64(out)
                ld_out = Int64(out.stride[0])
                n_vals = cute.size(acc)

                for tile in cutlass.range(Int32(bid), total_tiles, Int32(nctas), unroll=1):
                    m_tile, n_tile = self._tile_mn(tile, m_tiles)
                    acc.fill(0.0)
                    load_pipeline.consumer_wait(state)
                    if const_expr(self.prefetch_next):
                        sA0 = tSsA[None, None, None, state.index]
                        sB0 = tSsB[None, None, None, state.index]
                        cute.copy(cpA, sA0[None, None, 0], tSrA[None, None, 0])
                        cute.copy(cpB, sB0[None, None, 0], tSrB[None, None, 0])
                    for kt in cutlass.range(self.k_tiles, unroll=1):
                        sAk = tSsA[None, None, None, state.index]
                        sBk = tSsB[None, None, None, state.index]
                        if const_expr(not self.prefetch_next):
                            if kt > 0:
                                load_pipeline.consumer_wait(state)
                            cute.copy(cpA, sAk[None, None, 0], tSrA[None, None, 0])
                            cute.copy(cpB, sBk[None, None, 0], tSrB[None, None, 0])
                        for kb in cutlass.range_constexpr(num_kb):
                            if const_expr(kb < num_kb - 1):
                                cute.copy(cpA, sAk[None, None, kb + 1], tSrA[None, None, kb + 1])
                                cute.copy(cpB, sBk[None, None, kb + 1], tSrB[None, None, kb + 1])
                            if const_expr(kb == num_kb - 1):
                                load_pipeline.consumer_release(state)
                                state.advance()
                                if const_expr(self.prefetch_next):
                                    if kt < self.k_tiles - 1:
                                        load_pipeline.consumer_wait(state)
                                        sAn = tSsA[None, None, None, state.index]
                                        sBn = tSsB[None, None, None, state.index]
                                        cute.copy(cpA, sAn[None, None, 0], tSrA[None, None, 0])
                                        cute.copy(cpB, sBn[None, None, 0], tSrB[None, None, 0])
                            cute.gemm(thr_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb],
                                      acc)

                    # epilogue: register-local, packed bf16x2 global stores
                    row0 = m_tile * Int32(self.tile_m)
                    if const_expr(self.fused):
                        col0 = n_tile * Int32(self.half_n)
                        for i in cutlass.range_constexpr(0, n_vals // 2, 2):
                            crd = coords[i]
                            row = row0 + crd[0]
                            if row < m_rows:
                                packed = _swiglu_bf16x2(acc[i], acc[i + 1], acc[i + n_vals // 2],
                                                        acc[i + n_vals // 2 + 1])
                                addr = base + (Int64(row) * ld_out + Int64(col0 + crd[1])) * Int64(2)
                                _st_global_u32(addr, packed)
                    else:
                        col0 = n_tile * Int32(self.tile_n)
                        for i in cutlass.range_constexpr(0, n_vals, 2):
                            crd = coords[i]
                            row = row0 + crd[0]
                            if row < m_rows:
                                packed = _pack_bf16x2(acc[i], acc[i + 1])
                                addr = base + (Int64(row) * ld_out + Int64(col0 + crd[1])) * Int64(2)
                                _st_global_u32(addr, packed)

            elif warp_idx == self.producer_warp:
                if const_expr(self.use_setmaxnreg):
                    cute.arch.setmaxregister_decrease(self.load_regs)
                gA = cute.local_tile(mA, (self.tile_m, self.tile_k), (None, None))
                gB = cute.local_tile(mB, (self.half_n, self.tile_k), (None, None))
                cta_layout = cute.make_layout(1)
                tAsA, tAgA = cpasync.tma_partition(tma_atom_A, 0, cta_layout,
                                                   cute.group_modes(sA, 0, 2),
                                                   cute.group_modes(gA, 0, 2))
                tBsB, tBgB = cpasync.tma_partition(tma_atom_B, 0, cta_layout,
                                                   cute.group_modes(sBh, 0, 2),
                                                   cute.group_modes(gB, 0, 2))
                pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer,
                                                      self.stages)
                for tile in cutlass.range(Int32(bid), total_tiles, Int32(nctas), unroll=1):
                    m_tile, n_tile = self._tile_mn(tile, m_tiles)
                    if const_expr(self.fused):
                        nb_gate = n_tile
                        nb_up = n_tile + Int32(self.up_offset_tiles)
                    else:
                        nb_gate = n_tile * Int32(2)
                        nb_up = n_tile * Int32(2) + Int32(1)
                    for kt in cutlass.range(self.k_tiles, unroll=1):
                        load_pipeline.producer_acquire(pstate)
                        bar = load_pipeline.producer_get_barrier(pstate)
                        cute.copy(tma_atom_A, tAgA[(None, m_tile, kt)],
                                  tAsA[(None, pstate.index)], tma_bar_ptr=bar)
                        cute.copy(tma_atom_B, tBgB[(None, nb_gate, kt)],
                                  tBsB[(None, pstate.index * 2)], tma_bar_ptr=bar)
                        cute.copy(tma_atom_B, tBgB[(None, nb_up, kt)],
                                  tBsB[(None, pstate.index * 2 + 1)], tma_bar_ptr=bar)
                        load_pipeline.producer_commit(pstate)
                        pstate.advance()
                load_pipeline.producer_tail(pstate)


_COMPILED: Dict[tuple, object] = {}
COMPILE_COUNT = 0
# 8 mma.sync warps on a 256x128x64 tile (warp tile 64x64), 2 TMA stages, warp 0 issues the
# loads (no producer warp, so the 256-thread CTA keeps the 255-register cap), persistent
# over min(tiles, SMs) CTAs with a 16-m-tile grouped raster.
DEFAULT_CFG = dict(tile_m=256, tile_n=128, tile_k=64, stages=2, atom_mn=(4, 2), group_m=16,
                   inline_producer=True, persistent=True)
# Below this many rows cuBLAS may pick a different (e.g. split-K) kernel for F.linear, so the
# fused path is only bitwise-guaranteed where nvjet's single in-order K chain runs.
MIN_ROWS = 4096
FUSED_CALLS = 0
_SM_COUNT: Optional[int] = None


def gate_up_sm120_available() -> bool:
    if not _CUTE_OK or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (12, 0)


def gate_up_swiglu_sm120_shape_ok(k: int, n: int) -> bool:
    """K and the gate/up half width tile exactly (no K or N tails in the kernel)."""
    cfg = DEFAULT_CFG
    return (k % cfg["tile_k"] == 0 and n % 2 == 0 and (n // 2) % (cfg["tile_n"] // 2) == 0)


def _to_cute(t: torch.Tensor, dynamic: bool):
    ct = from_dlpack(t.detach(), assumed_align=16)
    if dynamic:
        ct = ct.mark_layout_dynamic(leading_dim=1)
    return ct


def _cfg_key(cfg: dict) -> tuple:
    return tuple(sorted((a, tuple(b) if isinstance(b, (list, tuple)) else b)
                        for a, b in cfg.items()))


def gate_up_sm120(x: torch.Tensor, weight: torch.Tensor, fused: bool = True,
                  out: Optional[torch.Tensor] = None, cfg: Optional[dict] = None) -> torch.Tensor:
    """x [M, K] bf16 contiguous rows, weight [N, K] bf16 contiguous.

    fused: returns silu(x @ Wg^T) * (x @ Wu^T) as [M, N/2] (gate rows first), bitwise equal to
    swiglu(F.linear(x, weight)). Not fused: returns F.linear(x, weight) bitwise.
    """
    global COMPILE_COUNT, _SM_COUNT
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    if cfg.get("persistent") and "max_ctas" not in cfg:
        if _SM_COUNT is None:
            _SM_COUNT = torch.cuda.get_device_properties(x.device).multi_processor_count
        cfg["max_ctas"] = _SM_COUNT
    m, k = x.shape
    n = weight.shape[0]
    n_out = n // 2 if fused else n
    if out is None:
        out = torch.empty((m, n_out), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    key = (bool(fused), n, k, _cfg_key(cfg))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    xc, wc, oc = _to_cute(x, True), _to_cute(weight, False), _to_cute(out, True)
    fn = _COMPILED.get(key)
    if fn is None:
        kern = GateUpSm120Kernel(n, k, fused, **cfg)
        fn = cute.compile(kern, xc, wc, oc, Int32(max(m, 1)), stream)
        _COMPILED[key] = fn
        COMPILE_COUNT += 1
    fn(xc, wc, oc, Int32(m), stream)
    return out


def _log_fused_call(m: int) -> None:
    global FUSED_CALLS
    FUSED_CALLS += 1
    if FUSED_CALLS == 1 or FUSED_CALLS % 1000 == 0:
        from tensorrt_llm.logger import logger
        rank = torch.distributed.get_rank() if (torch.distributed.is_available()
                                                and torch.distributed.is_initialized()) else 0
        logger.info(f"[rank {rank}] gate_up_swiglu_sm120: fused calls {FUSED_CALLS} "
                    f"(last M={m}, compiles {COMPILE_COUNT})")


@torch.library.custom_op("trtllm::gate_up_swiglu_bf16_sm120", mutates_args=())
def gate_up_swiglu_bf16_sm120(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """swiglu(F.linear(x, weight)) in one SM120 kernel; bitwise to the unfused chain."""
    out = gate_up_sm120(x, weight, fused=True)
    _log_fused_call(x.shape[0])
    return out


@gate_up_swiglu_bf16_sm120.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0] // 2))
