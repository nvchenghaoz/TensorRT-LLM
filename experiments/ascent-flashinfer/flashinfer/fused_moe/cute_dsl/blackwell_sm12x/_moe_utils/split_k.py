"""Device-side dynamic split-K for the SM12x decode fc1_act_q1 swap-AB tile.

At decode a rank has only D = 1..4 experts with tokens, so fc1 has T = 18*D (8,128) tiles on
188 persistent CTAs, and each CTA streams its 655 KB of weights through a 2-stage TMA ring at
~28 GB/s (the per-CTA rate holds even with the weights already in L2).  The scheduler warp
counts T on device (no host sync, CUDA-graph safe) and splits every tile's K into
S = min(grid // T, S_max, k_cycles) slices, one CTA each:

  * S == 1: the original schedule, loads and math (bitwise = the unsplit kernel).
  * S > 1:  CTA u = bidx works on tile t = u // S, K-cycles [s*C/S, (s+1)*C/S) with
            s = u % S.  Every slice writes its FP32 (up, gate) accumulators to scratch slot u,
            fences, and bumps counter[t]; the slice that arrives last resets counter[t] to 0
            and sums the S slots in slice order 0..S-1 (a fixed order, so the result is
            run-to-run deterministic), then runs the unchanged SwiGLU + MXFP8 epilogue.

``TRTLLM_SM12X_FI_SPLITK`` = 0 / 1 (default 1) turns it off / on; ``TRTLLM_SM12X_FI_SPLITK_MAX``
(default 4) caps S and ``TRTLLM_SM12X_FI_SPLITK_MIN`` (default 3) is the smallest S used
(below it the tile stays whole).  All are read when the kernel is built and are part of its
compile key.  The choice depends only on the live tile count, so it follows the per-rank
shape (e.g. 18 fc1 tiles/expert at EP4 inter 2304, 9 at EP2xTP2 inter 1152).
"""

import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, dsl_user_op

ENV = "TRTLLM_SM12X_FI_SPLITK"
ENV_MAX = "TRTLLM_SM12X_FI_SPLITK_MAX"
#: FP32 accumulator values per thread a slot can hold (fc1 swap (128 x 8) tile: 2 x 4).
SLOT_VALUES_PER_THREAD = 16
SLOT_THREADS = 256


def enabled() -> bool:
    return os.environ.get(ENV, "1") == "1"


ENV_MIN = "TRTLLM_SM12X_FI_SPLITK_MIN"


def max_splits() -> int:
    return max(1, int(os.environ.get(ENV_MAX, "4")))


def min_splits() -> int:
    return max(2, int(os.environ.get(ENV_MIN, "3")))


def key():
    return (enabled(), max_splits(), min_splits())


_WS: dict = {}


def workspace(device, grid_x: int):
    """(scratch fp32 [grid * slot], counters int32 [grid]); counters start at 0 and every
    split tile's last CTA puts its counter back to 0.  Allocated once per device, before any
    graph capture (the decode leaf warms every batch size first)."""
    k = (torch.device(device), grid_x)
    ws = _WS.get(k)
    if ws is None:
        scratch = torch.zeros(
            grid_x * SLOT_VALUES_PER_THREAD * SLOT_THREADS, dtype=torch.float32, device=device
        )
        counters = torch.zeros(grid_x, dtype=torch.int32, device=device)
        ws = (scratch, counters)
        _WS[k] = ws
    return ws


@dsl_user_op
def _addr(tensor: cute.Tensor, *, loc=None, ip=None) -> Int64:
    return Int64(llvm.ptrtoint(T.i64(), tensor.iterator.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def st_f32(addr: Int64, v: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_cg_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atom_add_i32(addr: Int64, v: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
            "atom.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_i32(addr: Int64, v: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "st.global.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def membar_gl(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _bar_red_or_asm(bar_id, nthreads):
    return (
        "{ .reg .pred q, r; setp.ne.s32 q, $1, 0; "
        f"bar.red.or.pred r, {bar_id}, {nthreads}, q; "
        "selp.s32 $0, 1, 0, r; }"
    )


@dsl_user_op
def bar_red_or(flag: Int32, bar_id: int, nthreads: int, *, loc=None, ip=None) -> Int32:
    """Named barrier over nthreads threads; returns 1 on every thread if any flag != 0."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(flag).ir_value(loc=loc, ip=ip)],
            _bar_red_or_asm(bar_id, nthreads),
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


_UNROLL_WINDOWS = 4  # groups 0..127 are counted with independent loads (one round trip)


@cute.jit
def count_m_blocks_lane(offsets: cute.Tensor, num_groups, pm):
    """Per-lane part of the m-block count.  Issue it before the scheduler's own lookups so
    the loads overlap; reduce with warp_total."""
    lane = cute.arch.lane_idx()
    ng = Int32(num_groups)
    total = Int32(0)
    for w in cutlass.range_constexpr(_UNROLL_WINDOWS):
        g = lane + Int32(32 * w)
        lo = offsets[cute.min(g, ng)]
        hi = offsets[cute.min(g + Int32(1), ng)]
        total = total + (hi - lo + Int32(pm - 1)) // Int32(pm)
    base = Int32(32 * _UNROLL_WINDOWS)
    while base < ng:
        g = base + lane
        lo = offsets[cute.min(g, ng)]
        hi = offsets[cute.min(g + Int32(1), ng)]
        total = total + (hi - lo + Int32(pm - 1)) // Int32(pm)
        base = base + Int32(32)
    return total


@cute.jit
def warp_total(v):
    lane = cute.arch.lane_idx()
    incl = v
    for i in cutlass.range_constexpr(5):
        d = 1 << i
        up = cute.arch.shuffle_sync_up(incl, Int32(d), mask_and_clamp=0)
        incl = incl + up * (Int32(1) + ((lane - Int32(d)) >> Int32(31)))
    return cute.arch.shuffle_sync(incl, Int32(31))


@cute.jit
def count_tiles(offsets: cute.Tensor, num_groups, pm, num_n_blocks):
    """Whole warp: work tiles MoeTileScheduler hands out (sum of ceil(rows_g/pm) x n-blocks)."""
    return warp_total(count_m_blocks_lane(offsets, num_groups, pm)) * Int32(num_n_blocks)


def splits_of(valid):
    """S of a published work tile (0 for an unsplit tile)."""
    return (valid >> Int32(16)) & Int32(0xFF)


def cycle_range(valid, bidx, cycles):
    """[lo, hi) K-cycles of this CTA's slice; (0, cycles) for an unsplit tile."""
    S = cute.max(splits_of(valid), Int32(1))
    s = bidx % S
    return (s * cycles) // S, ((s + Int32(1)) * cycles) // S


def choose_splits(T_tiles, grid_x: int, s_max: int, cycles, s_min: int):
    """S = min(grid // T, s_max, cycles), or 1 when that is below s_min (measured: at S = 2
    the partial write + ordered reduction costs about what the split saves)."""
    S = Int32(grid_x) // cute.max(T_tiles, Int32(1))
    S = cute.min(S, Int32(s_max))
    S = cute.min(S, Int32(cycles))
    S = cute.max(S, Int32(1))
    below = Int32(0) - ((S - Int32(s_min)) >> Int32(31))  # 1 if S < s_min else 0
    return S + (Int32(1) - S) * below


@cute.jit
def reduce_partials(
    acc_u,
    acc_g,
    gScratch: cute.Tensor,
    gCnt: cute.Tensor,
    bidx,
    S,
    tidx,
    nthreads: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
):
    """All nthreads math threads of a split CTA.  Returns 1 on the CTA that must finish the
    tile (with acc_u / acc_g holding the slice-ordered sum), 0 elsewhere."""
    nv = cute.size(acc_u)
    slot_bytes = 2 * nv * nthreads * 4
    base = _addr(gScratch)
    my = base + Int64(bidx) * Int64(slot_bytes) + Int64(tidx) * Int64(4)
    for i in cutlass.range_constexpr(nv):
        st_f32(my + Int64(i * nthreads * 4), acc_u[i])
        st_f32(my + Int64((nv + i) * nthreads * 4), acc_g[i])
    membar_gl()
    cute.arch.barrier(barrier_id=bar_id, number_of_threads=nthreads)
    t = bidx // S
    flag = Int32(0)
    if tidx == Int32(0):
        caddr = _addr(gCnt) + Int64(t) * Int64(4)
        old = atom_add_i32(caddr, Int32(1))
        if old == S - Int32(1):
            st_i32(caddr, Int32(0))
            flag = Int32(1)
    last = bar_red_or(flag, bar_id, nthreads)
    if last != Int32(0):
        membar_gl()
        first = base + Int64(t * S) * Int64(slot_bytes) + Int64(tidx) * Int64(4)
        for i in cutlass.range_constexpr(nv):
            acc_u[i] = ld_cg_f32(first + Int64(i * nthreads * 4))
            acc_g[i] = ld_cg_f32(first + Int64((nv + i) * nthreads * 4))
        j = Int32(1)
        while j < S:
            p = first + Int64(j) * Int64(slot_bytes)
            for i in cutlass.range_constexpr(nv):
                acc_u[i] = acc_u[i] + ld_cg_f32(p + Int64(i * nthreads * 4))
                acc_g[i] = acc_g[i] + ld_cg_f32(p + Int64((nv + i) * nthreads * 4))
            j = j + Int32(1)
    return last
