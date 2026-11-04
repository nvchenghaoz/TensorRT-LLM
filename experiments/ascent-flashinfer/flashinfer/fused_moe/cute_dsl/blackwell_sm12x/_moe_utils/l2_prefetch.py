"""L2 prefetch of a contiguous weight range from a producer warp.

The SM12x decode MoE GEMMs (fc1_act_q1, fc2_finalize) stream every weight byte exactly
once, and with only 2-4 TMA stages per CTA the bytes in flight per SM are too few to fill
DRAM when only a few dozen CTAs have work (decode: 1-4 active experts per rank).  At each
new tile the weight-producer warp asks L2 for the whole tile; the TMA loads that follow are
unchanged and simply hit L2.  It is a cache hint: same loads, same math, same order.

``TRTLLM_SM12X_FI_L2_PREFETCH`` (read when a kernel is built; part of the compile key):
  0   off
  1   cp.async.bulk.prefetch.L2.global, one range per lane (TMA unit)
  2   prefetch.global.L2 per 128 B line, lanes strided (LSU path)
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

ENV = "TRTLLM_SM12X_FI_L2_PREFETCH"


def mode() -> int:
    return int(os.environ.get(ENV, "0"))


def enabled() -> int:
    """Compile-key value (0 = off)."""
    return mode()


@dsl_user_op
def bulk_prefetch_l2(addr: Int64, size: Int32, *, loc=None, ip=None):
    """cp.async.bulk.prefetch.L2.global [addr], size  (addr 16 B aligned, size % 16 == 0)."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def line_prefetch_l2(addr: Int64, *, loc=None, ip=None):
    """prefetch.global.L2 [addr]."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _addr_of(tensor: cute.Tensor, offset, *, loc=None, ip=None) -> Int64:
    elem_ptr = tensor.iterator + offset
    return Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))


@cute.jit
def prefetch_range_warp(g_u8: cute.Tensor, byte_off, nbytes, pf_mode: cutlass.Constexpr):
    """Whole warp prefetches bytes [byte_off, byte_off + nbytes) of the uint8 tensor g_u8.

    nbytes must be a multiple of 4096.
    """
    lane = cute.arch.lane_idx()
    if cutlass.const_expr(pf_mode == 1):
        chunk = Int32(nbytes) // Int32(32)
        off = Int64(byte_off) + Int64(lane) * Int64(chunk)
        bulk_prefetch_l2(_addr_of(g_u8, off), chunk)
    else:
        lines = Int32(nbytes) // Int32(4096)  # 32 lanes x 128 B per step
        base = _addr_of(g_u8, Int64(byte_off) + Int64(lane) * Int64(128))
        for i in cutlass.range(lines, unroll=8):
            line_prefetch_l2(base + Int64(i) * Int64(4096))
