# Arm F exactness diagnosis.
# part1: fp32 q/k sum of squares: main-style 1-warp tl.sum (same code as the
#        T1 kernel) vs _ordered_sum_sq MODE 0..4 at 1 and 8 warps.
# part2: per F variant (qk_mode, norm_mode) over seeds and N: bitwise counts
#        of o, the whole pool and y vs main T1 + rms_norm_gated_token_major,
#        and fp32 rstd vs the multirow norm kernel run with SAVE_RSTD.
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from tensorrt_llm._torch.modules.fla import gdn_decode_t1 as t1mod  # noqa
from tensorrt_llm._torch.modules.mamba import layernorm_gated as lng  # noqa

EPS = 1e-6


def log(*a):
    print(*a, flush=True)


@triton.jit
def ref_sumsq(k, out, stride, H: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    i_n = pid // H
    i_h = pid % H
    o_k = tl.arange(0, K)
    b_k = tl.load(k + i_n * stride + i_h * K + o_k).to(tl.float32)
    tl.store(out + pid, tl.sum(b_k * b_k))


@triton.jit
def emu_sumsq(k, out, stride, H: tl.constexpr, K: tl.constexpr,
              MODE: tl.constexpr):
    pid = tl.program_id(0)
    i_n = pid // H
    i_h = pid % H
    o_k = tl.arange(0, K)
    b_k = tl.load(k + i_n * stride + i_h * K + o_k).to(tl.float32)
    tl.store(out + pid, t1mod._ordered_sum_sq(b_k, K // 4, 4, MODE))


def part1():
    n = 32
    gen = torch.Generator(device="cuda")
    gen.manual_seed(77)
    for trial in range(3):
        x = (torch.randn(n, common.QKVZ_DIM, device="cuda", generator=gen) *
             (0.7 + trial)).to(torch.bfloat16)
        kk = x[:, common.KEY_DIM:2 * common.KEY_DIM]
        o_r = torch.empty(n * common.H, device="cuda")
        ref_sumsq[(n * common.H, )](kk, o_r, kk.stride(0), common.H,
                                    common.K, num_warps=1)
        for mode in range(5):
            for w in (1, 8):
                o_e = torch.empty_like(o_r)
                emu_sumsq[(n * common.H, )](kk, o_e, kk.stride(0), common.H,
                                            common.K, mode, num_warps=w)
                torch.cuda.synchronize()
                d = (o_r.view(torch.int32) != o_e.view(torch.int32)).sum()
                log(f"part1 trial {trial} mode={mode} warps={w}: sumsq "
                    f"diffs {int(d)}/{o_r.numel()}")


def main_ref(lay):
    common.reset([lay])
    common.make_call(None, None)(lay)
    x = lay["out"].reshape(-1, common.V)
    y = lng.rms_norm_gated_token_major(x, lay["z"], lay["norm_w"], EPS)
    # Same kernel with SAVE_RSTD for an fp32 rstd reference.
    M = x.shape[0]
    y2 = torch.empty_like(x)
    rstd = torch.empty(M, dtype=torch.float32, device="cuda")
    lng._rms_norm_gated_fwd_multirow_kernel[(triton.cdiv(
        M, lng._MULTIROW_ROWS), )](
            x, y2, lay["norm_w"], lay["z"], rstd, None, x.stride(0),
            y2.stride(0), lay["z"].stride(0), M, EPS, N=common.V,
            ROWS=lng._MULTIROW_ROWS, HEADS_PER_TOK=common.HV,
            GATE_SIGMOID=False, WEIGHT_IS_DELTA=False, LAUNCH_WITH_PDL=False,
            num_warps=lng._MULTIROW_NUM_WARPS)
    torch.cuda.synchronize()
    assert torch.equal(y.view(torch.int16), y2.view(torch.int16))
    return lay["out"].clone(), lay["flat"].clone(), y.clone(), rstd


def run_f(lay, qk_mode, norm_mode):
    common.reset([lay])
    n = lay["n"]
    o_dbg = torch.zeros(n * common.HV, common.V, dtype=torch.bfloat16,
                        device="cuda")
    r_dbg = torch.zeros(n * common.HV, dtype=torch.float32, device="cuda")
    y = t1mod.fused_sigmoid_gating_delta_rule_update_t1_norm(
        A_log=lay["A_log"], a=lay["a"], dt_bias=lay["dt_bias"],
        softplus_beta=1.0, softplus_threshold=20.0, q=lay["q"], k=lay["k"],
        v=lay["v"], b=lay["b"], z=lay["z"], norm_weight=lay["norm_w"],
        eps=EPS, initial_state_source=lay["pool"],
        initial_state_indices=lay["idx"], scale=common.K**-0.5,
        use_qk_l2norm_in_kernel=True, o_debug=o_dbg, qk_per=4, norm_per=4,
        qk_mode=qk_mode, norm_mode=norm_mode, rstd_debug=r_dbg)
    torch.cuda.synchronize()
    return o_dbg, lay["flat"].clone(), y.clone(), r_dbg


def nd(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.element_size() == 4:
        return int((a.view(torch.int32) != b.view(torch.int32)).sum().item())
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


def part2():
    variants = [(1, 2), (2, 2), (3, 2), (4, 2), (2, 1), (2, 3), (2, 4),
                (2, 0)]
    tot = {v: [0, 0, 0, 0] for v in variants}
    for seed in range(3):
        for n in (8, 16, 32):
            gen = torch.Generator(device="cuda")
            gen.manual_seed(5000 + 100 * seed + n)
            lay = common.make_layer(n, gen, pad=True)
            o_r, s_r, y_r, r_r = main_ref(lay)
            for qm, nm in variants:
                o_f, s_f, y_f, r_f = run_f(lay, qm, nm)
                c = [nd(o_r, o_f), nd(s_r, s_f), nd(y_r, y_f), nd(r_r, r_f)]
                for i in range(4):
                    tot[(qm, nm)][i] += c[i]
                log(f"part2 seed {seed} N={n} qk_mode={qm} norm_mode={nm}: "
                    f"o {c[0]} state {c[1]} y {c[2]} rstd {c[3]}")
            del lay
            torch.cuda.empty_cache()
    for v, c in tot.items():
        log(f"TOTAL qk_mode={v[0]} norm_mode={v[1]}: o {c[0]} state {c[1]} "
            f"y {c[2]} rstd {c[3]}")


part1()
part2()
