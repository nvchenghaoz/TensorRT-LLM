"""Branch 115 (idea 182): folded FP8 scale reduce in qknorm_rope_fp8 (no one-CTA reduce launch).

Reference = landing 54's op in the same process (flux3_sm120_qknorm_rope_fp8.FOLD = False: mode 1
per-(row, warp) partials + one-program Triton reduce + mode 2), and main's pre-54 chain
(nvfp4_qkv_rope + quantize_qkv_e4m3 / qknorm_rope + quantize_qkv_e4m3) where op.py provides it.

modes:
  check : untimed, bitwise. (a) op level at M 4096/4176/4256 through nvfp4_qkv_rope_fp8_impl
          (GEMMs + fold): warmup 5 + 50 calls, each == landing-54 reference and == call 0 (qkv8,
          all 8 scales, gemm), inputs vs pristine before each reset, attention output bitwise.
          (b) pass level: branch 100's 12 random-magnitude / packed / planar / inf / NaN cases plus
          all-NaN q, k and v planes: fold == landing 54 == main (qknorm_rope + quantize_qkv_e4m3).
          (c) 178-layout partials (n_part = ceil(M/128)*24, entry [plane, m_tile*24 + head], max
          |BF16 mode-1 value| with init 0, NaN ignored, rows >= M masked) fed to mode 2 at
          M 4096/4176/4256 incl. inf/NaN/all-NaN: == landing 54; plus n_part 1 and 1024 layouts.
          (d) CUDA graph capture + 3 replays bitwise vs eager and reference.
  tune  : diagnostic timing of qknorm_rope_fp8 alone, one CUDA graph of 24 x M4176 + 4 x M4096,
          landing 54 vs fold at several (G1, G2), interleaved.
"""

import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b100_fp8direct"))
import op as b100  # noqa: E402

from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x  # noqa: E402
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)

H, HEADS = 3072, 24
SM_SCALE = 1.0 / (128 ** 0.5)
MS = (4096, 4176, 4256)
WARMUP, ITERS = 5, 50
log = b100.log
same = b100.same
u8 = b100.u8


def with_fold(flag, fn, *a, **k):
    old = qr8.FOLD
    qr8.FOLD = flag
    try:
        return fn(*a, **k)
    finally:
        qr8.FOLD = old


def eq_outputs(a, b):
    return {"qkv8": same(a[0], b[0]), "scales8": same(a[1], b[1]),
            **({"gemm": same(a[2], b[2])} if len(a) > 2 else {})}


# ------------------------------------------------------------------ (a) op level
def check_op(M, results):
    args = b100.mk(M, 3 * M + 1)
    ins = [t for t in args if isinstance(t, torch.Tensor)]
    pristine = [t.clone() for t in ins]
    ref = tuple(t.clone() for t in with_fold(False, b100.cand_chain, args))
    main = tuple(t.clone() for t in b100.main_chain(args))
    main_ok = all(b100.op_outputs_equal(main, ref, M).values())
    first = None
    n_ref = n_first = n_inputs = 0
    fails = {}
    for it in range(WARMUP + ITERS):
        n_inputs += all(torch.equal(t, p) for t, p in zip(ins, pristine))
        for t, p in zip(ins, pristine):
            t.copy_(p)
        junk = torch.empty(3 * M * H * 2, device="cuda", dtype=torch.uint8).fill_(0x7F)
        del junk
        o = with_fold(True, b100.cand_chain, args)
        if it < WARMUP:
            continue
        eq = eq_outputs(o, ref)
        for k, v in eq.items():
            if not v:
                fails[k] = fails.get(k, 0) + 1
        n_ref += all(eq.values())
        if first is None:
            first = tuple(t.clone() for t in o)
        n_first += all(same(a, b) for a, b in zip(o, first))
    inputs_final = all(torch.equal(t, p) for t, p in zip(ins, pristine))
    a_ref = fp8_sm12x.visual_gen_fp8_attention_sm120_prequant(ref[0], ref[1], 1, HEADS)
    a_c = fp8_sm12x.visual_gen_fp8_attention_sm120_prequant(first[0], first[1], 1, HEADS)
    attn_ok = same(a_ref, a_c)
    attn_main = b100.attention_equal(main, first, M)
    ok = (n_ref == ITERS and n_first == ITERS and n_inputs == WARMUP + ITERS and inputs_final
          and attn_ok and attn_main and main_ok and all(b100.op_outputs_equal(main, first, M).values()))
    results[f"op_M{M}"] = {
        "bitwise_vs_landing54(qkv8,scales[0:8],gemm)": f"{n_ref}/{ITERS}",
        "bitwise_vs_call0": f"{n_first}/{ITERS}",
        "inputs_unchanged_before_reset": f"{n_inputs}/{WARMUP + ITERS}",
        "landing54_equals_pre54_main_chain": main_ok,
        "attention_output_bitwise_vs_landing54": attn_ok,
        "attention_output_bitwise_vs_main_chain": attn_main,
        "field_failures": fails,
        "scales": [float(x) for x in first[1][:4].cpu()],
        "ok": ok,
    }
    log(f"op_M{M}", results[f"op_M{M}"])
    return ok


# ------------------------------------------------------------------ (b) pass level
def make_pass_case(case, g):
    M = MS[case % 3]
    mags = [10.0 ** float(torch.randint(-3, 4, (1,), generator=g, device="cuda")) for _ in range(3)]
    packed = case % 2 == 1
    src = torch.randn(M, 3 * H, device="cuda", generator=g, dtype=torch.float32)
    for i in range(3):
        src[:, i * H:(i + 1) * H] *= mags[i]
    src = src.bfloat16()
    special = None
    if case >= 8:
        r = int(torch.randint(0, M, (1,), generator=g, device="cuda"))
        special = ["inf_q", "nan_k", "inf_v", "nan_all", "allnan_q", "allnan_k", "allnan_v",
                   "allnan_qkv"][case - 8]
        if special == "inf_q":
            src[r, 5] = float("inf")
        elif special == "nan_k":
            src[r, H + 7] = float("nan")
        elif special == "inf_v":
            src[r, 2 * H + 9] = float("-inf")
        elif special == "nan_all":
            src[r, :] = float("nan")
        elif special == "allnan_q":
            src[:, :H] = float("nan")
        elif special == "allnan_k":
            src[:, H:2 * H] = float("nan")
        elif special == "allnan_v":
            src[:, 2 * H:] = float("nan")
        else:
            src[:] = float("nan")
    if packed:
        q, k, v = src[:, :H], src[:, H:2 * H], src[:, 2 * H:]
    else:
        planar = src.view(M, 3, H).permute(1, 0, 2).contiguous()
        q, k, v = planar[0], planar[1], planar[2]
    qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
    cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
    pos = torch.arange(M, device="cuda", dtype=torch.int64)
    return M, mags, packed, special, src, (q, k, v, qn, kn, cs, pos)


def check_pass(results):
    ok_all = True
    cases = []
    g = torch.Generator(device="cuda").manual_seed(7)
    for case in range(16):
        M, mags, packed, special, src, ins = make_pass_case(case, g)
        src_copy = src.clone()
        q, k, v, qn, kn, cs, pos = ins
        ref8, ref_sc = with_fold(False, qr8.qknorm_rope_fp8, *ins, SM_SCALE)
        main8, main_sc = b100.pass_ref(q.clone(), k.clone(), v.clone(), qn, kn, cs, pos)
        c8, c_sc = with_fold(True, qr8.qknorm_rope_fp8, *ins, SM_SCALE)
        rec = {"M": M, "packed": packed, "mags": mags, "special": special,
               "qkv8_vs_landing54": same(c8, ref8), "scales8_vs_landing54": same(c_sc, ref_sc),
               "qkv8_vs_main": same(c8, main8), "scales4_vs_main": same(c_sc[:4], main_sc),
               "mismatched_bytes": int((u8(c8) != u8(ref8)).sum()),
               "inputs_unchanged": same(src, src_copy),
               "scales": [float(x) for x in c_sc[:4].cpu()]}
        ok = rec["qkv8_vs_landing54"] and rec["scales8_vs_landing54"] and rec["inputs_unchanged"]
        if case < 12:  # branch 100's cases: landing 54 was shown == main on these
            ok &= rec["qkv8_vs_main"] and rec["scales4_vs_main"]
        rec["ok"] = ok
        ok_all &= ok
        cases.append(rec)
        log("pass case", rec)
    results["pass_cases"] = cases
    return ok_all


# ------------------------------------------------------------------ (c) 178-layout partials
def bf16_mode1_values(q, k, v, qn, kn, cs, pos):
    """The BF16 values mode 1 folds: norm+RoPE q/k exactly as main's qknorm_rope stores them, raw v."""
    M = q.shape[0]
    out = torch.empty(M, 3 * H, device="cuda", dtype=torch.bfloat16)
    b100.qr.qknorm_rope(q.clone(), k.clone(), v.clone(), out, qn, kn, cs, pos, planar=True)
    return out.view(3, M, H)


def partials_178(planes, M):
    """FP32 [3, ceil(M/128)*24]: entry [plane, m_tile*24 + head] = max |x| over a 128-row x
    128-col tile with init 0, NaN ignored, rows >= M masked."""
    tiles = math.ceil(M / 128)
    a = planes.float().abs()
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    pad = torch.zeros(3, tiles * 128, H, device="cuda")
    pad[:, :M] = a
    t = pad.view(3, tiles, 128, HEADS, 128).amax(dim=(2, 4))  # [3, tiles, 24]
    return t.reshape(3, tiles * HEADS).contiguous()


def regroup(parts, n):
    """Same per-plane max, different partial count: [3, n] by folding contiguous chunks."""
    m = parts.shape[1]
    pad = torch.zeros(3, math.ceil(m / n) * n, device="cuda")
    pad[:, :m] = parts
    return pad.view(3, n, -1).amax(dim=2).contiguous()


def check_178(results):
    ok_all = True
    recs = []
    g = torch.Generator(device="cuda").manual_seed(178)
    for case in range(16):
        M, mags, packed, special, src, ins = make_pass_case(case, g)
        q, k, v, qn, kn, cs, pos = ins
        ref8, ref_sc = with_fold(False, qr8.qknorm_rope_fp8, *ins, SM_SCALE)
        planes = bf16_mode1_values(q, k, v, qn, kn, cs, pos)
        parts = partials_178(planes, M)
        n_part = parts.shape[1]
        variants = {"178": parts, "n1": regroup(parts, 1), "n1024": regroup(parts, 1024)}
        rec = {"M": M, "special": special, "n_part_178": n_part,
               "expected_n_part": math.ceil(M / 128) * 24}
        ok = n_part == math.ceil(M / 128) * 24
        for name, p in variants.items():
            p_copy = p.clone()
            c8, c_sc = qr8.qknorm_rope_fp8(*ins, SM_SCALE, external_partials=p)
            e8, es = same(c8, ref8), same(c_sc, ref_sc)
            rec[name] = {"qkv8": e8, "scales8": es, "partials_unchanged": same(p, p_copy)}
            ok &= e8 and es and same(p, p_copy)
        # repeats of the 178 variant == call 0
        first = qr8.qknorm_rope_fp8(*ins, SM_SCALE, external_partials=parts)
        reps = sum(all(same(a, b) for a, b in zip(
            qr8.qknorm_rope_fp8(*ins, SM_SCALE, external_partials=parts), first))
            for _ in range(10))
        rec["repeats_eq_call0"] = f"{reps}/10"
        ok &= reps == 10
        rec["ok"] = ok
        ok_all &= ok
        recs.append(rec)
        log("178 case", rec)
    results["partials_178_cases"] = recs
    return ok_all


# ------------------------------------------------------------------ (d) graphs
def check_graph(results):
    ok = True
    for M in (4176, 4096, 4256):
        args = b100.mk(M, 11 + M)
        ref = tuple(t.clone() for t in with_fold(False, b100.cand_chain, args))
        eager = tuple(t.clone() for t in with_fold(True, b100.cand_chain, args))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        qr8.FOLD = True
        with torch.cuda.graph(graph):
            g_out = b100.cand_chain(args)
        torch.cuda.synchronize()
        gok = True
        for _ in range(3):
            for t in g_out:
                t.view(torch.uint8).fill_(0xFF)
            graph.replay()
            torch.cuda.synchronize()
            gok &= all(same(a, b) for a, b in zip(g_out, eager))
            gok &= all(eq_outputs(g_out, ref).values())
        results[f"cuda_graph_M{M}"] = {"replay_x3_bitwise_vs_eager_and_landing54": gok}
        ok &= gok
        log(f"cuda_graph_M{M}", gok)
    return ok


def check():
    results, passed = {}, True
    log("FOLD default", qr8.FOLD, "G1", qr8._FOLD_GRID1, "G2", qr8._FOLD_GRID2)
    for M in MS:
        passed &= check_op(M, results)
    passed &= check_pass(results)
    passed &= check_178(results)
    passed &= check_graph(results)
    results["compile_keys"] = [str(k) for k in qr8._COMPILED]
    print(json.dumps({
        "passed": bool(passed),
        "reference": "landing 54's op (FOLD=False: mode 1 + one-program Triton reduce + mode 2) "
                     "in-process, and main's pre-54 chain (nvfp4_qkv_rope / qknorm_rope + "
                     "fp8_sm12x.quantize_qkv_e4m3); bitwise",
        "results": results,
    }))


# ------------------------------------------------------------------ tune (diagnostic)
def tune():
    sets = {}
    for M in (4176, 4096):
        g = torch.Generator(device="cuda").manual_seed(M)
        planes = torch.randn(3, M, H, device="cuda", generator=g).bfloat16()
        qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
        kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
        th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
        cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
        pos = torch.arange(M, device="cuda", dtype=torch.int64)
        sets[M] = (planes[0], planes[1], planes[2], qn, kn, cs, pos)
    configs = [("l54", False, 0, 0)]
    for g1, g2 in [(752, 564), (752, 752), (564, 564), (376, 376), (1024, 564), (752, 376),
                   (752, 1128), (522, 522)]:
        configs.append((f"fold_{g1}_{g2}", True, g1, g2))
    configs.append(("l54b", False, 0, 0))
    graphs = {}
    for name, fold, g1, g2 in configs:
        qr8.FOLD, qr8._FOLD_GRID1, qr8._FOLD_GRID2 = fold, g1 or 752, g2 or 564

        def body():
            for M in [4176] * 24 + [4096] * 4:
                qr8.qknorm_rope_fp8(*sets[M], SM_SCALE)

        body()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            body()
        torch.cuda.synchronize()
        graphs[name] = gr
        log("captured", name)
    t0 = time.time()
    while time.time() - t0 < 8:
        for gr in graphs.values():
            gr.replay()
        torch.cuda.synchronize()
    ev = {k: [] for k in graphs}
    for it in range(60):
        for name, gr in graphs.items():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
            ev[name].append((s, e))
        torch.cuda.synchronize()
    res = {}
    for name, lst in ev.items():
        t = sorted(s.elapsed_time(e) for s, e in lst)
        med = statistics.median(t)
        res[name] = {"us_per_call": round(med * 1000 / 28, 2),
                     "noise": round((t[54] - t[6]) / med, 4)}
        log(name, res[name])
    print(json.dumps({"passed": True, "diagnostic": "tune, in-process", "res": res}))


if __name__ == "__main__":
    torch.manual_seed(0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"check": check, "tune": tune}[mode]()
