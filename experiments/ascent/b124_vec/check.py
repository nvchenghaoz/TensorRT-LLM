"""Branch 124 (idea 192): vectorized + pipelined fold kernel (TRTLLM_FLUX3_QKV_FP8_VEC).

Reuses branch 115's suite (experiments/ascent/b115_fold/check.py): with VEC on (default), its
FOLD=True arm runs the VEC kernel, so (a) op level 50 repeats, (b) 16 pass cases incl. inf/NaN and
all-NaN planes, (c) external-partials layouts (178 layout, n_part 1 and 1024), (d) CUDA graph x3
are all bitwise vs landing 54 (FOLD=0) and main's pre-54 chain. Added here, bitwise vs main's
landing-58 fold kernel (VEC=False, same process):
  (e) op level at M 4096/4176/4256: qkv8, scales[0:8], gemm, attention output;
  (f) the same 16 pass cases;
  (g) a VEC CUDA graph replayed 3x vs a landing-58 graph.
modes: smoke (e at M 4176 + a few pass cases), check (all), tune (diagnostic timing per mode).
"""

import json
import os
import statistics
import sys
import time

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "b115_fold"))
sys.path.insert(0, os.path.join(here, "..", "b100_fp8direct"))
import check as c115  # noqa: E402
import op as b100  # noqa: E402

from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x  # noqa: E402
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)

log = b100.log
same = b100.same
SM_SCALE = c115.SM_SCALE
HEADS = c115.HEADS


def with_vec(flag, fn, *a, **k):
    old_v, old_f = qr8.VEC, qr8.FOLD
    qr8.VEC, qr8.FOLD = flag, True
    try:
        return fn(*a, **k)
    finally:
        qr8.VEC, qr8.FOLD = old_v, old_f


def vec_op(M, results, iters=c115.ITERS):
    args = b100.mk(M, 5 * M + 3)
    ins = [t for t in args if isinstance(t, torch.Tensor)]
    pristine = [t.clone() for t in ins]
    ref = tuple(t.clone() for t in with_vec(False, b100.cand_chain, args))
    first = None
    n_ref = n_first = n_inputs = 0
    for it in range(c115.WARMUP + iters):
        n_inputs += all(torch.equal(t, p) for t, p in zip(ins, pristine))
        for t, p in zip(ins, pristine):
            t.copy_(p)
        o = with_vec(True, b100.cand_chain, args)
        if it < c115.WARMUP:
            continue
        n_ref += all(same(a, b) for a, b in zip(o, ref))
        if first is None:
            first = tuple(t.clone() for t in o)
        n_first += all(same(a, b) for a, b in zip(o, first))
    a_ref = fp8_sm12x.visual_gen_fp8_attention_sm120_prequant(ref[0], ref[1], 1, HEADS)
    a_c = fp8_sm12x.visual_gen_fp8_attention_sm120_prequant(first[0], first[1], 1, HEADS)
    ok = (n_ref == iters and n_first == iters and n_inputs == c115.WARMUP + iters
          and same(a_ref, a_c))
    mism = int((b100.u8(first[0]) != b100.u8(ref[0])).sum())
    results[f"vec_vs_l58_op_M{M}"] = {
        "bitwise_vs_landing58(qkv8,scales8,gemm)": f"{n_ref}/{iters}",
        "bitwise_vs_call0": f"{n_first}/{iters}",
        "inputs_unchanged_before_reset": f"{n_inputs}/{c115.WARMUP + iters}",
        "attention_output_bitwise": same(a_ref, a_c),
        "qkv8_mismatched_bytes_call0": mism,
        "scales_call0": [float(x) for x in first[1].cpu()],
        "ok": ok,
    }
    log(f"vec_vs_l58_op_M{M}", results[f"vec_vs_l58_op_M{M}"])
    return ok


def vec_pass(results, n_cases=16):
    ok_all = True
    recs = []
    g = torch.Generator(device="cuda").manual_seed(192)
    for case in range(n_cases):
        M, mags, packed, special, src, ins = c115.make_pass_case(case, g)
        src_copy = src.clone()
        r8, rs = with_vec(False, qr8.qknorm_rope_fp8, *ins, SM_SCALE)
        v8, vs = with_vec(True, qr8.qknorm_rope_fp8, *ins, SM_SCALE)
        rec = {"M": M, "packed": packed, "special": special,
               "qkv8": same(v8, r8), "scales8": same(vs, rs),
               "mismatched_bytes": int((b100.u8(v8) != b100.u8(r8)).sum()),
               "inputs_unchanged": same(src, src_copy)}
        if not rec["qkv8"]:
            bad = (b100.u8(v8) != b100.u8(r8)).nonzero()[:4].tolist()
            rec["first_bad"] = bad
        rec["ok"] = rec["qkv8"] and rec["scales8"] and rec["inputs_unchanged"]
        ok_all &= rec["ok"]
        recs.append(rec)
        log("vec pass case", rec)
    results["vec_vs_l58_pass_cases"] = recs
    return ok_all


def vec_graph(results):
    ok = True
    for M in (4176, 4096, 4256):
        args = b100.mk(M, 17 + M)
        ref = tuple(t.clone() for t in with_vec(False, b100.cand_chain, args))
        eager = tuple(t.clone() for t in with_vec(True, b100.cand_chain, args))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        qr8.VEC, qr8.FOLD = True, True
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
            gok &= all(same(a, b) for a, b in zip(g_out, ref))
        results[f"vec_graph_M{M}"] = {"replay_x3_bitwise_vs_eager_and_landing58": gok}
        ok &= gok
        log(f"vec_graph_M{M}", gok)
    return ok


def smoke():
    results, passed = {}, True
    log("VEC", qr8.VEC, "FOLD", qr8.FOLD)
    t0 = time.time()
    passed &= vec_pass(results, 16)
    log(f"pass cases done {time.time() - t0:.0f}s")
    passed &= vec_op(4176, results, iters=5)
    print(json.dumps({"passed": bool(passed), "mode": "smoke", "results": results}))


def rot():
    """Which explicit RoPE contraction reproduces main's fold kernel bitwise (diagnostic)."""
    res = {}
    for forms in [("E0", "O0"), ("E1", "O1"), ("E1", "O2"), ("E2", "O1"), ("E2", "O2")]:
        qr8._ROT_FORMS = forms
        for key in [k for k in qr8._COMPILED if "vec" in k]:
            del qr8._COMPILED[key]
        g = torch.Generator(device="cuda").manual_seed(192)
        tot_even = tot_odd = 0
        for case in range(6):
            M, mags, packed, special, src, ins = c115.make_pass_case(case, g)
            r8, rs = with_vec(False, qr8.qknorm_rope_fp8, *ins, SM_SCALE)
            v8, vs = with_vec(True, qr8.qknorm_rope_fp8, *ins, SM_SCALE)
            bad = b100.u8(v8) != b100.u8(r8)
            tot_even += int(bad[..., 0::2].sum())
            tot_odd += int(bad[..., 1::2].sum())
        res["".join(forms)] = {"even_col_mismatch": tot_even, "odd_col_mismatch": tot_odd}
        log("rot forms", forms, res["".join(forms)])
    print(json.dumps({"passed": True, "diagnostic": "rot forms", "res": res}))


def check():
    results, passed = {}, True
    log("VEC default", qr8.VEC, "FOLD default", qr8.FOLD, "G1", qr8._FOLD_GRID1,
        "G2", qr8._FOLD_GRID2)
    for M in c115.MS:
        passed &= vec_op(M, results)
    passed &= vec_pass(results)
    passed &= vec_graph(results)
    log("--- branch 115 suite (FOLD=True arm = VEC kernel) vs landing 54 ---")
    for M in c115.MS:
        passed &= c115.check_op(M, results)
    passed &= c115.check_pass(results)
    passed &= c115.check_178(results)
    passed &= c115.check_graph(results)
    results["compile_keys"] = [str(k) for k in qr8._COMPILED]
    print(json.dumps({
        "passed": bool(passed),
        "reference": "main's landing-58 fold kernel (VEC=False) and landing 54 (FOLD=False) "
                     "in-process, plus main's pre-54 chain via branch 115's suite; bitwise",
        "results": results,
    }))


def tune():
    """Diagnostic: per-mode and per-call timing of qknorm_rope_fp8 alone, one CUDA graph of
    28 x M4176 + 4 x M4096, landing-58 fold vs VEC at several grids, interleaved."""
    sets = {}
    for M in (4176, 4096):
        g = torch.Generator(device="cuda").manual_seed(M)
        planes = torch.randn(3, M, c115.H, device="cuda", generator=g).bfloat16()
        qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
        kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
        th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
        cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
        pos = torch.arange(M, device="cuda", dtype=torch.int64)
        sets[M] = (planes[0], planes[1], planes[2], qn, kn, cs, pos)
    configs = [("l58", False, 0, 0), ("vec", True, 0, 0)]
    for spec in (sys.argv[2:] if len(sys.argv) > 2 else []):
        g1, g2 = (int(x) for x in spec.split(","))
        configs.append((f"vec_{g1}_{g2}", True, g1, g2))
    configs.append(("l58b", False, 0, 0))
    graphs = {}
    for name, vec, g1, g2 in configs:
        qr8.VEC, qr8.FOLD, qr8._FOLD_GRID1, qr8._FOLD_GRID2 = vec, True, g1, g2

        def body():
            for M in [4176] * 28 + [4096] * 4:
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
        res[name] = {"us_per_call": round(med * 1000 / 32, 2),
                     "noise": round((t[54] - t[6]) / med, 4)}
        log(name, res[name])
    print(json.dumps({"passed": True, "diagnostic": "tune, in-process", "res": res}))


if __name__ == "__main__":
    torch.manual_seed(0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"smoke": smoke, "check": check, "tune": tune, "rot": rot}[mode]()
