"""Untimed correctness for idea 64 (down_proj wave split), branch tree only.

1. Every integer M in [4096, 8192]: production dispatch (Linear NVFP4 apply,
   flagged like Gemma4 down_proj) int16-equal to main's trtllm::nvfp4_gemm,
   for randn seed 1, randn seed 2 and heavy-tailed randn*2^U(-8,8) per row.
   Reports the plan (pieces/tactics) census and kernel names at sample M.
2. Kill switch (ENABLED=False): dispatch count 0, output == nvfp4_gemm.
3. FP32 reference (b) from dequantized NVFP4 inputs at M 4800/8000.
4. Repeated untimed A,B,A = (8000, 4800, 8000) with the bench's schedule
   (5 warmup + 20 calls each, 5 rotating weight copies), outputs 0xA5-filled
   before every call (direct out=) and through the production allocator;
   each call int16-equal to that copy's call 1 and to main; inputs unchanged.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_linears, call, set_act_scale  # noqa
from common import kernel_names, log, make_act, read_dims  # noqa

import tensorrt_llm  # noqa
from tensorrt_llm._torch.autotuner import autotune  # noqa
from tensorrt_llm._torch.modules.gemma4 import down_wave_split as dws  # noqa

torch.backends.cuda.matmul.allow_tf32 = False
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                    dtype=torch.float32, device="cuda")


def dequant_unscaled(fp4, sf, rows, k):
    """E2M1 * E4M3 block scale (no global scale), swizzled 128x4 SF layout."""
    sfk = ((k // 16) + 3) // 4 * 4
    b = fp4[:rows].view(torch.uint8)
    lo = E2M1[(b & 0xF).long()]
    hi = E2M1[(b >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(rows, k)
    r = torch.arange(rows, device="cuda").view(-1, 1)
    c = torch.arange(k // 16, device="cuda").view(1, -1)
    idx = (r // 128) * (sfk // 4) * 512 + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)
    s = sf.view(-1)[idx.flatten()].view(torch.float8_e4m3fn).float().view(rows, k // 16)
    return (v.view(rows, k // 16, 16) * s.unsqueeze(-1)).view(rows, k)


def i16ne(a, b):
    return (a.view(torch.int16) != b.view(torch.int16)).sum()


def main():
    n, k, _ = read_dims()
    sfk = ((k // 16) + 3) // 4 * 4
    lins = build_linears(n, k, 5)
    mmax = 8192
    inputs = []
    for kind, seed in (("randn_s1", 1), ("randn_s2", 2), ("heavy_s3", 3)):
        a, s, g = make_act(mmax, k, seed, "heavy" if kind.startswith("heavy") else "randn")
        inputs.append((kind, a, s.view(-1), g))
    set_act_scale(lins, inputs[0][3])
    with autotune():
        call(lins[0], inputs[0][1], inputs[0][2])
    torch.cuda.synchronize()
    call(lins[0], inputs[0][1][:4096], inputs[0][2][:4096 * sfk])
    assert dws._state["mod"] is not None, "FI module not loaded"

    count = {"n": 0}
    orig = dws.down_wave_split_gemm

    def counted(*a, **kw):
        count["n"] += 1
        return orig(*a, **kw)

    dws.down_wave_split_gemm = counted

    def main_op(lin, a, s):
        return torch.ops.trtllm.nvfp4_gemm(a, lin.weight, s, lin.weight_scale, lin.alpha, torch.bfloat16)

    ok = True
    report = {}
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    census = {}
    for m in range(4096, 8193):
        pcs = dws.plan(m, n, sms)
        key = ("split:" if len(pcs) > 1 else "whole:") + "+".join(f"FI{t}" for _, _, t in pcs)
        census[key] = census.get(key, 0) + 1
    report["plan_census"] = census
    report["plan_samples"] = {m: dws.plan(m, n, sms) for m in (4096, 4416, 4544, 4800, 5056, 5696, 6720,
                                                                6848, 7104, 7360, 7936, 8000, 8100, 8192)}
    log(f"plan census: {census}")

    # 1. every M
    for kind, a, s, g in inputs:
        set_act_scale(lins, g)
        c0 = count["n"]
        diffs = torch.zeros(8193 - 4096, dtype=torch.int64, device="cuda")
        for m in range(4096, 8193):
            sm = s[: ((m + 127) // 128) * 128 * sfk]
            y = call(lins[0], a[:m], sm)
            r = main_op(lins[0], a[:m], sm)
            diffs[m - 4096] = i16ne(y, r)
        torch.cuda.synchronize()
        nbad = int((diffs > 0).sum())
        first = None
        if nbad:
            first = int((diffs > 0).nonzero()[0]) + 4096
        disp = count["n"] - c0
        report[f"everyM_{kind}"] = dict(M_checked=4097, M_mismatch=nbad, first_bad_M=first,
                                        total_diff_elems=int(diffs.sum()), dispatches=disp)
        log(f"every-M {kind}: mismatching M {nbad} first {first} dispatches {disp}")
        ok &= nbad == 0 and disp == 4097

    # kernel names at sample M (randn_s1)
    kind, a, s, g = inputs[0]
    set_act_scale(lins, g)
    kn = {}
    for m in (4096, 4800, 6720, 8000, 8192):
        sm = s[: ((m + 127) // 128) * 128 * sfk]
        kn[m] = dict(split=kernel_names(lambda: call(lins[0], a[:m], sm)),
                     main=kernel_names(lambda: main_op(lins[0], a[:m], sm)))
        log(f"kernels M={m}: split={[x[:90] for x in kn[m]['split']]} main={[x[:90] for x in kn[m]['main']]}")
    report["kernel_names"] = kn

    # 2. kill switch
    dws.ENABLED = False
    c0 = count["n"]
    ks = {}
    for m in (4096, 4800, 8000, 8192):
        sm = s[: ((m + 127) // 128) * 128 * sfk]
        ks[m] = int(i16ne(call(lins[0], a[:m], sm), main_op(lins[0], a[:m], sm)))
    ks["dispatches"] = count["n"] - c0
    dws.ENABLED = True
    report["kill_switch"] = ks
    log(f"kill switch: {ks}")
    ok &= ks["dispatches"] == 0 and all(v == 0 for kk, v in ks.items() if kk != "dispatches")

    # 3. FP32 reference
    refs = {}
    wdq = dequant_unscaled(lins[0].weight, lins[0].weight_scale, n, k)
    for kind, a, s, g in (inputs[0], inputs[2]):
        set_act_scale(lins, g)
        for m in (4800, 8000):
            sm = s[: ((m + 127) // 128) * 128 * sfk]
            adq = dequant_unscaled(a, s, m, k)
            ref = (adq @ wdq.T) * lins[0].alpha.float()
            del adq
            y = call(lins[0], a[:m], sm).float()
            r = main_op(lins[0], a[:m], sm).float()

            def err(x):
                return dict(maxabs=float((x - ref).abs().max()),
                            relfro=float((x - ref).norm() / ref.norm()))
            refs[f"{kind}_{m}"] = dict(cand=err(y), main=err(r), cand_vs_main_maxabs=float((y - r).abs().max()),
                                       finite=bool(torch.isfinite(y).all()))
            log(f"ref {kind} M={m}: {refs[f'{kind}_{m}']}")
            ok &= refs[f"{kind}_{m}"]["finite"] and refs[f"{kind}_{m}"]["cand_vs_main_maxabs"] == 0.0
            del ref, y, r
    del wdq
    report["fp32_ref"] = refs

    # 4. repeated A,B,A
    kind, a, s, g = inputs[0]
    set_act_scale(lins, g)
    snap = [a.clone(), s.clone()] + [x for lin in lins for x in (lin.weight.clone(), lin.weight_scale.clone(),
                                                                lin.alpha.clone())]
    seq = []
    mains = {}
    for m in (8000, 4800):
        sm = s[: ((m + 127) // 128) * 128 * sfk]
        mains[m] = [main_op(lins[c], a[:m], sm) for c in range(5)]
    for mode in ("direct_outA5", "alloc_A5"):
        for m in (8000, 4800, 8000):
            sm = s[: ((m + 127) // 128) * 128 * sfk]
            firsts = {}
            bad_first = bad_main = 0
            for i in range(25):
                c = i % 5
                lin = lins[c]
                if mode == "direct_outA5":
                    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                    out.view(torch.int16).fill_(-23131)  # 0xA5A5
                    y = dws.down_wave_split_gemm(a[:m], lin.weight, sm, lin.weight_scale, lin.alpha, out=out)
                else:
                    poison = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                    poison.view(torch.int16).fill_(-23131)
                    del poison
                    y = call(lin, a[:m], sm)
                if c not in firsts:
                    firsts[c] = y.clone()
                bad_first += int(i16ne(y, firsts[c]) > 0)
                bad_main += int(i16ne(y, mains[m][c]) > 0)
            seq.append(dict(mode=mode, m=m, calls=25, bad_vs_first=bad_first, bad_vs_main=bad_main))
            log(f"repeat {mode} M={m}: bad_vs_first {bad_first} bad_vs_main {bad_main}")
            ok &= bad_first == 0 and bad_main == 0
    after = [a, s] + [x for lin in lins for x in (lin.weight, lin.weight_scale, lin.alpha)]
    unchanged = all(torch.equal(x, y) for x, y in zip(snap, after))
    report["repeated_ABA"] = dict(seq=seq, inputs_unchanged=unchanged)
    ok &= unchanged
    log(f"inputs unchanged: {unchanged}")

    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    json.dump(report, open(os.path.join(art, "check.json"), "w"), indent=1, default=str)
    print(json.dumps({"passed": bool(ok),
                      "compared": "production Linear NVFP4 dispatch (wave split) vs main trtllm::nvfp4_gemm int16 at "
                                  "every M 4096-8192 x 3 inputs; kill switch; FP32 dequant reference; repeated "
                                  "A,B,A (8000,4800,8000) from 0xA5 outputs",
                      **{kk: v for kk, v in report.items() if kk.startswith("everyM") or kk in
                         ("kill_switch", "plan_census")},
                      "fp32_ref": refs, "repeated_ABA": report["repeated_ABA"]}, default=str))


if __name__ == "__main__":
    main()
