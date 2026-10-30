"""Branch 57 (idea 96) correctness: cand (v_cf) vs base (main's served G3 modules), in-process.

Per S in --sizes, randn and peaky Q/K/V ([1,S,24,128] views of a 0xA5-guarded [3,S,3072] e4m3
buffer, fixed seeds, pristine copy compared after each S):
  nfo   cand fp4+sf == base fp4+sf bitwise; guards clean; --repeats repeated calls == call 0;
        3 garbage-filled graph replays == eager; no 0xA5 left in sf; SF padding rows all zero
  bf16  cand BF16 O == base bitwise (int16 view); guards; repeats; graph replays
  qrows nfo q_rows 16384 over kv 32848 (the served i2i block-27 launch): written prefix bitwise
  b2    B=2 BF16-out at S=4096 (swap23 map) and S=4176 (S%64 != 0: old map), cand == base
  census kernel names per call: cand runs *_g3_vcf where s % 16 == 0 (and b == 1 or s % 64 == 0),
        *_g3 otherwise; base always *_g3
  sass  REG/STACK and G3 opcode counts; REG <= 168, STACK 0, no STL/LDL in every G3 kernel
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as c  # noqa: E402

b = c.b
log = c.log
FILL = b.FILL


def pad_check(sf_ref, S, W):
    pad_rows = ((S + 127) // 128 * 128) - S
    if not pad_rows:
        return {}
    cols = (W // 16 + 3) // 4 * 4
    blk = sf_ref[(S // 128) * 128 * cols:].view(cols // 4, 32, 4, 4)
    rr = (torch.arange(4, device="cuda")[:, None] * 32 + torch.arange(32, device="cuda")[None, :])
    mask = (rr >= S % 128).t()
    padv = blk.permute(1, 2, 0, 3)[mask]
    return dict(pad_rows=pad_rows, pad_bytes=int(padv.numel()), pad_nonzero=int((padv != 0).sum()),
                pad_ok=bool(int((padv != 0).sum()) == 0 and padv.numel() == pad_rows * W // 16))


def run_seq(fns, tensors, checks, repeats, failures, name, views=None):
    """eager once (reference), `repeats` more eager calls, then 3 garbage-filled graph replays;
    compare `views` (default: the tensors) to the reference; guards."""
    views = views or (lambda ts: ts)
    for f in fns:
        f()
    torch.cuda.synchronize()
    ref = [t.clone() for t in views(tensors)]
    dirty = [ch() for ch in checks]
    rep_bad = 0
    for r in range(repeats):
        for t in tensors:
            t.fill_(0x5A if t.dtype == torch.uint8 else -3.0)
        for f in fns:
            f()
        if (r + 1) % 10 == 0 or r == repeats - 1:
            torch.cuda.synchronize()
            rep_bad += sum(int(not torch.equal(x, y)) for x, y in zip(views(tensors), ref))
            dirty += [ch() for ch in checks]
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    gbad = 0
    for _ in range(3):
        for t in tensors:
            if t.dtype == torch.uint8:
                t.fill_(0x11)
            else:
                t.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        gbad += sum(int(not torch.equal(x.view(torch.uint8) if x.dtype != torch.uint8 else x,
                                        y.view(torch.uint8) if y.dtype != torch.uint8 else y))
                    for x, y in zip(views(tensors), ref))
        dirty += [ch() for ch in checks]
    d = [x for x in dirty if x]
    row = dict(calls=len(fns), repeat_mismatch=rep_bad, graph_mismatch=gbad, guard_dirty=d[:4])
    if rep_bad or gbad or d:
        failures.append(name)
    del g
    return ref, row


def expect_vcf(S, B=1):
    return S % 16 == 0 and (B == 1 or S % 64 == 0) and B * (S // 16) >= 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--sizes", default="16384,16464,32848,16900,4176")
    p.add_argument("--skip_sass", action="store_true")
    p.add_argument("--skip_b2", action="store_true")
    p.add_argument("--skip_qrows", action="store_true")
    args = p.parse_args()
    lib = c.lib_mod()
    H, D, W = b.H, b.D, b.W
    sizes = [int(x) for x in args.sizes.split(",") if x]
    routed, levs = c.arm_levers(lib, b.make_qkv(4176, 7, False))
    log(f"[levers] routed={routed} base={levs['base']} cand={levs['cand']}")
    mods = c.build(lib, levs, ("base", "cand"))
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    failures, rows, bitwise, census = [], {}, {}, {}

    resources = {}
    if not args.skip_sass:
        for a in ("base", "cand"):
            for suf, lv in (("", levs[a]), ("_nfo", (*levs[a], "nvfp4_out"))):
                try:
                    resources[a + suf] = c.sass_census(lib, lv)
                    bad = c.resources_ok(resources[a + suf])
                    if bad:
                        failures += [f"res_{a}{suf}_{x}" for x in bad]
                except Exception as exc:  # noqa: BLE001
                    resources[a + suf] = {"error": repr(exc)}
        log(f"[sass] {json.dumps(resources)}")
        if not any(k.endswith("_g3_vcf") for k in resources.get("cand_nfo", {}).get("res_usage", {})):
            failures.append("sass_cand_has_no_vcf_kernel")

    def check_names(key, names, want_vcf):
        g3 = [n for n in names if c.is_g3(n) or "_g3" in n]
        has_vcf = any("_g3_vcf" in n for n in names)
        has_plain = any(n.endswith("_g3") for n in names)
        if not g3:
            failures.append(f"census_not_g3_{key}")
        if want_vcf and not has_vcf:
            failures.append(f"census_expected_vcf_{key}")
        if not want_vcf and (has_vcf or not has_plain):
            failures.append(f"census_expected_plain_g3_{key}")

    for S in sorted(set(sizes)):
        inp = b.make_qkv(S, 11 + S, False)
        out = b.NfoOut(S)
        o = torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)
        for a in ("base", "cand"):
            b.call_nfo(lib, mods[a]["nfo"], inp, out, gscale)
            k = f"{a}_nfo_S{S}"
            census[k] = b.names(lambda m=mods[a]["nfo"]: b.call_nfo(lib, m, inp, out, gscale))
            check_names(k, census[k], a == "cand" and expect_vcf(S))
            b.call_bf16(lib, mods[a]["bf16"], inp, o)
            k = f"{a}_bf16_S{S}"
            census[k] = b.names(lambda m=mods[a]["bf16"]: b.call_bf16(lib, m, inp, o))
            check_names(k, census[k], a == "cand" and expect_vcf(S))
        del inp, out, o
        torch.cuda.empty_cache()
    log(f"[census] {json.dumps(census)}")

    for peaky in (False, True):
        tag = "peaky" if peaky else "randn"
        for S in sizes:
            inp = b.make_qkv(S, 200 + S + int(peaky), peaky)
            refs = {}
            for a in ("base", "cand"):
                out = b.NfoOut(S)
                out.reset()
                ref, row = run_seq([lambda m=mods[a]["nfo"], oo=out: b.call_nfo(lib, m, inp, oo,
                                                                               gscale)],
                                   [out.fp4, out.sf], [out.check], args.repeats, failures,
                                   f"{tag}_nfo_{a}_S{S}")
                row["sf_unwritten_0xA5_bytes"] = int((ref[1] == FILL).sum())
                if row["sf_unwritten_0xA5_bytes"]:
                    failures.append(f"{tag}_sf_unwritten_{a}_S{S}")
                pc = pad_check(ref[1], S, W)
                row.update(pc)
                if pc and not pc["pad_ok"]:
                    failures.append(f"{tag}_pad_{a}_S{S}")
                rows[f"{tag}_nfo_{a}_S{S}"] = row
                log(f"[{tag} nfo {a} S{S}] {json.dumps(row)}")
                refs[a] = ref
                del out
            eq = all(torch.equal(x, y) for x, y in zip(refs["base"], refs["cand"]))
            bitwise[f"{tag}_nfo_S{S}"] = eq
            if not eq:
                failures.append(f"bitwise_{tag}_nfo_S{S}")
                d0 = int((refs["base"][0] != refs["cand"][0]).sum())
                d1 = int((refs["base"][1] != refs["cand"][1]).sum())
                log(f"[{tag} S{S}] NFO MISMATCH fp4 bytes {d0} sf bytes {d1}")
            del refs
            brefs = {}
            for a in ("base", "cand"):
                go = b.Guarded(S * W * 2)
                o = go.payload.view(torch.bfloat16).view(1, S, H, D)
                ref, row = run_seq([lambda m=mods[a]["bf16"], oo=o: b.call_bf16(lib, m, inp, oo)],
                                   [o], [go.check], args.repeats, failures, f"{tag}_bf16_{a}_S{S}")
                rows[f"{tag}_bf16_{a}_S{S}"] = row
                brefs[a] = ref[0]
                del go
            eq = torch.equal(brefs["base"].view(torch.int16), brefs["cand"].view(torch.int16))
            bitwise[f"{tag}_bf16_S{S}"] = eq
            if not eq:
                failures.append(f"bitwise_{tag}_bf16_S{S}")
                x, y = brefs["base"].float(), brefs["cand"].float()
                diff = (x != y).view(S, H, D)
                log(f"[{tag} S{S}] BF16 MISMATCH elems {int(diff.sum())} max abs "
                    f"{float((x - y).abs().max()):.4g}; mismatching keys/d (first 32): "
                    f"{torch.nonzero(diff.any(0).any(0)).flatten()[:32].tolist()}")
            if inp["guard"].check():
                failures.append(f"{tag}_qkv_guard_S{S}")
            if not torch.equal(inp["guard"].payload, inp["pristine"]):
                failures.append(f"{tag}_inputs_changed_S{S}")
            log(f"[{tag} S{S}] nfo_bitwise={bitwise[f'{tag}_nfo_S{S}']} bf16_bitwise={eq} "
                f"failures={len(failures)}")
            del inp, brefs
            torch.cuda.empty_cache()

        if not args.skip_qrows:
            S = 32848
            inp = b.make_qkv(S, 700 + int(peaky), peaky)
            nrow = lib.q_prefix_rows_written(c.QR)
            cols = (W // 16 + 3) // 4 * 4
            nsf = (nrow + 127) // 128 * 128 * cols
            qrefs = {}
            for a in ("base", "cand"):
                out = b.NfoOut(S)
                out.reset()
                if a == "cand":
                    k = f"cand_nfo_qrows_S{S}"
                    census[k] = b.names(lambda m=mods[a]["nfo"], oo=out:
                                        c.call_nfo_q(lib, m, inp, oo, gscale))
                    check_names(k, census[k], True)
                    out.reset()
                ref, row = run_seq([lambda m=mods[a]["nfo"], oo=out: c.call_nfo_q(lib, m, inp, oo,
                                                                                  gscale)],
                                   [out.fp4, out.sf], [out.check], args.repeats, failures,
                                   f"{tag}_qrows_{a}",
                                   views=lambda ts: [ts[0][:, :nrow], ts[1][:nsf]])
                rows[f"{tag}_qrows_{a}"] = row
                log(f"[{tag} qrows {a}] {json.dumps(row)}")
                qrefs[a] = ref
                del out
            eq = all(torch.equal(x, y) for x, y in zip(qrefs["base"], qrefs["cand"]))
            bitwise[f"{tag}_qrows_16384_of_{S}"] = eq
            if not eq:
                failures.append(f"bitwise_{tag}_qrows")
            # prefix of the full launch (same inputs) must equal the q_rows prefix
            out = b.NfoOut(S)
            out.reset()
            b.call_nfo(lib, mods["cand"]["nfo"], inp, out, gscale)
            torch.cuda.synchronize()
            fr = int(c.QR * (W // 2))
            eqf = torch.equal(out.fp4[:, :c.QR].reshape(-1)[:fr], qrefs["cand"][0][:, :c.QR]
                              .reshape(-1)[:fr])
            bitwise[f"{tag}_qrows_prefix_eq_full"] = eqf
            if not eqf:
                failures.append(f"{tag}_qrows_prefix_ne_full")
            if not torch.equal(inp["guard"].payload, inp["pristine"]):
                failures.append(f"{tag}_qrows_inputs_changed")
            log(f"[{tag} qrows] bitwise={eq} prefix_eq_full={eqf}")
            del inp, qrefs, out
            torch.cuda.empty_cache()

        if not args.skip_b2:
            from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import \
                quantize_qkv_e4m3

            for S in (4096, 4176):
                g = torch.Generator(device="cuda").manual_seed(900 + S + int(peaky))
                buf = torch.randn((3, 2, S, H, D), generator=g, device="cuda") * 2.0
                if peaky:
                    m = torch.rand((3, 2, S, H, 1), generator=g, device="cuda") < 0.01
                    buf = torch.where(m, buf * 40.0, buf)
                q, k, v = buf.to(torch.bfloat16).unbind(0)
                q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / (D ** 0.5))
                inp = dict(S=S, q=q8.contiguous(), k=k8.contiguous(), v=v8.contiguous(),
                           scales=scales)
                pr = [t.clone() for t in (inp["q"], inp["k"], inp["v"])]
                o2 = {}
                for a in ("base", "cand"):
                    go = b.Guarded(2 * S * W * 2)
                    o = go.payload.view(torch.bfloat16).view(2, S, H, D)
                    if a == "cand":
                        kk = f"cand_bf16_B2_S{S}"
                        census[kk] = b.names(lambda m=mods[a]["bf16"], oo=o:
                                             b.call_bf16(lib, m, inp, oo))
                        check_names(kk, census[kk], expect_vcf(S, 2))
                    ref, row = run_seq([lambda m=mods[a]["bf16"], oo=o: b.call_bf16(lib, m, inp,
                                                                                   oo)],
                                       [o], [go.check], 10, failures, f"{tag}_b2_{a}_S{S}")
                    rows[f"{tag}_b2_{a}_S{S}"] = row
                    o2[a] = ref[0]
                    del go
                eq = torch.equal(o2["base"].view(torch.int16), o2["cand"].view(torch.int16))
                bitwise[f"{tag}_bf16_B2_S{S}"] = eq
                if not eq:
                    failures.append(f"bitwise_{tag}_b2_S{S}")
                if not all(torch.equal(x, y) for x, y in zip(pr, (inp["q"], inp["k"], inp["v"]))):
                    failures.append(f"{tag}_b2_inputs_changed_S{S}")
                log(f"[{tag} B2 S{S}] bitwise={eq} vcf_expected={expect_vcf(S, 2)}")
                del inp, o2, pr
                torch.cuda.empty_cache()

    res = dict(passed=not failures, failures=failures, routed=list(routed),
               levers={k: list(v) for k, v in levs.items()}, bitwise=bitwise, census=census,
               resources=resources, cases=rows,
               compared=("cand (routed G3 + v_cf) vs base (main's served G3 BF16-out and nvfp4_out "
                         "modules = the TRTLLM_FMHA_V2_FLUX3_VCF=0 kill switch) in one process: "
                         "BF16 O and fp4+sf bitwise; 0xA5 guards around O/fp4/sf/qkv; garbage-"
                         "filled repeats == call 0; NaN/0x11-filled graph x3 == eager; SF padding "
                         "rows zero; q_rows 16384 of 32848; B=2; inputs unchanged"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
