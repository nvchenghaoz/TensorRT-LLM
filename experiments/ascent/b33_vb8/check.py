"""Branch 33 (idea 39) correctness: cand (v_b8t) vs base (main's served G3 modules), in-process.

Per S in --sizes, randn and peaky Q/K/V ([1,S,24,128] views of a 0xA5-guarded [3,S,3072] e4m3
buffer, fixed seeds, pristine copy compared after each S):
  nfo   cand fp4+sf == base fp4+sf bitwise; guards clean; --repeats repeated calls == call 0;
        3 graph replays == eager; no 0xA5 left in sf; SF padding rows all zero
  bf16  cand BF16 O == base bitwise (int16 view); guards; repeats; graph replays
  chain the per-step sets (t2i: 1 bf16 16384 + 4 nfo 16384 + 28 nfo 16464; i2i: 6 bf16 16384 +
        4 nfo 16384 + 28 nfo 32848), eager + 2 repeats + graph x3, cand vs base
  census kernel names per call (every call runs the _g3 kernel)
  sass  REG/STACK and G3 opcode counts (PRMT/LDSM/QMMA/SHFL/STL/LDL) for base and cand
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--sizes", default="16384,16464,32848,16900,4176")
    # n_bf16:n_nfo_img:S_img:S_full (+ 28 nfo at S_full); the last is i2i after landing 19
    p.add_argument("--chains", default="1:4:16384:16464,6:4:16384:32848,1:9:16384:32848")
    p.add_argument("--skip_sass", action="store_true")
    args = p.parse_args()
    lib = c.lib_mod()
    H, D, W = b.H, b.D, b.W
    sizes = [int(x) for x in args.sizes.split(",") if x]
    chains = [tuple(int(y) for y in s.split(":")) for s in args.chains.split(",") if s]
    routed, levs = c.arm_levers(lib, b.make_qkv(4176, 7, False))
    log(f"[levers] routed={routed} base={levs['base']} cand={levs['cand']}")
    mods = c.build(lib, levs, ("base", "cand"))
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    failures, rows, bitwise, census = [], {}, {}, {}

    resources = {}
    if not args.skip_sass:
        for a in ("base", "cand"):
            try:
                resources[a] = c.sass_census(lib, levs[a])
                resources[a + "_nfo"] = c.sass_census(lib, (*levs[a], "nvfp4_out"))
            except Exception as exc:  # noqa: BLE001
                resources[a] = {"error": repr(exc)}
        log(f"[sass] {json.dumps(resources)}")

    for S in sorted(set(sizes)):
        inp = b.make_qkv(S, 11 + S, False)
        out = b.NfoOut(S)
        o = torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)
        for a in ("base", "cand"):
            b.call_nfo(lib, mods[a]["nfo"], inp, out, gscale)
            census[f"{a}_nfo_S{S}"] = b.names(lambda m=mods[a]["nfo"]: b.call_nfo(lib, m, inp, out,
                                                                                 gscale))
            b.call_bf16(lib, mods[a]["bf16"], inp, o)
            census[f"{a}_bf16_S{S}"] = b.names(lambda m=mods[a]["bf16"]: b.call_bf16(lib, m, inp, o))
        del inp, out, o
        torch.cuda.empty_cache()
    log(f"[census] {json.dumps(census)}")
    for k, v in census.items():
        if not any(n.endswith("_g3") for n in v):
            failures.append(f"census_not_g3_{k}")
        if k.startswith("cand") and not any("vb8" in n for n in v):
            failures.append(f"census_cand_not_vb8_{k}")

    for peaky in (False, True):
        tag = "peaky" if peaky else "randn"
        for S in sizes:
            inp = b.make_qkv(S, 200 + S + int(peaky), peaky)
            refs = {}
            for a in ("base", "cand"):
                out = b.NfoOut(S)
                out.reset()
                ref, row = b.run_seq([lambda m=mods[a]["nfo"], oo=out: b.call_nfo(lib, m, inp, oo,
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
                ref, row = b.run_seq([lambda m=mods[a]["bf16"], oo=o: b.call_bf16(lib, m, inp, oo)],
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
                    f"{float((x - y).abs().max()):.4g}; mismatching d (first 32): "
                    f"{torch.nonzero(diff.any(0).any(0)).flatten()[:32].tolist()}")
            if inp["guard"].check():
                failures.append(f"{tag}_qkv_guard_S{S}")
            if not torch.equal(inp["guard"].payload, inp["pristine"]):
                failures.append(f"{tag}_inputs_changed_S{S}")
            log(f"[{tag} S{S}] nfo_bitwise={bitwise[f'{tag}_nfo_S{S}']} bf16_bitwise={eq} "
                f"failures={len(failures)}")
            del inp, brefs
            torch.cuda.empty_cache()

        for n_bf16, n_img, s_img, s_full in chains:
            inps = {S: b.make_qkv(S, 500 + S + int(peaky), peaky) for S in (s_img, s_full)}
            crefs = {}
            for a in ("base", "cand"):
                seq = [("b", s_img)] * n_bf16 + [("n", s_img)] * n_img + [("n", s_full)] * 28
                fns, tens, checks = [], [], []
                for kind, S in seq:
                    if kind == "b":
                        go = b.Guarded(S * W * 2)
                        o = go.payload.view(torch.bfloat16).view(1, S, H, D)
                        fns.append(lambda m=mods[a]["bf16"], i=inps[S], oo=o:
                                   b.call_bf16(lib, m, i, oo))
                        tens.append(o)
                        checks.append(go.check)
                    else:
                        out = b.NfoOut(S, post=4096)
                        fns.append(lambda m=mods[a]["nfo"], i=inps[S], oo=out:
                                   b.call_nfo(lib, m, i, oo, gscale))
                        tens += [out.fp4, out.sf]
                        checks.append(out.check)
                cn = f"{n_bf16}_{n_img}_{s_img}_{s_full}"
                ref, row = b.run_seq(fns, tens, checks, 2, failures, f"{tag}_chain{cn}_{a}")
                rows[f"{tag}_chain{cn}_{a}"] = row
                log(f"[{tag} chain {cn} {a}] {json.dumps(row)}")
                crefs[a] = [t.cpu() for t in ref]
                del fns, tens, checks, ref
                torch.cuda.empty_cache()
            eq = all(torch.equal(x, y) for x, y in zip(crefs["base"], crefs["cand"]))
            bitwise[f"{tag}_chain_{cn}"] = eq
            if not eq:
                failures.append(f"bitwise_{tag}_chain_{cn}")
            for S in inps:
                if not torch.equal(inps[S]["guard"].payload, inps[S]["pristine"]):
                    failures.append(f"{tag}_chain_inputs_changed_S{S}")
            del inps, crefs
            torch.cuda.empty_cache()

    res = dict(passed=not failures, failures=failures, routed=list(routed),
               levers={k: list(v) for k, v in levs.items()}, bitwise=bitwise, census=census,
               resources=resources, cases=rows,
               compared=("cand (routed G3 + v_b8t) vs base (main's served G3 BF16-out and nvfp4_out "
                         "modules) in one process: BF16 O and fp4+sf bitwise; 0xA5 guards around "
                         "O/fp4/sf/qkv; repeats == call 0; graph x3 == eager; SF padding rows zero; "
                         "per-step t2i/i2i chains; inputs unchanged"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
