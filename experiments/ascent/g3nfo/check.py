"""Idea 6: G3 fmha_v2 flux3 kernel with the NVFP4-out epilogue -- correctness + census.

Modules (this tree's fmha_v2_flux3, levers = load_trim_levers(ROUTED_LEVERS), which carry g3):
  nfo_4w   levers - g3 + nvfp4_out   the 4-warp nvfp4_out kernel = main's served nfo kernel
                                     (on main the G3 guard routes every nfo call to it)
  nfo_g3   levers + nvfp4_out        the served nfo module (this branch: G3 + NVFP4 epilogue)
  bf16_4w  levers - g3               4-warp BF16 out
  bf16_g3  levers                    G3 BF16 out (the unforked call)

Per S, randn and peaky Q/K/V ([1,S,24,128] views of a guarded [3,S,3072] e4m3 qkv8), fp4/sf/O
carved from 0xA5-guarded buffers:
  nfo   fp4+sf nfo_g3 == nfo_4w bitwise; guards clean; 100 repeats == call 0; graph x3 == eager;
        no 0xA5 left anywhere in sf (every valid scale written, padding rows zeroed); padding
        rows [S, round_up(S,128)) all 0
  bf16  nfo_g3.run() (fp4 null -> G3 BF16 path) == bf16_4w bitwise; guards; repeats; graph
  chain served pass (1 bf16 S_img + 4 nfo S_img + 28 nfo S_full), eager + graph x3, g3 vs 4w
  census kernel names per call (torch.profiler)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import b186_nfo as b  # noqa: E402

log = b.log
FILL = b.FILL


def build(lib, sample):
    lev = tuple(lib.load_trim_levers(tuple(lib.ROUTED_LEVERS), sample["k"], sample["v"]))
    assert "g3" in lev, lev
    no_g3 = tuple(x for x in lev if x != "g3")
    t0 = time.time()
    mods = {
        "nfo_4w": lib.get_flux3_d128_module((*no_g3, "nvfp4_out")),
        "nfo_g3": lib.get_flux3_d128_module((*lev, "nvfp4_out")),
        "bf16_4w": lib.get_flux3_d128_module(no_g3),
        "bf16_g3": lib.get_flux3_d128_module(lev),
    }
    log(f"[build] levers={lev} {time.time() - t0:.1f}s")
    return lev, mods


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--sizes", default="16384,16896,16900,33280,33300,4096,4100,4176")
    p.add_argument("--chains", default="16384:16896,16384:33280,4096:4176")
    args = p.parse_args()
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    H, D, W = b.H, b.D, b.W
    sizes = [int(x) for x in args.sizes.split(",") if x]
    chains = [tuple(int(y) for y in c.split(":")) for c in args.chains.split(",") if c]
    lev, mods = build(lib, b.make_qkv(4176, 7, False))
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    failures, rows, bitwise, census = [], {}, {}, {}

    # census
    for S in sorted(set(sizes)):
        inp = b.make_qkv(S, 11 + S, False)
        out = b.NfoOut(S)
        o = torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)
        for k in ("nfo_4w", "nfo_g3"):
            b.call_nfo(lib, mods[k], inp, out, gscale)
            census[f"{k}_S{S}"] = b.names(lambda m=mods[k]: b.call_nfo(lib, m, inp, out, gscale))
        census[f"nfo_g3_bf16run_S{S}"] = b.names(lambda: b.call_bf16(lib, mods["nfo_g3"], inp, o))
        del inp, out, o
        torch.cuda.empty_cache()
    log(f"[census] {json.dumps(census)}")
    for S in sizes:
        ks = census[f"nfo_g3_S{S}"]
        if not any(k.endswith("_g3") for k in ks):
            failures.append(f"census_nfo_g3_not_g3_S{S}")
        if any(k.endswith("_g3") for k in census[f"nfo_4w_S{S}"]):
            failures.append(f"census_nfo_4w_is_g3_S{S}")

    for peaky in (False, True):
        tag = "peaky" if peaky else "randn"
        for S in sizes:
            inp = b.make_qkv(S, 200 + S + int(peaky), peaky)
            refs = {}
            for k in ("nfo_4w", "nfo_g3"):
                out = b.NfoOut(S)
                out.reset()
                ref, row = b.run_seq([lambda m=mods[k], o=out: b.call_nfo(lib, m, inp, o, gscale)],
                                     [out.fp4, out.sf], [out.check], args.repeats, failures,
                                     f"{tag}_nfo_{k}_S{S}")
                row["sf_unwritten_0xA5_bytes"] = int((ref[1] == FILL).sum())
                pad_rows = ((S + 127) // 128 * 128) - S
                if pad_rows:
                    # rows >= S of the flat 128x4 layout: the last 128-row block, rows S%128..127
                    sf = ref[1]
                    cols = (W // 16 + 3) // 4 * 4
                    blk = sf[(S // 128) * 128 * cols:].view(cols // 4, 32, 4, 4)  # [cb, r32, r4, c4]
                    # row = r4 * 32 + r32 inside the block
                    rr = (torch.arange(4, device="cuda")[:, None] * 32 +
                          torch.arange(32, device="cuda")[None, :])  # [r4, r32]
                    mask = (rr >= S % 128).t()  # [r32, r4]
                    padv = blk.permute(1, 2, 0, 3)[mask]
                    row["pad_rows"] = pad_rows
                    row["pad_bytes"] = int(padv.numel())
                    row["pad_nonzero"] = int((padv != 0).sum())
                    if row["pad_nonzero"] or row["pad_bytes"] != pad_rows * W // 16:
                        failures.append(f"{tag}_pad_{k}_S{S}")
                if row["sf_unwritten_0xA5_bytes"]:
                    failures.append(f"{tag}_sf_unwritten_{k}_S{S}")
                rows[f"{tag}_nfo_{k}_S{S}"] = row
                log(f"[{tag} nfo {k} S{S}] {json.dumps(row)}")
                refs[k] = ref
                del out
            eq = all(torch.equal(x, y) for x, y in zip(refs["nfo_4w"], refs["nfo_g3"]))
            bitwise[f"{tag}_nfo_S{S}"] = eq
            if not eq:
                failures.append(f"bitwise_{tag}_nfo_S{S}")
                d = (refs["nfo_4w"][0] != refs["nfo_g3"][0]).sum().item()
                d2 = (refs["nfo_4w"][1] != refs["nfo_g3"][1]).sum().item()
                log(f"[{tag} S{S}] MISMATCH fp4 bytes {d} sf bytes {d2}")
            del refs
            brefs = {}
            for k, m in (("bf16_4w", mods["bf16_4w"]), ("nfo_g3_bf16run", mods["nfo_g3"])):
                go = b.Guarded(S * W * 2)
                o = go.payload.view(torch.bfloat16).view(1, S, H, D)
                ref, row = b.run_seq([lambda mm=m, oo=o: b.call_bf16(lib, mm, inp, oo)], [o],
                                     [go.check], max(10, args.repeats // 10), failures,
                                     f"{tag}_bf16_{k}_S{S}")
                rows[f"{tag}_bf16_{k}_S{S}"] = row
                brefs[k] = ref[0]
                del go
            eq = torch.equal(brefs["bf16_4w"].view(torch.int16),
                             brefs["nfo_g3_bf16run"].view(torch.int16))
            bitwise[f"{tag}_bf16run_S{S}"] = eq
            if not eq:
                failures.append(f"bitwise_{tag}_bf16run_S{S}")
            if inp["guard"].check():
                failures.append(f"{tag}_qkv_guard_S{S}")
            if not torch.equal(inp["guard"].payload, inp["pristine"]):
                failures.append(f"{tag}_inputs_changed_S{S}")
            log(f"[{tag} S{S}] nfo_bitwise={bitwise[f'{tag}_nfo_S{S}']} "
                f"bf16run_bitwise={eq} failures={len(failures)}")
            del inp, brefs
            torch.cuda.empty_cache()

        for s_img, s_full in chains:
            inps = {S: b.make_qkv(S, 500 + S + int(peaky), peaky) for S in (s_img, s_full)}
            crefs = {}
            for arm, nm, bm in (("4w", "nfo_4w", "bf16_4w"), ("g3", "nfo_g3", "bf16_g3")):
                seq = [("b", s_img)] + [("n", s_img)] * 4 + [("n", s_full)] * 28
                fns, tens, checks = [], [], []
                # outputs shared per S like the served pass's allocator reuse would not be; keep
                # one buffer per call so every call's output is compared
                for kind, S in seq:
                    if kind == "b":
                        go = b.Guarded(S * W * 2)
                        o = go.payload.view(torch.bfloat16).view(1, S, H, D)
                        fns.append(lambda m=mods[bm], i=inps[S], oo=o: b.call_bf16(lib, m, i, oo))
                        tens.append(o)
                        checks.append(go.check)
                    else:
                        out = b.NfoOut(S, post=4096)
                        fns.append(lambda m=mods[nm], i=inps[S], oo=out:
                                   b.call_nfo(lib, m, i, oo, gscale))
                        tens += [out.fp4, out.sf]
                        checks.append(out.check)
                ref, row = b.run_seq(fns, tens, checks, 2, failures, f"{tag}_chain{s_full}_{arm}")
                rows[f"{tag}_chain{s_full}_{arm}"] = row
                log(f"[{tag} chain {s_img}:{s_full} {arm}] {json.dumps(row)}")
                crefs[arm] = [t.cpu() for t in ref]
                del fns, tens, checks, ref
                torch.cuda.empty_cache()
            eq = all(torch.equal(x, y) for x, y in zip(crefs["4w"], crefs["g3"]))
            bitwise[f"{tag}_chain_{s_img}_{s_full}"] = eq
            if not eq:
                failures.append(f"bitwise_{tag}_chain_{s_full}")
            del inps, crefs
            torch.cuda.empty_cache()

    import b165_g3

    resources = {}
    for k, lv in (("nfo_g3", (*lev, "nvfp4_out")), ("bf16_g3", lev),
                  ("nfo_4w", (*[x for x in lev if x != "g3"], "nvfp4_out"))):
        try:
            r = b165_g3.resource_usage(lib, lv)
            resources[k] = {"functions": {n: v for n, v in r.get("functions", {}).items()
                                          if "flux3" in n},
                            "sass_stl_ldl": r.get("sass_stl_ldl"), "sass_sha": r.get("sass_sha")}
        except Exception as exc:  # noqa: BLE001
            resources[k] = {"error": str(exc)}
    log(f"[resources] {json.dumps(resources)}")
    res = dict(passed=not failures, failures=failures, levers=list(lev), bitwise=bitwise,
               census=census, cases=rows, resources=resources,
               compared=("nfo_g3 (G3 + NVFP4 epilogue, served nfo module) vs nfo_4w (main's served "
                         "4-warp nvfp4_out kernel): fp4+sf bitwise; nfo_g3 BF16 run vs bf16_4w; "
                         "0xA5 guards around fp4/sf/O/qkv; 100 repeats == call 0; graph x3 == "
                         "eager; SF padding rows zero; served chain g3 vs 4w"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
