"""Idea 82: G3 Q-row prefix launch (run_nvfp4_out_q_rows / run_q_rows) -- synthetic correctness.

Start check: git log -3, ancestry of main 7ab1b650, L15/L18 markers in the tree, no gridDim in the
G3 kernel header.

Per S in (32848 i2i single block, 16464 t2i single block), q_rows = 16384, randn and peaky Q/K/V
([1,S,24,128] views of a guarded [3,S,3072] e4m3 qkv8), module = the served nfo module
(load_trim_levers(ROUTED_LEVERS) + nvfp4_out, which carries g3):
  full     run_nvfp4_out (landing 18's call), reference.
  prefix   run_nvfp4_out_q_rows(q_rows): fp4 rows [0, q_rows) and the leading q_rows/128 SF row
           blocks torch.equal to full; rows >= q_prefix_rows_written stay 0xA5 (the launch really
           skips them; i2i only); eager x3 then CUDA-graph replay x3, each compared with full and
           with the first call; guards clean; qkv unchanged.
  zero     run_nvfp4_out_q_rows(0): fp4 + whole sf torch.equal to full.
  bf16     run_q_rows(q_rows) of the served BF16 G3 module vs run(): O rows [0, q_rows) equal.
  census   kernel name and grid of each arm (torch.profiler chrome trace).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import b186_nfo as b  # noqa: E402

log = b.log
FILL = b.FILL
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def sh(*cmd):
    p = subprocess.run(list(cmd), capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def start_check():
    tf = open(os.path.join(ROOT, "tensorrt_llm/_torch/visual_gen/models/flux3/"
                                 "transformer_flux3.py")).read()
    tres = os.path.join(ROOT, "experiments/ascent-flashinfer/csrc/flux3_fp4_pair_residual_sm120.cu")
    from flashinfer.jit import env as jit_env

    g3h = (jit_env.FLASHINFER_CSRC_DIR / "fmha_v2" /
           "fused_multihead_flash_attention_kernel_flux3_g3.h").read_text()
    res = {
        "git_log": sh("git", "-C", ROOT, "log", "--oneline", "-3")[1],
        "ancestor_7ab1b650_rc": sh("git", "-C", ROOT, "merge-base", "--is-ancestor", "7ab1b650",
                                   "HEAD")[0],
        "FLUX3_LAST_BLOCK_ROWS": "FLUX3_LAST_BLOCK_ROWS" in tf,
        "_Flux3LastRowsSingleStreamBlock": "class _Flux3LastRowsSingleStreamBlock" in tf,
        "flux3_tres": os.path.exists(tres) and "namespace flux3_tres" in open(tres).read(),
        "g3_header_gridDim_uses": g3h.count("gridDim"),
    }
    log(f"[start] {json.dumps(res)}")
    return res


def grids(fn):
    """{kernel name: [grid, ...]} of the fmha launches of fn (torch.profiler chrome trace)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(path)
    ev = json.load(open(path)).get("traceEvents", [])
    os.unlink(path)
    out = {}
    for e in ev:
        if e.get("cat") == "kernel" and "fmha" in e.get("name", ""):
            out.setdefault(e["name"], []).append(e.get("args", {}).get("grid"))
    return out


def sf_prefix_bytes(rows):
    return rows * ((b.W // 16 + 3) // 4 * 4)


def main():
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    failures = []
    start = start_check()
    if start["ancestor_7ab1b650_rc"] != 0 or not (start["FLUX3_LAST_BLOCK_ROWS"]
                                                   and start["flux3_tres"]):
        failures.append("start_check")
    if start["g3_header_gridDim_uses"]:
        failures.append("g3_reads_gridDim")
    sample = b.make_qkv(4176, 7, False)
    lev = tuple(lib.load_trim_levers(tuple(lib.ROUTED_LEVERS), sample["k"], sample["v"]))
    assert "g3" in lev, lev
    nfo = lib.get_flux3_d128_module((*lev, "nvfp4_out"))
    bmod = lib.get_flux3_d128_module(lev)
    log(f"[build] levers={lev}")
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    QR = 16384
    rows, census = {}, {}

    def call_q(inp, out, q_rows):
        sc = inp["scales"]
        lib.fmha_v2_flux3_d128_nvfp4_out_q_rows(
            inp["q"], inp["k"], inp["v"], out.fp4, out.sf, gscale, b.H, b.D, inp["S"], q_rows,
            scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], module=nfo)

    for S in (32848, 16464):
        written = lib.q_prefix_rows_written(QR)
        for peaky in (False, True):
            tag = f"{'peaky' if peaky else 'randn'}_S{S}"
            inp = b.make_qkv(S, 300 + S + int(peaky), peaky)
            full = b.NfoOut(S)
            full.reset()
            b.call_nfo(lib, nfo, inp, full, gscale)
            torch.cuda.synchronize()
            ref_fp4, ref_sf = full.fp4.clone(), full.sf.clone()
            sfp = sf_prefix_bytes(QR)
            row = {"written_rows": written}
            # zero: identical to run_nvfp4_out
            z = b.NfoOut(S)
            z.reset()
            call_q(inp, z, 0)
            torch.cuda.synchronize()
            row["zero_equal"] = bool(torch.equal(z.fp4, ref_fp4) and torch.equal(z.sf, ref_sf))
            if not row["zero_equal"]:
                failures.append(f"{tag}_zero")
            if z.check():
                failures.append(f"{tag}_zero_guard")
            del z
            # prefix: eager x3, then graph replay x3
            pre = b.NfoOut(S)
            bad, first = [], None
            for i in range(3):
                pre.reset()
                call_q(inp, pre, QR)
                torch.cuda.synchronize()
                a = pre.fp4[:, :QR].clone(), pre.sf[:sfp].clone()
                if first is None:
                    first = a
                    if S > written:
                        tail = pre.fp4[:, written:]
                        row["tail_rows_untouched"] = bool((tail == FILL).all())
                        if not row["tail_rows_untouched"]:
                            failures.append(f"{tag}_tail_written")
                        row["rows_written_lt_written"] = bool((pre.fp4[:, written - 1] != FILL)
                                                              .any())
                ok = torch.equal(a[0], ref_fp4[:, :QR]) and torch.equal(a[1], ref_sf[:sfp])
                ok = ok and torch.equal(a[0], first[0]) and torch.equal(a[1], first[1])
                if not ok:
                    bad.append(f"eager{i}")
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                call_q(inp, pre, QR)
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call_q(inp, pre, QR)
            for i in range(3):
                pre.reset()
                g.replay()
                torch.cuda.synchronize()
                ok = (torch.equal(pre.fp4[:, :QR], ref_fp4[:, :QR])
                      and torch.equal(pre.sf[:sfp], ref_sf[:sfp])
                      and torch.equal(pre.fp4[:, :QR], first[0]))
                if not ok:
                    bad.append(f"graph{i}")
            del g
            row["mismatch"] = bad
            row["guard_dirty"] = pre.check()
            if bad:
                failures.append(f"{tag}_prefix")
            if row["guard_dirty"]:
                failures.append(f"{tag}_guard")
            # bf16 run_q_rows vs run
            o_full = torch.full((1, S, b.H, b.D), 7.0, device="cuda", dtype=torch.bfloat16)
            o_pre = torch.full((1, S, b.H, b.D), 7.0, device="cuda", dtype=torch.bfloat16)
            b.call_bf16(lib, bmod, inp, o_full)
            sc = inp["scales"]
            lib.fmha_v2_flux3_d128_q_rows(inp["q"], inp["k"], inp["v"], o_pre, b.H, b.D, S, QR,
                                          scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], module=bmod)
            torch.cuda.synchronize()
            row["bf16_prefix_equal"] = bool(torch.equal(o_full[:, :QR].view(torch.int16),
                                                        o_pre[:, :QR].view(torch.int16)))
            if not row["bf16_prefix_equal"]:
                failures.append(f"{tag}_bf16")
            if inp["guard"].check() or not torch.equal(inp["guard"].payload, inp["pristine"]):
                failures.append(f"{tag}_inputs_changed")
            if not peaky:
                census[f"full_S{S}"] = grids(lambda: b.call_nfo(lib, nfo, inp, full, gscale))
                census[f"q_rows_S{S}"] = grids(lambda: call_q(inp, pre, QR))
                census[f"q_rows0_S{S}"] = grids(lambda: call_q(inp, pre, 0))
            rows[tag] = row
            log(f"[{tag}] {json.dumps(row)} failures={len(failures)}")
            del inp, full, pre, o_full, o_pre
            torch.cuda.empty_cache()
    log(f"[census] {json.dumps(census)}")
    for S in (32848, 16464):
        nf, nq = census[f"full_S{S}"], census[f"q_rows_S{S}"]
        if set(nf) != set(nq) or not all(k.endswith("_g3") for k in nq):
            failures.append(f"census_names_S{S}")
    gf = list(census["full_S32848"].values())[0][0]
    gq = list(census["q_rows_S32848"].values())[0][0]
    gt = list(census["q_rows_S16464"].values())[0][0]
    gtf = list(census["full_S16464"].values())[0][0]
    if gf[0] != 172 or gq[0] != 86 or gt != gtf:
        failures.append(f"census_grid {gf} {gq} {gt} {gtf}")
    print(json.dumps(dict(passed=not failures, failures=failures, start=start, levers=list(lev),
                          cases=rows, census=census,
                          compared="run_nvfp4_out_q_rows(16384) fp4 rows [0,16384) + leading 128 "
                                   "SF row blocks vs run_nvfp4_out (landing 18's call), eager x3 "
                                   "+ graph x3; q_rows=0 whole output vs run_nvfp4_out; run_q_rows "
                                   "BF16 O prefix vs run; guards; kernel name/grid census")))


if __name__ == "__main__":
    main()
