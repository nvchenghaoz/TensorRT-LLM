"""Per-instruction / per-source-line Nsight Compute counters for the d512 Tile64 kernel (idea 55).

Usage: ncu_source.py --report <file.ncu-rep[z]> --tag <name> [--report ... --tag ...]
For each report:
  * raw page: prints the metrics that size L2 sectors, LDGSTS, shared wavefronts / conflicts.
  * source page in SASS mode (per instruction, with the CUDA file:line correlation of a
    -lineinfo build): saves the CSV into $AUTO_DEV_ARTIFACTS, sums L2 theoretical sectors,
    shared wavefronts (actual / ideal / excessive) and warp-stall samples per opcode class and
    per CUDA source line, and prints the top rows.
"""
import argparse
import csv
import glob
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
from collections import defaultdict

RAW_KEYS = ("lts__t_sectors", "lts__t_requests", "ldgsts", "mem_shared", "shared_op",
            "bank_conflicts", "l1tex__data_pipe_lsu_wavefronts", "smsp__inst_executed.sum",
            "smsp__pcsamp_sample_count", "gpu__time_duration", "l1tex__t_sectors",
            "l1tex__t_requests", "sass_inst_executed_op", "sm__cycles_elapsed.avg ")


def find_ncu():
    cands = [shutil.which("ncu"), "/usr/local/cuda/bin/ncu"]
    cands += sorted(glob.glob("/tmp/*/*/ncu")) + sorted(glob.glob("/tmp/*/ncu"))
    cands += sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
    for c in cands:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:  # noqa: BLE001
        return None


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print("CMD", " ".join(cmd[1:]), "rc", p.returncode, "bytes", len(p.stdout), flush=True)
    if p.returncode != 0:
        print("ERR", p.stderr[-1500:], flush=True)
    return p.stdout if p.returncode == 0 else ""


def opclass(src):
    s = src.strip().split()
    if not s:
        return "?"
    op = s[0]
    if op.startswith("@"):
        op = s[1] if len(s) > 1 else op
    return op.split(".")[0] + ("." + op.split(".")[1] if op.startswith(("LDS", "LDSM", "STS",
                                                                         "LDGSTS", "LDG"))
                               and "." in op else "")


def raw_page(ncu, rep, tag):
    txt = run([ncu, "--import", rep, "--page", "raw", "--csv"])
    rows = list(csv.reader(io.StringIO(txt)))
    if len(rows) < 3:
        print("RAW empty", flush=True)
        return {}
    hdr, units = rows[0], rows[1]
    out = {}
    for ki, k in enumerate(hdr):
        if any(x in k for x in RAW_KEYS):
            vals = [r[ki] for r in rows[2:] if ki < len(r)]
            out[k] = (units[ki] if ki < len(units) else "", vals)
    for k in sorted(out):
        print("RAW", tag, k, out[k][0], out[k][1], flush=True)
    return {k: v[1] for k, v in out.items()}


def sass_page(ncu, rep, tag, art):
    got = {}
    for mode in ("sass", "cuda,sass"):
        txt = run([ncu, "--import", rep, "--page", "source", "--csv", "--print-source", mode])
        if not txt.strip():
            continue
        (art / f"source_{tag}_{mode.replace(',', '_')}.csv").write_text(txt)
        rows = list(csv.reader(io.StringIO(txt)))
        print("MODE", mode, "rows", len(rows), flush=True)
        got[mode] = rows
    return got


REGION_MARKS = [("struct RingLoader", "loader(set_tile/issue)"), ("struct Supported", "other"),
                ("void BatchPrefillPagedD512Tile64Kernel(", "prologue"),
                ("// Q tile: [64 chunks][64 rows]", "prologue_Q_fill"),
                ("// Prologue: sub-tiles 0 and 1", "prologue"),
                ("for (uint32_t iter = 0; iter < num_iterations; ++iter)", "loop_wait_sync_issue"),
                ("// ---------------- Q.K^T", "qk_mma+K_lds"),
                ("// ---------------- logits transform", "mask"),
                ("// ---------------- softmax", "softmax"),
                ("// ---------------- O += P . V", "pv_mma+V_ldsm"),
                ("cp_async_wait<0>();", "epilogue"),
                ("cudaError_t BatchPrefillPagedD512Tile64Dispatched", "host")]


def kernel_marks(arm):
    import importlib.util
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bench as b55
    spec = importlib.util.find_spec("flashinfer")
    kf = pathlib.Path(spec.origin).resolve().parent / "data" / b55.KFILE
    mtxt, _ = b55.main_text(kf.read_text())
    txt = b55.apply(mtxt, b55.ARMS[arm], arm)
    marks = []
    for i, ln in enumerate(txt.splitlines(), 1):
        for m, name in REGION_MARKS:
            if m in ln:
                marks.append((i, name))
    marks.sort()
    return marks, txt.splitlines()


def summarize_lines(rows, tag, arm):
    """cuda,sass page: per CUDA line aggregate rows (Address '-') of the Tile64 kernel file."""
    marks, ktxt = kernel_marks(arm)
    hdr = None
    cur_file = ""
    tot = defaultdict(float)
    reg = defaultdict(lambda: defaultdict(float))
    per_line = {}
    for r in rows:
        if not r:
            continue
        if r[0] == "File Path":
            cur_file = r[1] if len(r) > 1 else ""
            continue
        if r[0] == "Line No":
            hdr = r
            continue
        if hdr is None or len(r) < len(hdr) or r[2] != "-":
            continue
        vals = {}
        for i, c in enumerate(hdr):
            if i < 4:
                continue
            if re.search(r"Sectors|Wavefronts|Warp Stall Sampling \(All|Instructions Executed$|"
                         r"L1 Tag Requests", c):
                vals[c] = num(r[i]) or 0.0
        for c, v in vals.items():
            tot[c] += v
        if not cur_file.endswith("prefill_d512_tile64.cuh"):
            name = "other_file:" + cur_file.rsplit("/", 1)[-1]
            for c, v in vals.items():
                reg[name][c] += v
            continue
        ln = int(num(r[0]) or 0)
        rg = "header"
        for start, nm in marks:
            if ln >= start:
                rg = nm
        for c, v in vals.items():
            reg[rg][c] += v
        per_line[ln] = vals
    samp = next((c for c in tot if "Warp Stall Sampling (All" in c), None)
    print("LINE_TOTALS", tag, json.dumps({c: round(v) for c, v in tot.items()}), flush=True)
    for rg, d in sorted(reg.items(), key=lambda kv: -(kv[1].get(samp, 0) if samp else 0)):
        share = d.get(samp, 0) / tot[samp] if samp and tot.get(samp) else 0
        print("REGION", tag, f"{rg:28s}", f"stall_share={share:.4f}",
              json.dumps({c: round(v) for c, v in d.items() if v and c != samp}), flush=True)
    for ln in sorted(per_line):
        vals = per_line[ln]
        mem = {c: round(v) for c, v in vals.items() if v and ("Sectors" in c or "Wavefronts" in c
                                                               or "Tag" in c)}
        share = vals.get(samp, 0) / tot[samp] if samp and tot.get(samp) else 0
        if mem or share >= 0.004:
            src = ktxt[ln - 1].strip()[:90] if 0 < ln <= len(ktxt) else ""
            print("LINE", tag, ln, f"stall_share={share:.4f}", "|", src, "|", json.dumps(mem),
                  flush=True)
    return dict(totals={c: round(v) for c, v in tot.items()},
                regions={rg: {c: round(v) for c, v in d.items() if v} for rg, d in reg.items()})


def summarize_sass(rows, tag):
    hi = next((i for i, r in enumerate(rows) if "Source" in r and ("Address" in r or "# Address"
                                                                   in r)), None)
    if hi is None:
        print("SASS no header", flush=True)
        return {}
    hdr = rows[hi]
    ci = {c: i for i, c in enumerate(hdr)}
    print("SASS_COLUMNS", json.dumps(hdr), flush=True)
    src_i = ci["Source"]
    want = [c for c in hdr if re.search(r"Sectors|Wavefronts|Requests|Warp Stall Sampling \(All|"
                                        r"Instructions Executed|Bank Conflict|Shared", c)]
    by_op = defaultdict(lambda: defaultdict(float))
    tot = defaultdict(float)
    keep = []
    for r in rows[hi + 1:]:
        if len(r) < len(hdr):
            continue
        src = r[src_i]
        vals = {c: (num(r[ci[c]]) or 0.0) for c in want}
        op = opclass(src)
        for c, v in vals.items():
            by_op[op][c] += v
            tot[c] += v
        keep.append((src, vals))
    print("SASS_TOTALS", tag, json.dumps({c: round(v) for c, v in tot.items()}), flush=True)
    for op, d in sorted(by_op.items(), key=lambda kv: -max(kv[1].values() or [0])):
        nz = {c: round(v) for c, v in d.items() if v}
        if nz:
            print("OP", tag, op, json.dumps(nz), flush=True)
    samp = next((c for c in want if "Warp Stall Sampling (All" in c), None)
    keyc = [c for c in want if "Excessive" in c] + ([samp] if samp else [])
    for c in keyc:
        top = sorted(keep, key=lambda kv: -kv[1][c])[:25]
        for src, vals in top:
            if vals[c] <= 0:
                break
            print("TOP", tag, c, "|", src[:80], "|",
                  json.dumps({k: round(v) for k, v in vals.items() if v}), flush=True)
    return dict(totals={c: round(v) for c, v in tot.items()},
                by_op={op: {c: round(v) for c, v in d.items() if v} for op, d in by_op.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", required=True)
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--arm", action="append", help="idea-55 arm whose kernel text the report ran")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    arms = a.arm or []
    ncu = find_ncu()
    print("NCU", ncu, flush=True)
    if not ncu:
        print(json.dumps({"passed": False, "error": "no ncu"}))
        return
    art = pathlib.Path(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"))
    art.mkdir(parents=True, exist_ok=True)
    summary = {}
    for idx, (rep, tag) in enumerate(zip(a.report, a.tag)):
        print("REPORT", tag, rep, "exists", os.path.exists(rep), flush=True)
        if not os.path.exists(rep):
            summary[tag] = "missing"
            continue
        raw = raw_page(ncu, rep, tag) if a.raw else {}
        got = sass_page(ncu, rep, tag, art)
        s = summarize_sass(got["sass"], tag) if "sass" in got else {}
        arm = arms[idx] if idx < len(arms) else "main"
        lines = summarize_lines(got["cuda,sass"], tag, arm) if "cuda,sass" in got else {}
        summary[tag] = dict(raw={k: v for k, v in raw.items()}, sass=s, lines=lines)
    (art / "ncu_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"passed": True, "tags": list(summary)}), flush=True)


if __name__ == "__main__":
    main()
