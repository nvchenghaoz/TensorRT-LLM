"""Per-source-line Nsight Compute counters for the d512 Tile64 kernel (idea 55 stage 0).

Usage: ncu_source.py --report <file.ncu-rep[z]> [--report ...] [--kfile <prefill_d512_tile64.cuh>]
For each report: exports the source page (CUDA lines, -lineinfo build) as CSV into
$AUTO_DEV_ARTIFACTS, prints the per-line rows with L2 sectors / shared wavefronts / LDGSTS
counters / warp-stall samples, and sums warp-stall samples per kernel region.
"""
import argparse
import csv
import glob
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys


def find_ncu():
    cands = [shutil.which("ncu"), "/usr/local/cuda/bin/ncu"]
    cands += sorted(glob.glob("/tmp/*/*/ncu")) + sorted(glob.glob("/tmp/*/ncu"))
    cands += sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
    cands += sorted(glob.glob("/usr/local/cuda*/nsight-compute*/ncu"))
    for c in cands:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:  # noqa: BLE001
        return None


def regions_from(kfile):
    """(start_line, name) markers in the kernel file, 1-based."""
    marks = [("struct RingLoader", "loader"), ("struct Supported", "other"),
             ("void BatchPrefillPagedD512Tile64Kernel(", "prologue"),
             ("// ------------------------------------------------------------- loaders",
              "prologue_fill"),
             ("// ------------------------------------------------------------- per-thread",
              "prologue"),
             ("for (uint32_t iter = 0; iter < num_iterations; ++iter)", "loop_sync_issue"),
             ("// ---------------- Q.K^T", "qk"),
             ("// ---------------- logits transform", "mask_softmax"),
             ("// ---------------- O += P . V", "pv"),
             ("cp_async_wait<0>();", "epilogue"),
             ("cudaError_t BatchPrefillPagedD512Tile64Dispatched", "host")]
    out = []
    if not kfile or not os.path.exists(kfile):
        return out
    lines = pathlib.Path(kfile).read_text().splitlines()
    for i, ln in enumerate(lines, 1):
        for m, name in marks:
            if m in ln:
                out.append((i, name))
    out.sort()
    return out


def region_of(line, marks):
    r = "header/other"
    for start, name in marks:
        if line >= start:
            r = name
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", required=True)
    ap.add_argument("--tag", action="append")
    ap.add_argument("--kfile", action="append")
    a = ap.parse_args()
    ncu = find_ncu()
    print("NCU", ncu, flush=True)
    if not ncu:
        print(json.dumps({"passed": False, "error": "no ncu"}))
        return
    art = pathlib.Path(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"))
    art.mkdir(parents=True, exist_ok=True)
    summary = {}
    for idx, rep in enumerate(a.report):
        tag = (a.tag or [])[idx] if a.tag and idx < len(a.tag) else f"r{idx}"
        kfile = (a.kfile or [])[idx] if a.kfile and idx < len(a.kfile) else None
        if kfile and kfile.startswith("auto"):
            # auto: the job tree's kernel file; auto:<arm>: that idea-55 arm's text of it
            import importlib.util
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
            import bench as b55
            spec = importlib.util.find_spec("flashinfer")
            kf = pathlib.Path(spec.origin).resolve().parent / "data" / b55.KFILE
            mtxt, _ = b55.main_text(kf.read_text())
            arm = kfile.split(":", 1)[1] if ":" in kfile else "main"
            out = art / f"kfile_{tag}_{arm}.cuh"
            out.write_text(b55.apply(mtxt, b55.ARMS[arm], arm))
            kfile = str(out)
        print("REPORT", tag, rep, "exists", os.path.exists(rep), flush=True)
        if not os.path.exists(rep):
            summary[tag] = "missing"
            continue
        txt = None
        for extra in (["--print-source", "cuda"], ["--print-source=cuda"], []):
            cmd = [ncu, "--import", rep, "--page", "source", "--csv", *extra]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            print("CMD", " ".join(cmd), "rc", p.returncode, "bytes", len(p.stdout), flush=True)
            if p.returncode == 0 and p.stdout.strip():
                txt = p.stdout
                break
            print("ERR", p.stderr[-2000:], flush=True)
        if txt is None:
            summary[tag] = "export failed"
            continue
        (art / f"source_{tag}.csv").write_text(txt)
        # the csv may hold several kernels: split on header lines
        rows = list(csv.reader(io.StringIO(txt)))
        hdr_idx = [i for i, r in enumerate(rows) if r and r[0] in ("#", "Line", "# Address",
                                                                      "Address", "Source")]
        print("HEADER_ROWS", hdr_idx[:5], "total rows", len(rows), flush=True)
        for i in range(min(3, len(rows))):
            print("ROW", i, rows[i][:60], flush=True)
        if not hdr_idx:
            continue
        h0 = hdr_idx[0]
        hdr = rows[h0]
        end = hdr_idx[1] if len(hdr_idx) > 1 else len(rows)
        body = rows[h0 + 1:end]
        print("COLUMNS", json.dumps(hdr), flush=True)
        ci = {c: i for i, c in enumerate(hdr)}
        line_col = next((i for c, i in ci.items() if c in ("#", "Line", "Line #")), 0)
        src_col = next((i for c, i in ci.items() if c == "Source"), None)
        want = [c for c in hdr if any(k in c for k in (
            "L2 Theoretical Sectors Global", "L1 Wavefronts Shared", "Shared Wavefronts",
            "L1 Conflicts Shared", "L2 Explicit", "Warp Stall Sampling (All", "Instructions Executed",
            "Global Requests", "Shared Requests", "LDGSTS", "Sectors", "Wavefronts"))]
        print("WANT", json.dumps(want), flush=True)
        marks = regions_from(kfile)
        totals = {c: 0.0 for c in want}
        reg = {}
        top = []
        for r in body:
            if len(r) < len(hdr):
                continue
            ln = num(r[line_col])
            vals = {c: num(r[ci[c]]) or 0.0 for c in want}
            for c in want:
                totals[c] += vals[c]
            if ln is not None:
                rg = region_of(int(ln), marks)
                d = reg.setdefault(rg, {c: 0.0 for c in want})
                for c in want:
                    d[c] += vals[c]
            if any(vals[c] > 0 for c in want if "Sampling" not in c and "Executed" not in c) or \
                    any(vals[c] > 0 for c in want if "Sampling" in c):
                top.append((int(ln) if ln is not None else -1,
                            (r[src_col] if src_col is not None else "")[:110].strip(), vals))
        print("TOTALS", tag, json.dumps({c: round(v) for c, v in totals.items()}), flush=True)
        for rg, d in reg.items():
            print("REGION", tag, rg, json.dumps({c: round(v) for c, v in d.items() if v}),
                  flush=True)
        samp = next((c for c in want if "Warp Stall Sampling (All" in c), None)
        for ln, src, vals in top:
            nz = {c: round(v) for c, v in vals.items() if v}
            if not nz:
                continue
            share = ""
            if samp and totals.get(samp):
                share = f"stall_share={vals[samp] / totals[samp]:.4f}"
            print("LINE", tag, ln, share, "|", src, "|", json.dumps(nz), flush=True)
        summary[tag] = dict(totals={c: round(v) for c, v in totals.items()},
                            regions={rg: {c: round(v) for c, v in d.items() if v}
                                     for rg, d in reg.items()})
    print(json.dumps({"passed": True, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
