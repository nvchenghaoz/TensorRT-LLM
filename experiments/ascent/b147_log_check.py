"""Read-only log check for idea 255 (EP2xTP2 re-pair). usage: b147_log_check.py <label=path_or_glob> ...

Counts per log (and per rank where the line carries one): the VERIFY driver receipts, the
FI decode chain / fused fc1+fc2 (landing 45) receipts, TAIL_MOE_GRAPH capture lines, JIT in
capture, errors, and the MoE layout lines. Prints a JSON summary as the last line.
"""
import glob
import json
import re
import sys
from collections import Counter

PATS = {
    "verify_mode": r"\[verify\] mode=\S+ steps=[^\n]*",
    "mapping": r"\[verify\] MAPPING rank[^\n]*",
    "llm_args": r"\[verify\] llm\.args\.[^\n]*",
    "verify_summary": r"\[TAIL_MOE_GRAPH VERIFY SUMMARY\][^\n]*",
    "captured": r"\[TAIL_MOE_GRAPH\] rank \d+ layer \d+: captured[^\n]*",
    "capture_failed": r"capture failed[^\n]*",
    "fi_warm": r"FlashInfer decode chain warmed up[^\n]*",
    "fi_on": r"decode chain on[^\n]*",
    "fi_off": r"decode chain off[^\n]*",
    "fc12_off": r"SM12x fused fc1\+fc2 decode kernel off for[^\n]*",
    "fc12_unavail": r"SM12x fused fc1\+fc2 decode kernel unavailable[^\n]*",
    "moe_tp": r"moe_tensor_parallel_size\W+\d+|moe_tp_size\W+\d+",
    "moe_ep": r"moe_expert_parallel_size\W+\d+|moe_ep_size\W+\d+",
    "jit_capture": r"(JIT|jit_compile|Compiling)[^\n]*(captur|graph)[^\n]*|captur[^\n]*(JIT|compil)[^\n]*",
    "error": r"(Traceback|CUDA error|out of memory|OutOfMemory)[^\n]*",
}
RANK = re.compile(r"\[RANK (\d+)\]|rank (\d+)")


def check(path):
    t = open(path, errors="replace").read()
    marker = "[verify] ===== starting mode "
    if marker in t:  # the VERIFY driver's mode=both log: one result per mode segment
        segs = t.split(marker)[1:]
        return [check_text(f"{path}#{s.split(' ', 1)[0]}", s) for s in segs]
    return [check_text(path, t)]


def check_text(path, t):
    res = {"path": path, "bytes": len(t)}
    for k, pat in PATS.items():
        ms = [(m.group(0), t[max(0, t.rfind("\n", 0, m.start())):m.start()]) for m in re.finditer(pat, t)]
        ranks = Counter()
        for body, prefix in ms:
            r = RANK.search(prefix + body)
            ranks[(r.group(1) or r.group(2)) if r else "?"] += 1
        res[k] = {"n": len(ms), "per_rank": dict(ranks), "first": [b[:300] for b, _ in ms[:6]]}
    return res


def main():
    out = {}
    for arg in sys.argv[1:]:
        label, _, pat = arg.partition("=")
        paths = sorted(glob.glob(pat))
        print(f"{label}: {len(paths)} file(s) for {pat}", flush=True)
        out[label] = [r for p in paths for r in check(p)]
        for r in out[label]:
            print(label, json.dumps(r, indent=1), flush=True)
    summary = {}
    for label, rs in out.items():
        summary[label] = [{k: (v["n"], v["per_rank"]) for k, v in r.items() if isinstance(v, dict)}
                          for r in rs]
    print(json.dumps({"passed": all(out.values()), "summary": summary}))


if __name__ == "__main__":
    main()
