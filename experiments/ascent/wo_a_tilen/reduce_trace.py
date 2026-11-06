"""CPU-only reduction of profile 373 (job 466, main 1d696e72) decode traces: SM120BlockScaledKernel per stream/site."""
import collections
import glob
import gzip
import json
import sys

ROOT = sys.argv[1]


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


out = {}
for case in ("decode_c1", "decode_c4"):
    files = sorted(glob.glob(f"{ROOT}/{case}/*rank-0.json.gz"))
    if not files:
        print("missing", case, flush=True)
        continue
    tr = load(files[0])
    ev = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ev.sort(key=lambda e: e["ts"])
    print(case, "kernel events", len(ev), flush=True)
    by_stream = collections.defaultdict(list)
    for e in ev:
        by_stream[e.get("args", {}).get("stream", e.get("tid"))].append(e)
    res = {}
    for s, lst in by_stream.items():
        idx = [i for i, e in enumerate(lst) if "SM120BlockScaledKernel" in e["name"]]
        if not idx:
            continue
        nxt = collections.Counter()
        prv = collections.Counter()
        durs_by_next = collections.defaultdict(list)
        for i in idx:
            n = lst[i + 1]["name"][:60] if i + 1 < len(lst) else "END"
            p = lst[i - 1]["name"][:60] if i > 0 else "BEGIN"
            nxt[n] += 1
            prv[p] += 1
            durs_by_next[(p, n)].append(lst[i]["dur"])
        sites = {}
        for k, v in durs_by_next.items():
            v.sort()
            sites[f"prev={k[0]} | next={k[1]}"] = {"n": len(v), "mean_us": round(sum(v) / len(v), 2), "p50": v[len(v) // 2]}
        res[str(s)] = {"calls": len(idx), "mean_us": round(sum(lst[i]["dur"] for i in idx) / len(idx), 2),
                       "grid": lst[idx[0]].get("args", {}).get("grid"), "sites": sites}
    # aux-stream shared expert vs main stream: for each aux SM120 call, does it end before the next main-stream
    # kernel that starts after the aux chain? report gap of last aux kernel end vs first main kernel waiting
    out[case] = res
    print(json.dumps({case: res}, indent=1), flush=True)
print(json.dumps({"passed": True, "summary": out}), flush=True)
