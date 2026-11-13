"""Per-denoise-step census from an nsys sqlite export: DtoD memcpys and kernels launched inside each
`denoise_step N` NVTX range (matched through the CUDA runtime API call's correlation id and thread).

usage: python nsys_step_census.py <timeline.sqlite> [range_prefix]
"""
import json
import sqlite3
import sys
from collections import Counter


def main():
    path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "denoise_step"
    con = sqlite3.connect(path)
    cur = con.cursor()
    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    print("tables:", sorted(t for t in tables if "CUPTI" in t or "NVTX" in t), flush=True)

    def text_col():
        cols = [r[1] for r in cur.execute("pragma table_info(NVTX_EVENTS)")]
        return cols

    cols = text_col()
    if "textId" in cols:
        q = ("select n.start, n.end, n.globalTid, coalesce(n.text, s.value) from NVTX_EVENTS n "
             "left join StringIds s on n.textId = s.id where n.end is not null")
    else:
        q = "select start, end, globalTid, text from NVTX_EVENTS where end is not null"
    ranges = [r for r in cur.execute(q) if r[3] and r[3].startswith(prefix)]
    print(f"{len(ranges)} ranges with prefix {prefix!r}", flush=True)

    rt = list(cur.execute("select start, end, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"))
    rt.sort()
    memcpy = {}
    for corr, kind, nbytes, gpid in cur.execute(
            "select correlationId, copyKind, bytes, globalPid from CUPTI_ACTIVITY_KIND_MEMCPY"):
        memcpy.setdefault((gpid >> 24, corr), []).append((kind, nbytes))
    kname = {}
    knames = {}
    for corr, gpid, sid in cur.execute(
            "select correlationId, globalPid, coalesce(shortName, demangledName) from CUPTI_ACTIVITY_KIND_KERNEL"):
        kname.setdefault((gpid >> 24, corr), []).append(sid)
    for i, v in cur.execute("select id, value from StringIds"):
        knames[i] = v

    import bisect

    starts = [r[0] for r in rt]
    out = []
    for (s, e, tid, text) in sorted(ranges):
        pid = tid >> 24
        lo = bisect.bisect_left(starts, s)
        hi = bisect.bisect_right(starts, e)
        dtod = 0
        dtod_bytes = 0
        dtod_sizes = Counter()
        other_copies = Counter()
        kernels = Counter()
        for j in range(lo, hi):
            rs, re_, rtid, corr = rt[j]
            if rtid != tid:
                continue
            for kind, nbytes in memcpy.get((pid, corr), []):
                if kind == 8:
                    dtod += 1
                    dtod_bytes += nbytes
                    dtod_sizes[nbytes] += 1
                else:
                    other_copies[kind] += 1
            for kid in kname.get((pid, corr), []):
                kernels[knames.get(kid, str(kid))] += 1
        copyish = {k: v for k, v in kernels.items()
                   if any(w in k.lower() for w in ("copy", "clone", "cat"))}
        rec = {
            "range": text, "pid": pid, "dur_ms": (e - s) / 1e6, "memcpy_dtod": dtod,
            "memcpy_dtod_MB": dtod_bytes / 1e6,
            "dtod_sizes_bytes": {str(k): v for k, v in sorted(dtod_sizes.items())},
            "other_memcpy_kinds": dict(other_copies),
            "kernels_total": sum(kernels.values()), "copy_like_kernels": copyish,
            "top_kernels": dict(kernels.most_common(25)),
        }
        print(json.dumps(rec), flush=True)
        out.append(rec)
    print(json.dumps({"passed": True, "ranges": [
        {k: r[k] for k in ("range", "pid", "dur_ms", "memcpy_dtod", "memcpy_dtod_MB", "dtod_sizes_bytes",
                           "kernels_total", "copy_like_kernels")} for r in out]}))


if __name__ == "__main__":
    main()
