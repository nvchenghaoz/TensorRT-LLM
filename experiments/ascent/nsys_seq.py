"""Print the GPU-order sequence of kernels and memcpys of one device from an nsys sqlite export,
and classify each DtoD memcpy by the kernels around it.

usage: python nsys_seq.py <timeline.sqlite> [device_id] [n_print]
"""
import json
import sqlite3
import sys
from collections import Counter


def main():
    path = sys.argv[1]
    dev = int(sys.argv[2]) if len(sys.argv) > 2 else None
    n_print = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    con = sqlite3.connect(path)
    cur = con.cursor()
    names = dict(cur.execute("select id, value from StringIds"))
    ev = []
    for s, e, d, sid, stream, gp in cur.execute(
            "select start, end, deviceId, coalesce(shortName, demangledName), streamId, globalPid "
            "from CUPTI_ACTIVITY_KIND_KERNEL"):
        ev.append((s, e, (gp >> 24, d), "K", names.get(sid, str(sid))[:60], stream))
    for s, e, d, kind, nbytes, stream, gp in cur.execute(
            "select start, end, deviceId, copyKind, bytes, streamId, globalPid from CUPTI_ACTIVITY_KIND_MEMCPY"):
        ev.append((s, e, (gp >> 24, d), "M", f"memcpy kind={kind} bytes={nbytes}", stream))
    devs = sorted({x[2] for x in ev})
    print("(pid, device) keys:", devs, flush=True)
    for k in devs:
        sub = [x for x in ev if x[2] == k]
        print(k, "events", len(sub), "dtod", sum(1 for x in sub if x[3] == "M" and "kind=8" in x[4]),
              "nccl", sum(1 for x in sub if "AllReduce" in x[4]), flush=True)
    dev = devs[dev] if dev is not None and dev < len(devs) else devs[0]
    if dev is None:
        dev = devs[0]
    seq = sorted(x for x in ev if x[2] == dev)
    t0 = seq[0][0]
    dtod_idx = [i for i, x in enumerate(seq) if x[3] == "M" and "kind=8" in x[4]]
    shown = set()
    for c in dtod_idx[:3]:
        print(f"--- around DtoD at index {c}")
        for i in range(max(0, c - n_print), min(len(seq), c + 12)):
            s, e, d, t, n, st = seq[i]
            print(f"{i:5d} t={(s - t0) / 1e6:10.3f}ms dur={(e - s) / 1e3:9.1f}us st={st} {t} {n}")
    nccl_idx = [i for i, x in enumerate(seq) if "AllReduce" in x[4]]
    print("first nccl idx:", nccl_idx[:6], "dtod idx:", dtod_idx[:6], "total events:", len(seq))
    ctx = Counter()
    for i, x in enumerate(seq):
        if x[3] == "M" and "kind=8" in x[4]:
            prev = seq[i - 1][4] if i > 0 else None
            nxt = seq[i + 1][4] if i + 1 < len(seq) else None
            ctx[(prev, nxt, x[5])] += 1
    print(json.dumps({"passed": True, "dtod_context(prev,next,stream)": [[list(k), v] for k, v in ctx.most_common(20)]}))


if __name__ == "__main__":
    main()
