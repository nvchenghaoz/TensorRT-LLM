"""CPU launch-order view: for the thread that issued the first DtoD memcpy, print the runtime API
calls (with the kernel / memcpy each launched, via correlation id) around the first few DtoD copies.

usage: python nsys_cpu_seq.py <timeline.sqlite> [window]
"""
import json
import sqlite3
import sys


def main():
    path = sys.argv[1]
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    con = sqlite3.connect(path)
    cur = con.cursor()
    names = dict(cur.execute("select id, value from StringIds"))
    launched = {}
    for corr, gp, sid in cur.execute(
            "select correlationId, globalPid, coalesce(shortName, demangledName) from CUPTI_ACTIVITY_KIND_KERNEL"):
        launched[(gp >> 24, corr)] = "K " + names.get(sid, str(sid))[:70]
    for corr, gp, kind, nbytes in cur.execute(
            "select correlationId, globalPid, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
        launched[(gp >> 24, corr)] = f"M kind={kind} bytes={nbytes}"
    rt = list(cur.execute(
        "select r.start, r.globalTid, r.correlationId, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "left join StringIds s on r.nameId = s.id order by r.globalTid, r.start"))
    by_tid = {}
    for s, tid, corr, name in rt:
        by_tid.setdefault(tid, []).append((s, corr, name))
    summary = {}
    for tid, calls in by_tid.items():
        pid = tid >> 24
        seq = [(s, name, launched.get((pid, corr))) for s, corr, name in calls]
        seq = [x for x in seq if x[2] is not None]
        dtod = [i for i, x in enumerate(seq) if x[2].startswith("M kind=8")]
        summary[str(tid)] = {"launches": len(seq), "dtod": len(dtod)}
        if not dtod:
            continue
        print(f"=== tid {tid} launches={len(seq)} dtod={len(dtod)}")
        for c in dtod[:3]:
            print(f"--- around launch #{c}")
            for i in range(max(0, c - win), min(len(seq), c + 6)):
                print(f"{i:6d} {seq[i][1][:28]:28s} {seq[i][2]}")
    print(json.dumps({"passed": True, "threads": summary}))


if __name__ == "__main__":
    main()
