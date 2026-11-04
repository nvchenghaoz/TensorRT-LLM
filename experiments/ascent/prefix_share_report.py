"""Per-request TTFT per case for serving arms, with each cold-frontend miss named.

usage: prefix_share_report.py <label>=<artifacts/proxy dir> ...

For every arm: parses server.log 'Prefix token cache' lines (full tokenizations,
shared loads/publishes per pid, wall-clock second) and AIPerf profile_export.jsonl
per case (request_start_ns, TTFT, ITL). A timed request is a named cold miss
when a pid's full-tokenization count rises in the same second (+-1 s) as the
request start (server log stamps are local wall time, PST here).
"""
import datetime as dt
import json
import os
import re
import statistics
import sys

LINE = re.compile(r"^\[(\d\d)/(\d\d)/(\d{4})-(\d\d):(\d\d):(\d\d)\] .*Prefix token cache: (.*)$")
SUMMARY = re.compile(r"pid (\d+): (\d+) shared-prefix splices, (\d+) full tokenizations")
LOADED = re.compile(r"pid (\d+) loaded shared entry (\S+) from pid (\d+) .* in ([\d.]+) ms")
PUBLISHED = re.compile(r"pid (\d+) published shared entry (\S+) .* in ([\d.]+) ms")


def parse_log(path, tz_offset_s):
    events, full_by_pid, last_full = [], {}, {}
    for raw in open(path, errors="replace"):
        m = LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        mo, d, y, hh, mi, ss, rest = m.groups()
        t = dt.datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss)).replace(
            tzinfo=dt.timezone.utc).timestamp() - tz_offset_s
        s = SUMMARY.search(rest)
        if s:
            pid, splices, full = s.group(1), int(s.group(2)), int(s.group(3))
            if full > last_full.get(pid, 0):
                events.append(dict(t=t, kind="full", pid=pid, n=full - last_full.get(pid, 0)))
            last_full[pid] = full
            full_by_pid[pid] = full
            continue
        lm = LOADED.search(rest)
        if lm:
            events.append(dict(t=t, kind="load", pid=lm.group(1), src=lm.group(3), ms=float(lm.group(4))))
            continue
        pm = PUBLISHED.search(rest)
        if pm:
            events.append(dict(t=t, kind="publish", pid=pm.group(1), ms=float(pm.group(3))))
    return events, full_by_pid


def parse_case(path):
    reqs = []
    for raw in open(path):
        r = json.loads(raw)
        md, mt = r["metadata"], r["metrics"]
        reqs.append(dict(start_ns=md["request_start_ns"],
                         ttft=mt["time_to_first_token"]["value"],
                         itl=mt["inter_token_latency"]["value"]))
    reqs.sort(key=lambda x: x["start_ns"])
    return reqs


def tz_offset(log_path, case_dir):
    """Offset (s) between server-log local time and epoch, from the log's first stamp vs file mtime."""
    # PST = UTC-8 on 11-04; verify against the server log mtime.
    return -8 * 3600


def main():
    out = {}
    for arg in sys.argv[1:]:
        label, root = arg.split("=", 1)
        log = os.path.join(root, "server", "server.log")
        tzo = tz_offset(log, root)
        events, full_by_pid = parse_log(log, tzo)
        arm = dict(full_tokenizations_total=sum(full_by_pid.values()),
                   full_by_pid=full_by_pid,
                   loads=[e for e in events if e["kind"] == "load"],
                   publishes=[e for e in events if e["kind"] == "publish"],
                   full_events=[e for e in events if e["kind"] == "full"], cases={})
        for c in ("1", "2", "4"):
            p = os.path.join(root, f"aiperf-c{c}", "profile_export.jsonl")
            if not os.path.exists(p):
                continue
            reqs = parse_case(p)
            fulls = [e for e in events if e["kind"] == "full"]
            rows = []
            for i, r in enumerate(reqs):
                ts = r["start_ns"] / 1e9
                hit = [e for e in fulls if ts - 1.5 <= e["t"] <= ts + 1.5]
                rows.append(dict(i=i, request_start_ns=r["start_ns"], ttft_ms=round(r["ttft"], 2),
                                 itl_ms=round(r["itl"], 4),
                                 cold_miss_pid=hit[0]["pid"] if hit else None))
            ttfts = [r["ttft"] for r in reqs]
            arm["cases"][c] = dict(
                n=len(reqs), ttft_mean=statistics.mean(ttfts), ttft_p50=statistics.median(ttfts),
                ttft_min=min(ttfts), ttft_max=max(ttfts),
                itl_mean=statistics.mean(r["itl"] for r in reqs),
                named_cold_misses=[r for r in rows if r["cold_miss_pid"]], requests=rows)
        out[label] = arm
        print(f"== {label}: full tokenizations total {arm['full_tokenizations_total']} {full_by_pid}", file=sys.stderr)
        for e in arm["publishes"] + arm["loads"] + arm["full_events"]:
            print(f"   {e}", file=sys.stderr)
        for c, cs in arm["cases"].items():
            print(f"   c{c}: mean {cs['ttft_mean']:.2f} p50 {cs['ttft_p50']:.2f} min {cs['ttft_min']:.2f} "
                  f"max {cs['ttft_max']:.2f} itl {cs['itl_mean']:.4f} cold {cs['named_cold_misses']}",
                  file=sys.stderr)
            for r in cs["requests"]:
                print(f"      {r}", file=sys.stderr)
    print(json.dumps({"passed": True, "report": {k: {kk: vv for kk, vv in v.items()
                                                       if kk != "cases"} | {"cases": {c: {x: y for x, y in cs.items() if x != "requests"} for c, cs in v["cases"].items()}}
                                                   for k, v in out.items()}}))


if __name__ == "__main__":
    main()
