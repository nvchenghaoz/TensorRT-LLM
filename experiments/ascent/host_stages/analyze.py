"""Analysis for host_stages/run.py (branch 15, idea 21).

analyze_lifetime(client_results, trace_dir): per-request critical-path stage
  partition (sums to client TTFT exactly), per-step overlap hold, c=4 leader
  vs follower split. GPU times come from timing events mapped to host time
  with the minimum (gpu - host) offset over each case window.
analyze_nsys(sqlite, client_results, trace_dir): in-window GPU idle per case,
  split launch-starved vs blocking and by mechanism (NVTX AT: ranges).
"""
import bisect
import glob
import json
import os
import sqlite3
import statistics
from collections import defaultdict


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


def r3(x):
    return None if x is None else round(x, 3)


def load_trace(tdir):
    recs = []
    for f in sorted(glob.glob(os.path.join(str(tdir), "trace_*.jsonl"))):
        pid = os.path.basename(f)[6:-6]
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                r["pid"] = pid
                recs.append(r)
    return recs


class Trace:
    def __init__(self, tdir):
        self.recs = load_trace(tdir) if tdir else []
        by = defaultdict(list)
        for r in self.recs:
            by[r["n"]].append(r)
        for v in by.values():
            v.sort(key=lambda r: r["t"])
        self.by = by
        self.fe_req = {r["cid"]: r for r in by["fe_req"]}
        self.fe_head = {r.get("head"): r for r in by["fe_req"]}
        self.fe_out = {}
        for r in by["fe_out"]:
            self.fe_out.setdefault(r["cid"], r)
        self.fe_yield = {}
        for r in by["fe_yield"]:
            self.fe_yield.setdefault(r["cid"], r)
        self.px_send = {r["cid"]: r for r in by["px_send"]}
        self.w0 = {r["cid"]: r for r in by["w_submit0"]}
        self.w1 = {r["cid"]: r for r in by["w_submit1"]}

        def first_by_cids(name):
            d = {}
            for r in by[name]:
                for c in r.get("cids") or []:
                    d.setdefault(c, r)
            return d

        self.fetch = first_by_cids("ex_fetch")
        self.pub = first_by_cids("ex_pub")
        self.ipc0 = first_by_cids("w_ipc0")
        self.disp = first_by_cids("px_disp")
        self.passes = [p for p in by["ex_pass"] if p.get("cq")]
        self.pass_by_it = {p["it"]: p for p in self.passes}
        self.step_of = {}
        for p in self.passes:
            for c in p.get("ctx") or []:
                self.step_of.setdefault(c, p["it"])
        self.gpu = sorted([r for r in self.recs if "g" in r], key=lambda r: r["gs"])
        self.g_by_it = defaultdict(dict)
        for r in self.gpu:
            if r["n"] in ("g_fwd0", "g_fwd1", "g_s1"):
                self.g_by_it[r["it"]][r["n"]] = r
        # g_mc0/g_mc1/pi0/pi1/mc0/mc1 carry no iteration: attach by host time.
        for p in self.passes:
            if "fwd0" not in p or "fwd1" not in p:
                continue
            a, b = p["fwd0"], p["fwd1"]
            for nm in ("g_mc0", "g_mc1"):
                for r in self._between(self.by_g_name(nm), a, b):
                    self.g_by_it[p["it"]].setdefault(nm, r)
            for nm in ("pi0", "pi1", "mc0", "mc1"):
                for r in self._between(by[nm], a, b):
                    p.setdefault("_" + nm, r["t"])
                    break

    def by_g_name(self, nm):
        if not hasattr(self, "_gn"):
            self._gn = defaultdict(list)
            for r in self.gpu:
                self._gn[r["n"]].append(r)
            for v in self._gn.values():
                v.sort(key=lambda r: r["t"])
        return self._gn[nm]

    @staticmethod
    def _between(lst, a, b):
        ts = [r["t"] for r in lst]
        i = bisect.bisect_left(ts, a)
        out = []
        while i < len(lst) and lst[i]["t"] <= b:
            out.append(lst[i])
            i += 1
        return out

    def offset(self, t0, t1):
        """min over events in [t0,t1] of (gpu_s - host_s): maps GPU to host."""
        best = None
        for r in self.gpu:
            if t0 <= r["t"] <= t1:
                o = r["g"] / 1e3 - r["t"]
                if best is None or o < best:
                    best = o
        return best

    def gh(self, rec, off):
        """host time of a GPU record."""
        if rec is None or off is None:
            return None
        return rec["g"] / 1e3 - off


STAGES = ["http_receive", "tokenization", "ipc_to_executor", "queue_wait", "schedule_prepare_resources",
          "prepare_inputs_host", "wait_gpu_free", "model_forward_gpu_span", "post_model", "sample_gpu",
          "response_emission_hold", "ipc_back_to_frontend", "detok_sse_client"]


def request_rows(tr, case, off):
    rows = []
    for rq in case["requests"]:
        if not rq or "t_send" not in rq:
            continue
        fe = tr.fe_head.get(rq.get("head"))
        if fe is None:
            rows.append({"missing": "fe_req"})
            continue
        cid = fe["cid"]
        tc = rq.get("t_content", rq.get("t_first"))
        it = tr.step_of.get(cid)
        p = tr.pass_by_it.get(it) if it is not None else None
        g = tr.g_by_it.get(it, {}) if it is not None else {}
        fo = tr.fe_out.get(cid)
        px = tr.px_send.get(cid)
        w0, w1 = tr.w0.get(cid), tr.w1.get(cid)
        fx, pb = tr.fetch.get(cid), tr.pub.get(cid)
        i0, dp = tr.ipc0.get(cid), tr.disp.get(cid)
        if None in (p, fo, w1, fx, pb) or "fwd0" not in (p or {}):
            rows.append({"cid": cid, "missing": [k for k, v in dict(p=p, fo=fo, w1=w1, fx=fx, pb=pb).items()
                                                  if v is None]})
            continue
        G = {k: tr.gh(g.get(k), off) for k in ("g_fwd0", "g_mc0", "g_mc1", "g_fwd1", "g_s1")}
        mc0 = p.get("_mc0", p["fwd0"])
        pts = [("t_send", rq["t_send"]), ("entry", fe["entry"]), ("gen0", fe["gen0"]), ("w1", w1["t"]),
               ("fetch", fx["t"]), ("fwd0", p["fwd0"]), ("mc0", mc0), ("G_mc0", G["g_mc0"]),
               ("G_mc1", G["g_mc1"]), ("G_fwd1", G["g_fwd1"]), ("G_s1", G["g_s1"]), ("pub", pb["t"]),
               ("fe_out", fo["o0"]), ("client", tc)]
        if any(v is None for _, v in pts):
            rows.append({"cid": cid, "missing": [k for k, v in pts if v is None]})
            continue
        vals = [v for _, v in pts]
        seg = {STAGES[i]: 1e3 * (vals[i + 1] - vals[i]) for i in range(len(STAGES))}
        sub = {
            "tok_dispatch": 1e3 * (fe["tok0"] - fe["entry"]) if "tok0" in fe else None,
            "tok_call": 1e3 * (fe["tok1"] - fe["tok0"]) if "tok0" in fe else None,
            "tok_resume": 1e3 * (fe["gen0"] - fe["tok1"]) if "tok1" in fe else None,
            "gen_async_pre_send": 1e3 * (px["s0"] - fe["gen0"]) if px else None,
            "zmq_send": 1e3 * (px["t"] - px["s0"]) if px else None,
            "send_to_worker_recv": 1e3 * (w0["t"] - px["t"]) if (px and w0) else None,
            "worker_submit": 1e3 * (w1["t"] - w0["t"]) if w0 else None,
            "fetch_to_sched": 1e3 * (p["sched"] - fx["t"]) if "sched" in p else None,
            "sched_to_prep": 1e3 * (p["prep"] - p["sched"]) if "prep" in p else None,
            "prep_to_fwd0": 1e3 * (p["fwd0"] - p["prep"]) if "prep" in p else None,
            "prepare_inputs_call": 1e3 * (p["_pi1"] - p["_pi0"]) if "_pi1" in p else None,
            "model_enqueue_host": 1e3 * (p["_mc1"] - p["_mc0"]) if "_mc1" in p else None,
            "forward_step_host": 1e3 * (p["fwd1"] - p["fwd0"]),
            "gpu_lag_after_enqueue": 1e3 * (G["g_mc1"] - p["_mc1"]) if "_mc1" in p else None,
            "sample_host": 1e3 * (p["s1"] - p["s0"]) if "s1" in p else None,
            "pub_to_worker_ipc": 1e3 * (i0["t"] - pb["t"]) if i0 else None,
            "worker_ipc_to_proxy": 1e3 * (dp["t"] - i0["t"]) if (i0 and dp) else None,
            "proxy_to_frontend": 1e3 * (fo["o0"] - dp["t"]) if dp else None,
            "detok_postproc": 1e3 * (fo["pp1"] - fo["o0"]),
            "sse_to_client": 1e3 * (tc - fo["pp1"]),
        }
        rows.append({"cid": cid, "it": it, "ttft_ms": rq["ttft_ms"], "t_send": rq["t_send"],
                     "sum_ms": sum(seg.values()), "seg": seg, "sub": sub, "nctx": len(p.get("ctx") or []),
                     "ntok": p.get("ntok")})
    return rows


def summarize_rows(rows):
    ok = [r for r in rows if "seg" in r]
    out = {"n": len(ok), "missing": len(rows) - len(ok),
           "ttft_median": r3(med([r["ttft_ms"] for r in ok])),
           "ttft_mean": r3(statistics.mean([r["ttft_ms"] for r in ok])) if ok else None,
           "max_abs_sum_err_ms": r3(max((abs(r["sum_ms"] - r["ttft_ms"]) for r in ok), default=None)),
           "stage_median_ms": {k: r3(med([r["seg"][k] for r in ok])) for k in STAGES},
           "stage_mean_ms": {k: r3(statistics.mean([r["seg"][k] for r in ok])) if ok else None
                             for k in STAGES},
           "sub_median_ms": {k: r3(med([r["sub"][k] for r in ok])) for k in (ok[0]["sub"] if ok else {})}}
    if ok:
        out["sum_of_stage_medians"] = r3(sum(v for v in out["stage_median_ms"].values() if v is not None))
        out["sum_of_stage_means"] = r3(sum(v for v in out["stage_mean_ms"].values() if v is not None))
    out["missing_detail"] = [r for r in rows if "seg" not in r][:5]
    return out


def step_rows(tr, case, off):
    t0, t1 = case["t0"], case["t1"]
    rows = []
    passes = [p for p in tr.passes if t0 - 0.5 <= p["top"] <= t1 + 0.5]
    for i, p in enumerate(passes):
        if not p.get("ctx"):
            continue
        g = tr.g_by_it.get(p["it"], {})
        Gs1 = tr.gh(g.get("g_s1"), off)
        Gf0 = tr.gh(g.get("g_fwd0"), off)
        Gm0 = tr.gh(g.get("g_mc0"), off)
        Gf1 = tr.gh(g.get("g_fwd1"), off)
        pubs = [tr.pub[c]["t"] for c in p["ctx"] if c in tr.pub]
        pub = min(pubs) if pubs else None
        nxt = None
        for q in tr.passes:
            if q.get("prev_it") == p["it"]:
                nxt = q
                break
        row = {"it": p["it"], "nctx": len(p["ctx"]), "ntok": p.get("ntok"), "ngen": p.get("ngen"),
               "enqueue_ms": r3(1e3 * (p["fwd1"] - p["fwd0"])),
               "model_enqueue_ms": r3(1e3 * (p["_mc1"] - p["_mc0"])) if "_mc1" in p else None,
               "gpu_fwd_ms": r3(1e3 * (Gf1 - Gm0)) if (Gf1 and Gm0) else None,
               "gpu_sample_ms": r3(1e3 * (Gs1 - Gf1)) if (Gs1 and Gf1) else None,
               "gpu_start_lag_ms": r3(1e3 * (Gf0 - p["fwd0"])) if Gf0 else None,
               "hold_ms": r3(1e3 * (pub - Gs1)) if (pub and Gs1) else None}
        if nxt is not None and pub and Gs1:
            # Split [Gs1, pub] by what the loop was doing in pass N+1.
            marks = [("n1_before_fwd", nxt.get("top")), ("n1_fwd_enqueue", nxt.get("fwd0")),
                     ("n1_fwd_to_pp", nxt.get("fwd1")), ("n1_pp_to_pub", nxt.get("pp0"))]
            iv = []
            for j, (nm, st) in enumerate(marks):
                en = marks[j + 1][1] if j + 1 < len(marks) else pub
                if st is None or en is None:
                    continue
                iv.append((nm, st, en))
            split = {}
            for nm, st, en in iv:
                a, b = max(st, Gs1), min(en, pub)
                split[nm] = r3(1e3 * max(0.0, b - a))
            # Time before pass N+1 started (still in pass N tail) also counts.
            if nxt.get("top") and nxt["top"] > Gs1:
                split["pass_n_tail"] = r3(1e3 * (min(nxt["top"], pub) - Gs1))
            row["hold_split"] = split
            row["next_has_forward"] = "fwd0" in nxt
            row["next_enqueue_ms"] = r3(1e3 * (nxt["fwd1"] - nxt["fwd0"])) if "fwd1" in nxt else None
            row["next_ntok"] = nxt.get("ntok")
            if "upd0" in nxt and "upd1" in nxt:
                row["next_update_wait_ms"] = r3(1e3 * (nxt["upd1"] - nxt["upd0"]))
        rows.append(row)
    return rows


def waves(case_rows, reqs):
    """Burst split: requests sent within 5 ms of the previous send form one burst."""
    ok = sorted([r for r in case_rows if "seg" in r], key=lambda r: r["t_send"])
    lead, foll = [], []
    prev = None
    for r in ok:
        if prev is None or r["t_send"] - prev > 0.005:
            lead.append(r)
        else:
            foll.append(r)
        prev = r["t_send"]

    def s(rs):
        return {"n": len(rs), "ttft_median": r3(med([r["ttft_ms"] for r in rs])),
                "ttft_p10": r3(pct([r["ttft_ms"] for r in rs], 10)),
                "ttft_p90": r3(pct([r["ttft_ms"] for r in rs], 90)),
                "stage_median_ms": {k: r3(med([r["seg"][k] for r in rs])) for k in STAGES},
                "nctx_of_step_median": med([r["nctx"] for r in rs])}

    by_n = defaultdict(list)
    for r in ok:
        by_n[r["nctx"]].append(r)
    return {"leaders": s(lead), "followers": s(foll),
            "by_step_nctx": {str(k): s(v) for k, v in sorted(by_n.items())}}


def analyze_lifetime(results, tdir):
    tr = Trace(tdir)
    out = {"n_records": len(tr.recs), "record_types": {k: len(v) for k, v in tr.by.items()}}
    offs = {}
    for name, case in results.items():
        off = tr.offset(case["t0"] - 0.5, case["t1"] + 0.5)
        offs[name] = off
        rows = request_rows(tr, case, off)
        c = {"summary": summarize_rows(rows)}
        st = step_rows(tr, case, off)
        c["steps_n"] = len(st)
        c["steps"] = st[:40]
        hs = [s["hold_ms"] for s in st if s.get("hold_ms") is not None]
        c["hold_median_ms"] = r3(med(hs))
        c["hold_p10_p90"] = [r3(pct(hs, 10)), r3(pct(hs, 90))]
        if case["conc"] > 1:
            c["waves"] = waves(rows, case["requests"])
        c["client_ttft_mean"] = r3(statistics.mean([r["ttft_ms"] for r in case["requests"] if r and "ttft_ms" in r]))
        c["client_ttft_median"] = r3(med([r["ttft_ms"] for r in case["requests"] if r and "ttft_ms" in r]))
        out[name] = c
    vals = [v for v in offs.values() if v is not None]
    out["gpu_host_offset_spread_ms"] = r3(1e3 * (max(vals) - min(vals))) if vals else None
    return out


def client_only(results):
    out = {}
    for name, case in results.items():
        t = [r["ttft_ms"] for r in case["requests"] if r and "ttft_ms" in r]
        out[name] = {"n": len(t), "ttft_mean": r3(statistics.mean(t)) if t else None, "ttft_median": r3(med(t)),
                     "ttft_p10": r3(pct(t, 10)), "ttft_p90": r3(pct(t, 90))}
        if case["conc"] > 1:
            ok = sorted([r for r in case["requests"] if r and "ttft_ms" in r], key=lambda r: r["t_send"])
            lead, foll, prev = [], [], None
            for r in ok:
                (lead if prev is None or r["t_send"] - prev > 0.005 else foll).append(r["ttft_ms"])
                prev = r["t_send"]
            out[name]["leaders"] = {"n": len(lead), "median": r3(med(lead)), "p10": r3(pct(lead, 10)),
                                    "p90": r3(pct(lead, 90))}
            out[name]["followers"] = {"n": len(foll), "median": r3(med(foll)), "p10": r3(pct(foll, 10)),
                                      "p90": r3(pct(foll, 90))}
    return out


# ----------------------------------------------------------------- nsys
SYNC_WORDS = ("Synchronize", "cudaMemcpy", "cuMemcpy")


def analyze_nsys(sq, results, tdir):
    tr = Trace(tdir)
    con = sqlite3.connect(sq)
    cur = con.cursor()
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    out = {"tables_present": sorted(t for t in tables if t.startswith(("CUPTI", "NVTX")))}
    S = dict(cur.execute("SELECT id, value FROM StringIds"))

    gpu = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        for st, en, corr, nm in cur.execute(
                "SELECT start, end, correlationId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            gpu.append((st, en, corr, "K", S.get(nm, "")))
    for tb, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "C"), ("CUPTI_ACTIVITY_KIND_MEMSET", "S")):
        if tb in tables:
            for st, en, corr in cur.execute(f"SELECT start, end, correlationId FROM {tb}"):
                gpu.append((st, en, corr, kind, tb[-6:]))
    gpu.sort()
    rt = {}
    rt_by_tid = defaultdict(list)
    for st, en, nid, corr, tid in cur.execute(
            "SELECT start, end, nameId, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        nm = S.get(nid, "")
        rt[corr] = (st, en, tid, nm)
        rt_by_tid[tid].append((st, en, nm))
    for v in rt_by_tid.values():
        v.sort()
    nv = []
    cols = [r[1] for r in cur.execute("PRAGMA table_info(NVTX_EVENTS)")]
    has_text = "text" in cols
    q = "SELECT start, end, " + ("text, " if has_text else "NULL, ") + "textId, globalTid FROM NVTX_EVENTS"
    for st, en, tx, tid_, gt in cur.execute(q):
        name = tx if tx is not None else S.get(tid_, "")
        nv.append((st, en, name, gt))
    con.close()
    # Clock alignment: "AT:clk <perf>" marks.
    offs = []
    for st, en, name, gt in nv:
        if name and name.startswith("AT:clk "):
            try:
                offs.append(st - float(name.split()[1]) * 1e9)
            except ValueError:
                pass
    if not offs:
        out["error"] = "no AT:clk marks; cannot align"
        return out
    off_ns = statistics.median(offs)
    out["clock_align_spread_us"] = r3((max(offs) - min(offs)) / 1e3)
    out["n_gpu_ops"] = len(gpu)

    def ns(t):
        return t * 1e9 + off_ns

    # Executor thread: most kernel launches.
    cnt = defaultdict(int)
    for corr, (st, en, tid, nm) in rt.items():
        if "Launch" in nm:
            cnt[tid] += 1
    ex_tid = max(cnt, key=cnt.get) if cnt else None
    ranges = sorted([(st, en, name) for st, en, name, gt in nv
                     if gt == ex_tid and en is not None and en > st and name and name.startswith("AT:")])
    allr = sorted([(st, en, name) for st, en, name, gt in nv
                   if gt == ex_tid and en is not None and en > st and name and not name.startswith("AT:clk")])
    rstarts = [r[0] for r in ranges]
    astarts = [r[0] for r in allr]

    def innermost(x, lst, starts, look=400):
        i = bisect.bisect_right(starts, x)
        best = None
        for j in range(i - 1, max(-1, i - 1 - look), -1):
            st, en, name = lst[j]
            if en >= x and (best is None or st > best[0]):
                best = (st, en, name)
        return best[2] if best else None

    ex_rt = rt_by_tid.get(ex_tid, [])
    ex_starts = [r[0] for r in ex_rt]

    def api_spanning(x):
        """CUDA API call on the executor thread in progress at time x."""
        i = bisect.bisect_right(ex_starts, x)
        for j in range(i - 1, max(-1, i - 30), -1):
            st, en, nm = ex_rt[j]
            if en >= x:
                return (st, en, nm)
        return None

    def bucket(kind, rng, api):
        r = rng or ""
        if kind == "blocking":
            if "prepare_inputs" in r:
                return "plan_sync_prepare_inputs (attention 13)"
            if "model_forward" in r:
                return "sync_inside_model_forward (attention 13 / other)"
            if "update_requests" in r or "process_previous" in r or "sample" in r:
                return "sampler/update wait (overlap loop, engine 27 / host 26)"
            return "blocking_other"
        if "model_forward" in r:
            return "eager_enqueue_model (idea 19 BCG / 28)"
        if "prepare_inputs" in r or "forward_step" in r:
            return "eager_prepare_inputs_or_step (idea 28 / attention 13)"
        if any(k in r for k in ("update_requests", "process_previous", "sample_async")):
            return "overlap_loop_hold (engine 27 / host 26)"
        if "schedule" in r or "prepare_resources" in r:
            return "schedule/prepare_resources (admission)"
        return None

    # In-flight intervals (any request between fetch and publish), host->ns.
    inflight = []
    for c, f in tr.fetch.items():
        if c in tr.pub:
            inflight.append((ns(f["t"]), ns(tr.pub[c]["t"])))
    inflight.sort()
    inflight_u = []
    for fa, fb in inflight:
        if inflight_u and fa <= inflight_u[-1][1]:
            inflight_u[-1] = (inflight_u[-1][0], max(inflight_u[-1][1], fb))
        else:
            inflight_u.append((fa, fb))

    def in_flight(x):
        for a, b in inflight:
            if a <= x <= b:
                return True
            if a > x:
                break
        return False

    gstarts = [g[0] for g in gpu]
    gpu_by_corr = {g[2]: g for g in gpu}
    fwd_ranges = [r for r in ranges if r[2] == "AT:model_forward"]

    def forward_steps(w0, w1):
        """Per eager model forward in the window: kernels it launched, their
        busy union, first-to-last span, enqueue (NVTX) and in-span idle."""
        out_steps = []
        for st, en, _ in fwd_ranges:
            if not (w0 <= st <= w1):
                continue
            out_steps.append((st, en))
        res = []
        if not out_steps:
            return res
        # Build per-thread runtime list with correlation ids once.
        for st, en in out_steps:
            ops = [gpu_by_corr[c] for c, (rs, re_, tid, nm) in rt_items_sorted_window(st, en)
                   if c in gpu_by_corr]
            if not ops:
                continue
            ops.sort()
            busy, end = 0, None
            for o in ops:
                if end is None or o[0] > end:
                    busy += o[1] - o[0]
                    end = o[1]
                elif o[1] > end:
                    busy += o[1] - end
                    end = o[1]
            span = max(o[1] for o in ops) - ops[0][0]
            res.append({"enqueue_ms": r3((en - st) / 1e6), "kernels": sum(1 for o in ops if o[3] == "K"),
                        "gpu_busy_ms": r3(busy / 1e6), "gpu_span_ms": r3(span / 1e6),
                        "in_span_idle_ms": r3((span - busy) / 1e6),
                        "gpu_end_after_enqueue_ms": r3((max(o[1] for o in ops) - en) / 1e6)})
        return res

    ex_corr = sorted((rs, c) for c, (rs, re_, tid, nm) in rt.items() if tid == ex_tid)
    ex_corr_starts = [x[0] for x in ex_corr]

    def rt_items_sorted_window(st, en):
        i = bisect.bisect_left(ex_corr_starts, st)
        j = bisect.bisect_right(ex_corr_starts, en)
        return [(c, rt[c]) for _, c in ex_corr[i:j]]

    cases = {}
    for name, case in results.items():
        w0, w1 = ns(case["t0"]), ns(case["t1"])
        i0 = bisect.bisect_left(gstarts, w0)
        i1 = bisect.bisect_right(gstarts, w1)
        ops = gpu[i0:i1]
        if not ops:
            cases[name] = {"error": "no GPU ops in window"}
            continue
        span0, span1 = ops[0][0], max(o[1] for o in ops)
        busy, gaps, end = 0, [], None
        for st, en, corr, kind, nm in ops:
            if end is None:
                busy += en - st
                end = en
            elif st > end:
                gaps.append((end, st, corr))
                busy += en - st
                end = en
            elif en > end:
                busy += en - end
                end = en
        span = span1 - span0
        nreq = len(case["requests"])
        n_launch = sum(1 for c_, (st, en, tid, nm) in rt.items()
                       if w0 <= st <= w1 and "LaunchKernel" in nm)
        n_kern = sum(1 for o in ops if o[3] == "K")
        cat = defaultdict(float)
        cat_n = defaultdict(int)
        big = []
        small_total = 0.0
        inter = defaultdict(float)
        # Requests of this case with their host timeline (ns), for inter-request splits.
        req_tl = []
        for rq in case["requests"]:
            fe = tr.fe_head.get(rq.get("head")) if rq else None
            if fe is None or "t_send" not in rq:
                continue
            cid = fe["cid"]
            if cid not in tr.fetch:
                continue
            req_tl.append((ns(tr.fetch[cid]["t"]), [
                ("client_turnaround", None), ("http_receive", ns(rq["t_send"])),
                ("tokenization", ns(fe["entry"])), ("ipc_to_executor", ns(fe["gen0"])),
                ("queue_wait", ns(tr.w1[cid]["t"]) if cid in tr.w1 else None)]))
        req_tl.sort(key=lambda x: x[0])
        fstarts = [x[0] for x in req_tl]
        op_starts = [o[0] for o in ops]

        def pieces(a, b):
            """Split [a,b] into (lo, hi, in_flight) pieces by the in-flight union."""
            out_p, cur = [], a
            for fa, fb in inflight_u:
                if fb <= cur:
                    continue
                if fa >= b:
                    break
                if fa > cur:
                    out_p.append((cur, min(fa, b), False))
                lo = max(fa, cur)
                hi = min(fb, b)
                if hi > lo:
                    out_p.append((lo, hi, True))
                cur = max(cur, hi)
                if cur >= b:
                    break
            if cur < b:
                out_p.append((cur, b, False))
            return out_p

        for a, b, corr in gaps:
            d = b - a
            if d < 20e3:
                small_total += d
                continue
            for lo, hi, infl in pieces(a, b):
                pd = hi - lo
                api, where, detail = None, None, None
                if not infl:
                    kind, bk = "inter_request", "inter_request (HTTP/tokenize/IPC/admission)"
                    j = bisect.bisect_left(fstarts, lo)
                    if j < len(req_tl):
                        pts = [(k, v) for k, v in req_tl[j][1]]
                        pts[0] = ("client_turnaround", lo)
                        pts.append(("end", req_tl[j][0]))
                        pts = [(k, v) for k, v in pts if v is not None]
                        for q in range(len(pts) - 1):
                            x0, x1 = max(pts[q][1], lo), min(pts[q + 1][1], hi)
                            if x1 > x0:
                                inter[pts[q][0]] += x1 - x0
                        if hi > req_tl[j][0]:
                            inter["after_fetch"] += hi - max(lo, req_tl[j][0])
                    else:
                        inter["no_next_request"] += pd
                else:
                    api = api_spanning(lo + 1e3)
                    kind = "launch_starved"
                    if api is not None and any(w in api[2] for w in SYNC_WORDS) and api[0] < lo and api[1] >= lo:
                        kind = "blocking"
                        where = innermost(api[0], ranges, rstarts)
                        detail = innermost(api[0], allr, astarts)
                    else:
                        mid = (lo + hi) / 2
                        where = innermost(mid, ranges, rstarts)
                        detail = innermost(mid, allr, astarts)
                    bk = bucket(kind, where, api) or "loop_between_stages (other)"
                full = f"{kind}:{bk}"
                cat[full] += pd
                cat_n[full] += 1
                if pd > 0.3e6:
                    k2 = bisect.bisect_left(op_starts, b)
                    big.append({"ms": r3(pd / 1e6), "kind": kind, "bucket": bk, "at_range": where,
                                "nvtx": detail, "api": api[2] if api else None,
                                "next_op": ops[k2][4][:50] if k2 < len(ops) else None})
        idle = span - busy
        ls = sum(v for k, v in cat.items() if k.startswith("launch_starved"))
        bl = sum(v for k, v in cat.items() if k.startswith("blocking"))
        ir = sum(v for k, v in cat.items() if k.startswith("inter_request"))
        cases[name] = {
            "requests": nreq, "span_ms": r3(span / 1e6), "busy_ms": r3(busy / 1e6), "idle_ms": r3(idle / 1e6),
            "idle_share": r3(idle / span) if span else None,
            "busy_ms_per_request": r3(busy / 1e6 / nreq), "idle_ms_per_request": r3(idle / 1e6 / nreq),
            "launch_starved_ms_per_request": r3(ls / 1e6 / nreq), "blocking_ms_per_request": r3(bl / 1e6 / nreq),
            "inter_request_ms_per_request": r3(ir / 1e6 / nreq),
            "small_gaps_lt20us_ms_per_request": r3(small_total / 1e6 / nreq),
            "by_mechanism_ms_per_request": {k: r3(v / 1e6 / nreq) for k, v in sorted(cat.items(), key=lambda x: -x[1])},
            "by_mechanism_count": dict(cat_n),
            "inter_request_split_ms_per_request": {k: r3(v / 1e6 / nreq) for k, v in inter.items()},
            "kernel_records": n_kern, "launch_api_calls": n_launch,
            "gaps_over_0p3ms": len(big), "largest_gaps": sorted(big, key=lambda x: -x["ms"])[:12],
        }
        fs = forward_steps(w0, w1)
        cases[name]["forward_steps"] = fs[:12]
        if fs:
            cases[name]["forward_median"] = {k: r3(med([s[k] for s in fs])) for k in fs[0]}
        # Big D2H/H2D copies in the window (host-tier KV offload signature).
        cp = [o for o in ops if o[3] == "C"]
        cases[name]["memcpy_ms_per_request"] = r3(sum(o[1] - o[0] for o in cp) / 1e6 / nreq)
    out["cases"] = cases
    out["executor_tid"] = ex_tid
    return out


def compact(summary):
    """Drop per-step lists so the final JSON line stays small."""
    def strip(d, depth=0):
        if isinstance(d, dict):
            return {k: strip(v, depth + 1) for k, v in d.items() if k not in ("steps", "missing_detail")}
        return d
    return strip(summary)
