"""Idea 228 Stage 0 reducer for job 1066's kineto traces (EP4 vs EP2xTP2, c1 prefill, 3 requests per window).

Usage: b123_reduce.py <artifacts_dir>   (contains traces/{ep4,ep2tp2}/trtllm-trace-<ts>-<id>-rank-<r>.json)
Each trace id = one profile window (a run); windows are paired in time order (run 0, run 1).
Per window and rank: steps = GPU segments separated by > 60 ms idle (one per request).
AR-interval decomposition (kernel counts are identical across ranks, so the i-th AR on each rank is the same collective):
  compute_r,i = start(AR_i, r) - end(AR_{i-1}, r)   (rank r's own work + gaps between two collectives)
  crit_i      = max_r compute_r,i                    (the collective cannot finish before the last rank arrives)
  ar_min_i    = min_r dur(AR_i, r)                   (the last-arriving rank's AR = its intrinsic transfer time)
  step ~= sum_i crit_i + sum_i ar_min_i (+ head/tail)
Intervals are typed by their kernels: 'moe' if any fused-MoE kernel runs in them, else 'attn/other'.
Per-kernel tables (base name, mean per step) for rank 0 and for the slowest rank (largest non-AR busy).
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

GAP_US = 60000.0
AR_RE = re.compile(r"nccl|allreduce|all_reduce|lamport|oneshot|twoshot|reducescatter|allgather|ar_fusion", re.I)
MOE_RE = re.compile(r"moe|expert|permute|finalize|grouped|expandInputRows|doActivation|computeStrides|"
                    r"blockExpertPrefixSum|globalExpertPrefixSum|mergeExpertPrefixSum|buildExpertMaps|"
                    r"fusedBuildExpertMaps|cutlass.*Grouped|GemmUniversal.*Array", re.I)


def log(*a):
    print(*a, flush=True)


def base_name(n):
    n = n.strip()
    if n.startswith("void "):
        n = n[5:]
    for ch in "<(":
        i = n.find(ch)
        if i > 0:
            n = n[:i]
    return n[:90]


def load(path):
    with open(path) as f:
        data = json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    k = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"
         and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur") is not None]
    k.sort(key=lambda e: e["ts"])
    return k


def segments(kern):
    segs, cur, end = [], [], None
    for e in kern:
        if end is not None and e["ts"] - end > GAP_US:
            segs.append(cur)
            cur = []
        cur.append(e)
        t = e["ts"] + e["dur"]
        end = t if end is None else max(end, t)
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= 500]


def union(iv):
    tot, cs, ce = 0.0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        tot += ce - cs
    return tot


def step_info(seg):
    t0 = seg[0]["ts"]
    t1 = max(e["ts"] + e["dur"] for e in seg)
    ars = [e for e in seg if AR_RE.search(e.get("name", ""))]
    non = [e for e in seg if not AR_RE.search(e.get("name", ""))]
    table = defaultdict(lambda: [0, 0.0])
    for e in seg:
        x = table[base_name(e.get("name", ""))]
        x[0] += 1
        x[1] += e["dur"]
    # intervals between ARs: [prev AR end, AR start]
    ivs = []
    prev_end = t0
    for a in ars:
        inside = [e for e in non if e["ts"] >= prev_end - 1 and e["ts"] < a["ts"]]
        is_moe = any(MOE_RE.search(e.get("name", "")) for e in inside)
        names = []
        for e in inside:
            b = base_name(e.get("name", ""))[-40:]
            if b not in names:
                names.append(b)
        ivs.append({"compute": a["ts"] - prev_end, "ar_dur": a["dur"], "moe": is_moe,
                    "busy": union([(e["ts"], e["ts"] + e["dur"]) for e in inside]),
                    "ar_name": base_name(a.get("name", "")), "names": names[:4], "nk": len(inside)})
        prev_end = a["ts"] + a["dur"]
    tail = t1 - prev_end
    return {"span": t1 - t0, "n": len(seg), "n_ar": len(ars), "ar_sum": sum(a["dur"] for a in ars),
            "non_ar_busy": union([(e["ts"], e["ts"] + e["dur"]) for e in non]),
            "non_ar_sum": sum(e["dur"] for e in non), "table": dict(table), "ivs": ivs, "tail": tail}


def window(files):
    """files: {rank: path} -> per-rank list of step infos."""
    out = {}
    for r, p in sorted(files.items()):
        segs = segments(load(p))
        out[r] = [step_info(s) for s in segs]
        log(f"  rank {r}: {len(segs)} steps, spans {[round(s['span'] / 1e3, 2) for s in out[r]]} ms, "
            f"n_ar {[s['n_ar'] for s in out[r]]}")
    return out


def decompose(win):
    ranks = sorted(win)
    nsteps = min(len(win[r]) for r in ranks)
    res = []
    for s in range(nsteps):
        steps = [win[r][s] for r in ranks]
        n_iv = min(len(x["ivs"]) for x in steps)
        same = len({len(x["ivs"]) for x in steps}) == 1
        d = {"span_r0": steps[0]["span"], "span_max": max(x["span"] for x in steps), "consistent_ar_count": same,
             "crit": 0.0, "r0_compute": 0.0, "ar_min": 0.0, "crit_moe": 0.0, "crit_attn": 0.0,
             "r0_moe": 0.0, "r0_attn": 0.0, "skew_moe": 0.0, "skew_attn": 0.0, "n_moe_iv": 0, "n_attn_iv": 0,
             "ar_min_moe": 0.0, "ar_min_attn": 0.0, "argmax_rank_moe": defaultdict(int),
             "tail_max": max(x["tail"] for x in steps)}
        for i in range(n_iv):
            comp = [x["ivs"][i]["compute"] for x in steps]
            ard = [x["ivs"][i]["ar_dur"] for x in steps]
            moe = steps[0]["ivs"][i]["moe"]
            c, m = max(comp), min(ard)
            d["crit"] += c
            d["r0_compute"] += comp[0]
            d["ar_min"] += m
            key = "moe" if moe else "attn"
            d[f"crit_{key}"] += c
            d[f"r0_{key}"] += comp[0]
            d[f"skew_{key}"] += c - sum(comp) / len(comp)
            d[f"n_{key}_iv"] += 1
            d[f"ar_min_{key}"] += m
            if moe:
                d["argmax_rank_moe"][ranks[comp.index(c)]] += 1
        d["argmax_rank_moe"] = dict(d["argmax_rank_moe"])
        res.append(d)
    keys = [k for k in res[0] if isinstance(res[0][k], float)]
    mean = {k: round(sum(x[k] for x in res) / len(res) / 1e3, 3) for k in keys}  # ms
    mean["consistent_ar_count"] = all(x["consistent_ar_count"] for x in res)
    mean["n_moe_iv"] = res[0]["n_moe_iv"]
    mean["n_attn_iv"] = res[0]["n_attn_iv"]
    mean["argmax_rank_moe"] = [x["argmax_rank_moe"] for x in res]
    return mean


def rank_table(steps):
    n = len(steps)
    t = defaultdict(lambda: [0.0, 0.0])
    for s in steps:
        for k, (c, us) in s["table"].items():
            t[k][0] += c / n
            t[k][1] += us / n
    return t


def diff(ta, tb, top=30):
    rows = []
    for k in set(ta) | set(tb):
        ca, ua = ta.get(k, [0, 0.0])
        cb, ub = tb.get(k, [0, 0.0])
        rows.append([k, round(ca, 1), round(ua, 1), round(cb, 1), round(ub, 1), round(ub - ua, 1)])
    rows.sort(key=lambda r: -abs(r[5]))
    return rows[:top]


def main():
    root = sys.argv[1]
    arms = {}
    for arm in ("ep4", "ep2tp2"):
        byid = defaultdict(dict)
        for p in glob.glob(os.path.join(root, "traces", arm, "*.json")):
            m = re.search(r"trtllm-trace-(\d+)-([0-9a-f]+)-rank-(\d+)", os.path.basename(p))
            byid[(int(m.group(1)), m.group(2))][int(m.group(3))] = p
        arms[arm] = [byid[k] for k in sorted(byid)]
        log(f"{arm}: windows {sorted(byid)}")
    out = {"runs": []}
    for run in range(min(len(arms["ep4"]), len(arms["ep2tp2"]))):
        rr = {"run": run}
        wins = {}
        for arm in ("ep4", "ep2tp2"):
            log(f"run {run} arm {arm}")
            wins[arm] = window(arms[arm][run])
            rr[arm] = {"decomp_ms": decompose(wins[arm]),
                       "per_rank": {r: {"span_ms": round(sum(s["span"] for s in st) / len(st) / 1e3, 3),
                                        "non_ar_busy_ms": round(sum(s["non_ar_busy"] for s in st) / len(st) / 1e3, 3),
                                        "non_ar_sum_ms": round(sum(s["non_ar_sum"] for s in st) / len(st) / 1e3, 3),
                                        "ar_sum_ms": round(sum(s["ar_sum"] for s in st) / len(st) / 1e3, 3),
                                        "kernels": st[0]["n"], "n_ar": st[0]["n_ar"]}
                                    for r, st in wins[arm].items()}}
            log(json.dumps(rr[arm]["decomp_ms"]))
        # per-interval view: mean over steps of crit, per-rank compute and gap (compute - busy)
        per_iv = {}
        for arm in ("ep4", "ep2tp2"):
            ranks = sorted(wins[arm])
            ns = min(len(wins[arm][r]) for r in ranks)
            n_iv = min(len(wins[arm][r][s]["ivs"]) for r in ranks for s in range(ns))
            rows = []
            for i in range(n_iv):
                comp = [sum(wins[arm][r][s]["ivs"][i]["compute"] for s in range(ns)) / ns for r in ranks]
                gap = [sum(wins[arm][r][s]["ivs"][i]["compute"] - wins[arm][r][s]["ivs"][i]["busy"]
                           for s in range(ns)) / ns for r in ranks]
                crit = sum(max(wins[arm][r][s]["ivs"][i]["compute"] for r in ranks) for s in range(ns)) / ns
                iv0 = wins[arm][0][0]["ivs"][i]
                rows.append({"crit": crit, "comp": comp, "gap": gap, "moe": iv0["moe"], "names": iv0["names"],
                             "nk": iv0["nk"], "ar": iv0["ar_name"][:30]})
            per_iv[arm] = rows
        n_iv = min(len(per_iv["ep4"]), len(per_iv["ep2tp2"]))
        ivd = []
        for i in range(n_iv):
            a, b = per_iv["ep4"][i], per_iv["ep2tp2"][i]
            ivd.append({"i": i, "moe": a["moe"], "nk": [a["nk"], b["nk"]], "names": a["names"],
                        "crit_ep4": round(a["crit"], 1), "crit_ep2tp2": round(b["crit"], 1),
                        "d_crit": round(b["crit"] - a["crit"], 1),
                        "comp_ep4": [round(x, 1) for x in a["comp"]], "comp_ep2tp2": [round(x, 1) for x in b["comp"]],
                        "gap_ep4": [round(x, 1) for x in a["gap"]], "gap_ep2tp2": [round(x, 1) for x in b["gap"]]})
        rr["intervals"] = ivd
        for typ in (True, False):
            sel = [x for x in ivd if x["moe"] == typ]
            for half, lo, hi in (("first", 0, n_iv // 2), ("second", n_iv // 2, n_iv)):
                s2 = [x for x in sel if lo <= x["i"] < hi]
                log(f"run {run} {'moe' if typ else 'attn'} intervals {half} half (n={len(s2)}): d_crit "
                    f"{sum(x['d_crit'] for x in s2) / 1e3:.3f} ms; gap sum per rank ep4 "
                    f"{[round(sum(x['gap_ep4'][r] for x in s2) / 1e3, 3) for r in range(4)]} ep2tp2 "
                    f"{[round(sum(x['gap_ep2tp2'][r] for x in s2) / 1e3, 3) for r in range(4)]}")
        log(f"run {run} top intervals by d_crit:")
        for x in sorted(ivd, key=lambda x: -abs(x["d_crit"]))[:14]:
            log("   ", json.dumps(x))
        dd = {k: round(rr["ep2tp2"]["decomp_ms"][k] - rr["ep4"]["decomp_ms"][k], 3)
              for k in rr["ep4"]["decomp_ms"] if isinstance(rr["ep4"]["decomp_ms"][k], float)}
        rr["decomp_delta_ms"] = dd
        log(f"run {run} delta (ms): {json.dumps(dd)}")
        tabs = {}
        for arm in ("ep4", "ep2tp2"):
            slow = max(wins[arm], key=lambda r: rr[arm]["per_rank"][r]["non_ar_busy_ms"])
            tabs[arm] = {"slowest_rank": slow, "r0": rank_table(wins[arm][0]), "slow": rank_table(wins[arm][slow])}
        rr["slowest_rank"] = {a: tabs[a]["slowest_rank"] for a in tabs}
        rr["table_r0"] = diff(tabs["ep4"]["r0"], tabs["ep2tp2"]["r0"])
        rr["table_slowest"] = diff(tabs["ep4"]["slow"], tabs["ep2tp2"]["slow"])
        # per-rank full kernel table, all ranks, for the record
        rr["table_rank"] = {r: diff(rank_table(wins["ep4"][r]), rank_table(wins["ep2tp2"][r]), 20)
                            for r in sorted(set(wins["ep4"]) & set(wins["ep2tp2"]))}
        for name, rows in (("rank0", rr["table_r0"]), ("slowest", rr["table_slowest"])):
            log(f"run {run} {name} kernel diff [name, calls_ep4, us_ep4, calls_ep2tp2, us_ep2tp2, delta]:")
            for row in rows[:25]:
                log("   ", row)
        out["runs"].append(rr)
    art = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
    with open(os.path.join(art, "b123_reduce.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"passed": True, "runs": [{"run": r["run"], "delta_ms": r["decomp_delta_ms"],
                                                  "ep4": r["ep4"]["decomp_ms"], "ep2tp2": r["ep2tp2"]["decomp_ms"]}
                                                 for r in out["runs"]]}))


if __name__ == "__main__":
    main()
