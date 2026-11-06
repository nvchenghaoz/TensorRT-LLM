"""Combine pcie_latency_probe.py part outputs into one compact JSON (last stdout line) plus the sizing lines.
usage: python pcie_probe_summary.py <artifact dir>"""
import json
import os
import sys

art = sys.argv[1]


def load(name):
    p = os.path.join(art, f"probe_{name}.json")
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"passed": False, "missing": repr(e)[:120]}


lat, flo, br, bw, sv = (load(n) for n in ("lat", "floor", "bwread", "bwrest", "served"))
out = {"passed": all(x.get("passed") for x in (lat, flo, br, bw, sv)),
       "parts_passed": {n: x.get("passed") for n, x in zip(("lat", "floor", "bwread", "bwrest", "served"),
                                                          (lat, flo, br, bw, sv))},
       "topo": lat.get("topo")}
# (i)
out["i_oneway_by_class_ns"] = {s: lat.get(s, {}).get("by_class") for s in ("volatile", "release_acquire")}
out["i_oneway_per_pair_med_ns"] = {s: {p: r["oneway_med_ns"] for p, r in lat.get(s, {}).get("per_pair", {}).items()}
                                   for s in ("volatile", "release_acquire")}
# (ii)
out["ii_floor_us"] = {h: {k: {"med": v.get("med_us_max_rank"), "p90": v.get("p90_us_max_rank")}
                          for k, v in flo.get(h, {}).items()} for h in ("loop", "graph")}
# (iii)
tab = sv.get("table_us", {})
out["iii_served_us"] = {k: v.get("max_rank_us") for k, v in tab.items()}
out["iii_derived"] = sv.get("derived")
out["iii_dispatch"] = {m: {"lamport_all_ranks": d.get("lamport_all_ranks"), "nccl_any_rank": d.get("nccl_any_rank")}
                       for m, d in (sv.get("dispatch") or {}).items()}
out["iii_dispatch_kernels_rank0"] = {m: d.get("rank0_kernels") for m, d in (sv.get("dispatch") or {}).items()}
out["iii_checks_bitwise"] = {k: v.get("bitwise_all_ranks") for k, v in (sv.get("checks") or {}).items()}
# (iv)
out["iv_alone_by_class"] = {"read": br.get("alone", {}).get("by_class"), "write_ce": bw.get("alone", {}).get("by_class")}
conc = {}
for src in (br, bw):
    for k, v in (src.get("concurrent") or {}).items():
        conc[k] = {"agg_GBps_at_max_rank": v.get("agg_GBps_at_max_rank"), "max_rank_us": v.get("max_rank_us"),
                   "per_rank_agg_GBps": v.get("per_rank_agg_GBps")}
out["iv_concurrent"] = conc

# ---- sizing lines
lines = []
try:
    for M in (1, 4):
        fl_graph = flo["graph"][f"lamport_M{M}"]["med_us_max_rank"]
        fl_loop = flo["loop"][f"lamport_M{M}"]["med_us_max_rank"]
        ar_chain = tab[f"LOWPRECISION|M{M}|trig_end=False|ar"]["max_rank_us"]
        ar_pred = sv["derived"][f"M{M}|trig_end=False"]["ar_behind_pred_us"]
        for nm, served in (("AR chain", ar_chain), ("AR behind PDL pred", ar_pred)):
            for fn, fl in (("graph floor", fl_graph), ("loop floor", fl_loop)):
                d = served - fl
                lines.append(f"D-AR trim ceiling M={M} (served mode trig_end=False), {nm} {served:.3f} - {fn} "
                             f"{fl:.3f} = {d:.3f} us/call x 80 = {d * 80 / 1e3:.4f} ms/step")
    fl1 = flo["graph"]["lamport_M1"]["med_us_max_rank"]
    lines.append(f"floor-reachable = 3.2 - {fl1:.3f} (graph floor M=1) = {3.2 - fl1:.3f} us x 80 = "
                 f"{(3.2 - fl1) * 80 / 1e3:.4f} ms/step")
except Exception as e:  # noqa: BLE001
    lines.append(f"D-AR lines unavailable: {e!r}"[:200])
try:
    best = max(conc.items(), key=lambda kv: kv[1]["agg_GBps_at_max_rank"] or 0)
    rd = max((v["agg_GBps_at_max_rank"] for k, v in conc.items() if k.startswith("read")), default=None)
    t = 15.73e6 / (best[1]["agg_GBps_at_max_rank"] * 1e3) + 20.0
    lines.append(f"P-AR: best concurrent mode {best[0]} {best[1]['agg_GBps_at_max_rank']} GB/s/rank -> FP8 two-shot "
                 f"15.7 MB / that + 20 us = {t:.1f} us vs LP 478.9 us: {478.9 - t:.1f} us/call x 40 = "
                 f"{(478.9 - t) * 40 / 1e3:.2f} ms/req; c4 TTFT x3.1 = {(478.9 - t) * 40 / 1e3 * 3.1:.1f} ms")
    if rd:
        lines.append(f"P-AR: best concurrent read {rd} GB/s; best/read = {best[1]['agg_GBps_at_max_rank'] / rd:.3f}")
except Exception as e:  # noqa: BLE001
    lines.append(f"P-AR lines unavailable: {e!r}"[:200])
out["sizing_lines"] = lines
for ln in lines:
    print(ln, flush=True)
with open(os.path.join(art, "probe_summary.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out), flush=True)
