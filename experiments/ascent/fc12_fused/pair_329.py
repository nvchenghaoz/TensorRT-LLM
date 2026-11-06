"""Idea 329: ONE same-GPU job, BASE and CAND reps interleaved (ABAB...), fused fc12 decode kernel.

Arms are compile-key env settings of the SAME tree, so every build lives in one process:
  KACQ   -> TRTLLM_SM12X_FI_FC12_KACQ      (Stage A: per-SF-cycle fc2 acquire)
  EW2    -> TRTLLM_SM12X_FI_FC12_EARLY_W2  (Stage B: first fc2 tile's weights during the
                                            CTA's own fc1 epilogue)
  ACQREL -> TRTLLM_SM12X_FI_SPLITK_ACQREL  (Stage C: split-K / release fence chain trim)
PAIR_ARMS = "BASE:;CAND:KACQ,EW2,ACQREL" (name:knob,knob;...); the first arm is the reference.
Unlisted knobs are forced to 0, so BASE = main's schedule.  In main's tree (run_gpu_pair
baseline) the knobs do not exist and every arm is main.

PAIR_TARGETS / PAIR_SECONDARY: comma lists of case labels, declared and printed BEFORE any
timing.  Cases: mixM1/mixM2/mixM4 (uniform top-6 busiest-rank mix, BENCH_SEEDS routings, the
row value is the mean over the seeds) and M<m>_D<d> (forced D local experts).
Each rep times every arm once per case (order alternates per rep), Kineto per node like
bench.py (8 distinct layers, 512 MB read-only flush before every replay, p50 of BENCH_REPLAYS).
Per row and arm: mean, p10, p90, n over reps, (p90-p10)/median; per CAND arm: the paired
per-rep difference vs the first arm, mean / p10 / p90 in us/layer and %.
Last stdout line: {"batch": {...CAND...}, "rows": [...], "output": {"class": "reordered"}}.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import bench  # noqa: E402
from common12 import H, SHAPE, capture, log, make_state, routing, uniform_routing  # noqa: E402

KNOBS = {"KACQ": "TRTLLM_SM12X_FI_FC12_KACQ", "EW2": "TRTLLM_SM12X_FI_FC12_EARLY_W2",
         "ACQREL": "TRTLLM_SM12X_FI_SPLITK_ACQREL"}
REPS = int(os.environ.get("PAIR_REPS", "8"))
SEEDS = int(os.environ.get("BENCH_SEEDS", "6"))
LAYERS = bench.LAYERS


def parse_arms(spec):
    arms = []
    for part in spec.split(";"):
        name, _, knobs = part.partition(":")
        on = [k for k in knobs.split(",") if k]
        for k in on:
            assert k in KNOBS, k
        arms.append((name, {v: ("1" if k in on else "0") for k, v in KNOBS.items()}))
    return arms


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def stats(v):
    med = pct(v, 0.5)
    return {"mean": round(sum(v) / len(v), 3), "p10": round(pct(v, 0.1), 3),
            "p90": round(pct(v, 0.9), 3), "n": len(v),
            "spread": round((pct(v, 0.9) - pct(v, 0.1)) / med, 5)}


def node_info():
    for cmd in (["hostname"], ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,clocks.sm,"
                               "clocks.max.sm,temperature.gpu,power.draw", "--format=csv"],
                ["nvidia-smi", "topo", "-m"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
            log(f"NODE {' '.join(cmd[:2])}: {out.strip()}")
        except Exception as e:  # noqa: BLE001
            log(f"NODE {cmd[0]} failed: {e!r}")


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    arms = parse_arms(os.environ.get("PAIR_ARMS", "BASE:;CAND:KACQ,EW2,ACQREL"))
    targets = [c for c in os.environ.get("PAIR_TARGETS", "mixM2,mixM4").split(",") if c]
    secondary = [c for c in os.environ.get("PAIR_SECONDARY", "mixM1,M1_D3,M1_D4,M1_D5").split(",")
                 if c and c not in targets]
    node_info()
    log(f"PAIR DECLARED (before timing): shape={SHAPE} arms={[(n, e) for n, e in arms]} "
        f"TARGETS={targets} SECONDARY={secondary} reps={REPS} replays={bench.REPLAYS} "
        f"layers={LAYERS} seeds={SEEDS}")
    layers = []
    for li in range(LAYERS):
        _, st = make_state(seed=li)
        st.warmup(6)
        layers.append(st)
    torch.cuda.synchronize()
    bench.FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

    def chain(x, routes):
        out = None
        for st, (ids, w) in zip(layers, routes):
            out = st.run(x, ids, w)
        return out

    def case_routes(case):
        if case.startswith("mix"):
            m = int(case[4:])
            return m, [[(a, b) for a, b, _ in (uniform_routing(m, 1000 * s + li)
                                                for li in range(LAYERS))] for s in range(SEEDS)]
        m, d = case.split("_")
        m, d = int(m[1:]), int(d[1:])
        return m, [[routing(m, d, seed=li) for li in range(LAYERS)]]

    cases = targets + secondary
    graphs = {}
    kernels = {}
    for case in cases:
        m, rsets = case_routes(case)
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        for name, env in arms:
            os.environ.update(env)
            gl = []
            for routes in rsets:
                gl.append(capture(lambda: chain(x, routes)))
            graphs[(case, name)] = (x, rsets, gl)
        log(f"captured {case}: {len(rsets)} routings x {len(arms)} arms")
    for v in KNOBS.values():
        os.environ[v] = "0"
    vals = {k: [] for k in graphs}
    for r in range(REPS):
        order = arms if r % 2 == 0 else arms[::-1]
        for case in cases:
            line = []
            for name, _ in order:
                _, _, gl = graphs[(case, name)]
                sp = []
                for g in gl:
                    t = bench.trace(g)
                    sp.append(t["layer_us"])
                    kernels.setdefault(name, set()).update(t["kernels"])
                v = sum(sp) / len(sp)
                vals[(case, name)].append(v)
                line.append(f"{name}={v:.3f}")
            log(f"rep {r} {case}: " + " ".join(line))
    rows = []
    ref = arms[0][0]
    for case in cases:
        row = {"case": case, "role": "TARGET" if case in targets else "SECONDARY"}
        for name, _ in arms:
            row[name] = stats(vals[(case, name)])
        for name, _ in arms[1:]:
            d = [c - b for c, b in zip(vals[(case, name)], vals[(case, ref)])]
            base_mean = row[ref]["mean"]
            row[f"diff_{name}"] = {"mean_us": round(sum(d) / len(d), 3),
                                   "p10_us": round(pct(d, 0.1), 3),
                                   "p90_us": round(pct(d, 0.9), 3),
                                   "mean_pct": round(100.0 * sum(d) / len(d) / base_mean, 3),
                                   "p10_pct": round(100.0 * pct(d, 0.1) / base_mean, 3),
                                   "p90_pct": round(100.0 * pct(d, 0.9) / base_mean, 3)}
        rows.append(row)
        log(f"ROW {row['role']} {case}: " + " | ".join(
            f"{n} mean {row[n]['mean']} p10 {row[n]['p10']} p90 {row[n]['p90']} n {row[n]['n']} "
            f"spread {row[n]['spread']}" for n, _ in arms) + " || " + " | ".join(
            f"{n}-{ref} {row['diff_' + n]['mean_us']} us ({row['diff_' + n]['mean_pct']}%) "
            f"p10 {row['diff_' + n]['p10_pct']}% p90 {row['diff_' + n]['p90_pct']}%"
            for n, _ in arms[1:]))
    for name, ks in kernels.items():
        log(f"kernels [{name}]: {sorted(ks)}")
    cand = arms[-1][0]
    batch = {}
    for case in cases:
        if case.startswith("mix"):
            row = next(r for r in rows if r["case"] == case)
            batch[case[4:]] = {"kernel_ms": round(row[cand]["mean"] / 1000.0, 6),
                               "noise_floor": max(row[n]["spread"] for n, _ in arms)}
    print(json.dumps({"batch": batch, "rows": rows, "arms": [n for n, _ in arms], "reference_arm": ref,
                      "reported_arm": cand, "targets": targets, "secondary": secondary,
                      "shape": SHAPE, "output": {"class": "reordered"}}))


if __name__ == "__main__":
    main()
