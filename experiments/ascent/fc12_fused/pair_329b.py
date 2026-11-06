"""Idea 329 REFRESH pair (area lead ruling on verdict 10458): fine-grain interleaved bursts.

ONE same-GPU process holds BASE and CAND builds of the fused fc12 decode kernel (compile-key
envs of the same tree):
  BASE = TRTLLM_SM12X_FI_FC12_KACQ=0, TRTLLM_SM12X_FI_FC12_EARLY_W2=0, TRTLLM_SM12X_FI_SPLITK_ACQREL=0
  CAND = all three =1
In main's tree (run_gpu_pair baseline arm) the knobs do not exist, so both arms are main (A/A).

For every rep and every row: a burst of BASE then CAND (order ABAB on even reps, BABA on odd
reps), each burst ~PAIR_BURST graph replays in ONE Kineto session, per-node timing exactly as
bench.trace (8 distinct layers, 512 MB read-only flush before every replay, span = first
route_assign start -> last chain kernel end, / layers).  The per-rep value of an arm = the
MEDIAN of its burst; for rows with several routings (mix seeds / D draws) the burst is split
evenly over the routings and the value = mean over routings of each routing's burst median.
PAIR_WARM_REPS reps are declared up front as warm-up and excluded from the stats; their values
are still printed and kept in the JSON.  No other rep is ever dropped.
Per rep: /proc/loadavg and the SM clock (+ throttle reasons) sampled while the GPU is busy.
Per row and arm: mean, p10, p90, n, (p90-p10)/median and all per-rep values; per row the paired
per-rep diff CAND-BASE with p10/p90.
Last stdout line: {"batch": {...CAND mix rows...}, "rows": [...], "output": {"class": "reordered"}}.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KNOBS = {"KACQ": "TRTLLM_SM12X_FI_FC12_KACQ", "EW2": "TRTLLM_SM12X_FI_FC12_EARLY_W2",
         "ACQREL": "TRTLLM_SM12X_FI_SPLITK_ACQREL"}
REPS = int(os.environ.get("PAIR_REPS", "32"))
WARM = int(os.environ.get("PAIR_WARM_REPS", "2"))
BURST = int(os.environ.get("PAIR_BURST", "64"))
SEEDS = int(os.environ.get("BENCH_SEEDS", "6"))
DSETS = int(os.environ.get("PAIR_DSETS", "4"))
CLOCK_FLOOR = 2200.0


def log(msg):
    print(f"[pair_329b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{cmd[0]} failed: {e!r}>"


def loadavg():
    try:
        with open("/proc/loadavg") as f:
            return f.read().strip()
    except OSError as e:
        return f"<{e!r}>"


def slot_guard(limit_gib=4.0, wait_s=180, poll_s=10):
    """WAIT-ONLY dirty-slot guard (pitfalls 3067/3072): exit 3 before any CUDA allocation if
    this job's GPU stays above limit_gib.  Never kills processes."""
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    t0 = time.time()
    while True:
        out = sh(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"])
        used = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and (not vis or parts[0] in vis or parts[1] in vis):
                try:
                    used[parts[1]] = float(parts[2]) / 1024.0
                except ValueError:
                    pass
        dirty = {k: round(v, 2) for k, v in used.items() if v > limit_gib}
        log(f"slot guard t+{time.time() - t0:.0f}s memory.used GiB { {k: round(v, 2) for k, v in used.items()} }")
        if not dirty:
            return
        if time.time() - t0 >= wait_s:
            log(f"DIRTY SLOT {dirty}")
            print(json.dumps({"passed": False, "error": f"DIRTY SLOT {dirty} (infra, pitfall 3072)"}))
            sys.exit(3)
        time.sleep(poll_s)


def gpu_id():
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    return vis[0] if vis else None


def clock_probe():
    """Start an nvidia-smi SM clock sample in the background (call while the GPU is busy)."""
    cmd = ["nvidia-smi", "--query-gpu=clocks.sm,clocks_throttle_reasons.active,temperature.gpu,"
           "power.draw", "--format=csv,noheader,nounits"]
    if gpu_id():
        cmd.insert(1, f"--id={gpu_id()}")
    try:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except Exception:  # noqa: BLE001
        return None


def clock_read(p):
    if p is None:
        return None, "<no nvidia-smi>"
    try:
        out = p.communicate(timeout=30)[0].strip().splitlines()[0]
        return float(out.split(",")[0]), out
    except Exception as e:  # noqa: BLE001
        return None, f"<{e!r}>"


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
    return {"mean": round(sum(v) / len(v), 4), "p10": round(pct(v, 0.1), 4),
            "p50": round(med, 4), "p90": round(pct(v, 0.9), 4), "n": len(v),
            "spread": round((pct(v, 0.9) - pct(v, 0.1)) / med, 5),
            "values": [round(x, 4) for x in v]}


def main():
    slot_guard()
    import torch

    import bench
    from common12 import H, SHAPE, capture, make_state, routing, uniform_routing
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    log(f"NODE hostname: {sh(['hostname'])}")
    log(f"NODE loadavg at start: {loadavg()}")
    log("NODE nvidia-smi: " + sh(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,"
                                  "clocks.sm,power.draw", "--format=csv"]))
    log("NODE compute-apps: " + sh(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                    "--format=csv"]))
    arms = parse_arms(os.environ.get("PAIR_ARMS", "BASE:;CAND:KACQ,EW2,ACQREL"))
    assert len(arms) == 2, "this script pairs exactly two arms"
    targets = [c for c in os.environ.get("PAIR_TARGETS", "mixM1,M1_D3,M1_D4,M1_D5").split(",") if c]
    default_sec = ",".join(["mixM2", "mixM4"] + [f"M1_D{d}" for d in range(0, 7)]
                           + [f"M2_D{d}" for d in range(4, 10)] + [f"M4_D{d}" for d in range(8, 17)])
    secondary = [c for c in os.environ.get("PAIR_SECONDARY", default_sec).split(",")
                 if c and c not in targets]
    LAYERS = bench.LAYERS
    log(f"PAIR DECLARED (before timing): shape={SHAPE} layout={SHAPE} "
        f"arms={[(n, e) for n, e in arms]} (BASE = KACQ=0, EARLY_W2=0, SPLITK_ACQREL=0 in the same "
        f"process; in main's tree the knobs do not exist and both arms are main) "
        f"TARGETS={targets} SECONDARY={secondary} reps={REPS} warm_reps={WARM} (declared, "
        f"excluded) burst={BURST} replays/arm/rep layers={LAYERS} seeds={SEEDS} dsets={DSETS} "
        f"order=ABAB on even reps, BABA on odd reps; per-rep value = burst median; clock floor "
        f"{CLOCK_FLOOR:.0f} MHz")

    layers = []
    for li in range(LAYERS):
        _, st = make_state(seed=li)
        st.warmup(6)
        layers.append(st)
    torch.cuda.synchronize()
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

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
        return m, [[routing(m, d, seed=li + 97 * k) for li in range(LAYERS)] for k in range(DSETS)]

    from torch.profiler import ProfilerActivity, profile

    kernels = {}

    def burst(name, gl):
        """One Kineto session: per routing, n replays (flush before each). Returns the mean over
        routings of each routing's median per-layer span."""
        n = max(8, BURST // len(gl))
        for g in gl:
            for _ in range(2):
                flush.sum()
                g.replay()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for g in gl:
                for _ in range(n):
                    flush.sum()
                    g.replay()
            torch.cuda.synchronize()
        path = os.path.join(tempfile.gettempdir(), f"p329b_{os.getpid()}.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            tr = json.load(f)
        os.remove(path)
        evs = sorted((e for e in tr.get("traceEvents", [])
                      if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")),
                     key=lambda e: float(e["ts"]))
        reps, cur = [], None
        for e in evs:
            if "reduce_kernel" in e["name"] and float(e["dur"]) > 100.0:
                if cur is not None:
                    reps.append(cur)
                cur = []
            elif cur is not None:
                cur.append(e)
        if cur is not None:
            reps.append(cur)
        assert len(reps) == n * len(gl), f"segmented {len(reps)} replays, expected {n * len(gl)}"
        spans = []
        names = kernels.setdefault(name, set())
        for r in reps:
            st_i = next((i for i, e in enumerate(r) if "route_assign" in e["name"]), None)
            assert st_i is not None, "replay without route_assign"
            r = [e for e in r[st_i:] if "at::native" not in e["name"] and "Memset" not in e["name"]]
            t0 = float(r[0]["ts"])
            spans.append((max(float(e["ts"]) + float(e["dur"]) for e in r) - t0) / LAYERS)
            for e in r:
                names.add(e["name"][:48])
        meds = [pct(spans[i * n:(i + 1) * n], 0.5) for i in range(len(gl))]
        return sum(meds) / len(meds)

    cases = targets + secondary
    graphs = {}
    for case in cases:
        m, rsets = case_routes(case)
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        for name, env in arms:
            os.environ.update(env)
            graphs[(case, name)] = (x, rsets, [capture(lambda: chain(x, routes)) for routes in rsets])
        log(f"captured {case}: {len(rsets)} routings x {len(arms)} arms")
    for v in KNOBS.values():
        os.environ[v] = "0"

    vals = {k: [] for k in graphs}
    warm_vals = {k: [] for k in graphs}
    rep_log = []
    low_clock = 0
    for r in range(-WARM, REPS):
        order = arms if (r % 2 == 0) else arms[::-1]
        la = loadavg()
        probe = None
        t_rep = time.time()
        for ci, case in enumerate(cases):
            line = []
            for name, _ in order:
                if probe is None and ci == 0:
                    probe = clock_probe()  # sampled during the first bursts of the rep
                v = burst(name, graphs[(case, name)][2])
                (vals if r >= 0 else warm_vals)[(case, name)].append(v)
                line.append(f"{name}={v:.3f}")
            log(f"rep {r}{' (warm, excluded)' if r < 0 else ''} layout={SHAPE} {case} "
                f"order={''.join(n[0] for n, _ in order)}: " + " ".join(line))
        clk, raw = clock_read(probe)
        low = clk is not None and clk < CLOCK_FLOOR
        if r >= 0 and low:
            low_clock += 1
        rep_log.append({"rep": r, "warm": r < 0, "loadavg": la, "sm_clock_mhz": clk,
                        "nvsmi": raw, "low_clock": low, "secs": round(time.time() - t_rep, 1)})
        log(f"rep {r} loadavg [{la}] sm_clock {raw}{' BELOW FLOOR' if low else ''} "
            f"({time.time() - t_rep:.1f}s)")

    rows = []
    ref, cand = arms[0][0], arms[1][0]
    for case in cases:
        row = {"case": case, "role": "TARGET" if case in targets else "SECONDARY", "layout": SHAPE}
        for name, _ in arms:
            row[name] = stats(vals[(case, name)])
            row[name]["warm_values"] = [round(x, 4) for x in warm_vals[(case, name)]]
        d = [c - b for c, b in zip(vals[(case, cand)], vals[(case, ref)])]
        bm = row[ref]["mean"]
        row[f"diff_{cand}"] = {"mean_us": round(sum(d) / len(d), 4), "p10_us": round(pct(d, 0.1), 4),
                               "p50_us": round(pct(d, 0.5), 4), "p90_us": round(pct(d, 0.9), 4),
                               "mean_pct": round(100.0 * sum(d) / len(d) / bm, 3),
                               "p10_pct": round(100.0 * pct(d, 0.1) / bm, 3),
                               "p90_pct": round(100.0 * pct(d, 0.9) / bm, 3),
                               "values_us": [round(x, 4) for x in d]}
        rows.append(row)
        log(f"ROW {row['role']} layout={SHAPE} {case}: " + " | ".join(
            f"{n} mean {row[n]['mean']} p10 {row[n]['p10']} p90 {row[n]['p90']} n {row[n]['n']} "
            f"spread {row[n]['spread']}" for n, _ in arms) +
            f" || {cand}-{ref} {row['diff_' + cand]['mean_us']} us ({row['diff_' + cand]['mean_pct']}%) "
            f"p10 {row['diff_' + cand]['p10_pct']}% p90 {row['diff_' + cand]['p90_pct']}%")
    for name, ks in kernels.items():
        log(f"kernels [{name}]: {sorted(ks)}")
    tmax = max(max(rw[n]["spread"] for n, _ in arms) for rw in rows if rw["role"] == "TARGET")
    log(f"SUMMARY max per-arm target spread {tmax:.5f}; reps below {CLOCK_FLOOR:.0f} MHz: "
        f"{low_clock} of {REPS}; loadavg end {loadavg()}")
    batch = {}
    for case in cases:
        if case.startswith("mix"):
            row = next(rw for rw in rows if rw["case"] == case)
            batch[case[4:]] = {"kernel_ms": round(row[cand]["mean"] / 1000.0, 6),
                               "noise_floor": max(row[n]["spread"] for n, _ in arms)}
    print(json.dumps({"batch": batch, "rows": rows, "arms": [n for n, _ in arms], "reference_arm": ref,
                      "reported_arm": cand, "targets": targets, "secondary": secondary,
                      "shape": SHAPE, "layout": SHAPE, "low_clock_reps": low_clock,
                      "max_target_spread": tmax, "reps": rep_log,
                      "output": {"class": "reordered"}}))


if __name__ == "__main__":
    main()
