"""Idea 143 / branch 81, ruling 5110: ONE job = (b) repeated-execution check matched to bench_pair,
then (a) ncu CLI (SpeedOfLight + MemoryWorkloadAnalysis) on BASE and CAND, reduced to JSON.

(b) `final_check.py check` (BENCH_SHAPE=ep4 | ep2tp2): per forced-routing case, for BASE
    (TRTLLM_SM12X_FI_SPLITK=0, main's unsplit kernel) then CAND (=1): one graph holding a
    standalone fc1 (into q1/sf1, from the case's routed activations) followed by the production
    leaf Sm12xW4a8DecodeState.run; 3 eager warmups, capture, then 5 warm + 40 traced replays,
    each preceded by the 512 MB read-only flush (FLUSH.sum(), as bench_pair).  On EVERY one of
    the 45 replays: q1/sf1/leaf-out vs BASE's replay 0 (bitwise), vs the arm's own replay 0
    (bitwise, determinism), leaf-out vs the FP32 dequantized reference, split-K counters == 0.
    Then, for split cases, 5 more replays with the split-K scratch NaN-poisoned before each:
    must stay bitwise equal to the clean replay.
(a) `final_check.py probe` runs under ncu (driver): EP4, M = 1, forced D = 2, 3, 5, BASE then
    CAND, each launched once between cudaProfilerStart/Stop after the flush; only fc1/fc2
    kernels are profiled (warmup/JIT outside the region = past warmup).
`final_check.py` (no arg) = driver: check ep4, check ep2tp2, ncu probe; prints one JSON line.
"""

import csv
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))
sys.path.insert(0, os.path.join(HERE, "..", "fi_decode_moe_census"))

GRID = 188
SLOT_BYTES = 16 * 256 * 4  # split_k.SLOT_VALUES_PER_THREAD * SLOT_THREADS * fp32


def _setup_shape():
    import common as _c
    if os.environ.get("BENCH_SHAPE", "ep4") == "ep2tp2":
        _c.E, _c.I = 192, 1152
    return _c


def s_rule(T, s_min=4, s_max=4):
    s = 1
    for c in range(s_min, s_max + 1):
        if c * T <= GRID:
            s = c
    return s


def capture(fn):
    import torch
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def build_state(c):
    import torch
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf
    torch.manual_seed(0)
    w1, s1, w2, s2 = c.make_weights()
    w31, s31, w2v, s2s = c.trtllm_layout(w1, s1, w2, s2)
    st = leaf.Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=c.E, slot_start=0, swiglu_limit=10.0, enable_pdl=True)
    return leaf, st, (w1, s1, w2, s2)


def run_check():
    c = _setup_shape()
    import torch
    from census import forced_routing
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils import split_k
    leaf, st, (w1, s1, w2, s2) = build_state(c)
    E, H, I = c.E, c.H, c.I
    shape = os.environ.get("BENCH_SHAPE", "ep4")
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    route, _, fc1, out_sf_shape, _fc2, fi_act = leaf._load_ops()
    scratch, cnt = split_k.workspace(torch.device("cuda"), GRID)
    if shape == "ep4":
        cases = [(1, 1), (1, 2), (4, 1), (4, 2), (4, 3)]  # S=4,4,4,4 and S=1 control
    else:
        cases = [(1, 1), (1, 2), (1, 5), (4, 2), (4, 5), (4, 6)]  # S=4 x5, S=1 control
    if os.environ.get("FINAL_CASES"):  # e.g. "1x5,4x2"
        cases = [tuple(int(v) for v in s.split("x")) for s in os.environ["FINAL_CASES"].split(",")]
    rows, ok = [], True
    for m, d in cases:
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids, w = forced_routing(m, d, 0)
        local = st.local_ids(ids)
        pairs = local.numel()
        ws = leaf._workspace(x, local, E)
        r = route(x, local, w.float(), E, workspace=ws, enable_pdl=True)
        offsets, token_map, rw, a_q, a_sf = [t.clone() for t in r]
        T = int(((offsets[1:] - offsets[:-1] + 7) // 8).sum().item()) * (I // 128)
        S = s_rule(T)
        ref = c.reference(x, ids, w, w1, s1, w2, s2)
        q1 = torch.zeros(pairs, I, dtype=torch.float8_e4m3fn, device="cuda")
        sf1 = torch.zeros(out_sf_shape(pairs, I, E), dtype=torch.int32, device="cuda")
        holder = {}

        def fn():
            fc1(a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False, enable_pdl=True,
                out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=10.0)
            holder["o"] = st.run(x, ids, w)

        base = None
        row = {"shape": shape, "case": f"M{m}_D{d}", "T": T, "S": S}
        for mode in ("0", "1"):
            os.environ["TRTLLM_SM12X_FI_SPLITK"] = mode
            g = capture(fn)
            first = None
            n_cnt_bad = n_det_bad = n_base_q_bad = n_base_o_bad = n_nonfinite = n_qdet_bad = 0
            worst_cd, worst_rel = 0.0, 0.0
            for i in range(45):  # 5 warm + 40 traced, as bench_pair.trace
                flush.sum()
                g.replay()
                torch.cuda.synchronize()
                n_cnt_bad += int(int(cnt.abs().sum().item()) != 0)
                cur = (q1.clone(), sf1.clone(), holder["o"].clone())
                if first is None:
                    first = cur
                if not all(torch.equal(a, b) for a, b in zip(cur, first)):
                    n_det_bad += 1
                if not (torch.equal(cur[0], first[0]) and torch.equal(cur[1], first[1])):
                    n_qdet_bad += 1
                if base is not None:
                    if not (torch.equal(cur[0], base[0]) and torch.equal(cur[1], base[1])):
                        n_base_q_bad += 1
                    if not torch.equal(cur[2], base[2]):
                        n_base_o_bad += 1
                e = c.err_stats(cur[2], ref)
                n_nonfinite += int(not e["finite"])
                worst_cd = max(worst_cd, e["calc_diff"])
                worst_rel = max(worst_rel, e["max_abs_over_ref_absmax"])
            tag = "base" if mode == "0" else "cand"
            row[tag] = {"replays": 45, "counters_nonzero": n_cnt_bad, "nondeterministic": n_det_bad,
                        "fc1_nondeterministic": n_qdet_bad,
                        "nonfinite": n_nonfinite, "ref_calc_diff_max": worst_cd,
                        "ref_maxabs_over_absmax_max": round(worst_rel, 5)}
            if mode == "1":
                row["cand"]["fc1_q1sf1_not_bitwise_vs_base"] = n_base_q_bad
                row["cand"]["out_not_bitwise_vs_base"] = n_base_o_bad
                row["cand"]["q1_maxdiff_vs_base"] = (first[0].float() - base[0].float()).abs().max().item()
                row["cand"]["out_maxdiff_vs_base"] = (first[2].float() - base[2].float()).abs().max().item()
                if S > 1:
                    pbad = pobad = 0
                    pmax = 0.0
                    amax = max(first[2].float().abs().max().item(), 1e-6)
                    for _ in range(5):
                        scratch.fill_(float("nan"))
                        flush.sum()
                        g.replay()
                        torch.cuda.synchronize()
                        same = (torch.equal(q1, first[0]) and torch.equal(sf1, first[1])
                                and int(cnt.abs().sum().item()) == 0
                                and bool(torch.isfinite(holder["o"]).all().item()))
                        pbad += int(not same)
                        pobad += int(not torch.equal(holder["o"], first[2]))
                        pmax = max(pmax, (holder["o"].float() - first[2].float()).abs().max().item() / amax)
                    scratch.zero_()
                    row["cand"]["poison_replays"] = 5
                    # fc1 q1/sf1 bitwise + counters 0 + finite out; leaf out compared separately
                    # (bitwise unless fc2's scatter-add order is nondeterministic in BASE too)
                    row["cand"]["poison_not_bitwise"] = pbad
                    row["cand"]["poison_out_not_bitwise"] = pobad
                    row["cand"]["poison_out_maxdiff_over_absmax"] = pmax
            else:
                base = first
            del g
        b, cd = row["base"], row["cand"]
        good = (b["counters_nonzero"] == 0 and cd["counters_nonzero"] == 0
                and cd["fc1_nondeterministic"] == 0 and cd["nonfinite"] == 0
                and cd["ref_calc_diff_max"] < 2 * b["ref_calc_diff_max"] + 1e-4
                and cd.get("poison_not_bitwise", 0) == 0)
        base_out_det = b["nondeterministic"] == 0
        row["base_out_deterministic"] = base_out_det
        if base_out_det:  # leaf output must then be run-to-run bitwise in CAND too
            good &= cd["nondeterministic"] == 0 and cd.get("poison_out_not_bitwise", 0) == 0
        else:  # fc2 scatter-add order (main's own): out within 1.6e-2 of absmax
            good &= cd.get("poison_out_maxdiff_over_absmax", 0.0) <= 1.6e-2
        if S == 1:
            good &= cd["fc1_q1sf1_not_bitwise_vs_base"] == 0
            if base_out_det:
                good &= cd["out_not_bitwise_vs_base"] == 0
        row["pass"] = bool(good)
        ok &= bool(good)
        rows.append(row)
        c.log("CHECK " + json.dumps(row))
    print(json.dumps({"passed": bool(ok), "rows": rows}))


PROBE_D = (2, 3, 5)


def run_probe():
    c = _setup_shape()
    import torch
    from census import forced_routing
    leaf, st, _ = build_state(c)
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    m = 1
    x = torch.randn(m, c.H, dtype=torch.bfloat16, device="cuda")
    cases = [(mode, d, *forced_routing(m, d, 0)) for mode in ("0", "1") for d in PROBE_D]
    for mode, d, ids, w in cases:
        os.environ["TRTLLM_SM12X_FI_SPLITK"] = mode
        for _ in range(3):
            st.run(x, ids, w)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for mode, d, ids, w in cases:
        os.environ["TRTLLM_SM12X_FI_SPLITK"] = mode
        flush.sum()
        st.run(x, ids, w)
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(json.dumps({"probe_order": [f"{'CAND' if mo == '1' else 'BASE'}/M1_D{d}" for mo, d, *_ in cases]}))


def stream(cmd, env=None, timeout=900):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    last = None
    for line in p.stdout:
        line = line.rstrip("\n")
        if line.startswith("{"):
            try:
                last = json.loads(line)
                print(line[:400], flush=True)
                continue
            except Exception:  # noqa: BLE001
                pass
        print(line[:400], flush=True)
    p.wait(timeout=timeout)
    return p.returncode, last


def find_ncu():
    cands = [shutil.which("ncu")] + sorted(glob.glob("/usr/local/cuda*/bin/ncu")) + \
        sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu")) + sorted(glob.glob("/tmp/ad-*/*/ncu"))
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum",
           "dram__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_sector_hit_rate.pct",
           "gpu__time_duration.sum", "launch__grid_size", "launch__block_size"]


def run_ncu():
    ncu = find_ncu()
    if ncu is None:
        return {"error": "ncu not found"}
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", tempfile.gettempdir())
    rep = os.path.join(out_dir, "final_check_probe")
    cmd = [ncu, "--profile-from-start", "off", "-k", "regex:[Ff][Cc][12]",
           "--section", "SpeedOfLight", "--section", "MemoryWorkloadAnalysis",
           "--metrics", ",".join(METRICS), "-o", rep, "-f",
           sys.executable, os.path.abspath(__file__), "probe"]
    print("NCU CMD: " + " ".join(cmd), flush=True)
    env = dict(os.environ, BENCH_SHAPE="ep4")
    rc, last = stream(cmd, env=env, timeout=1200)
    found = sorted(glob.glob(rep + ".ncu-rep*"))  # ncu 2026.x writes .ncu-repz
    if rc != 0 or not found:
        return {"error": f"ncu rc {rc}, report {found}", "probe": last}
    raw = subprocess.run([ncu, "--import", found[0], "--csv", "--page", "raw",
                          "--metrics", ",".join(METRICS)],
                         capture_output=True, text=True).stdout
    lines = [ln for ln in raw.splitlines() if ln.startswith('"')]
    rdr = list(csv.reader(io.StringIO("\n".join(lines))))
    hdr, units, data = rdr[0], rdr[1], rdr[2:]
    order = (last or {}).get("probe_order", [])
    kern = []
    for i, r in enumerate(data):
        d = dict(zip(hdr, r))
        u = dict(zip(hdr, units))
        name = d.get("Kernel Name", "")
        kind = "fc1" if "fc1" in name.lower() else "fc2"
        rec = {"kernel": kind, "name": name[:80], "case": order[i // 2] if i // 2 < len(order) else None}
        for mname in METRICS:
            v = d.get(mname)
            try:
                v = float(v.replace(",", ""))
            except Exception:  # noqa: BLE001
                pass
            rec[mname] = v
            rec[mname + "_unit"] = u.get(mname)
        rec["grid"] = d.get("Grid Size") or d.get("launch__grid_size")
        kern.append(rec)
    # split-K scratch: S*T slots of SLOT_BYTES, written once and read once by the finishing CTAs
    for k in kern:
        case = k.get("case") or ""
        dd = int(case.split("_D")[-1]) if "_D" in case else 0
        T = 18 * dd  # EP4 fc1, M=1: one 8-row m-block per local expert, 18 tiles each
        S = s_rule(T) if case.startswith("CAND") and k["kernel"] == "fc1" else 1
        k["T"], k["S"] = T, S
        k["scratch_write_bytes"] = S * T * SLOT_BYTES if S > 1 else 0
        k["scratch_read_bytes"] = S * T * SLOT_BYTES if S > 1 else 0
    return {"kernels": kern}


def driver():
    res = {}
    ok = True
    shapes = [s for s in os.environ.get("FINAL_SHAPES", "ep4,ep2tp2").split(",") if s]
    for shape in shapes:
        env = dict(os.environ, BENCH_SHAPE=shape)
        rc, last = stream([sys.executable, os.path.abspath(__file__), "check"], env=env)
        res["check_" + shape] = last
        ok &= rc == 0 and bool(last and last.get("passed"))
    try:
        res["ncu"] = run_ncu()
    except Exception as e:  # noqa: BLE001
        res["ncu"] = {"error": repr(e)[:500]}
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", tempfile.gettempdir())
    with open(os.path.join(out_dir, "final_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    # compact last line
    summ = {"passed": bool(ok), "compared_against": "BASE (TRTLLM_SM12X_FI_SPLITK=0, main's kernel) bitwise and FP32 dequantized reference"}
    for shape in ("ep4", "ep2tp2"):
        rr = (res.get("check_" + shape) or {}).get("rows", [])
        summ["check_" + shape] = [{"case": r["case"], "S": r["S"], "pass": r["pass"],
                                   "cnt_bad": r["cand"]["counters_nonzero"],
                                   "det_bad": r["cand"]["nondeterministic"],
                                   "fc1_det_bad": r["cand"]["fc1_nondeterministic"],
                                   "base_out_det": r["base_out_deterministic"],
                                   "poison_out_maxrel": r["cand"].get("poison_out_maxdiff_over_absmax"),
                                   "q_vs_base_bad": r["cand"]["fc1_q1sf1_not_bitwise_vs_base"],
                                   "out_vs_base_bad": r["cand"]["out_not_bitwise_vs_base"],
                                   "base_det_bad": r["base"]["nondeterministic"],
                                   "cd_base": r["base"]["ref_calc_diff_max"],
                                   "cd_cand": r["cand"]["ref_calc_diff_max"],
                                   "poison_bad": r["cand"].get("poison_not_bitwise")} for r in rr]
    ncu = res["ncu"]
    if "kernels" in ncu:
        summ["ncu"] = [{"case": k["case"], "k": k["kernel"], "grid": k["grid"],
                        "dur": k["gpu__time_duration.sum"], "dur_unit": k["gpu__time_duration.sum_unit"],
                        "dram_rd": k["dram__bytes_read.sum"], "dram_wr": k["dram__bytes_write.sum"],
                        "dram_unit": k["dram__bytes_read.sum_unit"],
                        "dram_pct": k["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
                        "l2_hit": k["lts__t_sector_hit_rate.pct"], "S": k["S"],
                        "scratch_wr_B": k["scratch_write_bytes"], "scratch_rd_B": k["scratch_read_bytes"]}
                       for k in ncu["kernels"]]
    else:
        summ["ncu"] = ncu
    print(json.dumps(summ))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "check":
        run_check()
    elif arg == "probe":
        run_probe()
    else:
        driver()
