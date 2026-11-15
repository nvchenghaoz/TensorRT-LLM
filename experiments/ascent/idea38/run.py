"""Idea 38 driver: runs bench.py arms as separate processes (own workspace bases), then compares
their saved first-call outputs against each other and an FP32 reference.

  --arms main,v1,v1b      'main' = the job's tree (pin or integrated) at a fresh base;
                          'pin' = overlay that reverses v1 (main's op on an integrated tree).
  --rounds 2              round 2 reuses each arm's base (no rebuild) for an interleaved retime.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def log(*a):
    print(*a, flush=True)


def run_arm(arm, a, base, out_dir):
    bench_arm = "tree" if arm == "main" else arm
    cmd = [sys.executable, os.path.join(HERE, "bench.py"), "--arm", bench_arm, "--ckpt", a.ckpt,
           "--cases", a.cases, "--warmup", str(a.warmup), "--iters", str(a.iters),
           "--base", base]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    log("SPAWN", arm, " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        line = line.rstrip("\n")
        print(f"[{arm}] {line}", flush=True)
        if line.startswith("{"):
            last = line
    rc = p.wait()
    log("ARM_DONE", arm, "rc", rc, "wall_s", round(time.time() - t0, 1))
    if rc != 0 or last is None:
        return None
    # rename the saved outputs to the driver's arm name
    if out_dir and bench_arm != arm:
        for f in os.listdir(out_dir):
            if f.startswith(bench_arm + "_"):
                os.rename(os.path.join(out_dir, f),
                          os.path.join(out_dir, arm + f[len(bench_arm):]))
    return json.loads(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arms", default="main,v1,v1b")
    ap.add_argument("--cases", default="1x300,16x300,27x300,1x1000,8x1000,1x2000,4x2000,"
                    "16x512,1x4000,2x4000,1x8000")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ref-arm", default="main")
    a = ap.parse_args()
    log("DRIVER load", os.getloadavg())
    arms = a.arms.split(",")
    out_dir = tempfile.mkdtemp(prefix="idea38_out_")
    bases = {arm: tempfile.mkdtemp(prefix=f"idea38_{arm}_ws_") for arm in arms}
    res = {arm: [] for arm in arms}
    for r in range(a.rounds):
        for arm in arms:
            j = run_arm(arm, a, bases[arm], out_dir if r == 0 else "")
            res[arm].append(j)

    # cross-arm numerics
    import torch
    import fa_census as fc
    sys.path.insert(0, HERE)
    import cases as cz
    dims = fc.load_dims(a.ckpt)
    numer = {}
    passed = all(all(j is not None and j.get("passed") for j in res[arm]) for arm in arms)
    for cs in a.cases.split(","):
        c = cz.make_case(cs, dims)
        ref = c.reference()
        outs = {}
        for arm in arms:
            pth = os.path.join(out_dir, f"{arm}_{cs}.pt")
            if os.path.exists(pth):
                outs[arm] = torch.load(pth).cuda()
        row = {}
        m = outs.get(a.ref_arm)
        for arm, o in outs.items():
            e = fc.err_stats(o, ref)
            d = dict(max_abs_ref=e["max_abs"], rel_fro_ref=e["rel_fro"], finite=e["finite"])
            if m is not None and arm != a.ref_arm:
                d["bitwise_vs_main"] = cz.bytes_equal(o, m)
                d["maxabs_vs_main"] = float((o.float() - m.float()).abs().max())
                em = fc.err_stats(m, ref)
                # protocol gate: <= 1.10x main's error + 1 bf16 ulp at output magnitude
                ulp = float(ref.abs().max()) * 2 ** -7
                d["gate_ok"] = bool(e["max_abs"] <= 1.10 * em["max_abs"] + ulp and
                                    e["rel_fro"] <= 1.10 * em["rel_fro"] and
                                    e["finite"])
                passed &= d["gate_ok"]
            row[arm] = d
        numer[cs] = row
        log("NUMERICS", cs, json.dumps(row))
        del c, ref, outs, m
        torch.cuda.empty_cache()

    # timing table: per round median/p10/p90
    table = {}
    for cs in a.cases.split(","):
        table[cs] = {}
        for arm in arms:
            table[cs][arm] = [None if j is None else j["batch"].get(cs) for j in res[arm]]
    log("TABLE case | " + " | ".join(f"{arm} med(p10,p90) r{r}" for arm in arms
                                     for r in range(a.rounds)))
    for cs, row in table.items():
        cells = []
        for arm in arms:
            for t in row[arm]:
                cells.append("-" if t is None else
                             f"{t['kernel_ms']:.4f}({t['p10_ms']:.4f},{t['p90_ms']:.4f})")
        log("TABLE", cs, "|", " | ".join(cells))
    log("DRIVER END load", os.getloadavg())
    print(json.dumps({"passed": passed, "table": table, "numerics": numer}))


if __name__ == "__main__":
    main()
