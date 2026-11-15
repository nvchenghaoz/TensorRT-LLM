"""One-job driver for analysis idea 10: runs fa_census.py configurations sequentially
(each in its own process so a CUDA fault cannot poison the next) and prints a final
JSON summary. Usage: python fa_combined.py --ckpt <checkpoint>
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, "fa_census.py")

BIG = "1x2000,4x2000,16x512,1x8000"
ALL = "1x300,16x300,27x300,1x1000,8x1000,1x2000,4x2000,16x512,1x4000,2x4000,1x8000"

RUNS = [
    # name, env, args
    ("config_dump", {"FI_PF_DEBUG": "1"},
     ["--variants", "fa2,fa2@cq=16,fa2@cq=128", "--cases", ALL, "--only-run", "x",
      "--no-ref"]),
    ("sliding_tiles", {},
     ["--layers", "sliding", "--variants",
      "fa2,fa2@cq=16,fa2@cq=128,fa2@kv=1,fa2@kv=2,fa2@kv=4,fa2@ctas=1,fa2@ctas=2",
      "--cases", BIG]),
    ("global_tiles", {},
     ["--layers", "global", "--variants", "fa2,fa2@cq=16,fa2@kv=2,fa2@ctas=2",
      "--cases", BIG]),
    ("split_small", {},
     ["--variants", "fa2,fa2@nosplit=1", "--cases", "1x300,4x300,16x300,27x300,1x1000"]),
    ("page64_b12x", {},
     ["--page", "64", "--variants", "fa2,b12x_fp8kv", "--cases", BIG]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = {}
    ok = True
    for name, env, args in RUNS:
        if a.only and name not in a.only.split(","):
            continue
        e = dict(os.environ)
        e.update(env)
        cmd = [sys.executable, CENSUS, "--ckpt", a.ckpt] + args
        print(f"=== {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        p = subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        recs = []
        last = None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith("RESULT "):
                try:
                    recs.append(json.loads(line[7:]))
                except Exception:
                    pass
            if line.startswith("{"):
                last = line
        rc = p.wait()
        summ = []
        for r in recs:
            s = {k: r.get(k) for k in ("case", "variant", "median_ms", "p10_ms", "p90_ms",
                                       "tflops", "plan_info", "error")}
            if "repeat" in r:
                s["repeat_mismatches"] = r["repeat"]["mismatches"]
            if "err_vs_fp32ref" in r:
                s["rel_fro"] = r["err_vs_fp32ref"]["rel_fro"]
                s["max_abs"] = r["err_vs_fp32ref"]["max_abs"]
            if "maxabs_vs_main" in r:
                s["maxabs_vs_main"] = r["maxabs_vs_main"]
            if s.get("error"):
                s["error"] = s["error"][:200]
            summ.append(s)
            if r.get("repeat", {}).get("mismatches"):
                ok = False
        out[name] = dict(rc=rc, secs=round(time.time() - t0, 1), results=summ)
        print(f"=== {name} rc={rc} in {time.time() - t0:.0f}s", flush=True)
    print(json.dumps({"passed": ok, "runs": out}))


if __name__ == "__main__":
    main()
