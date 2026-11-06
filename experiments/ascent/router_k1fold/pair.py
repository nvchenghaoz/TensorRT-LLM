"""One job per arm for idea 313 (queue-bound ruling: everything in one job per decision).

Runs, as subprocesses in this tree: sass.py (SASS identity of the existing dsv4FusedRouterKernel instances), then
bench.py (the 40-layer chain timing), then, when the fold op exists (branch tree), the correctness gates after the
timing: check_kernel.py (bitwise kernel sweep, 8192 tokens per P) and check_module.py (module plumbing, repeated
graph sequence, env arms, skip-wait control). Prints bench.py's JSON with "sass", "checks" and "passed" added.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, timeout):
    print(f"=== {script}", flush=True)
    p = subprocess.Popen([sys.executable, os.path.join(HERE, script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        s = line.strip()
        if s.startswith("{"):
            try:
                last = json.loads(s)
            except json.JSONDecodeError:
                pass
    rc = p.wait(timeout=timeout)
    return rc, last


def main():
    import torch
    import tensorrt_llm  # noqa: F401
    has_fold = hasattr(torch.ops.trtllm, "dsv4_fused_router_k1fold")
    only = sys.argv[1] if len(sys.argv) > 1 else "all"  # all | timing (sass + bench) | checks
    if only == "checks":
        rc_k, ck = run("check_kernel.py", 1800)
        rc_m, cm = run("check_module.py", 2400)
        ok = bool(ck and ck.get("passed")) and bool(cm and cm.get("passed"))
        print(json.dumps({"passed": ok, "compared": "fold vs main K1 -> router (kernel sweep) and module plumbing "
                          "vs eager mHC.fused_hc_lagged + dsv4_fused_router",
                          "kernel": {"rc": rc_k, "passed": bool(ck and ck.get("passed"))},
                          "module": cm}))
        return 0 if ok else 1
    if only == "timing":
        has_fold = False  # the gates run in their own job
    rc_s, sass = run("sass.py", 1200)
    rc_b, bench = run("bench.py", 1800)
    out = dict(bench or {"batch": {}, "output": {"class": "exact"}})
    out["sass"] = {"rc": rc_s, "existing_digest": (sass or {}).get("existing_digest"),
                   "n_existing": (sass or {}).get("n_existing"),
                   "instances": {k.split(":", 1)[1]: v for k, v in ((sass or {}).get("instances") or {}).items()}}
    checks = {}
    passed = rc_b == 0 and bool(bench) and bench.get("checks_passed", True)
    if has_fold:
        rc_k, ck = run("check_kernel.py", 1800)
        rc_m, cm = run("check_module.py", 2400)
        checks = {"kernel": {"rc": rc_k, "passed": bool(ck and ck.get("passed"))},
                  "module": {"rc": rc_m, "passed": bool(cm and cm.get("passed")),
                             "failures": (cm or {}).get("failures"),
                             "skip_wait_bad_iterations": (cm or {}).get("skip_wait_bad_iterations"),
                             "hashes": (cm or {}).get("hashes")}}
        passed = passed and checks["kernel"]["passed"] and checks["module"]["passed"]
    out["checks"] = checks
    out["passed"] = bool(passed)
    print(json.dumps(out))
    # A failing gate is reported in "passed"; the job fails only if the timing itself did not run.
    return 0 if (rc_b == 0 and bench) else 1


if __name__ == "__main__":
    sys.exit(main())
