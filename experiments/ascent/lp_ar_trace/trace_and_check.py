"""One 4-GPU worker job (top lead ruling: at most one queued 4-GPU job per branch): runs
(1) experiments/ascent/lp_ar_trace/trace.py <checkpoint>  (in-server EXTEND/DECODE kernel split, acceptance 1)
(2) experiments/ascent/lp_ar/bench.py sweep              (every-iteration error, 1000-call run with IPC flag
    inspection before every reset barrier, routing; acceptance 3)
one after the other in fresh processes. Last stdout line: {"passed": ..., "trace": {...}, "check": {...}}.
"""
import json
import os
import subprocess
import sys
import time


def run(argv, timeout):
    t0 = time.time()
    print(f"[wrapper] start {argv}", flush=True)
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    last = ""
    try:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.strip():
                last = line.strip()
        p.wait(timeout=max(1, timeout - (time.time() - t0)))
    except subprocess.TimeoutExpired:
        p.kill()
    print(f"[wrapper] done rc={p.returncode} in {time.time()-t0:.0f}s", flush=True)
    try:
        return json.loads(last)
    except Exception:  # noqa: BLE001
        return {"passed": False, "unparsed_last_line": last[:500], "rc": p.returncode}


def main():
    ckpt = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    os.chdir(root)
    trace = run([sys.executable, os.path.join(here, "trace.py"), ckpt], 1500)
    check = run([sys.executable, os.path.join(root, "experiments", "ascent", "lp_ar", "bench.py"), "sweep"], 1000)
    rob = check.get("robust_LP_2048", {})
    out = {
        "passed": bool(trace.get("passed")) and bool(check.get("passed")),
        "trace": {k: trace.get(k) for k in ("passed", "rank0_prefill", "rank0_decode", "rank0_steps",
                                            "profiled_request", "error")},
        "check": {"passed": check.get("passed"), "reasons": check.get("reasons"),
                  "robust": {k: rob.get(k) for k in ("calls_per_rank", "reset_barrier_every", "nonfinite_calls",
                                                     "max_rel_rms", "max_e", "ipc_flags_ok_all_ranks",
                                                     "rerun_bitwise_mismatch", "flag_states_bad")},
                  "err_2048": (check.get("2048") or {}).get("err"),
                  "routing": {m: v.get("LP_bitwise_equal_AUTO_all_ranks")
                              for m, v in (check.get("routing") or {}).items()},
                  "unparsed": check.get("unparsed_last_line")},
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
