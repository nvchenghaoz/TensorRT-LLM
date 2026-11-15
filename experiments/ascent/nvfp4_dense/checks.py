"""Idea 9 / branch 30: every correctness check in ONE job (area-lead queue hold).

Runs, in order, each in its own process on this GPU:
 1. census   census.py: gate 0 on the shared real captures (layers 0/30/59) + synthetic
             seeds 1/2/heavy-tailed; main's and the dense pick in 2 extra processes.
 2. repeat   bench.py --mode check: untimed repeated sequence with the timing run's
             warmup+iteration count per case, shape order A,B,A, every call int16 vs
             call 1 and vs main's pick (switch OFF, same process), FP32 reference,
             inputs unchanged.
 3. cache    bench.py --mode cache: tune+save the autotuner cache in one process, load
             it in another; picks and output int16 hashes must match.
 4. capture  capture_check.py: op-level torch.cuda.graph capture (sync-debug warn mode),
             allocations equal to main's, 25 replays int16 vs eager and vs main.
 5. llm_ab   llm_ab.py (no recapture): in-process A/X/A' over 100 own prompts plus the
             same-process re-send, switch ON and OFF.
 6. bcg      bcg_inproc.py: BCG replay vs eager in one process, candidate ON, with the
             dispatch census inside BreakableCUDAGraphRunner.capture.
The last stdout line is {"passed": ...} over the steps that ran.

usage: checks.py <checkpoint> <capture_dir> [comma-separated steps to run; default all]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ("census", "repeat", "cache", "capture", "llm_ab", "bcg")


def run(cmd, tag):
    print(f"[checks] running {tag}: {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        sys.stdout.write(f"[{tag}] {line}")
        sys.stdout.flush()
        if line.strip().startswith("{"):
            last = line.strip()
    p.wait()
    try:
        return json.loads(last)
    except Exception:  # noqa: BLE001
        return {"passed": False, "error": f"rc={p.returncode}, no JSON"}


def main():
    ck, cap = sys.argv[1], sys.argv[2]
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else set(STEPS)
    py = sys.executable
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    out = {}
    if "census" in only:
        out["census"] = run([py, os.path.join(HERE, "census.py"), "--checkpoint", ck, "--captures", cap], "census")
    if "repeat" in only:
        out["repeat"] = run([py, os.path.join(HERE, "bench.py"), "--mode", "check", "--checkpoint", ck], "repeat")
    if "cache" in only:
        cache_path = os.path.join(art, "nvfp4_dense_autotuner_cache.json")
        save = run([py, os.path.join(HERE, "bench.py"), "--mode", "cache", "--phase", "save", "--cache-path",
                    cache_path, "--checkpoint", ck], "cache_save")
        load = run([py, os.path.join(HERE, "bench.py"), "--mode", "cache", "--phase", "load", "--cache-path",
                    cache_path, "--checkpoint", ck], "cache_load")
        diffs = [k for k in save if save.get(k) != load.get(k)] if "error" not in save else ["error"]
        out["cache"] = {"passed": not diffs and os.path.exists(cache_path), "diffs": diffs, "n_cases": len(save),
                        "save": save}
    if "capture" in only:
        out["capture"] = run([py, os.path.join(HERE, "capture_check.py"), ck], "capture")
    if "llm_ab" in only:
        out["llm_ab"] = run([py, os.path.join(HERE, "llm_ab.py"), ck, "-"], "llm_ab")
    if "bcg" in only:
        out["bcg"] = run([py, os.path.join(HERE, "bcg_inproc.py"), ck], "bcg")
    summary = {"passed": bool(out) and all(bool(v.get("passed")) for v in out.values()),
               "steps_run": sorted(out), "per_step_passed": {k: bool(v.get("passed")) for k, v in out.items()},
               "compared": "see the per-step 'compared' fields; " + ", ".join(sorted(out))}
    if "census" in out:
        summary["census"] = out["census"]
    if "repeat" in out:
        summary["repeat"] = {k: out["repeat"].get(k) for k in ("passed", "fails", "n_cases",
                                                                "autotuner_cache_mutated", "tune_seconds")}
    if "cache" in out:
        summary["cache"] = {k: out["cache"][k] for k in ("passed", "diffs", "n_cases")}
    if "capture" in out:
        summary["capture"] = {"passed": out["capture"].get("passed"), "error": out["capture"].get("error"),
                              "results": out["capture"].get("results")}
    if "llm_ab" in out:
        summary["llm_ab"] = {k: out["llm_ab"].get(k) for k in ("passed", "n_prompts", "A_vs_A2", "A_vs_X",
                                                               "resend", "dispatch_calls",
                                                               "chunked_gate_up_calls_total")}
    if "bcg" in out:
        b = out["bcg"]
        summary["bcg"] = {k: b.get(k) for k in ("passed", "dense_dispatches_during_capture",
                                                "main_dispatches_during_capture", "capture_hazards")}
        summary["bcg"]["inproc"] = {k: (b.get("inproc") or {}).get(k) for k in
                                    ("passed", "comparisons", "repeat_replay_all_equal_first",
                                     "repeat_replay_equal_eager", "context_steps_per_pass")}
    with open(os.path.join(art, "checks.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
