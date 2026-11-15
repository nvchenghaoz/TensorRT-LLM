"""Idea 9 / branch 30: every correctness check in ONE job (area-lead queue hold).

Runs, in order, each in its own process on this GPU:
 1. census.py: gate 0 on the shared real captures (layers 0/30/59) + synthetic
    seeds 1/2/heavy-tailed; main's and the dense pick in 2 extra processes.
 2. bench.py --mode check: untimed repeated sequence with the timing run's
    warmup+iteration count per case, shape order A,B,A, every call int16 vs
    call 1 and vs main's pick (switch OFF, same process), FP32 reference,
    inputs unchanged.
 3. bench.py --mode cache: tune+save the autotuner cache in one process, load
    it in another; picks and output int16 hashes must match.
 4. llm_ab.py (no recapture): in-process A/X/A' over 100 own prompts plus the
    same-process re-send, switch ON and OFF.
The last stdout line is {"passed": ...} over all three.

usage: checks.py <checkpoint> <capture_dir>
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


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
    py = sys.executable
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    out = {}
    out["census"] = run([py, os.path.join(HERE, "census.py"), "--checkpoint", ck, "--captures", cap], "census")
    out["repeat"] = run([py, os.path.join(HERE, "bench.py"), "--mode", "check", "--checkpoint", ck], "repeat")
    cache_path = os.path.join(art, "nvfp4_dense_autotuner_cache.json")
    save = run([py, os.path.join(HERE, "bench.py"), "--mode", "cache", "--phase", "save", "--cache-path",
                cache_path, "--checkpoint", ck], "cache_save")
    load = run([py, os.path.join(HERE, "bench.py"), "--mode", "cache", "--phase", "load", "--cache-path",
                cache_path, "--checkpoint", ck], "cache_load")
    diffs = [k for k in save if save.get(k) != load.get(k)] if "error" not in save else ["error"]
    out["cache_reload"] = {"passed": not diffs and os.path.exists(cache_path), "diffs": diffs,
                           "n_cases": len(save), "save": save}
    # 4. op-level CUDA-graph capture check
    out["capture"] = run([py, os.path.join(HERE, "capture_check.py"), ck], "capture")
    # 5. in-process LLM A/X/A' + re-send (own prompts; no recapture)
    out["llm_ab"] = run([py, os.path.join(HERE, "llm_ab.py"), ck, "-"], "llm_ab")
    # 6. BCG replay vs eager, one process, candidate ON
    out["bcg"] = run([py, os.path.join(HERE, "bcg_inproc.py"), ck], "bcg")
    rep = out["repeat"]
    summary = {
        "passed": bool(out["census"].get("passed") and rep.get("passed") and out["cache_reload"]["passed"]
                       and out["capture"].get("passed") and out["llm_ab"].get("passed")
                       and out["bcg"].get("passed")),
        "capture": {k: out["capture"].get(k) for k in ("passed",)},
        "bcg": {k: out["bcg"].get(k) for k in ("passed", "dense_dispatches_during_capture",
                                               "main_dispatches_during_capture", "capture_hazards")}
        | {"inproc": {k: (out["bcg"].get("inproc") or {}).get(k) for k in
                      ("passed", "comparisons", "repeat_replay_all_equal_first", "repeat_replay_equal_eager",
                       "context_steps_per_pass")}},
        "llm_ab": {k: out["llm_ab"].get(k) for k in ("passed", "n_prompts", "A_vs_A2", "A_vs_X", "resend",
                                                     "dispatch_calls", "chunked_gate_up_calls_total")},
        "compared": "gate 0 census (every offered dense tactic + dense pick vs main's pick, int16, real "
                    "captures L0/30/59 + synthetic seeds 1/2/heavy); repeated sequence vs call 1 and main; "
                    "autotuner cache save/reload picks + output hashes",
        "census": out["census"],
        "repeat": {k: rep.get(k) for k in ("passed", "fails", "n_cases", "autotuner_cache_mutated",
                                           "tune_seconds", "dense_available")},
        "cache_reload": {k: out["cache_reload"][k] for k in ("passed", "diffs", "n_cases")},
    }
    with open(os.path.join(art, "checks.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
