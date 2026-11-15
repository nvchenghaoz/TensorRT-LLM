"""One untimed check job for branch 23 on the synced main (single GPU; only bcg_check loads the model).

Runs, each in its own process with its log under $AUTO_DEV_ARTIFACTS:
  synced_test.py <ckpt>   items 5-8 on real _Gemma4GeluQuantMLP modules (chunk band live)
  kernel_test.py <ckpt>   FULL mode: 13 M x seed0/seed1/heavy, pinned ('cutlass', 2)
  correctness.py <ckpt>   bench.py-structured repeated execution, pinned ('cutlass', 2), 11 M
  bcg_check.py <ckpt>     BCG replay/eager/replay, EPI=1 vs EPI=0 (two model loads)
and prints {"passed": ..., per-script summaries} as the last line.
usage: checks.py <checkpoint>
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name, args, tag=None, env=None):
    tag = tag or name
    d = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    os.makedirs(d, exist_ok=True)
    logp = os.path.join(d, f"{tag}.log")
    t0 = time.time()
    print(f"=== {tag} start", flush=True)
    last = None
    with open(logp, "w") as f:
        p = subprocess.Popen([sys.executable, os.path.join(HERE, f"{name}.py"), *args],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env=dict(os.environ, **(env or {})))
        for line in p.stdout:
            f.write(line)
            f.flush()
            if len(line) < 600:
                print(f"[{tag}] {line}", end="", flush=True)
            else:
                print(f"[{tag}] <{len(line)} chars>", flush=True)
            if line.startswith("{"):
                last = line
        rc = p.wait()
    print(f"=== {tag} rc={rc} {time.time() - t0:.0f}s", flush=True)
    try:
        res = json.loads(last) if last else {}
    except json.JSONDecodeError:
        res = {}
    res["_rc"] = rc
    return res


def main():
    ck = sys.argv[1]
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None

    def want(n):
        return only is None or n in only

    skip = {"passed": True, "_rc": 0, "skipped": True}
    # synced_test once per TRTLLM_NVFP4_DENSE_TUNING value (landing 22's dense-bucket picks on / off)
    s = run("synced_test", [ck], env={"TRTLLM_NVFP4_DENSE_TUNING": "1"}) if want("synced_test") else skip
    s0 = (run("synced_test", [ck], tag="synced_test_dense0", env={"TRTLLM_NVFP4_DENSE_TUNING": "0"})
          if want("synced_test") else skip)
    k = run("kernel_test", [ck]) if want("kernel_test") else skip
    c = run("correctness", [ck]) if want("correctness") else skip
    b = run("bcg_check", [ck]) if want("bcg_check") else skip
    out = {
        "passed": bool(s.get("passed") and s0.get("passed") and k.get("passed") and c.get("passed")
                       and b.get("passed") and s["_rc"] == 0 and s0["_rc"] == 0 and k["_rc"] == 0
                       and c["_rc"] == 0 and b["_rc"] == 0),
        "ran": only or ["synced_test", "kernel_test", "correctness", "bcg_check"],
        "synced_test_dense0": {key: s0.get(key) for key in ("passed", "dense_tuning", "picks", "dense_picks",
                                                            "item6_forced", "item7_kill_switch",
                                                            "inputs_unchanged", "tuner_cache_unchanged",
                                                            "path_calls_total", "graph_capture")},
        "synced_item8_dense0": {mode: {"ok": v["ok"], "paths": {m: r["paths"] for m, r in v["rows"].items()},
                                       "bad": {m: [r["bad_vs_first"], r["bad_vs_main"]]
                                               for m, r in v["rows"].items()}}
                                for mode, v in (s0.get("item8_repeated") or {}).items()},
        "bcg_check": {key: b.get(key) for key in ("passed", "error", "comparisons", "steps",
                                                  "epi1_dispatch_in_capture", "epi1_dispatch_eager",
                                                  "epi1_capture_checks",
                                                  "epi1_capture_calls_during_startup", "mlp_permuted",
                                                  "path_calls", "gate_up_picks", "gate_up_picks_same",
                                                  "tuner_cache_sha256", "overlay")},
        "synced_test": {key: s.get(key) for key in ("passed", "dense_tuning", "modules", "picks", "dense_picks",
                                                     "item6_forced", "item7_kill_switch", "inputs_unchanged",
                                                     "tuner_cache_unchanged", "path_calls_total",
                                                     "graph_capture")},
        "synced_item8": {mode: {"ok": v["ok"], "paths": {m: r["paths"] for m, r in v["rows"].items()},
                                "bad": {m: [r["bad_vs_first"], r["bad_vs_main"]] for m, r in v["rows"].items()},
                                "calls": {m: r["calls"] for m, r in v["rows"].items()}}
                         for mode, v in (s.get("item8_repeated") or {}).items()},
        "kernel_test_full": {"passed": k.get("passed"), "n_rows": k.get("n_rows")},
        "correctness": {"passed": c.get("passed"),
                        "datasets": {ds: {"ok": v.get("ok"), "inputs_unchanged": v.get("inputs_unchanged"),
                                          "fused_dispatch_per_M": v.get("fused_dispatch_per_M"),
                                          "bad": {m: [r["bad_vs_first"], r["bad_vs_ref"]]
                                                  for m, r in (v.get("rows") or {}).items()}}
                                     for ds, v in (c.get("datasets") or {}).items()}},
    }
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
