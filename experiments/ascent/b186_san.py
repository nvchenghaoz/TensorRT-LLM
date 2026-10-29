"""Branch 186 acceptance item 2: racecheck + synccheck on the fixed merged tree, strict wrapper.

Each tool x case runs as its own subprocess (timeout per step), H=2, PYTORCH_NO_CUDA_MEMORY_CACHING=1,
--kernel-name kns=<G3 substring>. A step passes only if rc == 0, no usage / 'Filter ill-formed' text,
the child printed its final JSON, and the tool's summary line reads zero. A timeout is a FAIL.
Filter-match proof: the census (unsanitized) shows the G3 kernel name contains the substring, and
the per-launch wall time of the G3 launch under the tool is compared with the tool-off run.
memcheck is not rerun (job 2175: 'ERROR SUMMARY: 0 errors', rc=0 on the same commit).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NFO = os.path.join(HERE, "b186_nfo.py")
KNS = "kernel_nl_tiled_g3"
HEADS = "2"
TIMEOUT = 240
CASES = ("s4096", "s4100", "s4176", "graph", "chain")
ART = os.environ.get("AUTO_DEV_ARTIFACTS") or "/tmp/b186_san"
os.makedirs(ART, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def sh(*cmd):
    p = subprocess.run(list(cmd), capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def find_cs():
    for c in (shutil.which("compute-sanitizer"), "/usr/local/cuda/bin/compute-sanitizer",
              "/usr/local/cuda/compute-sanitizer/compute-sanitizer"):
        if c and os.path.exists(c):
            return c
    raise SystemExit("compute-sanitizer not found")


def run_step(name, argv):
    env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
    path = os.path.join(ART, f"{name}.log")
    log(f"=== step {name}: {' '.join(argv)}")
    t0 = time.time()
    last = t0
    with open(path, "w") as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                             start_new_session=True)
        timed_out = False
        while p.poll() is None:
            if time.time() - t0 > TIMEOUT:
                timed_out = True
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                break
            time.sleep(2)
            if time.time() - last > 30:
                last = time.time()
                log(f"  [wait] {name} {time.time() - t0:.0f}s")
    wall = time.time() - t0
    text = open(path).read()
    child_json = None
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.startswith("{") and '"mode": "san"' in s:
            try:
                child_json = json.loads(s)
            except Exception:
                pass
            break
    hb = [l.strip() for l in text.splitlines() if l.startswith("[hb]")]
    log(f"  rc={p.returncode} timed_out={timed_out} wall={wall:.1f}s heartbeats={len(hb)}")
    for l in hb[:6]:
        log(f"    {l}")
    if p.returncode != 0 or child_json is None:
        log("  --- tail ---\n" + "\n".join(text.splitlines()[-25:]))
    return dict(rc=p.returncode, timed_out=timed_out, wall_s=round(wall, 1), text=text,
                child=child_json, log=path)


def main():
    cs = find_cs()
    res = {"compute_sanitizer": cs, "version": sh(cs, "--version")[1].splitlines()[-1:],
           "kns": KNS, "heads": int(HEADS), "timeout_s": TIMEOUT,
           "env": "PYTORCH_NO_CUDA_MEMORY_CACHING=1"}
    res["git_log"] = sh("git", "-C", ROOT, "log", "--oneline", "-3")[1]
    for ref in ("69eac46e", "b1758d79", "fa8d9b90"):
        res[f"ancestor_{ref}_rc"] = sh("git", "-C", ROOT, "merge-base", "--is-ancestor", ref,
                                       "HEAD")[0]
    fx = open(os.path.join(ROOT, "tensorrt_llm/_torch/visual_gen/attention/backends/"
                                 "fmha_v2_flux3.py")).read()
    res["marker_nvfp4_out_lever"] = [l.strip() for l in fx.splitlines()
                                     if '_LEVER_FLAGS["nvfp4_out"]' in l][:2]
    log(f"[probe] {json.dumps(res)}")
    failures = []
    py = sys.executable
    base = [py, NFO, "--san", "--heads", HEADS]

    # 1. census (no sanitizer): which kernel each call runs at H=2; substring presence.
    c = run_step("census", base + ["--case", "census"])
    cen = (c["child"] or {}).get("census", {})
    g3_names = sorted({k for v in cen.values() for k in v if KNS in k})
    res["census"] = cen
    res["census_g3_names_with_kns"] = g3_names
    bf16_ok = bool(cen) and all(any(KNS in k for k in v) for n, v in cen.items()
                                if n.startswith("bf16_g3"))
    nfo_no_g3 = all(not any(KNS in k for k in v) for n, v in cen.items() if n.startswith("nfo"))
    res["census_bf16_calls_hit_g3"] = bf16_ok
    res["census_nfo_calls_avoid_g3"] = nfo_no_g3
    if c["rc"] != 0 or not cen or not bf16_ok or not nfo_no_g3:
        failures.append("census")
    log(f"[census] {json.dumps(cen)}")

    # 2. tool-off reference wall times.
    off = {}
    for case in ("s4176", "chain"):
        r = run_step(f"off_{case}", base + ["--case", case])
        off[case] = dict(rc=r["rc"], wall_s=r["wall_s"],
                         launch_wall_s=(r["child"] or {}).get("launch_wall_s"))
        if r["rc"] != 0 or r["child"] is None:
            failures.append(f"off_{case}")
    res["tool_off"] = off

    # 3. racecheck / synccheck per case.
    tools = {
        "racecheck": (["--tool", "racecheck", "--racecheck-report", "all"],
                      re.compile(r"RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, "
                                 r"(\d+) warnings?\)")),
        "synccheck": (["--tool", "synccheck"], re.compile(r"ERROR SUMMARY: (\d+) errors?")),
    }
    steps = {}
    for tool, (flags, rx) in tools.items():
        for case in CASES:
            name = f"{tool}_{case}"
            argv = [cs, *flags, "--kernel-name", f"kns={KNS}", "--report-api-errors", "no",
                    *base, "--case", case]
            r = run_step(name, argv)
            lines = [l.strip() for l in r["text"].splitlines() if rx.search(l)]
            m = rx.search(lines[-1]) if lines else None
            zero = bool(m) and all(int(x) == 0 for x in m.groups())
            bad_text = bool(re.search(r"Filter ill-formed|[Uu]sage:|unrecognised option|"
                                      r"unrecognized option|invalid filter", r["text"]))
            child = r["child"] or {}
            lw = dict(child.get("launch_wall_s") or [])
            row = dict(rc=r["rc"], timed_out=r["timed_out"], wall_s=r["wall_s"],
                       summary_line=lines[-1] if lines else None, summary_zero=zero,
                       usage_or_filter_error=bad_text, child_json=bool(r["child"]),
                       launch_wall_s=child.get("launch_wall_s"))
            ok = (r["rc"] == 0 and not r["timed_out"] and zero and not bad_text and
                  r["child"] is not None)
            if case == "s4176":
                g3_off = dict(off.get("s4176", {}).get("launch_wall_s") or []).get("g3_bf16_S4176")
                row["filter_match"] = dict(g3_sanitized_s=lw.get("g3_bf16_S4176"),
                                           g3_tool_off_s=g3_off,
                                           nfo_unmatched_same_proc_s=lw.get("nfo_S4176"))
            row["passed"] = ok
            if not ok:
                failures.append(name)
            steps[name] = row
            log(f"[{name}] passed={ok} rc={r['rc']} summary={row['summary_line']!r}")
    res["steps"] = steps
    res["memcheck_quoted"] = ("job 2175 (same commit fa8d9b90): memcheck --leak-check no "
                              "--padding 4096 --check-tensor-ops yes, S4096/4100/4176 + 33-call "
                              "chain eager+graph, H=24: 'ERROR SUMMARY: 0 errors', rc=0")
    res["failures"] = failures
    res["passed"] = not failures
    with open(os.path.join(ART, "b186_san_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
