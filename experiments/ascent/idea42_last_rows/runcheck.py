"""Run the FROZEN check.py on this tree (worker GPU worktrees lack experiments/ascent-benchmark).

1. Find runner copies of experiments/ascent-benchmark; keep those whose files have the protected
   sizes (read_benchmark) and agree by sha256 with every other such copy; copy one into
   <worktree>/experiments/ascent-benchmark (never edited).
2. Find a harness used by an earlier frozen adapter run (a JSON file holding workload_identity) or
   build it from the workload identity (fact 537/558), then run check.py with it.
The last stdout line is check.py's own JSON result plus the copy provenance.

usage: runcheck.py <checkpoint>
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ROOT = "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507"
LEASES = "/home/scratch.chengzhang/.auto-dev/leases"
SIZES = {"accuracy.py": 5824, "check.py": 10040, "common.py": 38578, "full_model.py": 4353,
         "inputs.json": 9200, "judge.py": 3579, "metrics.py": 2091, "profile_nsys.py": 20322,
         "prompts.json": 19918, "proxy.py": 4418, "reference.json": 10151, "refgen.py": 17048,
         "serve_launcher.py": 8212}


def log(*a):
    print(f"[runcheck {time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def find_copies():
    pats = [f"{ROOT}/runner/slots/*/workspace/*/experiments/ascent-benchmark",
            f"{ROOT}/runner/slots/*/workspace/*/*/experiments/ascent-benchmark",
            f"{ROOT}/runner/slots/*/*/*/experiments/ascent-benchmark",
            f"{ROOT}/runner/slots/*/jobs/*/*/experiments/ascent-benchmark",
            f"{ROOT}/*/experiments/ascent-benchmark", f"{ROOT}/*/*/experiments/ascent-benchmark",
            f"{ROOT}/*/*/*/experiments/ascent-benchmark"]
    found = set()
    for p in pats:
        found.update(d for d in glob.glob(p) if os.path.isdir(d))
    if not found:
        r = subprocess.run(["timeout", "180", "find", ROOT, "-maxdepth", "8", "-type", "d",
                            "-name", "ascent-benchmark", "-path", "*experiments*"],
                           capture_output=True, text=True)
        found.update(ln for ln in r.stdout.split() if ln)
    return sorted(found)


def find_harness():
    roots = glob.glob(f"{LEASES}/*/runner/jobs") + glob.glob(f"{ROOT}/runner/slots/*/jobs")
    r = subprocess.run(["timeout", "240", "grep", "-rl", "--include=*.json", "--include=*.txt",
                        "--include=*.sh", "--include=*cmd*", "--include=*command*", "--include=*.log",
                        "-m1", "workload_identity", *roots], capture_output=True, text=True)
    cands = [ln for ln in r.stdout.splitlines() if ln][:200]
    log(f"files mentioning workload_identity: {len(cands)} {cands[:10]}")
    dec = json.JSONDecoder()
    for path in cands:
        try:
            if os.path.getsize(path) > 4_000_000:
                continue
            text = open(path, errors="replace").read()
        except OSError:
            continue
        i = 0
        while True:
            i = text.find("{", i)
            if i < 0:
                break
            try:
                obj, end = dec.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            stack = [obj]
            while stack:
                o = stack.pop()
                if isinstance(o, dict):
                    wi = o.get("workload_identity")
                    if isinstance(wi, dict) and isinstance(wi.get("workload"), dict) \
                            and wi["workload"].get("cases"):
                        return o, path
                    stack.extend(o.values())
                elif isinstance(o, list):
                    stack.extend(o)
                elif isinstance(o, str) and "workload_identity" in o and o.lstrip().startswith("{"):
                    try:
                        stack.append(json.loads(o))
                    except ValueError:
                        pass
            i = end
    return None, None


def main():
    ckpt = sys.argv[1]
    dst = os.path.join(WT, "experiments", "ascent-benchmark")
    prov = {}
    if not os.path.isdir(dst):
        copies = find_copies()
        log(f"copies found: {copies}")
        good = {}
        for d in copies:
            files = {f: os.path.join(d, f) for f in SIZES}
            if all(os.path.isfile(p) and os.path.getsize(p) == SIZES[f] for f, p in files.items()):
                good[d] = {f: sha(p) for f, p in files.items()}
            else:
                log(f"skip {d}: sizes differ from the protected files")
        if not good:
            raise SystemExit("no runner copy of experiments/ascent-benchmark with the protected sizes")
        groups = {}
        for d, h in good.items():
            groups.setdefault(json.dumps(h, sort_keys=True), []).append(d)
        key, dirs = max(groups.items(), key=lambda kv: len(kv[1]))
        log(f"{len(good)} size-matching copies in {len(groups)} sha256 groups; using {dirs[0]} "
            f"(agreeing copies: {len(dirs)})")
        prov = {"source": dirs[0], "agreeing_copies": len(dirs), "sha_groups": len(groups),
                "sha256": json.loads(key)}
        shutil.copytree(dirs[0], dst, ignore=shutil.ignore_patterns("__pycache__"))
    sys.path.insert(0, dst)
    harness, hpath = find_harness()
    if harness is None:
        log("no harness found; building it from the workload identity")
        import inspect
        import common  # noqa: E402

        log(inspect.getsource(common.workload)[:3000])
        harness = {"workload_identity": {
            "workload": {"kind": "visual-gen", "model": "black-forest-labs/flux-3-image-klein",
                         "variant": "nvfp4", "num_inference_steps": 16, "concurrency": [1],
                         "tensor_parallel": 1, "cases": ["t2i-2048", "i2i-2048"]},
            "resolved_model": {"checkpoint": ckpt, "revision": "337337621b85"}}}
    else:
        harness = dict(harness)
        rm = harness["workload_identity"].get("resolved_model")
        log(f"harness from {hpath}: workload {json.dumps(harness['workload_identity']['workload'])[:1500]}"
            f" resolved_model {rm}")
    prov["harness_source"] = hpath
    log(f"loadavg start {open('/proc/loadavg').read().strip()}")
    proc = subprocess.Popen([sys.executable, os.path.join(dst, "check.py"), json.dumps(harness)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=WT)
    last = None
    for ln in proc.stdout:
        print(ln, end="", flush=True)
        if ln.strip():
            last = ln
    proc.wait()
    log(f"check.py rc={proc.returncode}; loadavg end {open('/proc/loadavg').read().strip()}")
    try:
        result = json.loads(last)
    except (TypeError, ValueError):
        result = {"raw_last_line": (last or "")[-2000:]}
    rep = result.get("repeated_execution") or {}
    comps = result.get("comparisons") or {}
    out = {"passed": bool(result.get("valid") and result.get("repeated_execution_passed")
                          and all(c.get("passed") == c.get("total") for c in comps.values())
                          and len(comps) == 2),
           "repeated_execution_passed": result.get("repeated_execution_passed"),
           "valid": result.get("valid"), "errors": result.get("errors"),
           "comparisons": {k: {"passed": v.get("passed"), "total": v.get("total"),
                               "tolerance": v.get("tolerance")} for k, v in comps.items()},
           "pairs": len(rep.get("pairs") or []), "expected_pairs": rep.get("expected_pairs"),
           "bitwise_equal_pairs": rep.get("bitwise_equal_pairs"),
           "compared": "frozen check.py: per-prompt LPIPS/PSNR vs reference.json, repeats t2i->i2i->t2i->i2i",
           "provenance": prov, "check_artifacts": result.get("artifacts")}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
