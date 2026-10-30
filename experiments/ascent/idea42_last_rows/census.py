"""Idea 42 served nsys launch census: default arm and TRTLLM_FLUX3_LAST_BLOCK_ROWS=0 in ONE job.

Per arm: nsys profile (cuda,nvtx, --cuda-graph-trace=node, capture gated by cudaProfilerApi) around
e2e.py with E2E_SEQ=short (one t2i and one i2i request after the server's warmup) and
TLLM_PROFILE_VISUAL_GEN_START_STOP=<step> (VisualGenProfiler opens one window on that denoise
step of every non-warmup request; --capture-range-end=repeat:2 writes one report per window).
Each report is exported to SQLite. In each step, the attention kernel launches are found; the last
28 are single_blocks[0..27]. seg[j] = kernels after attention launch j-1 up to and including
attention launch j (block j-1's tail + block j's head/QKV/FMHA); tail = kernels after block 27's
attention to the end of the step (block 27's side gate_up, pair, residual, then the final layer).
Printed per arm and case: every launch of seg[26], seg[27] and tail with kernel name, grid, block,
stream and duration; and the comparison of seg[0..27] (ordered and as multisets) between arms.

usage: census.py <checkpoint> [step]
"""

import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("AUTO_DEV_ARTIFACTS") or "/tmp/idea42_census"
os.makedirs(ART, exist_ok=True)
ATTN = re.compile(r"fmha|attention|flash|_attn|attn_", re.I)
N_SINGLE = 28


def log(*a):
    print(f"[census {time.strftime('%H:%M:%S')}]", *a, flush=True)


def loadavg():
    try:
        return open("/proc/loadavg").read().strip()
    except OSError:
        return None


def run_arm(ckpt, name, extra_env, step):
    base = os.path.join(ART, f"census_{name}")
    for f in glob.glob(base + "*"):
        os.remove(f)
    env = dict(os.environ)
    env.update(extra_env)
    env["E2E_SEQ"] = "short"
    env["TLLM_PROFILE_VISUAL_GEN_START_STOP"] = str(step)
    env["AUTO_DEV_ARTIFACTS"] = os.path.join(ART, f"e2e_{name}")
    cmd = ["nsys", "profile", "-o", base, "--force-overwrite=true", "-t", "cuda,nvtx",
           "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node",
           "--capture-range=cudaProfilerApi", "--capture-range-end=repeat:2",
           sys.executable, os.path.join(HERE, "e2e.py"), ckpt]
    log(f"[{name}] env {extra_env}: {' '.join(cmd)}")
    t0 = time.time()
    out_path = os.path.join(ART, f"census_{name}.stdout")
    with open(out_path, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        last = t0
        while proc.poll() is None:
            if time.time() - last > 60:
                last = time.time()
                tail = open(out_path, errors="replace").read().splitlines()[-1:]
                log(f"[{name}] running {time.time() - t0:.0f}s: {tail}")
            time.sleep(5)
    text = open(out_path, errors="replace").read()
    e2e = None
    for ln in reversed(text.splitlines()):
        if ln.startswith("{") and '"digests"' in ln:
            try:
                e2e = json.loads(ln)
            except ValueError:
                pass
            break
    log(f"[{name}] nsys rc={proc.returncode} in {time.time() - t0:.0f}s")
    reps = sorted(glob.glob(base + "*.nsys-rep"), key=os.path.getmtime)
    log(f"[{name}] reports {reps}")
    return reps, e2e, proc.returncode


def export(rep):
    db = rep[:-len(".nsys-rep")] + ".sqlite"
    r = subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, rep],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(db):
        raise RuntimeError(f"nsys export failed {r.returncode}: {r.stdout[-800:]} {r.stderr[-800:]}")
    return db


def kernels(db):
    con = sqlite3.connect(db)
    strings = dict(con.execute("select id, value from StringIds"))
    rows = con.execute("select start, end, shortName, demangledName, gridX, gridY, gridZ, blockX, blockY, "
                       "blockZ, streamId, globalPid from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    con.close()
    out = []
    for s, e, sn, dn, gx, gy, gz, bx, by, bz, st, pid in rows:
        out.append({"name": strings.get(sn) or "?", "full": (strings.get(dn) or "")[:300],
                    "grid": [gx, gy, gz], "block": [bx, by, bz], "stream": st, "pid": pid,
                    "us": round((e - s) / 1e3, 1), "start": s})
    return out


def sig(k):
    return f"{k['name']} g{tuple(k['grid'])} b{tuple(k['block'])}"


def segment(ks):
    attn_idx = [i for i, k in enumerate(ks) if ATTN.search(k["name"])]
    names = Counter(ks[i]["name"] for i in attn_idx)
    if len(attn_idx) < N_SINGLE:
        return None, names
    single = attn_idx[-N_SINGLE:]
    prev = attn_idx[-N_SINGLE - 1] if len(attn_idx) > N_SINGLE else -1
    segs, lo = [], prev + 1
    for i in single:
        segs.append(ks[lo:i + 1])
        lo = i + 1
    return {"segs": segs, "tail": ks[lo:], "pre": ks[:prev + 1], "attn_names": names}, names


def brief(k):
    return {"name": k["name"][:120], "grid": k["grid"], "block": k["block"], "stream": k["stream"],
            "us": k["us"]}


def main():
    ckpt = sys.argv[1]
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    log(f"loadavg start {loadavg()}")
    arms = [("default", {}), ("rows0", {"TRTLLM_FLUX3_LAST_BLOCK_ROWS": "0"})]
    res = {}
    for name, env in arms:
        reps, e2e, rc = run_arm(ckpt, name, env, step)
        cases = {}
        for idx, rep in enumerate(reps[:2]):
            case = ["t2i", "i2i"][idx]
            ks = kernels(export(rep))
            seg, names = segment(ks)
            log(f"[{name}/{case}] {len(ks)} kernels; attention kernels {dict(names)}")
            cases[case] = {"report": rep, "n_kernels": len(ks), "seg": seg, "attn_names": dict(names)}
            with open(os.path.join(ART, f"launches_{name}_{case}.txt"), "w") as fh:
                for i, k in enumerate(ks):
                    fh.write(f"{i}\t{k['us']}\t{k['stream']}\t{k['grid']}\t{k['block']}\t{k['name']}\n")
        res[name] = {"cases": cases, "e2e": e2e, "rc": rc, "reports": reps}
    log(f"loadavg end {loadavg()}")

    summary = {"step": step, "loadavg": loadavg(), "arms": {}}
    ok = True
    for name in res:
        arm = {"rc": res[name]["rc"], "reports": res[name]["reports"]}
        e2e = res[name]["e2e"] or {}
        rr = (e2e.get("results") or [{}])[0]
        arm["route_lines"] = rr.get("route_lines")
        arm["recompiles_after_ready"] = rr.get("recompiles_after_ready")
        arm["digests"] = rr.get("digests")
        arm["tracebacks"] = rr.get("tracebacks")
        for case, c in res[name]["cases"].items():
            seg = c["seg"]
            if seg is None:
                arm[case] = {"error": "fewer than 28 attention launches", "attn": c["attn_names"]}
                ok = False
                continue
            arm[case] = {
                "n_kernels_step": c["n_kernels"],
                "attention_kernels": c["attn_names"],
                "single_attn_sigs": sorted(Counter(sig(s[-1]) for s in seg["segs"]).items()),
                "block26_attn": brief(seg["segs"][26][-1]),
                "block27_attn": brief(seg["segs"][27][-1]),
                "seg27_head_qkv_attn": [brief(k) for k in seg["segs"][27]],
                "tail_after_block27_attn": [brief(k) for k in seg["tail"]],
                "seg26": [brief(k) for k in seg["segs"][26]],
            }
        summary["arms"][name] = arm
    # Compare the arms: seg[0..27] (block 0..26 entire + block 27 head/QKV/FMHA) and the pre-single part.
    cmp = {}
    for case in ("t2i", "i2i"):
        a = res["default"]["cases"].get(case, {}).get("seg")
        b = res["rows0"]["cases"].get(case, {}).get("seg")
        if not a or not b:
            ok = False
            continue
        diff_ordered = [j for j in range(N_SINGLE) if [sig(k) for k in a["segs"][j]] != [sig(k) for k in b["segs"][j]]]
        diff_multiset = [j for j in range(N_SINGLE)
                         if Counter(sig(k) for k in a["segs"][j]) != Counter(sig(k) for k in b["segs"][j])]
        pre_same = Counter(sig(k) for k in a["pre"]) == Counter(sig(k) for k in b["pre"])
        ta, tb = Counter(sig(k) for k in a["tail"]), Counter(sig(k) for k in b["tail"])
        cmp[case] = {"segs_differing_ordered": diff_ordered, "segs_differing_multiset": diff_multiset,
                     "pre_single_blocks_same_multiset": pre_same,
                     "tail_only_in_default": sorted((ta - tb).items()),
                     "tail_only_in_rows0": sorted((tb - ta).items()),
                     "n_tail_default": len(a["tail"]), "n_tail_rows0": len(b["tail"])}
        if diff_multiset or not pre_same:
            ok = False
    summary["compare_default_vs_rows0"] = cmp
    json.dump(summary, open(os.path.join(ART, "census_summary.json"), "w"), indent=1)
    print(json.dumps({"passed": ok, "compared": "nsys kernel launches (name, grid, block) per single block, "
                      "default vs TRTLLM_FLUX3_LAST_BLOCK_ROWS=0, one served t2i and i2i request, one step",
                      **summary}), flush=True)


if __name__ == "__main__":
    main()
