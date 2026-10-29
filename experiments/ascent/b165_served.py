"""Branch 165 (idea 245) acceptance 4: served-order nsys rows, A/B/A', base (g3 off) vs G3.

Three fresh processes, the pipeline as served (engine.json overlay, torch.compile + autotune,
CUDA graphs on, landing-70 fmha_first fork order = main's default), each under
nsys --cuda-graph-trace=node with capture between cudaProfilerStart/Stop around ONE request:
  A  : TRTLLM_FMHA_V2_FLUX3_G3=0 (main's routed 64-row kernel, byte for byte)
  B  : TRTLLM_FMHA_V2_FLUX3_G3=1 (this branch's default)
  A2 : TRTLLM_FMHA_V2_FLUX3_G3=0 again (drift control)
Each child generates A, A (warm), A (profiled), A and saves latents/images.
The parent pairs every side-stream gate_up (flux3_swiglu_fp4 / CUTLASS grid [33|32,144]) with its
FMHA (fmha_v2_flux3*) and reports per S: FMHA us, gate_up us, start(gate_up)-start(FMHA), overlap,
fork pair span (min start -> max end), the idle gap at the join (next kernel start - pair end),
the unforked FMHAs (image.0) as their own row, and the denoise window. Latents/images are compared
within each process (call j vs call 0) and across arms (info only: separate starts, pitfall 428).
nsys rows SIZE; the untraced same-slot pair decides (top-lead ruling).

usage: python b165_served.py <checkpoint>
"""

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1]
NSYS = "/usr/local/cuda/bin/nsys"
MAIN = "0b4693c4"


def log(*a):
    print(*a, flush=True)


def probe():
    def sh(*cmd):
        p = subprocess.run(list(cmd), capture_output=True, text=True)
        return p.returncode, p.stdout.strip()

    head = sh("git", "rev-parse", "HEAD")[1]
    rc = sh("git", "merge-base", "--is-ancestor", MAIN, "HEAD")[0]
    rc70 = sh("git", "merge-base", "--is-ancestor", "434118cf", "HEAD")[0]
    log(f"PROBE HEAD={head} is-ancestor {MAIN} rc={rc}, 434118cf (landing 70) rc={rc70}")
    log(sh("git", "log", "--oneline", "-3")[1])
    return {"head": head, "ancestor_rc_0b4693c4": rc, "ancestor_rc_landing70": rc70}


def child(out_path):
    import torch

    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine = json.load(open(os.path.join(HERE, "..", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    args = VisualGenArgs(
        model=CKPT,
        pipeline_config=pipeline_config,
        compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
        **engine,
    )
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    log(f"loaded {time.time() - t0:.0f}s G3={os.environ.get('TRTLLM_FMHA_V2_FLUX3_G3')} "
        f"ROUTED_LEVERS={lib.ROUTED_LEVERS} "
        f"ORDER={os.environ.get('TRTLLM_FLUX3_SIDE_GATE_UP_ORDER', '(default)')}")

    def generate(prompt, seed):
        with torch.inference_mode():
            ctx = pipe._encode_prompt(prompt, None)
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            latent, ids = pipe._prepare_t2i(
                ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
            )
            img = pipe._package_output(pipe._decode_latents((latent, ids))).image
        torch.cuda.synchronize()
        return latent.clone().cpu(), img.clone().cpu()

    prompt = ("A red fox standing in fresh snow at sunrise", 11)
    seq = []
    for i in range(4):
        t = time.time()
        if i == 2:
            torch.cuda.profiler.start()
        seq.append(("A", *generate(*prompt)))
        if i == 2:
            torch.cuda.profiler.stop()
        log(f"gen {i}: {time.time() - t:.2f}s")
    torch.save({"seq": seq, "levers": list(lib.ROUTED_LEVERS),
                "counters": dict(getattr(fp, "SIDE_GATE_UP_COUNTERS", {}))}, out_path)


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            name = r.get("Name", "")
            if not name or r.get("Duration (ns)") in (None, ""):
                continue
            try:
                grid = (int(float(r.get("GrdX") or 0)), int(float(r.get("GrdY") or 0)),
                        int(float(r.get("GrdZ") or 0)))
                block = int(float(r.get("BlkX") or 0))
            except ValueError:
                grid, block = (0, 0, 0), 0
            start = int(float(r["Start (ns)"]))
            rows.append((start, start + int(float(r["Duration (ns)"])), name, grid, block))
    rows.sort()
    return rows


def is_gate_up(r):
    return "flux3_swiglu_fp4" in r[2] or (
        "device_kernel" in r[2] and r[3][1] == 144 and r[3][0] in (32, 33))


def is_fmha(r):
    return "fmha_v2_flux3" in r[2]


def med(v):
    return round(statistics.median(v), 2) if v else None


def reduce(rows):
    gate = [r for r in rows if is_gate_up(r)]
    fmha = [r for r in rows if is_fmha(r)]
    starts = [r[0] for r in rows]
    used, pairs = set(), []
    for g in gate:
        best, bi = None, None
        for i, f in enumerate(fmha):
            if i in used:
                continue
            d = abs(g[0] - f[0])
            if best is None or d < best:
                best, bi = d, i
        if bi is None or best > 2_000_000:
            continue
        used.add(bi)
        pairs.append((g, fmha[bi]))
    import bisect

    per_s = {}
    for g, f in pairs:
        s = 4176 if g[3][0] == 33 else 4096
        d = per_s.setdefault(s, {"n": 0, "dstart": [], "overlap": [], "fd": [], "gd": [],
                                 "span": [], "join_idle": [], "gate_after_fmha": [],
                                 "fmha_grid": f[3], "fmha_block": f[4], "fmha_name": f[2]})
        d["n"] += 1
        d["dstart"].append((g[0] - f[0]) / 1e3)
        d["overlap"].append(max(0, min(g[1], f[1]) - max(g[0], f[0])) / 1e3)
        d["fd"].append((f[1] - f[0]) / 1e3)
        d["gd"].append((g[1] - g[0]) / 1e3)
        end = max(g[1], f[1])
        d["span"].append((end - min(g[0], f[0])) / 1e3)
        d["gate_after_fmha"].append(max(0, g[1] - f[1]) / 1e3)
        j = bisect.bisect_left(starts, end)
        if j < len(starts):
            d["join_idle"].append((starts[j] - end) / 1e3)
    out = {}
    for s, d in sorted(per_s.items()):
        out[str(s)] = {
            "forks": d["n"], "fmha_us_median": med(d["fd"]), "fmha_us_total": round(sum(d["fd"]), 1),
            "gate_up_us_median": med(d["gd"]),
            "start_gate_minus_fmha_us_median": med(d["dstart"]),
            "overlap_us_median": med(d["overlap"]),
            "gate_up_past_fmha_end_us_median": med(d["gate_after_fmha"]),
            "pair_span_us_median": med(d["span"]), "pair_span_us_total": round(sum(d["span"]), 1),
            "join_idle_us_median": med(d["join_idle"]),
            "join_idle_us_total": round(sum(d["join_idle"]), 1),
            "fmha_grid": d["fmha_grid"], "fmha_block": d["fmha_block"],
            "fmha_kernel": d["fmha_name"][:120],
        }
    unp = {}
    for i, f in enumerate(fmha):
        if i in used:
            continue
        unp.setdefault(str(f[3]), []).append((f[1] - f[0]) / 1e3)
    fm = [r for r in rows if is_fmha(r)]
    return {
        "kernels": len(rows), "gate_up_count": len(gate), "fmha_count": len(fmha),
        "fmha_us_total_all": round(sum((f[1] - f[0]) / 1e3 for f in fm), 1),
        "pairs": len(pairs), "per_s": out,
        "unpaired_fmha": {k: {"n": len(v), "us_median": med(v), "us_total": round(sum(v), 1)}
                          for k, v in unp.items()},
        "denoise_window_ms_first_to_last_fmha_or_gate_up": round(
            (max(r[1] for r in gate + fmha) - min(r[0] for r in gate + fmha)) / 1e6, 3)
        if gate + fmha else None,
        "gpu_span_ms": round((rows[-1][1] - rows[0][0]) / 1e6, 3) if rows else None,
    }


def parent():
    import torch

    info = probe()
    tmp = os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp()
    arms = (("A", "0"), ("B", "1"), ("A2", "0"))
    res, traces = {}, {}
    for tag, g3 in arms:
        rep = os.path.join(tmp, f"b165_served_{tag}")
        pt = os.path.join(tmp, f"b165_served_{tag}.pt")
        env = dict(os.environ, TRTLLM_FMHA_V2_FLUX3_G3=g3)
        cmd = [NSYS, "profile", "-t", "cuda", "--cuda-graph-trace=node",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-f", "true",
               "-o", rep, sys.executable, __file__, CKPT, "--child", pt]
        log(f"=== {tag}: TRTLLM_FMHA_V2_FLUX3_G3={g3}")
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        log(p.stdout[-2500:])
        log(p.stderr[-1500:])
        if p.returncode != 0 or not os.path.exists(pt):
            print(json.dumps({"passed": False, "error": f"{tag} child rc={p.returncode}"}))
            return
        res[tag] = torch.load(pt)
        s = subprocess.run([NSYS, "stats", "-r", "cuda_gpu_trace", "-f", "csv", "-o", rep,
                            "--force-export=true", rep + ".nsys-rep"],
                           capture_output=True, text=True)
        log(s.stderr[-600:])
        csv_path = rep + "_cuda_gpu_trace.csv"
        traces[tag] = reduce(load_rows(csv_path)) if os.path.exists(csv_path) else {"error": "no csv"}
        traces[tag]["levers"] = res[tag]["levers"]
        traces[tag]["counters"] = res[tag]["counters"]
        log(tag, json.dumps(traces[tag]))

    def within(tag):
        s = res[tag]["seq"]
        return {f"call{j}_vs_call0": bool(torch.equal(s[0][1], s[j][1]) and torch.equal(s[0][2], s[j][2]))
                for j in (1, 2, 3)}

    def across(x, y):
        return {"latent_bitwise": bool(torch.equal(res[x]["seq"][2][1], res[y]["seq"][2][1])),
                "image_bitwise": bool(torch.equal(res[x]["seq"][2][2], res[y]["seq"][2][2]))}

    w = {t: within(t) for t in res}
    rows = {}
    for s in ("4176", "4096"):
        rows[s] = {}
        for t in ("A", "B", "A2"):
            d = traces[t].get("per_s", {}).get(s)
            if d:
                rows[s][t] = {k: d[k] for k in ("forks", "fmha_us_median", "gate_up_us_median",
                                                 "overlap_us_median", "gate_up_past_fmha_end_us_median",
                                                 "pair_span_us_median", "join_idle_us_median",
                                                 "fmha_grid", "fmha_block")}
    passed = all(all(v.values()) for v in w.values()) and info["ancestor_rc_0b4693c4"] == 0
    print(json.dumps({
        "passed": bool(passed), "probe": info,
        "rows_per_S": rows, "traces": traces, "within_process_repeats": w,
        "across_arms_info": {"B_vs_A": across("B", "A"), "A2_vs_A": across("A2", "A")},
        "note": "nsys --cuda-graph-trace=node rows size only; the untraced pair decides. "
                "join_idle = next kernel start (any stream) - max(FMHA end, gate_up end).",
    }), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child(sys.argv[3])
    else:
        parent()
