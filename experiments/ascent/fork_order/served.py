"""Idea 234 STEP 0(a) + STEP 3 served order rows + full-forward latent bitwise check.

Three fresh processes, the pipeline as served (engine.json overlay, torch.compile + autotune,
CUDA graphs on), each under nsys --cuda-graph-trace=node with capture between cudaProfilerStart/
Stop around ONE request:
  S : TRTLLM_FLUX3_SIDE_GATE_UP=0 (sequential gate_up after FMHA; solo kernel durations)
  A : mode 3, TRTLLM_FLUX3_SIDE_GATE_UP_ORDER=gate_up_first (main's order, byte for byte)
  B : mode 3, TRTLLM_FLUX3_SIDE_GATE_UP_ORDER=fmha_first (this branch's default)
Each child generates A, A (warm), A (profiled), B (T=160 bucket), A and saves latents/images.
The parent pairs every gate_up (flux3_swiglu_fp4 / CUTLASS grid [33|32,144]) with its FMHA
(fmha_v2_flux3), reports per S (gate_up grid x 33 -> S4176, 32 -> S4096) the share of blocks with
gate_up starting first, start(gate_up) - start(FMHA), overlap, durations, pair span, and the
saved time vs the solo (arm S) durations; and compares latents/images bitwise B vs A vs S.

usage: python served.py <checkpoint>
       python served.py <checkpoint> --child <out.pt>
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
MAIN_AT_START = "1185e8f9"
# Empty-SM probe of the FMHA's last-wave idle, full-GPU-equivalent us per call (fact 1114).
FMHA_TAIL_IDLE_US = {4176: 25.8, 4096: 34.5}


def log(*a):
    print(*a, flush=True)


def probe():
    import tensorrt_llm

    def sh(*cmd):
        p = subprocess.run(list(cmd), capture_output=True, text=True)
        return p.returncode, p.stdout.strip()

    head = sh("git", "rev-parse", "HEAD")[1]
    rc = sh("git", "merge-base", "--is-ancestor", MAIN_AT_START, "HEAD")[0]
    log(f"PROBE HEAD={head} MAIN_AT_START={MAIN_AT_START} "
        f"tensorrt_llm={tensorrt_llm.__file__} merge_base_is_ancestor rc={rc}")
    log(sh("git", "log", "--oneline", "-3")[1])
    return {"head": head, "main_at_start": MAIN_AT_START, "ancestor_rc": rc,
            "tensorrt_llm": tensorrt_llm.__file__}


def child(out_path):
    import torch

    sys.path.insert(0, os.path.dirname(HERE))
    from te_common import prompt_for_bucket

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
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
    log(f"loaded {time.time() - t0:.0f}s SIDE_GATE_UP={os.environ.get('TRTLLM_FLUX3_SIDE_GATE_UP')} "
        f"ORDER={os.environ.get('TRTLLM_FLUX3_SIDE_GATE_UP_ORDER')}")
    te = pipe.text_encoder

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

    prompts = {"A": ("A red fox standing in fresh snow at sunrise", 11),
               "B": (prompt_for_bucket(te, 2), 22)}
    seq = []
    for i, name in enumerate(("A", "A", "A", "B", "A")):
        t = time.time()
        if i == 2:
            torch.cuda.profiler.start()
        seq.append((name, *generate(*prompts[name])))
        if i == 2:
            torch.cuda.profiler.stop()
        log(f"{name}: {time.time() - t:.2f}s")
    torch.save({"seq": seq, "counters": dict(fp.SIDE_GATE_UP_COUNTERS)}, out_path)
    log("CHILD_JSON " + json.dumps({"counters": dict(fp.SIDE_GATE_UP_COUNTERS)}))


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
            except ValueError:
                grid = (0, 0, 0)
            start = int(float(r["Start (ns)"]))
            rows.append((start, start + int(float(r["Duration (ns)"])), name, grid))
    rows.sort()
    return rows


def is_gate_up(r):
    return "flux3_swiglu_fp4" in r[2] or (
        "device_kernel" in r[2] and r[3][1] == 144 and r[3][0] in (32, 33)
    )


def is_fmha(r):
    return "fmha_v2_flux3" in r[2]


def med(v):
    return round(statistics.median(v), 2) if v else None


def reduce(rows):
    gate = [r for r in rows if is_gate_up(r)]
    fmha = [r for r in rows if is_fmha(r)]
    used = set()
    pairs = []
    for g in gate:
        best, bi = None, None
        for i, f in enumerate(fmha):
            if i in used:
                continue
            d = abs(g[0] - f[0])
            if best is None or d < best:
                best, bi = d, i
        if bi is None or best > 2_000_000:  # > 2 ms apart: not the same block
            continue
        used.add(bi)
        pairs.append((g, fmha[bi]))
    unpaired_fmha = [f for i, f in enumerate(fmha) if i not in used]
    per_s = {}
    for g, f in pairs:
        s = 4176 if g[3][0] == 33 else 4096
        d = per_s.setdefault(s, {"n": 0, "gate_first": 0, "dstart": [], "overlap": [], "fd": [],
                                 "gd": [], "span": [], "tail_window": [], "fmha_grid": f[3]})
        d["n"] += 1
        d["gate_first"] += g[0] < f[0]
        d["dstart"].append((g[0] - f[0]) / 1e3)
        d["overlap"].append(max(0, min(g[1], f[1]) - max(g[0], f[0])) / 1e3)
        d["fd"].append((f[1] - f[0]) / 1e3)
        d["gd"].append((g[1] - g[0]) / 1e3)
        d["span"].append((max(g[1], f[1]) - min(g[0], f[0])) / 1e3)
        # Second-starting kernel's run inside the first one's span (FMHA end - gate_up start when
        # FMHA leads; gate_up end - FMHA start when gate_up leads).
        d["tail_window"].append(max(0, min(g[1], f[1]) - max(g[0], f[0])) / 1e3)
    out = {}
    for s, d in sorted(per_s.items()):
        out[str(s)] = {
            "blocks": d["n"], "gate_up_first_share": round(d["gate_first"] / d["n"], 4),
            "start_gate_minus_fmha_us_median": med(d["dstart"]),
            "overlap_us_median": med(d["overlap"]), "overlap_us_total": round(sum(d["overlap"]), 1),
            "fmha_us_median": med(d["fd"]), "gate_up_us_median": med(d["gd"]),
            "pair_span_us_median": med(d["span"]), "pair_span_us_total": round(sum(d["span"]), 1),
            "fmha_end_last_share": round(
                sum(1 for (g, f) in pairs if (4176 if g[3][0] == 33 else 4096) == s and f[1] >= g[1])
                / d["n"], 4),
            "fmha_grid": d["fmha_grid"],
        }
    unp = {}
    for f in unpaired_fmha:
        k = str(f[3])
        unp.setdefault(k, []).append((f[1] - f[0]) / 1e3)
    all_k = gate + fmha
    span = None
    if all_k:
        span = round((max(r[1] for r in all_k) - min(r[0] for r in all_k)) / 1e6, 3)
    return {
        "kernels": len(rows), "gate_up_count": len(gate), "fmha_count": len(fmha),
        "pairs": len(pairs), "per_s": out,
        "unpaired_fmha": {k: {"n": len(v), "us_median": med(v)} for k, v in unp.items()},
        "denoise_span_ms_first_to_last_fmha_or_gate_up": span,
        "gpu_span_ms": round((rows[-1][1] - rows[0][0]) / 1e6, 3) if rows else None,
    }


def parent():
    import torch

    info = probe()
    tmp = os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp()
    arms = (("S", "0", "gate_up_first"), ("A", "3", "gate_up_first"), ("B", "3", "fmha_first"))
    res, traces = {}, {}
    for tag, mode, order in arms:
        rep = os.path.join(tmp, f"fork_order_{tag}")
        pt = os.path.join(tmp, f"fork_order_{tag}.pt")
        env = dict(os.environ, TRTLLM_FLUX3_SIDE_GATE_UP=mode,
                   TRTLLM_FLUX3_SIDE_GATE_UP_ORDER=order)
        cmd = [NSYS, "profile", "-t", "cuda", "--cuda-graph-trace=node",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-f", "true",
               "-o", rep, sys.executable, __file__, CKPT, "--child", pt]
        log(f"=== {tag}: mode={mode} order={order}")
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
        traces[tag]["counters"] = res[tag]["counters"]
        log(tag, json.dumps(traces[tag]))
    # Solo durations from arm S: saved = solo_fmha + solo_gate_up - pair span (us/call); residual
    # FMHA last-wave idle ~= fact 1114's empty-SM probe fill - saved (full-GPU-equivalent us).
    solo = {}
    sS = traces["S"].get("per_s", {})
    for s in ("4176", "4096"):
        if s in sS:
            solo[s] = (sS[s]["fmha_us_median"], sS[s]["gate_up_us_median"])
    rows = {}
    for tag in ("S", "A", "B"):
        rows[tag] = {}
        for s, d in traces[tag].get("per_s", {}).items():
            r = dict(d)
            if s in solo and solo[s][0] and solo[s][1]:
                saved = solo[s][0] + solo[s][1] - d["pair_span_us_median"]
                r["saved_vs_solo_us"] = round(saved, 2)
                r["residual_fmha_tail_idle_us_est"] = round(FMHA_TAIL_IDLE_US[int(s)] - saved, 2)
            rows[tag][s] = r
    # Bitwise latents/images.
    def cmp(x, y):
        return [{"call": i, "prompt": a[0], "latent_bitwise": torch.equal(a[1], b[1]),
                 "image_bitwise": torch.equal(a[2], b[2])} for i, (a, b) in
                enumerate(zip(res[x]["seq"], res[y]["seq"]))]

    b_vs_a, a_vs_s = cmp("B", "A"), cmp("A", "S")

    def within(tag):
        s = res[tag]["seq"]
        return {f"call{j}_vs_call0": torch.equal(s[0][1], s[j][1]) and torch.equal(s[0][2], s[j][2])
                for j in (1, 2, 4)}

    w = {t: within(t) for t in ("S", "A", "B")}
    cb = res["B"]["counters"]
    passed = (all(p["latent_bitwise"] and p["image_bitwise"] for p in b_vs_a + a_vs_s)
              and w["B"] == w["A"] and cb["issued"] == cb["joined"] > 0
              and cb["sync_fallback"] == 0 and info["ancestor_rc"] == 0)
    print(json.dumps({
        "passed": bool(passed), "probe": info,
        "reference": "arm A (mode 3, gate_up_first = main's order) and arm S (mode 0, main's "
                     "sequential ops) fresh graph-on processes, same prompts/seeds A,A,A,B,A",
        "B_vs_A": b_vs_a, "A_vs_S": a_vs_s, "within_process_repeats": w,
        "served_order": rows,
        "unpaired_fmha": {t: traces[t].get("unpaired_fmha") for t in traces},
        "denoise_span_ms": {t: traces[t].get("denoise_span_ms_first_to_last_fmha_or_gate_up")
                            for t in traces},
        "counters": {t: traces[t].get("counters") for t in traces},
        "residual_idle_definition": "FMHA_TAIL_IDLE_US (fact 1114 empty-SM probe: 25.8/34.5 us "
                                    "full-GPU per call at S4176/S4096) minus saved_vs_solo_us "
                                    "(arm S solo fmha + gate_up medians - pair span median)",
    }), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child(sys.argv[3])
    else:
        parent()
