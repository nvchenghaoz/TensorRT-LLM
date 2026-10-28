"""Idea 161 acceptance 2: nsys trace of one 1024^2 t2i request (pipeline as served: engine.json
overlay, torch.compile + autotune, CUDA graphs on) with TRTLLM_FLUX3_SIDE_GATE_UP=0 and =<mode>,
in two fresh processes. Each child warms (A, A: compile, autotune, graph capture), then profiles
one replayed request A between cudaProfilerStart/Stop. nsys --cuda-graph-trace=node; the parent
reduces cuda_gpu_trace: gate_up kernels (flux3_swiglu_fp4) per stream, FMHA (fmha_v2_flux3) per
stream and duration, gate_up/FMHA time overlap, denoise GPU span.

usage: python served_trace.py <checkpoint> [mode]
       python served_trace.py <checkpoint> --child
"""

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1]
NSYS = "/usr/local/cuda/bin/nsys"


def log(*a):
    print(*a, flush=True)


def child():
    import torch

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
    pipe = PipelineLoader(args, device="cuda:0").load()

    def generate(prompt, seed):
        with torch.inference_mode():
            ctx = pipe._encode_prompt(prompt, None)
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            latent, ids = pipe._prepare_t2i(
                ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
            )
            pipe._package_output(pipe._decode_latents((latent, ids)))
        torch.cuda.synchronize()

    for _ in range(2):
        generate("A red fox standing in fresh snow at sunrise", 11)
    torch.cuda.profiler.start()
    generate("A red fox standing in fresh snow at sunrise", 11)
    torch.cuda.profiler.stop()
    print("CHILD_JSON " + json.dumps({"counters": dict(fp.SIDE_GATE_UP_COUNTERS)}), flush=True)


def reduce(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            name = r.get("Name", "")
            if not name or r.get("Duration (ns)") in (None, ""):
                continue
            start = int(float(r["Start (ns)"]))
            dur = int(float(r["Duration (ns)"]))
            rows.append((start, start + dur, name, r.get("Strm", r.get("Stream", "?"))))
    gate = [r for r in rows if "flux3_swiglu_fp4" in r[2]]
    fmha = [r for r in rows if "fmha_v2_flux3" in r[2]]

    def by_stream(rs):
        out = {}
        for r in rs:
            out[r[3]] = out.get(r[3], 0) + 1
        return out

    overlap_us, overlapped = 0.0, 0
    for g in gate:
        o = sum(max(0, min(g[1], f[1]) - max(g[0], f[0])) for f in fmha)
        overlap_us += o / 1e3
        overlapped += o > 0
    fd = [(f[1] - f[0]) / 1e3 for f in fmha]
    gd = [(g[1] - g[0]) / 1e3 for g in gate]
    big = [f for f in fmha]
    return {
        "kernels": len(rows),
        "gate_up_count": len(gate), "gate_up_by_stream": by_stream(gate),
        "gate_up_us_median": round(statistics.median(gd), 2) if gd else None,
        "fmha_count": len(fmha), "fmha_by_stream": by_stream(fmha),
        "fmha_us_median": round(statistics.median(fd), 2) if fd else None,
        "fmha_us_sum": round(sum(fd), 1),
        "fmha_us_median_top448": round(statistics.median(sorted(fd)[-448:]), 2) if fd else None,
        "gate_up_overlapping_fmha": overlapped, "gate_up_fmha_overlap_us_total": round(overlap_us, 1),
        "gpu_span_ms": round((max(r[1] for r in rows) - min(r[0] for r in rows)) / 1e6, 3)
        if rows else None,
        "streams": sorted({r[3] for r in rows}),
        "_n_big": len(big),
    }


def parent(mode):
    tmp = os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp()
    out = {}
    for tag, env_mode in (("off", "0"), ("on", mode)):
        rep = os.path.join(tmp, f"idea161_{tag}")
        env = dict(os.environ, TRTLLM_FLUX3_SIDE_GATE_UP=env_mode)
        cmd = [NSYS, "profile", "-t", "cuda", "--cuda-graph-trace=node",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-f", "true",
               "-o", rep, sys.executable, __file__, CKPT, "--child"]
        log(f"=== {tag}: " + " ".join(cmd))
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        child_json = None
        for line in p.stdout.splitlines():
            if line.startswith("CHILD_JSON "):
                child_json = json.loads(line[len("CHILD_JSON "):])
        log(p.stdout[-1500:])
        log(p.stderr[-1500:])
        s = subprocess.run([NSYS, "stats", "-r", "cuda_gpu_trace", "-f", "csv", "-o", rep,
                            "--force-export=true", rep + ".nsys-rep"],
                           capture_output=True, text=True)
        log(s.stdout[-800:], s.stderr[-800:])
        csv_path = rep + "_cuda_gpu_trace.csv"
        out[tag] = reduce(csv_path) if os.path.exists(csv_path) else {"error": "no csv"}
        out[tag]["child"] = child_json
        log(tag, json.dumps(out[tag]))
    on = out["on"]
    # Under --cuda-graph-trace=node nsys reports graph kernels on the graph executor's internal
    # streams, not the capture streams; a gate_up that runs concurrently with an FMHA is on
    # another stream than that FMHA, so concurrency (time overlap) is the side-stream evidence.
    out["summary"] = {
        "gate_up_concurrent_with_fmha_on": on.get("gate_up_overlapping_fmha"),
        "gate_up_concurrent_with_fmha_off": out["off"].get("gate_up_overlapping_fmha"),
        "gate_up_total_on": on.get("gate_up_count"),
        "sync_fallback_on": (on.get("child") or {}).get("counters", {}).get("sync_fallback"),
        "fmha_us_median_off": out["off"].get("fmha_us_median"),
        "fmha_us_median_on": on.get("fmha_us_median"),
        "gpu_span_ms_off": out["off"].get("gpu_span_ms"),
        "gpu_span_ms_on": on.get("gpu_span_ms"),
    }
    out["passed"] = bool(
        on.get("gate_up_count")
        and out["summary"]["gate_up_concurrent_with_fmha_on"] == on["gate_up_count"]
        and out["summary"]["sync_fallback_on"] == 0
    )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child()
    else:
        parent(sys.argv[2] if len(sys.argv) > 2 else "3")
