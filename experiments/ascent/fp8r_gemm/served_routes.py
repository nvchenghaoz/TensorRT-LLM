"""Idea 122: served-path route count + full-model exactness, ONE job, two fresh processes.

Arms (pipeline as served: engine.json overlay, torch.compile + autotune, CUDA graphs on):
  A: TRTLLM_FLUX3_FP8R_GEMM=0 (main's torch._scaled_mm at the 4 FP8R sites)
  B: TRTLLM_FLUX3_FP8R_GEMM=1 (routed, default)
Each child generates P, P, P (profiled under nsys --cuda-graph-trace=node, cudaProfilerApi range),
P and saves latents/images. The parent counts, in the profiled request, every CUTLASS GemmUniversal
kernel with E4M3 operands by owner (torch = at::cuda::detail enable_3x kernel, flux3 = this
branch's overlay) and grid, and compares latents/images bitwise B vs A and call j vs call 0.

usage: python served_routes.py <checkpoint>      (parent)
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1]
NSYS = "/usr/local/cuda/bin/nsys"
PROMPT = ("A red fox standing in fresh snow at sunrise", 100)


def log(*a):
    print(*a, flush=True)


def child(out_path):
    import torch

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
    log(f"loaded {time.time() - t0:.0f}s TRTLLM_FLUX3_FP8R_GEMM="
        f"{os.environ.get('TRTLLM_FLUX3_FP8R_GEMM')}")

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

    seq = []
    for i in range(4):
        t = time.time()
        if i == 2:
            torch.cuda.profiler.start()
        seq.append(generate(*PROMPT))
        if i == 2:
            torch.cuda.profiler.stop()
        log(f"gen {i}: {time.time() - t:.2f}s")
    routes = {}
    try:
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg

        routes = {str(k): v for k, v in fg.ROUTES.items()}
    except ImportError:
        pass
    torch.save({"seq": seq, "routes_python": routes}, out_path)
    log("CHILD_JSON " + json.dumps({"routes_python_capture_time": routes}))


def classify(rows):
    groups = {}
    for name, grid, dur in rows:
        if "GemmUniversal" not in name or "e4m3" not in name:
            continue
        if "e2m1" in name:
            continue
        owner = "torch_scaled_mm" if "enable_3x_kernel" in name else (
            "flux3_fp8r_overlay" if "Pingpong" in name else "other_e4m3")
        key = f"{owner} grid={list(grid)}"
        g = groups.setdefault(key, {"calls": 0, "us_total": 0.0})
        g["calls"] += 1
        g["us_total"] += dur / 1e3
    for g in groups.values():
        g["us_total"] = round(g["us_total"], 1)
        g["us_mean"] = round(g["us_total"] / g["calls"], 1)
    return groups


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
            rows.append((name, grid, int(float(r["Duration (ns)"]))))
    return rows


def parent():
    import torch

    sys.path.insert(0, HERE)
    import step0

    step0.header()
    tmp = os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp()
    res, kern = {}, {}
    for tag, flag in (("A", "0"), ("B", "1")):
        rep = os.path.join(tmp, f"fp8r_routes_{tag}")
        pt = os.path.join(tmp, f"fp8r_routes_{tag}.pt")
        env = dict(os.environ, TRTLLM_FLUX3_FP8R_GEMM=flag)
        cmd = [NSYS, "profile", "-t", "cuda", "--cuda-graph-trace=node",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-f", "true",
               "-o", rep, sys.executable, os.path.abspath(__file__), CKPT, "--child", pt]
        log(f"=== {tag}: TRTLLM_FLUX3_FP8R_GEMM={flag}")
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
        kern[tag] = classify(load_rows(csv_path)) if os.path.exists(csv_path) else {"error": "no csv"}
        log(tag, json.dumps(kern[tag]))

    def same(a, b):
        return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    b_vs_a = [same(a, b) for a, b in zip(res["A"]["seq"], res["B"]["seq"])]
    within = {t: [same(res[t]["seq"][0], res[t]["seq"][j]) for j in (1, 2, 3)] for t in res}
    kb, ka = kern["B"], kern["A"]
    flux3_calls = {k: v["calls"] for k, v in kb.items() if k.startswith("flux3_fp8r_overlay")}
    torch_calls_b = {k: v["calls"] for k, v in kb.items() if k.startswith("torch_scaled_mm")}
    torch_calls_a = {k: v["calls"] for k, v in ka.items() if k.startswith("torch_scaled_mm")}
    passed = (all(b_vs_a) and all(all(v) for v in within.values()) and not torch_calls_b
              and sum(flux3_calls.values()) == sum(torch_calls_a.values()) > 0
              and not any(k.startswith("flux3") for k in ka))
    print(json.dumps({
        "passed": bool(passed),
        "reference": "arm A = TRTLLM_FLUX3_FP8R_GEMM=0 (main's torch._scaled_mm), fresh graph-on "
                     "process in the same job, same prompt/seed, 4 requests",
        "latent_and_image_bitwise_B_vs_A_per_call": b_vs_a,
        "within_process_call_j_vs_call0": within,
        "profiled_request_e4m3_gemm_kernels": {"A": ka, "B": kb},
        "routes_python_capture_time": {t: res[t]["routes_python"] for t in res},
    }), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child(sys.argv[3])
    else:
        parent()
