"""In-process end-to-end check for branch 3 (FP8R on the BF16 checkpoint-excluded blocks).

Parent mode (default): runs two child processes on the same GPU, one after the other:
  off: TRTLLM_FLUX3_SM120_FP8_EXCLUDED=0 (exactly main's BF16 path for those blocks)
  on:  the branch default (FP8R)
Each child loads VisualGen with the benchmark's engine defaults (VANILLA attention, per-block
torch.compile with autotune, no CUDA graphs, warmup [[1024,1024]] t2i), with TORCH_LOGS=recompiles.
After warmup it prints a TIMED-START marker, then generates prompts whose text lengths span several
pad_multiple buckets, each twice (repeat determinism), and saves the uint8 images.

The parent reports per mode: which blocks the branch switched (log line), Dynamo recompiles and
CuTe/compile messages after the marker, repeat determinism, per-request denoise time, and per
prompt PSNR of on vs off (precision cost, information only).
"""
import json
import math
import os
import re
import subprocess
import sys
import time

PROMPTS = [
    "A red fox standing in fresh snow at sunrise",
    "A cozy reading nook with a green velvet armchair, a brass floor lamp, stacks of old books "
    "on a wooden side table, a sleeping tabby cat curled on a knitted blanket, rain streaking "
    "down a tall window, warm evening light, photorealistic interior photograph",
    "An isometric illustration of a small futuristic harbor town at dusk: terraced houses with "
    "solar roofs climbing a green hill, a lighthouse with a rotating beam, fishing boats with "
    "glowing lanterns moored along a curved stone pier, a cable car crossing the bay, market "
    "stalls selling fruit under striped awnings, people walking dogs on the promenade, seagulls, "
    "soft pastel colors, clean vector lines, high detail, wide establishing view, gentle haze "
    "over the water and the first stars appearing in a violet sky above distant mountains",
    "Close-up macro photo of a dewdrop on a fern leaf",
]
SEEDS = [11, 23, 37, 41]


def child(out_dir: str, ckpt: str) -> None:
    import torch

    from tensorrt_llm import VisualGen, VisualGenArgs

    yaml_path = os.path.join(out_dir, "args.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            "parallel_config:\n  tp_size: 1\nattention_config:\n  backend: VANILLA\n"
            "cuda_graph_config:\n  enable: false\ncompilation_config:\n"
            "  resolutions: [[1024, 1024]]\n  num_frames: [1]\n  modes: [t2i]\n"
        )
    config = VisualGenArgs.from_yaml(yaml_path)
    config.pipeline_config["variant"] = "nvfp4"
    model = VisualGen(model=ckpt, args=config)
    print("=== TIMED-START ===", flush=True)
    sys.stderr.write("=== TIMED-START ===\n")
    sys.stderr.flush()
    res = []
    for rep in range(int(os.environ.get("FP8X_REPS", "2"))):
        for i, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
            params = model.default_params
            params.height, params.width, params.seed = 1024, 1024, seed
            t0 = time.time()
            out = model.generate(inputs=prompt, params=params)
            wall = time.time() - t0
            img = out.image.detach().cpu()
            torch.save(img, os.path.join(out_dir, f"img_{i}_rep{rep}.pt"))
            m = out.metrics
            res.append(
                {
                    "prompt": i,
                    "rep": rep,
                    "wall_s": wall,
                    "denoise": getattr(m, "denoise", None),
                    "generation": getattr(m, "generation", None),
                }
            )
            print("REQ", json.dumps(res[-1]), flush=True)
    print("=== TIMED-END ===", flush=True)
    with open(os.path.join(out_dir, "requests.json"), "w") as f:
        json.dump(res, f)


def psnr(a, b):
    import torch

    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def parent(ckpt: str) -> None:
    import torch

    base = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/fp8x_e2e")
    summary = {}
    reps = int(os.environ.get("FP8X_REPS", "2"))
    modes = os.environ.get("FP8X_MODES", "off,on").split(",")
    for mode in modes:
        d = os.path.join(base, mode)
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ)
        env["TORCH_LOGS"] = "recompiles"
        env["TRTLLM_FLUX3_SM120_FP8_EXCLUDED"] = "0" if mode == "off" else "1"
        log_path = os.path.join(d, "log.txt")
        print("running mode", mode, flush=True)
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                [sys.executable, __file__, "child", d, ckpt],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
            for line in proc.stdout:
                log.write(line)
                if line.startswith(("REQ", "===", "running")) or "FP8R for BF16" in line:
                    print(mode, line.rstrip()[:400], flush=True)
            rc = proc.wait()
        text = open(log_path).read()
        timed = text.split("=== TIMED-START ===", 1)[-1] if "=== TIMED-START ===" in text else ""
        warm = text.split("=== TIMED-START ===", 1)[0]
        recompile_re = re.compile(r"Recompiling function|torch._dynamo.*recompil", re.I)
        switched = [ln for ln in text.splitlines() if "FP8R for BF16" in ln]
        reqs = json.load(open(os.path.join(d, "requests.json"))) if rc == 0 else []
        per_prompt = []
        if rc == 0:
            for i in range(len(PROMPTS)):
                ref = torch.load(os.path.join(d, f"img_{i}_rep0.pt"))
                per_prompt.append(
                    [
                        torch.equal(ref, torch.load(os.path.join(d, f"img_{i}_rep{r}.pt")))
                        for r in range(1, reps)
                    ]
                )
        det = rc == 0 and all(all(p) for p in per_prompt)
        summary[mode] = {
            "returncode": rc,
            "switched_log": switched[-1][-600:] if switched else None,
            "recompiles_during_warmup": len(recompile_re.findall(warm)),
            "recompiles_during_timed": len(recompile_re.findall(timed)),
            "recompile_lines_timed": [ln[:300] for ln in timed.splitlines() if recompile_re.search(ln)][:10],
            "repeat_bitwise_identical": det,
            "repeat_equal_to_rep0_per_prompt": per_prompt,
            "denoise_per_request": [r["denoise"] for r in reqs],
            "wall_s_per_request": [round(r["wall_s"], 4) for r in reqs],
        }
        print(mode, json.dumps(summary[mode]), flush=True)
    psnrs = []
    if all(m in summary and summary[m]["returncode"] == 0 for m in ("off", "on")):
        for i in range(len(PROMPTS)):
            a = torch.load(os.path.join(base, "off", f"img_{i}_rep0.pt"))
            b = torch.load(os.path.join(base, "on", f"img_{i}_rep0.pt"))
            psnrs.append(round(psnr(a, b), 3))
    passed = all(summary[m]["returncode"] == 0 for m in summary) and all(
        summary[m]["repeat_bitwise_identical"] for m in summary
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "compared": "FP8R branch vs the same tree with TRTLLM_FLUX3_SM120_FP8_EXCLUDED=0 "
                "(main's BF16 path for those blocks), same GPU, separate processes; repeat "
                "determinism within each process; Dynamo recompiles after warmup",
                "psnr_on_vs_off_db": psnrs,
                "modes": summary,
            }
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    else:
        parent(sys.argv[1])
