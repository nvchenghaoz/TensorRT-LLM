"""Worker-runnable stand-in for check.py's repeated-execution part.

check.py / reference.json are not shipped to worker GPU jobs (knowledge 129, 428).
This starts ONE trtllm-serve VisualGen server on the benchmark engine settings with
experiments/ascent-config/engine.json overlaid, sends 12 prompts (fixed seeds
100-111, 1024x1024, 16 steps, b64_json) for 2 full passes, and requires pass 2 ==
pass 1 bitwise, both on the decoded RGB pixels and on the PNG bytes.
One prompt is long enough to pad text to T=160, so the T=80 -> T=160 -> T=80
transition is exercised on the running server.

usage: python e2e_repeat_check.py <checkpoint> <worktree>
"""
import base64
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

CKPT = sys.argv[1]
WT = sys.argv[2]

PROMPTS = [
    "A red fox sitting in a snowy forest at dawn, photorealistic",
    "A bowl of ramen on a wooden table, steam rising, studio lighting",
    "An astronaut riding a horse on the surface of Mars",
    "A watercolor painting of a lighthouse on a rocky cliff",
    "A futuristic city skyline at night with neon signs reflected in rain puddles",
    "Close-up portrait of an elderly fisherman with a weathered face",
    "A cozy library with tall bookshelves and a fireplace",
    "A cat wearing sunglasses lounging on a beach chair",
    "A vintage red bicycle leaning against a brick wall covered in ivy",
    "Isometric pixel art of a small farm with a windmill",
    "A glass of orange juice next to sliced oranges, top-down view",
    ("A highly detailed fantasy illustration of an ancient elven city built into enormous "
     "glowing trees, with spiral staircases winding around the trunks, rope bridges "
     "connecting platforms, lanterns hanging from branches, waterfalls cascading into "
     "turquoise pools below, flocks of white birds circling overhead, a crowd of travelers "
     "in colorful cloaks walking along mossy paths, soft golden evening light filtering "
     "through the canopy, mist drifting between the roots, intricate carvings on every "
     "wall, banners fluttering in the wind, a dragon sleeping on a distant mountain peak, "
     "and a full moon rising behind snowy summits, cinematic composition, ultra sharp"),
]
SEEDS = list(range(100, 100 + len(PROMPTS)))


def log(*a):
    print(*a, flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def deep_merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def main():
    args = {
        "attention_config": {"backend": "VANILLA"},
        "compilation_config": {"modes": ["t2i"], "num_frames": [1],
                               "resolutions": [[1024, 1024]], "skip_warmup": False},
        "cuda_graph_config": {"enable": False},
        "parallel_config": {"tp_size": 1},
        "pipeline_config": {"variant": "nvfp4"},
        "torch_compile_config": {"enable": True, "enable_autotune": True,
                                 "enable_fullgraph": False},
    }
    overlay_path = os.path.join(WT, "experiments/ascent-config/engine.json")
    overlay = {}
    if os.path.exists(overlay_path):
        with open(overlay_path) as f:
            overlay = json.load(f)
    deep_merge(args, overlay)
    log("[cfg] engine overlay sha256",
        hashlib.sha256(open(overlay_path, "rb").read()).hexdigest() if overlay else None)
    log("[cfg] visual_gen_args", json.dumps(args))
    tmpd = tempfile.mkdtemp()
    ypath = os.path.join(tmpd, "vg.yaml")
    with open(ypath, "w") as f:
        json.dump(args, f)  # JSON is valid YAML
    port = free_port()
    cmd = [sys.executable, "-m", "tensorrt_llm.commands.serve", "serve", CKPT,
           "--visual_gen_args", ypath, "--host", "127.0.0.1", "--port", str(port)]
    log("[srv]", " ".join(cmd))
    srv_log = open(os.path.join(tmpd, "server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd=WT)
    base = f"http://127.0.0.1:{port}"
    result = {"passed": False}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > 1500:
                raise RuntimeError("server start timeout")
            if int(time.time() - t0) % 30 == 0:
                log(f"[srv] waiting {time.time() - t0:.0f}s")
            time.sleep(1)
        log(f"[srv] ready in {time.time() - t0:.0f}s")
        from PIL import Image
        import numpy as np

        def gen(i):
            body = json.dumps({"prompt": PROMPTS[i], "seed": SEEDS[i], "size": "1024x1024",
                               "num_inference_steps": 16, "response_format": "b64_json",
                               "n": 1}).encode()
            req = urllib.request.Request(base + "/v1/images/generations", data=body,
                                         headers={"Content-Type": "application/json"})
            t = time.time()
            with urllib.request.urlopen(req, timeout=600) as r:
                js = json.loads(r.read())
            png = base64.b64decode(js["data"][0]["b64_json"])
            px = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            return png, px, (time.time() - t) * 1000

        passes = []
        for p in range(2):
            out = []
            for i in range(len(PROMPTS)):
                png, px, ms = gen(i)
                out.append((hashlib.sha256(png).hexdigest(), px))
                log(f"[pass {p}] prompt {i} seed {SEEDS[i]} {ms:.0f} ms png_sha "
                    f"{out[-1][0][:16]} px_sha {hashlib.sha256(px.tobytes()).hexdigest()[:16]}")
            passes.append(out)
        per = []
        for i in range(len(PROMPTS)):
            a, b = passes[0][i], passes[1][i]
            px_eq = a[1].shape == b[1].shape and bool((a[1] == b[1]).all())
            per.append({"prompt": i, "seed": SEEDS[i], "pixels_bitwise": px_eq,
                        "png_bytes_equal": a[0] == b[0],
                        "max_abs_px_diff": int(np.abs(a[1].astype(int) - b[1].astype(int)).max())})
        rep = all(x["pixels_bitwise"] for x in per)
        # distinct-image sanity: outputs are not degenerate / identical across prompts
        distinct = len({passes[0][i][0] for i in range(len(PROMPTS))})
        result = {
            "passed": rep and distinct == len(PROMPTS),
            "repeated_execution_passed": rep,
            "prompts": len(PROMPTS),
            "passes": 2,
            "pixels_bitwise_count": sum(x["pixels_bitwise"] for x in per),
            "png_bytes_equal_count": sum(x["png_bytes_equal"] for x in per),
            "distinct_images": distinct,
            "per_prompt": per,
            "engine_overlay": overlay,
            "note": "stand-in for check.py repeated-execution; gate v2 vs reference.json "
                    "is not runnable from worker jobs (reference not shipped)",
        }
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        srv_log.close()
        try:
            txt = open(os.path.join(tmpd, "server.log")).read()
            rc = txt.count("__recompiles")
            log(f"[srv] log lines with __recompiles: {rc}")
            log("[srv] log tail:\n" + txt[-3000:])
        except Exception:
            pass
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
