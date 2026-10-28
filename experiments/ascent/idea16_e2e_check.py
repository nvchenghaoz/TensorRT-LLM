"""Idea 16 end-to-end check on one running trtllm-serve VisualGen server (b64_json path).

The frozen experiments/ascent-benchmark/check.py is not shipped to worker GPU jobs
(knowledge 129), so this replicates its repeated-execution part on the benchmark's own
engine settings and the frozen timed workload (12 prompts, seeds 100-111):
  1. one server (benchmark visual_gen_args, engine.json overlay if present), client
     test request first, then the 12 prompts twice (pass 1, pass 2) at c1 via
     POST /v1/images/generations response_format=b64_json;
  2. repeated_execution: pass-2 decoded pixels == pass-1 bitwise per prompt;
  3. body identity: every raw HTTP body == Starlette JSONResponse(json.loads(body)).body
     (the exact bytes main's JSONResponse(model_dump()) path renders);
  4. against the base tree: decoded pixels vs the baseline arm (job 255, main) and
     candidate arm (job 256) PNGs of serving pair f6bbbff0, when that directory is
     readable (cross-process: start-up mode can differ, knowledge 271; PSNR reported).
"""
import base64
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

CKPT = sys.argv[1]
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/idea16_e2e")
os.makedirs(ART, exist_ok=True)
PROMPTS = [
    "A red fox standing in fresh snow at sunrise",
    "A lighthouse on a cliff during a thunderstorm, oil painting",
    "A bowl of ramen with a soft boiled egg, studio photo",
    "A futuristic city skyline at night with flying cars",
    "Portrait of an elderly fisherman with a white beard, dramatic lighting",
    "A cozy reading nook with a window, plants and warm lamp light",
    "A watercolor painting of a hot air balloon over green hills",
    "A golden retriever puppy playing with a tennis ball on grass",
    "A vintage travel poster that says VISIT MARS in bold letters",
    "Macro photo of a dew drop on a green leaf",
    "A medieval castle surrounded by autumn forest, aerial view",
    "A robot chef cooking pancakes in a modern kitchen",
]
SEEDS = list(range(100, 112))
ROOT = "/home/scratch.chengzhang/trtllm-dev-flux3-klein-20261027-125409/runner/slots/1/jobs"
PAIR_DIRS = {
    "base_job255": f"{ROOT}/ce4de9dbacc74103b82a31df472537fb/artifacts/proxy/media_proxy_c1",
    "cand_job256": f"{ROOT}/af0cdfb55c5a4fe6bd3c44fa0fa5fb75/artifacts/proxy/media_proxy_c1",
}
ENGINE_YAML = """attention_config:
  backend: VANILLA
compilation_config:
  modes:
  - t2i
  num_frames:
  - 1
  resolutions:
  - - 1024
    - 1024
  skip_warmup: false
cuda_graph_config:
  enable: false
parallel_config:
  tp_size: 1
pipeline_config:
  variant: nvfp4
torch_compile_config:
  enable: true
  enable_autotune: true
  enable_fullgraph: false
"""


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def decode(png_bytes):
    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    overlay = os.path.join(os.getcwd(), "experiments/ascent-config/engine.json")
    yaml_path = os.path.join(ART, "engine.yaml")
    with open(yaml_path, "w") as f:
        f.write(ENGINE_YAML)
    if os.path.exists(overlay):
        import yaml

        cfg = yaml.safe_load(ENGINE_YAML)
        ov = json.load(open(overlay))

        def merge(d, o):
            for k, v in o.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    merge(d[k], v)
                else:
                    d[k] = v

        merge(cfg, ov)
        with open(yaml_path, "w") as f:
            yaml.safe_dump(cfg, f)
        log(f"engine.json overlay applied: {ov}")
    port = free_port()
    srv_log = open(os.path.join(ART, "server.log"), "w")
    cmd = [sys.executable, "-c",
           "import sys; from tensorrt_llm.commands.serve import main; "
           "sys.argv=['trtllm-serve']+sys.argv[1:]; main()",
           CKPT, "--host", "127.0.0.1", "--port", str(port), "--visual_gen_args", yaml_path]
    log(f"starting server: {' '.join(cmd[3:])}")
    srv = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"passed": False}
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"server exited rc={srv.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > 1500:
                raise RuntimeError("server not ready in 1500 s")
            if int(time.time() - t0) % 60 < 5:
                log(f"waiting for server ({time.time() - t0:.0f}s)")
            time.sleep(5)
        log(f"server ready after {time.time() - t0:.0f}s")
        from starlette.responses import JSONResponse

        def gen(prompt, seed):
            payload = {"prompt": prompt, "model": "flux-3-image-klein",
                       "response_format": "b64_json", "height": 1024, "width": 1024,
                       "num_inference_steps": 16, "seed": seed}
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/images/generations",
                data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer unused"})
            t = time.perf_counter()
            with urllib.request.urlopen(req, timeout=600) as r:
                raw = r.read()
                timing = r.headers.get("Server-Timing")
                ctype = r.headers.get("content-type")
            ms = (time.perf_counter() - t) * 1000
            body = json.loads(raw)
            ref = JSONResponse(content=body)
            same = ref.body == raw and ctype == "application/json"
            pngs = [base64.b64decode(d["b64_json"]) for d in body["data"]]
            return pngs, same, timing is not None, ms

        gen(PROMPTS[0], SEEDS[0])  # client's initial test request (untimed)
        log("test request done")
        passes = []
        body_ok, timing_ok = [], []
        for p in range(2):
            imgs = []
            for i, (pr, sd) in enumerate(zip(PROMPTS, SEEDS)):
                pngs, same, has_t, ms = gen(pr, sd)
                assert len(pngs) == 1
                imgs.append(pngs[0])
                body_ok.append(same)
                timing_ok.append(has_t)
                log(f"pass {p + 1} prompt {i:2d}: {ms:7.1f} ms png {len(pngs[0])} B "
                    f"body==JSONResponse {same}")
            passes.append(imgs)
        arrays = [[decode(b) for b in imgs] for imgs in passes]
        repeat_eq = [bool(np.array_equal(a, b)) for a, b in zip(arrays[0], arrays[1])]
        png_bytes_eq = [a == b for a, b in zip(passes[0], passes[1])]
        for i, b in enumerate(passes[0]):
            with open(os.path.join(ART, f"check_{i:02d}.png"), "wb") as f:
                f.write(b)
        vs = {}
        for name, d in PAIR_DIRS.items():
            if not os.path.isdir(d):
                vs[name] = "unreadable"
                continue
            eq, ps = [], []
            for i in range(12):
                ref_arr = decode(open(os.path.join(d, f"{i:04d}_0.png"), "rb").read())
                eq.append(bool(np.array_equal(ref_arr, arrays[0][i])))
                ps.append(psnr(ref_arr, arrays[0][i]))
            vs[name] = {"bitwise_equal": sum(eq), "of": 12,
                        "min_psnr_db": min(ps) if ps else None}
            log(f"vs {name}: {vs[name]}")
        rep = all(repeat_eq)
        result = {
            "passed": rep and all(body_ok) and all(timing_ok),
            "repeated_execution_passed": rep,
            "repeat_pixels_equal": f"{sum(repeat_eq)}/12",
            "repeat_png_bytes_equal": f"{sum(png_bytes_eq)}/12",
            "body_identical_to_JSONResponse": f"{sum(body_ok)}/{len(body_ok)}",
            "server_timing_present": f"{sum(timing_ok)}/{len(timing_ok)}",
            "vs_serving_pair_images": vs,
            "compared": "12 frozen workload prompts (seeds 100-111), b64_json, one server, "
                        "pass 2 vs pass 1 bitwise decoded pixels; raw body vs Starlette "
                        "JSONResponse render; decoded pixels vs serving pair 255 (main) / 256",
        }
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=60)
        except Exception:
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except Exception:
                pass
        srv_log.close()
    print(json.dumps(result), flush=True)


main()
