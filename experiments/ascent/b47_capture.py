"""Branch 47 (idea 42 Part 1): capture real served image tensors + stage stamps.

One trtllm-serve VisualGen server (benchmark engine settings + engine.json overlay)
with TRTLLM_VISUALGEN_DUMP_DIR set: client test request, 24 timed requests (the 12
frozen workload prompts, seeds 100-111, twice), then 12 check requests (seeds
500-511). image_to_bytes saves each served uint8 tensor (shape, strides,
storage_offset) and its encode/b64 stamps. Then the server is stopped and the
deflate sweep (b47_sweep.py) runs on the saved tensors in the same job.
Usage: python b47_capture.py <checkpoint>
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
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b47")
DUMP = os.path.join(ART, "b47_dump")
os.makedirs(DUMP, exist_ok=True)
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


def main():
    import yaml

    overlay = os.path.join(os.getcwd(), "experiments/ascent-config/engine.json")
    cfg = yaml.safe_load(ENGINE_YAML)
    if os.path.exists(overlay):
        ov = json.load(open(overlay))

        def merge(d, o):
            for k, v in o.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    merge(d[k], v)
                else:
                    d[k] = v

        merge(cfg, ov)
        log(f"engine.json overlay applied: {ov}")
    yaml_path = os.path.join(ART, "engine.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)
    port = free_port()
    srv_log = open(os.path.join(ART, "server.log"), "w")
    env = dict(os.environ, TRTLLM_VISUALGEN_DUMP_DIR=DUMP)
    cmd = [sys.executable, "-c",
           "import sys; from tensorrt_llm.commands.serve import main; "
           "sys.argv=['trtllm-serve']+sys.argv[1:]; main()",
           CKPT, "--host", "127.0.0.1", "--port", str(port), "--visual_gen_args", yaml_path]
    log(f"starting server, dump dir {DUMP}")
    srv = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT,
                           start_new_session=True, env=env)
    reqs = []
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
            ms = (time.perf_counter() - t) * 1000
            b64 = json.loads(raw)["data"][0]["b64_json"]
            return b64, ms

        plan = [("test", PROMPTS[0], 100)]
        plan += [("timed", p, s) for p, s in zip(PROMPTS, range(100, 112))] * 2
        plan += [("check", p, s) for p, s in zip(PROMPTS, range(500, 512))]
        for seq, (kind, pr, sd) in enumerate(plan):
            b64, ms = gen(pr, sd)
            png = base64.b64decode(b64)
            with open(os.path.join(DUMP, f"served_{seq:04d}.png"), "wb") as f:
                f.write(png)
            reqs.append({"seq": seq, "kind": kind, "seed": sd, "client_ms": ms,
                         "png_bytes": len(png)})
            log(f"{seq:2d} {kind} seed {sd}: {ms:7.1f} ms png {len(png)} B")
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
    with open(os.path.join(DUMP, "requests.json"), "w") as f:
        json.dump(reqs, f)
    stamps = [json.loads(l) for l in open(os.path.join(DUMP, "stamps.jsonl"))]
    timed = [s for s, r in zip(stamps, reqs) if r["kind"] == "timed"]
    enc = sorted(s["encode_ms"] for s in timed)
    b64s = sorted(s["b64_ms"] for s in timed)
    log(f"timed encode_ms median {enc[len(enc)//2]:.2f} p90 {enc[int(.9*len(enc))]:.2f}; "
        f"b64 median {b64s[len(b64s)//2]:.2f}")
    log(f"strides seen: {sorted({str(s['stride']) for s in stamps})}")
    # served PNG decode == saved tensor (sanity of the capture)
    import torch
    ok = 0
    for s in stamps:
        t = torch.load(os.path.join(DUMP, f"img_{s['seq']:04d}.pt"))["data"]
        a = t[0].numpy() if t.dim() == 4 else t.numpy()
        d = np.asarray(Image.open(io.BytesIO(open(
            os.path.join(DUMP, f"served_{s['seq']:04d}.png"), "rb").read())))
        ok += int(np.array_equal(a, d))
    log(f"served decode == saved tensor: {ok}/{len(stamps)}")
    print(f"DUMP_DIR {os.path.abspath(DUMP)}", flush=True)
    sys.argv = ["b47_sweep.py", DUMP]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import b47_sweep
    b47_sweep.main(DUMP, stamps_timed={"encode_ms": enc, "b64_ms": b64s})


main()
