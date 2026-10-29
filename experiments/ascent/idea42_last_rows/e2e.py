"""Idea 42 end to end: served outputs of both cases, bitwise, through trtllm-serve.

Starts ONE trtllm-serve VisualGen server per configuration (the benchmark defaults: nvfp4, TP1,
torch.compile + autotune, warmup of [[2048, 2048]] t2i + i2i, overlaid with this tree's
experiments/ascent-config/engine.json: FLASHINFER FP8 attention, CUDA graphs, VAE settings) with
TORCH_LOGS=recompiles, then sends at concurrency 1, 2048x2048, 16 steps, fixed seeds:
  t2i A, i2i A, t2i B, i2i B, t2i A, i2i A, t2i A, i2i A   (t2i -> i2i -> t2i -> i2i, repeats)
and prints the sha256 of every decoded output (RGB pixels). Per configuration:
  repeat_ok      every repeat of a (case, prompt) is bitwise equal to its first output,
  recompiles     Dynamo "Recompiling" lines in the server log after the server became ready,
  route_lines    the server log lines of the last-block keep_rows cut (transformer_flux3).
Configurations: default, plus one per --env KEY=VAL given (e.g. TRTLLM_FLUX3_LAST_BLOCK_ROWS=0).
Compare the digests across run_gpu_pair arms (base tree vs this tree) for the bitwise check.

usage: e2e.py <checkpoint> [--env KEY=VAL ...]
"""

import base64
import glob
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

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
STORE = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
         "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
PROMPTS = {
    "A": ("A red fox sitting in a snowy forest at dawn, a wooden sign that says \"NORTH\", "
          "photorealistic"),
    "B": "Three blue ceramic cups on a kitchen table next to a bowl of lemons, soft daylight",
}
EDITS = {
    "A": "Make it look like a watercolor painting, keep the layout",
    "B": "Turn the sky orange at sunset and add warm lights",
}
SEQ = [("t2i", "A"), ("i2i", "A"), ("t2i", "B"), ("i2i", "B"),
       ("t2i", "A"), ("i2i", "A"), ("t2i", "A"), ("i2i", "A")]
SEEDS = {"A": 42, "B": 7}


def log(*a):
    print(f"[e2e {time.strftime('%H:%M:%S')}]", *a, flush=True)


def deep_merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def server_args():
    args = {
        "pipeline_config": {"variant": "nvfp4"},
        "parallel_config": {"tp_size": 1},
        "attention_config": {"backend": "VANILLA"},
        "cuda_graph_config": {"enable": False},
        "torch_compile_config": {"enable": True, "enable_fullgraph": False,
                                 "enable_autotune": True},
        "compilation_config": {"resolutions": [[2048, 2048]], "num_frames": [1],
                               "modes": ["t2i", "i2i"]},
    }
    path = os.path.join(WT, "experiments/ascent-config/engine.json")
    overlay = json.load(open(path)) if os.path.exists(path) else {}
    env = overlay.pop("env", {}) if isinstance(overlay.get("env"), dict) else {}
    vga = overlay.pop("visual_gen_args", None)
    deep_merge(args, vga if isinstance(vga, dict) else overlay)
    return args, env


def reference_png():
    files = sorted(f for f in glob.glob(os.path.join(STORE, "**", "*"), recursive=True)
                   if os.path.isfile(f) and f.lower().endswith(".png"))
    if not files:
        files = sorted(f for f in glob.glob(os.path.join(STORE, "**", "*"), recursive=True)
                       if os.path.isfile(f))
    for f in files:
        data = open(f, "rb").read()
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return f, data
    raise SystemExit(f"no PNG reference in {STORE}")


def digest(b64):
    from PIL import Image
    import numpy as np

    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    arr = np.asarray(img)
    return hashlib.sha256(arr.tobytes()).hexdigest(), list(arr.shape)


def post(url, body, timeout=1200):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_config(ckpt, name, extra_env, ref_b64):
    args, env_over = server_args()
    tmpd = tempfile.mkdtemp(prefix=f"idea42_{name}_")
    ypath = os.path.join(tmpd, "vg.yaml")
    json.dump(args, open(ypath, "w"))
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in env_over.items()})
    env.update(extra_env)
    env["TORCH_LOGS"] = "recompiles"
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    art = os.environ.get("AUTO_DEV_ARTIFACTS") or tmpd
    os.makedirs(art, exist_ok=True)
    log_path = os.path.join(art, f"server_{name}.log")
    cmd = [sys.executable, "-m", "tensorrt_llm.commands.serve", "serve", ckpt,
           "--visual_gen_args", ypath, "--host", "127.0.0.1", "--port", str(port)]
    log(f"[{name}] env {extra_env} args {json.dumps(args)}")
    srv_log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd=WT, env=env)
    base = f"http://127.0.0.1:{port}"
    outs, lat = [], []
    ready_offset = 0
    try:
        t0, last = time.time(), 0.0
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > 2400:
                raise RuntimeError("server start timeout")
            if time.time() - last > 60:
                last = time.time()
                log(f"[{name}] waiting {time.time() - t0:.0f}s")
            time.sleep(2)
        srv_log.flush()
        ready_offset = os.path.getsize(log_path)
        log(f"[{name}] ready in {time.time() - t0:.0f}s")
        for i, (case, key) in enumerate(SEQ):
            t = time.time()
            if case == "t2i":
                js = post(base + "/v1/images/generations",
                          {"prompt": PROMPTS[key], "seed": SEEDS[key], "size": "2048x2048",
                           "num_inference_steps": 16, "response_format": "b64_json", "n": 1})
            else:
                js = post(base + "/v1/images/edits",
                          {"prompt": EDITS[key], "image": ref_b64, "seed": SEEDS[key],
                           "size": "2048x2048", "num_inference_steps": 16,
                           "response_format": "b64_json", "model": "flux-3-image-klein"})
            ms = (time.time() - t) * 1e3
            d, shape = digest(js["data"][0]["b64_json"])
            outs.append({"i": i, "case": case, "prompt": key, "sha256": d, "shape": shape,
                         "ms": round(ms, 1)})
            log(f"[{name}] {outs[-1]}")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=120)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        srv_log.close()
    text = open(log_path, errors="replace").read()
    after = text[ready_offset:]
    recompiles = [ln for ln in after.splitlines() if "Recompiling" in ln]
    route = [ln[-260:] for ln in text.splitlines()
             if "final-layer rows" in ln or "kept rows" in ln]
    first = {}
    repeat_ok = True
    for o in outs:
        k = (o["case"], o["prompt"])
        if k in first and first[k] != o["sha256"]:
            repeat_ok = False
        first.setdefault(k, o["sha256"])
    return {"name": name, "env": extra_env, "outputs": outs, "repeat_ok": repeat_ok,
            "digests": {f"{c}/{p}": d for (c, p), d in first.items()},
            "recompiles_after_ready": len(recompiles), "recompile_lines": recompiles[:5],
            "route_lines": route[:8], "server_log": log_path,
            "tracebacks": text.count("Traceback")}


def main():
    ckpt = sys.argv[1]
    envs = [a.split("=", 1) for a in sys.argv[2:] if "=" in a]
    try:
        sha = subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = None
    ref_path, ref = reference_png()
    log(f"tree {WT} sha {sha}; reference {ref_path} sha256 {hashlib.sha256(ref).hexdigest()}")
    ref_b64 = base64.b64encode(ref).decode()
    results = [run_config(ckpt, "default", {}, ref_b64)]
    for k, v in envs:
        results.append(run_config(ckpt, f"{k}={v}", {k: v}, ref_b64))
    same = all(r["digests"] == results[0]["digests"] for r in results)
    passed = all(r["repeat_ok"] and len(r["outputs"]) == len(SEQ) for r in results)
    print(json.dumps({"passed": bool(passed), "configs_equal": bool(same), "git_sha": sha,
                      "compared": "decoded RGB sha256 per (case, prompt); repeats within a server; "
                                  "configs within this job; compare digests across pair arms",
                      "digests": {r["name"]: r["digests"] for r in results},
                      "results": results}), flush=True)


if __name__ == "__main__":
    main()
