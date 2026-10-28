"""Branch 70 (idea 101): VisualGen to_tensor CUDA IPC stamps + exactness/memory run.

One fresh trtllm-serve VisualGen server (benchmark engine settings + engine.json overlay),
launched with a thin wrapper that times PipelineOutput.to_tensor in the server process
(works on main and branch alike: "outer" stamp) and, optionally, flips the fast path per
pass (B70_TOGGLE=1: pass 1 runs with TRTLLM_VISUALGEN_IPC_FAST semantics off, pass 2 on,
same process -> bitwise pass1 vs pass2 is main path vs fast path).

Sequence: client test request (untimed) + 24 c1 requests (12 check prompts x 2 passes),
optional c2 phase (B70_C2=1: 6 rounds x 2 concurrent requests, each image vs its c1 image),
per-request nvidia-smi memory of server + worker pids.
Last stdout line: JSON summary.
"""
import base64
import concurrent.futures as cf
import hashlib
import io
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

CKPT = sys.argv[1]
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b70")
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
ENGINE_YAML = {
    "attention_config": {"backend": "VANILLA"},
    "compilation_config": {"modes": ["t2i"], "num_frames": [1], "resolutions": [[1024, 1024]],
                           "skip_warmup": False},
    "cuda_graph_config": {"enable": False},
    "parallel_config": {"tp_size": 1},
    "pipeline_config": {"variant": "nvfp4"},
    "torch_compile_config": {"enable": True, "enable_autotune": True, "enable_fullgraph": False},
}

LAUNCH = r'''
import json, os, sys, time
from tensorrt_llm._torch.visual_gen import output as _o
_orig = _o.PipelineOutput.to_tensor
_path = os.environ["B70_OUTER_FILE"]
_toggle = os.environ.get("B70_TOGGLE") == "1"
_state = {"n": 0}
def _wrapped(self):
    n = _state["n"]; _state["n"] += 1
    if _toggle and hasattr(_o, "_set_ipc_fast"):
        # request 0 = client test request, 1..12 pass 1 (main path), 13.. pass 2 (fast path)
        _o._set_ipc_fast(n >= 13)
    t0 = time.perf_counter()
    r = _orig(self)
    ms = (time.perf_counter() - t0) * 1e3
    with open(_path, "a") as f:
        f.write(json.dumps({"n": n, "to_tensor_ms": ms, "t": time.time(),
                            "fast": bool(getattr(_o, "_IPC_FAST_ACTIVE", False))}) + "\n")
    return r
_o.PipelineOutput.to_tensor = _wrapped
from tensorrt_llm.commands.serve import main
sys.argv = ["trtllm-serve"] + sys.argv[1:]
main()
'''


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


def descendants(pid):
    out = [pid]
    try:
        kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    except Exception:
        return out
    for k in kids:
        out += descendants(int(k))
    return out


def gpu_mem(pids):
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)}
    mem = {}
    for line in r.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in pids:
            mem[parts[0]] = int(parts[1]) if parts[1].isdigit() else parts[1]
    return mem


def stats(xs):
    if not xs:
        return None
    s = sorted(xs)
    return {"n": len(s), "mean": statistics.mean(s), "median": statistics.median(s),
            "p90": s[min(len(s) - 1, (9 * len(s)) // 10)], "max": s[-1], "min": s[0]}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip()]


def main():
    cfg = json.loads(json.dumps(ENGINE_YAML))
    overlay = os.path.join(os.getcwd(), "experiments/ascent-config/engine.json")
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
    import yaml

    yaml_path = os.path.join(ART, "engine.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)
    outer = os.path.join(ART, "outer_to_tensor.jsonl")
    stamps = os.path.join(ART, "ipc_stamps.jsonl")
    for p in (outer, stamps):
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ)
    env["B70_OUTER_FILE"] = outer
    env["TRTLLM_VISUALGEN_IPC_STAMPS_FILE"] = stamps
    log(f"env: STAMPS={env.get('TRTLLM_VISUALGEN_IPC_STAMPS')} FAST={env.get('TRTLLM_VISUALGEN_IPC_FAST')}"
        f" TOGGLE={env.get('B70_TOGGLE')} C2={env.get('B70_C2')}")
    port = free_port()
    srv_log = open(os.path.join(ART, "server.log"), "w")
    cmd = [sys.executable, "-c", LAUNCH, CKPT, "--host", "127.0.0.1", "--port", str(port),
           "--visual_gen_args", yaml_path]
    srv = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT, start_new_session=True,
                           env=env)
    result = {"passed": False}
    try:
        t0 = time.time()
        last = 0
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
            if time.time() - last > 60:
                last = time.time()
                log(f"waiting for server ({time.time() - t0:.0f}s)")
            time.sleep(3)
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
            body = json.loads(raw)
            pngs = [base64.b64decode(d["b64_json"]) for d in body["data"]]
            assert len(pngs) == 1
            return pngs[0], ms

        pids = set(descendants(srv.pid))
        mem_trace = [{"after": "ready", "mem": gpu_mem(pids)}]
        gen(PROMPTS[0], SEEDS[0])
        log("test request done")
        mem_trace.append({"after": "test", "mem": gpu_mem(pids)})
        passes, client_ms = [], []
        for p in range(2):
            imgs = []
            for i, (pr, sd) in enumerate(zip(PROMPTS, SEEDS)):
                png, ms = gen(pr, sd)
                imgs.append(png)
                client_ms.append(ms)
                mem_trace.append({"after": f"c1_p{p + 1}_{i}", "mem": gpu_mem(pids)})
                log(f"pass {p + 1} prompt {i:2d}: {ms:7.1f} ms png {len(png)} B "
                    f"sha {hashlib.sha256(png).hexdigest()[:12]}")
            passes.append(imgs)
        arrays = [[decode(b) for b in imgs] for imgs in passes]
        repeat_px = [bool(np.array_equal(a, b)) for a, b in zip(arrays[0], arrays[1])]
        repeat_png = [a == b for a, b in zip(passes[0], passes[1])]
        for i, b in enumerate(passes[0]):
            with open(os.path.join(ART, f"c1_{i:02d}.png"), "wb") as f:
                f.write(b)
        c2 = None
        if os.environ.get("B70_C2") == "1":
            c2_px, c2_png = [], []
            with cf.ThreadPoolExecutor(2) as ex:
                for rnd in range(6):
                    a, b = 2 * rnd, 2 * rnd + 1
                    fa = ex.submit(gen, PROMPTS[a], SEEDS[a])
                    fb = ex.submit(gen, PROMPTS[b], SEEDS[b])
                    for idx, fut in ((a, fa), (b, fb)):
                        png, ms = fut.result()
                        c2_px.append(bool(np.array_equal(decode(png), arrays[0][idx])))
                        c2_png.append(png == passes[0][idx])
                    mem_trace.append({"after": f"c2_r{rnd}", "mem": gpu_mem(pids)})
                    log(f"c2 round {rnd}: px {c2_px[-2:]} png {c2_png[-2:]}")
            c2 = {"pixels_equal_c1": f"{sum(c2_px)}/{len(c2_px)}",
                  "png_equal_c1": f"{sum(c2_png)}/{len(c2_png)}", "passed": all(c2_px) and all(c2_png)}
        time.sleep(1.0)
        o = read_jsonl(outer)
        st = read_jsonl(stamps)
        cons = [r for r in st if r.get("side") == "consumer"]
        prod = [r for r in st if r.get("side") == "producer"]
        o_timed = [r["to_tensor_ms"] for r in o if 1 <= r["n"] <= 24]
        o_p1 = [r["to_tensor_ms"] for r in o if 1 <= r["n"] <= 12]
        o_p2 = [r["to_tensor_ms"] for r in o if 13 <= r["n"] <= 24]
        keys = ["from_dict_ms", "lookup_ms", "ipc_open_ms", "event_ms", "view_ms", "clone_ms",
                "release_ms", "total_ms", "fast_ms", "deferred_release_ms"]
        # first consumer stamp of a real request: skip the READY handshake (no media)
        cons_timed = cons[1:25] if len(cons) >= 25 else cons[1:]
        stamp_stats = {k: stats([r[k] for r in cons_timed if k in r]) for k in keys}
        stamp_stats = {k: v for k, v in stamp_stats.items() if v}
        handles = [r.get("handle") for r in cons]
        prod_handles = [r.get("handle") for r in prod]
        mem_per_pid = {}
        for m in mem_trace:
            for pid, v in m["mem"].items():
                mem_per_pid.setdefault(pid, []).append(v)
        mem_flat = {pid: {"first_after_test": v[1] if len(v) > 1 else v[0], "last": v[-1],
                          "max": max(v), "min": min(v[1:]) if len(v) > 1 else v[0]}
                    for pid, v in mem_per_pid.items() if all(isinstance(x, int) for x in v)}
        result = {
            "passed": all(repeat_px) and all(repeat_png) and (c2 is None or c2["passed"]),
            "repeated_execution_passed": all(repeat_px),
            "repeat_pixels_equal": f"{sum(repeat_px)}/12",
            "repeat_png_bytes_equal": f"{sum(repeat_png)}/12",
            "c2": c2,
            "client_ms": stats(client_ms),
            "outer_to_tensor_ms": {"timed24": stats(o_timed), "pass1": stats(o_p1),
                                   "pass2": stats(o_p2),
                                   "first_untimed": o[0]["to_tensor_ms"] if o else None,
                                   "n_records": len(o)},
            "stamps": stamp_stats,
            "stamp_rows": len(cons),
            "consumer_handles_distinct": len(set(h for h in handles if h)),
            "producer_handles_distinct": len(set(h for h in prod_handles if h)),
            "producer_seg": sorted(set((r.get("seg_bytes"), r.get("seg_off")) for r in prod
                                       if r.get("seg_bytes") is not None))[:12],
            "producer_to_handle_ms": stats([r["to_handle_ms"] for r in prod[1:]]),
            "producer_mem_reserved": [r.get("mem_reserved") for r in prod][:3] +
                                     [r.get("mem_reserved") for r in prod][-3:],
            "gpu_mem_mib_per_pid": mem_flat,
        }
        log(f"consumer rows: {json.dumps(cons[:3])}")
        log(f"producer rows: {json.dumps(prod[:3])}")
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
