"""Idea 82: fresh-process full-model A/B/A (method 729 / pitfall 1096), env fixed at load.

Arms (each a fresh trtllm-serve VisualGen server on the benchmark engine settings with
experiments/ascent-config/engine.json overlaid):
  A  : TRTLLM_FLUX3_SM120_QKV_SCHED=pingpong  (main's 3-stream per-plane QKV GEMMs)
  B  : TRTLLM_FLUX3_SM120_QKV_SCHED=batched   (one L=3 launch, this branch's default)
  A' : pingpong again
Per arm: WARM untimed requests, then TIMED requests of one fixed prompt/seed; server denoise
from the Server-Timing header (CUDA-event GPU-stream span), NVML power / SM clock sampled at
20 ms during the timed requests. Images must be bitwise equal across all three arms (exact).
GO only if B's denoise median is below both A medians.

usage: python aba_served.py <checkpoint> <worktree>
"""
import base64
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

CKPT = sys.argv[1]
WT = sys.argv[2]
WARM = int(os.environ.get("ABA_WARM", "3"))
TIMED = int(os.environ.get("ABA_TIMED", "12"))
PROMPT = "A red fox sitting in a snowy forest at dawn, photorealistic"
SEED = 100
ARMS = [("A", "pingpong"), ("B", "batched"), ("A2", "pingpong")]


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


class Nvml:
    def __init__(self):
        self.ok = False
        try:
            import pynvml

            pynvml.nvmlInit()
            uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
            if uuid.startswith("GPU-"):
                self.h = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            else:
                self.h = pynvml.nvmlDeviceGetHandleByIndex(int(uuid or 0))
            self.nv = pynvml
            self.ok = True
        except Exception as e:  # noqa: BLE001
            log("[nvml] unavailable", repr(e))
        self.samples = []
        self._stop = None

    def start(self):
        self.samples = []
        if not self.ok:
            return
        self._stop = threading.Event()

        def run():
            while not self._stop.wait(0.02):
                try:
                    self.samples.append((self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
                                         self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)))
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        if self._stop is not None:
            self._stop.set()
            time.sleep(0.05)
        if not self.samples:
            return None
        p = [s[0] for s in self.samples]
        c = [s[1] for s in self.samples]
        return {"n": len(p), "power_w_median": round(statistics.median(p), 1),
                "power_w_max": round(max(p), 1), "sm_clock_mhz_median": statistics.median(c),
                "sm_clock_mhz_min": min(c)}


def server_args():
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
    path = os.path.join(WT, "experiments/ascent-config/engine.json")
    if os.path.exists(path):
        with open(path) as f:
            over = json.load(f)
        over = {k: v for k, v in over.items() if k != "env"}
        deep_merge(args, over.get("visual_gen_args", over))
    return args


def run_arm(tag, sched, nvml):
    args = server_args()
    tmpd = tempfile.mkdtemp()
    ypath = os.path.join(tmpd, "vg.yaml")
    with open(ypath, "w") as f:
        json.dump(args, f)
    port = free_port()
    env = dict(os.environ, TRTLLM_FLUX3_SM120_QKV_SCHED=sched, TORCH_LOGS="recompiles")
    cmd = [sys.executable, "-m", "tensorrt_llm.commands.serve", "serve", CKPT,
           "--visual_gen_args", ypath, "--host", "127.0.0.1", "--port", str(port)]
    log(f"[{tag}] sched={sched} start {' '.join(cmd)}")
    logp = os.path.join(tmpd, "server.log")
    srv_log = open(logp, "w")
    proc = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True, cwd=WT)
    base = f"http://127.0.0.1:{port}"
    out = {"tag": tag, "sched": sched}
    try:
        t0 = time.time()
        last = t0
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}: "
                                   + open(logp).read()[-3000:])
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > 2400:
                raise RuntimeError("server start timeout")
            if time.time() - last > 30:
                last = time.time()
                log(f"[{tag}] waiting {time.time() - t0:.0f}s")
            time.sleep(2)
        out["startup_s"] = round(time.time() - t0, 1)
        log(f"[{tag}] ready in {out['startup_s']}s")

        def gen():
            body = json.dumps({"prompt": PROMPT, "seed": SEED, "size": "1024x1024",
                               "num_inference_steps": 16, "response_format": "b64_json",
                               "n": 1}).encode()
            req = urllib.request.Request(base + "/v1/images/generations", data=body,
                                         headers={"Content-Type": "application/json"})
            t = time.time()
            with urllib.request.urlopen(req, timeout=600) as r:
                hdr = r.headers.get("Server-Timing", "")
                js = json.loads(r.read())
            wall = (time.time() - t) * 1000
            tim = {}
            for e in hdr.split(","):
                parts = [p.strip() for p in e.split(";")]
                for p in parts[1:]:
                    k, _, v = p.partition("=")
                    if k.strip() == "dur":
                        tim[parts[0]] = float(v)
            png = base64.b64decode(js["data"][0]["b64_json"])
            return hashlib.sha256(png).hexdigest(), tim, wall

        for i in range(WARM):
            sha, tim, wall = gen()
            log(f"[{tag}] warm {i} denoise {tim.get('denoise', -1):.2f} ms wall {wall:.0f}")
        log_mark = len(open(logp).read())
        nvml.start()
        den, shas, walls, gens = [], [], [], []
        for i in range(TIMED):
            sha, tim, wall = gen()
            den.append(tim.get("denoise", float("nan")))
            gens.append(tim.get("generation", float("nan")))
            shas.append(sha)
            walls.append(wall)
            log(f"[{tag}] timed {i} denoise {den[-1]:.2f} ms gen {gens[-1]:.1f} wall {wall:.0f}")
        out["nvml"] = nvml.stop()
        timed_log = open(logp).read()[log_mark:]
        out["timed_log_flags"] = {
            "recompiles": timed_log.count("Recompiling function") + timed_log.count("__recompiles"),
            "jit_build": sum(timed_log.count(s) for s in ("Building", "ninja", "build_and_load")),
            "autotune": timed_log.lower().count("autotun"),
        }
        xs = sorted(den)
        n = len(xs)
        out["denoise_ms"] = {"median": round(statistics.median(xs), 3),
                             "mean": round(statistics.fmean(xs), 3),
                             "p10": round(xs[int(0.1 * (n - 1))], 3),
                             "p90": round(xs[int(0.9 * (n - 1))], 3), "n": n,
                             "all": [round(x, 2) for x in den]}
        out["generation_ms_median"] = round(statistics.median(gens), 2)
        out["wall_ms_median"] = round(statistics.median(walls), 1)
        out["image_sha"] = sorted(set(shas))
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=90)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        srv_log.close()
    return out


def main():
    nvml = Nvml()
    res = {"arms": []}
    for tag, sched in ARMS:
        r = run_arm(tag, sched, nvml)
        log(json.dumps(r))
        res["arms"].append(r)
        time.sleep(10)
    a, b, a2 = (r["denoise_ms"]["median"] for r in res["arms"])
    shas = [tuple(r["image_sha"]) for r in res["arms"]]
    res["summary"] = {
        "A": a, "B": b, "A2": a2, "B_minus_meanA_ms": round(b - (a + a2) / 2, 3),
        "AA_spread_ms": round(abs(a - a2), 3),
        "go": bool(b < a and b < a2),
        "images_bitwise_across_arms": len(set(shas)) == 1 and len(shas[0]) == 1,
    }
    res["passed"] = res["summary"]["images_bitwise_across_arms"]
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
