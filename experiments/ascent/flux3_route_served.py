"""Route check in a SERVED request (idea 110 acceptance item 1). Diagnostic only.

Starts ONE trtllm-serve VisualGen server on the benchmark defaults with
experiments/ascent-config/engine.json overlaid (as e2e_repeat_check.py does), with a
sitecustomize.py (run_adapter.SITE, on PYTHONPATH for this job only) that wraps
flashinfer.run_fmha_v2_fp8_sm12x in every server process and logs, per call, S, causal/LSE,
the contiguity and strides of the E4M3 Q/K/V it receives from
trtllm::visual_gen_fp8_attention_sm120's K2 and whether the branch's guard routes it to
fmha_v2_flux3_b2_ms_rq_*. Sends 2 requests (1024^2, 16 steps). Run it under profile_timeline
to get the kernel list. Nothing in the source tree changes.

usage: python flux3_route_served.py <checkpoint> <worktree>
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_adapter import SITE  # noqa: E402

CKPT, WT = sys.argv[1], sys.argv[2]


def log(*a):
    print(*a, flush=True)


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
        "torch_compile_config": {"enable": True, "enable_autotune": True, "enable_fullgraph": False},
    }
    overlay_path = os.path.join(WT, "experiments/ascent-config/engine.json")
    overlay = json.load(open(overlay_path)) if os.path.exists(overlay_path) else {}
    env_over = overlay.pop("env", {}) if isinstance(overlay.get("env"), dict) else {}
    vga = overlay.pop("visual_gen_args", None)
    deep_merge(args, vga if isinstance(vga, dict) else overlay)
    log("[cfg] engine overlay", json.dumps(overlay if vga is None else vga), "env", env_over)
    log("[cfg] visual_gen_args", json.dumps(args))
    try:
        sha = subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = None
    log("[cfg] worktree git sha", sha)
    import tensorrt_llm._torch.visual_gen.attention.backends.flashinfer as fib

    log("[cfg] flashinfer backend", fib.__file__, "b2 flag", getattr(fib, "_USE_FMHA_V2_FLUX3_B2", None))
    tmpd = tempfile.mkdtemp()
    with open(os.path.join(tmpd, "sitecustomize.py"), "w") as f:
        f.write(SITE)
    rl = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS") or tmpd, "flux3_route_log.jsonl")
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in env_over.items()})
    env["FLUX3_ROUTE_LOG"] = rl
    env["PYTHONPATH"] = tmpd + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    ypath = os.path.join(tmpd, "vg.yaml")
    with open(ypath, "w") as f:
        json.dump(args, f)
    s = __import__("socket").socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cmd = [sys.executable, "-m", "tensorrt_llm.commands.serve", "serve", CKPT,
           "--visual_gen_args", ypath, "--host", "127.0.0.1", "--port", str(port)]
    log("[srv]", " ".join(cmd))
    srv_log = open(os.path.join(tmpd, "server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT, start_new_session=True,
                            cwd=WT, env=env)
    base = f"http://127.0.0.1:{port}"
    lat = []
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
            if time.time() - t0 > 1800:
                raise RuntimeError("server start timeout")
            if time.time() - last > 30:
                last = time.time()
                log(f"[srv] waiting {time.time() - t0:.0f}s")
            time.sleep(1)
        log(f"[srv] ready in {time.time() - t0:.0f}s")
        for i in range(2):
            body = json.dumps({"prompt": "A red fox sitting in a snowy forest at dawn, photorealistic",
                               "seed": 100 + i, "size": "1024x1024", "num_inference_steps": 16,
                               "response_format": "b64_json", "n": 1}).encode()
            req = urllib.request.Request(base + "/v1/images/generations", data=body,
                                         headers={"Content-Type": "application/json"})
            t = time.time()
            with urllib.request.urlopen(req, timeout=900) as r:
                r.read()
            lat.append((time.time() - t) * 1000)
            log(f"[req {i}] {lat[-1]:.0f} ms")
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
    recs = []
    if os.path.exists(rl):
        recs = [json.loads(x) for x in open(rl).read().splitlines() if x.strip()]
    calls = [r for r in recs if "call" in r]
    counts = {}
    for r in calls:
        k = (f"S={r['S']} causal={r['causal']} lse={r['lse']} contig={r['contiguous_qkv']} "
             f"b2={r['guard_routes_b2']}")
        counts[k] = counts.get(k, 0) + 1
    for r in calls[:8]:
        log(json.dumps(r))
    summ = [r for r in recs if "counts" in r]
    res = {"passed": bool(calls) and all(r["guard_routes_b2"] for r in calls
                                          if r["head_dim"] == 128 and not r["causal"] and not r["lse"]),
           "patched": [r for r in recs if "patched" in r][:4],
           "first_calls_counts": counts, "periodic_counts": summ[-1] if summ else None,
           "first_calls": calls[:4], "request_ms": lat, "git_sha": sha}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
