"""Branch 44 (idea 92): served-path stand-in for check.py's repeated-execution part, plus
Dynamo recompile counts per phase on the running server.

Same protocol as experiments/ascent/attend_fastpath/e2e_repeat_check.py (ruling 657): ONE
trtllm-serve VisualGen server on the benchmark defaults + experiments/ascent-config/engine.json,
12 prompts (seeds 100-111, 1024x1024, 16 steps), P1/P2 = 11 T=80 prompts twice, L1/L2 = the
T=160 prompt twice, P3 = T=80 prompts after T=160. Pass 2 == pass 1 bitwise on pixels and PNG
bytes. Additionally the server runs with TORCH_LOGS=recompiles and the server log is split at
each request, so the number of Dynamo recompile messages is reported for the first request at
T=80, the first request at T=160, and all later requests. FLUX3_ROPE_TABLE stays at its default
(True) as served.

usage: python b44_served_check.py <checkpoint> <worktree>
"""
import base64
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

CKPT = sys.argv[1]
WT = sys.argv[2]
sys.path.insert(0, os.path.join(WT, "experiments/ascent/attend_fastpath"))
sys.argv = [sys.argv[0], CKPT, WT]
import e2e_repeat_check as base_chk  # noqa: E402  (only PROMPTS/SEEDS/helpers; main not run)

PROMPTS, SEEDS = base_chk.PROMPTS, base_chk.SEEDS
log = base_chk.log
RECOMP_PATTERNS = ("__recompiles", "Recompiling function")


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
    base_chk.deep_merge(args, overlay)
    log("[cfg] visual_gen_args", json.dumps(args))
    tmpd = tempfile.mkdtemp()
    ypath = os.path.join(tmpd, "vg.yaml")
    with open(ypath, "w") as f:
        json.dump(args, f)
    port = base_chk.free_port()
    cmd = [sys.executable, "-m", "tensorrt_llm.commands.serve", "serve", CKPT,
           "--visual_gen_args", ypath, "--host", "127.0.0.1", "--port", str(port)]
    env = dict(os.environ)
    env["TORCH_LOGS"] = "recompiles"
    env.pop("FLUX3_ROPE_TABLE", None)
    log("[srv]", " ".join(cmd), "TORCH_LOGS=recompiles")
    log_path = os.path.join(tmpd, "server.log")
    srv_log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd=WT, env=env)
    base = f"http://127.0.0.1:{port}"
    result = {"passed": False}

    def log_text():
        srv_log.flush()
        with open(log_path, errors="replace") as f:
            return f.read()

    def recomp_lines(txt):
        return [ln for ln in txt.splitlines() if any(p in ln for p in RECOMP_PATTERNS)]

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
        recomp_warmup = len(recomp_lines(log_text()))
        log(f"[dynamo] recompile messages during startup/warmup: {recomp_warmup}")
        from PIL import Image
        import numpy as np

        per_request = []

        def gen(i, tag):
            before = len(log_text())
            body = json.dumps({"prompt": PROMPTS[i], "seed": SEEDS[i], "size": "1024x1024",
                               "num_inference_steps": 16, "response_format": "b64_json",
                               "n": 1}).encode()
            req = urllib.request.Request(base + "/v1/images/generations", data=body,
                                         headers={"Content-Type": "application/json"})
            t = time.time()
            with urllib.request.urlopen(req, timeout=600) as r:
                js = json.loads(r.read())
            ms = (time.time() - t) * 1000
            time.sleep(0.5)
            rl = recomp_lines(log_text()[before:])
            per_request.append({"tag": tag, "prompt": i, "ms": round(ms), "recompile_msgs": len(rl),
                                "sample": [x[-240:] for x in rl[:4]]})
            png = base64.b64decode(js["data"][0]["b64_json"])
            px = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            return png, px, ms, len(rl)

        short = list(range(len(PROMPTS) - 1))
        long_i = len(PROMPTS) - 1

        def run_pass(tag, idxs):
            out = {}
            for i in idxs:
                png, px, ms, nrc = gen(i, tag)
                out[i] = (hashlib.sha256(png).hexdigest(), px)
                log(f"[{tag}] prompt {i} seed {SEEDS[i]} {ms:.0f} ms recompiles {nrc} png_sha "
                    f"{out[i][0][:16]} px_sha {hashlib.sha256(px.tobytes()).hexdigest()[:16]}")
            return out

        p1 = run_pass("P1", short)
        p2 = run_pass("P2", short)
        l1 = run_pass("L1", [long_i])
        l2 = run_pass("L2", [long_i])
        p3 = run_pass("P3", short)

        def cmp(a, b, idxs):
            per = []
            for i in idxs:
                x, y = a[i], b[i]
                eq = x[1].shape == y[1].shape and bool((x[1] == y[1]).all())
                per.append({"prompt": i, "pixels_bitwise": eq, "png_bytes_equal": x[0] == y[0],
                            "png_sha256": x[0],
                            "max_abs_px_diff": int(np.abs(x[1].astype(int) - y[1].astype(int)).max())})
            return per

        r12 = cmp(p1, p2, short)
        rL = cmp(l1, l2, [long_i])
        r13 = cmp(p1, p3, short)
        rep = all(x["pixels_bitwise"] and x["png_bytes_equal"] for x in r12 + rL)
        distinct = len({p1[i][0] for i in short} | {l1[long_i][0]})
        first_t80 = per_request[0]["recompile_msgs"]
        first_t160 = next(r["recompile_msgs"] for r in per_request if r["tag"] == "L1")
        t80_rest_before_long = sum(r["recompile_msgs"] for r in per_request[1:]
                                   if r["tag"] in ("P1", "P2"))
        after_long = sum(r["recompile_msgs"] for r in per_request if r["tag"] in ("L2", "P3"))
        result = {
            "passed": rep and distinct == len(PROMPTS),
            "repeated_execution_passed": rep,
            "T80_repeat_bitwise": sum(x["pixels_bitwise"] for x in r12),
            "T160_repeat_bitwise": sum(x["pixels_bitwise"] for x in rL),
            "T80_after_T160_bitwise_count": sum(x["pixels_bitwise"] for x in r13),
            "prompts": len(PROMPTS),
            "distinct_images": distinct,
            "dynamo_recompile_msgs": {
                "startup_warmup": recomp_warmup,
                "first_T80_request": first_t80,
                "other_T80_requests_before_T160": t80_rest_before_long,
                "first_T160_request": first_t160,
                "requests_after_first_T160": after_long,
            },
            "per_request": per_request,
            "P1_vs_P2": r12, "L1_vs_L2": rL, "P1_vs_P3": r13,
            "engine_overlay": overlay,
            "FLUX3_ROPE_TABLE": "default (True)",
            "note": "stand-in for check.py repeated-execution (ruling 657); gate v2 vs "
                    "reference.json needs the reference store, which is not shipped to worker jobs",
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
            txt = open(log_path, errors="replace").read()
            log(f"[srv] total recompile lines: {len(recomp_lines(txt))}")
            log("[srv] log tail:\n" + txt[-2500:])
        except Exception:
            pass
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
