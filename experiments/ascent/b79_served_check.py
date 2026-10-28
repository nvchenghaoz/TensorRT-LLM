"""Branch 79 (idea 123): served A/B/A repeated-execution check + per-phase Dynamo recompile
counts, total and for the final layer (Flux3LastLayer.forward) alone.

Same protocol as b44_served_check.py (ruling 657 stand-in): ONE trtllm-serve VisualGen server on
the benchmark defaults + experiments/ascent-config/engine.json, TORCH_LOGS=recompiles,
P1/P2 = 11 T=80 prompts twice, L1/L2 = the T=160 prompt twice, P3 = T=80 prompts after T=160.
Pass: P2 == P1 and L2 == L1 bitwise (pixels + PNG bytes); P3 == P1 reported; final-layer
recompile lines are those naming transformer_flux3.py:<line of Flux3LastLayer.forward's def>.
Run it on main and on the branch (run_gpu_pair) to compare.

usage: python b79_served_check.py <checkpoint> <worktree>
"""
import base64
import hashlib
import io
import json
import os
import re
import shutil
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
import e2e_repeat_check as base_chk  # noqa: E402

PROMPTS, SEEDS = base_chk.PROMPTS, base_chk.SEEDS
log = base_chk.log
RECOMP_PATTERNS = ("__recompiles", "Recompiling function")


def final_layer_marker():
    path = os.path.join(WT, "tensorrt_llm/_torch/visual_gen/models/flux3/transformer_flux3.py")
    lines = open(path).read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("class Flux3LastLayer"))
    for i in range(start, len(lines)):
        if re.match(r"\s+def forward\(", lines[i]):
            return f"transformer_flux3.py:{i + 1}"
    raise RuntimeError("Flux3LastLayer.forward not found")


def main():
    marker = final_layer_marker()
    log("[cfg] final-layer recompile marker", marker)
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
    overlay = json.load(open(overlay_path)) if os.path.exists(overlay_path) else {}
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

    def fl_lines(txt):
        return [ln for ln in recomp_lines(txt) if marker in ln]

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
        txt = log_text()
        warm_all, warm_fl = len(recomp_lines(txt)), len(fl_lines(txt))
        compiled_fl = "torch.compile: transformer.final_layer" in txt
        log(f"[dynamo] startup/warmup recompile lines: all {warm_all}, final layer {warm_fl}; "
            f"final layer compiled: {compiled_fl}")
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
            new = log_text()[before:]
            ra, rf = recomp_lines(new), fl_lines(new)
            per_request.append({"tag": tag, "prompt": i, "ms": round(ms),
                                "recompile_lines": len(ra), "final_layer_recompile_lines": len(rf),
                                "final_layer_sample": [x[-240:] for x in rf[:3]]})
            png = base64.b64decode(js["data"][0]["b64_json"])
            px = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            return png, px, ms, len(ra), len(rf)

        short = list(range(len(PROMPTS) - 1))
        long_i = len(PROMPTS) - 1

        def run_pass(tag, idxs):
            out = {}
            for i in idxs:
                png, px, ms, na, nf = gen(i, tag)
                out[i] = (hashlib.sha256(png).hexdigest(), px)
                log(f"[{tag}] prompt {i} seed {SEEDS[i]} {ms:.0f} ms recompiles {na} "
                    f"final-layer {nf} png_sha {out[i][0][:16]}")
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
                            "max_abs_px_diff":
                                int(np.abs(x[1].astype(int) - y[1].astype(int)).max())})
            return per

        r12, rL, r13 = cmp(p1, p2, short), cmp(l1, l2, [long_i]), cmp(p1, p3, short)
        rep = all(x["pixels_bitwise"] and x["png_bytes_equal"] for x in r12 + rL)
        distinct = len({p1[i][0] for i in short} | {l1[long_i][0]})

        def phase(key):
            return {
                "first_T80": per_request[0][key],
                "other_T80_before_T160": sum(r[key] for r in per_request[1:]
                                             if r["tag"] in ("P1", "P2")),
                "first_T160": next(r[key] for r in per_request if r["tag"] == "L1"),
                "after_first_T160": sum(r[key] for r in per_request
                                        if r["tag"] in ("L2", "P3")),
            }

        fl = phase("final_layer_recompile_lines")
        p3_eq = sum(x["pixels_bitwise"] for x in r13)
        result = {
            "passed": rep and distinct == len(PROMPTS),
            "repeated_execution_passed": rep,
            "T80_repeat_bitwise": sum(x["pixels_bitwise"] for x in r12),
            "T160_repeat_bitwise": sum(x["pixels_bitwise"] for x in rL),
            "T80_after_T160_bitwise_count": p3_eq,
            "P3_equals_P1_all": p3_eq == len(short),
            "prompts": len(PROMPTS),
            "distinct_images": distinct,
            "final_layer_compiled": compiled_fl,
            "final_layer_marker": marker,
            "recompile_lines_all": {"startup_warmup": warm_all,
                                    **phase("recompile_lines")},
            "recompile_lines_final_layer": {"startup_warmup": warm_fl, **fl},
            "final_layer_recompiles_after_warmup": sum(fl.values()),
            "mean_ms_T80_P2": round(sum(r["ms"] for r in per_request if r["tag"] == "P2")
                                    / len(short), 1),
            "per_request": per_request,
            "P1_vs_P2": r12, "L1_vs_L2": rL, "P1_vs_P3": r13,
            "engine_overlay": overlay,
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
            art = os.environ.get("AUTO_DEV_ARTIFACTS")
            if art:
                os.makedirs(art, exist_ok=True)
                shutil.copy(log_path, os.path.join(art, "server.log"))
            txt = open(log_path, errors="replace").read()
            log(f"[srv] total recompile lines: {len(recomp_lines(txt))}, "
                f"final layer: {len(fl_lines(txt))}")
            log("[srv] log tail:\n" + txt[-2000:])
        except Exception:
            pass
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
