"""Branch 52 (idea 104): served-path checks for items 3 and 4.

Stand-in for check.py's repeated-execution part (ruling 657), same prompts/seeds as
experiments/ascent/attend_fastpath/e2e_repeat_check.py, on trtllm-serve VisualGen with the
benchmark defaults + this tree's experiments/ascent-config/engine.json (cuda_graph_config on).

Server 1 (graphs as in engine.json, TORCH_LOGS=recompiles):
  P1 = 11 T=80 prompts, P2 = same again, L1/L2 = the T=160 prompt twice, P3 = T=80 again.
  A(T=80)/B(T=160)/A: P1 == P3 bitwise. Repeat: P1 == P2, L1 == L2 bitwise (pixels + PNG).
  Per request: Dynamo recompile lines and 'Capturing graph for key' lines from the server log.
Server 2 (same overlay, cuda_graph_config.enable=False): P1 + L1; graph-on == graph-off bitwise.

usage: python served_check.py <checkpoint> <worktree>
"""
import base64
import copy
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
import e2e_repeat_check as base_chk  # noqa: E402

PROMPTS, SEEDS = base_chk.PROMPTS, base_chk.SEEDS
log = base_chk.log
RECOMP = ("__recompiles", "Recompiling function")
CAPTURE = ("Capturing graph for key",)


def server_args(graphs_override=None):
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
    with open(os.path.join(WT, "experiments/ascent-config/engine.json")) as f:
        overlay = json.load(f)
    base_chk.deep_merge(args, copy.deepcopy(overlay))
    if graphs_override is not None:
        args["cuda_graph_config"] = {"enable": graphs_override}
    return args, overlay


def run_server(tag, args, passes):
    from PIL import Image
    import numpy as np

    log(f"[{tag}] visual_gen_args", json.dumps(args))
    tmpd = tempfile.mkdtemp()
    ypath = os.path.join(tmpd, "vg.yaml")
    with open(ypath, "w") as f:
        json.dump(args, f)
    port = base_chk.free_port()
    cmd = [sys.executable, "-m", "tensorrt_llm.commands.serve", "serve", CKPT,
           "--visual_gen_args", ypath, "--host", "127.0.0.1", "--port", str(port)]
    env = dict(os.environ)
    env["TORCH_LOGS"] = "recompiles"
    log_path = os.path.join(tmpd, "server.log")
    srv_log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd=WT, env=env)
    base = f"http://127.0.0.1:{port}"

    def log_text():
        srv_log.flush()
        with open(log_path, errors="replace") as f:
            return f.read()

    def count(txt, pats):
        return [ln for ln in txt.splitlines() if any(p in ln for p in pats)]

    results, per_request, startup = {}, [], {}
    try:
        t0 = time.time()
        last = 0
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}: {log_text()[-3000:]}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > 1500:
                raise RuntimeError("server start timeout")
            if time.time() - last > 30:
                last = time.time()
                log(f"[{tag}] waiting {time.time() - t0:.0f}s")
            time.sleep(1)
        txt = log_text()
        startup = {"ready_s": round(time.time() - t0), "recompile_msgs": len(count(txt, RECOMP)),
                   "graph_captures": len(count(txt, CAPTURE))}
        log(f"[{tag}] ready {startup}")

        def gen(i, ptag):
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
            rc, cap = count(new, RECOMP), count(new, CAPTURE)
            per_request.append({"tag": ptag, "prompt": i, "ms": round(ms),
                                "recompile_msgs": len(rc), "graph_captures": len(cap),
                                "sample": [x[-200:] for x in (rc + cap)[:3]]})
            png = base64.b64decode(js["data"][0]["b64_json"])
            px = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            log(f"[{tag}/{ptag}] prompt {i} {ms:.0f} ms recompiles {len(rc)} captures "
                f"{len(cap)} png {hashlib.sha256(png).hexdigest()[:16]}")
            return hashlib.sha256(png).hexdigest(), px

        for ptag, idxs in passes:
            results[ptag] = {i: gen(i, ptag) for i in idxs}
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
            log(f"[{tag}] log tail:\n" + open(log_path, errors="replace").read()[-2000:])
        except Exception:
            pass
    return results, per_request, startup


def cmp(a, b):
    import numpy as np

    out = []
    for i in a:
        x, y = a[i], b[i]
        eq = x[1].shape == y[1].shape and bool((x[1] == y[1]).all())
        out.append({"prompt": i, "pixels_bitwise": eq, "png_bytes_equal": x[0] == y[0],
                    "max_abs_px_diff": int(np.abs(x[1].astype(int) - y[1].astype(int)).max())})
    return out


def main():
    short = list(range(len(PROMPTS) - 1))
    long_i = len(PROMPTS) - 1
    args_on, overlay = server_args()
    on, req_on, start_on = run_server(
        "graphs_on", args_on,
        [("P1", short), ("P2", short), ("L1", [long_i]), ("L2", [long_i]), ("P3", short)])
    args_off, _ = server_args(graphs_override=False)
    off, req_off, start_off = run_server("graphs_off", args_off, [("P1", short), ("L1", [long_i])])

    r12, rL, r13 = cmp(on["P1"], on["P2"]), cmp(on["L1"], on["L2"]), cmp(on["P1"], on["P3"])
    g80, g160 = cmp(on["P1"], off["P1"]), cmp(on["L1"], off["L1"])

    def allbit(rs):
        return all(r["pixels_bitwise"] and r["png_bytes_equal"] for r in rs)

    rep = allbit(r12) and allbit(rL)
    aba = allbit(r13)
    graph_eq = allbit(g80) and allbit(g160)
    first_l1 = next(k for k, r in enumerate(req_on) if r["tag"] == "L1")
    steady = [r for k, r in enumerate(req_on) if k not in (0, first_l1)]
    captures_ok = all(r["graph_captures"] == 0 for r in steady)
    recompiles_steady = sum(r["recompile_msgs"] for r in steady)
    distinct = len({on["P1"][i][0] for i in short} | {on["L1"][long_i][0]})
    result = {
        "passed": rep and aba and graph_eq and captures_ok and recompiles_steady == 0
        and distinct == len(PROMPTS),
        "repeated_execution_passed": rep,
        "ABA_T80_T160_T80_bitwise": aba,
        "graph_on_eq_graph_off_bitwise": graph_eq,
        "graph_captures": {"startup_warmup": start_on["graph_captures"],
                           "first_T80_request": req_on[0]["graph_captures"],
                           "first_T160_request": req_on[first_l1]["graph_captures"],
                           "steady_requests": sum(r["graph_captures"] for r in steady)},
        "dynamo_recompile_msgs": {"startup_warmup": start_on["recompile_msgs"],
                                  "first_T80_request": req_on[0]["recompile_msgs"],
                                  "first_T160_request": req_on[first_l1]["recompile_msgs"],
                                  "steady_requests": recompiles_steady},
        "distinct_images": distinct,
        "P1_vs_P2": r12, "L1_vs_L2": rL, "P1_vs_P3": r13,
        "graphs_on_vs_off_T80": g80, "graphs_on_vs_off_T160": g160,
        "per_request_graphs_on": req_on, "per_request_graphs_off": req_off,
        "startup": {"on": start_on, "off": start_off},
        "engine_overlay": overlay,
        "note": "stand-in for check.py repeated execution (ruling 657); gate v2 vs reference.json "
                "is not shipped to worker jobs and runs at landing",
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
