"""In-server decode AR trace, two arms in one 4-GPU job, without the frozen adapter (pitfall 784, method 797).

Arm A (main's AR path): TLLM_PCIE_CUSTOM_AR_MAX_BYTES=0 and TLLM_NCCL_PCIE_DEFAULTS=0 -> all NCCL, main's NCCL env.
Arm B (branch default): lamport ONESHOT up to 512 KiB, branch NCCL defaults.
Each arm: one trtllm-serve lifetime from this tree with the protocol's server settings (no engine overlay on this
branch), prime the 32768-token prefix, 2 warmups, then c1 and c4 decode windows traced with POST /start_profile
{num_steps: 20, start_step: 2, activities: [GPU]} started after every request has its first token. Prompts are token
id lists: fixed-seed 32768-token prefix + unique 2048-token suffix (unique first token), 256 tokens out, ignore_eos.
Reduced in this process (no pool) with ar_trace_job.analyze_window. Usage: ar_serve_trace.py <checkpoint>
"""
import glob
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ar_trace_job as atj  # noqa: E402

STEPS = 20
atj.DECODE_STEPS = STEPS
PREFIX, SUFFIX, OUT = 32768, 2048, 256
MODEL = "dsv41-flash"
SETTINGS = dict(atj.SERVER_SETTINGS)
ART = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/ar_serve_trace"))
log = atj.log

_rng = random.Random(1234)
PREFIX_IDS = [_rng.randrange(1000, 60000) for _ in range(PREFIX)]
_suffix_idx = [0]


def suffix():
    i = _suffix_idx[0]
    _suffix_idx[0] += 1
    r = random.Random(10_000 + i)
    return [70000 + i] + [r.randrange(1000, 60000) for _ in range(SUFFIX - 1)]


def http(url, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode(errors="replace")
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        return {"text": txt[:300]}


def stream(url, ids, n, first, out):
    body = {"model": MODEL, "prompt": ids, "max_tokens": n, "temperature": 0.0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    usage, chunks = None, 0
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for line in resp:
                line = line.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    d = json.loads(line[5:])
                except Exception:  # noqa: BLE001
                    continue
                if d.get("choices"):
                    chunks += 1
                    if chunks >= 2:
                        first.set()
                if d.get("usage"):
                    usage = d["usage"]
    except Exception as exc:  # noqa: BLE001
        out.append({"error": repr(exc)[:300]})
    finally:
        first.set()
    u = usage or {}
    out.append({"chunks": chunks, "prompt_tokens": u.get("prompt_tokens"),
                "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "completion_tokens": u.get("completion_tokens")})


def run_requests(url, c, profile_dir=None):
    firsts = [threading.Event() for _ in range(c)]
    outs = []
    ths = [threading.Thread(target=stream, args=(url, PREFIX_IDS + suffix(), OUT, firsts[i], outs)) for i in range(c)]
    for t in ths:
        t.start()
    for e in firsts:
        e.wait(300)
    ctl = {}
    if profile_dir is not None:
        time.sleep(0.3)
        try:
            ctl["start"] = http(f"{url}/start_profile", {"output_dir": str(profile_dir), "num_steps": STEPS,
                                                        "start_step": 2, "activities": ["GPU"]}, timeout=120)
        except Exception as exc:  # noqa: BLE001
            ctl["start"] = {"error": repr(exc)[:300]}
    for t in ths:
        t.join(900)
    if profile_dir is not None:
        try:
            ctl["stop"] = http(f"{url}/stop_profile", {}, timeout=120)
        except Exception as exc:  # noqa: BLE001
            ctl["stop"] = {"error": repr(exc)[:300]}
    ok = all(o.get("prompt_tokens") == PREFIX + SUFFIX and o.get("cached_tokens") == PREFIX
             and o.get("completion_tokens") == OUT for o in outs)
    return {"requests": outs, "usage_ok": ok, "control": ctl}


def serve_cmd(ckpt, port, cfg):
    exe = Path(sys.executable).parent / "trtllm-serve"
    pre = [str(exe)] if exe.exists() else [sys.executable, "-c",
                                           "import sys;from tensorrt_llm.commands.serve import main;sys.exit(main())"]
    s = SETTINGS
    return pre + [ckpt, "--host", "127.0.0.1", "--port", str(port), "--backend", "pytorch",
                  "--tp_size", "4", "--ep_size", "4", "--max_batch_size", str(s["max_batch_size"]),
                  "--max_num_tokens", str(s["max_num_tokens"]), "--max_seq_len", str(s["max_seq_len"]),
                  "--kv_cache_free_gpu_memory_fraction", "0.5", "--served_model_name", MODEL,
                  "--config", str(cfg)]


def run_arm(name, env_extra, ckpt, port):
    d = ART / "traces" / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = ART / f"{name}_llm_api.yaml"
    cfg.write_text(json.dumps(SETTINGS))  # JSON is valid YAML
    env = dict(os.environ)
    env.update(env_extra)
    env["TLLM_TORCH_PROFILER_DIR"] = str(d)
    env.pop("TLLM_PROFILE_START_STOP", None)
    logf = open(ART / f"{name}_server.log", "w")
    cmd = serve_cmd(ckpt, port, cfg)
    log(f"[{name}] start server env={env_extra}: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    res = {"env": env_extra}
    try:
        t0 = time.time()
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"server exited rc={p.returncode}")
            try:
                urllib.request.urlopen(f"{url}/health", timeout=5)
                break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > 2400:
                raise RuntimeError("server start timeout")
            if int(time.time() - t0) % 60 < 5:
                log(f"[{name}] waiting for /health {time.time() - t0:.0f}s")
            time.sleep(5)
        log(f"[{name}] healthy after {time.time() - t0:.0f}s")
        res["prime"] = run_requests(url, 1)
        res["warmup"] = run_requests(url, 2)
        log(f"[{name}] primed/warm: {res['prime']['requests']} {res['warmup']['requests']}")
        for c in (1, 4):
            wd = d / f"decode_c{c}"
            wd.mkdir(parents=True, exist_ok=True)
            r = run_requests(url, c, wd)
            log(f"[{name}] c{c}: {r}")
            deadline = time.time() + 300
            while not glob.glob(str(wd / "*.json*")) and time.time() < deadline:
                time.sleep(5)
            time.sleep(10)
            r["files"] = sorted(os.path.basename(f) for f in glob.glob(str(wd / "*")))
            res[f"c{c}"] = r
    except Exception as exc:  # noqa: BLE001
        res["error"] = repr(exc)[:500]
        try:
            res["log_tail"] = (ART / f"{name}_server.log").read_text(errors="replace")[-3000:]
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(90)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        logf.close()
        time.sleep(15)
    return res


def gpu_busy_and_graph(files):
    """Per rank: GPU-busy us per step (union of kernel intervals / STEPS) and graph-replay evidence of AR kernels."""
    out = {}
    for f in files:
        m = re.search(r"rank-(\d+)", os.path.basename(f))
        r = int(m.group(1)) if m else len(out)
        evs = atj.load_trace(f)
        iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs)
        busy, cs, ce = 0.0, None, None
        for s, e in iv:
            if ce is None or s > ce:
                if ce is not None:
                    busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        if ce is not None:
            busy += ce - cs
        span = (iv[-1][1] - iv[0][0]) if iv else 0
        ar = [e for e in evs if atj.AR_RE.search(e.get("name", ""))]
        names = {}
        for e in ar:
            k = ("lamport" if "lamport" in e["name"] else "nccl" if "nccl" in e["name"].lower() else "other")
            names[k] = names.get(k, 0) + 1
        graph_keys = {}
        for e in ar:
            for k, v in (e.get("args") or {}).items():
                if "graph" in k.lower():
                    graph_keys.setdefault(k, set()).add(str(v))
        lam = [e["dur"] for e in ar if "lamport" in e["name"]]
        nc = [e["dur"] for e in ar if "nccl" in e["name"].lower()]

        def st(x):
            x = sorted(x)
            return {"n": len(x), "per_step": round(len(x) / STEPS, 1),
                    "mean_us": round(sum(x) / len(x), 2) if x else None, "p50_us": x[len(x) // 2] if x else None}
        out[r] = {"kernels": len(evs), "gpu_busy_us_per_step": round(busy / STEPS, 1),
                  "span_us_per_step": round(span / STEPS, 1),
                  "ar_us_per_step": round(sum(e["dur"] for e in ar) / STEPS, 1),
                  "ar_kinds": names, "lamport": st(lam), "nccl": st(nc),
                  "ar_graph_arg_keys": {k: len(v) for k, v in graph_keys.items()},
                  "ar_with_graph_arg": sum(1 for e in ar if any("graph" in k.lower() for k in (e.get("args") or {}))),
                  "ar_names": sorted({e["name"][:90] for e in ar})[:6]}
    return out


def successors(files):
    """Per rank: the kernel launched right after each AR on the same stream (e.g. fused_hc / post_mapping)."""
    out = {}
    for f in files:
        m = re.search(r"rank-(\d+)", os.path.basename(f))
        r = int(m.group(1)) if m else len(out)
        evs = atj.load_trace(f)
        by_stream = {}
        for e in evs:
            by_stream.setdefault((e.get("pid"), e.get("tid")), []).append(e)
        nxt = {}
        for lst in by_stream.values():
            lst.sort(key=lambda e: e["ts"])
            for a, b in zip(lst, lst[1:]):
                if atj.AR_RE.search(a.get("name", "")) and not atj.AR_RE.search(b.get("name", "")):
                    k = b["name"][:80]
                    d = nxt.setdefault(k, {"n": 0, "sum_us": 0.0, "gap_sum_us": 0.0})
                    d["n"] += 1
                    d["sum_us"] += b["dur"]
                    d["gap_sum_us"] += b["ts"] - (a["ts"] + a["dur"])
        out[r] = sorted(({"name": k, "n": v["n"], "mean_us": round(v["sum_us"] / v["n"], 2),
                          "mean_gap_after_ar_us": round(v["gap_sum_us"] / v["n"], 2)} for k, v in nxt.items()),
                        key=lambda x: -x["n"])[:6]
    return out


def main():
    ckpt = sys.argv[1]
    ART.mkdir(parents=True, exist_ok=True)
    arm_set = sys.argv[2] if len(sys.argv) > 2 else "nccl_vs_branch"
    if arm_set == "sm120_launch":
        # idea 70: arm A = branch 22's launch as landed on main (1 CTA per token, no PDL attribute), selected by the
        # branch's env switches, which make the launcher identical to main's; arm B = the branch default.
        arms = [("armA_branch22_launch", {"TLLM_AR_SM120_PDL": "0", "TLLM_AR_SM120_CTAS_PER_TOKEN": "1"}),
                ("armB_candidate", {})]
    else:
        arms = [("armA_main_ar", {"TLLM_PCIE_CUSTOM_AR_MAX_BYTES": "0", "TLLM_NCCL_PCIE_DEFAULTS": "0"}),
                ("armB_branch", {})]
    summary = {"prompt_shape": f"{PREFIX} shared prefix (primed) + {SUFFIX} unique suffix, {OUT} out, token ids",
               "steps_per_window": STEPS, "arms": {}}
    for i, (name, env) in enumerate(arms):
        summary["arms"][name] = {"run": run_arm(name, env, ckpt, 8311 + i)}
    log("reducing traces (single process)")
    for name, _ in arms:
        for c in (1, 4):
            wd = ART / "traces" / name / f"decode_c{c}"
            files = sorted(f for f in glob.glob(str(wd / "**" / "*.json*"), recursive=True))
            log(f"{name} c{c}: {len(files)} files {[os.path.basename(f) for f in files]}")
            w = {}
            try:
                w["ar_match"] = atj.analyze_window(files)
            except Exception as exc:  # noqa: BLE001
                w["ar_match"] = {"error": repr(exc)[:300]}
            try:
                w["per_rank"] = gpu_busy_and_graph(files)
            except Exception as exc:  # noqa: BLE001
                w["per_rank"] = {"error": repr(exc)[:300]}
            try:
                w["after_ar"] = successors(files)
            except Exception as exc:  # noqa: BLE001
                w["after_ar"] = {"error": repr(exc)[:300]}
            summary["arms"][name][f"c{c}"] = w
            log(f"{name} c{c}: {json.dumps(w)[:3000]}")
    (ART / "ar_serve_trace_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    ok = all(isinstance(summary["arms"][n].get(f"c{c}", {}).get("ar_match"), dict)
             and "by_class" in summary["arms"][n][f"c{c}"]["ar_match"] for n, _ in arms for c in (1, 4))
    print(json.dumps({"passed": ok, **summary}, default=str))


if __name__ == "__main__":
    main()
