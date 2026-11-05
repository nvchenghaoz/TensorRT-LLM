"""Idea 228 Stage 0: in-server per-kernel diff of one c1 prefill (32768 cached + 2048 new), EP4 vs EP2xTP2.

One 4-GPU job, two server lifetimes back to back on the same GPUs (arm order given on the command line):
  ep4    : protocol server settings + this tree's engine.json with moe_expert_parallel_size=4, moe_tp removed
  ep2tp2 : same, moe_expert_parallel_size=2, moe_tensor_parallel_size=2
Per arm: start trtllm-serve, prime the 32768-token prefix, 3 warmups, N_TTFT sequential streaming requests
(max_tokens=2) timed client-side (TTFT), then a GPU-only torch-profiler window over N_PROF sequential
max_tokens=1 requests. Each rank's trace is cut into prefill steps by GPU idle gaps > GAP_US, and every step
is tabulated per kernel (base name and full name): calls, total us; plus span, busy and idle per step.
Usage: b123_prefill_trace.py <checkpoint> [arms=ep4,ep2tp2]
"""
import glob
import gzip
import json
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

PREFIX, SUFFIX = 32768, 2048
MODEL = "dsv41-flash"
N_TTFT = int(os.environ.get("B123_N_TTFT", "12"))
N_PROF = int(os.environ.get("B123_N_PROF", "3"))
GAP_US = float(os.environ.get("B123_GAP_US", "60000"))  # requests are >=300 ms apart; one segment = one request
ART = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b123"))
T0 = time.time()

# Protocol server settings (ar_trace_job.SERVER_SETTINGS, frozen harness.server), overlaid by engine.json.
BASE_SETTINGS = {
    "context_parallel_size": 1, "cuda_graph_config": {"batch_sizes": [1, 2, 4, 8], "enable_padding": True},
    "enable_attention_dp": False, "enable_chunked_prefill": True,
    "kv_cache_config": {"enable_block_reuse": True, "free_gpu_memory_fraction": 0.5, "tokens_per_block": 128},
    "max_batch_size": 8, "max_num_tokens": 8192, "max_seq_len": 36864, "moe_expert_parallel_size": 4,
    "pipeline_parallel_size": 1, "tensor_parallel_size": 4, "trust_remote_code": True}
_ENGINE = Path(__file__).resolve().parents[1] / "ascent-config" / "engine.json"
ENGINE = json.loads(_ENGINE.read_text() or "{}") if _ENGINE.exists() else {}

AR_RE = re.compile(r"nccl|allreduce|all_reduce|lamport|oneshot|twoshot|reducescatter|allgather", re.I)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.1f}s]", *a, flush=True)


def arm_settings(arm):
    s = json.loads(json.dumps(BASE_SETTINGS))
    for k, v in ENGINE.items():
        if k in ("env", "serve_args"):
            continue
        s[k] = v
    s.pop("moe_tensor_parallel_size", None)
    if arm == "ep4":
        s["moe_expert_parallel_size"] = 4
    elif arm == "ep2tp2":
        s["moe_expert_parallel_size"] = 2
        s["moe_tensor_parallel_size"] = 2
    else:
        raise SystemExit(f"unknown arm {arm}")
    return s


_rng = random.Random(1234)
PREFIX_IDS = [_rng.randrange(1000, 60000) for _ in range(PREFIX)]
_sfx = [0]


def suffix():
    i = _sfx[0]
    _sfx[0] += 1
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


_BUILDER = [None]


def real_prompt():
    """B123_REAL=1: the frozen harness's real-text prompts (common.PromptBuilder, 32768 shared + 2048 unique)."""
    if _BUILDER[0] is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ascent-benchmark"))
        import common  # noqa: E402
        tok = common.load_tokenizer(CKPT[0])
        _BUILDER[0] = common.PromptBuilder(tok, PREFIX, SUFFIX)
    i = _sfx[0]
    _sfx[0] += 1
    return _BUILDER[0].prompt(f"prof-prefill-{i}").text


CKPT = [None]
REAL = os.environ.get("B123_REAL", "0") == "1"


def stream_ttft(url, n):
    """One streaming request; returns ttft_s (first choices chunk), total_s and usage."""
    body = {"model": MODEL, "prompt": real_prompt() if REAL else PREFIX_IDS + suffix(), "max_tokens": n,
            "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttft, usage = None, None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[5:])
            except Exception:  # noqa: BLE001
                continue
            if d.get("choices") and ttft is None:
                ttft = time.perf_counter() - t0
            if d.get("usage"):
                usage = d["usage"]
    u = usage or {}
    return {"ttft_ms": round(1e3 * ttft, 2) if ttft else None, "total_ms": round(1e3 * (time.perf_counter() - t0), 2),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": u.get("completion_tokens")}


def serve_cmd(ckpt, port, cfg, s):
    exe = Path(sys.executable).parent / "trtllm-serve"
    pre = [str(exe)] if exe.exists() else [sys.executable, "-c",
                                           "import sys;from tensorrt_llm.commands.serve import main;sys.exit(main())"]
    return pre + [ckpt, "--host", "127.0.0.1", "--port", str(port), "--backend", "pytorch",
                  "--tp_size", "4", "--ep_size", str(s["moe_expert_parallel_size"]),
                  "--max_batch_size", str(s["max_batch_size"]), "--max_num_tokens", str(s["max_num_tokens"]),
                  "--max_seq_len", str(s["max_seq_len"]), "--kv_cache_free_gpu_memory_fraction", "0.5",
                  "--served_model_name", MODEL, "--config", str(cfg)]


LOG_PATTERNS = re.compile(r"moe_tp|moe_ep|moe_tensor_parallel|moe_expert_parallel|chain off|chain|inter=|"
                          r"experts=|mapping|Mapping|JIT|jit|autotun", re.I)


def log_lines(path, limit=60):
    out, seen = [], set()
    try:
        for line in Path(path).read_text(errors="replace").splitlines():
            if LOG_PATTERNS.search(line):
                key = re.sub(r"\d+", "#", line)[:200]
                if key in seen:
                    continue
                seen.add(key)
                out.append(line[:300])
                if len(out) >= limit:
                    break
    except Exception as exc:  # noqa: BLE001
        out.append(repr(exc))
    return out


def run_arm(arm, ckpt, port):
    s = arm_settings(arm)
    d = ART / "traces" / arm
    d.mkdir(parents=True, exist_ok=True)
    cfg = ART / f"{arm}_llm_api.yaml"
    cfg.write_text(json.dumps(s))
    env = dict(os.environ)
    env["TLLM_TORCH_PROFILER_DIR"] = str(d)
    env.pop("TLLM_PROFILE_START_STOP", None)
    logp = ART / f"{arm}_server.log"
    logf = open(logp, "w")
    cmd = serve_cmd(ckpt, port, cfg, s)
    log(f"[{arm}] settings {json.dumps(s)}")
    log(f"[{arm}] start: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    res = {"arm": arm, "settings": s}
    try:
        t0 = time.time()
        last = 0
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"server exited rc={p.returncode}")
            try:
                urllib.request.urlopen(f"{url}/health", timeout=5)
                break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > 2700:
                raise RuntimeError("server start timeout")
            if time.time() - last > 60:
                last = time.time()
                log(f"[{arm}] waiting for /health {time.time() - t0:.0f}s")
            time.sleep(5)
        res["startup_s"] = round(time.time() - t0, 1)
        log(f"[{arm}] healthy after {res['startup_s']}s")
        res["prime"] = stream_ttft(url, 2)
        res["warmup"] = [stream_ttft(url, 8) for _ in range(3)]
        log(f"[{arm}] prime {res['prime']} warm {res['warmup']}")
        tt = []
        for i in range(N_TTFT):
            r = stream_ttft(url, 2)
            tt.append(r)
            log(f"[{arm}] ttft {i}: {r}")
        res["ttft_requests"] = tt
        v = sorted(x["ttft_ms"] for x in tt if x["ttft_ms"])
        res["ttft_ms"] = {"mean": round(statistics.mean(v), 2), "median": round(statistics.median(v), 2),
                          "min": v[0], "max": v[-1], "n": len(v)}
        res["usage_ok"] = all(x["prompt_tokens"] == PREFIX + SUFFIX and x["cached_tokens"] == PREFIX
                              for x in tt)
        # profile window
        ctl = {}
        acts = os.environ.get("B123_ACTIVITIES", "GPU").split(",")
        ctl["start"] = http(f"{url}/start_profile", {"output_dir": str(d), "activities": acts}, timeout=120)
        time.sleep(1.0)
        prof = []
        for i in range(N_PROF):
            prof.append(stream_ttft(url, 1))
            time.sleep(0.3)
        ctl["stop"] = http(f"{url}/stop_profile", {}, timeout=300)
        res["profile_requests"] = prof
        res["profile_control"] = ctl
        log(f"[{arm}] profile {prof} {ctl}")
        deadline = time.time() + 300
        while len(glob.glob(str(d / "**" / "*.json*"), recursive=True)) < 4 and time.time() < deadline:
            time.sleep(5)
        time.sleep(15)
        res["files"] = sorted(glob.glob(str(d / "**" / "*.json*"), recursive=True))
    except Exception as exc:  # noqa: BLE001
        res["error"] = repr(exc)[:500]
        try:
            res["log_tail"] = logp.read_text(errors="replace")[-3000:]
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(120)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        logf.close()
        time.sleep(20)
    res["log_lines"] = log_lines(logp)
    return res


# ----------------------------------------------------------------------------- trace reduction
def load_kernels(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    return sorted((e for e in evs if isinstance(e, dict) and e.get("ph") == "X"
                   and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur") is not None),
                  key=lambda e: e["ts"])


def base_name(n):
    n = n.strip()
    if n.startswith("void "):
        n = n[5:]
    for ch in "<(":
        i = n.find(ch)
        if i > 0:
            n = n[:i]
    return n[:100]


def segments(kern):
    segs, cur, end = [], [], None
    for e in kern:
        if end is not None and e["ts"] - end > GAP_US:
            segs.append(cur)
            cur = []
        cur.append(e)
        t = e["ts"] + e["dur"]
        end = t if end is None else max(end, t)
    if cur:
        segs.append(cur)
    return segs


def seg_stats(seg):
    start = seg[0]["ts"]
    stop = max(e["ts"] + e["dur"] for e in seg)
    busy, cs, ce = 0.0, None, None
    for e in seg:
        s, t = e["ts"], e["ts"] + e["dur"]
        if ce is None or s > ce:
            if ce is not None:
                busy += ce - cs
            cs, ce = s, t
        else:
            ce = max(ce, t)
    busy += ce - cs
    by_base, by_full = {}, {}
    ar = 0.0
    for e in seg:
        nm = e.get("name", "")
        b = base_name(nm)
        x = by_base.setdefault(b, [0, 0.0])
        x[0] += 1
        x[1] += e["dur"]
        a = e.get("args") or {}
        fk = f"{nm[:170]} | grid={a.get('grid')} block={a.get('block')}"
        y = by_full.setdefault(fk, [0, 0.0])
        y[0] += 1
        y[1] += e["dur"]
        if AR_RE.search(nm):
            ar += e["dur"]
    sum_dur = sum(e["dur"] for e in seg)
    return {"span_us": stop - start, "busy_us": busy, "idle_us": stop - start - busy, "kernels": len(seg),
            "sum_dur_us": sum_dur, "ar_us": ar, "non_ar_us": sum_dur - ar, "by_base": by_base, "by_full": by_full}


def reduce_arm(files):
    """Per rank: the prefill steps (segments with >= 500 kernels) and their mean per-kernel table."""
    out = {}
    for f in files:
        m = re.search(r"rank-(\d+)", os.path.basename(f))
        r = int(m.group(1)) if m else len(out)
        try:
            kern = load_kernels(f)
        except Exception as exc:  # noqa: BLE001
            out[r] = {"error": repr(exc)[:300], "file": f}
            continue
        segs = segments(kern)
        info = [{"n": len(s), "span_us": round(s[-1]["ts"] + s[-1]["dur"] - s[0]["ts"], 1)} for s in segs]
        big = [s for s in segs if len(s) >= 500]
        stats = [seg_stats(s) for s in big]
        log(f"rank {r}: {len(kern)} kernels, {len(segs)} segments {info[:12]}, {len(big)} prefill steps")
        if not stats:
            out[r] = {"segments": info, "error": "no prefill step"}
            continue
        n = len(stats)
        mean = {k: sum(s[k] for s in stats) / n for k in ("span_us", "busy_us", "idle_us", "kernels", "sum_dur_us",
                                                         "ar_us", "non_ar_us")}
        tabs = {}
        for key in ("by_base", "by_full"):
            t = {}
            for s in stats:
                for k, (c, us) in s[key].items():
                    x = t.setdefault(k, [0.0, 0.0])
                    x[0] += c / n
                    x[1] += us / n
            tabs[key] = t
        out[r] = {"segments": info, "steps": n, "per_step_spans_us": [round(s["span_us"], 1) for s in stats],
                  "mean": {k: round(v, 1) for k, v in mean.items()}, **tabs}
    return out


def diff_tables(a, b, key, top=45):
    ta, tb = a.get(key, {}), b.get(key, {})
    rows = []
    for k in set(ta) | set(tb):
        ca, ua = ta.get(k, [0, 0.0])
        cb, ub = tb.get(k, [0, 0.0])
        rows.append({"kernel": k, "calls_a": round(ca, 1), "us_a": round(ua, 1), "calls_b": round(cb, 1),
                     "us_b": round(ub, 1), "delta_us": round(ub - ua, 1)})
    rows.sort(key=lambda r: -abs(r["delta_us"]))
    return rows[:top]


def main():
    ckpt = sys.argv[1]
    CKPT[0] = ckpt
    arms = (sys.argv[2] if len(sys.argv) > 2 else "ep4,ep2tp2").split(",")
    ART.mkdir(parents=True, exist_ok=True)
    log(f"engine overlay in tree: {ENGINE}; arms {arms}")
    results, reduced = {}, {}
    for i, arm in enumerate(arms):
        tag = f"{arm}#{i}" if arms.count(arm) > 1 else arm
        results[tag] = run_arm(arm, ckpt, 8711 + i)
        log(f"[{tag}] ttft {results[tag].get('ttft_ms')} err {results[tag].get('error')}")
        reduced[tag] = reduce_arm(results[tag].get("files", []))
        (ART / "b123_reduced.json").write_text(json.dumps(reduced))
        (ART / "b123_results.json").write_text(json.dumps(results, indent=1))
    summary = {"arms": {}}
    for tag, red in reduced.items():
        summary["arms"][tag] = {"ttft_ms": results[tag].get("ttft_ms"), "usage_ok": results[tag].get("usage_ok"),
                                "error": results[tag].get("error"),
                                "log_lines": results[tag].get("log_lines", [])[:40],
                                "ranks": {r: {k: v for k, v in d.items() if k not in ("by_base", "by_full")}
                                          for r, d in red.items()}}
    tags = list(reduced)
    if len(tags) >= 2:
        a, b = tags[0], tags[1]
        summary["diff"] = {"a": a, "b": b, "ranks": {}}
        for r in sorted(set(reduced[a]) & set(reduced[b])):
            ra, rb = reduced[a][r], reduced[b][r]
            if "mean" not in ra or "mean" not in rb:
                continue
            summary["diff"]["ranks"][r] = {
                "mean_delta": {k: round(rb["mean"][k] - ra["mean"][k], 1) for k in ra["mean"]},
                "by_base": diff_tables(ra, rb, "by_base", 40),
                "by_full": diff_tables(ra, rb, "by_full", 40)}
        ta, tb = results[a].get("ttft_ms") or {}, results[b].get("ttft_ms") or {}
        if ta and tb:
            summary["diff"]["ttft_delta_ms"] = {k: round(tb[k] - ta[k], 2) for k in ("mean", "median", "min")}
    (ART / "b123_summary.json").write_text(json.dumps(summary, indent=1))
    compact = {"passed": all(not results[t].get("error") for t in tags),
               "ttft": {t: results[t].get("ttft_ms") for t in tags},
               "ttft_delta_ms": (summary.get("diff") or {}).get("ttft_delta_ms"),
               "mean_delta_rank": {r: v["mean_delta"] for r, v in (summary.get("diff") or {}).get("ranks", {}).items()},
               "summary": "b123_summary.json"}
    print(json.dumps(compact))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
