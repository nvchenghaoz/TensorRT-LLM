#!/usr/bin/env python3
"""Host-stage analysis driver (branch 15, idea 21). Analysis only.

Usage: run.py <checkpoint> <mode> [out_dir]
  mode = timing : tokenization test, then lifetime A (ASCENT_TRACE_DIR set,
                  all cases) and lifetime B (no instrumentation, c=1 and c=4).
  mode = nsys   : one lifetime under `nsys launch`, ASCENT_TRACE_DIR +
                  ASCENT_NVTX=1, one nsys start/stop window around the cases;
                  the report is exported to sqlite and analyzed in this job.
Server: trtllm-serve (python -m tensorrt_llm.commands.serve serve), main
defaults + kv_cache_config.enable_partial_reuse=false, fresh server per
lifetime, untimed warmup at every ISL, unique exact-length prompts,
1 output token, streaming /v1/completions (as AIPerf does).
"""
import http.client
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

T0 = time.time()
OWNER = f"ascent-hs-{uuid.uuid4().hex[:10]}"
_procs = []


def log(*a):
    print(f"[+{time.time() - T0:7.1f}s]", *a, flush=True)


def _hb():
    while True:
        time.sleep(30)
        log("heartbeat")


def _cleanup(*_):
    for p in _procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass
    t = time.time()
    for p in _procs:
        try:
            p.wait(timeout=max(1, 30 - (time.time() - t)))
        except Exception:  # noqa: BLE001
            pass
    for p in _procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    # Owner-marked orphans (MPI-spawned workers) only.
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            env = open(f"/proc/{pid}/environ", "rb").read()
        except Exception:  # noqa: BLE001
            continue
        if f"TLLM_ASCENT_OWNER={OWNER}".encode() in env:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
    _procs.clear()


def _sig(signum, frame):  # noqa: ARG001
    raise SystemExit(128 + signum)


# ----------------------------------------------------------------- prompts
class Prompts:
    def __init__(self, ckpt):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
        words = []
        for s in self.tok.get_vocab():
            if s.startswith("▁"):
                w = s[1:]
                if 3 <= len(w) <= 8 and w.isascii() and w.isalpha() and w.islower():
                    words.append(w)
        words.sort()
        self.words = words
        self.seen = set()
        log(f"prompt vocab words: {len(words)}")

    def ntok(self, text):
        return len(self.tok(text)["input_ids"])

    def make(self, n, rng):
        ws = [rng.choice(self.words) for _ in range(n - 1)]
        for _ in range(80):
            text = " ".join(ws)
            L = self.ntok(text)
            if L == n and text not in self.seen:
                self.seen.add(text)
                return text
            if L > n:
                del ws[-max(1, (L - n) // 2):]
            elif L < n:
                ws.extend(rng.choice(self.words) for _ in range(max(1, (n - L) // 2)))
            else:
                ws[0] = rng.choice(self.words)
        raise RuntimeError(f"could not build an exact {n}-token prompt")


# ----------------------------------------------------------------- server
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, ckpt, out, tag, env_extra, nsys_session=None, extra_cli=None):
        self.port = free_port()
        cfg = {"kv_cache_config": {"enable_partial_reuse": False}}
        cfgp = Path(out) / f"{tag}_config.yaml"
        cfgp.write_text(json.dumps(cfg))
        cmd = [sys.executable, "-m", "tensorrt_llm.commands.serve", "serve", ckpt,
               "--host", "127.0.0.1", "--port", str(self.port), "--config", str(cfgp)] + list(extra_cli or [])
        if nsys_session:
            cmd = ["nsys", "launch", f"--session-new={nsys_session}", "--trace=cuda,nvtx"] + cmd
        env = dict(os.environ)
        env.update(env_extra)
        env["TLLM_ASCENT_OWNER"] = OWNER
        self.log_path = Path(out) / f"{tag}_server.log"
        self.lf = open(self.log_path, "w")
        log(f"[{tag}] starting server: {' '.join(cmd)}")
        log(f"[{tag}] env extra: {env_extra}")
        self.proc = subprocess.Popen(cmd, stdout=self.lf, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        _procs.append(self.proc)
        self.tag = tag
        self.t_start = time.time()

    def wait_ready(self, timeout=2700):
        last = 0
        while time.time() - self.t_start < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}\n" + tail(self.log_path))
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                c.request("GET", "/health")
                r = c.getresponse()
                r.read()
                if r.status == 200:
                    log(f"[{self.tag}] server ready after {time.time() - self.t_start:.0f}s")
                    return
            except Exception:  # noqa: BLE001
                pass
            if time.time() - last > 60:
                last = time.time()
                log(f"[{self.tag}] waiting for server... last log: {tail(self.log_path, 1)[:200]}")
            time.sleep(2)
        raise RuntimeError("server start timeout")

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        if self.proc in _procs:
            _procs.remove(self.proc)
        self.lf.close()


def tail(path, n=40):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# ----------------------------------------------------------------- client
def one_request(port, text, max_tokens=1):
    body = json.dumps({"model": "model", "prompt": text, "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}})
    rec = {"head": text[:40]}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    rec["t_send"] = time.perf_counter()
    c.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    rec["t_hdr"] = time.perf_counter()
    rec["status"] = r.status
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            rec["t_done"] = time.perf_counter()
            break
        try:
            d = json.loads(payload)
        except Exception:  # noqa: BLE001
            continue
        if "t_first" not in rec:
            rec["t_first"] = time.perf_counter()
        ch = d.get("choices") or []
        if ch and ch[0].get("text") and "t_content" not in rec:
            rec["t_content"] = time.perf_counter()
        u = d.get("usage")
        if u:
            rec["prompt_tokens"] = u.get("prompt_tokens")
            rec["completion_tokens"] = u.get("completion_tokens")
            det = u.get("prompt_tokens_details") or {}
            rec["cached_tokens"] = det.get("cached_tokens")
    c.close()
    rec.setdefault("t_done", time.perf_counter())
    rec["ttft_ms"] = 1e3 * (rec.get("t_content", rec.get("t_first", rec["t_done"])) - rec["t_send"])
    return rec


def run_case(port, texts, conc):
    import queue
    q = queue.Queue()
    for i, t in enumerate(texts):
        q.put((i, t))
    out = [None] * len(texts)

    def worker():
        while True:
            try:
                i, t = q.get_nowait()
            except queue.Empty:
                return
            try:
                out[i] = one_request(port, t)
            except Exception as e:  # noqa: BLE001
                out[i] = {"error": repr(e)}

    th = [threading.Thread(target=worker) for _ in range(conc)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    return out


ISL = {"isl300": 300, "isl1k": 1000, "isl2k": 2000, "isl4k": 4000, "isl8k": 8000}


def case_list(spec):
    out = []
    for name, nreq in spec:
        isl_lbl, c = name.split("@")
        out.append((name, ISL[isl_lbl], int(c), nreq))
    return out


def warmup(port, pf, rng):
    log("warmup start")
    for n in (300, 1000, 2000, 4000, 8000):
        for _ in range(2):
            one_request(port, pf.make(n, rng))
    run_case(port, [pf.make(300, rng) for _ in range(16)], 16)
    run_case(port, [pf.make(1000, rng) for _ in range(8)], 4)
    run_case(port, [pf.make(2000, rng) for _ in range(8)], 8)
    log("warmup done")


def run_cases(port, pf, rng, cases, tag):
    results = {}
    for name, isl, conc, nreq in cases:
        texts = [pf.make(isl, rng) for _ in range(nreq)]
        time.sleep(1.0)
        t0 = time.perf_counter()
        res = run_case(port, texts, conc)
        t1 = time.perf_counter()
        bad = [r for r in res if not r or r.get("error") or r.get("status") != 200]
        ttfts = sorted(r["ttft_ms"] for r in res if r and "ttft_ms" in r)
        med = ttfts[len(ttfts) // 2] if ttfts else None
        mean = sum(ttfts) / len(ttfts) if ttfts else None
        pt = sorted({r.get("prompt_tokens") for r in res if r}, key=str)
        ct = sorted({r.get("cached_tokens") for r in res if r}, key=str)
        log(f"[{tag}] {name}: n={len(res)} bad={len(bad)} ttft mean={mean and round(mean, 2)} "
            f"med={med and round(med, 2)} prompt_tokens={pt} cached={ct}")
        results[name] = {"isl": isl, "conc": conc, "t0": t0, "t1": t1, "requests": res}
        time.sleep(1.0)
    return results


# ----------------------------------------------------------------- tokenization test
def tokenization_test(ckpt, pf, rng):
    log("tokenization test")
    from tensorrt_llm.inputs.registry import create_input_processor
    from tensorrt_llm.sampling_params import SamplingParams
    from tensorrt_llm.tokenizer.tokenizer import tokenizer_factory
    tok = tokenizer_factory(ckpt)
    ip = create_input_processor(ckpt, tok)
    log(f"input processor: {type(ip).__name__}; tokenizer {type(tok).__name__}/{type(tok.tokenizer).__name__}")
    hf = tok.tokenizer
    import torch
    sp = SamplingParams(max_tokens=1)

    def m_serving(t):
        return ip({"prompt": t}, sp)[0]

    def m_pt_tolist(t):
        return hf(t, return_tensors="pt")["input_ids"][0].to(torch.int32).tolist()

    def m_list(t):
        return hf(t)["input_ids"]

    def m_encode(t):
        return hf.encode(t)

    def m_rust(t):
        return hf.backend_tokenizer.encode(t, add_special_tokens=True).ids

    methods = [("serving_input_processor", m_serving), ("hf_call_pt_tolist", m_pt_tolist),
               ("hf_call_list", m_list), ("hf_encode", m_encode), ("rust_backend_encode", m_rust)]
    out = {}
    for n in (300, 1000, 2000, 4000, 8000):
        texts = [pf.make(n, rng) for _ in range(6)]
        ref = [m_serving(t) for t in texts]
        row = {}
        for name, fn in methods:
            ts = []
            same = True
            try:
                for rep in range(4):
                    for i, t in enumerate(texts):
                        a = time.perf_counter()
                        ids = fn(t)
                        ts.append(1e3 * (time.perf_counter() - a))
                        if rep == 0:
                            same = same and (list(ids) == list(ref[i]))
            except Exception as e:  # noqa: BLE001
                row[name] = {"error": repr(e)[:200]}
                continue
            ts.sort()
            row[name] = {"median_ms": round(ts[len(ts) // 2], 4), "p10_ms": round(ts[len(ts) // 10], 4),
                         "p90_ms": round(ts[(9 * len(ts)) // 10], 4), "identical_to_serving": same,
                         "len": len(ref[0])}
        out[str(n)] = row
        log(f"tok n={n}: " + ", ".join(f"{k}={v.get('median_ms')}ms({'=' if v.get('identical_to_serving') else 'DIFF'})"
                                       for k, v in row.items()))
    return out


# ----------------------------------------------------------------- main
def lifetime(ckpt, out, tag, pf, rng, cases, env_extra, nsys_session=None, extra_cli=None):
    tdir = Path(out) / f"{tag}_trace"
    if env_extra.get("ASCENT_TRACE_DIR") == "@":
        tdir.mkdir(parents=True, exist_ok=True)
        env_extra = dict(env_extra, ASCENT_TRACE_DIR=str(tdir))
    srv = Server(ckpt, out, tag, env_extra, nsys_session, extra_cli)
    rep = None
    try:
        srv.wait_ready()
        if tdir.exists():
            (tdir / "ARM").write_text("1")  # GPU events only after warmup/capture
        warmup(srv.port, pf, rng)
        time.sleep(2)
        if nsys_session:
            rep = str(Path(out) / f"{tag}_nsys")
            r = subprocess.run(["nsys", "start", f"--session={nsys_session}", f"--output={rep}",
                                "--force-overwrite=true", "--sample=none", "--cpuctxsw=none"],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                log(f"nsys start failed ({r.stderr[-300:]}); retrying without sampling flags")
                r = subprocess.run(["nsys", "start", f"--session={nsys_session}", f"--output={rep}",
                                    "--force-overwrite=true"], capture_output=True, text=True, timeout=120)
            log(f"nsys start rc={r.returncode} {r.stdout[-300:]} {r.stderr[-300:]}")
            time.sleep(2)
        res = run_cases(srv.port, pf, rng, cases, tag)
        if nsys_session:
            time.sleep(2)
            r = subprocess.run(["nsys", "stop", f"--session={nsys_session}"], capture_output=True, text=True,
                               timeout=900)
            log(f"nsys stop rc={r.returncode} {r.stdout[-500:]} {r.stderr[-500:]}")
        time.sleep(3)  # let the trace flushers drain
    finally:
        srv.stop()
    (Path(out) / f"{tag}_client.json").write_text(json.dumps(res))
    return res, (tdir if tdir.exists() else None), rep


def main():
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    threading.Thread(target=_hb, daemon=True).start()
    ckpt, mode = sys.argv[1], sys.argv[2]
    out = Path(sys.argv[3] if len(sys.argv) > 3 else os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/hs_out"))
    out = out / f"host_stages_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"artifacts: {out}  owner={OWNER}")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,clocks.sm,temperature.gpu",
                            "--format=csv"], capture_output=True, text=True, timeout=30)
        log(r.stdout.strip())
        log("loadavg", open("/proc/loadavg").read().strip())
    except Exception:  # noqa: BLE001
        pass
    import analyze
    summary = {"mode": mode, "artifacts": str(out)}
    rng = random.Random(f"hs-{mode}-{time.time()}")
    try:
        pf = Prompts(ckpt)
        if mode == "timing":
            summary["tokenization"] = tokenization_test(ckpt, pf, rng)
            casesA = case_list([("isl300@1", 10), ("isl1k@1", 10), ("isl2k@1", 8), ("isl4k@1", 6),
                                ("isl8k@1", 6), ("isl300@4", 16), ("isl1k@4", 16), ("isl300@16", 32),
                                ("isl2k@64", 128)])
            resA, tdirA, _ = lifetime(ckpt, out, "A_instr", pf, rng, casesA, {"ASCENT_TRACE_DIR": "@"})
            summary["A"] = analyze.analyze_lifetime(resA, tdirA)
            casesB = case_list([("isl300@1", 10), ("isl1k@1", 10), ("isl2k@1", 8), ("isl4k@1", 6),
                                ("isl8k@1", 6), ("isl300@4", 16), ("isl1k@4", 16)])
            resB, _, _ = lifetime(ckpt, out, "B_plain", pf, rng, casesB, {})
            summary["B"] = analyze.client_only(resB)
        elif mode == "nsys":
            cases = case_list([("isl300@1", 6), ("isl1k@1", 6), ("isl8k@1", 4), ("isl300@4", 12),
                               ("isl1k@4", 12), ("isl300@16", 16), ("isl2k@64", 64)])
            env = {"ASCENT_TRACE_DIR": "@", "ASCENT_NVTX": "1", "TLLM_NVTX_DEBUG": "1"}
            try:
                sess = f"asc{uuid.uuid4().hex[:8]}"
                res, tdir, rep = lifetime(ckpt, out, "N_nsys", pf, rng, cases, env, nsys_session=sess)
                summary["topology"] = "multi-process (MPI worker, default frontends)"
            except RuntimeError as e:
                log(f"MPI-topology nsys lifetime failed ({str(e)[:300]}); falling back to single process")
                _cleanup()
                time.sleep(5)
                sess = f"asc{uuid.uuid4().hex[:8]}"
                res, tdir, rep = lifetime(ckpt, out, "N1_nsys", pf, rng, cases,
                                          dict(env, TLLM_WORKER_USE_SINGLE_PROCESS="1"), nsys_session=sess,
                                          extra_cli=["--num_serve_frontends", "1"])
                summary["topology"] = "single process worker, 1 frontend (fallback)"
            summary["N"] = analyze.analyze_lifetime(res, tdir)
            repf = (rep or "") + ".nsys-rep"
            if rep and os.path.exists(repf):
                sq = rep + ".sqlite"
                r = subprocess.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true", f"--output={sq}",
                                    repf], capture_output=True, text=True, timeout=1800)
                log(f"nsys export rc={r.returncode} {r.stderr[-300:]}")
                summary["N_gpu"] = analyze.analyze_nsys(sq, res, tdir)
            else:
                summary["N_gpu"] = {"error": f"no report at {repf}", "ls": os.listdir(out)}
    except BaseException as e:  # noqa: BLE001
        import traceback
        summary["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
        log(summary["error"])
    finally:
        _cleanup()
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    small = analyze.compact(summary)
    (out / "summary_compact.json").write_text(json.dumps(small, indent=1, default=str))
    print(json.dumps(small, default=str), flush=True)


if __name__ == "__main__":
    main()
