#!/usr/bin/env python3
"""In-server tokenization stage + TTFT (branch 32, idea 62). Same script on base and candidate.

Usage: serve_tok.py <checkpoint> [n_per_case]
One fresh trtllm-serve per arm (main defaults + kv_cache_config block reuse off, so re-sent
prompts re-run the fresh prefill; env TRTLLM_FI_HOST_PREFILL_PLAN=1), with the experiment-only
sitecustomize hook (site/) that times Gemma4InputProcessor.call_with_text_prompt inside the
serving frontend and records live piece-pool threads + TOKENIZERS_PARALLELISM.
Unique workload prompts (benchmark PromptFactory), untimed warmup, cases isl{300,1k,2k,4k,8k}@1
and isl8k@4 with cached_tokens checked, then the re-send check: 2 prompts per ISL sent with
top-20 logprobs, re-sent alone in the same server, compared bitwise.
"""
import http.client
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "host_stages"))
import run as hs  # noqa: E402

sys.path.insert(0, str(HERE))
from tok_bench import load_common  # noqa: E402

SERVER_CFG = {"kv_cache_config": {"enable_partial_reuse": False, "enable_block_reuse": False}}
SERVER_ENV = {"TRTLLM_FI_HOST_PREFILL_PLAN": "1"}


def log(*a):
    hs.log(*a)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


class Server(hs.Server):
    def __init__(self, ckpt, out, tag, env_extra):  # same as hs.Server, own config
        self.port = hs.free_port()
        cfgp = Path(out) / f"{tag}_config.yaml"
        cfgp.write_text(json.dumps(SERVER_CFG))
        cmd = [sys.executable, "-m", "tensorrt_llm.commands.serve", "serve", ckpt,
               "--host", "127.0.0.1", "--port", str(self.port), "--config", str(cfgp)]
        env = dict(os.environ)
        env.update(env_extra)
        env["TLLM_ASCENT_OWNER"] = hs.OWNER
        self.log_path = Path(out) / f"{tag}_server.log"
        self.lf = open(self.log_path, "w")
        log(f"[{tag}] starting server: {' '.join(cmd)} config={SERVER_CFG} env_extra={env_extra}")
        self.proc = subprocess.Popen(cmd, stdout=self.lf, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        hs._procs.append(self.proc)
        self.tag = tag
        self.t_start = time.time()


def logprob_request(port, text):
    body = json.dumps({"model": "model", "prompt": text, "max_tokens": 1, "temperature": 0.0,
                       "logprobs": 20})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    d = json.loads(r.read())
    c.close()
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    usage = d.get("usage") or {}
    return {"text": ch.get("text"), "tokens": lp.get("tokens"), "token_logprobs": lp.get("token_logprobs"),
            "top_logprobs": lp.get("top_logprobs"), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def main():
    ckpt = sys.argv[1]
    nper = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    out = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)
    toklog = out / "toklog"
    toklog.mkdir(exist_ok=True)
    threading.Thread(target=hs._hb, daemon=True).start()
    common = load_common()
    tok = common.load_tokenizer(ckpt)
    cls = next(getattr(common, n) for n in dir(common)
               if isinstance(getattr(common, n), type) and hasattr(getattr(common, n), "_cut"))
    pf = cls(tok, 4242)

    def make(n):
        r = pf.make(n)
        return r if isinstance(r, str) else max((x for x in r if isinstance(x, str)), key=len)

    cases = [("isl300@1", 300, 1, nper), ("isl1k@1", 1000, 1, nper), ("isl2k@1", 2000, 1, nper),
             ("isl4k@1", 4000, 1, nper), ("isl8k@1", 8000, 1, nper), ("isl8k@4", 8000, 4, 16)]
    prompts = {name: [make(isl) for _ in range(nreq)] for name, isl, c, nreq in cases}
    resend = {n: [make(n) for _ in range(2)] for n in (300, 1000, 2000, 4000, 8000)}
    warm = [make(n) for n in (300, 1000, 2000, 4000, 8000) for _ in range(2)]
    log("prompts ready")
    site = str(HERE / "site")
    env = dict(SERVER_ENV)
    env["ASCENT_TOK_LOG"] = str(toklog)
    env["PYTHONPATH"] = site + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    srv = Server(ckpt, str(out), "tok", env)
    res, info, rs, rs_summary = {}, {}, {}, {}
    try:
        srv.wait_ready()
        for t in warm:
            hs.one_request(srv.port, t)
        hs.run_case(srv.port, [make(8000) for _ in range(4)], 4)
        log("warmup done")

        def read_log():
            recs = []
            for p in toklog.glob("tok_*.jsonl"):
                for line in p.read_text().splitlines():
                    try:
                        recs.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        pass
            return recs

        log(f"hook records after warmup: {len(read_log())}")
        for name, isl, conc, nreq in cases:
            time.sleep(1.0)
            t0 = time.time()
            rr = hs.run_case(srv.port, prompts[name], conc)
            t1 = time.time()
            recs = [r for r in read_log() if t0 <= r["t_end"] <= t1]
            ttft = [r["ttft_ms"] for r in rr if r and "ttft_ms" in r]
            bad = [r for r in rr if not r or r.get("error") or r.get("status") != 200]
            tk = [r["ms"] for r in recs]
            med = statistics.median(ttft)
            res[name] = {
                "step_ms": round(med, 3), "p10_ms": round(pct(ttft, 0.1), 3), "p90_ms": round(pct(ttft, 0.9), 3),
                "noise_floor": round((pct(ttft, 0.9) - pct(ttft, 0.1)) / med, 4), "n": len(ttft), "bad": len(bad),
                "prompt_tokens": sorted({r.get("prompt_tokens") for r in rr if r}, key=str),
                "cached_tokens": sorted({r.get("cached_tokens") for r in rr if r}, key=str),
                "tok_ms_median": round(statistics.median(tk), 3) if tk else None,
                "tok_ms_p10": round(pct(tk, 0.1), 3) if tk else None,
                "tok_ms_p90": round(pct(tk, 0.9), 3) if tk else None, "tok_n": len(tk),
                "tok_cpu_over_wall_median": round(statistics.median(r["process_cpu_ms"] / max(r["ms"], 1e-6)
                                                                    for r in recs), 3) if recs else None,
                "tok_pids": sorted({r["pid"] for r in recs}),
                "pool_threads_after_max": max((r["pool_threads_after"] for r in recs), default=None),
                "tokenizers_parallelism_env": sorted({str(r["TOKENIZERS_PARALLELISM"]) for r in recs}),
            }
            log(f"{name}: {res[name]}")
        # Re-send check (same server process): first send, then re-send alone; bitwise.
        n_eq, n_tot, cached_bad = 0, 0, 0
        for n, texts in resend.items():
            for t in texts:
                a = logprob_request(srv.port, t)
                b = logprob_request(srv.port, t)
                same = (a["tokens"] == b["tokens"] and a["token_logprobs"] == b["token_logprobs"]
                        and a["top_logprobs"] == b["top_logprobs"])
                n_eq += int(same)
                n_tot += 1
                cached_bad += int(a["cached_tokens"] not in (0, None)) + int(b["cached_tokens"] not in (0, None))
                rs.setdefault(str(n), []).append({"bitwise_equal": same, "token": a["tokens"],
                                                  "logprob": a["token_logprobs"], "prompt_tokens": a["prompt_tokens"],
                                                  "cached": [a["cached_tokens"], b["cached_tokens"]]})
        rs_summary = {"equal": n_eq, "total": n_tot, "cached_nonzero": cached_bad}
        log("re-send:", rs_summary)
        allr = read_log()
        info = {"server_pid": srv.proc.pid, "hook_pids": sorted({r["pid"] for r in allr}),
                "hook_ppids": sorted({r["ppid"] for r in allr}),
                "pool_threads_max": max((r["pool_threads_after"] for r in allr), default=None),
                "TOKENIZERS_PARALLELISM": sorted({str(r["TOKENIZERS_PARALLELISM"]) for r in allr})}
        log("process info:", info)
    finally:
        srv.stop()
        hs._cleanup()
    print(json.dumps({"batch": res, "process": info, "resend": {"summary": rs_summary, "detail": rs},
                      "output": {"class": "exact"}}))


if __name__ == "__main__":
    main()
