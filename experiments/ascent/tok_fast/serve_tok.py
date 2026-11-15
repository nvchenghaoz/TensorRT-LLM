#!/usr/bin/env python3
"""In-server tokenization stage + TTFT at c=1 (branch 32, idea 62). Same script on base and candidate.

Usage: serve_tok.py <checkpoint> [n_per_case]
Starts trtllm-serve (main defaults + partial reuse off; host_stages/run.py Server) with the
experiment-only sitecustomize hook (site/) that times Gemma4InputProcessor.call_with_text_prompt
inside the serving frontend. Unique workload prompts (benchmark PromptFactory), untimed warmup,
then cases isl{300,1k,2k,4k,8k}@1 and isl300@4. Prints per case TTFT and tok-stage ms.
"""
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "host_stages"))
import run as hs  # noqa: E402

sys.path.insert(0, str(HERE))
from tok_bench import load_common  # noqa: E402  (module-level code only parses argv)


def log(*a):
    hs.log(*a)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


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
             ("isl4k@1", 4000, 1, nper), ("isl8k@1", 8000, 1, nper), ("isl300@4", 300, 4, 4 * nper)]
    prompts = {name: [make(isl) for _ in range(nreq)] for name, isl, c, nreq in cases}
    warm = [make(n) for n in (300, 1000, 2000, 4000, 8000) for _ in range(2)]
    log("prompts ready")
    site = str(HERE / "site")
    env = {"ASCENT_TOK_LOG": str(toklog),
           "PYTHONPATH": site + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    srv = hs.Server(ckpt, str(out), "tok", env)
    res = {}
    try:
        srv.wait_ready()
        for t in warm:
            hs.one_request(srv.port, t)
        hs.run_case(srv.port, [make(300) for _ in range(8)], 4)
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

        n_warm = len(read_log())
        log(f"hook records after warmup: {n_warm}")
        for name, isl, conc, nreq in cases:
            time.sleep(1.0)
            t0 = time.time()
            rr = hs.run_case(srv.port, prompts[name], conc)
            t1 = time.time()
            recs = [r for r in read_log() if t0 <= r["t_end"] <= t1]
            ttft = [r["ttft_ms"] for r in rr if r and "ttft_ms" in r]
            bad = [r for r in rr if not r or r.get("error") or r.get("status") != 200]
            pts = sorted({r.get("prompt_tokens") for r in rr if r}, key=str)
            cached = sorted({r.get("cached_tokens") for r in rr if r}, key=str)
            tk = [r["ms"] for r in recs]
            med = statistics.median(ttft)
            res[name] = {
                "step_ms": round(med, 3), "p10_ms": round(pct(ttft, 0.1), 3), "p90_ms": round(pct(ttft, 0.9), 3),
                "noise_floor": round((pct(ttft, 0.9) - pct(ttft, 0.1)) / med, 4), "n": len(ttft), "bad": len(bad),
                "prompt_tokens": pts, "cached_tokens": cached,
                "tok_ms_median": round(statistics.median(tk), 3) if tk else None,
                "tok_ms_p10": round(pct(tk, 0.1), 3) if tk else None,
                "tok_ms_p90": round(pct(tk, 0.9), 3) if tk else None, "tok_n": len(tk),
                "tok_pids": sorted({r["pid"] for r in recs}),
                "pool_threads_after_max": max((r["pool_threads_after"] for r in recs), default=None),
                "tokenizers_parallelism_env": sorted({str(r["TOKENIZERS_PARALLELISM"]) for r in recs}),
            }
            log(f"{name}: {res[name]}")
        allr = read_log()
        info = {"server_pid": srv.proc.pid, "hook_pids": sorted({r["pid"] for r in allr}),
                "hook_ppids": sorted({r["ppid"] for r in allr})}
        log("process info:", info)
    finally:
        srv.stop()
        hs._cleanup()
    print(json.dumps({"batch": res, "process": info, "output": {"class": "exact"}}))


if __name__ == "__main__":
    main()
