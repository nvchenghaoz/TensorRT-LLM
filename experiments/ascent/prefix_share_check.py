"""Cross-process exactness and cold-miss timing for the shared prefix token cache.

usage: prefix_share_check.py <checkpoint> [check|bench]

Roles (each a fresh python process, so caches never share memory):
  publisher: tokenizes the prime prompt (full) with the shared dir set -> publishes.
  consumer : fresh cache, encodes N benchmark-shaped prompts; each must equal
             a full tokenization of the same text; reports loads/splices/full counts.
Modes:
  check (default): exactness + negative files, prints {"passed": ...}.
  bench          : times the first request of a cold process after a prime
                   published by another process; prints step_ms JSON.
                   On a base tree (no sharing) the first request is a full
                   tokenization.

Prompts mimic the benchmark: one shared ~32K-token preamble, then a unique
~2K-token suffix whose first word differs per prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ENV = "TLLM_PREFIX_TOKEN_CACHE_SHARED_DIR"
N_PROMPTS = int(os.environ.get("PSC_N", "64"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


_BASE = [
    "the", "model", "serves", "requests", "while", "cache", "holds", "tokens", "and",
    "document", "retrieval", "question", "answer", "context", "window", "attention",
    "layer", "expert", "router", "kernel", "memory", "bandwidth", "latency", "throughput",
    "Alpha", "Beta", "Gamma", "delta", "epsilon", "über", "café", "naïve", "résumé",
    "data-driven", "state-of-the-art", "e.g.", "i.e.", "(see", "below)", "[note]",
    "x=1;", "value:", "\"quoted\"", "'single'", "path/to/file.txt", "C++", "#tag", "@user",
    "中文", "テスト", "ok!", "why?", "end.", "50%", "$12", "1999", "3.14159", "a,b,c",
]


def _words(rng, n):
    parts = []
    for _ in range(n):
        w = rng.choice(_BASE)
        if rng.random() < 0.05:
            w = str(rng.randint(0, 99999))
        parts.append(" " + w)
        r = rng.random()
        if r < 0.03:
            parts.append(".\n\n")
        elif r < 0.07:
            parts.append(",")
    return "".join(parts)


def prompts(n, seed=0):
    prefix = "Document 0." + _words(random.Random(1234), 22000)
    leads = ["Apple", "Bridge", "Castle", "Dragon", "Engine", "Forest", "Garden", "Harbor",
             "Island", "Jungle", "Kettle", "Ladder", "Meadow", "Needle", "Orange", "Pencil"]
    out = []
    for i in range(n + 1):
        r = random.Random(seed * 100003 + i)
        out.append(prefix + "\n\n" + leads[i % len(leads)] + str(seed * 1000 + i) + _words(r, 1400))
    return out  # out[0] is the prime


def load_tok(ckpt):
    from tensorrt_llm.tokenizer.tokenizer import load_hf_tokenizer
    tok = load_hf_tokenizer(ckpt)
    return getattr(tok, "tokenizer", tok)


def role_main(role, args):
    tok = load_tok(args["ckpt"])
    from tensorrt_llm.inputs import prefix_token_cache as ptc
    cache = ptc.create_prefix_token_cache(tok)
    assert cache is not None and cache._lcp_splice, "lcp splice gate is off"
    ps = prompts(int(args.get("n", N_PROMPTS)), seed=int(args.get("seed", 0)))
    for attr in ("shared_loads", "shared_skipped", "shared_publishes"):
        if not hasattr(cache, attr):  # base tree: no sharing
            setattr(cache, attr, 0)
    res = {"pid": os.getpid(), "fp": getattr(cache, "_fingerprint", None)}
    if role == "publisher":
        t = time.perf_counter()
        ids = cache.encode(tok, ps[0])
        res["prime_ms"] = (time.perf_counter() - t) * 1e3
        full = tok(ps[0], add_special_tokens=False)["input_ids"]
        res["prime_exact"] = list(ids) == list(full)
        res["n_prime_ids"] = len(ids)
        res["publishes"] = cache.shared_publishes
    else:
        bad, digest, times = 0, hashlib.sha1(), []
        first_loads = first_full = None
        for i, p in enumerate(ps[1:]):
            t = time.perf_counter()
            ids = cache.encode(tok, p)
            times.append((time.perf_counter() - t) * 1e3)
            if i == 0:
                first_loads, first_full = cache.shared_loads, cache._full_count
            full = tok(p, add_special_tokens=False)["input_ids"]
            bad += list(ids) != list(full)
            digest.update(json.dumps(list(ids)).encode())
        res.update(
            n=len(ps) - 1, mismatches=bad, digest=digest.hexdigest(), first_ms=times[0],
            rest_ms=statistics.median(times[1:]) if len(times) > 1 else None,
            shared_loads=cache.shared_loads, shared_skipped=cache.shared_skipped,
            shared_publishes=cache.shared_publishes, first_loads=first_loads,
            first_full=first_full, full=cache._full_count, splices=cache.splices,
            splice_fallbacks=cache.splice_fallbacks, splice_no_cut=cache.splice_no_cut,
            n_ids_first=None, summary=cache.summary(),
        )
    print("ROLE_RESULT " + json.dumps(res), flush=True)


def run_role(ckpt, role, shared_dir, **args):
    env = dict(os.environ)
    env.pop(ENV, None)
    if shared_dir is not None:
        env[ENV] = shared_dir
    args["ckpt"] = ckpt
    cmd = [sys.executable, os.path.abspath(__file__), "--role", role, json.dumps(args)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    for line in p.stdout.splitlines():
        if line.startswith("ROLE_RESULT "):
            return json.loads(line[len("ROLE_RESULT "):])
    sys.stderr.write(p.stdout[-3000:] + p.stderr[-6000:])
    raise RuntimeError(f"{role} failed rc={p.returncode}")


def all_files(root):
    out = []
    for dp, _, fns in os.walk(root):
        out += [os.path.join(dp, f) for f in fns]
    return sorted(out)


def brief(r):
    return json.dumps({k: v for k, v in r.items() if k not in ("summary", "digest")})


def check(ckpt):
    work = tempfile.mkdtemp(prefix="psc_")
    results, ok = {}, True
    try:
        # 1. publisher A, then consumer B twice (identical), on one shared dir.
        d1 = os.path.join(work, "d1")
        os.makedirs(d1)
        a = run_role(ckpt, "publisher", d1)
        log(f"publisher: {brief(a)}")
        files = all_files(d1)
        log(f"published files: {files} sizes {[os.path.getsize(f) for f in files]}")
        b1 = run_role(ckpt, "consumer", d1)
        log(f"consumer run 1: {brief(b1)}")
        log(f"consumer run 1 summary: {b1['summary']}")
        b2 = run_role(ckpt, "consumer", d1)
        log(f"consumer run 2: {brief(b2)}")
        a2 = run_role(ckpt, "publisher", d1)  # republish: must skip (file exists)
        log(f"republisher: {brief(a2)}")
        files2 = all_files(d1)
        c1 = bool(
            a["prime_exact"] and a["publishes"] == 1 and len(files) == 1 and files == files2
            and a2["publishes"] == 0 and a2["prime_exact"]
            and b1["n"] >= 64 and b1["mismatches"] == 0 and b1["first_loads"] == 1
            and b1["first_full"] == 0 and b1["full"] == 0 and b1["splices"] == b1["n"]
            and b1["splice_fallbacks"] == 0 and b1["shared_publishes"] == 0
            and b1["digest"] == b2["digest"] and b2["mismatches"] == 0
            and b2["splices"] == b2["n"]
        )
        results["cross_process"] = dict(
            passed=c1, n_prompts=b1["n"], prime_full_ms=a["prime_ms"],
            cold_first_ms=b1["first_ms"], warm_ms=b1["rest_ms"], splices=b1["splices"],
            fallbacks=b1["splice_fallbacks"], full=b1["full"], loads=b1["shared_loads"],
            repeat_digest_equal=b1["digest"] == b2["digest"], n_files=len(files), fp=a["fp"],
        )
        ok &= c1
        good = files[0]
        with open(good, "rb") as f:
            data = f.read()
        nl = data.find(b"\n")
        header = json.loads(data[:nl])
        payload = data[nl + 1:]
        fp = header["fp"]
        log(f"header: {header}")

        def neg(name, files_to_write):
            d = os.path.join(work, name)
            for rel, blob in files_to_write.items():
                path = os.path.join(d, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(blob)
            r = run_role(ckpt, "consumer", d, n=16)
            expect_skip = sum(1 for k in files_to_write
                              if k.startswith(fp + "/") and k.endswith(".ptc"))
            passed = bool(
                r["mismatches"] == 0 and r["shared_loads"] == 0
                and r["shared_skipped"] == expect_skip and r["first_full"] == 1
                and r["splice_fallbacks"] == 0 and r["splices"] == r["n"] - 1
            )
            log(f"negative {name}: passed={passed} {brief(r)}")
            return passed, r

        rel = os.path.relpath(good, d1)
        bname = os.path.basename(good)
        flipped = bytearray(payload)
        flipped[header["n_text_bytes"] + 40] ^= 0x01  # one id byte
        h_foreign = dict(header, fp="0123456789ab")
        cases = {
            "truncated": {rel: data[: len(data) - 1000]},
            "header_only": {rel: data[:nl]},
            "mismatched_n_ids": {rel: json.dumps(dict(header, n_ids=header["n_ids"] - 1)).encode()
                                 + b"\n" + payload},
            "mismatched_split": {rel: json.dumps(dict(header, split_token=header["n_ids"] + 5)).encode()
                                 + b"\n" + payload},
            "checksum_flip": {rel: data[: nl + 1] + bytes(flipped)},
            "foreign_fp_in_dir": {rel: json.dumps(h_foreign).encode() + b"\n" + payload},
            "foreign_fp_subdir": {f"0123456789ab/{bname}": json.dumps(h_foreign).encode()
                                  + b"\n" + payload},
            "tmp_only": {rel + ".123.4.tmp": data},
        }
        neg_res = {}
        for name, fw in cases.items():
            p, r = neg(name, fw)
            neg_res[name] = dict(passed=p, skipped=r["shared_skipped"], loads=r["shared_loads"],
                                 mismatches=r["mismatches"], full=r["full"])
            ok &= p
        results["negative"] = neg_res

        # 3. env unset: no file I/O, behaviour as today (one full, then splices).
        r = run_role(ckpt, "consumer", None, n=16)
        log(f"env unset: {brief(r)}")
        c3 = bool(r["mismatches"] == 0 and r["full"] == 1 and r["shared_loads"] == 0
                  and r["shared_publishes"] == 0 and "shared:" not in r["summary"])
        results["env_unset"] = dict(passed=c3, full=r["full"], splices=r["splices"])
        ok &= c3
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"passed": bool(ok),
                      "compared": "cache ids vs full tokenizer(text) (the protected definition of "
                                  "prompt ids), across processes",
                      **results}))


def bench(ckpt):
    """First-request time on a cold process, prime published by another process."""
    trials = int(os.environ.get("PSC_TRIALS", "9"))
    work = tempfile.mkdtemp(prefix="psb_")
    firsts = []
    try:
        d = os.path.join(work, "d")
        os.makedirs(d)
        a = run_role(ckpt, "publisher", d)
        log(f"publisher prime_ms={a['prime_ms']:.1f} files={all_files(d)}")
        for t in range(trials):
            r = run_role(ckpt, "consumer", d, n=4, seed=t + 1)
            log(f"trial {t}: first_ms={r['first_ms']:.2f} rest_ms={r['rest_ms']:.2f} "
                f"loads={r['shared_loads']} full={r['full']} mismatches={r['mismatches']}")
            assert r["mismatches"] == 0
            firsts.append(r["first_ms"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    s = sorted(firsts)
    med = statistics.median(s)
    q = lambda f: s[min(len(s) - 1, max(0, round(f * (len(s) - 1))))]
    print(json.dumps({"batch": {"1": {"step_ms": med, "noise_floor": (q(0.9) - q(0.1)) / med}},
                      "first_request_ms": firsts, "output": {"class": "exact"}}))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--role":
        role_main(sys.argv[2], json.loads(sys.argv[3]))
    else:
        ckpt = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else "check"
        bench(ckpt) if mode == "bench" else check(ckpt)
