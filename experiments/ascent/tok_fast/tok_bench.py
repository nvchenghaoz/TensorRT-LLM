#!/usr/bin/env python3
"""Gemma4 text-only tokenization: correctness + timing (branch 32, idea 62).

Usage: tok_bench.py <checkpoint> <mode>
  mode = check : ids from the input processor (and the split encoder with forced splits)
                 vs the stock call on workload prompts (5 ISLs x N) + a fuzz set, 3 repeats,
                 then under 4 concurrent threads. Prints {"passed": ...}.
  mode = time  : per-ISL input-processor ms (median/p10/p90) and 4/16-thread 8k makespan.
Runs identically on base and candidate trees (only calls the input processor).
"""
import importlib.util
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T0 = time.time()


def log(*a):
    print(f"[+{time.time() - T0:6.1f}s]", *a, flush=True)


def _hb():
    while True:
        time.sleep(30)
        log("heartbeat")


ISLS = [300, 1000, 2000, 4000, 8000]
N_PER_ISL = int(os.environ.get("N_PER_ISL", "40"))


SHARED = Path("/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/shared/"
              "host_runtime_benchfiles/a929b157")
_BENCH_FILES = ("common.py", "corpus.txt")


def _publish_shared(src_dir):
    """Copy the frozen common.py + corpus.txt to SHARED (tmp dir, manifest last, rename)."""
    import hashlib
    import shutil
    if (SHARED / "manifest.json").exists():
        return
    tmp = SHARED.parent / f".tmp-{os.environ.get('AUTO_DEV_JOB_ID', os.getpid())}"
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        man = {"source": str(src_dir), "job": os.environ.get("AUTO_DEV_JOB_ID"), "files": {}}
        for f in _BENCH_FILES:
            shutil.copyfile(src_dir / f, tmp / f)
            man["files"][f] = hashlib.sha256((tmp / f).read_bytes()).hexdigest()
        (tmp / "manifest.json").write_text(json.dumps(man))
        os.rename(tmp, SHARED)
        log("published frozen benchmark files to", SHARED, man["files"])
    except OSError as e:
        log("publish to shared dir skipped:", repr(e))
        shutil.rmtree(tmp, ignore_errors=True)


def _shared_ok():
    import hashlib
    try:
        man = json.loads((SHARED / "manifest.json").read_text())
        return all(hashlib.sha256((SHARED / f).read_bytes()).hexdigest() == man["files"][f] for f in _BENCH_FILES)
    except (OSError, ValueError, KeyError):
        return False


def load_common():
    import glob
    import hashlib
    local = Path(__file__).resolve().parents[2] / "ascent-benchmark"
    if (local / "common.py").exists() and (local / "corpus.txt").exists():
        _publish_shared(local)
    cands = [Path("experiments/ascent-benchmark/common.py"), local / "common.py"]
    # The base tree of a pair does not carry the frozen benchmark: read the verified shared copy.
    if _shared_ok():
        cands.append(SHARED / "common.py")
    # The base tree of a pair does not carry the frozen benchmark; any copy on the node is the
    # same frozen file (sha256 logged so both arms can be compared).
    root = "/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/runner/slots"
    cands += [Path(p) for p in sorted(glob.glob(f"{root}/*/workspace/trtllm/experiments/ascent-benchmark/common.py"))]
    cands += [Path(p) for p in sorted(glob.glob(f"{root}/*/jobs/*/trtllm/experiments/ascent-benchmark/common.py"))]
    for cand in cands:
        if cand.exists() and (cand.parent / "corpus.txt").exists():
            log("benchmark common sha256", hashlib.sha256(cand.read_bytes()).hexdigest()[:16],
                "corpus sha256", hashlib.sha256((cand.parent / "corpus.txt").read_bytes()).hexdigest()[:16])
            spec = importlib.util.spec_from_file_location("bench_common", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["bench_common"] = mod
            spec.loader.exec_module(mod)
            log("loaded benchmark common from", cand)
            return mod
    log("benchmark common.py not found")
    return None


def workload_prompts(common, n_per):
    tok = common.load_tokenizer(ckpt)
    cls = None
    for name in dir(common):
        obj = getattr(common, name)
        if isinstance(obj, type) and hasattr(obj, "make") and hasattr(obj, "_cut"):
            cls = obj
            break
    log("prompt factory:", cls.__name__ if cls else None)
    pf = cls(tok, 12345)
    out = {}

    def text_of(r):
        if isinstance(r, str):
            return r
        if isinstance(r, dict):
            return next(v for v in r.values() if isinstance(v, str) and len(v) > 50)
        return max((x for x in r if isinstance(x, str)), key=len)

    r0 = pf.make(300)
    log("make() returns", type(r0).__name__, [type(x).__name__ for x in r0] if isinstance(r0, tuple) else "")
    for n in ISLS:
        out[n] = [text_of(pf.make(n)) for _ in range(n_per)]
        log(f"isl {n}: {len(out[n])} prompts, chars median {statistics.median(len(t) for t in out[n])}")
    return out


def fuzz_texts(base_texts, n, seed=7):
    rng = random.Random(seed)
    specials = ["<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>", "<pad>", "<unk>", "<mask>",
                "<start_of_image>", "<|turn>", "<turn|>", "<|channel>", "<channel|>"]
    odd = [" ", "  ", "   ", "\n", "\n\n", "\t", " \n", "\n ", ">", "> ", " >", "▁", "▁ ",
           " ▁", "<", "</s>", "<br> ", "->", "=> ", ">>> ", " ", "　", "\r\n",
           "\U0001F600", "中文 ", "مرحبا ", "é", "﻿",
           "\x00", "\x07", "\U000E0041", "##", "​", "  \t ", " <bos> ", "<start_of_turn>user\n"]
    corpus = "\n".join(base_texts)
    out = []
    for k in range(n):
        L = rng.choice([200, 800, 2000, 3500, 5000, 8000, 12000, 20000, 34000])
        parts = []
        cur = 0
        while cur < L:
            r = rng.random()
            if r < 0.6:
                s = rng.randrange(0, len(corpus) - 400)
                seg = corpus[s:s + rng.randrange(5, 400)]
            elif r < 0.85:
                seg = rng.choice(odd)
            elif r < 0.92:
                seg = rng.choice(specials)
            else:
                seg = "".join(chr(rng.choice([rng.randrange(32, 127), rng.randrange(0x80, 0x3000),
                                              rng.randrange(0x1F300, 0x1FAFF)])) for _ in range(rng.randrange(1, 20)))
            parts.append(seg)
            cur += len(seg)
        out.append("".join(parts))
    return out


def build_ip():
    from tensorrt_llm.inputs.registry import create_input_processor
    from tensorrt_llm.sampling_params import SamplingParams
    from tensorrt_llm.tokenizer.tokenizer import tokenizer_factory
    tok = tokenizer_factory(ckpt)
    ip = create_input_processor(ckpt, tok)
    log(f"input processor {type(ip).__name__}, tokenizer {type(tok).__name__}/{type(tok.tokenizer).__name__}")
    return ip, tok, SamplingParams(max_tokens=1)


def stock_ids(ip, text):
    # Exactly the base tree's text-only tokenizer call + call_with_text_prompt conversion.
    import torch
    return ip._tokenizer(text, return_tensors="pt")["input_ids"][0].to(torch.int32).tolist()


def check():
    common = load_common()
    prompts = workload_prompts(common, N_PER_ISL)
    ip, tok, sp = build_ip()
    import torch  # noqa: F401
    allw = [t for n in ISLS for t in prompts[n]]
    fuzz = fuzz_texts(allw, int(os.environ.get("N_FUZZ", "2000")))
    log(f"fuzz texts: {len(fuzz)}, chars total {sum(map(len, fuzz))}")
    texts = allw + fuzz
    labels = [f"isl{n}" for n in ISLS for _ in prompts[n]] + ["fuzz"] * len(fuzz)
    refs = [stock_ids(ip, t) for t in texts]
    log("references computed")
    res = {"n_workload": len(allw), "n_fuzz": len(fuzz)}
    enc = getattr(ip, "_text_encoder", None)
    if enc is not None:
        import tensorrt_llm._torch.models.gemma4_text_tokenize as gt
        res["can_split"] = enc.can_split
        res["bad_prev"] = sorted(enc._bad_prev) if enc._bad_prev else None
        res["prefix"], res["suffix"] = enc._prefix, enc._suffix
        log("encoder can_split", enc.can_split, "bad_prev", res["bad_prev"], "prefix", enc._prefix,
            "suffix", enc._suffix)
        # Pre-tokenizer does not split the normalized text (one BPE word).
        b = enc._backend
        multi = 0
        for t in texts[:500]:
            if len(b.pre_tokenizer.pre_tokenize_str(b.normalizer.normalize_str(t))) != 1:
                multi += 1
        res["pretok_multi_piece_texts_of_500"] = multi
        log("pre-tokenizer multi-piece texts (of 500):", multi)
    fails = {}

    def cmp(name, fn, idxs):
        bad = 0
        for i in idxs:
            got = fn(texts[i])
            if list(got) != refs[i]:
                bad += 1
                if bad <= 3:
                    a, r = list(got), refs[i]
                    j = next((k for k in range(min(len(a), len(r))) if a[k] != r[k]), min(len(a), len(r)))
                    log(f"MISMATCH {name} {labels[i]} len {len(a)} vs {len(r)} first diff at {j}: "
                        f"{a[max(0, j - 3):j + 3]} vs {r[max(0, j - 3):j + 3]}")
        fails[name] = fails.get(name, 0) + bad
        return bad

    idx = list(range(len(texts)))
    for rep in range(3):
        b1 = cmp("input_processor", lambda t: ip({"prompt": t}, sp)[0], idx)
        log(f"rep {rep}: input_processor mismatches {b1}")
        if enc is not None and enc.can_split:
            # Forced split at every length: small pieces, many cuts.
            saved = (gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS)
            gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS = 64, 64
            try:
                b2 = cmp("forced_split_64", enc.encode, idx)
            finally:
                gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS = saved
            log(f"rep {rep}: forced split (64-char pieces) mismatches {b2}")
            saved = (gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS, gt.MAX_CONCURRENT_SPLITS)
            gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS, gt.MAX_CONCURRENT_SPLITS = 64, 64, 0
            try:
                b3 = cmp("forced_split_64_serial", enc.encode, idx)
            finally:
                gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS, gt.MAX_CONCURRENT_SPLITS = saved
            log(f"rep {rep}: forced serial split (64-char pieces) mismatches {b3}")
    # Concurrency: 4 threads, both paths.
    for name, fn in [("input_processor_4thr", lambda t: ip({"prompt": t}, sp)[0])]:
        out = [None] * len(texts)

        def work(i):
            out[i] = fn(texts[i])

        with ThreadPoolExecutor(4) as ex:
            list(ex.map(work, idx))
        bad = sum(1 for i in idx if list(out[i]) != refs[i])
        fails[name] = bad
        log(f"{name} mismatches {bad}")
    if enc is not None and enc.can_split:
        n_cut_texts = sum(1 for t in texts if enc._cuts(t))
        res["texts_with_cuts_default"] = n_cut_texts
        log("texts with cuts at default settings:", n_cut_texts)
    res["mismatches"] = fails
    res["passed"] = all(v == 0 for v in fails.values())
    res["compared_against"] = "stock tokenizer(text, return_tensors='pt')['input_ids'][0].to(int32).tolist()"
    return res


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def timing():
    common = load_common()
    prompts = workload_prompts(common, N_PER_ISL)
    ip, tok, sp = build_ip()
    call = lambda t: ip({"prompt": t}, sp)[0]  # noqa: E731
    # warmup
    for n in ISLS:
        for t in prompts[n][:5]:
            call(t)
    batch = {}
    for n in ISLS:
        ts = []
        for rep in range(3):
            for t in prompts[n]:
                a = time.perf_counter()
                call(t)
                ts.append(1e3 * (time.perf_counter() - a))
        med = statistics.median(ts)
        p10, p90 = pct(ts, 0.1), pct(ts, 0.9)
        batch[f"isl{n}"] = {"step_ms": round(med, 4), "p10_ms": round(p10, 4), "p90_ms": round(p90, 4),
                            "noise_floor": round((p90 - p10) / med, 4), "n": len(ts)}
        log(f"isl{n}: median {med:.3f} ms p10 {p10:.3f} p90 {p90:.3f}")
        enc = getattr(ip, "_text_encoder", None)
        if enc is not None:
            # Same process, same prompts: main's stock call and stage 1 (unsplit list path).
            for name, fn in (("stock_inproc", lambda t: stock_ids(ip, t)),
                             ("stage1_inproc", enc._encode_unsplit)):
                ts2 = []
                for rep in range(3):
                    for t in prompts[n]:
                        a = time.perf_counter()
                        fn(t)
                        ts2.append(1e3 * (time.perf_counter() - a))
                m2 = statistics.median(ts2)
                batch[f"isl{n}"][f"{name}_ms"] = [round(m2, 4), round(pct(ts2, 0.1), 4), round(pct(ts2, 0.9), 4)]
                log(f"isl{n} {name}: median {m2:.3f}")
    for thr in (4, 16):
        mks = []
        texts = prompts[8000]
        for rep in range(7):
            sel = [texts[(rep * thr + k) % len(texts)] for k in range(thr)]
            bar = threading.Barrier(thr + 1)
            done = [0.0] * thr

            def w(k):
                bar.wait()
                call(sel[k])
                done[k] = time.perf_counter()

            ths = [threading.Thread(target=w, args=(k,)) for k in range(thr)]
            for x in ths:
                x.start()
            bar.wait()
            a = time.perf_counter()
            for x in ths:
                x.join()
            mks.append(1e3 * (max(done) - a))
        med = statistics.median(mks)
        p10, p90 = pct(mks, 0.1), pct(mks, 0.9)
        batch[f"isl8000_x{thr}thr_makespan"] = {"step_ms": round(med, 4), "p10_ms": round(p10, 4),
                                                 "p90_ms": round(p90, 4), "noise_floor": round((p90 - p10) / med, 4),
                                                 "n": len(mks)}
        log(f"8k x{thr} threads makespan: median {med:.3f} p10 {p10:.3f} p90 {p90:.3f}")
    return {"batch": batch, "output": {"class": "exact"}}


def sweep():
    common = load_common()
    prompts = workload_prompts(common, 20)
    ip, tok, sp = build_ip()
    import tensorrt_llm._torch.models.gemma4_text_tokenize as gt
    enc = ip._text_encoder
    hf = tok.tokenizer
    import torch
    out = {}
    combos = [("stock", None), ("unsplit", (10**9, 1536, 8)), ("p1024x8", (2048, 1024, 8)),
              ("p1536x8", (3072, 1536, 8)), ("p2048x8", (4096, 2048, 8)), ("p1536x4", (3072, 1536, 4)),
              ("p1024x16", (2048, 1024, 16)), ("p3072x8", (6144, 3072, 8)), ("p768x8", (1536, 768, 8))]
    for n in ISLS:
        row = {}
        for name, cfg in combos:
            if cfg is None:
                fn = lambda t: hf(t, return_tensors="pt")["input_ids"][0].to(torch.int32).tolist()  # noqa: E731
            else:
                gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS, gt.MAX_PIECES = cfg
                gt.POOL_WORKERS = 32
                gt._PIECE_POOL = gt._PiecePool()
                fn = enc.encode
            for t in prompts[n][:3]:
                fn(t)
            ts = []
            for rep in range(3):
                for t in prompts[n]:
                    a = time.perf_counter()
                    fn(t)
                    ts.append(1e3 * (time.perf_counter() - a))
            row[name] = round(statistics.median(ts), 3)
        out[n] = row
        log(f"isl{n}: {row}")
    print(json.dumps({"passed": True, "sweep_median_ms": out}))


if __name__ == "__main__":
    threading.Thread(target=_hb, daemon=True).start()
    ckpt, mode = sys.argv[1], sys.argv[2]
    if mode == "publish":
        load_common()
        print(json.dumps({"passed": _shared_ok(), "shared": str(SHARED)}))
    elif mode == "check":
        print(json.dumps(check()))
    elif mode == "sweep":
        sweep()
    elif mode == "all":
        # Timing first (clean process state), then the untimed id-equality check.
        t = timing()
        c = check()
        t["check"] = c
        t["passed"] = c["passed"]
        print(json.dumps(t))
    else:
        print(json.dumps(timing()))
