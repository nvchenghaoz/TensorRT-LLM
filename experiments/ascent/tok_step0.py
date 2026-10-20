# SPDX-License-Identifier: Apache-2.0
"""STEP 0 of host-runtime idea 109: per-segment timing of the c1 prefix-token-cache
hit intake on main, plus whether a batched / threaded tail tokenize can run in
parallel in this process. CPU only: tokenizer + input processor, no model weights.

Segments per c1 real-prompt cache hit (instance-level wrappers, main's code runs):
  (a) PrefixTokenCache._lookup
  (b) the tail tokenize (PrefixTokenCache._tokenize on text[split_char:])
  (c) the rest of encode: resync, splice, grid, insert, ids.tolist()
  (d) proxy submit: GenerationRequest(ids) + pickle.dumps (the IPC put) + loads
  (e) executor token round trip: tllm.Request(int32 buffer) ->
      .input_token_ids -> list -> LlmRequest (executor_request_to_llm_request)
Then, on the observed hit tails: single HF call (main) vs k pieces split at
pre-token boundaries through one HF batch call, backend encode_batch, and a
thread pool; exactness of ids AND offsets for each; GIL release test.

Last stdout line: JSON.
"""

import argparse
import json
import os
import pickle
import subprocess
import threading
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

from tok_common import (PromptMaker, common_prefix, env_info, hf_tokenizer, load_real_prompts,
                        load_tokenizer, log, make_processor, pct, processor_cache)

BASE_COMMIT = "07a287408d9a3a2313ae5f3ec22730f94f932b51"


def tree_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=20).stdout.strip()
        d = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return {"TREE_COMMIT": h or "?", "DIRTY": len(d)}
    except Exception as e:  # noqa: BLE001
        return {"TREE_COMMIT": "?", "err": repr(e)}


def ms(t0):
    return (time.perf_counter() - t0) * 1e3


def med(xs):
    return round(pct(xs, 0.5), 4)


def is_space_split(text, c):
    return 0 < c < len(text) and text[c] == " " and not text[c - 1].isspace()


def cut_points(tail, k, search=512):
    n = len(tail)
    cuts = [0]
    for j in range(1, k):
        target = max(j * n // k, cuts[-1] + 1)
        p = tail.find(" ", target)
        while p != -1 and p - target <= search and not is_space_split(tail, p):
            p = tail.find(" ", p + 1)
        if p == -1 or p - target > search:
            continue
        cuts.append(p)
    return cuts


def pieces_of(tail, cuts):
    return [tail[a:b] for a, b in zip(cuts, cuts[1:] + [len(tail)])]


def concat(cuts, ids_list, offs_list):
    ids, offs = [], []
    for base, i, o in zip(cuts, ids_list, offs_list):
        ids.extend(i)
        if base:
            offs.extend([(s + base, e + base) for s, e in o])
        else:
            offs.extend(o)
    return ids, offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()
    ti = tree_info()
    log(f"TREE_COMMIT={ti.get('TREE_COMMIT')} DIRTY={ti.get('DIRTY')} BASE_COMMIT={BASE_COMMIT}")
    env = env_info()
    try:
        aff = len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        aff = None
    cpu = {"cpu_count": os.cpu_count(), "affinity": aff, "loadavg": os.getloadavg(),
           "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
           "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    log(f"env {env} cpu {cpu}")

    from tensorrt_llm.sampling_params import SamplingParams

    tok = load_tokenizer(args.ckpt)
    hf = hf_tokenizer(tok)
    bt = hf.backend_tokenizer
    tk_info = {"tok_type": type(tok).__name__, "hf_type": type(hf).__name__,
               "normalizer": repr(bt.normalizer), "pre_tokenizer": repr(bt.pre_tokenizer),
               "post_processor": repr(bt.post_processor), "model": type(bt.model).__name__,
               "dropout": getattr(bt.model, "dropout", "?")}
    try:
        added = bt.get_added_tokens_decoder()
        flags = {"n": len(added),
                 "with_space": sum(1 for t in added.values() if any(ch.isspace() for ch in t.content)),
                 "lstrip": sum(1 for t in added.values() if t.lstrip),
                 "rstrip": sum(1 for t in added.values() if t.rstrip),
                 "single_word": sum(1 for t in added.values() if t.single_word),
                 "normalized": sum(1 for t in added.values() if t.normalized)}
    except Exception as e:  # noqa: BLE001
        flags = {"err": repr(e)}
    tk_info["added_tokens"] = flags
    for k, v in tk_info.items():
        log(f"tokenizer {k}: {v}")

    proc = make_processor(args.ckpt, tok, enable_cache=True)
    cache = processor_cache(proc)
    log(f"processor {type(proc).__name__} cache={cache is not None} "
        f"cache_tokenizer={type(getattr(proc, '_prefix_token_cache_tokenizer', None)).__name__}")
    sp = SamplingParams(max_tokens=256, add_special_tokens=True)

    rec = {}
    orig_lookup = cache._lookup
    orig_tok = type(cache)._tokenize
    orig_encode = cache.encode

    def lookup(text):
        t0 = time.perf_counter()
        r = orig_lookup(text)
        rec["lookup"] = ms(t0)
        return r

    def tokenize(tokenizer, text):
        t0 = time.perf_counter()
        r = orig_tok(tokenizer, text)
        rec.setdefault("tok", []).append((ms(t0), len(text), len(r[0])))
        rec["tail"] = text
        return r

    def encode(tokenizer, text):
        t0 = time.perf_counter()
        r = orig_encode(tokenizer, text)
        rec["encode"] = ms(t0)
        return r

    cache._lookup = lookup
    cache._tokenize = tokenize
    cache.encode = encode

    real, info = load_real_prompts(args.ckpt)
    log(f"real prompts: {json.dumps(info)}")
    assert real is not None, "no real prompts found"
    allr = real[1] + real[2] + real[4]
    shared = common_prefix(allr)
    maker = PromptMaker(hf, seed=4321)
    prime = f"{shared.rstrip()} {maker.sample(2048)}"
    t0 = time.perf_counter()
    ids0, _ = proc({"prompt": prime}, sp)
    log(f"prime: {ms(t0):.2f} ms, {len(ids0)} tokens, {len(prime)} chars; {cache.summary()}")

    rows = []
    tails = []
    for p in real[1]:
        rec.clear()
        t0 = time.perf_counter()
        ids, _ = proc({"prompt": p}, sp)
        total = ms(t0)
        tok_calls = rec.get("tok", [])
        tok_ms = sum(t[0] for t in tok_calls)
        row = {"proc_ms": total, "encode_ms": rec["encode"], "lookup_ms": rec["lookup"],
               "tok_ms": tok_ms, "tok_calls": len(tok_calls),
               "tail_chars": tok_calls[0][1] if tok_calls else 0,
               "tail_tokens": tok_calls[0][2] if tok_calls else 0, "n_ids": len(ids)}
        row["rest_ms"] = row["encode_ms"] - row["lookup_ms"] - row["tok_ms"]
        rows.append(row)
        if len(tok_calls) == 1:
            tails.append(rec["tail"])
        # (c) components, replicated on the returned ids
        arr = array("i", ids)
        n_split = len(ids) - row["tail_tokens"] + 64 if row["tail_tokens"] else len(ids)
        tl = ids[n_split:]
        t0 = time.perf_counter()
        a2 = arr[:n_split]
        a2.extend(tl)
        row["c_splice_ms"] = ms(t0)
        t0 = time.perf_counter()
        a2.tolist()
        row["c_tolist_ms"] = ms(t0)
        # (d) submit
        try:
            from tensorrt_llm.executor.request import GenerationRequest
            t0 = time.perf_counter()
            gr = GenerationRequest(ids, sampling_params=sp, streaming=True)
            row["d_genreq_ms"] = ms(t0)
            gr.set_id(len(rows))
            t0 = time.perf_counter()
            data = pickle.dumps(gr)
            row["d_pickle_ms"] = ms(t0)
            row["d_bytes"] = len(data)
            t0 = time.perf_counter()
            gr2 = pickle.loads(data)
            row["d_unpickle_ms"] = ms(t0)
            buf = gr2._prompt_token_ids_i32
        except Exception as e:  # noqa: BLE001
            row["d_err"] = repr(e)
            buf = None
        # (e) executor round trip
        try:
            import numpy as np

            from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
            from tensorrt_llm.bindings import executor as tllm
            if buf is None:
                buf = np.asarray(ids, dtype=np.int32)
            t0 = time.perf_counter()
            er = tllm.Request(input_token_ids=buf, max_tokens=256, streaming=True,
                              sampling_config=sp._get_sampling_config(),
                              output_config=sp._get_output_config(is_pytorch_backend=True),
                              end_id=-1)
            row["e_request_ms"] = ms(t0)
            t0 = time.perf_counter()
            lst = er.input_token_ids
            row["e_get_list_ms"] = ms(t0)
            row["e_list_equal"] = list(lst) == ids
            t0 = time.perf_counter()
            llm_req = executor_request_to_llm_request(len(rows), er, [], False)
            row["e_to_llm_request_ms"] = ms(t0)
            t0 = time.perf_counter()
            llm_req.get_tokens(0)
            row["e_get_tokens_ms"] = ms(t0)
        except Exception as e:  # noqa: BLE001
            row["e_err"] = repr(e)[:300]
        log("c1 hit: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in row.items()}))
    log(f"cache summary after c1 real: {cache.summary()}")
    seg = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and
                not isinstance(r.get(key), bool)]
        if vals:
            seg[key] = {"median": med(vals), "p10": round(pct(vals, 0.1), 4),
                        "p90": round(pct(vals, 0.9), 4)}
    for k, v in seg.items():
        log(f"segment {k}: {v}")

    # Parallel exploration on the observed hit tails.
    def hf_single(tail):
        enc = tok(tail, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def hf_batch(tail, cuts):
        enc = tok(pieces_of(tail, cuts), add_special_tokens=False, return_offsets_mapping=True)
        return concat(cuts, enc["input_ids"], enc["offset_mapping"])

    def be_batch(tail, cuts):
        encs = bt.encode_batch(pieces_of(tail, cuts), add_special_tokens=False)
        return concat(cuts, [e.ids for e in encs], [e.offsets for e in encs])

    pool = ThreadPoolExecutor(max_workers=8)

    def tp_batch(tail, cuts):
        encs = list(pool.map(lambda p: bt.encode_batch([p], add_special_tokens=False)[0],
                             pieces_of(tail, cuts)))
        return concat(cuts, [e.ids for e in encs], [e.offsets for e in encs])

    def be_raw(tail, cuts):
        bt.encode_batch(pieces_of(tail, cuts), add_special_tokens=False)
        return None

    variants = {"hf_single": lambda t: hf_single(t)}
    for k in (1, 2, 3, 4, 5, 6, 8):
        variants[f"hf_batch_{k}"] = (lambda kk: lambda t: hf_batch(t, cut_points(t, kk)))(k)
        variants[f"be_batch_{k}"] = (lambda kk: lambda t: be_batch(t, cut_points(t, kk)))(k)
        variants[f"be_raw_{k}"] = (lambda kk: lambda t: be_raw(t, cut_points(t, kk)))(k)
        if k > 1:
            variants[f"tp_{k}"] = (lambda kk: lambda t: tp_batch(t, cut_points(t, kk)))(k)
    timing = {name: [] for name in variants}
    exact = {name: True for name in variants}
    for t in tails:
        ref = hf_single(t)
        ref = (list(ref[0]), [tuple(o) for o in ref[1]])
        for name, fn in variants.items():
            for _ in range(2):
                fn(t)
            best = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                out = fn(t)
                best.append(ms(t0))
            timing[name].append(pct(best, 0.5))
            if out is not None:
                got = (list(out[0]), [tuple(o) for o in out[1]])
                if got != ref:
                    exact[name] = False
    par = {name: {"median": med(v), "exact": exact[name]} for name, v in timing.items()}
    for name, v in par.items():
        log(f"tail variant {name}: {v}")
    log(f"tail pieces at k=4 for the first tail: {len(cut_points(tails[0], 4))} pieces, "
        f"chars {[len(p) for p in pieces_of(tails[0], cut_points(tails[0], 4))]}")

    # GIL release test: a Python spin thread's progress while the main thread
    # runs a call in a loop for ~0.3 s, relative to the spin rate alone.
    def spin_rate(work):
        stop = threading.Event()
        cnt = [0]

        def spin():
            c = 0
            while not stop.is_set():
                c += 1
            cnt[0] = c

        th = threading.Thread(target=spin)
        t0 = time.perf_counter()
        th.start()
        if work is None:
            time.sleep(0.3)
        else:
            while time.perf_counter() - t0 < 0.3:
                work()
        stop.set()
        th.join()
        return cnt[0] / (time.perf_counter() - t0)

    tail0 = tails[0]
    alone = spin_rate(None)
    gil = {
        "be_encode_single": spin_rate(lambda: bt.encode(tail0, add_special_tokens=False)) / alone,
        "be_encode_batch_1": spin_rate(lambda: bt.encode_batch([tail0], add_special_tokens=False))
        / alone,
        "hf_call_single": spin_rate(lambda: hf_single(tail0)) / alone,
    }
    log(f"GIL: spin-thread progress fraction while tokenizing: {gil}")

    # Concurrent wave: 4 threads each tokenizing a tail (c4 leaders), single vs batched.
    def wave(fn):
        with ThreadPoolExecutor(max_workers=4) as wp:
            t0 = time.perf_counter()
            list(wp.map(fn, tails[:4]))
            return ms(t0)

    wv = {}
    for name in ("hf_single", "hf_batch_4", "be_batch_4"):
        fn = variants[name]
        xs = [wave(fn) for _ in range(5)]
        wv[name] = med(xs)
    log(f"4 concurrent tails wall ms: {wv}")

    out = {"tree": ti, "base_commit": BASE_COMMIT, "env": env, "cpu": cpu, "tokenizer": tk_info,
           "segments_c1_real": seg, "n_hits": len(rows), "tail_variants": par, "gil": gil,
           "wave4": wv, "cache_summary": cache.summary()}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
