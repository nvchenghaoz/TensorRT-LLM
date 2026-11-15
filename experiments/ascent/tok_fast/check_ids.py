#!/usr/bin/env python3
"""Branch 32 evaluator job: frozen check adapter + in-server id digests + cut-rule proof. One job.

Usage: check_ids.py <checkpoint>
 (0) Cut-rule proof: every vocab token containing '>' + '▁', every BPE merge whose left part ends
     in '>' and right part starts with '▁' (flagging byte-fallback '<0xNN>' / '<unk>' symbols), the
     encoder's bad set and its unknown-character guard. A byte-fallback fuzz (characters with no
     single-character token, each followed by a space) is compared against the stock call with
     forced 64-char splitting under the current rule and, for information, with the guard off.
     Cut positions on the workload prompts are compared between the guarded and unguarded rule.
 (1) The offline id check (tok_bench.py check: 200 workload prompts + 2000 fuzz, 3x, 4 threads).
 (2) The frozen check adapter (experiments/ascent-benchmark/check.py, reference.json, protected
     tolerances, its own fresh server with block reuse off) on this tree over all 20 cases,
     including its repeated-execution re-send, with the sitecustomize hook logging, per tokenizer
     call inside the server, sha256 of out[0] and the prompt text.
 (3) Offline: stock ip._tokenizer(text, return_tensors='pt')['input_ids'][0].to(int32).tolist()
     for every logged prompt; digests compared (0 mismatches required).
The job produces all its own inputs and exits when its work is done.
"""
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
T0 = time.time()


def log(*a):
    print(f"[+{time.time() - T0:7.1f}s]", *a, flush=True)


def _hb():
    while True:
        time.sleep(30)
        log("heartbeat")


def sha_ids(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def run_stream(cmd, env=None, tag="sub"):
    log(f"[{tag}] run: {' '.join(cmd[:3])} ...")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                         env=env)
    last = ""
    for line in p.stdout:
        s = line.rstrip("\n")
        if s.strip():
            last = s.strip()
        # The adapter's result line is large; print a prefix only (the full JSON is saved).
        print(f"[{tag}] {s[:2000]}", flush=True)
    rc = p.wait()
    log(f"[{tag}] exit {rc}")
    try:
        return rc, json.loads(last)
    except ValueError:
        return rc, None


_STOCK = {}


def stock_ids_cached(ip, t):
    from tok_bench import stock_ids
    h = hashlib.sha256(t.encode()).hexdigest()
    if h not in _STOCK:
        _STOCK[h] = stock_ids(ip, t)
    return _STOCK[h]


def proof(ckpt):
    import tok_bench
    tok_bench.ckpt = ckpt
    import tensorrt_llm._torch.models.gemma4_text_tokenize as gt
    from tok_bench import build_ip, stock_ids
    ip, tok, sp = build_ip()
    enc = ip._text_encoder
    b = enc._backend
    res = {"can_split": enc.can_split, "bad_prev": sorted(enc._bad_prev or []),
           "guard_unknown": enc._guard_unknown, "single_char_tokens": len(enc._single)}
    vocab = b.get_vocab(with_added_tokens=True)
    gt_meta = sorted((t, i) for t, i in vocab.items() if ">▁" in t)
    res["vocab_tokens_containing_gt_meta"] = [[t, i] for t, i in gt_meta]
    model = json.loads(b.to_str())["model"]
    res["byte_fallback"] = model.get("byte_fallback")
    res["unk_token"] = model.get("unk_token")
    merges = model.get("merges") or []
    byte_re = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    hits = []
    for m in merges:
        left, right = (m.split(" ", 1) if isinstance(m, str) else (m[0], m[1]))
        if left.endswith(">") and right.startswith("▁"):
            hits.append({"left": left, "right": right,
                         "left_is_byte_or_unk": bool(byte_re.match(left)) or left == model.get("unk_token")})
    res["merges_count"] = len(merges)
    res["merges_gt_left_meta_right"] = hits
    res["byte_tokens_in_vocab"] = sum(1 for t in vocab if byte_re.match(t))
    log("cut proof:", json.dumps(res, ensure_ascii=False)[:4000])

    # Byte-fallback fuzz: characters with no single-character model token, then a space.
    rng = random.Random(20261115)
    rare = []
    cps = (list(range(0x80, 0x3000)) + list(range(0x3400, 0x9FFF, 7)) + list(range(0xE000, 0xE100))
           + list(range(0x1F300, 0x1FAFF)) + list(range(0x20000, 0x2A6DF, 97)) + list(range(0xE0020, 0xE0080)))
    for cp in cps:
        ch = chr(cp)
        if (ch.isprintable() or cp >= 0xE000) and ch not in enc._single and ch != " ":
            rare.append(ch)
    if not rare:
        rare = ["\U000E0041", ""]
    res["rare_chars_found"] = len(rare)
    log(f"rare (no single-char token) characters: {len(rare)}, e.g. {[hex(ord(c)) for c in rare[:12]]}")
    cp_path = Path("experiments/ascent-benchmark/corpus.txt")
    corpus = cp_path.read_text(errors="replace") if cp_path.exists() else "the quick brown fox " * 500
    texts = []
    for _ in range(600):
        parts, cur, L = [], 0, rng.choice([300, 1500, 4000, 9000])
        while cur < L:
            r = rng.random()
            if r < 0.45:
                s = rng.randrange(0, max(1, len(corpus) - 300))
                seg = corpus[s:s + rng.randrange(3, 200)]
            elif r < 0.8:
                seg = "".join(rng.choice(rare) for _ in range(rng.randrange(1, 3))) + rng.choice(
                    [" ", "  ", " a", " ▁"])
            else:
                seg = rng.choice([" ", "> ", ">", "▁ ", "x "])
            parts.append(seg)
            cur += len(seg)
        texts.append("".join(parts))
    refs = [stock_ids(ip, t) for t in texts]
    saved = (gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS)
    fz = {}
    try:
        gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS = 64, 64
        for guard in (True, False):
            enc._guard_unknown = guard
            bad, cuts_total, first = 0, 0, None
            for t, r in zip(texts, refs):
                cuts_total += len(enc._cuts(t))
                got = enc.encode(t)
                if got != r:
                    bad += 1
                    if first is None:
                        j = next((q for q in range(min(len(got), len(r))) if got[q] != r[q]), min(len(got), len(r)))
                        first = {"at": j, "got": got[max(0, j - 3):j + 3], "ref": r[max(0, j - 3):j + 3]}
            fz["guard_on" if guard else "guard_off_informational"] = {"texts": len(texts), "mismatches": bad,
                                                                       "cuts": cuts_total, "first": first}
            log(f"byte-fallback fuzz forced-64 guard={guard}: mismatches {bad} cuts {cuts_total} first {first}")
    finally:
        gt.SPLIT_MIN_CHARS, gt.PIECE_CHARS = saved
        enc._guard_unknown = res["guard_unknown"]
    bad_def = sum(1 for t, r in zip(texts, refs) if enc.encode(t) != r)
    fz["default_settings_mismatches"] = bad_def
    log(f"byte-fallback fuzz default settings: mismatches {bad_def}")
    res["byte_fallback_fuzz"] = fz
    # Workload prompts: cut positions identical with and without the guard (timing pair stays valid).
    common = tok_bench.load_common()
    wp = tok_bench.workload_prompts(common, 40)
    same, total = 0, 0
    for _, ts in wp.items():
        for t in ts:
            enc._guard_unknown = True
            a = enc._cuts(t)
            enc._guard_unknown = False
            c = enc._cuts(t)
            total += 1
            same += int(a == c)
    enc._guard_unknown = res["guard_unknown"]
    res["workload_cut_positions_same_as_unguarded"] = [same, total]
    log(f"workload prompts with identical cuts guarded vs unguarded: {same}/{total}")
    return ip, res


def main():
    threading.Thread(target=_hb, daemon=True).start()
    ckpt = sys.argv[1]
    out = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or "/tmp/ascent_b32")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"errors": []}

    # (0) cut proof + byte-fallback fuzz
    ip, pr = proof(ckpt)
    summary["cut_proof"] = pr

    # (1) offline id check on this tree
    rc, tb = run_stream([sys.executable, str(HERE / "tok_bench.py"), ckpt, "check"], tag="tok_bench")
    summary["offline_id_check"] = tb
    if not tb or not tb.get("passed"):
        summary["errors"].append(f"offline id check failed rc={rc}")
    if len(sys.argv) > 2 and sys.argv[2] == "offline":
        # Offline only: cut proof, byte-fallback fuzz, workload + fuzz id equality (no server).
        fz = pr["byte_fallback_fuzz"]
        summary["passed"] = bool(not summary["errors"] and fz["guard_on"]["mismatches"] == 0
                                 and fz["default_settings_mismatches"] == 0)
        (out / "summary.json").write_text(json.dumps(summary))
        print(json.dumps(summary, ensure_ascii=True))
        return

    # (2) frozen check adapter with the id-digest hook in its server
    ref = json.loads(Path("experiments/ascent-benchmark/reference.json").read_text())
    cfg = {
        "workload_identity": {
            "workload": {"runtime": "tensorrt-llm", "model": "nvidia/Gemma-4-31B-IT-NVFP4", "gpus_per_instance": 1,
                         "cached_prefix_tokens": 0, "output_tokens": 1, "concurrency": [1, 4, 16, 64],
                         "cases": [{"name": "isl300", "unique_input_tokens": 300},
                                   {"name": "isl1k", "unique_input_tokens": 1000},
                                   {"name": "isl2k", "unique_input_tokens": 2000},
                                   {"name": "isl4k", "unique_input_tokens": 4000},
                                   {"name": "isl8k", "unique_input_tokens": 8000}]},
            "resolved_model": {"id": "nvidia/Gemma-4-31B-IT-NVFP4",
                               "revision": "e5ef03afa233c35cb000323ff098d4291e1dd07c", "checkpoint": ckpt}},
        "served_model_name": "gemma-4-31b-it-nvfp4",
        "server": {"max_batch_size": 2048, "max_num_tokens": 8192,
                   "kv_cache_config": {"enable_block_reuse": False, "free_gpu_memory_fraction": 0.9}},
        "warmup": {"single": 2, "burst": 8, "small_isl_burst": 32},
        "server_start_timeout_s": 2700, "output_class": "exact",
        "prompt_seed": ref.get("prompt_seed"), "prompts_per_case": ref.get("prompts_per_case"),
        "repeats_per_case": ref.get("repeats_per_case"),
    }
    log("harness config:", json.dumps(cfg))
    log("reference:", {k: ref.get(k) for k in ("prompt_seed", "prompts_per_case", "repeats_per_case",
                                                  "prompt_digest", "tolerance", "body_sha256")})
    toklog = out / "toklog"
    toklog.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["ASCENT_TOK_LOG"] = str(toklog)
    env["ASCENT_TOK_TEXT"] = "1"
    env["PYTHONPATH"] = str(HERE / "site") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["ASCENT_ARTIFACT_DIR"] = str(out)
    rc, cr = run_stream([sys.executable, "experiments/ascent-benchmark/check.py", json.dumps(cfg)], env=env,
                        tag="check")
    if cr is not None:
        (out / "check_result.json").write_text(json.dumps(cr))
    if not cr or not cr.get("valid"):
        summary["errors"].append(f"check adapter invalid rc={rc}: {str((cr or {}).get('errors', ''))[:1500]}")
    per_isl = {}
    if cr and cr.get("comparisons"):
        for label, c in cr["comparisons"].items():
            isl = label.split("@")[0]
            e = per_isl.setdefault(isl, {"first_pass_passed": 0, "first_pass_total": 0, "failed_keys": [],
                                         "repeat_passed": 0, "repeat_total": 0, "repeat_detail": []})
            e["first_pass_passed"] += c["passed_count"]
            e["first_pass_total"] += c["total"]
            e["failed_keys"] += [[p["key"], p.get("distance"), p.get("tolerance")] for p in c["per_prompt"]
                                 if not p["passed"]]
        for r in cr.get("repeated_execution", []):
            isl = r["key"].split("@")[0]
            e = per_isl[isl]
            e["repeat_total"] += 1
            e["repeat_passed"] += int(bool(r.get("passed")))
            e["repeat_detail"].append({"key": r["key"], "passed": r.get("passed"),
                                       "to_first": r.get("distance_to_first_execution"),
                                       "to_reference": r.get("distance_to_reference"), "tol": r.get("tolerance")})
        for isl, e in per_isl.items():
            e["passed"] = (e["first_pass_passed"] == e["first_pass_total"] and e["repeat_passed"] == e["repeat_total"])
            log(f"check {isl}: first pass {e['first_pass_passed']}/{e['first_pass_total']} "
                f"repeat {e['repeat_passed']}/{e['repeat_total']} -> {'PASS' if e['passed'] else 'FAIL'} "
                f"failed {e['failed_keys']} repeats {e['repeat_detail']}")
    summary["check"] = {"valid": bool(cr and cr.get("valid")), "per_isl": per_isl,
                        "repeated_execution_passed": (cr or {}).get("repeated_execution_passed"),
                        "summary": (cr or {}).get("summary"), "reference": (cr or {}).get("reference"),
                        "server_import_mode": ((cr or {}).get("server") or {}).get("import_mode")}

    # (3) offline digest compare for every prompt the server tokenized
    recs = []
    for p in sorted(toklog.glob("tok_*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    mism, by_len, seen = [], {}, {}
    for r in recs:
        t = r.get("text")
        if t is None:
            mism.append({"reason": "no text logged", "n_ids": r.get("n_ids")})
            continue
        stock = stock_ids_cached(ip, t)
        ok = sha_ids(stock) == r["ids_sha256"]
        n = str(r["n_ids"])
        d = by_len.setdefault(n, {"calls": 0, "mismatches": 0, "texts_tokenized_twice_or_more": 0})
        d["calls"] += 1
        d["mismatches"] += int(not ok)
        seen[(n, r["text_sha256"])] = seen.get((n, r["text_sha256"]), 0) + 1
        if not ok:
            mism.append({"n_ids": r["n_ids"], "stock_len": len(stock), "text_sha256": r["text_sha256"]})
    for (n, _), k in seen.items():
        if k >= 2:
            by_len[n]["texts_tokenized_twice_or_more"] += 1
    summary["in_server_ids"] = {
        "calls": len(recs), "mismatches": len(mism), "by_n_ids": by_len, "first_mismatches": mism[:5],
        "hook_pids": sorted({r.get("pid") for r in recs}),
        "pool_threads_max": max((r.get("pool_threads_after", 0) for r in recs), default=None),
        "compared_against": "stock ip._tokenizer(text, return_tensors='pt')['input_ids'][0].to(int32).tolist(), "
                            "sha256 of ','.join(ids)"}
    log("in-server ids:", json.dumps(summary["in_server_ids"])[:3000])
    if not recs:
        summary["errors"].append("hook logged no tokenizer calls")
    fz = pr["byte_fallback_fuzz"]
    passed = (not summary["errors"] and len(mism) == 0 and bool(recs)
              and len(per_isl) == 5 and all(e["passed"] for e in per_isl.values())
              and fz["guard_on"]["mismatches"] == 0 and fz["default_settings_mismatches"] == 0)
    summary["passed"] = bool(passed)
    (out / "summary.json").write_text(json.dumps(summary))
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
