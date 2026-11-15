"""Idea 39 full-model numerics gate: in-process LLM A/B of the 64-row d512 prefill kernel.

One process (TLLM_WORKER_USE_SINGLE_PROCESS=1), main's engine defaults (FlashInfer FA2 on
SM120, FP8 KV from the checkpoint), block reuse off, max_num_tokens 8192. The kernel is
switched between passes with FLASHINFER_FA2_D512_TILE64 (read by the FlashInfer planner on
every plan): A = 0 (main's 32-row d512 kernel), B = 1 (the 64-row kernel). Passes A, B, A', B'.
Prompts: natural text (Python stdlib docstrings) in the chat template, cut to each ISL.
Classes: isl300@1, isl1k@4, isl8k@1, isl8k@4 (c=4 groups sent together with
batch_wait_timeout_ms so batch composition is fixed). 1 output token, greedy, top-20 logprobs.
Reports per class: A vs A' bitwise (harness determinism), B vs A bitwise, token flips and
max |delta logprob| of the reference token; the repeated sends (A' vs A, B' vs B) are the
repeated-execution evidence. Host prefill plan flag: main's default, same in every pass.
Last stdout line: JSON {"passed": ...}.
"""
import argparse
import inspect
import json
import math
import os
import struct
import sys
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
FLAG = "FLASHINFER_FA2_D512_TILE64"


def log(*a):
    print(*a, flush=True)


STDLIB = ["argparse", "collections", "json", "email", "http.client", "unittest", "logging",
          "asyncio", "typing", "pathlib", "subprocess", "threading", "itertools", "functools",
          "inspect", "os", "re", "string", "textwrap", "datetime", "decimal", "fractions",
          "statistics", "random", "sqlite3", "xml.etree.ElementTree", "urllib.request",
          "zipfile", "tarfile", "csv", "configparser", "shutil", "tempfile", "heapq", "bisect",
          "contextlib", "dataclasses", "enum", "abc", "io", "pickle", "copy", "pprint",
          "difflib", "calendar", "locale", "gettext", "optparse", "socket", "ssl", "selectors",
          "signal", "multiprocessing", "concurrent.futures", "queue", "sched", "doctest", "pdb",
          "timeit", "trace", "traceback", "warnings", "weakref", "types", "operator", "numbers",
          "array", "struct", "codecs", "html.parser", "mailbox", "mimetypes", "base64",
          "hashlib", "hmac", "secrets", "uuid", "ipaddress", "glob", "fnmatch", "tokenize",
          "ast", "dis", "platform", "sysconfig", "importlib", "smtplib", "ftplib", "imaplib",
          "poplib", "telnetlib", "xmlrpc.client", "wsgiref.simple_server", "turtle", "cmd",
          "shlex", "netrc", "plistlib", "zoneinfo", "graphlib", "tomllib"]


def corpus_text():
    import importlib
    parts = []
    for name in STDLIB:
        try:
            m = importlib.import_module(name)
        except Exception:
            continue
        d = inspect.getdoc(m)
        if d:
            parts.append(d)
        for _, obj in inspect.getmembers(m):
            if inspect.isclass(obj) or inspect.isfunction(obj):
                try:
                    d = inspect.getdoc(obj)
                except Exception:
                    d = None
                if d and len(d) > 300:
                    parts.append(d)
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return "\n\n".join(out)


def build_prompts(tok, classes):
    text = corpus_text()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    log("corpus tokens", len(ids))
    ph = "⁣PLACEHOLDER⁣"
    tmpl = tok.apply_chat_template([{"role": "user", "content": ph}], tokenize=False,
                                   add_generation_prompt=True)
    pre, suf = tmpl.split(ph)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    suf_ids = tok(suf, add_special_tokens=False)["input_ids"]
    off = 0
    prompts = {}
    for name, isl, conc, count in classes:
        body = isl - len(pre_ids) - len(suf_ids)
        lst = []
        for _ in range(count):
            if off + body > len(ids):
                off = (off * 7919 + 13) % max(1, len(ids) - body)  # wrap (distinct start)
            lst.append(pre_ids + ids[off:off + body] + suf_ids)
            off += body
        prompts[name] = (isl, conc, lst)
    return prompts


def summarize(out):
    o = out.outputs[0]
    tid = int(o.token_ids[0])
    lp = o.logprobs[0] if o.logprobs else {}
    items = []
    if isinstance(lp, dict):
        for k, v in lp.items():
            val = getattr(v, "logprob", v)
            items.append((int(k), float(val)))
    else:
        for v in lp:
            items.append((int(getattr(v, "token_id", -1)), float(getattr(v, "logprob", v))))
    items.sort(key=lambda x: (-x[1], x[0]))
    return tid, items


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def same(a, b):
    return a[0] == b[0] and len(a[1]) == len(b[1]) and all(
        p[0] == q[0] and bits(p[1]) == bits(q[1]) for p, q in zip(a[1], b[1]))


def ref_delta(ref, cand):
    """|logprob_cand(ref token) - logprob_ref(ref token)|; None if missing."""
    tok = ref[0]
    lr = dict(ref[1]).get(tok)
    lc = dict(cand[1]).get(tok)
    if lr is None or lc is None:
        return None
    return abs(lc - lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n300", type=int, default=10)
    ap.add_argument("--n1k", type=int, default=12)
    ap.add_argument("--n8k1", type=int, default=10)
    ap.add_argument("--n8k4", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    classes = [("isl300@1", 300, 1, a.n300), ("isl1k@4", 1000, 4, a.n1k),
               ("isl8k@1", 8000, 1, a.n8k1), ("isl8k@4", 8000, 4, a.n8k4)]
    prompts = build_prompts(tok, classes)
    for k, (isl, conc, lst) in prompts.items():
        log("class", k, "isl", isl, "c", conc, "n", len(lst), "lens", sorted({len(p) for p in lst}))

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    log("host prefill plan flag TRTLLM_FI_HOST_PREFILL_PLAN =",
        os.environ.get("TRTLLM_FI_HOST_PREFILL_PLAN", "<default>"))
    t0 = time.time()
    llm = LLM(model=a.ckpt, max_batch_size=64, max_num_tokens=8192, max_seq_len=9216,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.8),
              batch_wait_timeout_ms=1500)
    log("LLM ready in", round(time.time() - t0, 1), "s")
    sp = SamplingParams(max_tokens=1, logprobs=20)  # default sampling is greedy

    def run_pass(name, flag):
        os.environ[FLAG] = flag
        res = {}
        t = time.time()
        for cls, (isl, conc, lst) in prompts.items():
            outs = []
            for i in range(0, len(lst), conc):
                group = lst[i:i + conc]
                got = llm.generate(group, sp, use_tqdm=False)
                outs.extend(summarize(g) for g in got)
            res[cls] = outs
        log("pass", name, f"{FLAG}={flag}", "secs", round(time.time() - t, 1))
        return res

    # warm both paths once (JIT + autotune) outside the compared passes
    for flag in ("0", "1"):
        os.environ[FLAG] = flag
        llm.generate([prompts["isl1k@4"][2][0]], sp, use_tqdm=False)
    passes = {}
    for name, flag in (("A", "0"), ("B", "1"), ("A2", "0"), ("B2", "1")):
        passes[name] = run_pass(name, flag)

    report = {}
    ok = True
    for cls in prompts:
        A, B, A2, B2 = (passes[p][cls] for p in ("A", "B", "A2", "B2"))
        n = len(A)
        aa = sum(same(x, y) for x, y in zip(A, A2))
        bb = sum(same(x, y) for x, y in zip(B, B2))
        ba = sum(same(x, y) for x, y in zip(B, A))
        flips = [i for i in range(n) if A[i][0] != B[i][0]]
        deltas = [ref_delta(A[i], B[i]) for i in range(n)]
        md = max([d for d in deltas if d is not None], default=0.0)
        missing = sum(d is None for d in deltas)
        report[cls] = dict(n=n, A_vs_A2_bitwise=aa, B_vs_B2_bitwise=bb, B_vs_A_bitwise=ba,
                           token_flips=len(flips), flip_idx=flips,
                           max_abs_delta_ref_logprob=md, ref_token_missing=missing)
        ok &= aa == n and ba == n and bb == n
        log("CLASS", cls, json.dumps(report[cls]))
    total = sum(r["n"] for r in report.values())
    summary = {"passed": bool(ok), "prompts": total,
               "B_vs_A_bitwise_total": sum(r["B_vs_A_bitwise"] for r in report.values()),
               "A_vs_A2_bitwise_total": sum(r["A_vs_A2_bitwise"] for r in report.values()),
               "classes": report, "flag": FLAG,
               "host_prefill_plan": os.environ.get("TRTLLM_FI_HOST_PREFILL_PLAN", "default")}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
