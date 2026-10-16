# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the tokenization-cache microbenchmark and checks.

Prompts mirror the frozen serving benchmark: AIPerf's synthetic text generator
(Shakespeare corpus tokenized in 10K-char chunks, random token windows decoded
back to text), a 32768-token shared prefix and a 2048-token unique suffix per
request, joined the way AIPerf joins a prefix prompt (``f"{prefix} {content}"``).
No model weights are loaded; only the tokenizer and the input processor.
"""

import os
import random
import sys
import time

PREFIX_TOKENS = 32768
SUFFIX_TOKENS = 2048


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tokenizer(ckpt):
    from tensorrt_llm.tokenizer import tokenizer_factory
    return tokenizer_factory(ckpt, trust_remote_code=True, use_fast=True)


def hf_tokenizer(tok):
    return getattr(tok, "tokenizer", tok)


def corpus_tokens(hf):
    import aiperf
    path = os.path.join(os.path.dirname(aiperf.__file__), "dataset", "generator",
                        "assets", "shakespeare.txt")
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    chunks, buf, count = [], [], 0
    for line in lines:
        buf.append(line)
        count += len(line)
        if count >= 10_000:
            chunks.append(" ".join(buf))
            buf, count = [], 0
    if buf:
        chunks.append(" ".join(buf))
    enc = hf.backend_tokenizer.encode_batch(chunks, add_special_tokens=False)
    toks = [t for e in enc for t in e.ids]
    return toks


class PromptMaker:

    def __init__(self, hf, seed=1234):
        self.hf = hf
        self.corpus = corpus_tokens(hf)
        self.rng = random.Random(seed)

    def sample(self, n):
        c = self.corpus
        start = self.rng.randrange(len(c))
        out = c[start:start + n]
        if len(out) < n:
            out += c[:n - len(out)]
        return self.hf.decode(out)

    def prefix(self):
        return self.sample(PREFIX_TOKENS)

    def request(self, prefix):
        return f"{prefix} {self.sample(SUFFIX_TOKENS)}"


def load_real_prompts(ckpt):
    """The frozen benchmark's own prompts, from the lead's proxy job artifacts.

    Returns ({conc: [text, ...]}, info) or (None, info) if none are found.
    """
    import glob
    import hashlib
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    pattern = os.path.join(root, "runner", "slots", "*", "jobs", "*", "artifacts", "proxy",
                           "c{}-input", "prompts.jsonl")
    jobs = {}
    for conc in (1, 2, 4):
        for path in glob.glob(pattern.format(conc)):
            job = path.split(os.sep + "artifacts" + os.sep)[0]
            jobs.setdefault(job, {})[conc] = path
    complete = sorted(j for j, files in jobs.items() if len(files) == 3)
    info = {"proxy_jobs_with_prompts": len(complete)}
    if not complete:
        return None, info
    digests = set()
    for job in complete:
        h = hashlib.sha256()
        for conc in (1, 2, 4):
            with open(jobs[job][conc], "rb") as f:
                h.update(f.read())
        digests.add(h.hexdigest()[:16])
    info["distinct_prompt_sets"] = sorted(digests)
    chosen = jobs[complete[0]]
    info["files"] = chosen
    prompts = {}
    for conc in (1, 2, 4):
        rows = []
        with open(chosen[conc]) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", row.get("texts", row.get("prompt")))
                if isinstance(text, list):
                    assert len(text) == 1, f"multi-text row: {len(text)}"
                    text = text[0]
                rows.append(text)
        if conc == 1:
            info["row_keys"] = sorted(row.keys())
        prompts[conc] = rows
    info["rows"] = {c: len(v) for c, v in prompts.items()}
    return prompts, info


def request_plan(maker, prefix, warmup, measured, concs=(1, 2, 4)):
    """Per concurrency: (conc, warmup prompts, measured prompts), in benchmark order."""
    plan = []
    for conc in concs:
        w = [maker.request(prefix) for _ in range(warmup)]
        m = [maker.request(prefix) for _ in range(measured)]
        plan.append((conc, w, m))
    return plan


def make_processor(ckpt, tok, enable_cache=True):
    from tensorrt_llm.inputs.registry import create_input_processor
    try:
        return create_input_processor(ckpt, tok,
                                      enable_tokenization_cache=enable_cache)
    except TypeError:
        return create_input_processor(ckpt, tok)


def processor_cache(proc):
    return getattr(proc, "_prefix_token_cache", None)


def pct(xs, q):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def env_info():
    import transformers
    try:
        import tokenizers
        tv = tokenizers.__version__
    except Exception:  # noqa: BLE001
        tv = "?"
    return {"python": sys.version.split()[0], "transformers": transformers.__version__,
            "tokenizers": tv}
