#!/usr/bin/env python3
"""Probe Gemma4 tokenizer shape + vocab scan for exact cut boundaries (branch 32, idea 62).

Usage: probe.py <checkpoint> <worktree>
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

T0 = time.time()


def log(*a):
    print(f"[+{time.time() - T0:6.1f}s]", *a, flush=True)


ckpt, wt = sys.argv[1], sys.argv[2]
tj = json.loads(Path(ckpt, "tokenizer.json").read_text())
model = tj["model"]
log("normalizer:", json.dumps(tj.get("normalizer"))[:800])
log("pre_tokenizer:", json.dumps(tj.get("pre_tokenizer"))[:800])
log("post_processor:", json.dumps(tj.get("post_processor"))[:1200])
log("decoder:", json.dumps(tj.get("decoder"))[:400])
log("truncation:", tj.get("truncation"), "padding:", tj.get("padding"))
log("model keys:", {k: (v if not isinstance(v, (dict, list)) else f"<{type(v).__name__} {len(v)}>")
                    for k, v in model.items()})
vocab = model["vocab"]
if isinstance(vocab, list):  # unigram
    toks = [t[0] for t in vocab]
else:
    toks = list(vocab.keys())
added = tj.get("added_tokens", [])
log(f"vocab size {len(toks)}, added tokens {len(added)}")
flag_counts = Counter((a.get("special"), a.get("normalized"), a.get("lstrip"), a.get("rstrip"),
                       a.get("single_word")) for a in added)
log("added flags (special,normalized,lstrip,rstrip,single_word):", dict(flag_counts))
nonspecial = [a for a in added if not a.get("special")]
log("non-special added tokens:", [a["content"] for a in nonspecial][:80])
ws_added = [a["content"] for a in added if any(ch.isspace() or ch == "▁" for ch in a["content"])]
log("added tokens containing whitespace/▁:", ws_added[:80], len(ws_added))
log("max added token len:", max(len(a["content"]) for a in added))
log("max vocab token len:", max(len(t) for t in toks))
merges = model.get("merges")
if merges is not None:
    log("merges:", len(merges), "first:", merges[:3])

# Vocab scan: chars c such that c followed by '▁' occurs in some token (model vocab or added).
alltok = toks + [a["content"] for a in added] + [a["content"].replace(" ", "▁") for a in added]
bad_space = Counter()
bad_nl = Counter()
for t in alltok:
    for j in range(1, len(t)):
        if t[j] == "▁":
            bad_space[t[j - 1]] += 1
        if t[j] == "\n" and t[j - 1] != "\n":
            bad_nl[t[j - 1]] += 1
log(f"chars c with c+'▁' inside a token: {len(bad_space)}; top:", bad_space.most_common(40))
alpha_bad = sorted(c for c in bad_space if c.isalnum())
log(f"alnum chars among them: {len(alpha_bad)}:", "".join(alpha_bad[:200]))
log(f"chars c with c+'\\n' inside a token: {len(bad_nl)}; top:", bad_nl.most_common(40))
# Sample tokens that contain c+'▁' for letters
ex = [t for t in toks if re.search(r"[A-Za-z]▁", t)][:30]
log("example tokens with letter+▁:", ex)
ex2 = [t for t in toks if re.search(r"[.,;:!?]▁", t)][:30]
log("example tokens with punct+▁:", ex2)

# Benchmark prompt construction
cp = Path(wt, "experiments/ascent-benchmark/common.py")
if cp.exists():
    src = cp.read_text()
    for i, line in enumerate(src.splitlines()):
        if re.match(r"\s*def ", line) or "corpus" in line or "completions" in line or "chat" in line:
            print(f"common.py:{i + 1}: {line}")
else:
    log("no common.py at", cp)
print(json.dumps({"passed": True, "n_bad_space_chars": len(bad_space), "n_bad_nl_chars": len(bad_nl)}))
