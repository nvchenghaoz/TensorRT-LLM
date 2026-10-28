"""Idea 9 step 0b: padded text buckets of the protected prompt sets (tokenizer only)."""

import json
import math
import os
import sys
from collections import Counter

CKPT = sys.argv[1]
root = "experiments/ascent-benchmark"
# The protected benchmark dir may live outside the job's source snapshot:
# argv[2:] are candidate roots (e.g. from `find`); the first with files wins.
for c in [root] + sys.argv[2:]:
    if os.path.isdir(c) and any(fs for _, _, fs in os.walk(c)):
        root = c
        break
print("files under", root, ":", flush=True)
for dp, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(dp, f)
        print(" ", p, os.path.getsize(p), flush=True)

from transformers import AutoProcessor  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint  # noqa: E402

pkg = Flux3Checkpoint(CKPT, "nvfp4")
cfg = pkg.config["text_encoder"]
proc = AutoProcessor.from_pretrained(str(pkg.asset(cfg["tokenizer"])))
tok = proc.tokenizer


def bucket(p):
    text = proc.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
    )
    real = len(tok(text, truncation=True, max_length=cfg["max_tokens"])["input_ids"])
    return real, min(math.ceil(real / cfg["pad_multiple"]) * cfg["pad_multiple"], cfg["max_tokens"])


def walk(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("prompt", "prompts", "text", "caption") and isinstance(v, str):
                out.append((path + "/" + k, v))
            elif k in ("prompts",) and isinstance(v, list) and all(isinstance(x, str) for x in v):
                out.extend((f"{path}/{k}[{i}]", x) for i, x in enumerate(v))
            else:
                walk(v, path + "/" + str(k), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", out)


summary = {}
for dp, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(dp, f)
        found = []
        try:
            if f.endswith(".json"):
                walk(json.load(open(p)), "", found)
            elif f.endswith(".jsonl"):
                for i, line in enumerate(open(p)):
                    if line.strip():
                        walk(json.loads(line), f"[{i}]", found)
            elif f.endswith(".txt"):
                found = [(f"line{i}", x.rstrip("\n")) for i, x in enumerate(open(p)) if x.strip()]
            else:
                continue
        except Exception as e:  # noqa: BLE001
            print("skip", p, e, flush=True)
            continue
        if not found:
            continue
        uniq = list(dict.fromkeys(v for _, v in found))
        info = [bucket(v) for v in uniq]
        reals = [r for r, _ in info]
        c = Counter(b for _, b in info)
        summary[p] = {"n_unique": len(uniq), "buckets": dict(c), "real_min": min(reals),
                      "real_max": max(reals)}
        print(p, json.dumps(summary[p]), flush=True)
        for (_, v), (r, b) in zip([(None, u) for u in uniq], info):
            if b > 80:
                print("   >80:", r, b, repr(v[:120]), flush=True)
print(json.dumps({"passed": True, "summary": summary}))
