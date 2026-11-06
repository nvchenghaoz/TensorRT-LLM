"""Shared helpers for idea 307 (DeepSeek-V4 embed_tokens use_gather_kernel).

The flag is read from THIS tree's DeepseekV4Model.__init__ source, so run_gpu_pair's
baseline arm (main) builds the masked lookup and the candidate arm builds the gather
lookup, each exactly as its own model would.
"""
import json
import os
import re
import sys
import time

import torch


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def tree_flag():
    """(use_gather_kernel as DeepseekV4Model.__init__ passes it, path of the file read)."""
    import tensorrt_llm
    root = os.path.dirname(tensorrt_llm.__file__)
    path = os.path.join(root, "_torch", "models", "modeling_deepseekv4.py")
    with open(path) as f:
        src = f.read()
    hits = [m.start() for m in re.finditer(r"self\.embed_tokens = Embedding\(", src)]
    if len(hits) != 1:
        raise RuntimeError(f"expected 1 embed_tokens ctor in {path}, found {len(hits)}")
    i = src.index("(", hits[0])
    depth, j = 0, i
    while True:
        c = src[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        j += 1
    call = src[i:j + 1]
    flag = bool(re.search(r"use_gather_kernel\s*=\s*True", call))
    return flag, path, " ".join(call.split())


def model_dims(ckpt):
    V, H, dt = 129280, 4096, torch.bfloat16
    hc = 4
    try:
        with open(os.path.join(ckpt, "config.json")) as f:
            cfg = json.load(f)
        V = int(cfg["vocab_size"])
        H = int(cfg["hidden_size"])
        hc = int(cfg.get("hc_mult", 4))
    except Exception as e:  # noqa: BLE001
        log("config.json not read, using defaults:", repr(e))
    return V, H, hc, dt


def make_embedding(V, H, dtype, use_gather_kernel, weight=None):
    from tensorrt_llm._torch.modules.embedding import Embedding
    with torch.device("cuda"):
        emb = Embedding(V, H, dtype=dtype, use_gather_kernel=use_gather_kernel)
    emb = emb.cuda()
    if weight is not None:
        emb.weight.data = weight
    assert emb.tp_size == 1
    return emb


def nvsmi():
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,clocks.sm",
             "--format=csv"], capture_output=True, text=True, timeout=30).stdout
        log("nvidia-smi:\n" + out)
    except Exception as e:  # noqa: BLE001
        log("nvidia-smi failed", repr(e))


def ckpt_arg():
    return sys.argv[1] if len(sys.argv) > 1 else ""
