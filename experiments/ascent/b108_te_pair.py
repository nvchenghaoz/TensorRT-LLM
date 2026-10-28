"""Branch 108 paired microbenchmark: graphed T=80 text encode() GPU time (CUDA events).

Runs in either tree. Loads the FLUX.3 text encoder as served (torch_compile on), warms it
like the pipeline (warmup_buckets() + encode(['warmup'])), then times 96 encode([prompt])
calls over 12 T=80 prompts with CUDA events (start before encode, end after, synced).
argv[-1] = checkpoint root.
"""
import hashlib
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent"))
from te_common import load_embedder  # noqa: E402

PROMPTS = [
    "A cat sitting on a red sofa.",
    'A sign that says "OPEN 24/7" with \'neon\' letters\nand a newline.',
    "Café au lait, 東京の夜景, émojis 🎨🚀, Ünïcödé.",
    "A red fox standing in fresh snow at sunrise",
    "A bowl of ramen with a soft-boiled egg on a wooden table, studio lighting",
    "A futuristic city skyline at night with neon signs reflected in rain puddles",
    "An old lighthouse on a rocky coast during a storm, dramatic clouds",
    "A watercolor painting of a hummingbird drinking from a purple flower",
    "A vintage red bicycle leaning against a yellow brick wall",
    "A close-up portrait of an elderly fisherman with a weathered face",
    "A cozy library with tall bookshelves and a fireplace, warm light",
    "Three green apples and a glass of water on a white tablecloth",
]

with torch.inference_mode():
    emb = load_embedder(sys.argv[-1])
    t0 = time.time()
    emb.warmup_buckets()
    emb.encode(["warmup"])
    torch.cuda.synchronize()
    print(f"warmup {time.time() - t0:.1f}s graphs {list(emb._graphs)}", flush=True)
    digest = hashlib.sha256()
    for p in PROMPTS:
        digest.update(emb.encode([p])[0].view(torch.int16).cpu().numpy().tobytes())
    ms = []
    for i in range(12 + 96):
        p = PROMPTS[i % len(PROMPTS)]
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        emb.encode([p])
        e.record()
        e.synchronize()
        if i >= 12:
            ms.append(s.elapsed_time(e))
        if i % 24 == 0:
            print(f"iter {i}: {s.elapsed_time(e):.3f} ms", flush=True)
    s_ = sorted(ms)
    med = statistics.median(ms)
    noise = (s_[int(0.9 * len(s_))] - s_[int(0.1 * len(s_))]) / med
    print(f"encode median {med:.4f} ms p10 {s_[int(0.1 * len(s_))]:.4f} "
          f"p90 {s_[int(0.9 * len(s_))]:.4f} ctx sha256 {digest.hexdigest()}", flush=True)
    print(json.dumps({
        "batch": {"1": {"kernel_ms": round(med, 4), "noise_floor": round(noise, 4)}},
        "output": {"class": "exact"},
        "ctx_sha256": digest.hexdigest(),
    }))
