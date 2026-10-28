"""Idea 9 paired microbenchmark: Qwen3-VL text encode() wall time at T=80 (encoder only).

Self-contained (run via python -c in both trees). Loads the FLUX.3 text encoder with
torch_compile on (as served), warms it the way the tree's pipeline warmup does
(warmup_buckets() + encode(["warmup"]) on the branch, encode(["warmup"]) on main), then
times encode([prompt]) synchronized, 100 iterations over 12 prompts in the T=80 bucket.
"""
import hashlib, json, statistics, sys, time  # noqa: E401
import torch

CKPT = sys.argv[-1]
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
from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.text_encoder import (  # noqa: E402
    Qwen3VLEmbedder, Qwen3VLEmbedderParams)

torch._dynamo.config.cache_size_limit = 128
pkg = Flux3Checkpoint(CKPT, "nvfp4")
cfg = pkg.config["text_encoder"]
emb = Qwen3VLEmbedder(
    str(pkg.asset(cfg["path"])),
    Qwen3VLEmbedderParams(output_layer=cfg["layers"], max_length=cfg["max_tokens"],
                          pad_multiple=cfg["pad_multiple"], torch_dtype=cfg["dtype"],
                          use_compile=True),
    tokenizer_spec=str(pkg.asset(cfg["tokenizer"])),
).to("cuda").eval()
branch = hasattr(emb, "warmup_buckets")
print("tree:", "branch" if branch else "base", flush=True)
t0 = time.time()
if branch:
    emb.warmup_buckets()
emb.encode(["warmup"])
torch.cuda.synchronize()
print(f"warmup {time.time() - t0:.1f}s", flush=True)
shapes = set()
for p in PROMPTS:
    shapes.add(tuple(emb.encode([p])[0].shape))
torch.cuda.synchronize()
print("ctx shapes", shapes, flush=True)
digest = hashlib.sha256()
for p in PROMPTS:
    digest.update(emb.encode([p])[0].view(torch.int16).cpu().numpy().tobytes())
walls, rets = [], []
for i in range(100):
    p = PROMPTS[i % len(PROMPTS)]
    torch.cuda.synchronize()
    a = time.perf_counter()
    emb.encode([p])
    b = time.perf_counter()
    torch.cuda.synchronize()
    c = time.perf_counter()
    walls.append((c - a) * 1e3)
    rets.append((b - a) * 1e3)
    if i % 25 == 0:
        print(f"iter {i}: {walls[-1]:.3f} ms (host return {rets[-1]:.3f} ms)", flush=True)
s = sorted(walls)
med = statistics.median(walls)
noise = (s[int(0.9 * len(s))] - s[int(0.1 * len(s))]) / med
print(f"encode wall median {med:.3f} ms, host-return median {statistics.median(rets):.3f} ms, "
      f"ctx sha256 {digest.hexdigest()[:16]}", flush=True)
print(json.dumps({
    "batch": {"1": {"step_ms": round(med, 4), "noise_floor": round(noise, 4)}},
    "output": {"class": "exact"},
    "host_return_ms_median": round(statistics.median(rets), 4),
    "ctx_sha256": digest.hexdigest(),
}))
