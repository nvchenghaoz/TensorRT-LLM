"""Idea 9 item 4: nsys view of text_encode spans (encoder only, as served: torch_compile on).

Loads the FLUX.3 text encoder, warms it like the pipeline warmup (warmup_buckets() then
encode(["warmup"])), then runs 12 encode() calls of T=80 prompts, each in an NVTX range
"text_encode", all inside NVTX range "te_requests". TRTLLM_FLUX3_TEXT_GRAPH=0 gives the
compiled graphs-off path in the same tree for the busy/idle comparison.
"""
import os
import sys
import time

import torch

CKPT = sys.argv[-1]
PROMPTS = [
    "A red fox standing in fresh snow at sunrise",
    "A lighthouse on a cliff during a thunderstorm, oil painting",
    "A bowl of ramen with a soft boiled egg, studio photo",
    "A futuristic city skyline at night with flying cars",
    "Portrait of an elderly fisherman with a white beard, dramatic lighting",
    "A cozy reading nook with a window, plants and warm lamp light",
    "A watercolor painting of a hot air balloon over green hills",
    "A golden retriever puppy playing with a tennis ball on grass",
    "A vintage travel poster that says VISIT MARS in bold letters",
    "Macro photo of a dew drop on a green leaf",
    "A medieval castle surrounded by autumn forest, aerial view",
    "A robot chef cooking pancakes in a modern kitchen",
]
from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.text_encoder import (  # noqa: E402
    Qwen3VLEmbedder,
    Qwen3VLEmbedderParams,
)

torch._dynamo.config.cache_size_limit = 128
pkg = Flux3Checkpoint(CKPT, "nvfp4")
cfg = pkg.config["text_encoder"]
emb = (
    Qwen3VLEmbedder(
        str(pkg.asset(cfg["path"])),
        Qwen3VLEmbedderParams(
            output_layer=cfg["layers"],
            max_length=cfg["max_tokens"],
            pad_multiple=cfg["pad_multiple"],
            torch_dtype=cfg["dtype"],
            use_compile=True,
        ),
        tokenizer_spec=str(pkg.asset(cfg["tokenizer"])),
    )
    .to("cuda")
    .eval()
)
mode = "graphs_off" if os.environ.get("TRTLLM_FLUX3_TEXT_GRAPH", "1") == "0" else "graphs_on"
print("mode:", mode, flush=True)
t0 = time.time()
if hasattr(emb, "warmup_buckets"):
    emb.warmup_buckets()
emb.encode(["warmup"])
for p in PROMPTS:
    emb.encode([p])
torch.cuda.synchronize()
print(f"warmup {time.time() - t0:.1f}s", flush=True)
spans = []
torch.cuda.nvtx.range_push("te_requests")
for i in range(24):
    p = PROMPTS[i % len(PROMPTS)]
    torch.cuda.synchronize()
    a = time.perf_counter()
    torch.cuda.nvtx.range_push("text_encode")
    ctx = emb.encode([p])[0]
    torch.cuda.synchronize()  # the pipeline consumes ctx right after; span ends when GPU is done
    torch.cuda.nvtx.range_pop()
    spans.append((time.perf_counter() - a) * 1e3)
    time.sleep(0.005)  # keep requests apart, as served
torch.cuda.nvtx.range_pop()
spans.sort()
med = spans[len(spans) // 2]
print(f"{mode}: text_encode span median {med:.3f} ms (min {spans[0]:.3f}, max {spans[-1]:.3f})",
      flush=True)
import json  # noqa: E402

print(json.dumps({"passed": True, "mode": mode, "text_encode_span_ms_median": round(med, 3)}))
