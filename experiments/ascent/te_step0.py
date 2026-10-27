"""Step 0 for idea 9: baseline facts of the FLUX.3 Qwen3-VL text encoder (no denoiser load)."""

import os

os.environ.setdefault("TORCH_LOGS", "graph_breaks,recompiles")
import inspect
import json
import sys
import time

import torch

CKPT = sys.argv[1]
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
os.makedirs(ART, exist_ok=True)

import transformers
from transformers.models.qwen3_vl import modeling_qwen3_vl as mq

print("transformers", transformers.__version__, mq.__file__, flush=True)
with open(os.path.join(ART, "modeling_qwen3_vl.py"), "w") as f:
    f.write(inspect.getsource(mq))
try:
    from transformers import masking_utils

    with open(os.path.join(ART, "masking_utils.py"), "w") as f:
        f.write(inspect.getsource(masking_utils))
except Exception as e:  # noqa
    print("no masking_utils", e)

from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.text_encoder import (
    Qwen3VLEmbedder,
    Qwen3VLEmbedderParams,
)

pkg = Flux3Checkpoint(CKPT, "nvfp4")
cfg = pkg.config["text_encoder"]
print("te config", cfg, flush=True)
torch._dynamo.config.cache_size_limit = 128
emb = Qwen3VLEmbedder(
    str(pkg.asset(cfg["path"])),
    Qwen3VLEmbedderParams(
        output_layer=cfg["layers"],
        max_length=cfg["max_tokens"],
        pad_multiple=cfg["pad_multiple"],
        torch_dtype=cfg["dtype"],
        use_compile=True,
    ),
    tokenizer_spec=str(pkg.asset(cfg["tokenizer"])),
).to("cuda").eval()
print("model config", emb.model.config.text_config, flush=True)

prompts = {
    "warmup": "warmup",
    "empty": "",
    "short": "A cat sitting on a red sofa.",
    "quote": 'A sign that says "OPEN 24/7" with \'neon\' letters\nand a newline.',
    "nonascii": "Café au lait, 東京の夜景, émojis 🎨🚀, Ünïcödé.",
    "long": " ".join(["a majestic mountain landscape with rivers, forests and snow"] * 12),
}
tok = emb.processor.tokenizer
info = {}
for k, p in prompts.items():
    text = emb.processor.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
    )
    real = tok(text, return_tensors="pt", truncation=True, max_length=emb.max_length)["input_ids"].shape[1]
    info[k] = real
print("real lengths", info, "pad_token_id", tok.pad_token_id, "padding_side", tok.padding_side, flush=True)
print("chat text for warmup:", repr(emb.processor.apply_chat_template(
    [{"role": "user", "content": "warmup"}], tokenize=False, add_generation_prompt=True)), flush=True)


def make(T, real_prompt="warmup"):
    text = emb.processor.apply_chat_template(
        [{"role": "user", "content": real_prompt}], tokenize=False, add_generation_prompt=True
    )
    t = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=T)
    return t["input_ids"].cuda(), t["attention_mask"].cuda()


eager = lambda ids, m: type(emb)._forward_tensors(emb, ids, m)  # noqa

# Sync debug on eager path
ids, m = make(80)
with torch.inference_mode():
    eager(ids, m)
    torch.cuda.synchronize()
    print("=== sync debug eager T=80 ===", flush=True)
    torch.cuda.set_sync_debug_mode("warn")
    eager(ids, m)
    torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()

# compile time per bucket
for T in (80, 160):
    ids, m = make(T)
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.time()
        out = emb._forward_tensors(ids, m)
        torch.cuda.synchronize()
        print(f"compile+first T={T}: {time.time()-t0:.1f}s out {tuple(out.shape)} {out.dtype}", flush=True)
        t0 = time.time()
        for _ in range(10):
            out = emb._forward_tensors(ids, m)
        torch.cuda.synchronize()
        print(f"steady T={T}: {(time.time()-t0)*100:.2f} ms/call", flush=True)
        e = eager(ids, m)
        print(f"T={T} compiled vs eager max abs {(out.float()-e.float()).abs().max().item():.4g} equal={torch.equal(out, e)}", flush=True)

print("=== sync debug compiled T=80 ===", flush=True)
ids, m = make(80)
with torch.inference_mode():
    torch.cuda.set_sync_debug_mode("warn")
    emb._forward_tensors(ids, m)
    torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()

# kernel count compiled T=80
from torch.profiler import ProfilerActivity, profile

with torch.inference_mode(), profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    emb._forward_tensors(ids, m)
    torch.cuda.synchronize()
evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
print("cuda kernel events compiled T=80:", len(evs), flush=True)
cpu_names = {}
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CPU and ("Memcpy" in e.name or "item" in e.name or "_local_scalar" in e.name or "nonzero" in e.name or "synchron" in e.name.lower()):
        cpu_names[e.name] = cpu_names.get(e.name, 0) + 1
print("cpu sync-like ops:", cpu_names, flush=True)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)

from torch._dynamo.utils import counters

print("dynamo counters:", {k: dict(v) for k, v in counters.items()}, flush=True)
print("rope_deltas attr:", getattr(emb.model.model, "rope_deltas", "n/a"), flush=True)
print(json.dumps({"passed": True, "real_lengths": info}))
