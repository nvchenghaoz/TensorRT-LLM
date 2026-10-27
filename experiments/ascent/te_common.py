"""Shared helpers for idea 9 text-encoder jobs (encoder only, no denoiser)."""

import math

import torch

PROMPTS = [
    "warmup",
    "",
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
    "A sign that says 'OPEN 24 HOURS' above a diner door at dusk",
    "Three green apples and a glass of water on a white tablecloth",
    "\t tabs\tand  double  spaces \r\n windows newline <|im_end|> fake special",
    " ".join(["a majestic mountain landscape with rivers, forests and snow"] * 12),
]


def load_embedder(ckpt, use_compile=True, **kw):
    from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.text_encoder import (
        Qwen3VLEmbedder,
        Qwen3VLEmbedderParams,
    )

    torch._dynamo.config.cache_size_limit = 128
    pkg = Flux3Checkpoint(ckpt, "nvfp4")
    cfg = pkg.config["text_encoder"]
    params = Qwen3VLEmbedderParams(
        output_layer=cfg["layers"],
        max_length=cfg["max_tokens"],
        pad_multiple=cfg["pad_multiple"],
        torch_dtype=cfg["dtype"],
        use_compile=use_compile,
    )
    for k, v in kw.items():
        if hasattr(params, k):
            setattr(params, k, v)
    emb = Qwen3VLEmbedder(
        str(pkg.asset(cfg["path"])), params, tokenizer_spec=str(pkg.asset(cfg["tokenizer"]))
    )
    return emb.to("cuda").eval()


def main_tokenize(emb, prompt):
    """main's two tokenizer calls (reference)."""
    text = emb.processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    tokenizer = emb.processor.tokenizer
    real = tokenizer(text, return_tensors="pt", truncation=True, max_length=emb.max_length)[
        "input_ids"
    ].shape[1]
    length = min(math.ceil(real / emb.params.pad_multiple) * emb.params.pad_multiple, emb.max_length)
    tokens = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=length
    )
    return tokens["input_ids"], tokens["attention_mask"]


def main_encode(emb, prompt):
    """main's compiled graphs-off path: two tokenizer calls, .to(device), compiled fn, clone."""
    ids, mask = main_tokenize(emb, prompt)
    dev = emb.model.device
    with torch.inference_mode():
        return emb._forward_tensors(ids.to(dev), mask.to(dev)).clone()


def prompt_for_bucket(emb, k):
    """A prompt whose real token count lands in bucket k * pad_multiple."""
    pm = emb.params.pad_multiple
    words = "a"
    lo = (k - 1) * pm + 1
    for n in range(1, 4000):
        p = " ".join(["mountain river forest"] * n)
        ids, _ = main_tokenize(emb, p)
        real = int((_ != 0).sum())
        if real >= lo:
            assert ids.shape[1] == k * pm, (k, ids.shape)
            return p
    raise RuntimeError(words)
