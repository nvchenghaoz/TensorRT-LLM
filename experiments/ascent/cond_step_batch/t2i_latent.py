"""Idea 87 switch-0 proof for t2i: denoised t2i latent hashes at 2048^2, one process per tree.

usage: t2i_latent.py <checkpoint>

Loads the pipeline in-process (PipelineLoader; FLASHINFER FP8 attention, nvfp4), hooks
_decode_latents to keep the denoised latent, and runs 3 t2i prompts (seeds 100-102) at
2048x2048, 16 steps, each twice (repeats must be bitwise). In the candidate tree it repeats
the set with TRTLLM_FLUX3_COND_STEP_BATCH=0 in the same process. Prints sha256 of every
latent; run in both trees of a run_gpu_pair: t2i is unchanged when the base tree's hashes equal
the candidate's (the t2i denoised latent is cross-process bitwise on main, fact 600; decoded
t2i images are not, so check.py image hashes cannot show this).
"""

import hashlib
import json
import os
import sys
import time
import types

PROMPTS = [
    ("p0", "A red fox standing in fresh snow at sunrise, photorealistic", 100),
    ("p1", "A cozy reading nook with a green velvet armchair and a sleeping tabby cat", 101),
    ("p2", "A street sign that reads 'MAPLE AVE' next to a red mailbox", 102),
]


def log(msg):
    print(f"[t2i_latent {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import torch

    from tensorrt_llm import VisualGenArgs
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig
    from tensorrt_llm.visual_gen.params import VisualGenParams

    ckpt = sys.argv[1]
    cfg = VisualGenArgs()
    att = AttentionConfig(
        backend="FLASHINFER",
        quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
    )
    cfg = cfg.model_copy(update={"attention_config": att})
    cfg.pipeline_config["variant"] = "nvfp4"
    t0 = time.time()
    pipe = PipelineLoader(cfg, device="cuda:0").load(checkpoint_dir=ckpt)
    log(f"loaded in {time.time() - t0:.0f}s")
    from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf

    has_switch = hasattr(tf, "_cond_step_batch_enabled")
    state = {}
    orig = pipe._decode_latents

    def decode(latent_result):
        state["latent"] = latent_result[0].detach().clone()
        return orig(latent_result)

    pipe._decode_latents = decode

    def gen(prompt, seed):
        params = VisualGenParams(height=2048, width=2048, seed=seed, num_inference_steps=16)
        out = pipe.infer(types.SimpleNamespace(prompt=prompt, params=params))
        torch.cuda.synchronize()
        lat = state["latent"].contiguous().cpu()
        return hashlib.sha256(lat.view(torch.uint8).numpy().tobytes()).hexdigest(), out

    modes = [("default", None)] + ([("switch0", "0")] if has_switch else [])
    hashes, ok = {}, True
    for mode, val in modes:
        if val is None:
            os.environ.pop("TRTLLM_FLUX3_COND_STEP_BATCH", None)
        else:
            os.environ["TRTLLM_FLUX3_COND_STEP_BATCH"] = val
        for rep in range(2):
            for key, prompt, seed in PROMPTS:
                t1 = time.time()
                h, _ = gen(prompt, seed)
                log(f"{mode} rep{rep} {key} seed {seed}: latent sha256 {h[:16]} ({time.time() - t1:.1f}s)")
                prev = hashes.setdefault(mode, {}).get(key)
                if prev is not None and prev != h:
                    ok = False
                    log(f"  MISMATCH vs earlier {mode} {key}")
                hashes[mode][key] = h
    if "switch0" in hashes and hashes["switch0"] != hashes["default"]:
        ok = False
    print(json.dumps({
        "passed": ok,
        "has_cond_step_batch_switch": has_switch,
        "latent_sha256": hashes,
        "compared": "t2i 2048^2 denoised latents: repeats bitwise in-process; candidate default == "
                    "switch 0 in-process; compare latent_sha256.default across the pair's two trees",
    }), flush=True)


if __name__ == "__main__":
    main()
