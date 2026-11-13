"""Untimed check of the MiniMax-H3 step gate for FLASHINFER per-tensor FP8 attention.

1. Config: engine.json's visual_gen_args validate as VisualGenArgs; step_precision reaches
   DiffusionPipelineConfig; invalid combinations are refused.
2. Gate: a FlashInferAttention (fp8/fp8 recipe) driven by a DenoiseStepController over a
   49-forward schedule at K=14 (and K=None). Every forward's output is compared with
   (a) the VANILLA backend's SDPA on the same inputs (must be bitwise equal on forwards < K
   and on calls outside the loop) and (b) the ungated FP8 path (bitwise equal on forwards >= K).
   Warmup forwards must run FP8. Repeated passes must be bitwise identical.
Shape: B=1, S (default 4096, small to stay quick), H=28, D=128; V a strided view (row stride 3*H*D).
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=28)
    ap.add_argument("--forwards", type=int, default=49)
    ap.add_argument("--k", type=int, default=14)
    args = ap.parse_args()

    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
    from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.step_precision import (
        ATTENTION_BF16_FORWARDS_STATE_KEY, STEP_PRECISION_STATE_KEY, DenoiseStepController)
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    failures = []
    # ---------------------------------------------------------------- 1. config
    eng = json.loads((Path(__file__).resolve().parents[1] / "ascent-config" / "engine.json").read_text())
    vg = VisualGenArgs(**eng["visual_gen_args"])
    log("engine.json visual_gen_args ->", vg.attention_config.backend,
        vg.attention_config.quant_attention_config, vg.step_precision)
    if vg.step_precision is None or vg.step_precision.attention_bf16_forwards != args.k:
        failures.append(f"engine.json step_precision {vg.step_precision} != K={args.k}")
    for bad in ({"step_precision": {"attention_bf16_forwards": 3}},
                {"attention_config": {"backend": "VANILLA"}, "step_precision": {"attention_bf16_forwards": 3}},
                {"attention_config": {"backend": "FLASHINFER"}, "step_precision": {"bogus": 1}}):
        try:
            VisualGenArgs(**bad)
            failures.append(f"config not refused: {bad}")
        except Exception as e:  # noqa: BLE001
            log("refused as expected:", bad, "->", type(e).__name__)
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
    for cls in (DiffusionModelConfig, DiffusionPipelineConfig):
        if "step_precision" not in cls.model_fields:
            failures.append(f"{cls.__name__} has no step_precision field")

    # ---------------------------------------------------------------- 2. gate
    S, H, D = args.seq, args.heads, 128
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, S, H, D, device="cuda", generator=g).bfloat16()
    k = torch.randn(1, S, H, D, device="cuda", generator=g).bfloat16()
    qkv = (torch.randn(1, S, 3 * H * D, device="cuda", generator=g) * 2.0).bfloat16()
    v = qkv[..., 2 * H * D:].unflatten(-1, (H, D))
    pristine = [t.clone() for t in (q, k, qkv)]

    qcfg = vg.attention_config.quant_attention_config
    from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention

    cudnn_ref = CuDNNAttention(layer_idx=0, num_heads=H, head_dim=D, dtype=torch.bfloat16)
    ref_cudnn = cudnn_ref.forward(q, k, v).clone()
    vanilla = VanillaAttention(layer_idx=0, num_heads=H, head_dim=D, dtype=torch.bfloat16)
    ref_fa2 = vanilla.forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    # Main's BF16 attention is the VANILLA backend (SDPA FA2): the gate's BF16 phase must match it.
    ref_bf16 = ref_fa2
    cudnn_vs_fa2 = ((ref_cudnn.float() - ref_fa2.float()).norm() / ref_fa2.float().norm()).item()
    log(f"cuDNN BF16 vs VANILLA FA2 rel-L2: {cudnn_vs_fa2:.3e}")

    results = {}
    for K in (args.k, None):
        controller = DenoiseStepController()
        state = {STEP_PRECISION_STATE_KEY: controller, ATTENTION_BF16_FORWARDS_STATE_KEY: K}
        attn = FlashInferAttention(layer_idx=0, num_heads=H, head_dim=D, dtype=torch.bfloat16,
                                   quant_attention_config=qcfg, attention_metadata_state=state)
        ref_fp8, _ = attn._run_fp8_sm120(q, k, v, is_causal=False, with_lse=False)
        ref_fp8 = ref_fp8.clone()
        paths = []
        passes = []
        for rep in range(2):
            outs = []
            # warmup forward (startup warmup): must be FP8
            controller.set_step(0, 1, warmup=True)
            w = attn.forward(q, k, v)
            if not torch.equal(w, ref_fp8):
                failures.append(f"K={K} warmup forward is not the FP8 output")
            controller.reset()
            # outside the loop (text refiner)
            o = attn.forward(q, k, v)
            want = ref_bf16 if K else ref_fp8
            if not torch.equal(o, want):
                failures.append(f"K={K} rep {rep} outside-loop call mismatched {'bf16' if K else 'fp8'}")
            for i in range(args.forwards):
                controller.set_step(i, args.forwards)
                o = attn.forward(q, k, v)
                exp_bf16 = bool(K) and i < K
                want = ref_bf16 if exp_bf16 else ref_fp8
                if not torch.equal(o, want):
                    failures.append(f"K={K} rep {rep} forward {i}: output != {'bf16' if exp_bf16 else 'fp8'} ref")
                if rep == 0:
                    paths.append("bf16" if exp_bf16 else "fp8")
                outs.append(o.clone())
                for t, p in zip((q, k, qkv), pristine):
                    if not torch.equal(t, p):
                        failures.append(f"K={K} input modified at forward {i}")
            controller.reset()
            passes.append(outs)
        for i, (a, b) in enumerate(zip(*passes)):
            if not torch.equal(a, b):
                failures.append(f"K={K} forward {i} differs between repeated passes")
        err = ((ref_fp8.float() - ref_bf16.float()).norm() / ref_bf16.float().norm()).item()
        results[str(K)] = {"bf16_forwards": paths.count("bf16"), "fp8_forwards": paths.count("fp8"),
                           "fp8_vs_bf16_rel_l2": err}
        log(f"K={K}: {results[str(K)]}")

    out = {"passed": not failures, "failures": failures[:20], "gate": results,
           "cudnn_bf16_vs_fa2_rel_l2": cudnn_vs_fa2,
           "compared": "every forward vs a standalone VanillaAttention (main's SDPA) call (bitwise, forwards < K and outside-loop) or the "
                       "ungated FP8 path (bitwise, forwards >= K and warmup); two repeated passes "
                       "bitwise; inputs vs pristine", "shape": [1, S, H, D]}
    print(json.dumps(out), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
