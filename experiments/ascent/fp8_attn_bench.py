"""Per-call time of the MiniMax-H3 denoiser self-attention at a gated (FP8) forward, one rank of TP2.

Shape: B=1, S=37966, H=28, D=128, BF16, non-causal, V a strided view (row stride 3*H*D) as in serving.
Arm selection follows the tree's engine.json (experiments/ascent-config/engine.json):
  - no file / no quant_attention_config (main): VanillaAttention, torch SDPA FA2 (what main runs);
  - FLASHINFER fp8/fp8 + step_precision (branch): FlashInferAttention.forward with the step controller
    at forward 20 (>= K), i.e. the per-tensor E4M3 quantize x3 + sm120_fmha_fp8_ragged_prefill path.
Timing: 5 warmup + 25 timed calls, CUDA events, median and (p90-p10)/median.
Correctness (untimed, same warmup/iteration count after timing): every iteration's output equals the
first (deterministic) and inputs stay bitwise unchanged; error vs an FP32 math reference (TF32 off) on
heads 0-1, all query rows, reported as rel-L2 (and the FA2 arm's own error for comparison).
Last stdout line: {"batch": {"1": {"kernel_ms", "noise_floor"}}, "output": {...}}.
"""

import json
import statistics
from pathlib import Path

import torch

S, H, D = 37966, 28, 128
WARMUP, ITERS = 5, 25


def log(*a):
    print(*a, flush=True)


def ref_fp32(q, k, v, heads=(0, 1), chunk=2048):
    torch.backends.cuda.matmul.allow_tf32 = False
    outs = []
    scale = D ** -0.5
    for h in heads:
        kh = k[0, :, h].float()
        vh = v[0, :, h].float()
        rows = []
        for s0 in range(0, S, chunk):
            qh = q[0, s0:s0 + chunk, h].float()
            p = torch.softmax((qh @ kh.T) * scale, dim=-1)
            rows.append(p @ vh)
        outs.append(torch.cat(rows))
    return torch.stack(outs, dim=1)  # [S, len(heads), D]


def main():
    import tensorrt_llm  # noqa: F401

    eng_path = Path(__file__).resolve().parents[1] / "ascent-config" / "engine.json"
    eng = json.loads(eng_path.read_text()) if eng_path.is_file() else {}
    vga = eng.get("visual_gen_args", {})
    qcfg_dict = (vga.get("attention_config") or {}).get("quant_attention_config")
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, S, H, D, device="cuda", generator=g).bfloat16()
    k = torch.randn(1, S, H, D, device="cuda", generator=g).bfloat16()
    qkv = torch.randn(1, S, 3 * H * D, device="cuda", generator=g).bfloat16()
    v = qkv[..., 2 * H * D:].unflatten(-1, (H, D))
    pristine = [t.clone() for t in (q, k, qkv)]

    if qcfg_dict:
        from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
        from tensorrt_llm._torch.visual_gen.models.minimax_h3.step_precision import (
            ATTENTION_BF16_FORWARDS_STATE_KEY, STEP_PRECISION_STATE_KEY, DenoiseStepController)
        from tensorrt_llm.visual_gen.args import VisualGenArgs

        vg = VisualGenArgs(**vga)
        K = vg.step_precision.attention_bf16_forwards
        ctl = DenoiseStepController()
        ctl.set_step(20, 49)
        state = {STEP_PRECISION_STATE_KEY: ctl, ATTENTION_BF16_FORWARDS_STATE_KEY: K}
        attn = FlashInferAttention(layer_idx=1, num_heads=H, head_dim=D, dtype=torch.bfloat16,
                                   quant_attention_config=vg.attention_config.quant_attention_config,
                                   attention_metadata_state=state)
        arm = f"FLASHINFER per-tensor fp8/fp8 (gated forward 20, K={K})"

        def call():
            return attn.forward(q, k, v)
    else:
        from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention

        attn = VanillaAttention(layer_idx=1, num_heads=H, head_dim=D, dtype=torch.bfloat16)
        arm = "VANILLA torch SDPA (FA2), main"

        def call():
            return attn.forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    log(f"arm: {arm}; shape B=1 S={S} H={H} D={D}")

    # ---- timing
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    times = []
    for i in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        call()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    med = statistics.median(times)
    p10, p90 = times[int(0.1 * (ITERS - 1))], times[int(0.9 * (ITERS - 1))]
    noise = (p90 - p10) / med
    log(f"timed {ITERS}: median {med:.3f} ms, p10 {p10:.3f}, p90 {p90:.3f}, noise {noise:.4f}")

    # ---- untimed correctness, same warmup/iteration count
    failures = []
    first = None
    for i in range(WARMUP + ITERS):
        o = call()
        if first is None:
            first = o.clone()
        elif not torch.equal(o, first):
            failures.append(f"iteration {i} differs from the first")
        for t, p in zip((q, k, qkv), pristine):
            if not torch.equal(t, p):
                failures.append(f"input modified at iteration {i}")
    log(f"repeat check: {len(failures)} failures")
    ref = ref_fp32(q, k, v)
    out2 = first[0, :, :2].float()
    rel = ((out2 - ref).norm() / ref.norm()).item()
    maxabs = (out2 - ref).abs().max().item()
    # The unmodified BF16 op's own error, for scale.
    from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention

    van = VanillaAttention(layer_idx=1, num_heads=H, head_dim=D, dtype=torch.bfloat16)
    fa2 = van.forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    fa2_rel = ((fa2[0, :, :2].float() - ref).norm() / ref.norm()).item()
    cls = "mode_switch" if qcfg_dict else "exact"
    out = {"batch": {"1": {"kernel_ms": med, "noise_floor": noise, "p10_ms": p10, "p90_ms": p90}},
           "output": {"class": cls, "rel_l2_vs_fp32_ref_heads01": rel, "max_abs_vs_fp32_ref": maxabs,
                      "fa2_bf16_rel_l2_vs_fp32_ref_heads01": fa2_rel},
           "arm": arm, "shape": [1, S, H, D], "passed": not failures, "failures": failures[:10]}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
