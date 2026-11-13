"""Real MiniMaxH3Transformer3DModel (checkpoint config, few layers, random weights, 1 GPU, TP1) with
per-block torch.compile exactly as pipeline.torch_compile does it.

  check: forwards at several timesteps with the deferred MLP residual (this branch's default) and
         with it off (main's block path), outputs compared bitwise; Dynamo frame/graph-break counters
         before and after the timed steps (no per-step recompiles); kernel launch counts per forward
         from torch.profiler.
  trace: warmup, then forwards inside an NVTX range "denoise" for nsys.
Last stdout line: JSON.
"""

import json
import os
import sys
from collections import Counter
from types import SimpleNamespace

import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import build_packed_sequence, build_row_timesteps
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "check"
L = int(os.environ.get("H3_LAYERS", "4"))
STEPS = [0.05, 0.3, 0.55, 0.8, 0.95]


def build():
    with open(os.path.join(CKPT, "transformer", "config.json")) as f:
        cfg = json.load(f)
    cfg["num_layers"] = L
    if "num_refiner_layers" in cfg:
        cfg["num_refiner_layers"] = 1
    mc = DiffusionModelConfig(pretrained_config=SimpleNamespace(**cfg))
    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(mc)
    model.to("cuda")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 and "norm" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p, dtype=torch.float32).to(p.dtype))
            elif p.ndim == 1:
                p.copy_(0.02 * torch.randn_like(p, dtype=torch.float32).to(p.dtype))
            else:
                p.copy_((torch.randn_like(p, dtype=torch.float32) / p.shape[-1] ** 0.5).to(p.dtype))
    model.eval()
    # pipeline.torch_compile: each block of every ModuleList compiled on its own.
    model.transformer_blocks = nn.ModuleList(
        [torch.compile(b, mode="default", dynamic=None, fullgraph=False) for b in model.transformer_blocks]
    )
    return model, cfg


def inputs(model, cfg):
    patch = tuple(cfg["patch_size"])
    text_len = 77
    layout = build_packed_sequence(torch.zeros(text_len, dtype=torch.long), 4, 32, 48, 25, patch)
    dev = "cuda"
    vid_dim = cfg["in_channels"] * patch[0] * patch[1] * patch[2]
    n_vid = layout.video_indices.numel()
    n_aud = layout.audio_indices.numel()
    g = torch.Generator(device=dev).manual_seed(1)
    common = dict(
        hidden_states=torch.randn(1, n_vid, vid_dim, generator=g, device=dev),
        audio_hidden_states=torch.randn(1, n_aud, cfg["audio_in_channels"], generator=g, device=dev),
        token_tags=layout.token_tags.to(dev),
        video_indices=layout.video_indices.to(dev),
        audio_indices=layout.audio_indices.to(dev),
        text_indices=layout.text_indices.to(dev),
    )
    enc = torch.randn(1, text_len, cfg["text_dim"], generator=g, device=dev).to(torch.bfloat16)
    with torch.inference_mode():
        static = model.prepare_static_context(enc, layout.position_ids.to(dev))
    plans = []
    for t in STEPS:
        uniq, tidx = build_row_timesteps(layout, t, min(1.0, t + 0.1), t, 1.0)
        plans.append((uniq.to(dev), tidx.to(dev), torch.tensor([1.0 - t], device=dev)))
    return layout, common, static, plans


def fwd(model, common, static, plan):
    uniq, tidx, att_t = plan
    return model(
        encoder_hidden_states=None, timestep=att_t, conditioning_timesteps=uniq, timestep_indices=tidx,
        position_ids=None, return_dict=False, static_context=static, **common,
    )


def main():
    from torch._dynamo.utils import counters

    model, cfg = build()
    layout, common, static, plans = inputs(model, cfg)
    print(f"layers={L} M={layout.sequence_length} D={cfg['hidden_size']} defer={model._defer_mlp_residual}", flush=True)
    out = {"layers": L, "M": layout.sequence_length}
    with torch.inference_mode():
        if MODE == "trace":
            for p in plans[:2]:
                fwd(model, common, static, p)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push("denoise")
            for p in plans[2:]:
                fwd(model, common, static, p)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            out["traced_forwards"] = len(plans) - 2
            print(json.dumps(out))
            return
        # warmup (first forward compiles), then the timed-equivalent steps
        new_first = fwd(model, common, static, plans[0])
        torch.cuda.synchronize()
        c0 = {k: dict(v) for k, v in counters.items() if k in ("stats", "graph_break", "recompiles")}
        new = [new_first] + [fwd(model, common, static, p) for p in plans[1:]]
        torch.cuda.synchronize()
        c1 = {k: dict(v) for k, v in counters.items() if k in ("stats", "graph_break", "recompiles")}
        out["dynamo_after_first_forward"] = c0
        out["dynamo_after_all_forwards"] = c1
        out["no_recompile_after_first_forward"] = c0 == c1
        # kernel counts of one deferred forward
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fwd(model, common, static, plans[2])
            torch.cuda.synchronize()
        cnt = Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                cnt[e.name] += 1
        out["kernels_per_forward"] = {
            "poi_add_index_select_mul": sum(v for k, v in cnt.items() if "triton_poi_fused_add_index_select_mul" in k),
            "adaln": sum(v for k, v in cnt.items() if "_minimax_h3_adaln_kernel" in k),
            "gate_res_tail": sum(v for k, v in cnt.items() if "_minimax_h3_gate_res_kernel" in k),
            "mod_out_tail": sum(v for k, v in cnt.items() if "_minimax_h3_mod_out_kernel" in k),
            "qk_norm_rope": sum(v for k, v in cnt.items() if "qk_norm_rope" in k),
        }
        # main's path: same model, residual applied inside each block (compiled), eager norm_out
        model._defer_mlp_residual = False
        ref = [fwd(model, common, static, p) for p in plans]
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fwd(model, common, static, plans[2])
            torch.cuda.synchronize()
        cnt2 = Counter(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
        out["main_path_kernels_per_forward"] = {
            "poi_add_index_select_mul": sum(v for k, v in cnt2.items() if "triton_poi_fused_add_index_select_mul" in k),
            "adaln": sum(v for k, v in cnt2.items() if "_minimax_h3_adaln_kernel" in k),
        }
        mism = []
        for i, (a, b) in enumerate(zip(new, ref)):
            for j in range(2):
                mism.append(int((a[j] != b[j]).sum()))
        out["bitwise_mismatch_per_output"] = mism
        out["finite"] = all(bool(torch.isfinite(a[j]).all()) for a in new for j in range(2))
    k = out["kernels_per_forward"]
    out["passed"] = bool(
        all(m == 0 for m in mism) and out["no_recompile_after_first_forward"] and out["finite"]
        and k["poi_add_index_select_mul"] == 0 and k["adaln"] == 2 * L
        and out["main_path_kernels_per_forward"]["adaln"] == 2 * L
    )
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
