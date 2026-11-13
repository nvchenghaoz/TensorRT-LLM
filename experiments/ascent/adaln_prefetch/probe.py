"""Stage 0 (idea 105): is the opaque adaln_table op bitwise equal to main's compiled prologue?

1 GPU, synthetic. W [96768, 2688] = randn*0.02 bf16 + bias, inside the real
MiniMaxH3AdaLayerNormModulation Linear. temb (FP32, as the model feeds it) = randn and randn*5 at
M = 1 and 2 rows (row_timestep_plan's unique timesteps).

  main : torch.compile(default) of  mod = adaln_proj.table(temb); h3_norm_mod(x, w, mod, idx)
  new  : torch.compile(default) of  x_act = silu(temb).to(bf16);
                                    mod = trtllm::minimax_h3_adaln_table(temb, x_act, W, b, slot)
                                    h3_norm_mod(x, w, mod, idx)
  new is run as a miss (empty slot) and as a hit (slot filled eagerly from the x_act the compiled
  code returned, as block i's drain does), each repeated; mod and the norm output are compared
  with torch.equal against main's. Kernel names of one call of each are listed (torch.profiler).
Last stdout line: {"passed": ..., ...}
"""

import json
import os
import sys

import torch
import torch.nn.functional as F


def log(*a):
    print(*a, flush=True)


def main():
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import adaln_prefetch as ap
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_adaln import h3_norm_mod

    torch.cuda.set_device(0)
    D, T = 5376, 2688
    vgm = VisualGenMapping(1, 0, tp_size=1)
    mc = DiffusionModelConfig(pretrained_config=None, mapping=vgm.to_llm_mapping(), visual_gen_mapping=vgm)
    mod_layer = h3.MiniMaxH3AdaLayerNormModulation(T, D, mc).cuda()
    lin = mod_layer.linear
    g = torch.Generator(device="cuda")
    g.manual_seed(105)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(lin.weight.shape, device="cuda", generator=g) * 0.02)
        lin.bias.copy_(torch.randn(lin.bias.shape, device="cuda", generator=g) * 0.02)
    log(f"W {tuple(lin.weight.shape)} {lin.weight.dtype}, bias {tuple(lin.bias.shape)} {lin.bias.dtype}")
    norm_w = (1.0 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    rows = 4096
    x = torch.randn(1, rows, D, device="cuda", dtype=torch.bfloat16, generator=g)
    slot = ap.H3AdalnTableSlot(None)

    def f_main(temb, x, idx):
        mod = mod_layer.table(temb)
        return mod, h3_norm_mod(x, norm_w, mod, idx, 1e-6, 1, 0)

    def f_new(temb, x, idx):
        x_act = F.silu(temb).to(lin.dtype)
        mod = ap.h3_adaln_table(temb, x_act, lin.weight, lin.bias, slot.slot_id).view(-1, 6 * D)
        return mod, h3_norm_mod(x, norm_w, mod, idx, 1e-6, 1, 0), x_act

    c_main = torch.compile(f_main, mode="default", dynamic=None)
    c_new = torch.compile(f_new, mode="default", dynamic=None)
    results = {}
    ok = True
    kernels = {}
    for m in (1, 2):
        for scale in (1.0, 5.0):
            case = f"M{m}_x{scale:g}"
            gg = torch.Generator(device="cuda")
            gg.manual_seed(1000 * m + int(scale))
            temb = torch.randn(m, T, device="cuda", generator=gg) * scale
            idx = torch.randint(0, 3 * m, (rows,), device="cuda", generator=gg)
            pristine = temb.clone()
            with torch.inference_mode():
                ref_mod, ref_y = c_main(temb, x, idx)
                eager_mod = mod_layer.table(temb)
                r = {"eager_table_vs_compiled_main": bool(torch.equal(eager_mod, ref_mod))}
                miss_eq, hit_eq, n_hit, n_miss = [], [], 0, 0
                for it in range(6):
                    h0, m0 = slot.hits, slot.misses
                    mod, y, x_act = c_new(temb, x, idx)
                    miss_eq.append(bool(torch.equal(mod, ref_mod) and torch.equal(y, ref_y)))
                    n_miss += slot.misses - m0
                    # block i's drain: fill the next consumer (same weight here) from x_act
                    slot.fill(1, lin, temb, x_act)
                    mod2, y2, _ = c_new(temb, x, idx)
                    hit_eq.append(bool(torch.equal(mod2, ref_mod) and torch.equal(y2, ref_y)))
                    n_hit += slot.hits - h0
                    if not torch.equal(temb, pristine):
                        r["temb_changed"] = True
                torch.cuda.synchronize()
                d = (mod.float() - ref_mod.float()).abs()
                r.update(
                    miss_bitwise=all(miss_eq),
                    hit_bitwise=all(hit_eq),
                    hits=n_hit,
                    misses=n_miss,
                    mismatch_count=int((mod != ref_mod).sum().item()),
                    max_abs=float(d.max().item()),
                )
            results[case] = r
            ok = ok and r["miss_bitwise"] and r["hit_bitwise"] and n_hit == 6 and not r.get("temb_changed")
            log(f"{case}: {r}")
            if scale == 1.0:
                from torch.profiler import ProfilerActivity, profile

                for name, fn in (("main", c_main), ("new", c_new)):
                    with torch.inference_mode(), profile(activities=[ProfilerActivity.CUDA]) as prof:
                        fn(temb, x, idx)
                        torch.cuda.synchronize()
                    kernels[f"{name}_M{m}"] = [
                        e.name for e in prof.events() if str(e.device_type).endswith("CUDA")
                    ]
                    log(f"kernels {name} M{m}: {kernels[f'{name}_M{m}']}")
    out = {"passed": ok, "cases": results, "kernels": kernels,
           "torch": torch.__version__}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    sys.exit(main())
