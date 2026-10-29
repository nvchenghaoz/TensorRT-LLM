"""Idea 227 STEP 0 diagnostics: per-phase, per-plan correctness of flux3_vae::subpixel_up_conv's
cuDNN FE graphs writing a strided phase view of a channels_last y. For every plan: fill y with a
sentinel, run the phase, then check (1) the phase's positions vs the fp32 reference of the same
sub-pixel math, (2) the other phases' positions still hold the sentinel; plus time per plan.
Variants: bias fp32 tensor (as the op), bias bf16 tensor, no bias."""

import json
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


r = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
rc = subprocess.run(["git", "merge-base", "--is-ancestor", "fb730d34", "HEAD"]).returncode
log("git", r.stdout.strip().replace("\n", " | "), "ancestor_rc", rc)

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import subpixel_conv as sp  # noqa: E402

cudnn = sp.cudnn
C, H = 256, int(sys.argv[1]) if len(sys.argv) > 1 else 512
torch.manual_seed(0)
up = iv.Flux3ImageUpsample(C).cuda()
with torch.no_grad():
    up.conv.weight.normal_(0, 0.02)
    up.conv.bias.normal_(0, 0.1)
up.refresh_subpixel()
up = up.to(torch.bfloat16).to(memory_format=torch.channels_last)
w4 = up.subpixel_weight.detach()
b4 = up.subpixel_bias.detach()
bias = up.conv.bias.detach()
x = torch.randn(1, C, H, H, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
with torch.inference_mode():
    o = F.conv2d(x.float().contiguous(), w4.float().contiguous(), b4.float(), padding=1)
    n, c, h, w = x.shape
    o = o.view(n, 2, 2, C, h + 1, w + 1)
    rows = [torch.stack([o[:, a, b, :, a:a + h, b:b + w] for b in range(2)], dim=-1) for a in range(2)]
    R = torch.stack(rows, dim=3).reshape(n, C, 2 * h, 2 * w).contiguous(memory_format=torch.channels_last)
    del o, rows
SENT = 1234.0


def build(a, b, bias_kind):
    """Same graph as sp._PhaseGraph but with a selectable bias dtype."""
    key = (1, C, C, H, H, a, b, bias_kind != "none")
    if bias_kind in ("none", "f32"):
        return sp.get_graph(key, x.device)
    # bf16 bias variant: patch the graph builder's bias dtype
    g0 = sp._PhaseGraph.__new__(sp._PhaseGraph)
    handle = sp._fp8c._get_handle(x.device)
    bf16, f32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    g = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=f32, compute_data_type=f32, handle=handle)
    t = {}
    t["X"] = g.tensor(name="X", dim=[1, C, H, H], stride=sp._fp8c._nhwc(C, H, H), data_type=bf16)
    t["W"] = g.tensor(name="W", dim=[C, C, 2, 2], stride=sp._fp8c._nhwc(C, 2, 2), data_type=bf16)
    (ph0, ph1), (pw0, pw1) = sp._pads(a), sp._pads(b)
    y = g.conv_fprop(image=t["X"], weight=t["W"], pre_padding=[ph0, pw0], post_padding=[ph1, pw1],
                     stride=[1, 1], dilation=[1, 1], compute_data_type=f32, name="conv")
    y.set_data_type(f32)
    t["B"] = g.tensor(name="bias", dim=[1, C, 1, 1], stride=[C, 1, C, C], data_type=bf16)
    y = g.add(a=y, b=t["B"], compute_data_type=f32, name="bias_add")
    y.set_output(True).set_data_type(bf16).set_dim([1, C, H, H]).set_stride([4 * H * H * C, 1, 4 * H * C, 2 * C])
    t["Y"] = y
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans(cudnn.build_plan_policy.ALL)
    g0.g, g0.t, g0.handle, g0.key = g, t, handle, key
    g0.names = [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]
    g0.ws = [int(g.get_workspace_size_plan_at_index(i)) for i in range(len(g0.names))]
    g0.plan = 0
    return g0


out = {"H": H}
with torch.inference_mode():
    for bias_kind in ("f32", "bf16", "none"):
        res = {}
        for a in range(2):
            for b in range(2):
                p = 2 * a + b
                try:
                    gr = build(a, b, bias_kind)
                except Exception as ex:  # noqa: BLE001
                    res[f"{a}{b}"] = f"REFUSED {ex!r}"[:1500]
                    log(bias_kind, a, b, res[f"{a}{b}"])
                    continue
                wp = w4[p * C:(p + 1) * C].contiguous(memory_format=torch.channels_last)
                rows = {}
                for i, nm in enumerate(gr.names):
                    if nm.startswith("eng8") or nm.startswith("eng0") or nm.startswith("eng71"):
                        continue  # slow reference engines
                    y = torch.full((1, C, 2 * H, 2 * H), SENT, dtype=torch.bfloat16, device="cuda").contiguous(
                        memory_format=torch.channels_last)
                    views = {(aa, bb): v for aa, bb, v in sp._phase_views(y)}
                    vp = {gr.t["X"]: x, gr.t["W"]: wp, gr.t["Y"]: views[(a, b)]}
                    if bias_kind == "f32":
                        vp[gr.t["B"]] = bias.float().view(1, C, 1, 1)
                    elif bias_kind == "bf16":
                        vp[gr.t["B"]] = bias.view(1, C, 1, 1)
                    try:
                        gr.run(vp, i)
                        torch.cuda.synchronize()
                    except Exception as ex:  # noqa: BLE001
                        rows[nm] = f"EXEC FAIL {ex!r}"[:300]
                        continue
                    rv = {(aa, bb): v for aa, bb, v in sp._phase_views(R)}[(a, b)]
                    mine = views[(a, b)].float()
                    ref = rv.float() if bias_kind != "none" else rv.float() - bias.float().view(1, C, 1, 1)
                    d = (mine - ref).abs()
                    others = all(bool((views[k] == SENT).all()) for k in views if k != (a, b))
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    for _ in range(2):
                        gr.run(vp, i)
                    s.record()
                    for _ in range(5):
                        gr.run(vp, i)
                    e.record()
                    torch.cuda.synchronize()
                    rows[nm] = {"ms": round(s.elapsed_time(e) / 5, 4), "nan": int(torch.isnan(mine).sum()),
                                "sentinel_left": int((views[(a, b)] == SENT).sum()),
                                "max_abs": round(float(torch.nan_to_num(d, nan=1e9).max()), 5),
                                "mean_abs": float(torch.nan_to_num(d, nan=0).mean()),
                                "others_untouched": others}
                res[f"{a}{b}"] = rows
                log(bias_kind, a, b, json.dumps(rows))
        out[bias_kind] = res
print(json.dumps(out))
