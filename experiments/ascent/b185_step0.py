"""Idea 227 STEP 0: up[1] upsample (x [1,256,512,512] bf16 NHWC -> y [1,256,1024,1024]).
flux3_vae::subpixel_up_conv (one cuDNN FE stride-2 dgrad + fp32 bias epilogue) vs main's compiled
sub-pixel fprop + interleave. Synthetic tensors, no model load.

Arms (CUDA events per call, 10 warmup + 60 timed, median/p10/p90; M and P interleaved):
  M  : Flux3ImageUpsample.forward with use_fused_bias_op=False (main's path), torch.compile'd
       like the decoder (layout_optimization False, emulate_precision_casts True)
  Pc : the same module with use_fused_bias_op=True, torch.compile'd (opaque custom op)
  P  : the op called eagerly
Plus a plan sweep (every FE plan timed) and accuracy vs an fp32 reference of the same sub-pixel
math (fp32 conv of the bf16 x with the bf16 W4 + b4).
"""

import json
import subprocess
import time

import torch
import torch.nn.functional as F

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


r = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
rc = subprocess.run(["git", "merge-base", "--is-ancestor", "fb730d34", "HEAD"]).returncode
log("git", r.stdout.strip().replace("\n", " | "), "ancestor_fb730d34_rc", rc)

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import subpixel_conv as sp  # noqa: E402

WARM, ITERS = 10, 60
C, H = 256, 512
torch.manual_seed(0)
up = iv.Flux3ImageUpsample(C).cuda()
with torch.no_grad():
    up.conv.weight.normal_(0, 0.02)
    up.conv.bias.normal_(0, 0.1)
up.refresh_subpixel()
up = up.to(torch.bfloat16).to(memory_format=torch.channels_last).eval()
up.use_transpose = False
x = torch.randn(1, C, H, H, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
x0 = x.clone()
opts = {"layout_optimization": False, "emulate_precision_casts": True}


class Wrap(torch.nn.Module):
    def __init__(self, flag):
        super().__init__()
        self.flag = flag

    def forward(self, t):
        up.use_fused_bias_op = self.flag
        return up(t)


def mk(flag):
    m = Wrap(flag)
    m.compile(dynamic=False, options=opts)
    return m


M, Pc = mk(False), mk(True)


def P(t):
    return torch.ops.flux3_vae.subpixel_up_conv(t, up.transpose_weight, up.conv.bias)


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": round(med, 4), "p10_ms": round(p10, 4), "p90_ms": round(p90, 4),
            "noise_floor": round((p90 - p10) / med, 4)}


def one(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn(x)
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


res = {"git": {"log": r.stdout.strip(), "ancestor_fb730d34_rc": rc}}
with torch.inference_mode():
    o = F.conv2d(x.float().contiguous(), up.subpixel_weight.float().contiguous(), up.subpixel_bias.float(), padding=1)
    n, c, h, w = x.shape
    o = o.view(n, 2, 2, C, h + 1, w + 1)
    rows = [torch.stack([o[:, a, b, :, a:a + h, b:b + w] for b in range(2)], dim=-1) for a in range(2)]
    R = torch.stack(rows, dim=3).reshape(n, C, 2 * h, 2 * w)
    del o, rows
    outs = {}
    for name, fn in (("M", M), ("Pc", Pc), ("P", P)):
        outs[name] = fn(x).clone()
        log(name, tuple(outs[name].shape), outs[name].stride())
    res["graphs"] = sp.graph_table()
    log("graphs", json.dumps(res["graphs"]))

    def err(a, ref):
        d = (a.float() - ref).abs()
        return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "nan": int(torch.isnan(a).sum())}

    acc = {k + "_vs_R": err(v, R) for k, v in outs.items()}
    acc["P_vs_M"] = err(outs["P"], outs["M"].float())
    acc["Pc_bitwise_P"] = bool(torch.equal(outs["P"], outs["Pc"]))
    res["accuracy"] = acc
    log("accuracy", json.dumps(acc))

    # plan sweep (writes only into a scratch output)
    gr = sp._graphs[(1, C, C, H, H)]
    ys = torch.empty_like(outs["P"])
    vp = {gr.t["X"]: x, gr.t["W"]: up.transpose_weight, gr.t["B"]: up.conv.bias.view(1, C, 1, 1), gr.t["Y"]: ys}
    res["plan_sweep_ms"] = gr.autotune(vp, warm=3, iters=10)
    log("sweep", json.dumps(res["plan_sweep_ms"]))

    # timing: M, Pc, P interleaved
    for fn in (M, Pc, P):
        for _ in range(WARM):
            fn(x)
    torch.cuda.synchronize()
    ts = {"M": [], "Pc": [], "P": []}
    for i in range(ITERS):
        for k, fn in ((("M", M), ("Pc", Pc), ("P", P)) if i % 2 == 0 else (("P", P), ("Pc", Pc), ("M", M))):
            ts[k].append(one(fn))
    res["timing"] = {k: stats(v) for k, v in ts.items()}
    log("timing", json.dumps(res["timing"]))

    # untimed repeat sequence (same warmup + iteration count), bitwise vs iteration 0
    rep = {}
    for name, fn in (("Pc", Pc), ("M", M)):
        for _ in range(WARM):
            fn(x)
        first, ok, inp = None, True, True
        for _ in range(ITERS):
            inp &= bool(torch.equal(x, x0))
            y = fn(x)
            if first is None:
                first = y.clone()
            else:
                ok &= bool(torch.equal(y, first))
        rep[name] = {"repeat_bitwise": ok, "inputs_unchanged": inp, "bitwise_vs_first_call": bool(torch.equal(first, outs[name]))}
    res["repeat"] = rep
    log("repeat", json.dumps(rep))

pc, m = res["timing"]["Pc"], res["timing"]["M"]
res["delta_Pc_minus_M_ms"] = round(pc["median_ms"] - m["median_ms"], 4)
res["passed"] = bool(all(v["repeat_bitwise"] and v["inputs_unchanged"] for v in rep.values())
                     and acc["Pc_vs_R"]["nan"] == 0 and acc["Pc_vs_R"]["max_abs"] <= acc["M_vs_R"]["max_abs"])
res["batch"] = {"1": {"kernel_ms": pc["median_ms"], "noise_floor": max(pc["noise_floor"], m["noise_floor"])}}
res["output"] = {"class": "reordered", "error_vs_fp32_R": acc["Pc_vs_R"], "main_error_vs_fp32_R": acc["M_vs_R"]}
res["wall_s"] = round(time.time() - T0, 1)
print(json.dumps(res))
