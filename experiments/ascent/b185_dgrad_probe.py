"""Idea 227 STEP 0b: up[1] upsample as ONE cuDNN FE conv_dgrad (stride-2 4x4 transposed conv,
the same weights as Flux3ImageUpsample.transpose_weight) with an fp32 bias epilogue, writing the
interleaved channels_last y directly. Per plan: correctness vs the fp32 sub-pixel reference
(sentinel-filled output) and time. Also: aten conv_transpose2d (bias-free) and main's compiled
fprop+interleave for scale. Variants: bias f32, bias bf16, no bias."""

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

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import fp8_conv as fc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv  # noqa: E402

cudnn = fc.cudnn
C, H = 256, 512
torch.manual_seed(0)
up = iv.Flux3ImageUpsample(C).cuda()
with torch.no_grad():
    up.conv.weight.normal_(0, 0.02)
    up.conv.bias.normal_(0, 0.1)
up.refresh_subpixel()
up = up.to(torch.bfloat16).to(memory_format=torch.channels_last)
up.use_transpose = False
w4 = up.subpixel_weight.detach()
b4 = up.subpixel_bias.detach()
bias = up.conv.bias.detach()
wt = up.transpose_weight.detach()
log("wt", tuple(wt.shape), wt.stride())
x = torch.randn(1, C, H, H, device="cuda").to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
with torch.inference_mode():
    o = F.conv2d(x.float().contiguous(), w4.float().contiguous(), b4.float(), padding=1)
    n, c, h, w = x.shape
    o = o.view(n, 2, 2, C, h + 1, w + 1)
    rows = [torch.stack([o[:, a, b, :, a:a + h, b:b + w] for b in range(2)], dim=-1) for a in range(2)]
    R = torch.stack(rows, dim=3).reshape(n, C, 2 * h, 2 * w).contiguous(memory_format=torch.channels_last)
    del o, rows
SENT = 1234.0


def tm(fn, warm=3, iters=10):
    for _ in range(warm):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return round(s.elapsed_time(e) / iters, 4)


out = {}
with torch.inference_mode():
    # aten references
    out["aten_dgrad_nobias_ms"] = tm(lambda: F.conv_transpose2d(x, wt, None, stride=2, padding=1))
    ya = F.conv_transpose2d(x, wt, None, stride=2, padding=1)
    out["aten_dgrad_out_stride"] = list(ya.stride())
    out["aten_dgrad_nobias_err"] = float((ya.float() + bias.float().view(1, -1, 1, 1) - R).abs().max())
    out["aten_dgrad_plus_bias_eager_ms"] = tm(lambda: F.conv_transpose2d(x, wt, None, stride=2, padding=1)
                                              + bias.view(1, -1, 1, 1))
    log(json.dumps(out))
    for bias_kind in ("f32", "bf16", "none"):
        res = {}
        try:
            handle = fc._get_handle(x.device)
            bf16, f32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
            g = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=f32, compute_data_type=f32, handle=handle)
            DY = g.tensor(name="DY", dim=[1, C, H, H], stride=fc._nhwc(C, H, H), data_type=bf16)
            # dgrad filter [K = dy channels, C = dx channels, R, S]; conv_transpose2d weight is [Cin, Cout, 4, 4]
            Wt = g.tensor(name="W", dim=[C, C, 4, 4], stride=fc._nhwc(C, 4, 4), data_type=bf16)
            dx = g.conv_dgrad(loss=DY, filter=Wt, padding=[1, 1], stride=[2, 2], dilation=[1, 1],
                              compute_data_type=f32, name="dgrad")
            dx.set_dim([1, C, 2 * H, 2 * H]).set_stride(fc._nhwc(C, 2 * H, 2 * H))
            B = None
            if bias_kind != "none":
                dx.set_data_type(f32)
                B = g.tensor(name="bias", dim=[1, C, 1, 1], stride=[C, 1, C, C],
                             data_type=f32 if bias_kind == "f32" else bf16)
                y = g.add(a=dx, b=B, compute_data_type=f32, name="bias_add")
                y.set_output(True).set_data_type(bf16)
            else:
                y = dx
                y.set_output(True).set_data_type(bf16)
            g.validate()
            g.build_operation_graph()
            modes = [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
            g.create_execution_plans(modes)
            g.check_support()
            g.build_plans(cudnn.build_plan_policy.ALL)
            names = [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]
            wss = [int(g.get_workspace_size_plan_at_index(i)) for i in range(len(names))]
            log(bias_kind, "plans", names)
        except Exception as ex:  # noqa: BLE001
            out[bias_kind] = f"REFUSED {ex!r}"[:3000]
            log(bias_kind, out[bias_kind])
            continue
        bt = None if B is None else (bias.float() if bias_kind == "f32" else bias).view(1, C, 1, 1)
        for i, nm in enumerate(names):
            yb = torch.full((1, C, 2 * H, 2 * H), SENT, dtype=torch.bfloat16, device="cuda").contiguous(
                memory_format=torch.channels_last)
            vp = {DY: x, Wt: wt, y: yb}
            if B is not None:
                vp[B] = bt
            ws = torch.empty(max(wss[i], 1), dtype=torch.uint8, device="cuda")

            def run():
                cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
                g.execute_plan_at_index(vp, ws, i, handle=handle)
            try:
                run()
                torch.cuda.synchronize()
            except Exception as ex:  # noqa: BLE001
                res[nm] = f"EXEC FAIL {ex!r}"[:300]
                continue
            ref = R if B is not None else R - bias.float().view(1, -1, 1, 1)
            d = (yb.float() - ref).abs()
            row = {"sentinel_left": int((yb == SENT).sum()), "nan": int(torch.isnan(yb).sum()),
                   "max_abs": float(torch.nan_to_num(d, nan=1e9).max()), "mean_abs": float(torch.nan_to_num(d).mean()),
                   "workspace": wss[i]}
            row["ms"] = tm(run) if row["max_abs"] < 1 and not nm.startswith("eng0") else None
            r1 = yb.clone()
            run()
            torch.cuda.synchronize()
            row["repeat_bitwise"] = bool(torch.equal(r1, yb))
            res[nm] = row
            log(bias_kind, nm, json.dumps(row))
        out[bias_kind] = res
print(json.dumps(out))
