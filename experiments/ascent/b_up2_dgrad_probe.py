"""Idea 279 STEP 0: up[2] (U2, 512@256^2) and up[3] (U1, 512@128^2) upsample as one cuDNN FE
conv_dgrad (stride 2, pad 1, transpose_weight [C,C,4,4]) + fp32 bias epilogue (the graph of
subpixel_conv._UpGraph), every plan from heur A / B / FALLBACK, vs main's compiled aten call.
Real checkpoint weights; real x captured by pre-hooks on the eager bf16-trunk decoder, seed-0 latent.
Usage: python b_up2_dgrad_probe.py <checkpoint>"""

import json
import os
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import fp8_conv as fc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype  # noqa: E402

T0 = time.time()
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


r = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
rcs = {c: subprocess.run(["git", "merge-base", "--is-ancestor", c, "HEAD"]).returncode
       for c in ("da43d58f", "59981b4e")}
log("git", r.stdout.strip().replace("\n", " | "), "ancestor_rcs", rcs)
cudnn = fc.cudnn
CL = torch.channels_last
SENT = 1234.0
root = sys.argv[1]


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    return {"median": round(statistics.median(ts), 4), "p10": round(ts[int(0.1 * n)], 4),
            "p90": round(ts[min(n - 1, int(0.9 * n))], 4)}


def time_fn(fn, warm=10, iters=60):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return stats([s.elapsed_time(e) for s, e in ev])


def kernels_of(fn):
    fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    rows = []
    for ev in p.key_averages():
        if ev.device_type is not None and "CUDA" in str(ev.device_type) and ev.self_device_time_total > 0:
            rows.append([ev.key[:90], round(ev.self_device_time_total / 1000.0, 4)])
    return rows


with torch.inference_mode():
    vae, _ = gn_check.build_vae(root)
    set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    cap = {}
    hooks = []
    for lvl in (2, 3):
        def mk(lvl):
            def pre(mod, args):
                cap[lvl] = args[0].detach().to(torch.bfloat16).contiguous(memory_format=CL).clone()
            return pre
        hooks.append(vae.decoder.up[lvl].upsample.register_forward_pre_hook(mk(lvl)))
    g0 = torch.Generator(device="cuda").manual_seed(0)
    z = torch.randn(1, 128, 64, 64, device="cuda", generator=g0)
    vae.decode(z)
    for h in hooks:
        h.remove()
    log("captured", {k: tuple(v.shape) for k, v in cap.items()})

    out = {"git": r.stdout.strip(), "ancestor_rcs": rcs, "sites": {}}
    for lvl, site in ((2, "U2"), (3, "U1")):
        up = vae.decoder.up[lvl].upsample
        up = up.to(memory_format=CL)
        x0 = cap[lvl]
        x = x0.clone()
        n, C, h, w = x.shape
        wt = up.transpose_weight.detach().to(torch.bfloat16).contiguous(memory_format=CL)
        bias = up.conv.bias.detach().to(torch.bfloat16).contiguous()
        cout = wt.shape[1]
        R = F.conv_transpose2d(x.float(), wt.float(), bias.float(), stride=2, padding=1)
        res = {"x": [n, C, h, w], "plans": {}}
        # --- main's compiled call as the decoder runs it
        up.use_transpose = True
        if lvl == 2:
            fmain = torch.compile(lambda t: up.forward_nobias(t), dynamic=False)
        else:
            fmain = torch.compile(lambda t: up.forward(t), dynamic=False)
        fmain(x)
        ym = fmain(x)
        if lvl == 2:
            ym_b = ym.float() + bias.float().view(1, -1, 1, 1)
        else:
            ym_b = ym.float()
        res["main"] = {"mode": "forward_nobias" if lvl == 2 else "forward(dgrad+bias)",
                       "max_abs": float((ym_b - R).abs().max()), "mean_abs": float((ym_b - R).abs().mean()),
                       "out_stride": list(ym.stride())}
        assert torch.equal(x, x0)
        res["main"]["time"] = time_fn(lambda: fmain(x))
        res["main"]["kernels"] = kernels_of(lambda: fmain(x))
        log(site, "main", json.dumps(res["main"]))
        # aten eager no-bias for scale
        res["aten_eager_nobias"] = time_fn(lambda: F.conv_transpose2d(x, wt, None, stride=2, padding=1))
        # --- FE plans per heuristic mode
        handle = fc._get_handle(x.device)
        bf16, f32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
        for mode_name in ("A", "B", "FALLBACK"):
            try:
                g = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=f32, compute_data_type=f32,
                                  handle=handle)
                X = g.tensor(name="X", dim=[n, C, h, w], stride=fc._nhwc(C, h, w), data_type=bf16)
                W = g.tensor(name="W", dim=[C, cout, 4, 4], stride=fc._nhwc(cout, 4, 4), data_type=bf16)
                y = g.conv_dgrad(loss=X, filter=W, padding=[1, 1], stride=[2, 2], dilation=[1, 1],
                                 compute_data_type=f32, name="dgrad")
                y.set_dim([n, cout, 2 * h, 2 * w]).set_stride(fc._nhwc(cout, 2 * h, 2 * w))
                y.set_data_type(f32)
                B = g.tensor(name="bias", dim=[1, cout, 1, 1], stride=[cout, 1, cout, cout], data_type=bf16)
                Y = g.add(a=y, b=B, compute_data_type=f32, name="bias_add")
                Y.set_output(True).set_data_type(bf16)
                g.validate()
                g.build_operation_graph()
                g.create_execution_plans([getattr(cudnn.heur_mode, mode_name)])
                g.check_support()
                g.build_plans(cudnn.build_plan_policy.ALL)
                names = [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]
                wss = [int(g.get_workspace_size_plan_at_index(i)) for i in range(len(names))]
            except Exception as ex:  # noqa: BLE001
                res["plans"][mode_name] = f"REFUSED {ex!r}"[:600]
                log(site, mode_name, res["plans"][mode_name])
                continue
            log(site, mode_name, "plans", names)
            for i, nm in enumerate(names):
                key = f"{mode_name}:{nm}"
                if any(k.split(":", 1)[1] == nm for k in res["plans"] if ":" in k):
                    res["plans"][key] = "dup"
                    continue
                yb = torch.full((n, cout, 2 * h, 2 * w), SENT, dtype=torch.bfloat16, device="cuda").contiguous(
                    memory_format=CL)
                vp = {X: x, W: wt, B: bias.view(1, cout, 1, 1), Y: yb}
                ws = torch.empty(max(wss[i], 1), dtype=torch.uint8, device="cuda")

                def run(vp=vp, ws=ws, i=i, g=g):
                    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
                    g.execute_plan_at_index(vp, ws, i, handle=handle)
                try:
                    run()
                    torch.cuda.synchronize()
                except Exception as ex:  # noqa: BLE001
                    res["plans"][key] = f"EXEC FAIL {ex!r}"[:300]
                    continue
                d = (yb.float() - R).abs()
                row = {"workspace": wss[i], "sentinel_left": int((yb == SENT).sum()),
                       "max_abs": float(torch.nan_to_num(d, nan=1e9).max()),
                       "mean_abs": float(torch.nan_to_num(d, nan=1e9).mean())}
                if row["max_abs"] < 1.0:
                    assert torch.equal(x, x0)
                    row["time"] = time_fn(run, warm=10, iters=60 if not nm.startswith("eng0") else 12)
                    # correctness pass: restore pristine x, sentinel, 5 repeats bitwise
                    x.copy_(x0)
                    yb.fill_(SENT)
                    run()
                    torch.cuda.synchronize()
                    first = yb.clone()
                    rep = True
                    for _ in range(5):
                        yb.fill_(SENT)
                        run()
                        torch.cuda.synchronize()
                        rep = rep and torch.equal(first, yb)
                    row["repeat5_bitwise"] = bool(rep)
                    row["inputs_unchanged"] = bool(torch.equal(x, x0))
                res["plans"][key] = row
                log(site, key, json.dumps(row))
                del yb, ws
            del g
        # verdict
        mt = res["main"]["time"]
        best = None
        for k, v in res["plans"].items():
            if isinstance(v, dict) and "time" in v and v["sentinel_left"] == 0 and v.get("repeat5_bitwise") \
                    and v["max_abs"] <= max(2 * res["main"]["max_abs"], 0.05):
                if best is None or v["time"]["median"] < res["plans"][best]["time"]["median"]:
                    best = k
        if best is not None:
            bt = res["plans"][best]["time"]
            margin = (mt["p90"] - mt["p10"]) + (bt["p90"] - bt["p10"])
            res["best"] = best
            res["gain_ms"] = round(mt["median"] - bt["median"], 4)
            res["needed_ms"] = round(margin, 4)
            res["GO"] = bool(res["gain_ms"] > margin)
        else:
            res["GO"] = False
        log(site, "VERDICT", best, res.get("gain_ms"), res.get("needed_ms"), res["GO"])
        out["sites"][site] = res
        del R, ym, ym_b
        torch.cuda.empty_cache()
out["passed"] = True
print(json.dumps(out))
