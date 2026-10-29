"""Idea 297 (finish branch 198) correctness on the synced tree: up[2]/up[3] upsample on
flux3_vae::subpixel_up_conv ON vs OFF (OFF = use_fused_bias_op False at up[2], up[3] = main's aten
dgrad path + bias fold / bias pass). Seeds 0-7, real checkpoint VAE, one process.

Order: the same-process bf16 control (OFF vs OFF + 1e-5 latent perturbation, code of diag
b198_site_diag.py @0b6b9f6d) runs FIRST and is REPORT-ONLY (written to $AUTO_DEV_ARTIFACTS/b1_control.json
before any ON arm runs; its T_u8 is printed, never gated).
Declared gates (passed holds only these):
- B1 (frozen from control 2233 only, top lead 7107): bf16 decoder, both sites ON vs OFF, seeds 0-7:
  max u8_max <= 6 AND min PSNR >= 65.0 dB.
- B2: served FP8 vs fp32 eager R: mean_abs(mean over seeds) <= 1.05x OFF's; PSNR min >= OFF's - 0.3 dB.
- A: fp32 eager decode shas (seeds 0-7) and the fp32 compiled Dynamo FX graph sha equal ON vs OFF.
- per op: at U1 (up[3]) and U2 (up[2]), max_abs of the op's bf16 output vs fp64 conv_transpose2d + bias
  <= main's path's bf16 output (dgrad -> bf16, + bias -> bf16); unrounded main reported, not gated.
- repeat (FP8 ON): 10 warm + 30 decodes bitwise vs iteration 0, latent unchanged, Dynamo counters and
  cuDNN FE graph counts unchanged after warmup.
- census (FP8 ON): 28 fp8_conv3x3 calls/decode, 1 Dynamo frame, 0 graph breaks, no strided_dgrad
  kernels. _scaled_mm count reported (FP8-PV path marker), LPIPS / u8 reported.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import fp8_conv as fc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import subpixel_conv as spc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype  # noqa: E402

T0 = time.time()
root = sys.argv[1]
ART = os.environ.get("AUTO_DEV_ARTIFACTS", tempfile.mkdtemp())
os.makedirs(ART, exist_ok=True)
torch._dynamo.config.cache_size_limit = 64
SEEDS = list(range(8))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


r = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
rcs = {c: subprocess.run(["git", "merge-base", "--is-ancestor", c, "HEAD"]).returncode
       for c in ("59981b4e", "bae40207")}
iv_src = open(os.path.join("tensorrt_llm", "_torch", "visual_gen", "models", "flux3", "flux3_core",
                           "image_vae", "image_vae.py")).read()
git_info = {"log": r.stdout.strip(), "ancestor_rcs": rcs, "UP_FE_DGRAD_EXTRA": spc.UP_FE_DGRAD_EXTRA,
            "env_FLUX3_UP_FE_DGRAD_EXTRA": os.environ.get("FLUX3_UP_FE_DGRAD_EXTRA"),
            "image_vae_scaled_mm_mentions": iv_src.count("_scaled_mm")}
log("git", json.dumps(git_info))

prep_calls = []
_orig_prep = fc.prepare_fp8_convs


def _counting_prep(dec):
    n = _orig_prep(dec)
    prep_calls.append(n)
    return n


fc.prepare_fp8_convs = _counting_prep


def lat(s, eps=0.0):
    g = torch.Generator(device="cuda").manual_seed(s)
    z = torch.randn(1, 128, 64, 64, device="cuda", generator=g)
    if eps:
        g2 = torch.Generator(device="cuda").manual_seed(1000 + s)
        z = z + eps * torch.randn(1, 128, 64, 64, device="cuda", generator=g2)
    return z


# Plans as committed in the branch's subpixel_conv.PLAN_NAME (no-op when the tree already has them).
spc.PLAN_NAME.setdefault((512, 512, 256), "eng1_k24=1")
spc.PLAN_NAME.setdefault((512, 512, 128), "eng1_k24=4")
log("PLAN_NAME", {str(k): v for k, v in spc.PLAN_NAME.items()})


def set_levels(vae, levels):
    # ON: op path; OFF: main's aten conv_transpose2d path. Same meaning in the branch tree
    # (fused_op_path checked first) and in main's tree (transpose_path checked first).
    for lvl in (2, 3):
        vae.decoder.up[lvl].upsample.use_fused_bias_op = lvl in levels
        vae.decoder.up[lvl].upsample.use_transpose = lvl not in levels
    assert vae.decoder.up[1].upsample.use_fused_bias_op and not vae.decoder.up[1].upsample.use_transpose


def build(bf16, kw, levels, backend=None):
    vae, _ = gn_check.build_vae(root)
    if bf16:
        set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    set_levels(vae, levels)
    if kw is not None:
        if backend is not None:
            dec = vae.decoder
            orig = dec.compile

            def compile_with(*a, **k):
                k.pop("options", None)
                return orig(*a, backend=backend, **k)

            dec.compile = compile_with
        vae.compile_decoder(**kw)
    return vae


_LP = {}


def lpips_fn():
    if "m" not in _LP:
        try:
            import lpips
            try:
                m = lpips.LPIPS(net="alex", version="0.1", verbose=False)
            except TypeError:
                import lpips.pretrained_networks as pn
                import torchvision.models as tvm
                original = tvm.alexnet

                def alexnet(pretrained=False, **kwargs):
                    if pretrained:
                        kwargs.setdefault("weights", tvm.AlexNet_Weights.IMAGENET1K_V1)
                    return original(**kwargs)

                pn.tv.alexnet = alexnet
                m = lpips.LPIPS(net="alex", version="0.1", verbose=False)
            _LP["m"] = m.to("cuda").eval()
        except Exception as e:  # noqa: BLE001
            log("lpips unavailable", repr(e)[:200])
            _LP["m"] = None
    return _LP["m"]


def cmp(a, b, lp=False):
    a, b = a.float(), b.float()
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2).mean().item()
    du = (gn_check.to_u8(a).int() - gn_check.to_u8(b).int()).abs()
    row = {"psnr": round(10 * torch.log10(torch.tensor(4.0 / mse)).item(), 3) if mse > 0 else 999.0,
           "mean_abs": round((a - b).abs().mean().item(), 7), "max_abs": round((a - b).abs().max().item(), 4),
           "u8_max": du.max().item(), "u8_gt2": int((du > 2).sum())}
    if lp and lpips_fn() is not None:
        with torch.no_grad():
            row["lpips"] = round(float(lpips_fn()(a.clamp(-1, 1), b.clamp(-1, 1)).item()), 6)
    return row


def agg(rows):
    out = {"psnr_min": min(r["psnr"] for r in rows),
           "mean_abs_mean": round(sum(r["mean_abs"] for r in rows) / len(rows), 7),
           "u8_max": max(r["u8_max"] for r in rows), "u8_gt2_total": sum(r["u8_gt2"] for r in rows)}
    if all("lpips" in r for r in rows):
        out["lpips_max"] = max(r["lpips"] for r in rows)
    return out


def counters():
    import torch._dynamo.utils as du
    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break", "stats")}


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def strip(code):
    return "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("#"))


def census(vae, z):
    from torch.profiler import ProfilerActivity, profile
    vae.decode(z)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        vae.decode(z)
        torch.cuda.synchronize()
    tp = os.path.join(tempfile.mkdtemp(), "trace.json")
    prof.export_chrome_trace(tp)
    tr = json.load(open(tp))
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern = [e for e in evs if e.get("cat") == "kernel"]
    opc = Counter(e.get("name") for e in evs if e.get("cat") == "cpu_op")
    kc, kd = Counter(), Counter()
    for k in kern:
        kc[k["name"]] += 1
        kd[k["name"]] += k.get("dur", 0) / 1e3
    up_rows = {nm[:110]: [kc[nm], round(kd[nm], 4)] for nm in kc
               if any(s in nm for s in ("dgrad", "Dgrad", "strided", "nhwcToNchw", "nchwToNhwc"))}
    return {"kernels": len(kern), "kernel_sum_ms": round(sum(kd.values()), 4),
            "fp8_conv3x3_calls": opc.get("flux3_vae::fp8_conv3x3", 0),
            "subpixel_up_conv_calls": opc.get("flux3_vae::subpixel_up_conv", 0),
            "scaled_mm_calls": opc.get("aten::_scaled_mm", 0),
            "scaled_mm_named_kernels": sum(v for k, v in kc.items() if "scaled_mm" in k),
            "attn_fp8_pv_flags": [bool(m._fp8_pv) for m in vae.decoder.modules()
                                  if type(m).__name__ == "Flux3ImageAttnBlock"],
            "strided_dgrad_kernels": sum(v for k, v in kc.items() if "strided_dgrad" in k),
            "upsample_related_kernels": up_rows}


out = {"git": git_info}
gates = {}
with torch.inference_mode():
    # ------------------------------------------------ B1 control FIRST (bf16 OFF vs OFF + 1e-5)
    v = build(True, {}, ())
    O = {"OFF": {s: v.decode(lat(s)).clone() for s in SEEDS}}
    ctrl_rows = [cmp(v.decode(lat(s, 1e-5)), O["OFF"][s]) for s in SEEDS]
    T_u8 = max(rw["u8_max"] for rw in ctrl_rows)
    out["B1_control"] = {"rows": ctrl_rows, "agg": agg(ctrl_rows), "T_u8": T_u8}
    with open(os.path.join(ART, "b1_control.json"), "w") as f:
        json.dump(out["B1_control"], f)
    log("B1 CONTROL (report-only; OFF vs OFF+1e-5, before any ON arm)", json.dumps(out["B1_control"]["agg"]),
        "T_u8", T_u8, "| frozen B1 thresholds: u8_max <= 6, PSNR min >= 65.0 dB (control 2233: u8 6, 65.3 dB)",
        "written", os.path.join(ART, "b1_control.json"))
    del v
    torch.cuda.empty_cache()
    # ------------------------------------------------ B1 ON arms (per site + both)
    for name, levels in (("U1_only", (3,)), ("U2_only", (2,)), ("both", (2, 3))):
        v = build(True, {}, levels)
        rows = [cmp(v.decode(lat(s)), O["OFF"][s]) for s in SEEDS]
        out[f"B1_{name}_vs_OFF"] = {"rows": rows, "agg": agg(rows)}
        log("B1", name, json.dumps(agg(rows)))
        del v
        torch.cuda.empty_cache()
    del O
    b1 = out["B1_both_vs_OFF"]["agg"]
    # Frozen from control 2233 only (area lead / top lead 7107); this job's control is report-only.
    gates["B1_u8_max_le_6"] = b1["u8_max"] <= 6
    gates["B1_psnr_min_ge_65.0"] = b1["psnr_min"] >= 65.0

    # ------------------------------------------------ fp32 eager R + gate A
    ref = build(False, None, (2, 3))
    R = {s: ref.decode(lat(s)).clone() for s in SEEDS}
    sh_on = [sha(R[s]) for s in SEEDS]
    set_levels(ref, ())
    sh_off = [sha(ref.decode(lat(s))) for s in SEEDS]
    del ref
    torch.cuda.empty_cache()
    fx = {}
    for name, levels in (("ON", (2, 3)), ("OFF", ())):
        graphs = []

        def cap(gm, ex, graphs=graphs):
            graphs.append(gm.code)
            return gm.forward

        v = build(False, {}, levels, backend=cap)
        v.decode(lat(0))
        torch.cuda.synchronize()
        fx[name] = {"n_graphs": len(graphs),
                    "sha": hashlib.sha256("\n#==\n".join(strip(g) for g in graphs).encode()).hexdigest()[:16]}
        del v
        torch.cuda.empty_cache()
    out["A"] = {"eager_fp32_sha_ON": sh_on, "eager_fp32_sha_OFF": sh_off, "fx_fp32": fx}
    gates["A_fp32_eager_shas_equal"] = sh_on == sh_off
    gates["A_fp32_fx_sha_equal"] = fx["ON"]["sha"] == fx["OFF"]["sha"] and fx["ON"]["n_graphs"] == fx["OFF"]["n_graphs"]
    log("A", json.dumps(out["A"]))

    # ------------------------------------------------ served FP8 ON / OFF vs R (B2), repeat, census
    import torch._dynamo.utils as _du
    for name, levels in (("fp8_ON", (2, 3)), ("fp8_OFF", ())):
        _du.counters.clear()
        v = build(True, {"fp8_convs": True}, levels)
        n_prep = prep_calls[-1] if prep_calls else 0
        rows = [cmp(v.decode(lat(s)), R[s], lp=True) for s in SEEDS]
        out[name] = {"rows": rows, "agg": agg(rows), "n_prepared": n_prep}
        log(name, "vs R", json.dumps(agg(rows)), "prepared", n_prep)
        if name == "fp8_ON":
            z = lat(0)
            z0 = z.clone()
            for _ in range(10):
                v.decode(z)
            torch.cuda.synchronize()
            c0, g0 = counters(), (len(fc._graphs), len(spc._graphs))
            first, rep, inp = None, True, True
            for _ in range(30):
                inp &= bool(torch.equal(z, z0))
                y = v.decode(z)
                if first is None:
                    first = y.clone()
                elif not torch.equal(y, first):
                    rep = False
            torch.cuda.synchronize()
            c1, g1 = counters(), (len(fc._graphs), len(spc._graphs))
            out["repeat"] = {"bitwise": rep, "inputs_unchanged": inp, "no_recompile": c1 == c0,
                             "no_new_fe_graphs": g1 == g0, "counters": c1, "fe_graphs": list(g1),
                             "subpixel_graphs": spc.graph_table()}
            log("repeat", json.dumps(out["repeat"])[:2000])
        cen = census(v, lat(0))
        cnt = counters()
        cen["frames_ok"] = cnt.get("frames", {}).get("ok")
        cen["graph_breaks"] = sum(cnt.get("graph_break", {}).values()) if cnt.get("graph_break") else 0
        out[name]["census"] = cen
        log(name, "census", json.dumps(cen)[:3000])
        del v
        torch.cuda.empty_cache()
    on, off = out["fp8_ON"]["agg"], out["fp8_OFF"]["agg"]
    gates["B2_mean_abs_le_1.05x_off"] = on["mean_abs_mean"] <= 1.05 * off["mean_abs_mean"]
    gates["B2_psnr_min_ge_off_minus_0.3"] = on["psnr_min"] >= off["psnr_min"] - 0.3
    rp = out["repeat"]
    gates.update({"repeat_bitwise": rp["bitwise"], "inputs_unchanged": rp["inputs_unchanged"],
                  "no_recompile_after_warmup": rp["no_recompile"],
                  "no_cudnn_fe_graph_build_after_warmup": rp["no_new_fe_graphs"]})
    ce = out["fp8_ON"]["census"]
    gates["n_prepared_28"] = out["fp8_ON"]["n_prepared"] == 28
    gates["fp8_conv3x3_calls_28"] = ce["fp8_conv3x3_calls"] == 28
    gates["frames_1_breaks_0"] = ce["frames_ok"] == 1 and ce["graph_breaks"] == 0
    gates["strided_dgrad_gone"] = ce["strided_dgrad_kernels"] == 0
    del R
    torch.cuda.empty_cache()

    # ------------------------------------------------ per op vs fp64 at U1 (up[3]) / U2 (up[2])
    vae = build(True, None, ())
    capd = {}
    hooks = []
    for lvl in (2, 3):
        def mk(lvl):
            def pre(mod, args):
                capd[lvl] = args[0].detach().to(torch.bfloat16).contiguous(memory_format=torch.channels_last).clone()
            return pre
        hooks.append(vae.decoder.up[lvl].upsample.register_forward_pre_hook(mk(lvl)))
    vae.decode(lat(0))
    for h in hooks:
        h.remove()
    out["per_op"] = {}
    for lvl, site in ((3, "U1"), (2, "U2")):
        up = vae.decoder.up[lvl].upsample
        x = capd[lvl]
        wt = up.transpose_weight
        b = up.conv.bias
        R64 = F.conv_transpose2d(x.double(), wt.double(), b.double(), stride=2, padding=1)
        y_op = torch.ops.flux3_vae.subpixel_up_conv(x, wt, b)
        d_nb = F.conv_transpose2d(x, wt.to(x.dtype), None, stride=2, padding=1)
        main_r = (d_nb + b.to(x.dtype).view(1, -1, 1, 1))
        main_u = d_nb.double() + b.double().view(1, -1, 1, 1)
        row = {"x": list(x.shape), "op_max_abs": float((y_op.double() - R64).abs().max()),
               "op_mean_abs": float((y_op.double() - R64).abs().mean()),
               "main_bf16_max_abs": float((main_r.double() - R64).abs().max()),
               "main_bf16_mean_abs": float((main_r.double() - R64).abs().mean()),
               "main_unrounded_max_abs": float((main_u - R64).abs().max())}
        out["per_op"][site] = row
        gates[f"per_op_{site}_max_abs_le_main"] = row["op_max_abs"] <= row["main_bf16_max_abs"]
        log("per_op", site, json.dumps(row))
    del vae
    torch.cuda.empty_cache()

out["gates"] = gates
out["report_only"] = {"fp8_ON_scaled_mm_calls": out["fp8_ON"]["census"]["scaled_mm_calls"],
                      "fp8_ON_scaled_mm_named_kernels": out["fp8_ON"]["census"]["scaled_mm_named_kernels"],
                      "fp8_ON_attn_fp8_pv_flags": out["fp8_ON"]["census"]["attn_fp8_pv_flags"],
                      "fp8_OFF_census": {k: out["fp8_OFF"]["census"][k] for k in
                                         ("kernels", "kernel_sum_ms", "strided_dgrad_kernels", "subpixel_up_conv_calls")},
                      "fp8_ON_census": {k: ce[k] for k in
                                        ("kernels", "kernel_sum_ms", "strided_dgrad_kernels", "subpixel_up_conv_calls")}}
log("gates", json.dumps(gates), "report", json.dumps(out["report_only"]))
out["passed"] = bool(all(gates.values()))
out["wall_s"] = round(time.time() - T0, 1)
print(json.dumps(out))
