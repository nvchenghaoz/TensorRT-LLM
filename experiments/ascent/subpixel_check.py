"""Checks + census for the sub-pixel Flux3ImageUpsample (vae_decode idea 27, branch 29).

usage: python experiments/ascent/subpixel_check.py <checkpoint_root> <mode>
modes:
  unit     module test at 512@128^2, 512@256^2, 256@512^2 with the checkpoint's real
           upsample conv weights, fp32 (TF32 convs, like-for-like) and bf16 (weights
           cast from the fp32 masters). Declared tolerance (before the run): new vs
           original max abs <= 2x the original's own max abs vs an fp64 reference.
           Also: repeated calls bitwise, input unchanged, compiled module == eager
           module within the same tolerance, per-shape compiled module timing.
  decoder  full decoder, fp32 and bf16: candidate compiled decoder vs the original
           Upsample (same tree, instance-patched forward) compiled decoder, both vs
           an fp64 eager decode with the original Upsample. Tolerance: u8 max diff
           <= 2 vs original same dtype; candidate mean_abs vs fp64 <= 1.25x the
           original's (+1e-6). One Dynamo frame, no graph breaks, no recompiles across
           warmup + repeated decodes + 2 consecutive "requests" (seed 1 then seed 0
           again, bitwise == first seed-0 output); state_dict keys unchanged.
  census   torch.profiler kernel census of one warm compiled decode, original vs
           candidate arm (fp32), plus per-level compiled Upsample module kernels.
Last stdout line is a JSON object.
"""

import json
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

iv = gn_check.iv
ROOT, MODE = sys.argv[1], sys.argv[2]
OUT = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
LEVELS = [(3, 512, 128), (2, 512, 256), (1, 256, 512)]  # (up index, C, H)


def log(*a):
    print(*a, flush=True)


def orig_forward(self, x):
    x = F.interpolate(x, scale_factor=2.0, mode="nearest")
    return self.conv(x)


def make_orig_mod(m):
    m.forward = types.MethodType(orig_forward, m)
    return m


def make_orig(vae):
    for m in vae.decoder.modules():
        if isinstance(m, iv.Flux3ImageUpsample):
            make_orig_mod(m)
    return vae


def build(dtype=None, original=False):
    vae, _ = gn_check.build_vae(ROOT)
    if dtype is not None:
        vae = vae.to(dtype)
    if original:
        make_orig(vae)
    return vae


def timeit(fn, warm=5, iters=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def compile_mod(m, dtype):
    opts = {"layout_optimization": False}
    if dtype != torch.float32:
        opts["emulate_precision_casts"] = True
    return torch.compile(m, dynamic=False, options=opts)


@torch.no_grad()
def unit():
    import torch._dynamo

    vae32 = build()
    recs, ok_all = [], True
    for dtype in (torch.float32, torch.bfloat16):
        for lvl, C, H in LEVELS:
            torch._dynamo.reset()
            up32 = vae32.decoder.up[lvl].upsample
            new = iv.Flux3ImageUpsample(C)
            new.load_state_dict(up32.state_dict())  # fires the hook on fp32 masters
            new = new.cuda().to(dtype)
            orig = iv.Flux3ImageUpsample(C)
            orig.load_state_dict(up32.state_dict())
            orig = make_orig_mod(orig.cuda().to(dtype))
            g = torch.Generator(device="cuda").manual_seed(100 + C + H)
            x = torch.randn(1, C, H, H, device="cuda", generator=g).to(dtype)
            x0 = x.clone()
            yo = orig(x)
            yn = new(x)
            yn2 = new(x)
            # fp64 reference: original math, fp32 master weights, the same (rounded) input
            r64 = F.conv2d(F.interpolate(x.double(), scale_factor=2.0, mode="nearest"),
                           up32.conv.weight.double(), up32.conv.bias.double(), padding=1)
            d_no = (yn.double() - yo.double()).abs()
            e_o = (yo.double() - r64).abs()
            e_n = (yn.double() - r64).abs()
            cn = compile_mod(new, dtype)
            yc = cn(x)
            d_c = (yc.double() - yn.double()).abs().max().item()
            co = compile_mod(orig, dtype)
            t_orig = timeit(lambda: co(x))
            t_new = timeit(lambda: cn(x))
            rec = dict(dtype=str(dtype).split(".")[-1], C=C, H=H, level=lvl,
                       new_vs_orig_max_abs=d_no.max().item(), new_vs_orig_mean_abs=d_no.mean().item(),
                       orig_vs_fp64_max=e_o.max().item(), new_vs_fp64_max=e_n.max().item(),
                       orig_vs_fp64_mean=e_o.mean().item(), new_vs_fp64_mean=e_n.mean().item(),
                       compiled_vs_eager_new_max=d_c, repeat_bitwise=torch.equal(yn, yn2),
                       inputs_unchanged=torch.equal(x, x0), state_dict_keys=sorted(new.state_dict()),
                       compiled_ms_orig=t_orig, compiled_ms_new=t_new)
            tol = 2 * rec["orig_vs_fp64_max"]
            ok = (rec["new_vs_orig_max_abs"] <= tol and d_c <= tol and rec["repeat_bitwise"]
                  and rec["inputs_unchanged"] and rec["state_dict_keys"] == ["conv.bias", "conv.weight"])
            rec["ok"] = bool(ok)
            ok_all &= bool(ok)
            log(json.dumps(rec))
            recs.append(rec)
            del new, orig, cn, co, x, x0, yo, yn, yn2, yc, r64, d_no, e_o, e_n
            torch.cuda.empty_cache()
    return {"passed": bool(ok_all), "check": "sub-pixel Upsample vs original interpolate+conv3x3, "
            "real weights, 3 shapes x fp32/bf16; tol new-vs-orig max abs <= 2x orig-vs-fp64 max abs",
            "cases": recs}


def counters():
    import torch._dynamo.utils as du

    return {k: dict(v) for k, v in du.counters.items()
            if k in ("frames", "recompiles", "graph_break", "stats")}


def latent(seed, dtype):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g).to(dtype)


@torch.no_grad()
def ref64(seed):
    vae = build(torch.float64, original=True)
    out = vae.decode(latent(seed, torch.float32).double())
    del vae
    torch.cuda.empty_cache()
    return out


def stat(a, b):
    d = (a.double() - b.double()).abs()
    du = (gn_check.to_u8(a.float()).int() - gn_check.to_u8(b.float()).int()).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "u8_max_diff": du.max().item(),
            "u8_frac_diff": (du > 0).float().mean().item()}


@torch.no_grad()
def decoder():
    import torch._dynamo

    keys_ref = sorted(iv.Flux3ImageVAE(iv.Flux3ImageVAEConfig()).state_dict())
    r64 = ref64(0)
    log("fp64 reference done")
    res, ok_all = {}, True
    for dtype in (torch.float32, torch.bfloat16):
        outs = {}
        for arm in ("orig", "new"):
            torch._dynamo.reset()
            vae = build(dtype, original=(arm == "orig"))
            keys_ok = sorted(vae.state_dict()) == keys_ref
            vae.compile_decoder()
            z0, z1 = latent(0, dtype), latent(1, dtype)
            zc = z0.clone()
            c0 = counters()
            first = vae.decode(z0).clone()
            for _ in range(4):
                vae.decode(z0)
            c_warm = counters()
            rep = all(torch.equal(vae.decode(z0), first) for _ in range(6))
            # 2 consecutive requests: a different latent, then the first again
            vae.decode(z1)
            again = torch.equal(vae.decode(z0), first)
            torch.cuda.synchronize()
            c_end = counters()
            outs[arm] = first
            rec = {"arm": arm, "dtype": str(dtype), "state_dict_keys_unchanged": keys_ok,
                   "repeat_bitwise": rep, "second_request_bitwise": again,
                   "inputs_unchanged": torch.equal(z0, zc),
                   "counters_after_warmup": {k: c_warm.get(k, {}) for k in ("frames", "graph_break", "recompiles")},
                   "counters_before": {k: c0.get(k, {}) for k in ("frames", "graph_break", "recompiles")},
                   "no_new_frames_after_warmup": c_end == c_warm,
                   "vs_fp64": stat(first, r64)}
            res[f"{arm}_{str(dtype).split('.')[-1]}"] = rec
            log(json.dumps(rec))
            del vae
            torch.cuda.empty_cache()
        d = stat(outs["new"], outs["orig"])
        tag = str(dtype).split(".")[-1]
        res[f"new_vs_orig_{tag}"] = d
        n, o = res[f"new_{tag}"], res[f"orig_{tag}"]
        gb = sum(n["counters_after_warmup"].get("graph_break", {}).values()) - \
            sum(n["counters_before"].get("graph_break", {}).values())
        ok = (d["u8_max_diff"] <= 2 and n["vs_fp64"]["mean_abs"] <= 1.25 * o["vs_fp64"]["mean_abs"] + 1e-6
              and n["repeat_bitwise"] and n["second_request_bitwise"] and n["inputs_unchanged"]
              and n["no_new_frames_after_warmup"] and n["state_dict_keys_unchanged"] and gb == 0)
        res[f"ok_{tag}"] = bool(ok)
        res[f"graph_breaks_new_{tag}"] = gb
        ok_all &= bool(ok)
        log(json.dumps({"dtype": tag, "new_vs_orig": d, "ok": bool(ok)}))
    res["passed"] = bool(ok_all)
    res["check"] = ("compiled decoder with sub-pixel Upsample vs compiled decoder with the original "
                    "Upsample (same dtype) and both vs fp64 eager original decode; dynamo frames/graph "
                    "breaks/recompiles; repeated decodes + 2 consecutive requests bitwise")
    return res


def kernel_census(fn):
    from collections import Counter

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    seq, cnt, dur = [], Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            seq.append((ev.name[:120], round(ev.device_time_total / 1e3, 4)))
            cnt[ev.name[:120]] += 1
            dur[ev.name[:120]] += ev.device_time_total / 1e3

    def match(*ss):
        ks = [k for k in cnt if any(s in k for s in ss)]
        return {"launches": sum(cnt[k] for k in ks), "total_ms": round(sum(dur[k] for k in ks), 3)}

    return {"total_launches": sum(cnt.values()), "gpu_total_ms": round(sum(dur.values()), 3),
            "nchwToNhwc": match("nchwToNhwc"), "nhwcToNchw": match("nhwcToNchw"),
            "conv_gemm": match("fprop", "implicit", "conv"), "inductor_triton": match("triton_"),
            "by_kernel": sorted(((k, cnt[k], round(dur[k], 3)) for k in cnt), key=lambda t: -t[2])}, seq


@torch.no_grad()
def census():
    import torch._dynamo

    res = {}
    for arm in ("orig", "new"):
        torch._dynamo.reset()
        vae = build(original=(arm == "orig"))
        vae.compile_decoder()
        z = latent(0, torch.float32)
        for _ in range(4):
            vae.decode(z)
        torch.cuda.synchronize()
        summ, seq = kernel_census(lambda: vae.decode(z))
        with open(os.path.join(OUT, f"census_seq_{arm}.json"), "w") as f:
            json.dump(seq, f)
        levels = {}
        for lvl, C, H in LEVELS:
            m = vae.decoder.up[lvl].upsample
            cm = compile_mod(m, torch.float32)
            x = torch.randn(1, C, H, H, device="cuda")
            for _ in range(3):
                cm(x)
            ms = timeit(lambda: cm(x))
            s, _ = kernel_census(lambda: cm(x))
            levels[f"up{lvl}_{C}@{H}"] = {"median_ms": ms, "kernels": s["by_kernel"]}
        summ["by_kernel"] = summ["by_kernel"][:25]
        res[arm] = {"decode": summ, "upsample_module": levels}
        log(json.dumps({arm: res[arm]}))
        del vae
        torch.cuda.empty_cache()
    res["passed"] = True
    return res


if __name__ == "__main__":
    t0 = time.time()
    log("torch", torch.__version__, "cudnn.allow_tf32", torch.backends.cudnn.allow_tf32)
    out = {"unit": unit, "decoder": decoder, "census": census}[MODE]()
    out["elapsed_s"] = time.time() - t0
    out["mode"] = MODE
    print(json.dumps(out))
