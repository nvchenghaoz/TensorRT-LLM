"""Idea 218 decode pair: FP8 ResnetBlock convs (compile_decoder(fp8_convs=True)) vs main's bf16 decoder.

usage: python experiments/ascent/b218_pair.py <checkpoint_root> --arm base|cand [--warmup 10] [--iters 60]

Real checkpoint VAE, served recipe (set_vae_decoder_trunk_dtype(bf16), conv_out fp32, compile_decoder),
latent 1x128x64x64 fp32 from torch.Generator('cuda').manual_seed(0) (as vae_decode_bench.py).

base (main's tree): M = compile_decoder() bf16; timed (CUDA events per decode, --warmup then --iters);
   untimed repeat sequence of the same length (bitwise vs iteration 0, latent unchanged, Dynamo
   counters unchanged). A/A: M2 = a second bf16 compile with Inductor caches disabled (envelope).
cand (this tree): ON = compile_decoder(fp8_convs=True), OFF = compile_decoder() (flag off), timed
   interleaved in one process; repeat sequence for ON and OFF.
Both arms: Dynamo FX graph code (capture backend, comment lines stripped) of the bf16 and the fp32
   decoder -> sha256 + files; accuracy on seeds 0-2 vs the fp32 eager decoder R (PSNR, u8 diff).
   The base arm leaves its FX code and seed outputs in /tmp/b218_xchg; the cand arm diffs against
   them when present (same container) and reports "unavailable" otherwise.
Declared gates (the passed flag holds only these):
   base: repeat bitwise, latent unchanged, no recompile after warmup.
   cand: ON and OFF repeat bitwise, latent unchanged, no Dynamo recompile and no cuDNN FE plan build
   after warmup, 28 convs prepared on ON and prepare_fp8_convs not called on OFF, ON PSNR vs fp32 R
   >= 40 dB on seeds 0-2, decoder fused-tail kernel (_conv3x3_to3_gather_kernel) in the ON census.
Last stdout line: pair evidence JSON (batch.1.kernel_ms = this arm's timed decoder median).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--arm", required=True, choices=["base", "cand"])
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--iters", type=int, default=60)
args = p.parse_args()
WARMUP, ITERS = args.warmup, args.iters
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b218_art")
os.makedirs(ART, exist_ok=True)
XCHG = "/tmp/b218_xchg"
MAIN = "1185e8f970fa1f56c64deb4efbaae6f5d939e02c"
torch._dynamo.config.cache_size_limit = 64
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def git(*a):
    try:
        r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)


rc_anc, _ = git("merge-base", "--is-ancestor", MAIN, "HEAD")
_, head = git("log", "--oneline", "-3")
git_info = {"log": head, "is_ancestor_main_rc": rc_anc, "cwd": os.getcwd()}
log("git", json.dumps(git_info))

from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype  # noqa: E402

iv = gn_check.iv
try:
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import fp8_conv as fc
except Exception:  # noqa: BLE001  (main's tree has no fp8_conv)
    fc = None
log("tree has fp8_conv:", fc is not None)

prep_calls = []
if fc is not None:
    _orig_prep = fc.prepare_fp8_convs

    def _counting_prep(dec):
        n = _orig_prep(dec)
        prep_calls.append(n)
        return n

    fc.prepare_fp8_convs = _counting_prep  # compile_decoder calls it through the module attribute

gen = torch.Generator(device="cuda").manual_seed(0)
Z = torch.randn(1, 128, 64, 64, device="cuda", generator=gen)
Z0 = Z.clone()


def build(trunk_bf16=True, compile_kw=None, backend=None):
    vae, _ = gn_check.build_vae(args.root)
    if trunk_bf16:
        set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    if compile_kw is not None:
        if backend is not None:
            dec = vae.decoder
            orig = dec.compile

            def compile_with(*a, **kw):
                kw.pop("options", None)
                return orig(*a, backend=backend, **kw)

            dec.compile = compile_with
        vae.compile_decoder(**compile_kw)
    return vae


@torch.inference_mode()
def decode(vae, z=None):
    return vae.decode(Z if z is None else z)


def frames():
    import torch._dynamo.utils as du

    return json.dumps({k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")},
                      sort_keys=True)


def n_plans():
    return len(fc._graphs) if fc is not None else 0


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": round(med, 4), "p10_ms": round(p10, 4), "p90_ms": round(p90, 4),
            "noise_floor": (p90 - p10) / med}


def timed(vae):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    decode(vae)
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def warm(vae, tag):
    t0 = time.time()
    for _ in range(WARMUP):
        decode(vae)
    torch.cuda.synchronize()
    log(f"[{tag}] warm {WARMUP} decodes in {time.time() - t0:.1f}s")


def repeat_seq(vae):
    for _ in range(WARMUP):
        decode(vae)
    first, rep, inp = None, True, True
    for _ in range(ITERS):
        inp &= torch.equal(Z, Z0)
        out = decode(vae)
        if first is None:
            first = out.clone()
        elif not torch.equal(out, first):
            rep = False
    torch.cuda.synchronize()
    return first, rep, inp


def to_u8(x):
    return gn_check.to_u8(x.float())


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    mse = ((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)) ** 2).mean().item()
    du = (to_u8(a).int() - to_u8(b).int()).abs()
    return {"psnr_db": float("inf") if mse == 0 else round(10 * torch.log10(torch.tensor(4.0 / mse)).item(), 3),
            "max_abs": round(d.max().item(), 5), "u8_max_diff": du.max().item(),
            "u8_frac_diff": round((du > 0).float().mean().item(), 6), "bitwise": bool(torch.equal(a, b))}


def strip(code):
    return "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("#"))


def fx_capture(trunk_bf16, tag):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm.code)
        return gm.forward

    vae = build(trunk_bf16, {}, backend)
    decode(vae)
    torch.cuda.synchronize()
    code = "\n# ==== next graph ====\n".join(strip(g) for g in graphs)
    path = os.path.join(ART, f"fx_{tag}.py")
    with open(path, "w") as f:
        f.write(code)
    del vae
    torch.cuda.empty_cache()
    return {"n_graphs": len(graphs), "sha256": hashlib.sha256(code.encode()).hexdigest(), "path": path,
            "lines": code.count("\n") + 1}, code


def seed_latent(s):
    g = torch.Generator(device="cuda").manual_seed(s)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g)


res = {"arm": args.arm, "git": git_info, "warmup": WARMUP, "iters": ITERS}

# ---------------------------------------------------------------- FX graphs (flag off / main)
prep_before_fx = len(prep_calls)
fx_bf16, code_bf16 = fx_capture(True, "bf16")
fx_fp32, code_fp32 = fx_capture(False, "fp32")
res["fx"] = {"bf16": fx_bf16, "fp32": fx_fp32}
log("fx", json.dumps(res["fx"]))

# ---------------------------------------------------------------- fp32 reference R (eager)
R = {}
ref_vae = build(trunk_bf16=False, compile_kw=None)
for s in range(3):
    R[s] = decode(ref_vae, seed_latent(s)).clone()
del ref_vae
torch.cuda.empty_cache()
log("fp32 eager reference built")

xchg_ok = os.path.isdir(XCHG) and os.path.exists(os.path.join(XCHG, "stamp.json"))
if args.arm == "base":
    os.makedirs(XCHG, exist_ok=True)
    M = build(True, {})
    warm(M, "M")
    f0 = frames()
    ts = []
    for i in range(ITERS):
        ts.append(timed(M))
        if i % 20 == 0:
            log(f"iter {i} M={ts[-1]:.3f}")
    st = stats(ts)
    first, rep, inp = repeat_seq(M)
    no_recompile = frames() == f0
    acc, outs = [], {}
    for s in range(3):
        outs[s] = decode(M, seed_latent(s)).clone()
        acc.append({"seed": s, **cmp(outs[s], R[s])})
    log("main vs R", json.dumps(acc))
    # A/A envelope: second compile of main's bf16 decoder with Inductor caches disabled
    import torch._inductor.config as icfg

    old = icfg.force_disable_caches
    icfg.force_disable_caches = True
    M2 = build(True, {})
    warm(M2, "M2")
    icfg.force_disable_caches = old
    aa = [{"seed": s, **cmp(decode(M2, seed_latent(s)), outs[s])} for s in range(3)]
    log("A/A", json.dumps(aa))
    for s in range(3):
        torch.save(outs[s].cpu(), os.path.join(XCHG, f"main_s{s}.pt"))
    with open(os.path.join(XCHG, "fx_bf16.py"), "w") as f:
        f.write(code_bf16)
    with open(os.path.join(XCHG, "fx_fp32.py"), "w") as f:
        f.write(code_fp32)
    with open(os.path.join(XCHG, "stamp.json"), "w") as f:
        json.dump({"t": time.time(), "git": git_info}, f)
    passed = rep and inp and no_recompile
    res.update({"batch": {"1": {"kernel_ms": st["median_ms"], "step_ms": st["median_ms"],
                                "noise_floor": st["noise_floor"]}},
                "output": {"class": "exact", "note": "main's compiled bf16 decoder (the base)"},
                "timing": st, "repeat_bitwise": rep, "inputs_unchanged": inp, "no_recompile_after_warmup": no_recompile,
                "main_vs_fp32_R": acc, "main_AA_nocache_vs_main": aa,
                "psnr_min_main_vs_R": min(a["psnr_db"] for a in acc), "passed": bool(passed)})
else:
    import torch._inductor.config as icfg

    n_prep0 = len(prep_calls)
    OFF = build(True, {})
    off_prep_called = len(prep_calls) != n_prep0
    ON = build(True, {"fp8_convs": True})
    n_prepared = prep_calls[-1] if prep_calls else 0
    n_fp8_blocks = sum(1 for m in ON.decoder.modules() if isinstance(m, iv.Flux3ImageResnetBlock) and m._fp8)
    warm(OFF, "OFF")
    warm(ON, "ON")
    f0, pl0 = frames(), n_plans()
    ts = {"ON": [], "OFF": []}
    for i in range(ITERS):
        order = ["ON", "OFF"] if i % 2 == 0 else ["OFF", "ON"]
        for k in order:
            ts[k].append(timed(ON if k == "ON" else OFF))
        if i % 20 == 0:
            log(f"iter {i} ON={ts['ON'][-1]:.3f} OFF={ts['OFF'][-1]:.3f}")
    st = {k: stats(v) for k, v in ts.items()}
    on_first, on_rep, on_inp = repeat_seq(ON)
    off_first, off_rep, off_inp = repeat_seq(OFF)
    no_recompile = frames() == f0
    no_new_plans = n_plans() == pl0
    acc_on, acc_off, on_vs_off, outs_off, outs_on = [], [], [], {}, {}
    for s in range(3):
        outs_on[s] = decode(ON, seed_latent(s)).clone()
        outs_off[s] = decode(OFF, seed_latent(s)).clone()
        acc_on.append({"seed": s, **cmp(outs_on[s], R[s])})
        acc_off.append({"seed": s, **cmp(outs_off[s], R[s])})
        on_vs_off.append({"seed": s, **cmp(outs_on[s], outs_off[s])})
    log("ON vs R", json.dumps(acc_on))
    log("OFF vs R", json.dumps(acc_off))
    no_new_plans &= n_plans() == pl0
    # ON census (one warm decode)
    from collections import Counter

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        decode(ON)
        torch.cuda.synchronize()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name] += 1
            dur[ev.name] += ev.device_time_total
    tail = {k: cnt[k] for k in cnt if "conv3x3_to3_gather" in k}
    fp8k = {k[:120]: (cnt[k], round(dur[k] / 1e3, 3)) for k in cnt
            if any(t in k.lower() for t in ("e4m3", "fp8", "f8"))}
    census = {"total_launches": sum(cnt.values()), "gpu_total_ms": round(sum(dur.values()) / 1e3, 3),
              "tail_kernel": tail, "fp8_kernels": fp8k,
              "fp8_kernel_launches": sum(v[0] for v in fp8k.values()),
              "top": [(k[:120], cnt[k], round(dur[k] / 1e3, 3)) for k, _ in dur.most_common(20)]}
    log("census", json.dumps(census)[:3000])
    # cross-arm (same container only)
    cross = "unavailable (no /tmp/b218_xchg from the base arm in this container)"
    if xchg_ok:
        stamp = json.load(open(os.path.join(XCHG, "stamp.json")))
        cross = {"base_stamp_age_s": round(time.time() - stamp["t"], 1), "base_git": stamp["git"]}
        for tag, code in (("bf16", code_bf16), ("fp32", code_fp32)):
            base_code = open(os.path.join(XCHG, f"fx_{tag}.py")).read()
            cross[f"fx_{tag}_identical_to_main"] = base_code == code
        cross["off_vs_main"], cross["on_vs_main"] = [], []
        for s in range(3):
            m = torch.load(os.path.join(XCHG, f"main_s{s}.pt")).cuda()
            cross["off_vs_main"].append({"seed": s, **cmp(outs_off[s], m)})
            cross["on_vs_main"].append({"seed": s, **cmp(outs_on[s], m)})
    log("cross", json.dumps(cross))
    psnr_min_on = min(a["psnr_db"] for a in acc_on)
    gates = {"on_repeat_bitwise": on_rep, "off_repeat_bitwise": off_rep, "inputs_unchanged": on_inp and off_inp,
             "no_recompile_after_warmup": no_recompile, "no_cudnn_plan_build_after_warmup": no_new_plans,
             "n_prepared_28": n_prepared == 28 and n_fp8_blocks == 14, "off_prepare_not_called": not off_prep_called,
             "on_psnr_vs_fp32_R_ge_40dB": psnr_min_on >= 40.0, "tail_kernel_in_on_census": bool(tail)}
    res.update({"batch": {"1": {"kernel_ms": st["ON"]["median_ms"], "step_ms": st["ON"]["median_ms"],
                                "noise_floor": max(st["ON"]["noise_floor"], st["OFF"]["noise_floor"])}},
                "output": {"class": "precision",
                           "error_vs_fp32_R": {"psnr_min_db": psnr_min_on,
                                               "u8_max_diff": max(a["u8_max_diff"] for a in acc_on),
                                               "max_abs": max(a["max_abs"] for a in acc_on)},
                           "flag_off_error_vs_fp32_R": {"psnr_min_db": min(a["psnr_db"] for a in acc_off),
                                                        "u8_max_diff": max(a["u8_max_diff"] for a in acc_off)}},
                "timing": st, "on_minus_off_ms": round(st["ON"]["median_ms"] - st["OFF"]["median_ms"], 4),
                "gates": gates, "n_prepared": n_prepared, "n_fp8_blocks": n_fp8_blocks,
                "cudnn_plans": fc.graph_table() if fc else None,
                "on_vs_fp32_R": acc_on, "off_vs_fp32_R": acc_off, "on_vs_off": on_vs_off,
                "cross_arm": cross, "census_on": census, "passed": bool(all(gates.values()))})
res["wall_s"] = round(time.time() - T0, 1)
print(json.dumps(res, default=str), flush=True)
