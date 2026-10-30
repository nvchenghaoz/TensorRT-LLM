"""Idea 21 (branch 46): served FLUX.3 VAE decode + encode at 2048^2, timing and in-process check.

usage: python vae_pair.py <checkpoint_root> <mode>
  time  : one served VAE as pipeline_flux3 builds it (fp32 package, bf16 decoder trunk +
          compile_decoder(fp8_convs=True), compile_encoder()). Warm (5), then ITERS rounds of
          decode(latent seed 0) and encode(real 2048^2 reference), order AB/BA by round, CUDA
          events. Then an untimed repeat sequence with the same warmup and iteration count:
          every output bitwise vs its first, inputs unchanged. Runs on any tree (base or branch).
          kernel_ms: t2i-2048 = decode median; i2i-2048 = decode + encode medians.
  check : branch tree only. ONE process, arms A = main's softmax (row_softmax_triton.ENABLED
          False) and B = the op. Per arm: Dynamo graph-break counter delta of its compile,
          kernel census (torch.profiler), outputs; then B vs A (decode pixels max abs, u8 diff;
          encode moments max abs / PSNR), and an alternating untimed repeat sequence with
          error_on_recompile (no recompile once both arms are warm), bitwise vs each arm's first.

DECLARED TOLERANCE (op level, brief / plan 351; job 562 measured it on the op): pq bitwise on
>= 99.99% and <= 1 e4m3 ulp; scale_a max rel <= 1e-5; bf16 P <= 1 ulp. Module level (stated
before this run): decode B vs A u8 max diff <= 2 levels on <= 0.5% of pixels; encode B vs A
PSNR >= 60 dB (the compiled-encoder vs eager gate of branch 4).
"""

import glob
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter

import torch

ROOT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "time"
ITERS = int(os.environ.get("ITERS", "40"))
WARM = int(os.environ.get("WARM", "5"))
STORE = (
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
    "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs"
)
T0 = time.time()
torch._dynamo.config.cache_size_limit = 64


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def git_head():
    try:
        return subprocess.run(["git", "log", "-3", "--oneline"], capture_output=True, text=True).stdout
    except Exception as ex:  # noqa: BLE001
        return repr(ex)


def build():
    from safetensors.torch import load_file

    from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (
        Flux3ImageVAE,
        Flux3ImageVAEConfig,
    )
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype

    pkg = Flux3Checkpoint(ROOT, "nvfp4")
    cfg = pkg.config["vae"]
    vae = Flux3ImageVAE(Flux3ImageVAEConfig())
    vae.load_state_dict(load_file(str(pkg.asset(cfg["path"]))))
    vae = vae.to(device="cuda", dtype=getattr(torch, cfg["dtype"])).eval()
    set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    vae.compile_decoder(fp8_convs=True)
    vae.compile_encoder()
    return vae, getattr(torch, cfg["dtype"])


def latent(seed, dt):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, 128, 128, 128, device="cuda", generator=g).to(dt)


def images(n=2):
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels

    files = sorted(glob.glob(os.path.join(STORE, "*.png")))
    if not files:
        files = sorted(f for f in glob.glob(os.path.join(STORE, "**", "*"), recursive=True) if os.path.isfile(f))
    log("references", [os.path.basename(f) for f in files[:n]])
    return [_load_image_pixels(f, device=torch.device("cuda")).to(torch.float32) for f in files[:n]]


def timed(fn, x):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    fn(x)
    b.record()
    b.synchronize()
    return a.elapsed_time(b)


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "noise_floor": (p90 - p10) / med, "n": n}


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def to_u8(x):
    x = (x.float() / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
    return (x * 255).round().to(torch.uint8)


def u8cmp(a, b):
    du = (to_u8(a).int() - to_u8(b).int()).abs()
    return {"bitwise": bool(torch.equal(a, b)), "u8_max": int(du.max().item()),
            "u8_frac_diff": (du > 0).float().mean().item(),
            "max_abs": (a.float() - b.float()).abs().max().item()}


def errcmp(a, ref):
    d = (a.double() - ref.double()).abs()
    mse = d.pow(2).mean().item()
    rng = (ref.max() - ref.min()).item()
    return {"bitwise": bool(torch.equal(a, ref)), "max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "psnr_db": float("inf") if mse == 0 else 10 * math.log10(rng * rng / mse)}


def census(fn, x):
    from torch.profiler import ProfilerActivity, profile

    fn(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(x)
        torch.cuda.synchronize()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name] += 1
            dur[ev.name] += ev.device_time_total / 1e3
    top = [(k[:120], cnt[k], round(dur[k], 3)) for k, _ in dur.most_common(12)]
    soft = {k[:120]: round(dur[k], 3) for k in dur if "softmax" in k.lower() or "_scaled_mm" in k and "red" in k}
    return {"gpu_total_ms": round(sum(dur.values()), 3), "top": top, "softmax_like": soft}


@torch.inference_mode()
def run_time():
    log(git_head())
    vae, dt = build()
    z = latent(0, dt)
    (x,) = images(1)
    z0, x0 = z.clone(), x.clone()
    zeros = torch.zeros(1, 3, 2048, 2048, device="cuda")
    t = time.time()
    vae.encode(zeros)  # the server's warmup encode
    vae.decode(z)
    torch.cuda.synchronize()
    log(f"compile+first calls {time.time() - t:.1f}s")
    for _ in range(WARM):
        vae.decode(z)
        vae.encode(x)
    torch.cuda.synchronize()
    td, te = [], []
    for i in range(ITERS):
        if i % 2 == 0:
            td.append(timed(vae.decode, z))
            te.append(timed(vae.encode, x))
        else:
            te.append(timed(vae.encode, x))
            td.append(timed(vae.decode, z))
        if i % 10 == 0:
            log(f"iter {i}: decode {td[-1]:.2f} encode {te[-1]:.2f}")
    sd, se = stats(td), stats(te)
    log("decode", json.dumps(sd), "encode", json.dumps(se))
    # untimed repeat sequence: same warmup + iteration count
    for _ in range(WARM):
        vae.decode(z)
        vae.encode(x)
    fd = fe = None
    rep_d = rep_e = inp = True
    for i in range(ITERS):
        inp &= bool(torch.equal(z, z0) and torch.equal(x, x0))
        yd, ye = vae.decode(z), vae.encode(x)
        if fd is None:
            fd, fe = yd.clone(), ye.clone()
        else:
            rep_d &= bool(torch.equal(yd, fd))
            rep_e &= bool(torch.equal(ye, fe))
    torch.cuda.synchronize()
    noise_i2i = max(sd["noise_floor"], se["noise_floor"])
    res = {
        "batch": {"t2i-2048": {"kernel_ms": sd["median_ms"], "noise_floor": sd["noise_floor"]},
                  "i2i-2048": {"kernel_ms": sd["median_ms"] + se["median_ms"], "noise_floor": noise_i2i}},
        "output": {"class": "reordered"},
        "decode": sd, "encode": se,
        "decode_sha": sha(fd), "encode_sha": sha(fe),
        "repeat_bitwise": {"decode": rep_d, "encode": rep_e}, "inputs_unchanged": inp,
        "passed": bool(rep_d and rep_e and inp),
        "bench": "idea21/vae_pair.py time: served bf16+fp8 compiled decoder (1x128x128x128 latent) + "
                 "compiled fp32 channels_last encoder (real 2048^2 reference)",
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }
    print(json.dumps(res))


@torch.inference_mode()
def run_check():
    from torch._dynamo.utils import counters

    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import row_softmax_triton as rs

    log(git_head())
    vae, dt = build()
    za, zb = latent(0, dt), latent(1, dt)
    xa, xb = images(2)
    keep = [t.clone() for t in (za, zb, xa, xb)]
    zeros = torch.zeros(1, 3, 2048, 2048, device="cuda")
    res = {"arms": {}}
    gates = {}
    for arm, en in (("A", False), ("B", True)):
        rs.ENABLED = en
        gb0 = sum(counters["graph_break"].values())
        fr0 = dict(counters["frames"])
        t = time.time()
        vae.encode(zeros)
        vae.decode(za)
        torch.cuda.synchronize()
        gb1 = sum(counters["graph_break"].values())
        res["arms"][arm] = {"compile_s": round(time.time() - t, 1), "graph_breaks": gb1 - gb0,
                            "graph_break_reasons": dict(counters["graph_break"]),
                            "frames_before": fr0, "frames_after": dict(counters["frames"]),
                            "census_decode": census(vae.decode, za), "census_encode": census(vae.encode, xa)}
        log(arm, json.dumps(res["arms"][arm]))
    gates["no_new_graph_breaks"] = res["arms"]["B"]["graph_breaks"] == res["arms"]["A"]["graph_breaks"] == 0
    cd, ce = res["arms"]["B"]["census_decode"], res["arms"]["B"]["census_encode"]
    names_d = " ".join(n for n, _, _ in cd["top"])
    names_e = " ".join(n for n, _, _ in ce["top"])
    gates["decoder_uses_op"] = "_softmax_rows_e4m3_kernel" in names_d
    gates["encoder_uses_op"] = "_softmax_rows_bf16_kernel" in names_e
    gates["no_inductor_softmax_over_S"] = not any(
        n.startswith("triton_red") and ("softmax" in n or "exp" in n) and ms > 2.0
        for n, _, ms in cd["top"] + ce["top"])
    # outputs and an alternating repeat sequence, no recompiles from here on
    torch._dynamo.config.error_on_recompile = True
    first = {}
    rep = True
    recompile_err = None
    try:
        for i in range(int(os.environ.get("REPS", "4"))):
            for arm, en in (("A", False), ("B", True)) if i % 2 == 0 else (("B", True), ("A", False)):
                rs.ENABLED = en
                for name, fn, inp in (("dec_a", vae.decode, za), ("enc_a", vae.encode, xa),
                                      ("dec_b", vae.decode, zb), ("enc_b", vae.encode, xb)):
                    y = fn(inp)
                    k = (arm, name)
                    if k in first:
                        rep &= bool(torch.equal(y, first[k]))
                    else:
                        first[k] = y.clone()
            ok_inp = all(torch.equal(a, b) for a, b in zip((za, zb, xa, xb), keep))
            gates[f"inputs_unchanged_{i}"] = ok_inp
            log(f"rep round {i}: repeat_bitwise so far {rep}, inputs unchanged {ok_inp}")
    except Exception as ex:  # noqa: BLE001
        recompile_err = repr(ex)[:500]
        log("FAILED", recompile_err)
    torch._dynamo.config.error_on_recompile = False
    gates["no_recompile"] = recompile_err is None
    gates["repeat_bitwise"] = rep
    cmp = {}
    for name in ("dec_a", "dec_b"):
        if ("A", name) in first and ("B", name) in first:
            cmp[name] = u8cmp(first[("B", name)], first[("A", name)])
            gates[f"{name}_tol"] = cmp[name]["u8_max"] <= 2 and cmp[name]["u8_frac_diff"] <= 0.005
    for name in ("enc_a", "enc_b"):
        if ("A", name) in first and ("B", name) in first:
            cmp[name] = errcmp(first[("B", name)], first[("A", name)])
            gates[f"{name}_tol"] = cmp[name]["psnr_db"] >= 60
    res["B_vs_A"] = cmp
    res["hashes"] = {f"{a}:{n}": sha(t) for (a, n), t in first.items()}
    res["recompile_error"] = recompile_err
    res["gates"] = gates
    res["passed"] = bool(all(gates.values()))
    res["output"] = {"class": "reordered"}
    log("B_vs_A", json.dumps(cmp))
    print(json.dumps(res))


if __name__ == "__main__":
    {"time": run_time, "check": run_check}[MODE]()
