"""Idea 276 decode pair: FP8 e4m3 P.V (_attention_tc fp8_pv) in the served FP8 decoder.

usage: python experiments/ascent/fp8_pv/pair.py <checkpoint_root> [--warmup 10] [--iters 60]

Served FP8 recipe both arms: gn_check.build_vae -> set_vae_decoder_trunk_dtype(bf16) ->
compile_decoder(fp8_convs=True) (28 fp8 convs); latent 1x128x64x64 fp32 Generator('cuda').seed(0).
Arm = cand iff Flux3ImageAttnBlock has _fp8_pv.
Steps:
  A  gate A (fp32 path unchanged): Dynamo FX code of the fp32 compiled decoder (capture backend)
     -> sha256 (cross-arm must match); cand: _fp8_pv False on every attn block of the fp32 decoder.
  R  fp32 compiled decoder, seeds 0-2 = this run's reference R.
  O  ON first decode under run_and_get_code: _scaled_mm extern count, (16384,16384) buffers + dtype.
  T  ON timed (warmup, iters, CUDA events). cand: MAINEMU (ON with _fp8_pv forced False = main's
     served decoder) timed interleaved with ON in the same process.
  Q  untimed repeat sequence (same warmup + iters) bitwise vs iteration 0, latent unchanged;
     Dynamo counters unchanged after warmup; Dynamo frames added by ON.
  S  seeds 0-2: ON vs R (PSNR, LPIPS alex, u8), gate-v2 thresholds; cand: MAINEMU vs R, ON vs MAINEMU.
Declared gates (passed holds only these):
  both: repeat bitwise, latent unchanged, no recompile after warmup, ON frames added == 1,
        ON PSNR vs R >= 40 dB seeds 0-2, gate v2 (LPIPS <= 0.265977, PSNR >= 17.658) seeds 0-2.
  cand: exactly 1 _scaled_mm extern, no bf16 (16384,16384) buffer, an e4m3 (16384,16384) buffer,
        fp32 decoder flag off.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import torch
import torch._dynamo
import torch._inductor.config

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--iters", type=int, default=60)
args = p.parse_args()
WARMUP, ITERS = args.warmup, args.iters
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/fp8pv_art")
os.makedirs(ART, exist_ok=True)
torch._dynamo.config.cache_size_limit = 64
torch._inductor.config.fx_graph_cache = False  # output code must be generated in this process
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def git(*a):
    try:
        r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)


git_info = {"log": git("log", "--oneline", "-3")[1]}
for x in ("da43d58f", "59981b4e", "b1758d79"):
    git_info[f"is_ancestor_{x}_rc"] = git("merge-base", "--is-ancestor", x, "HEAD")[0]
log("git", json.dumps(git_info))

from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype  # noqa: E402

iv = gn_check.iv
ARM = "cand" if hasattr(iv.Flux3ImageAttnBlock, "_fp8_pv") else "base"
log("arm", ARM)
LPIPS_MAX, PSNR_MIN = 0.265977, 17.658

gen = torch.Generator(device="cuda").manual_seed(0)
Z = torch.randn(1, 128, 64, 64, device="cuda", generator=gen)
Z0 = Z.clone()


def seed_latent(s):
    g = torch.Generator(device="cuda").manual_seed(s)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g)


def sha(t):
    t = t.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def attn_blocks(vae):
    return [m for m in vae.decoder.modules() if isinstance(m, iv.Flux3ImageAttnBlock)]


def build(trunk_bf16=True, fp8=True, backend=None, force_pv=None):
    vae, _ = gn_check.build_vae(args.root)
    if trunk_bf16:
        set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    if backend is not None:
        dec = vae.decoder
        orig = dec.compile

        def compile_with(*a, **kw):
            kw.pop("options", None)
            return orig(*a, backend=backend, **kw)

        dec.compile = compile_with
    vae.compile_decoder(fp8_convs=True) if fp8 else vae.compile_decoder()
    if force_pv is not None:
        for m in attn_blocks(vae):
            m._fp8_pv = force_pv
    return vae


@torch.inference_mode()
def decode(vae, z=None):
    return vae.decode(Z if z is None else z)


def frames():
    import torch._dynamo.utils as du

    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}


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


_LP = {}


def lpips_model():
    if "m" in _LP:
        return _LP["m"]
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


def cmp(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2).mean().item()
    au, bu = gn_check.to_u8(a), gn_check.to_u8(b)
    du = (au.int() - bu.int()).abs()
    mse8 = (du.double() ** 2).mean().item()
    out = {"psnr_db": float("inf") if mse == 0 else round(10 * torch.log10(torch.tensor(4.0 / mse)).item(), 3),
           "psnr_u8_db": 100.0 if mse8 == 0 else round(10 * torch.log10(torch.tensor(255.0 ** 2 / mse8)).item(), 3),
           "mean_abs": float(f"{d.mean().item():.4e}"), "max_abs": round(d.max().item(), 5),
           "u8_max": du.max().item(), "u8_frac": round((du > 0).float().mean().item(), 6),
           "bitwise": bool(torch.equal(a, b))}
    m = lpips_model()
    if m is not None:
        with torch.no_grad():
            out["lpips"] = round(float(m(a.clamp(-1, 1), b.clamp(-1, 1)).item()), 6)
    return out


def strip(code):
    return "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("#"))


res = {"arm": ARM, "git": git_info, "warmup": WARMUP, "iters": ITERS}
gates = {}

# ------------------------------------------------------------------ A: fp32 FX code (gate A)
graphs = []


def cap_backend(gm, example_inputs):
    graphs.append(gm.code)
    return gm.forward


F32fx = build(trunk_bf16=False, fp8=False, backend=cap_backend)
decode(F32fx)
torch.cuda.synchronize()
code = "\n# ==== next graph ====\n".join(strip(g) for g in graphs)
with open(os.path.join(ART, "fx_fp32.py"), "w") as f:
    f.write(code)
res["gate_A"] = {"fp32_fx_sha256": hashlib.sha256(code.encode()).hexdigest(), "n_graphs": len(graphs),
                 "scaled_mm_in_fp32_fx": "_scaled_mm" in code}
if ARM == "cand":
    res["gate_A"]["fp32_flag_off"] = all(not m._fp8_pv for m in attn_blocks(F32fx))
    gates["fp32_flag_off_and_no_scaled_mm"] = res["gate_A"]["fp32_flag_off"] and not res["gate_A"]["scaled_mm_in_fp32_fx"]
del F32fx
torch.cuda.empty_cache()
log("gate A", json.dumps(res["gate_A"]))

# ------------------------------------------------------------------ R: fp32 compiled decode
R = {}
F32 = build(trunk_bf16=False, fp8=False)
for s in range(3):
    R[s] = decode(F32, seed_latent(s)).clone()
res["fp32_R_sha"] = {s: sha(R[s]) for s in R}
del F32
torch.cuda.empty_cache()
log("fp32 R", json.dumps(res["fp32_R_sha"]))

# ------------------------------------------------------------------ O: ON output code
from torch._inductor.utils import run_and_get_code  # noqa: E402

fr_pre = frames()
ON = build()
res["on_fp8_pv_flags"] = [getattr(m, "_fp8_pv", None) for m in attn_blocks(ON)]
_, codes = run_and_get_code(decode, ON)
oc = "\n".join(codes)
with open(os.path.join(ART, f"on_output_code_{ARM}.py"), "w") as f:
    f.write(oc)
big = [ln.strip()[:200] for ln in oc.splitlines() if re.search(r"empty_strided_cuda\(\(1?,? ?16384, 16384\)", ln)]
n_smm = len(re.findall(r"extern_kernels\._scaled_mm\(", oc))
smm_lines = [ln.strip()[:240] for ln in oc.splitlines() if "extern_kernels._scaled_mm(" in ln]
res["on_output_code"] = {"n_scaled_mm_extern": n_smm, "scaled_mm_lines": smm_lines, "TxT_buffers": big,
                         "n_bmm_extern": len(re.findall(r"extern_kernels\.bmm", oc)),
                         "n_files": len(codes)}
log("output code", json.dumps(res["on_output_code"]))
if ARM == "cand":
    gates["one_scaled_mm_extern"] = n_smm == 1
    gates["no_bf16_TxT_buffer"] = not any("bfloat16" in b for b in big)
    gates["e4m3_TxT_buffer"] = any("float8_e4m3fn" in b for b in big)

for _ in range(WARMUP):
    decode(ON)
torch.cuda.synchronize()
fr_warm = frames()
on_frames_added = fr_warm.get("frames", {}).get("total", 0) - fr_pre.get("frames", {}).get("total", 0)
log("ON warm; frames added", on_frames_added)
EMU = None
if ARM == "cand":
    EMU = build(force_pv=False)
    for _ in range(WARMUP):
        decode(EMU)
    torch.cuda.synchronize()
    fr_warm = frames()
    log("EMU warm")

# ------------------------------------------------------------------ T: timing
ts = {"ON": [], "EMU": []}
for i in range(ITERS):
    order = ["ON", "EMU"] if i % 2 == 0 else ["EMU", "ON"]
    for k in order:
        if k == "EMU" and EMU is None:
            continue
        ts[k].append(timed(ON if k == "ON" else EMU))
    if i % 20 == 0:
        log(f"iter {i} ON={ts['ON'][-1]:.3f}" + (f" EMU={ts['EMU'][-1]:.3f}" if ts["EMU"] else ""))
st = {k: stats(v) for k, v in ts.items() if v}
res["timing"] = st
log("timing", json.dumps(st))


# ------------------------------------------------------------------ Q: repeat sequence
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


q_first, q_rep, q_inp = repeat_seq(ON)
fr_after = frames()
res["repeat"] = {"timed_loop_seq_bitwise_vs_iter0": q_rep, "latent_unchanged": q_inp,
                 "no_recompile_after_warmup": fr_after == fr_warm, "on_dynamo_frames_added": on_frames_added,
                 "recompiles": fr_after.get("recompiles", {}), "graph_break": fr_after.get("graph_break", {}),
                 "seq_sha": sha(q_first)}
if EMU is not None:
    e_first, e_rep, e_inp = repeat_seq(EMU)
    res["repeat"]["emu_seq_bitwise"] = e_rep
log("repeat", json.dumps(res["repeat"]))
gates.update({"repeat_bitwise": q_rep, "latent_unchanged": q_inp,
              "no_recompile_after_warmup": fr_after == fr_warm, "on_frames_added_1": on_frames_added == 1})

# ------------------------------------------------------------------ S: seeds
seeds = []
for s in range(3):
    z = seed_latent(s)
    o = decode(ON, z).clone()
    row = {"seed": s, "sha": sha(o), "repeat5_bitwise": all(torch.equal(decode(ON, z), o) for _ in range(4)),
           "on_vs_R": cmp(o, R[s])}
    if EMU is not None:
        e = decode(EMU, z).clone()
        row["emu_vs_R"] = cmp(e, R[s])
        row["on_vs_emu"] = cmp(o, e)
    seeds.append(row)
    log("seed", json.dumps(row))
res["seeds"] = seeds
psnr_min = min(r["on_vs_R"]["psnr_db"] for r in seeds)
gv2 = all((r["on_vs_R"].get("lpips", 1.0) <= LPIPS_MAX) and r["on_vs_R"]["psnr_u8_db"] >= PSNR_MIN for r in seeds)
gates.update({"on_psnr_vs_R_ge_40dB": psnr_min >= 40.0, "gate_v2_vs_R": gv2})

# ------------------------------------------------------------------ census (info)
from collections import Counter  # noqa: E402

from torch.profiler import ProfilerActivity, profile  # noqa: E402


def census(vae, n=3):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            decode(vae)
        torch.cuda.synchronize()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name] += 1
            dur[ev.name] += ev.device_time_total
    return {"gpu_ms_per_decode": round(sum(dur.values()) / 1e3 / n, 4),
            "top": [(k[:90], cnt[k] // n, round(dur[k] / 1e3 / n, 4)) for k, _ in dur.most_common(14)]}


res["census_on"] = census(ON)
if EMU is not None:
    res["census_emu"] = census(EMU)
log("census", json.dumps({k: v for k, v in res.items() if k.startswith("census")})[:5000])

passed = all(gates.values())
res["gates"] = gates
on = st["ON"]
res["batch"] = {"1": {"kernel_ms": on["median_ms"], "step_ms": on["median_ms"],
                      "noise_floor": max(v["noise_floor"] for v in st.values())}}
if ARM == "cand":
    res["output"] = {"class": "precision",
                     "error_vs_fp32_R": {"psnr_min_db": psnr_min,
                                         "lpips_max": max(r["on_vs_R"].get("lpips", -1) for r in seeds),
                                         "u8_max": max(r["on_vs_R"]["u8_max"] for r in seeds)},
                     "main_emu_error_vs_fp32_R": {"psnr_min_db": min(r["emu_vs_R"]["psnr_db"] for r in seeds),
                                                  "lpips_max": max(r["emu_vs_R"].get("lpips", -1) for r in seeds)},
                     "on_minus_emu_ms": round(st["ON"]["median_ms"] - st["EMU"]["median_ms"], 4)}
else:
    res["output"] = {"class": "precision", "note": "main's served FP8 decoder (the base)",
                     "error_vs_fp32_R": {"psnr_min_db": psnr_min}}
res["passed"] = bool(passed)
res["wall_s"] = round(time.time() - T0, 1)
with open(os.path.join(ART, f"result_{ARM}.json"), "w") as f:
    json.dump(res, f, default=str)
print(json.dumps(res, default=str), flush=True)
