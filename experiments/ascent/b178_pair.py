"""Branch 178 / idea 260: nin_shortcut bias (and up[1].block[0]'s b_fold) as an fp32 bias in
conv2's FP8 cuDNN epilogue. One script for both pair arms (base = main, cand = this branch).

usage: python experiments/ascent/b178_pair.py <checkpoint_root> [--warmup 10] [--iters 60]

Served FP8 recipe: gn_check.build_vae -> set_vae_decoder_trunk_dtype(bf16) ->
compile_decoder(fp8_convs=True); latent 1x128x64x64 fp32 torch.Generator('cuda').manual_seed(0).
Steps (both arms):
  F  fp32 compiled decoder (compile_decoder(), served vae dtype) on seeds 0-2: R = this arm's
     fp32 decode; sha256 of raw bytes (cross-arm: must be equal, the fp32 path is untouched).
  T  ON timed: --warmup decodes, --iters CUDA-event-timed decodes; then the untimed
     timed-loop-shaped sequence (same warmup + iters) bitwise vs iteration 0, latent unchanged,
     Dynamo counters and cuDNN FE graph count unchanged; Dynamo frames added by ON == 1.
  S  seeds 0-2: ON decode 5x (bitwise vs the first, latent unchanged); vs R: PSNR, mean_abs,
     u8_max, LPIPS(alex); sha256.
  C  census: torch.profiler over 5 warm ON decodes; fp8_conv3x3 CPU-op count per decode; the
     ordered kernel list of each decode; the kernels between FP8 conv launches 16->17
     (up[1].block[0] conv2) and 22->23 (up[0].block[0] conv2), per-position median over 5.
  K  per-key synthetic cuDNN check: every graph key with has_bias in {0,1}: seeded e4m3/bf16
     operands -> sha256 of the fp8_conv3x3 output (cross-arm: must be equal) + plan name;
     fp32-bias keys (cand): plan names.
  E  eager FP8 per-call record (28 calls: key, output sha256).
cand only:
  M  main emulation in-process: main's two FP8 branches (Flux3ImageResnetBlock.forward,
     _resnet_block_with_input_bias) patched in on a second compiled ON instance -> main's served
     decode per seed; cand vs main-emu: PSNR, mean_abs, u8_max, LPIPS; eager per-call: cand vs
     main-emu keys (only calls 17 and 23 change, to has_bias=2) and outputs (calls 0-16 bitwise,
     call 17 error vs main's).
Last stdout line: pair evidence JSON (batch.1.kernel_ms = ON decode median).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--iters", type=int, default=60)
p.add_argument("--skip-fp32", action="store_true")
args = p.parse_args()
WARMUP, ITERS = args.warmup, args.iters
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b178_art")
os.makedirs(ART, exist_ok=True)
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


rc_anc, _ = git("merge-base", "--is-ancestor", "fb730d34", "HEAD")
_, head = git("log", "--oneline", "-3")
git_info = {"log": head, "is_ancestor_fb730d34_rc": rc_anc}
log("git", json.dumps(git_info))

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import fp8_conv as fc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype  # noqa: E402

iv = gn_check.iv
ARM = "cand" if hasattr(fc, "BIAS_FP32") else "base"
log("arm", ARM)

gen = torch.Generator(device="cuda").manual_seed(0)
Z = torch.randn(1, 128, 64, 64, device="cuda", generator=gen)
Z0 = Z.clone()


def seed_latent(s):
    g = torch.Generator(device="cuda").manual_seed(s)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g)


def sha(t):
    t = t.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def build(trunk_bf16=True, fp8=True, compiled=True):
    vae, _ = gn_check.build_vae(args.root)
    if trunk_bf16:
        set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
    if not compiled:
        vae.decoder.compile = lambda *a, **k: None
    vae.compile_decoder(fp8_convs=fp8) if fp8 else vae.compile_decoder()
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
    du = (gn_check.to_u8(a).int() - gn_check.to_u8(b).int()).abs()
    out = {"psnr_db": float("inf") if mse == 0 else round(10 * torch.log10(torch.tensor(4.0 / mse)).item(), 3),
           "mean_abs": float(f"{d.mean().item():.4e}"), "max_abs": round(d.max().item(), 5),
           "u8_max": du.max().item(), "u8_frac": round((du > 0).float().mean().item(), 6),
           "bitwise": bool(torch.equal(a, b))}
    m = lpips_model()
    if m is not None:
        with torch.no_grad():
            out["lpips"] = round(float(m(a.clamp(-1, 1), b.clamp(-1, 1)).item()), 6)
    return out


res = {"arm": ARM, "git": git_info, "warmup": WARMUP, "iters": ITERS}

# ------------------------------------------------------------------ F: fp32 decode (R)
R, fp32_sha = {}, {}
if not args.skip_fp32:
    F32 = build(trunk_bf16=False, fp8=False)
    for s in range(3):
        R[s] = decode(F32, seed_latent(s)).clone()
        r2 = decode(F32, seed_latent(s))
        fp32_sha[s] = {"sha": sha(R[s]), "repeat_bitwise": bool(torch.equal(r2, R[s]))}
    del F32
    torch.cuda.empty_cache()
log("fp32 R", json.dumps(fp32_sha))
res["fp32_decode_sha"] = fp32_sha

# ------------------------------------------------------------------ T: ON timed + repeat
ON = build()
n_fp8_blocks = sum(1 for m in ON.decoder.modules() if isinstance(m, iv.Flux3ImageResnetBlock) and m._fp8)
fr_pre = frames()
t0 = time.time()
for _ in range(WARMUP):
    decode(ON)
torch.cuda.synchronize()
log(f"ON warm {WARMUP} in {time.time() - t0:.1f}s")
fr_warm, pl_warm = frames(), len(fc._graphs)
ts = []
for i in range(ITERS):
    ts.append(timed(ON))
    if i % 20 == 0:
        log(f"iter {i} {ts[-1]:.3f} ms")
st = stats(ts)
log("timing", json.dumps(st))
for _ in range(WARMUP):
    decode(ON)
first, rep, inp = None, True, True
for i in range(ITERS):
    inp &= torch.equal(Z, Z0)
    out = decode(ON)
    if first is None:
        first = out.clone()
    elif not torch.equal(out, first):
        rep = False
torch.cuda.synchronize()
fr_after, pl_after = frames(), len(fc._graphs)
on_frames_added = fr_warm.get("frames", {}).get("total", 0) - fr_pre.get("frames", {}).get("total", 0)
res["timing"] = st
res["repeat"] = {"timed_loop_seq_bitwise": rep, "latent_unchanged": inp,
                 "no_recompile_after_warmup": fr_after == fr_warm, "no_plan_build_after_warmup": pl_after == pl_warm,
                 "on_dynamo_frames_added": on_frames_added, "graph_break": fr_after.get("graph_break", {}),
                 "recompiles": fr_after.get("recompiles", {}), "seq_sha": sha(first)}
log("repeat", json.dumps(res["repeat"]))

# ------------------------------------------------------------------ S: seeds
seeds, on_out = [], {}
for s in range(3):
    z = seed_latent(s)
    z0 = z.clone()
    o = decode(ON, z).clone()
    rb = all(torch.equal(decode(ON, z), o) for _ in range(4))
    on_out[s] = o
    row = {"seed": s, "sha": sha(o), "repeat5_bitwise": rb, "latent_unchanged": bool(torch.equal(z, z0))}
    if s in R:
        row["vs_R"] = cmp(o, R[s])
    seeds.append(row)
    log("seed", json.dumps(row))
res["seeds"] = seeds

# ------------------------------------------------------------------ C: census
from torch.profiler import ProfilerActivity, profile  # noqa: E402

NDEC = 5
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(NDEC):
        decode(ON)
        torch.cuda.synchronize()
evs = [e for e in prof.events()]
n_fp8_ops = sum(1 for e in evs if e.name == "flux3_vae::fp8_conv3x3" and e.device_type != torch.autograd.DeviceType.CUDA)
kern = sorted([e for e in evs if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: e.time_range.start)
names = Counter(e.name for e in kern)
per_dec = len(kern) // NDEC
fp8_name_cands = {k: v for k, v in names.items()
                  if v == 28 * NDEC and not k.startswith("triton_")}
log("kernel names with 28/decode:", json.dumps({k[:100]: v for k, v in fp8_name_cands.items()}))


def is_fp8(name):
    return name in fp8_name_cands or (any(t in name.lower() for t in ("e4m3", "f8f8", "fp8"))
                                      and not name.startswith("triton_"))


decs = [kern[i * per_dec:(i + 1) * per_dec] for i in range(NDEC)] if len(kern) % NDEC == 0 else []
census = {"fp8_conv3x3_cpu_ops_per_decode": n_fp8_ops / NDEC, "kernels_per_decode": len(kern) / NDEC,
          "gpu_ms_per_decode": round(sum(e.device_time_total for e in kern) / 1e3 / NDEC, 4)}
windows = {}
if decs and all([e.name for e in d] == [e.name for e in decs[0]] for d in decs):
    d0 = decs[0]
    fp8_pos = [i for i, e in enumerate(d0) if is_fp8(e.name)]
    census["fp8_kernel_launches_per_decode"] = len(fp8_pos)

    def med(i):
        v = sorted(d[i].device_time_total for d in decs)
        return round(v[len(v) // 2], 2)

    with open(os.path.join(ART, f"census_{ARM}.json"), "w") as f:
        json.dump([(i, e.name[:160], med(i)) for i, e in enumerate(d0)], f)
    if len(fp8_pos) >= 24:
        for tag, (a, b) in {"up1_block0_conv2": (16, 17), "up0_block0_conv2": (22, 23)}.items():
            lo, hi = fp8_pos[a], fp8_pos[b]
            windows[tag] = [(i, d0[i].name[:90], med(i)) for i in range(lo, hi + 1)]
        census["all_fp8_conv_us"] = [med(i) for i in fp8_pos]
    census["windows_us"] = windows
else:
    census["note"] = "decodes not aligned; see top"
census["top"] = [(k[:100], v) for k, v in names.most_common(12)]
res["census"] = census
log("census", json.dumps(census)[:6000])

# ------------------------------------------------------------------ K: per-key synthetic
FP8 = torch.float8_e4m3fn


def synth_run(key):
    n, cin, cout, h, w, hb, hr = key
    g = torch.Generator(device="cuda").manual_seed(cin * 7 + cout * 13 + h)
    xq = (torch.randn(n, cin, h, w, device="cuda", generator=g) * 64).clamp(-448, 448).to(FP8)
    xq = xq.contiguous(memory_format=torch.channels_last)
    wq = (torch.randn(cout, 3, 3, cin, device="cuda", generator=g) * 64).clamp(-448, 448).to(FP8)
    sx = torch.rand(1, device="cuda", generator=g) * 1e-2 + 1e-3
    sw = torch.rand(cout, device="cuda", generator=g) * 1e-3 + 1e-4
    b = torch.randn(cout, device="cuda", generator=g)
    r = torch.randn(n, cout, h, w, device="cuda", generator=g).to(torch.bfloat16).contiguous(
        memory_format=torch.channels_last)
    bias = None if not hb else (b if hb == 2 else b.to(torch.bfloat16))
    with torch.inference_mode():
        o = torch.ops.flux3_vae.fp8_conv3x3(xq, sx, wq, sw, bias, r if hr else None)
        o2 = torch.ops.flux3_vae.fp8_conv3x3(xq, sx, wq, sw, bias, r if hr else None)
    return o, bool(torch.equal(o, o2))


keys_tab = []
for d in fc.graph_table():
    key = tuple(int(v) for v in d["key"])
    row = {"key": key, "plan": d["plan"], "plan_name": d["plan_name"], "n_plans": d["n_plans"]}
    o, rep2 = synth_run(key)
    row["synth_sha"], row["synth_repeat_bitwise"] = sha(o), rep2
    keys_tab.append(row)
keys_tab.sort(key=lambda r: r["key"])
res["keys"] = keys_tab
log("keys", json.dumps(keys_tab))


# ------------------------------------------------------------------ E: eager per-call record
class Rec:
    def __init__(self):
        self.calls = []
        self.orig = fc.fp8_conv3x3

    def __enter__(self):
        orig = self.orig

        def wrapped(xq, s_x, wq, s_w, bias=None, residual=None):
            o = orig(xq, s_x, wq, s_w, bias, residual)
            hb = 0 if bias is None else (2 if bias.dtype == torch.float32 else 1)
            self.calls.append({"key": [xq.shape[0], xq.shape[1], wq.shape[0], xq.shape[2], xq.shape[3], hb,
                                       int(residual is not None)], "out": o.clone()})
            return o

        fc.fp8_conv3x3 = wrapped
        return self

    def __exit__(self, *a):
        fc.fp8_conv3x3 = self.orig


EAG = build(compiled=False)
with Rec() as rec_c:
    decode(EAG)
torch.cuda.synchronize()
res["eager_calls"] = [{"i": i, "key": c["key"], "sha": sha(c["out"])} for i, c in enumerate(rec_c.calls)]
log("eager calls", len(rec_c.calls))

# ------------------------------------------------------------------ M: main emulation (cand only)
if ARM == "cand":
    _fp8c = fc
    ORIG_FWD = iv.Flux3ImageResnetBlock.forward
    ORIG_FOLD = iv._resnet_block_with_input_bias

    def main_forward(self, x):  # main 0761b30b's FP8 branch verbatim
        if self._fp8:
            hq, sx = self._fp8_conv2_input(x)
            sc = self.nin_shortcut(x) if self.in_channels != self.out_channels else x
            c2 = self.conv2
            return _fp8c.fp8_conv3x3(hq, sx, c2.fp8_wq, c2.fp8_sw, c2.bias, sc)
        return ORIG_FWD(self, x)

    def main_fold(block, x_nb, b_in):  # main 0761b30b's FP8 branch verbatim
        if block._fp8:
            x = x_nb + b_in.view(1, -1, 1, 1)
            hq, sx = block._fp8_conv2_input(x)
            nin = block.nin_shortcut
            w = nin.weight
            b_fold = (nin.bias.double() + torch.mv(w[:, :, 0, 0].double(), b_in.double())).float()
            s = torch.nn.functional.conv2d(x_nb, w, None)
            s = (s.float() + b_fold.view(1, -1, 1, 1)).to(x_nb.dtype)
            c2 = block.conv2
            return _fp8c.fp8_conv3x3(hq, sx, c2.fp8_wq, c2.fp8_sw, c2.bias, s)
        return ORIG_FOLD(block, x_nb, b_in)

    iv.Flux3ImageResnetBlock.forward = main_forward
    iv._resnet_block_with_input_bias = main_fold
    try:
        with Rec() as rec_m:
            decode(EAG)
        torch.cuda.synchronize()
        EMU = build()
        for _ in range(3):
            decode(EMU)
        emu = {}
        for s in range(3):
            o = decode(EMU, seed_latent(s)).clone()
            emu[s] = {"sha": sha(o), "repeat_bitwise": bool(torch.equal(decode(EMU, seed_latent(s)), o)),
                      "vs_R": cmp(o, R[s]) if s in R else None, "cand_vs_main_emu": cmp(on_out[s], o)}
            log("emu seed", s, json.dumps(emu[s]))
        del EMU
    finally:
        iv.Flux3ImageResnetBlock.forward = ORIG_FWD
        iv._resnet_block_with_input_bias = ORIG_FOLD
    res["main_emu_seeds"] = emu
    pc = []
    first_div = None
    for i, (c, m) in enumerate(zip(rec_c.calls, rec_m.calls)):
        same = torch.equal(c["out"], m["out"])
        row = {"i": i, "key_cand": c["key"], "key_main": m["key"], "bitwise": bool(same)}
        if not same:
            mx = m["out"].float().abs().max().item()
            row["max_abs"] = round((c["out"].float() - m["out"].float()).abs().max().item(), 5)
            row["rel_max"] = round(row["max_abs"] / mx, 6)
            if first_div is None:
                first_div = i
        pc.append(row)
    key_changes = [r["i"] for r in pc if r["key_cand"] != r["key_main"]]
    res["eager_per_call_vs_main_emu"] = {"n_cand": len(rec_c.calls), "n_main": len(rec_m.calls),
                                        "key_changed_calls": key_changes, "first_divergent_call": first_div,
                                        "calls": pc}
    log("per-call", json.dumps(res["eager_per_call_vs_main_emu"])[:4000])

rp = res["repeat"]
gates = {"ancestry_rc0": rc_anc == 0, "timed_loop_seq_bitwise": rp["timed_loop_seq_bitwise"],
         "latent_unchanged": rp["latent_unchanged"], "no_recompile_after_warmup": rp["no_recompile_after_warmup"],
         "no_plan_build_after_warmup": rp["no_plan_build_after_warmup"], "one_dynamo_frame": on_frames_added == 1,
         "seeds_repeat5": all(r["repeat5_bitwise"] and r["latent_unchanged"] for r in seeds),
         "fp8_ops_28": census["fp8_conv3x3_cpu_ops_per_decode"] == 28, "n_fp8_blocks_14": n_fp8_blocks == 14,
         "eager_calls_28": len(rec_c.calls) == 28, "synth_repeat": all(r["synth_repeat_bitwise"] for r in keys_tab)}
if ARM == "cand":
    e = res["eager_per_call_vs_main_emu"]
    gates["only_calls_17_23_rekeyed"] = e["key_changed_calls"] == [17, 23]
    gates["calls_0_16_bitwise"] = e["first_divergent_call"] == 17
    gates["emu_psnr_vs_cand_ge_40"] = all(v["cand_vs_main_emu"]["psnr_db"] >= 40 for v in emu.values())
res["gates"] = gates
res["batch"] = {"1": {"kernel_ms": st["median_ms"], "noise_floor": st["noise_floor"]}}
res["output"] = {"class": "reordered" if ARM == "cand" else "exact"}
res["passed"] = bool(all(gates.values()))
res["wall_s"] = round(time.time() - T0, 1)
with open(os.path.join(ART, f"result_{ARM}.json"), "w") as f:
    json.dump(res, f, default=str)
print(json.dumps(res, default=str), flush=True)
