"""Idea 227 decode pair: up[1] upsample as flux3_vae::subpixel_up_conv (one cuDNN FE stride-2 dgrad
with an fp32 bias epilogue) vs main's sub-pixel fprop + interleave. Runs UNCHANGED in both trees.

usage: python experiments/ascent/b185_pair.py <checkpoint_root> [--warmup 10] [--iters 60]

Served config in both arms: gn_check.build_vae -> set_vae_decoder_trunk_dtype(bf16) ->
compile_decoder(fp8_convs=True) (prepare_fp8_convs call counted; 28 convs must be prepared).
Latent 1x128x64x64 fp32 from torch.Generator('cuda').manual_seed(0), pristine copy kept.
- timed: --warmup decodes, then --iters decodes, one CUDA event pair each (median/p10/p90).
- untimed repeat sequence, same warmup + iteration count: every output bitwise vs iteration 0,
  latent unchanged before every call; Dynamo counters and cuDNN FE graph counts (fp8_conv and,
  when the tree has it, subpixel_conv) unchanged after warmup.
- census (one warm decode, torch.profiler kineto trace): every kernel attributed to the CPU op that
  launched it (fp8_conv3x3, subpixel_up_conv, aten conv ops, or none = Inductor Triton); regions
  between consecutive fp8_conv3x3 calls listed: #10-#11 (up[3].upsample), #16-#17 (up[2].upsample +
  fold), #22-#23 (up[1].upsample + up[0].block[0].norm1 stats/apply), #23-#24 (up[0].block[0]
  norm2 + nin_shortcut + its bias add); nchwToNhwc / nhwcToNchw counts.
- accuracy seeds 0-2 vs the fp32 eager decoder R (PSNR, max abs, mean abs, u8 diff).
- fp32 exactness: sha256 of the fp32 eager (NCHW) decode of seed 0 and of the fp32 compiled
  decoder's Dynamo FX graph code (capture backend), to be compared across arms.
Declared gates (passed holds only these): repeat bitwise, latent unchanged, no recompile and no
cuDNN FE graph build after warmup, 28 convs prepared, 28 fp8_conv3x3 calls per decode.
Last stdout line: pair evidence JSON (batch.1.kernel_ms = timed decode median).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--iters", type=int, default=60)
args = p.parse_args()
WARMUP, ITERS = args.warmup, args.iters
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
git_info = {"log": head, "ancestor_fb730d34_rc": rc_anc, "cwd": os.getcwd()}
log("git", json.dumps(git_info))

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import fp8_conv as fc  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import set_vae_decoder_trunk_dtype  # noqa: E402

try:
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import subpixel_conv as spc
except Exception:  # noqa: BLE001  (main's tree has no subpixel_conv)
    spc = None
iv = gn_check.iv
log("tree has subpixel_conv:", spc is not None)

prep_calls = []
_orig_prep = fc.prepare_fp8_convs


def _counting_prep(dec):
    n = _orig_prep(dec)
    prep_calls.append(n)
    return n


fc.prepare_fp8_convs = _counting_prep

gen = torch.Generator(device="cuda").manual_seed(0)
Z = torch.randn(1, 128, 64, 64, device="cuda", generator=gen)
Z0 = Z.clone()


def seed_latent(s):
    g = torch.Generator(device="cuda").manual_seed(s)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g)


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


def counters():
    import torch._dynamo.utils as du

    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break", "stats")}


def n_graphs():
    return len(fc._graphs) + (len(spc._graphs) if spc is not None else 0)


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": round(med, 4), "p10_ms": round(p10, 4), "p90_ms": round(p90, 4),
            "noise_floor": round((p90 - p10) / med, 5)}


def to_u8(x):
    return gn_check.to_u8(x.float())


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    mse = ((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)) ** 2).mean().item()
    du = (to_u8(a).int() - to_u8(b).int()).abs()
    return {"psnr_db": float("inf") if mse == 0 else round(10 * torch.log10(torch.tensor(4.0 / mse)).item(), 3),
            "max_abs": round(d.max().item(), 5), "mean_abs": round(d.mean().item(), 7),
            "u8_max_diff": du.max().item(), "u8_frac_diff": round((du > 0).float().mean().item(), 6),
            "bitwise": bool(torch.equal(a, b))}


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def strip(code):
    return "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("#"))


res = {"git": git_info, "warmup": WARMUP, "iters": ITERS, "tree_has_subpixel_conv": spc is not None}

# ---------------------------------------------------------------- fp32 exactness rows
graphs = []


def cap_backend(gm, example_inputs):
    graphs.append(gm.code)
    return gm.forward


v32 = build(False, {}, cap_backend)
decode(v32)
torch.cuda.synchronize()
code32 = "\n# ==== next graph ====\n".join(strip(g) for g in graphs)
res["fp32_compiled_fx"] = {"n_graphs": len(graphs), "sha256": hashlib.sha256(code32.encode()).hexdigest()}
del v32
R = {}
ref_vae = build(trunk_bf16=False, compile_kw=None)
for s in range(3):
    R[s] = decode(ref_vae, seed_latent(s)).clone()
res["fp32_eager_nchw_seed0_sha256"] = sha(R[0])
res["fp32_eager_up1_upsample_flag"] = getattr(ref_vae.decoder.up[1].upsample, "use_fused_bias_op", None)
del ref_vae
torch.cuda.empty_cache()
import torch._dynamo.utils as _du  # noqa: E402

_du.counters.clear()
log("fp32 rows", json.dumps({k: res[k] for k in ("fp32_compiled_fx", "fp32_eager_nchw_seed0_sha256")}))

# ---------------------------------------------------------------- served FP8 decoder
ON = build(True, {"fp8_convs": True})
n_prepared = prep_calls[-1] if prep_calls else 0
n_fp8_blocks = sum(1 for m in ON.decoder.modules() if isinstance(m, iv.Flux3ImageResnetBlock) and m._fp8)
log(f"prepare_fp8_convs prepared {n_prepared} convs ({n_fp8_blocks} FP8 ResnetBlocks)")
t0 = time.time()
for _ in range(WARMUP):
    decode(ON)
torch.cuda.synchronize()
log(f"warm {WARMUP} decodes in {time.time() - t0:.1f}s")
c0, g0 = counters(), n_graphs()
log("dynamo counters after warmup", json.dumps(c0))
ts = []
for i in range(ITERS):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    decode(ON)
    e.record()
    e.synchronize()
    ts.append(s.elapsed_time(e))
    if i % 20 == 0:
        log(f"iter {i} {ts[-1]:.3f} ms")
st = stats(ts)
log("timing", json.dumps(st))

# untimed repeat sequence, same warmup + iteration count
for _ in range(WARMUP):
    decode(ON)
first, rep, inp = None, True, True
for _ in range(ITERS):
    inp &= bool(torch.equal(Z, Z0))
    out = decode(ON)
    if first is None:
        first = out.clone()
    elif not torch.equal(out, first):
        rep = False
torch.cuda.synchronize()
c1, g1 = counters(), n_graphs()
no_recompile = c1 == c0
no_new_graphs = g1 == g0
log("repeat", rep, "inputs", inp, "no_recompile", no_recompile, "no_new_fe_graphs", no_new_graphs)

acc = []
outs = {}
for s in range(3):
    outs[s] = decode(ON, seed_latent(s)).clone()
    acc.append({"seed": s, **cmp(outs[s], R[s])})
log("served vs fp32 R", json.dumps(acc))
res["served_seed0_sha256"] = sha(outs[0])

# ---------------------------------------------------------------- census (kineto trace attribution)
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    decode(ON)
    torch.cuda.synchronize()
tdir = tempfile.mkdtemp()
tp = os.path.join(tdir, "trace.json")
prof.export_chrome_trace(tp)
tr = json.load(open(tp))
evs = tr["traceEvents"] if isinstance(tr, dict) else tr
kern = sorted([e for e in evs if e.get("cat") == "kernel"], key=lambda e: e["ts"])
rt = {e["args"]["correlation"]: e for e in evs
      if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
INTEREST = ("flux3_vae::fp8_conv3x3", "flux3_vae::subpixel_up_conv", "aten::cudnn_convolution_transpose",
            "aten::cudnn_convolution", "aten::convolution", "aten::_convolution", "aten::conv_transpose2d")
ops = [e for e in evs if e.get("cat") == "cpu_op" and e.get("name") in INTEREST]


def owner(k):
    r = rt.get(k.get("args", {}).get("correlation"))
    if r is None:
        return "?"
    enc = [o for o in ops if o["tid"] == r["tid"] and o["ts"] <= r["ts"] <= o["ts"] + o.get("dur", 0)]
    if not enc:
        return "-"
    custom = [o for o in enc if o["name"].startswith("flux3_vae")]
    if custom:
        return custom[0]["name"]
    return min(enc, key=lambda o: o.get("dur", 0))["name"]


seq = [(k["name"], round(k.get("dur", 0) / 1e3, 4), owner(k)) for k in kern]
fp8_idx = []
prev_fp8 = False
for i, (nm, d, ow) in enumerate(seq):
    is_fp8 = ow == "flux3_vae::fp8_conv3x3"
    if is_fp8 and not prev_fp8:
        fp8_idx.append(i)
    prev_fp8 = is_fp8
fp8_calls = len(fp8_idx)


def region(a, b):
    """kernels after the a-th fp8 conv call (1-based) and before the b-th."""
    if len(fp8_idx) < b:
        return None
    i0 = fp8_idx[a - 1]
    while i0 < len(seq) and seq[i0][2] == "flux3_vae::fp8_conv3x3":
        i0 += 1
    rows = seq[i0:fp8_idx[b - 1]]
    return {"total_ms": round(sum(r[1] for r in rows), 4), "n": len(rows),
            "kernels": [[r[0][:110], r[1], r[2]] for r in rows]}


cnt, dur = Counter(), Counter()
for nm, d, _ in seq:
    cnt[nm] += 1
    dur[nm] += d
census = {
    "total_kernels": len(seq), "kernel_sum_ms": round(sum(r[1] for r in seq), 4),
    "fp8_conv3x3_calls": fp8_calls,
    "subpixel_up_conv_kernels": [[r[0][:110], r[1]] for r in seq if r[2] == "flux3_vae::subpixel_up_conv"],
    "interleave_kernels": [[r[0][:110], r[1]] for r in seq if "select_slice_stack" in r[0]],
    "nchwToNhwc": sum(v for k, v in cnt.items() if "nchwToNhwc" in k),
    "nhwcToNchw": sum(v for k, v in cnt.items() if "nhwcToNchw" in k),
    "R10_11_up3_upsample": region(10, 11), "R16_17_up2_upsample_fold": region(16, 17),
    "R22_23_up1_upsample_up0b0_norm1": region(22, 23), "R23_24_up0b0_norm2_nin": region(23, 24),
    "top": [(k[:110], cnt[k], round(dur[k], 4)) for k, _ in dur.most_common(25)],
}
log("census", json.dumps(census)[:6000])
res["census"] = census
res["counters_after_warmup"] = c0
res["counters_end"] = c1
frames_ok = c1.get("frames", {}).get("ok")
gb = sum(c1.get("graph_break", {}).values()) if c1.get("graph_break") else 0
res["dynamo"] = {"frames_ok": frames_ok, "graph_breaks": gb,
                 "unique_graphs": c1.get("stats", {}).get("unique_graphs")}
res["fe_graphs"] = {"fp8": fc.graph_table(), "subpixel": spc.graph_table() if spc is not None else None}

gates = {"repeat_bitwise": rep, "inputs_unchanged": inp, "no_recompile_after_warmup": no_recompile,
         "no_cudnn_fe_graph_build_after_warmup": no_new_graphs,
         "n_prepared_28": n_prepared == 28 and n_fp8_blocks == 14, "fp8_conv3x3_calls_per_decode_28": fp8_calls == 28}
res.update({
    "batch": {"1": {"kernel_ms": st["median_ms"], "step_ms": st["median_ms"], "noise_floor": st["noise_floor"]}},
    "output": {"class": "reordered" if spc is not None else "exact",
               "error_vs_fp32_R": {"psnr_min_db": min(a["psnr_db"] for a in acc),
                                   "max_abs": max(a["max_abs"] for a in acc),
                                   "mean_abs_max": max(a["mean_abs"] for a in acc),
                                   "u8_max_diff": max(a["u8_max_diff"] for a in acc)}},
    "timing": st, "gates": gates, "n_prepared": n_prepared, "served_vs_fp32_R": acc,
    "passed": bool(all(gates.values())), "wall_s": round(time.time() - T0, 1)})
print(json.dumps(res))
