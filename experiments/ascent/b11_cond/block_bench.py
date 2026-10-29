"""Branch 11 (idea 26): i2i reference-stream mode blocks (content_mode_blocks.image_cond) on the
prepared SM120 NVFP4 block route.

ONE process per arm: the pipeline as served (engine.json overlay: FLASHINFER FP8 attention;
torch.compile + autotune; warmup [[2048, 2048]] i2i), CUDA graphs off (block-level timing).
One i2i request (2048^2 frozen reference PNG, 16 steps) captures every image_cond block's real
step-0 inputs (x, freqs, mod, sample_bounds, timestep) plus image block 1's (control).

Modes (argv[2]):
  step0  STEP 0 facts: each cond linear's quant method, input_scale (activation global scale),
         force_dynamic_quantization, the route attributes; per-block time of the loaded route.
  bench  per-block CUDA-event time (5 warmup + 50 timed calls, median) of every cond block on
         the loaded route, plus image block 1 (control). Last line: evidence JSON with
         kernel_ms = sum of the 5 cond block medians.
  check  untimed: on the loaded route, the same 5 + 50 call sequence per cond block, every call
         compared with call 0 (bitwise) and the input checked unchanged; then (candidate tree)
         the generic route (adapters cleared = main's route) and an fp32 torch reference of the
         same block math on the dequantized NVFP4 weights; max abs / rel errors per block.
         Candidate also times/compares the attention-out variants (nfo / side / off).
"""

import glob
import hashlib
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "bench"
WARMUP, ITERS = 5, 50
INPUTS_DIR = (
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/ascent-benchmark-data/"
    "flux3-klein-nvfp4-2048/v1/inputs"
)
fails = []


def log(*a):
    print(f"[b11 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


from tensorrt_llm._torch.visual_gen.models.flux3 import primsts  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    _load_image_pixels,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
engine.pop("env", None)
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
engine["cuda_graph_config"] = {"enable": False}
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    parallel_config={"tp_size": 1},
    torch_compile_config={"enable": True, "enable_fullgraph": False, "enable_autotune": True},
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["i2i"]},
    **engine,
)
log("cond env", {k: v for k, v in os.environ.items() if k.startswith("TRTLLM_FLUX3_COND")})
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
log(f"loaded+warmup {time.time() - t0:.0f}s")
tr = pipe.transformer
cond_blocks = list(tr.content_mode_blocks["image_cond"])
img_block1 = tr.content_mode_blocks["image"][1]


def orig(b):
    return getattr(b, "_orig_mod", b)


def linears(b):
    b = orig(b)
    att = b.attn
    qkv = getattr(att, "qkv_proj", None)
    proj = {"qkv_proj": qkv} if qkv is not None else {
        "to_q": att.to_q, "to_k": att.to_k, "to_v": att.to_v}
    return {**proj, "attn_out": b.attn_out, "gate_up": b.ff.gate_up_proj, "down": b.ff.down_proj}


def route_of(b):
    b = orig(b)
    return {
        "nvfp4_qkv": getattr(b.attn, "_flux3_primsts_nvfp4_qkv", None) is not None,
        "adaln_nvfp4": bool(getattr(b, "_flux3_adaln_nvfp4", False)),
        "tail_pair": bool(getattr(b, "_sm120_tail_pair", False)),
        "side_gate_up": int(getattr(b, "_sm120_side_gate_up", 0)),
        "attn_nvfp4_out": bool(getattr(b, "_sm120_attn_nvfp4_out", False)),
        "static_qk": bool(getattr(b.attn, "_flux3_static_qk", True)),
        "fp8r": getattr(b.attn, "_flux3_sm120_fp8_qkv", None) is not None,
    }


# ---------------------------------------------------------------- STEP 0(a): layer facts
facts = {}
for j, b in enumerate(cond_blocks):
    rec = {}
    for name, lin in linears(b).items():
        r = {"quant_method": type(lin.quant_method).__name__,
             "weight_dtype": str(lin.weight.dtype), "weight_shape": list(lin.weight.shape),
             "force_dynamic_quantization": bool(getattr(lin, "force_dynamic_quantization", False)),
             "missing_input_scale_flag": bool(getattr(lin, "_flux3_nvfp4_missing_input_scale", False))}
        if getattr(lin, "input_scale", None) is not None:
            r["input_scale(global)"] = float(lin.input_scale.float().reshape(-1)[0])
        if getattr(lin, "weight_scale_2", None) is not None:
            r["weight_scale_2"] = float(lin.weight_scale_2.float().reshape(-1)[0])
        rec[name] = r
    facts[f"image_cond.{j}"] = {"linears": rec, "route": route_of(b)}
    log(f"image_cond.{j}", json.dumps(facts[f"image_cond.{j}"]))
log("image.1 route", json.dumps(route_of(img_block1)))

# ---------------------------------------------------------------- capture real inputs
captured = {}


def _clone(v):
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    if isinstance(v, (tuple, list)):
        return type(v)(_clone(e) for e in v)
    return v


def make_hook(key):
    def hook(mod, a, kw):
        if key not in captured:
            captured[key] = (_clone(a), _clone(kw))

    return hook


handles = [b.register_forward_pre_hook(make_hook(f"image_cond.{j}"), with_kwargs=True)
           for j, b in enumerate(cond_blocks)]
handles.append(img_block1.register_forward_pre_hook(make_hook("image.1"), with_kwargs=True))
pngs = sorted(glob.glob(os.path.join(INPUTS_DIR, "*.png")))
log("reference png", pngs[0] if pngs else None, len(pngs))
with open(pngs[0], "rb") as f:
    ref_bytes = f.read()
with torch.inference_mode():
    ctx = pipe._encode_prompt("Make it look like a watercolor painting", None)
    image = _load_image_pixels(ref_bytes, device=pipe.device)
    with autotune_context(pipe.device, pipe.rank, tune=False):
        t1 = time.time()
        pipe._prepare_i2i(ctx=ctx, image_cond=[image], height=2048, width=2048,
                          num_inference_steps=16, seed=7)
        torch.cuda.synchronize()
        log(f"capture request {time.time() - t1:.1f}s")
for h in handles:
    h.remove()
log("captured", sorted(captured))
for key, (a, kw) in captured.items():
    x = a[0]
    mod = a[2]
    log(key, "x", list(x.shape), str(x.dtype), "mod shapes", [list(m.shape) for m in mod],
        "freqs", [list(f.shape) for f in a[1]], "kw", {k: (list(v.shape) if isinstance(v, torch.Tensor) else v)
                                                       for k, v in kw.items()})


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def call(b, key):
    a, kw = captured[key]
    return b(*a, **kw)


def time_block(b, key, warmup=WARMUP, iters=ITERS):
    with torch.inference_mode():
        for _ in range(warmup):
            call(b, key)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            call(b, key)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
    ts.sort()
    med = statistics.median(ts)
    p10, p90 = ts[int(0.1 * (len(ts) - 1))], ts[int(0.9 * (len(ts) - 1))]
    return {"median_ms": round(med, 4), "p10": round(p10, 4), "p90": round(p90, 4),
            "noise": round((p90 - p10) / med, 4)}


def sequence(b, key, warmup=WARMUP, iters=ITERS):
    """Untimed sequence matching time_block: returns (first output, all-equal flag)."""
    x_sha = sha(captured[key][0][0])
    with torch.inference_mode():
        first = call(b, key).clone()
        same = True
        for _ in range(warmup + iters - 1):
            out = call(b, key)
            same = same and torch.equal(out, first)
        torch.cuda.synchronize()
    note(sha(captured[key][0][0]) == x_sha, f"{key}: input unchanged across the sequence")
    return first, same


def errs(a, ref):
    d = (a.float() - ref.float()).abs()
    r = ref.float().abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "max_rel_at_ref>1": float((d / r.clamp_min(1.0)).max()),
            "rel_l2": float(d.norm() / r.norm().clamp_min(1e-30))}


results = {"facts": facts}
cond_keys = [f"image_cond.{j}" for j in range(len(cond_blocks))]

if MODE in ("step0", "bench"):
    times = {}
    for j, key in enumerate(cond_keys):
        times[key] = time_block(cond_blocks[j], key)
        log("TIME", key, json.dumps(times[key]), json.dumps(route_of(cond_blocks[j])))
    times["image.1"] = time_block(img_block1, "image.1")
    log("TIME image.1 (control)", json.dumps(times["image.1"]))
    # a second pass interleaved (cond 0..4) for stability
    times2 = {key: time_block(cond_blocks[j], key) for j, key in enumerate(cond_keys)}
    log("TIME pass2", json.dumps(times2))
    total = sum(min(times[k]["median_ms"], times2[k]["median_ms"]) for k in cond_keys)
    noise = max(max(times[k]["noise"], times2[k]["noise"]) for k in cond_keys)
    results.update(times=times, times_pass2=times2)
    log("RESULTS", json.dumps(results)[:20000])
    print(json.dumps({
        "batch": {"i2i-2048": {"kernel_ms": round(total, 4), "noise_floor": round(noise, 4)}},
        "control_image1_ms": times["image.1"]["median_ms"],
        "per_block_ms": {k: min(times[k]["median_ms"], times2[k]["median_ms"]) for k in cond_keys},
        "route": route_of(cond_blocks[0]),
        "output": {"class": "mode_switch" if route_of(cond_blocks[0])["nvfp4_qkv"] else "exact"},
    }))
    sys.exit(0)

# ---------------------------------------------------------------- check
from tensorrt_llm._torch.utils import unswizzle_sf  # noqa: E402
from tensorrt_llm._torch.visual_gen.attention.attention import apply_rotary_emb  # noqa: E402

_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                     dtype=torch.float32, device="cuda")


def dequant(lin):
    w = lin.weight
    if w.dtype != torch.uint8:
        return w.float()
    n, kh = w.shape
    k = kh * 2
    lo = (w & 0xF).long()
    hi = (w >> 4).long()
    vals = torch.stack([_E2M1[lo], _E2M1[hi]], dim=-1).reshape(n, k)
    sf = unswizzle_sf(lin.weight_scale.detach(), n, k).reshape(n, k // 16)
    sf = sf.contiguous().view(torch.uint8).view(torch.float8_e4m3fn).float()
    return vals * sf.repeat_interleave(16, dim=1) * float(lin.weight_scale_2.float().reshape(-1)[0])


def fp32_block(b, key):
    """fp32 torch reference of the mode block on the dequantized NVFP4 weights (no activation
    quantization; fp32 LN, adaLN, QK-norm, RoPE, attention, SwiGLU, residual)."""
    a, kw = captured[key]
    x, freqs, mod = a[0], a[1], a[2]
    b = orig(b)
    shift, scale, gate = (m.float() for m in mod[:3])
    xf = x.float()
    h = F.layer_norm(xf, (xf.shape[-1],), eps=b.pre_norm.variance_epsilon)
    h = (1 + scale) * h + shift
    lin = linears(b)
    if "qkv_proj" in lin:
        qkv = h @ dequant(lin["qkv_proj"]).t()
        q, k, v = qkv.split(qkv.shape[-1] // 3, dim=-1)
    else:
        q, k, v = (h @ dequant(lin[n]).t() for n in ("to_q", "to_k", "to_v"))
    att = b.attn
    B, S = x.shape[:2]
    H, D = att.local_num_attention_heads, att.head_dim
    q, k, v = (t.view(B, S, H, D) for t in (q, k, v))
    eps = att.norm_q.variance_epsilon
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps) * att.norm_q.weight.float()
    k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + att.norm_k.variance_epsilon) * \
        att.norm_k.weight.float()
    cos, sin = freqs[0].float(), freqs[1].float()
    q = apply_rotary_emb(q, cos, sin).float()
    k = apply_rotary_emb(k, cos, sin).float()
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       scale=1.0 / math.sqrt(D))
    o = o.transpose(1, 2).reshape(B, S, H * D)
    attn_y = o @ dequant(lin["attn_out"]).t()
    gu = h @ dequant(lin["gate_up"]).t()
    g, u = gu.chunk(2, dim=-1)
    mlp_y = (F.silu(g) * u) @ dequant(lin["down"]).t()
    return xf + gate * (attn_y + mlp_y)


def clear_route(b):
    """Main's generic route for this block: drop every prepared adapter / static switch."""
    t = orig(b)
    saved = {}
    for name, lin in linears(b).items():
        saved[("lin", name)] = getattr(lin, "_flux3_primsts_nvfp4", None)
        lin._flux3_primsts_nvfp4 = None
    for attr in ("_flux3_primsts_nvfp4_qkv", "_flux3_primsts_nvfp4_qkv_planar"):
        saved[("attn", attr)] = getattr(t.attn, attr, None)
    t.attn._flux3_primsts_nvfp4_qkv = None
    t.attn._flux3_primsts_nvfp4_qkv_planar = False
    for attr, off in (("_flux3_adaln_nvfp4", False), ("_sm120_tail_pair", False),
                      ("_sm120_side_gate_up", 0), ("_sm120_attn_nvfp4_out", False)):
        saved[("blk", attr)] = getattr(t, attr, off)
        setattr(t, attr, off)
    return saved


def restore_route(b, saved):
    t = orig(b)
    lins = linears(b)
    for (kind, name), val in saved.items():
        if kind == "lin":
            lins[name]._flux3_primsts_nvfp4 = val
        elif kind == "attn":
            setattr(t.attn, name, val)
        else:
            setattr(t, name, val)


torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
prepared_tree = route_of(cond_blocks[0])["nvfp4_qkv"]
blocks_out = {}
for j, key in enumerate(cond_keys):
    b = cond_blocks[j]
    rec = {"route": route_of(b)}
    out, same = sequence(b, key)
    rec["repeat_bitwise_55"] = same
    rec["out_sha"] = sha(out)
    rec["x_sha"] = sha(captured[key][0][0])
    note(same, f"{key}: 55 repeated calls bitwise equal to call 0")
    with torch.inference_mode():
        ref = fp32_block(b, key)
    rec["loaded_vs_fp32"] = errs(out, ref)
    if prepared_tree:
        saved = clear_route(b)
        gen, gsame = sequence(b, key, warmup=1, iters=4)
        restore_route(b, saved)
        rec["generic_out_sha"] = sha(gen)
        rec["generic_repeat_bitwise"] = gsame
        rec["generic(main)_vs_fp32"] = errs(gen, ref)
        rec["prepared_vs_generic(main)"] = errs(out, gen)
        again, _ = sequence(b, key, warmup=0, iters=2)
        rec["prepared_after_generic_bitwise"] = bool(torch.equal(again, out))
        note(rec["prepared_after_generic_bitwise"], f"{key}: prepared route bitwise after flips")
        # attention-out variants
        t = orig(b)
        base = (t._sm120_attn_nvfp4_out, t._sm120_side_gate_up)
        var = {}
        nfo_ok = primsts._sm120_attn_nvfp4_out_eligible  # noqa: F841
        for vname, flags in (("nfo", (True, 3)), ("side", (False, 3)), ("off", (False, 0))):
            t._sm120_attn_nvfp4_out, t._sm120_side_gate_up = flags
            vo, vsame = sequence(b, key, warmup=1, iters=4)
            var[vname] = {"sha": sha(vo), "repeat": vsame, "vs_fp32": errs(vo, ref),
                          "vs_generic": errs(vo, gen), "time": time_block(b, key)}
            log("VARIANT", key, vname, json.dumps(var[vname]))
        t._sm120_attn_nvfp4_out, t._sm120_side_gate_up = base
        rec["variants"] = var
        note(var["side"]["sha"] == var["off"]["sha"], f"{key}: side == off bitwise (same math)")
    del ref
    blocks_out[key] = rec
    log("CHECK", key, json.dumps(rec))
results["blocks"] = blocks_out
log("RESULTS", json.dumps(results)[:30000])
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "prepared_tree": prepared_tree,
    "compared_against": "per cond block, real step-0 i2i inputs: 55-call sequence bitwise vs call 0; "
    "main's generic route (adapters cleared, same process) and an fp32 torch reference on the "
    "dequantized NVFP4 weights",
    "summary": {k: {kk: v.get(kk) for kk in ("loaded_vs_fp32", "generic(main)_vs_fp32",
                                              "prepared_vs_generic(main)", "repeat_bitwise_55")}
                for k, v in blocks_out.items()},
}))
