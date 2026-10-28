"""Branch 108 (idea 67) probe: exact 3-kernel TE RMSNorm, all steps in ONE process.

argv[-1] = checkpoint root.
 (a) exhaustive Triton rsqrt vs torch.rsqrt over every fp32 in [f32(1e-6), +inf]
 (b) random op tests fused vs eager body, bitwise, real shapes, >=1e5 rows per shape
 (c) base (switch 0) and fused (switch 1) embedders, each warmup_buckets()+encode(['warmup']);
     fused/eager call counts during graph capture
 (d) per-call hook over one non-graphed compiled T=80 forward of the fused instance
 (e) ctx sha256 over 12 T=80 prompts, both instances
 (f) interleaved graphed encode timing B, F, B2 (CUDA events)
 (g) matching untimed sequence: F == B per iteration and F == F(iter 0 of the prompt)
Everything runs inside one torch.inference_mode() block (pitfall 1068).
"""
import hashlib
import json
import math
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.getcwd(), "experiments/ascent"))
from te_common import load_embedder  # noqa: E402

import tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.text_encoder as te_mod  # noqa: E402
import tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.text_rms_norm as fn_mod  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.text_rms_norm import (  # noqa: E402
    _rsqrt as _prod_rsqrt,
)

CKPT = sys.argv[-1]
PROMPTS = [
    "A cat sitting on a red sofa.",
    'A sign that says "OPEN 24/7" with \'neon\' letters\nand a newline.',
    "Café au lait, 東京の夜景, émojis 🎨🚀, Ünïcödé.",
    "A red fox standing in fresh snow at sunrise",
    "A bowl of ramen with a soft-boiled egg on a wooden table, studio lighting",
    "A futuristic city skyline at night with neon signs reflected in rain puddles",
    "An old lighthouse on a rocky coast during a storm, dramatic clouds",
    "A watercolor painting of a hummingbird drinking from a purple flower",
    "A vintage red bicycle leaning against a yellow brick wall",
    "A close-up portrait of an elderly fisherman with a weathered face",
    "A cozy library with tall bookshelves and a fireplace, warm light",
    "Three green apples and a glass of water on a white tablecloth",
]
T0 = time.time()
RES = {}


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


@triton.jit
def _rsqrt_probe(v_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(v_ptr + offs, mask=m, other=1.0)
    tl.store(o_ptr + offs, _prod_rsqrt(v), mask=m)


def step_a():
    lo = int(torch.tensor([1e-6], dtype=torch.float32).view(torch.int32).item())
    hi = 0x7F800000  # +inf inclusive
    total = hi - lo + 1
    chunk = 1 << 27
    bad = 0
    first_bad = None
    for s in range(lo, hi + 1, chunk):
        e = min(s + chunk, hi + 1)
        bits = torch.arange(s, e, dtype=torch.int32, device="cuda")
        v = bits.view(torch.float32)
        ref = torch.rsqrt(v)
        out = torch.empty_like(v)
        n = v.numel()
        _rsqrt_probe[(triton.cdiv(n, 1024),)](v, out, n, BLOCK=1024, num_warps=4)
        diff = ref.view(torch.int32) != out.view(torch.int32)
        nb = int(diff.sum())
        if nb and first_bad is None:
            i = int(diff.nonzero()[0])
            first_bad = (float(v[i]), float(ref[i]), float(out[i]))
        bad += nb
        del bits, ref, out, diff
    log(f"(a) rsqrt exhaustive: {total} values, {bad} mismatches, first {first_bad}")
    return {"values": total, "mismatches": bad, "first_bad": first_bad, "bitwise": bad == 0}


def step_b(weights, eps):
    gen = torch.Generator(device="cuda").manual_seed(1234)
    shapes = [
        ("ln_T80", (1, 80, 2560), "ln"), ("ln_T160", (1, 160, 2560), "ln"),
        ("q_T80", (1, 80, 32, 128), "q"), ("q_T160", (1, 160, 32, 128), "q"),
        ("k_T80", (1, 80, 8, 128), "k"), ("k_T160", (1, 160, 8, 128), "k"),
    ]
    out = {}
    ok = True
    for name, shape, wk in shapes:
        d = shape[-1]
        rows_per = math.prod(shape[:-1])
        ncalls = math.ceil(100000 / rows_per)
        per_type = min(math.ceil(1000 / ncalls), rows_per // 4)
        rows = mism = calls_bad = 0
        outl = 0
        for c in range(ncalls):
            x = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=gen)
            x2 = x.view(-1, d)
            idx = torch.randperm(rows_per, device="cuda", generator=gen)[: 3 * per_type]
            x2[idx[:per_type]] *= 1e3
            x2[idx[per_type: 2 * per_type]] *= 1e-3
            x2[idx[2 * per_type:]] = 0
            outl += 3 * per_type
            if c % 2 == 0:
                w = weights[wk]
            else:
                w = torch.randn(d, dtype=torch.bfloat16, device="cuda", generator=gen)
            assert fn_mod.supports(x, w)
            ref = te_mod._text_rms_norm_eager(x, w, eps)
            got = fn_mod.rms_norm(x, w, eps)
            nb = int((ref.view(torch.int16) != got.view(torch.int16)).sum())
            mism += nb
            calls_bad += nb > 0
            rows += rows_per
        # also through the registered op
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=gen)
        via_op = torch.equal(
            torch.ops.trtllm.flux3_text_rms_norm(x, weights[wk], eps).view(torch.int16),
            te_mod._text_rms_norm_eager(x, weights[wk], eps).view(torch.int16))
        out[name] = {"rows": rows, "calls": ncalls, "outlier_rows": outl,
                     "mismatched_elems": mism, "bad_calls": calls_bad, "via_op_equal": via_op}
        ok = ok and mism == 0 and via_op
        log(f"(b) {name}: {out[name]}")
    torch.cuda.synchronize()
    return ok, out


# capture-aware counting wrappers around the two op paths
_orig_fused = fn_mod.rms_norm
_orig_eager = te_mod._text_rms_norm_eager
COUNTS = {}


def _count(path):
    key = f"{path}_{'capture' if torch.cuda.is_current_stream_capturing() else 'run'}"
    COUNTS[key] = COUNTS.get(key, 0) + 1


def _fused_counting(x, w, eps):
    _count("fused")
    return _orig_fused(x, w, eps)


def _eager_counting(x, w, eps):
    _count("eager")
    return _orig_eager(x, w, eps)


fn_mod.rms_norm = _fused_counting
te_mod._text_rms_norm_eager = _eager_counting


def encode_one(emb, p):
    return emb.encode([p])[0]


def sha(ctxs):
    h = hashlib.sha256()
    for c in ctxs:
        h.update(c.view(torch.int16).cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    passed = True
    RES["rsqrt"] = step_a()
    if not RES["rsqrt"]["bitwise"]:
        fn_mod.ATEN_RSQRT = True
        log("Triton rsqrt not bitwise: using ATen-rsqrt fallback for the rest of the probe")
    RES["aten_rsqrt_fallback"] = fn_mod.ATEN_RSQRT

    # (c) base instance
    os.environ["TRTLLM_FLUX3_TEXT_FUSED_NORM"] = "0"
    log("loading base embedder (switch 0)")
    base = load_embedder(CKPT)
    lm = base.model.model.language_model
    l0 = lm.layers[0]
    weights = {"ln": l0.input_layernorm.weight.detach(), "q": l0.self_attn.q_norm.weight.detach(),
               "k": l0.self_attn.k_norm.weight.detach()}
    eps = float(l0.input_layernorm.variance_epsilon)
    RES["eps"] = eps
    log(f"eps {eps}, norm modules {[type(m).__name__ for m in (l0.input_layernorm, l0.self_attn.q_norm)]}")

    ok_b, RES["op_tests"] = step_b(weights, eps)
    passed &= ok_b

    COUNTS.clear()
    t = time.time()
    base.warmup_buckets()
    encode_one(base, "warmup")
    torch.cuda.synchronize()
    RES["base_warm_s"] = round(time.time() - t, 1)
    RES["base_counts"] = dict(COUNTS)
    log(f"(c) base warm {RES['base_warm_s']}s counts {COUNTS} graphs {list(base._graphs)}")
    passed &= COUNTS.get("eager_capture", 0) == 128 and COUNTS.get("fused_capture", 0) == 0

    os.environ["TRTLLM_FLUX3_TEXT_FUSED_NORM"] = "1"
    log("loading fused embedder (switch 1)")
    fused = load_embedder(CKPT)
    COUNTS.clear()
    t = time.time()
    fused.warmup_buckets()
    encode_one(fused, "warmup")
    torch.cuda.synchronize()
    RES["fused_warm_s"] = round(time.time() - t, 1)
    RES["fused_counts"] = dict(COUNTS)
    log(f"(c) fused warm {RES['fused_warm_s']}s counts {COUNTS} graphs {list(fused._graphs)}")
    passed &= COUNTS.get("fused_capture", 0) == 128 and COUNTS.get("eager_capture", 0) == 0
    assert 80 in base._graphs and 80 in fused._graphs

    # (d) per-call hook over a non-graphed compiled T=80 forward of the fused instance
    hook = {"calls": 0, "bad": 0, "shapes": set()}

    def _fused_check(x, w, e):
        o = _orig_fused(x, w, e)
        r = _orig_eager(x, w, e)
        hook["calls"] += 1
        hook["shapes"].add(tuple(x.shape))
        hook["bad"] += int(not torch.equal(o.view(torch.int16), r.view(torch.int16)))
        return o

    fn_mod.rms_norm = _fused_check
    ids, mask = fused._tokenize(PROMPTS[3])
    assert ids.shape[1] == 80
    ng = fused._forward_tensors(ids.cuda(), mask.cuda()).clone()
    torch.cuda.synchronize()
    fn_mod.rms_norm = _fused_counting
    graphed = encode_one(fused, PROMPTS[3])
    RES["hook"] = {"calls": hook["calls"], "bad": hook["bad"],
                   "shapes": sorted(str(s) for s in hook["shapes"]),
                   "nongraphed_eq_graphed": torch.equal(ng, graphed)}
    log(f"(d) {RES['hook']}")
    passed &= hook["calls"] == 128 and hook["bad"] == 0

    # (e) ctx equality
    cb = [encode_one(base, p).clone() for p in PROMPTS]
    cf = [encode_one(fused, p).clone() for p in PROMPTS]
    shapes = {tuple(c.shape) for c in cb + cf}
    eq = [torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(cb, cf)]
    RES["ctx"] = {"shapes": sorted(str(s) for s in shapes), "equal": f"{sum(eq)}/{len(eq)}",
                  "sha_base": sha(cb), "sha_fused": sha(cf)}
    log(f"(e) {RES['ctx']}")
    passed &= all(eq)

    arms = [("B", base), ("F", fused), ("B2", base)]
    WARM, ITERS = 6, 72

    def untimed(tag):
        bad = {"F_vs_B": 0, "F_vs_F0": 0, "B2_vs_B": 0, "vs_probe_ref": 0}
        f0 = {}
        for i in range(WARM + ITERS):
            p = PROMPTS[i % len(PROMPTS)]
            outs = {a: encode_one(e, p).clone() for a, e in arms}
            if i < WARM:
                continue
            k = i % len(PROMPTS)
            f0.setdefault(k, outs["F"])
            bad["F_vs_B"] += not torch.equal(outs["F"], outs["B"])
            bad["F_vs_F0"] += not torch.equal(outs["F"], f0[k])
            bad["B2_vs_B"] += not torch.equal(outs["B2"], outs["B"])
            bad["vs_probe_ref"] += not torch.equal(outs["B"], cb[k])
        log(f"untimed[{tag}] {bad}")
        return bad

    # smoke (2 rounds) before timing
    smoke = {"F_vs_B": 0}
    for i in range(2):
        for k, p in enumerate(PROMPTS[:2]):
            smoke["F_vs_B"] += not torch.equal(encode_one(fused, p), encode_one(base, p))
    log(f"smoke {smoke}")
    passed &= smoke["F_vs_B"] == 0

    # (f) timing
    ev = {a: [] for a, _ in arms}
    wall = {a: [] for a, _ in arms}
    for i in range(WARM + ITERS):
        p = PROMPTS[i % len(PROMPTS)]
        for a, e in arms:
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            f = torch.cuda.Event(enable_timing=True)
            w0 = time.perf_counter()
            s.record()
            encode_one(e, p)
            f.record()
            f.synchronize()
            w1 = time.perf_counter()
            if i >= WARM:
                ev[a].append(s.elapsed_time(f))
                wall[a].append((w1 - w0) * 1e3)
        if i % 20 == 0:
            log(f"timing iter {i}: " + ", ".join(f"{a} {ev[a][-1]:.3f}" for a in ev if ev[a]))

    def stats(xs):
        s = sorted(xs)
        med = statistics.median(xs)
        return {"median": round(med, 4), "p10": round(s[int(0.1 * len(s))], 4),
                "p90": round(s[int(0.9 * len(s))], 4),
                "noise": round((s[int(0.9 * len(s))] - s[int(0.1 * len(s))]) / med, 4)}

    RES["timing_event_ms"] = {a: stats(v) for a, v in ev.items()}
    RES["timing_wall_ms"] = {a: stats(v) for a, v in wall.items()}
    mb, mf, mb2 = (RES["timing_event_ms"][a]["median"] for a in ("B", "F", "B2"))
    RES["F_minus_B_ms"] = round(mf - mb, 4)
    RES["B2_minus_B_ms"] = round(mb2 - mb, 4)
    log(f"(f) event {RES['timing_event_ms']} F-B {RES['F_minus_B_ms']} B2-B {RES['B2_minus_B_ms']}")
    log(f"(f) wall {RES['timing_wall_ms']}")

    # (g) matching untimed sequence
    RES["untimed"] = untimed("g")
    passed &= all(v == 0 for v in RES["untimed"].values())
    RES["final_counts"] = dict(COUNTS)
    RES["passed"] = bool(passed)
    RES["compared"] = ("fused (switch 1) vs eager body / base instance (switch 0) in one process; "
                       "bitwise on op outputs and ctx")
    print(json.dumps(RES), flush=True)


with torch.inference_mode():
    main()
