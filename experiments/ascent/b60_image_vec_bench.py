"""Idea 96 microbenchmark: FLUX.3 flow-map image-stream vec path, per-step (main) vs step-batched.

No checkpoint: builds a Flux3Model (flow_map=True, hidden 3072, 24 heads, depth 1) whose
modules are all BF16 (static-NVFP4 config excluding every module, as the served checkpoint
leaves time_in / ref_time_in / modulations / final layer in BF16). Random weights
(std 1/sqrt(fan_in)). Inputs packed with the pipeline's pack(): 4096 image tokens (1024^2
after 2x2 patching), T=80 text tokens.

Per request (16 steps) the image-stream vec work is:
  per-step (main):  16 x [ref_time_in, time_in (M=1), emb_in (M=4096), early + single image
                    modulations (M=1), vec cat-expand to [1,4096,D], final adaLN row (M=1)]
  batched (branch): 1 x prepare_flow_map_image (ref_time_in, time_in, 2 modulations,
                    final adaLN at M=16) + 16 x [emb_in (M=4096), row views]
modes:
  bench : kernel_ms per request of this tree's path (batched when the tree has
          prepare_flow_map_image and TRTLLM_FLUX3_IMAGE_VEC_STEP_BATCH != 0). On the branch
          also an interleaved in-process A/B/A. CUDA events, median.
  check : untimed sequence matching the timed one; batched vs per-step per step rows within
          the declared tolerance (plus both vs an fp32 reference for scale); every iteration
          bitwise vs iteration 0; inputs unchanged; row layout (shape/stride) equal to main's.
  kprof : torch.profiler (CUPTI) kernel table per request for both paths.
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
STEPS = 16
N_IMG = 4096
T_TXT = 80
ITERS = int(os.environ.get("B60_ITERS", "60"))
WARMUP = 5
# Declared before running (reordered: cuBLAS picks other kernels at M=16 than the M=1 split-K
# GEMVs): per step and per row tensor, rel-L2 ||batched - per_step|| / ||per_step|| <= 1e-2
# and max |diff| <= 3% of max |per_step|.
TOL_REL_L2 = 1e-2
TOL_MAX_FRAC = 3e-2
FIELDS = ("vec", "early_shift", "early_scale", "early_gate", "single_shift", "single_scale",
          "single_gate", "final_shift", "final_scale")


def log(*a):
    print(*a, flush=True)


def build():
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3Model
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    pretrained = SimpleNamespace(
        flow_map=True,
        hidden_size=3072,
        num_heads=24,
        depth=1,
        depth_single_blocks=1,
        context_in_dim=20480,
        mlp_ratio=3.0,
    )
    cfg = DiffusionModelConfig(
        pretrained_config=pretrained,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, exclude_modules=["*"]),
        skip_create_weights_in_init=True,
    )
    with torch.device("cuda"):
        model = Flux3Model(cfg)
    model = model.cuda().eval()
    g = torch.Generator(device="cuda").manual_seed(1234)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g, device="cuda") / p.shape[1] ** 0.5)
            else:
                p.fill_(1.0)
    return model


def make_inputs(model, seed=7):
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import pack
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import prc_txt

    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.randn(T_TXT, model.context_in_dim, generator=g, device="cuda").to(torch.bfloat16)
    _, text_ids = prc_txt(ctx)
    img = torch.randn(N_IMG, 128, generator=g, device="cuda").to(torch.bfloat16)
    side = int(N_IMG ** 0.5)
    img_ids = torch.stack(
        [torch.zeros(N_IMG), torch.arange(N_IMG) // side, torch.arange(N_IMG) % side,
         torch.zeros(N_IMG)],
        dim=1,
    ).to("cuda", torch.long)
    packed = pack(
        [{"text": (ctx, text_ids), "image": (img, img_ids)}],
        {"text": model.context_in_dim, **model.in_channels},
    )
    inputs = packed.kwargs()
    inputs["_flux3_uniform_timesteps"] = True
    return inputs


def scheduler():
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.scheduler import (
        Flux3FlowMatchScheduler,
    )

    s = Flux3FlowMatchScheduler(3.0)
    s.set_timesteps(STEPS)
    return s


def per_step_request(model, inputs, keep=False):
    """Main's per-step image-stream vec work (forward's ref + _packed_stream('image') + the
    final layer's _final_layer_vec + adaLN_modulation), exactly as forward runs it."""
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
        flux3_timestep_embedding,
    )

    sched = scheduler()
    lengths = dict(inputs["layout"])["image"]
    tref = inputs["x_text_timesteps"].new_empty(1)
    outs = []
    for i, timestep in enumerate(sched.timesteps):
        inputs["x_image_timesteps"].fill_(timestep.item())
        tref.fill_(sched.sigmas[i + 1].item())
        target = tref.to(inputs["x_text_timesteps"])
        ref = model.ref_time_in(flux3_timestep_embedding(target, 256, out_dtype=target.dtype))
        state = model._packed_stream("image", inputs, lengths, ref)
        vec = model._final_layer_vec(state["vec"], inputs, lengths)
        mod = model.final_layer["image"].adaLN_modulation(vec)
        if keep:
            row = {f: state[f] for f in FIELDS[1:7]}
            row["vec"] = vec
            row["final_shift"], row["final_scale"] = mod.chunk(2, dim=-1)
            row["tokens"] = state["tokens"]
            outs.append(row)
    return outs


def batched_request(model, inputs, keep=False):
    sched = scheduler()
    ts = inputs["x_image_timesteps"]
    targets = sched.sigmas[1 : len(sched.timesteps) + 1].to(
        device=ts.device, dtype=inputs["x_text_timesteps"].dtype
    )
    image_ts = sched.timesteps.to(device=ts.device, dtype=ts.dtype)
    prep = model.prepare_flow_map_image(image_ts, targets, **inputs)
    assert prep is not None, "prepare_flow_map_image fell back"
    lengths = dict(inputs["layout"])["image"]
    outs = []
    for i in range(STEPS):
        inputs["x_image_timesteps"].fill_(sched.timesteps[i].item())
        step = model.flow_map_image_step(prep, i)
        state = model._image_step_stream({**inputs, **step}, lengths)
        if keep:
            row = {f: state[f] for f in FIELDS[1:7]}
            row["final_shift"], row["final_scale"] = state["final_mod"].chunk(2, dim=-1)
            row["tokens"] = state["tokens"]
            outs.append(row)
    return outs


def batched_vec_rows(model, inputs):
    """The [N, D] vec rows prepare_flow_map_image feeds its GEMMs (for the vec comparison)."""
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
        flux3_timestep_embedding,
    )

    sched = scheduler()
    ts = inputs["x_image_timesteps"]
    target = sched.sigmas[1 : STEPS + 1].to(device=ts.device, dtype=inputs["x_text_timesteps"].dtype)
    image_ts = sched.timesteps.to(device=ts.device, dtype=ts.dtype)
    ref = model.ref_time_in(flux3_timestep_embedding(target, 256, out_dtype=target.dtype))
    vec = model.time_in(flux3_timestep_embedding(image_ts, 256, out_dtype=torch.bfloat16))
    return vec + ref


def fp32_rows(model, inputs):
    """fp32 reference of the same math (fp32 weights and activations)."""
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
        flux3_timestep_embedding,
    )

    def mlp(m, x):
        h = torch.nn.functional.linear(x, m.in_layer.weight.float())
        return torch.nn.functional.linear(torch.nn.functional.silu(h), m.out_layer.weight.float())

    sched = scheduler()
    ts = inputs["x_image_timesteps"]
    target = sched.sigmas[1 : STEPS + 1].to(device=ts.device, dtype=inputs["x_text_timesteps"].dtype)
    image_ts = sched.timesteps.to(device=ts.device, dtype=ts.dtype)
    vec = mlp(model.time_in, flux3_timestep_embedding(image_ts, 256, out_dtype=torch.float32))
    vec = vec + mlp(model.ref_time_in, flux3_timestep_embedding(target, 256, out_dtype=torch.float32))
    s = torch.nn.functional.silu(vec)
    out = []
    e = torch.nn.functional.linear(s, model.early_stream_modulations["image"].lin.weight.float())
    si = torch.nn.functional.linear(s, model.single_stream_modulations["image"].lin.weight.float())
    fi = torch.nn.functional.linear(s, model.final_layer["image"].adaLN_modulation[1].weight.float())
    for i in range(STEPS):
        row = {"vec": vec[i]}
        for stage, t in (("early", e), ("single", si)):
            for f, v in zip(("shift", "scale", "gate"), t[i].chunk(3)):
                row[stage + "_" + f] = v
        row["final_shift"], row["final_scale"] = fi[i].chunk(2)
        out.append(row)
    return out


def has_batched(model):
    return hasattr(model, "prepare_flow_map_image") and os.environ.get(
        "TRTLLM_FLUX3_IMAGE_VEC_STEP_BATCH", "1"
    ) != "0"


def time_fn(fn, iters):
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return ts


def stats(ts):
    s = sorted(ts)
    n = len(s)
    med = s[n // 2]
    return {
        "median_ms": med,
        "p10_ms": s[int(0.1 * (n - 1))],
        "p90_ms": s[int(0.9 * (n - 1))],
        "noise": (s[int(0.9 * (n - 1))] - s[int(0.1 * (n - 1))]) / med,
        "n": n,
    }


def snapshot(inputs):
    return {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}


def restore(inputs, snap):
    for k, v in snap.items():
        inputs[k].copy_(v)


def unchanged(inputs, snap, skip=("x_image_timesteps",)):
    return all(torch.equal(inputs[k], v) for k, v in snap.items() if k not in skip)


def cmp(a, b):
    x, y = a.float().flatten(), b.float().flatten()
    d = (x - y).abs()
    return {
        "rel_l2": ((x - y).norm() / y.norm().clamp_min(1e-30)).item(),
        "max_abs": d.max().item(),
        "max_frac": (d.max() / y.abs().max().clamp_min(1e-30)).item(),
    }


def layout_of(t):
    return [list(t.shape), list(t.stride()), str(t.dtype)]


def kprof(model, inputs, snap):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    reps = 3
    out = {}
    for name, fn in (("per_step_main", lambda: per_step_request(model, inputs)),
                     ("batched_branch", lambda: batched_request(model, inputs))):
        restore(inputs, snap)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
        gemm = {}
        for ev in prof.key_averages(group_by_input_shape=True):
            dev = getattr(ev, "self_device_time_total", 0)
            if ev.device_type != DeviceType.CPU or dev <= 0:
                continue
            if ev.key in ("aten::mm", "aten::addmm", "aten::linear", "aten::matmul"):
                key = f"{ev.key} {ev.input_shapes}"
                gemm[key] = [round(dev / 1e3 / reps, 4), ev.count / reps]
        kernels = {}
        for ev in prof.events():
            if ev.device_type != DeviceType.CUDA:
                continue
            k = kernels.setdefault(ev.name[:100], [0.0, 0])
            k[0] += (ev.time_range.end - ev.time_range.start) / 1e3 / reps
            k[1] += 1 / reps
        top = sorted(kernels.items(), key=lambda kv: -kv[1][0])
        res = {
            "kernel_sum_ms_per_request": round(sum(v[0] for v in kernels.values()), 4),
            "kernel_launches_per_request": round(sum(v[1] for v in kernels.values()), 1),
            "kernels": [{"name": n, "ms": round(v[0], 4), "calls": round(v[1], 1)} for n, v in top[:25]],
            "gemm_ops_ms_per_request": gemm,
        }
        log(name, json.dumps(res, indent=1))
        out[name] = res
    return out


def main():
    torch.manual_seed(0)
    log("building model")
    t0 = time.time()
    model = build()
    batched = has_batched(model)
    log(f"built in {time.time() - t0:.1f}s; batched path available: {batched}")
    inputs = make_inputs(model)
    snap = snapshot(inputs)

    with torch.inference_mode():
        restore(inputs, snap)
        for _ in range(WARMUP):
            per_step_request(model, inputs)
            if batched:
                batched_request(model, inputs)
        torch.cuda.synchronize()
        log("warmed")

        if MODE == "bench":
            fn = (lambda: batched_request(model, inputs)) if batched else (
                lambda: per_step_request(model, inputs))
            restore(inputs, snap)
            st = stats(time_fn(fn, ITERS))
            log("this tree's path:", "batched" if batched else "per-step", json.dumps(st))
            extra = {}
            if batched:
                ab = {"A_per_step": [], "B_batched": [], "A2_per_step": []}
                restore(inputs, snap)
                for it in range(ITERS):
                    for key, f in (("A_per_step", lambda: per_step_request(model, inputs)),
                                   ("B_batched", lambda: batched_request(model, inputs)),
                                   ("A2_per_step", lambda: per_step_request(model, inputs))):
                        ab[key] += time_fn(f, 1)
                    if it % 20 == 0:
                        log("interleaved iter", it)
                extra = {k: stats(v) for k, v in ab.items()}
                log("interleaved A/B/A", json.dumps(extra))
            assert unchanged(inputs, snap)
            print(json.dumps({
                "batch": {"1": {"kernel_ms": st["median_ms"], "noise_floor": st["noise"]}},
                "output": {"class": "reordered" if batched else "exact"},
                "path": "batched" if batched else "per_step",
                "steps": STEPS, "image_tokens": N_IMG,
                "interleaved": extra,
            }))
            return

        if MODE == "kprof":
            res = kprof(model, inputs, snap)
            print(json.dumps({"passed": True, **res}))
            return

        # check: untimed sequence matching the timed one (WARMUP done above, ITERS iterations,
        # restore before the sequence).
        assert batched, "check needs the branch"
        fails = []
        restore(inputs, snap)
        ref_rows = per_step_request(model, inputs, keep=True)
        ref_rows = [{k: v.clone() for k, v in r.items()} for r in ref_rows]
        vec_b = batched_vec_rows(model, inputs)
        f32 = fp32_rows(model, inputs)
        first = None
        for it in range(ITERS):
            if not unchanged(inputs, snap):
                fails.append(f"inputs changed before iter {it}")
            restore(inputs, snap)
            rows = batched_request(model, inputs, keep=True)
            rows = [{k: v.clone() for k, v in r.items()} for r in rows]
            if first is None:
                first = rows
            elif not all(torch.equal(a[k], b[k]) for a, b in zip(rows, first) for k in a):
                fails.append(f"iter {it} not bitwise equal to iter 0")
        per_step = []
        worst = {}
        for i in range(STEPS):
            rec = {"step": i}
            for f in FIELDS:
                if f == "vec":
                    a = vec_b[i]
                else:
                    a = first[i][f]
                b = ref_rows[i][f]
                c = cmp(a, b)
                c["base_vs_fp32_rel_l2"] = cmp(b, f32[i][f])["rel_l2"]
                c["batched_vs_fp32_rel_l2"] = cmp(a, f32[i][f])["rel_l2"]
                rec[f] = c
                w = worst.setdefault(f, {"rel_l2": 0.0, "max_abs": 0.0, "max_frac": 0.0,
                                         "base_vs_fp32_rel_l2": 0.0})
                for k in w:
                    w[k] = max(w[k], c[k])
                if c["rel_l2"] > TOL_REL_L2 or c["max_frac"] > TOL_MAX_FRAC:
                    fails.append(f"step {i} {f} out of tolerance {c}")
                if f != "vec" and layout_of(a) != layout_of(b):
                    fails.append(f"step {i} {f} layout {layout_of(a)} != main {layout_of(b)}")
            if not torch.equal(first[i]["tokens"], ref_rows[i]["tokens"]):
                fails.append(f"step {i} emb_in tokens differ")
            per_step.append(rec)
        log("per-step comparison", json.dumps(per_step))
        log("worst", json.dumps(worst))
        layouts = {f: layout_of(first[0][f]) for f in FIELDS[1:]}
        print(json.dumps({
            "passed": not fails,
            "failures": fails[:20],
            "output": {"class": "reordered"},
            "compared_against": "main's per-step image vec path (forward ref + _packed_stream("
                                "'image') + _final_layer_vec + adaLN) on the same inputs; fp32 "
                                "reference of the same math for scale",
            "tolerance": {"rel_l2": TOL_REL_L2, "max_frac": TOL_MAX_FRAC},
            "iterations": ITERS, "warmup": WARMUP,
            "worst": worst,
            "row_layouts": layouts,
        }))


if __name__ == "__main__":
    main()
