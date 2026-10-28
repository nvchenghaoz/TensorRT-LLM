"""Idea 55 microbenchmark: FLUX.3 flow-map text stream, per-step (main) vs step-batched.

No checkpoint: builds a Flux3Model (flow_map=True, hidden 3072, 24 heads, 5 mode blocks,
context_in_dim 20480) with a static-NVFP4 quant config that excludes every module, so the text
path is exactly the served one for the BF16-excluded txt blocks (SEPARATE_QKV, eager QK-norm,
GatedMLP + attn_out, cuBLAS BF16). Random weights (std 1/sqrt(fan_in)), txt_mode_blocks
compiled per block like Flux3Pipeline.torch_compile. Inputs packed with the pipeline's pack().

Per request the text-stream work is:
  per-step (main):  16 x [ref_time_in, time_in, txt_in, 2 modulations, 5 txt blocks at M=T]
  batched (branch): 1 x prepare_flow_map_text (ref_time_in M=16, modulations M=16, txt_in M=T,
                    5 txt blocks at M=16T)
modes:
  bench  : kernel_ms of this tree's path per request (batched if the tree has
           prepare_flow_map_text and TRTLLM_FLUX3_TEXT_STEP_BATCH != 0, else per-step);
           on the branch also an interleaved in-process A/B and A/A. CUDA events, median.
  check  : untimed sequence (same warmup/iterations), batched vs per-step per step tensors
           within the declared tolerance, iteration determinism, inputs unchanged, and an
           A (T=80) / B (T=160) / A sequence with A bitwise equal.
  profile: NVTX-marked requests of both paths for nsys.
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
STEPS = 16
T_A, T_B = 80, 160
ITERS = int(os.environ.get("B33_ITERS", "60"))
WARMUP = 5
# Declared before running (reordered: other cuBLAS tactics at M=16T vs M=T split-K, M=16 vs
# M=1 modulation GEMVs): per step, rel-L2 ||batched - per_step|| / ||per_step|| <= 1e-2 and
# max |diff| <= 3% of max |per_step| for tokens and single_*.
TOL_REL_L2 = 1e-2
TOL_MAX_FRAC = 3e-2


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
        depth=5,
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
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g, device="cuda") / p.shape[1] ** 0.5)
            else:
                p.fill_(1.0)
    for i in range(len(model.txt_mode_blocks)):
        model.txt_mode_blocks[i] = torch.compile(
            model.txt_mode_blocks[i], dynamic=None, options={"emulate_precision_casts": True}
        )
    return model


def make_inputs(model, T, seed):
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import pack
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import prc_txt

    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.randn(T, model.context_in_dim, generator=g, device="cuda").to(torch.bfloat16)
    _, text_ids = prc_txt(ctx)
    n_img = 64
    img = torch.randn(n_img, 128, generator=g, device="cuda").to(torch.bfloat16)
    img_ids = torch.stack(
        [torch.zeros(n_img), torch.arange(n_img) // 8, torch.arange(n_img) % 8, torch.zeros(n_img)],
        dim=1,
    ).to("cuda", torch.long)
    packed = pack(
        [{"text": (ctx, text_ids), "image": (img, img_ids)}],
        {"text": model.context_in_dim, **model.in_channels},
    )
    inputs = packed.kwargs()
    inputs["_flux3_uniform_timesteps"] = True
    with torch.inference_mode():
        prepared = model.prepare_static_inputs(**inputs)
    return inputs, prepared


def sigmas():
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.scheduler import (
        Flux3FlowMatchScheduler,
    )

    s = Flux3FlowMatchScheduler(3.0)
    s.set_timesteps(STEPS)
    return s


def per_step_request(model, inputs, prepared):
    """Main's forward text-stream code, once per step (forward L'for name, lengths in layout')."""
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import bounds
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
        flux3_timestep_embedding,
    )

    sched = sigmas()
    layout = inputs["layout"]
    lengths = dict(layout)["text"]
    tref = inputs["x_text_timesteps"].new_empty(1)
    outs = []
    for i, timestep in enumerate(sched.timesteps):
        tref.fill_(sched.sigmas[i + 1].item())
        target = tref.to(inputs["x_text_timesteps"])
        ref = model.ref_time_in(flux3_timestep_embedding(target, 256, out_dtype=target.dtype))
        state = model._packed_stream("text", inputs, lengths, ref)
        freqs = prepared["_flux3_text_cos"], prepared["_flux3_text_sin"]
        mod = tuple(state["early_" + f] for f in ("shift", "scale", "gate"))
        state["tokens"] = model._early_adaln_runtime["text"].prepare(state["tokens"], *mod[:2])
        for block in model.txt_mode_blocks:
            state["tokens"] = block(
                state["tokens"], freqs, mod, sample_bounds=bounds(lengths), timestep=timestep
            )
        outs.append({f: state[f] for f in ("tokens", "single_shift", "single_scale", "single_gate")})
    return outs


def batched_request(model, inputs, prepared):
    sched = sigmas()
    ts = inputs["x_text_timesteps"]
    targets = sched.sigmas[1 : len(sched.timesteps) + 1].to(device=ts.device, dtype=ts.dtype)
    return model.prepare_flow_map_text(targets, **inputs, **prepared)


def batched_steps(model, res):
    out = []
    for i in range(STEPS):
        s = model.flow_map_text_step(res, i)
        out.append(
            {f: s["_flux3_text_step_" + f] for f in ("tokens", "single_shift", "single_scale", "single_gate")}
        )
    return out


def has_batched(model):
    return hasattr(model, "prepare_flow_map_text") and os.environ.get(
        "TRTLLM_FLUX3_TEXT_STEP_BATCH", "1"
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


def snapshot(inputs, prepared):
    return {k: v.clone() for k, v in {**inputs, **prepared}.items() if isinstance(v, torch.Tensor)}


def unchanged(inputs, prepared, snap):
    cur = {**inputs, **prepared}
    return all(torch.equal(cur[k], v) for k, v in snap.items())


def compare(a, b):
    """a vs reference b, per field: rel-L2 and max abs / max |ref|."""
    out = {}
    for f in b:
        x, y = a[f].float(), b[f].float()
        d = (x - y).abs()
        out[f] = {
            "rel_l2": ((x - y).norm() / y.norm().clamp_min(1e-30)).item(),
            "max_abs": d.max().item(),
            "max_frac": (d.max() / y.abs().max().clamp_min(1e-30)).item(),
        }
    return out


def main():
    torch.manual_seed(0)
    log("building model")
    t0 = time.time()
    model = build()
    log(f"built in {time.time() - t0:.1f}s; batched path available: {has_batched(model)}")
    inputs, prepared = make_inputs(model, T_A, 7)
    snap = snapshot(inputs, prepared)
    batched = has_batched(model)

    with torch.inference_mode():
        # Warm both paths (compiles M=T and, on the branch, M=16T).
        for _ in range(WARMUP):
            per_step_request(model, inputs, prepared)
            if batched:
                batched_request(model, inputs, prepared)
        torch.cuda.synchronize()
        log("warmed")

        if MODE == "bench":
            fn = (lambda: batched_request(model, inputs, prepared)) if batched else (
                lambda: per_step_request(model, inputs, prepared)
            )
            assert unchanged(inputs, prepared, snap)
            ts = time_fn(fn, ITERS)
            st = stats(ts)
            log("this tree's path:", "batched" if batched else "per-step", json.dumps(st))
            extra = {}
            if batched:
                # In-process interleaved A/B plus A/A (per-step vs per-step).
                ab = {"per_step": [], "per_step_aa": [], "batched": []}
                for it in range(ITERS):
                    for key, f in (
                        ("per_step", lambda: per_step_request(model, inputs, prepared)),
                        ("batched", lambda: batched_request(model, inputs, prepared)),
                        ("per_step_aa", lambda: per_step_request(model, inputs, prepared)),
                    ):
                        ab[key] += time_fn(f, 1)
                    if it % 10 == 0:
                        log("interleaved iter", it)
                extra = {k: stats(v) for k, v in ab.items()}
                log("interleaved", json.dumps(extra))
            assert unchanged(inputs, prepared, snap)
            print(
                json.dumps(
                    {
                        "batch": {"1": {"kernel_ms": st["median_ms"], "noise_floor": st["noise"]}},
                        "output": {"class": "reordered" if batched else "exact"},
                        "path": "batched" if batched else "per_step",
                        "text_len": T_A,
                        "steps": STEPS,
                        "interleaved": extra,
                    }
                )
            )
            return

        if MODE == "profile":
            import torch.cuda.nvtx as nvtx

            # B33_PROFILE_PATH=per_step|batched|both selects what the capture region holds.
            which = os.environ.get("B33_PROFILE_PATH", "both")
            paths = []
            if which in ("per_step", "both"):
                paths.append(("per_step_request", lambda: per_step_request(model, inputs, prepared)))
            if batched and which in ("batched", "both"):
                paths.append(("batched_request", lambda: batched_request(model, inputs, prepared)))
            reps = 3
            wall = {name: [] for name, _ in paths}
            torch.cuda.cudart().cudaProfilerStart()
            for i in range(reps):
                for name, fn in paths:
                    torch.cuda.synchronize()
                    nvtx.range_push(name)
                    t = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    wall[name].append((time.perf_counter() - t) * 1e3)
                    nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            log("wall ms per request (under nsys)", json.dumps(wall))
            print(json.dumps({"passed": True, "profiled": True, "requests_per_path": reps, "wall_ms": wall}))
            return

        # check
        assert batched, "check needs the branch's batched path"
        fails = []
        ref_steps = [{k: v.clone() for k, v in s.items()} for s in per_step_request(model, inputs, prepared)]
        # Scale: per-step main path with eager (uncompiled) txt blocks vs compiled.
        compiled = list(model.txt_mode_blocks)
        for i, b in enumerate(compiled):
            model.txt_mode_blocks[i] = getattr(b, "_orig_mod", b)
        eager_steps = per_step_request(model, inputs, prepared)
        for i, b in enumerate(compiled):
            model.txt_mode_blocks[i] = b
        scale = [compare(e, r) for e, r in zip(eager_steps, ref_steps)]
        worst_scale = {
            f: max(s[f]["rel_l2"] for s in scale) for f in ("tokens", "single_shift", "single_scale", "single_gate")
        }
        first = None
        worst = {f: {"rel_l2": 0.0, "max_frac": 0.0} for f in ref_steps[0]}
        for it in range(WARMUP + ITERS):
            if not unchanged(inputs, prepared, snap):
                fails.append(f"inputs changed before iter {it}")
            res = batched_request(model, inputs, prepared)
            steps = batched_steps(model, res)
            flat = torch.cat([res[k].flatten().float() for k in sorted(res)])
            if first is None:
                first = flat
                for i, (s, r) in enumerate(zip(steps, ref_steps)):
                    c = compare(s, r)
                    for f in c:
                        worst[f]["rel_l2"] = max(worst[f]["rel_l2"], c[f]["rel_l2"])
                        worst[f]["max_frac"] = max(worst[f]["max_frac"], c[f]["max_frac"])
                        if c[f]["rel_l2"] > TOL_REL_L2 or c[f]["max_frac"] > TOL_MAX_FRAC:
                            fails.append(f"step {i} {f} {c[f]}")
                for f in ("single_shift", "single_scale", "single_gate"):
                    # Slices must be exactly the per-token expansion of their rows.
                    if not torch.equal(steps[3][f], steps[3][f][:, :1].expand_as(steps[3][f])):
                        fails.append(f"{f} slice not a row expansion")
            elif not torch.equal(flat, first):
                fails.append(f"iter {it} not bitwise equal to iter 0")
        log("batched vs per-step worst", json.dumps(worst), "scale(eager vs compiled per-step)", json.dumps(worst_scale))

        # A (T=80), B (T=160), A: A runs bitwise equal; B checked against its per-step path.
        inputs_b, prepared_b = make_inputs(model, T_B, 8)
        a1 = batched_request(model, inputs, prepared)
        a1 = {k: v.clone() for k, v in a1.items()}
        b1 = batched_request(model, inputs_b, prepared_b)
        b_ref = per_step_request(model, inputs_b, prepared_b)
        b_cmp = [compare(s, r) for s, r in zip(batched_steps(model, b1), b_ref)]
        b_worst = {f: max(c[f]["rel_l2"] for c in b_cmp) for f in b_cmp[0]}
        if max(b_worst.values()) > TOL_REL_L2:
            fails.append(f"T=160 batched vs per-step {b_worst}")
        a2 = batched_request(model, inputs, prepared)
        aba_equal = all(torch.equal(a1[k], a2[k]) for k in a1)
        if not aba_equal:
            fails.append("A after B not bitwise equal to A")
        # Env switch falls back to per-step (returns None).
        os.environ["TRTLLM_FLUX3_TEXT_STEP_BATCH"] = "0"
        off = batched_request(model, inputs, prepared)
        os.environ["TRTLLM_FLUX3_TEXT_STEP_BATCH"] = "1"
        if off is not None:
            fails.append("env switch did not disable batching")
        # FP8 switch-over below T (TRTLLM_VISUAL_GEN_FP8_ATTN_MIN_SEQ lowered) must fall back.
        backend = model.txt_mode_blocks[0].attn.attn
        old_min = backend.__dict__.get("_fp8_min_seq_len", None)
        backend._fp8_min_seq_len = T_A
        fp8_fallback = batched_request(model, inputs, prepared) is None
        if old_min is None:
            del backend._fp8_min_seq_len
        else:
            backend._fp8_min_seq_len = old_min
        if not fp8_fallback:
            fails.append("FP8 min-seq <= T did not fall back to per-step")
        # A backend that may use `timestep` (anything but VANILLA/FLASHINFER) must fall back.

        class _TimestepBackend:  # backends are plain AttentionBackend objects, not Modules
            pass

        attn_mod = model.txt_mode_blocks[0].attn
        attn_mod.attn = _TimestepBackend()
        backend_fallback = batched_request(model, inputs, prepared) is None
        attn_mod.attn = backend
        if not backend_fallback:
            fails.append("non VANILLA/FLASHINFER backend did not fall back to per-step")
        if batched_request(model, inputs, prepared) is None:
            fails.append("batching not restored after fallback checks")
        mem_mb = sum(v.numel() * v.element_size() for v in a1.values()) / 2**20
        print(
            json.dumps(
                {
                    "passed": not fails,
                    "failures": fails[:20],
                    "compared": "batched prepare_flow_map_text slices vs main's per-step text-stream code "
                    "(same model, same inputs, compiled txt blocks), every step; tolerance declared "
                    f"rel_l2<={TOL_REL_L2}, max_frac<={TOL_MAX_FRAC}",
                    "worst_batched_vs_per_step": worst,
                    "scale_eager_vs_compiled_per_step_rel_l2": worst_scale,
                    "iterations_bitwise_equal": not any("bitwise" in f for f in fails),
                    "iterations": WARMUP + ITERS,
                    "T160_batched_vs_per_step_rel_l2": b_worst,
                    "A_B_A_bitwise_equal": aba_equal,
                    "precomputed_mb_T80": mem_mb,
                }
            )
        )


if __name__ == "__main__":
    main()
