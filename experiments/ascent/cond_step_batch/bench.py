"""Idea 87 bench: FLUX.3 flow-map i2i text stream + image vec rows, per-step vs step-batched.

usage: bench.py <bench|check> [checkpoint]

No model load: builds a Flux3Model (flow_map=True; hidden/heads/depth/context_in_dim from the
checkpoint's config.json "dit" block when readable, else 3072/24/5/20480) with every module
BF16 (static-NVFP4 config excluding all modules, as the served checkpoint keeps the text
blocks, time_in / ref_time_in, modulations and final adaLN in BF16), random weights
(std 1/sqrt(fan_in)), txt_mode_blocks torch.compiled per block, FLASHINFER attention with FP8
Q/K/V (text T=80 < 1024 runs BF16 attention, as served). Inputs packed with pack() in the
served i2i layout: text T=80, image 16384 and image_cond 16384 tokens (2048^2).

One "request" = the stream-preparation work forward does for 16 steps, written exactly as
Flux3Model.forward / Flux3Pipeline._run_denoise run it:
  prepare_flow_map_text + prepare_flow_map_image (return None on the per-step path), then per
  step: ref_time_in, the text stream (per-step: _packed_stream + 5 txt blocks at M=T; batched:
  the step's slice), the image stream (per-step: _packed_stream; batched: _image_step_stream)
  and the final adaLN row (per-step: adaLN_modulation(_final_layer_vec); batched: final_mod).
The image_cond stream, the joint single blocks and the final layer's norm/linear run the same
code on both paths and are left out.

Arms, each its own CUDA graph of one whole request (served runs the denoiser in graphs):
  P  : TRTLLM_FLUX3_COND_STEP_BATCH=0 (main's per-step i2i path; on the base tree: always)
  P2 : P captured again (A/A)
  B  : TRTLLM_FLUX3_COND_STEP_BATCH=1 (this branch; absent on the base tree)
bench: interleaved replays in seeded random order, CUDA events; last line is the pair JSON for
       this tree's default path (unset env). Also prints per-step max-abs diffs B vs P, P2 vs P.
check: untimed sequence with the same warmup / replay count and reset points; every replay of
       every arm compared with P (bitwise for P2, declared tolerance for B) and with its own
       replay 0 (bitwise); inputs unchanged before each reset.
Declared tolerance for B vs P (reordered: other cuBLAS/Inductor reductions at M=16T vs M=T and
M=16 vs M=1), per step and per tensor: rel-L2 <= 1e-2 and max|diff| <= 3% of max|P|.
"""

import json
import os
import random
import sys
import time
from types import SimpleNamespace

import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
CKPT = sys.argv[2] if len(sys.argv) > 2 else None
STEPS = 16
T_TXT = 80
N_IMG = 16384
N_COND = 16384
WARMUP = 3
ROUNDS = int(os.environ.get("CSB_ROUNDS", "40"))
TOL_REL_L2 = 1e-2
TOL_MAX_FRAC = 3e-2
ENV = "TRTLLM_FLUX3_COND_STEP_BATCH"
FIELDS = ("text_tokens", "text_single_shift", "text_single_scale", "text_single_gate",
          "image_tokens", "image_early_shift", "image_early_scale", "image_early_gate",
          "image_single_shift", "image_single_scale", "image_single_gate", "final_mod")


def log(*a):
    print(f"[csb {time.strftime('%H:%M:%S')}]", *a, flush=True)


def dit_config():
    cfg = dict(hidden_size=3072, num_heads=24, depth=5, depth_single_blocks=1,
               context_in_dim=20480, mlp_ratio=3.0)
    src = "default"
    if CKPT:
        path = os.path.join(CKPT, "config.json")
        try:
            with open(path) as f:
                dit = json.load(f).get("dit", {})
            for k in list(cfg):
                if k in dit and isinstance(dit[k], (int, float)):
                    cfg[k] = dit[k]
            src = path
        except Exception as exc:  # noqa: BLE001
            src = f"default ({type(exc).__name__})"
    cfg["depth_single_blocks"] = 1  # single blocks are not part of this work
    return cfg, src


def build():
    from tensorrt_llm._torch.visual_gen.config import (
        DiffusionModelConfig,
        create_attention_metadata_state,
    )
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3Model
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    cfg, src = dit_config()
    log(f"dit config from {src}: {cfg}")
    pretrained = SimpleNamespace(flow_map=True, **cfg)
    kw = dict(
        pretrained_config=pretrained,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, exclude_modules=["*"]),
        skip_create_weights_in_init=True,
    )
    backend = "FLASHINFER fp8"
    try:
        from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

        kw["attention"] = AttentionConfig(
            backend="FLASHINFER",
            quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
        )
        kw["attention_metadata_state"] = create_attention_metadata_state()
        mc = DiffusionModelConfig(**kw)
        with torch.device("cuda"):
            model = Flux3Model(mc)
    except Exception as exc:  # noqa: BLE001
        backend = f"default ({type(exc).__name__}: {exc})"
        kw.pop("attention", None)
        kw.pop("attention_metadata_state", None)
        mc = DiffusionModelConfig(**kw)
        with torch.device("cuda"):
            model = Flux3Model(mc)
    model = model.cuda().eval()
    g = torch.Generator(device="cuda").manual_seed(1234)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g, device="cuda") / p.shape[1] ** 0.5)
            else:
                p.fill_(1.0)
    for i in range(len(model.txt_mode_blocks)):
        model.txt_mode_blocks[i] = torch.compile(
            model.txt_mode_blocks[i], dynamic=None, options={"emulate_precision_casts": True}
        )
    log(f"attention backend: {backend}; txt block backend "
        f"{type(model.txt_mode_blocks[0]._orig_mod.attn.attn).__name__}; in_channels {model.in_channels}")
    return model


def grid_ids(n, t):
    side = int(n ** 0.5)
    return torch.stack(
        [torch.full((n,), float(t)), torch.arange(n) // side, torch.arange(n) % side, torch.zeros(n)],
        dim=1,
    ).to("cuda", torch.long)


def make_inputs(model, seed=7):
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import pack
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import prc_txt

    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.randn(T_TXT, model.context_in_dim, generator=g, device="cuda").to(torch.bfloat16)
    _, text_ids = prc_txt(ctx)
    img = torch.randn(N_IMG, model.in_channels["image"], generator=g, device="cuda").to(torch.bfloat16)
    cond = torch.randn(N_COND, model.in_channels["image_cond"], generator=g, device="cuda").to(
        torch.bfloat16)
    packed = pack(
        [{"text": (ctx, text_ids), "image": (img, grid_ids(N_IMG, 0)),
          "image_cond": (cond, grid_ids(N_COND, 10))}],
        {"text": model.context_in_dim, **model.in_channels},
    )
    inputs = packed.kwargs()
    inputs["_flux3_uniform_timesteps"] = True
    with torch.inference_mode():
        prepared = model.prepare_static_inputs(**inputs)
    return inputs, prepared


def schedule(inputs):
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.scheduler import (
        Flux3FlowMatchScheduler,
    )

    s = Flux3FlowMatchScheduler(3.0)
    s.set_timesteps(STEPS)
    dev = inputs["x_text_timesteps"].device
    targets = s.sigmas[1 : len(s.timesteps) + 1].to(device=dev, dtype=inputs["x_text_timesteps"].dtype)
    image_ts = s.timesteps.to(device=dev, dtype=inputs["x_image_timesteps"].dtype)
    return s, targets, image_ts


def request(model, inputs, prepared, sched, targets, image_ts, tref):
    """16 steps of forward's stream preparation, as _run_denoise + forward run them."""
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import bounds
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
        _block_freqs,
        flux3_timestep_embedding,
    )

    text_steps = model.prepare_flow_map_text(targets, **inputs, **prepared)
    image_steps = model.prepare_flow_map_image(image_ts, targets, **inputs, **prepared)
    layout = inputs["layout"]
    lengths = dict(layout)
    outs = []
    for i, timestep in enumerate(sched.timesteps):
        inputs["x_image_timesteps"].fill_(timestep.item())
        tref.fill_(sched.sigmas[i + 1].item())
        kw = {**inputs, **prepared}
        if text_steps is not None:
            kw.update(model.flow_map_text_step(text_steps, i))
        if image_steps is not None:
            kw.update(model.flow_map_image_step(image_steps, i))
        image_rows = "_flux3_image_step_early" in kw
        ref = None
        if not ("_flux3_text_step_tokens" in kw and image_rows
                and {n for n, l in layout if any(l)} <= {"text", "image"}):
            target = tref.to(kw["x_text_timesteps"])
            ref = model.ref_time_in(flux3_timestep_embedding(target, 256, out_dtype=target.dtype))
        row = {}
        if "_flux3_text_step_tokens" in kw:
            for f in ("tokens", "single_shift", "single_scale", "single_gate"):
                row["text_" + f] = kw["_flux3_text_step_" + f]
        else:
            state = model._packed_stream("text", kw, lengths["text"], ref)
            freqs = _block_freqs(prepared, "_flux3_text_")
            mod = tuple(state["early_" + f] for f in ("shift", "scale", "gate"))
            state["tokens"] = model._early_adaln_runtime["text"].prepare(state["tokens"], *mod[:2])
            for block in model.txt_mode_blocks:
                state["tokens"] = block(state["tokens"], freqs, mod,
                                        sample_bounds=bounds(lengths["text"]), timestep=timestep)
            for f in ("tokens", "single_shift", "single_scale", "single_gate"):
                row["text_" + f] = state[f]
        if image_rows:
            state = model._image_step_stream(kw, lengths["image"])
            row["final_mod"] = state["final_mod"]
        else:
            state = model._packed_stream("image", kw, lengths["image"], ref)
            vec = model._final_layer_vec(state["vec"], kw, lengths["image"])
            row["final_mod"] = model.final_layer["image"].adaLN_modulation(vec)
        row["image_tokens"] = state["tokens"]
        for st in ("early", "single"):
            for f in ("shift", "scale", "gate"):
                row[f"image_{st}_{f}"] = state[f"{st}_{f}"]
        outs.append(row)
    return outs, text_steps is not None and image_steps is not None


def with_env(value, fn):
    old = os.environ.get(ENV)
    if value is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = value
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old


def snapshot(inputs, prepared):
    return {k: v.clone() for k, v in {**inputs, **prepared}.items() if isinstance(v, torch.Tensor)}


def restore(inputs, prepared, snap):
    cur = {**inputs, **prepared}
    for k, v in snap.items():
        cur[k].copy_(v)


def unchanged(inputs, prepared, snap, skip=("x_image_timesteps",)):
    cur = {**inputs, **prepared}
    return [k for k, v in snap.items() if k not in skip and not torch.equal(cur[k], v)]


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def clone_outs(outs):
    return [{k: v.detach().clone() for k, v in row.items()} for row in outs]


def cmp(a, b):
    x, y = a.float(), b.float()
    if x.shape != y.shape:
        x, y = torch.broadcast_tensors(x, y)
    x, y = x.flatten(), y.flatten()
    d = (x - y).abs()
    return {
        "max_abs": d.max().item(),
        "rel_l2": ((x - y).norm() / y.norm().clamp_min(1e-30)).item(),
        "max_frac": (d.max() / y.abs().max().clamp_min(1e-30)).item(),
        "bitwise": bool(a.shape == b.shape and torch.equal(a, b)),
    }


def diff_table(outs, ref):
    rows, worst = [], {}
    for i, (o, r) in enumerate(zip(outs, ref)):
        step = {}
        for f in FIELDS:
            c = cmp(o[f], r[f])
            step[f] = c["max_abs"]
            w = worst.setdefault(f, {"max_abs": 0.0, "rel_l2": 0.0, "max_frac": 0.0, "bitwise": True})
            for k in ("max_abs", "rel_l2", "max_frac"):
                w[k] = max(w[k], c[k])
            w["bitwise"] = w["bitwise"] and c["bitwise"]
        rows.append({"step": i, "max_abs": step})
    return rows, worst


def within(worst):
    return all(w["rel_l2"] <= TOL_REL_L2 and w["max_frac"] <= TOL_MAX_FRAC for w in worst.values())


def stats(ts):
    s = sorted(ts)
    n = len(s)
    med = s[n // 2]
    p10, p90 = s[int(0.1 * (n - 1))], s[int(0.9 * (n - 1))]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "noise": (p90 - p10) / med, "n": n}


def main():
    torch.manual_seed(0)
    t0 = time.time()
    model = build()
    log(f"built in {time.time() - t0:.1f}s")
    inputs, prepared = make_inputs(model)
    snap = snapshot(inputs, prepared)
    sched, targets, image_ts = schedule(inputs)
    tref = inputs["x_text_timesteps"].new_empty(1)

    def req():
        return request(model, inputs, prepared, sched, targets, image_ts, tref)

    with torch.inference_mode():
        # Which paths this tree has; warm (compile M=T and M=16T) outside the graphs.
        _, batched_ok = with_env("1", req)
        _, default_batched = with_env(None, req)
        _, off_batched = with_env("0", req)
        log(f"tree: batched with {ENV}=1: {batched_ok}; default: {default_batched}; "
            f"with {ENV}=0: {off_batched}")
        if off_batched:
            print(json.dumps({"passed": False, "error": f"{ENV}=0 did not take the per-step path"}))
            return
        for _ in range(WARMUP):
            with_env("0", req)
            if batched_ok:
                with_env("1", req)
        torch.cuda.synchronize()
        log("warmed")

        arms = {}
        restore(inputs, prepared, snap)
        arms["P"] = with_env("0", lambda: capture(req))
        restore(inputs, prepared, snap)
        arms["P2"] = with_env("0", lambda: capture(req))
        if batched_ok:
            restore(inputs, prepared, snap)
            arms["B"] = with_env("1", lambda: capture(req))
        graphs = {k: v[0] for k, v in arms.items()}
        outs = {k: v[1][0] for k, v in arms.items()}
        log(f"captured arms {list(graphs)}")

        rng = random.Random(1234)
        if MODE == "bench":
            soak = time.time()
            while time.time() - soak < 8.0:
                for g in graphs.values():
                    g.replay()
                torch.cuda.synchronize()
            times = {k: [] for k in graphs}
            for r in range(ROUNDS):
                order = list(graphs)
                rng.shuffle(order)
                for k in order:
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    graphs[k].replay()
                    b.record()
                    torch.cuda.synchronize()
                    times[k].append(a.elapsed_time(b))
                if r % 10 == 9:
                    log(f"round {r + 1}/{ROUNDS}: " + ", ".join(
                        f"{k} {sorted(v)[len(v) // 2]:.3f}" for k, v in times.items()))
            st = {k: stats(v) for k, v in times.items()}
            paired = {}
            for k in graphs:
                if k == "P":
                    continue
                d = sorted(x - y for x, y in zip(times[k], times["P"]))
                paired[f"{k}-P"] = {"median_ms": d[len(d) // 2], "p10_ms": d[int(0.1 * (len(d) - 1))],
                                    "p90_ms": d[int(0.9 * (len(d) - 1))]}
            log("per-request in-graph ms: " + json.dumps(st))
            log("paired differences: " + json.dumps(paired))
            # Per-step diffs from the timed graphs' last replay (same inputs every replay).
            ref = clone_outs(outs["P"])
            diffs = {}
            for k in graphs:
                if k == "P":
                    continue
                rows, worst = diff_table(outs[k], ref)
                diffs[k] = {"worst": worst, "within_tolerance": within(worst)}
                log(f"{k} vs P per-step max_abs:")
                for row in rows:
                    log("  step %2d " % row["step"] + " ".join(
                        f"{f}={v:.3g}" for f, v in row["max_abs"].items()))
                log(f"{k} vs P worst: " + json.dumps(worst))
            arm = "B" if default_batched else "P"
            out = {
                "batch": {"i2i-2048": {"step_ms": st[arm]["median_ms"],
                                       "noise_floor": max(st[arm]["noise"], st["P2"]["noise"])}},
                "output": {"class": "reordered" if default_batched else "exact"},
                "default_arm": arm,
                "arms": st,
                "paired": paired,
                "diffs_vs_P": diffs,
                "what": "in-graph ms per i2i request (16 steps) of text stream + image vec rows",
            }
            print(json.dumps(out), flush=True)
            return

        # check: the same warmup/capture, then the same replay count, untimed.
        failures = []
        ref0 = {}
        for k, g in graphs.items():
            g.replay()
        torch.cuda.synchronize()
        for k in graphs:
            ref0[k] = clone_outs(outs[k])
        P = ref0["P"]
        summary = {}
        for r in range(ROUNDS):
            order = list(graphs)
            rng.shuffle(order)
            for k in order:
                graphs[k].replay()
                torch.cuda.synchronize()
                cur = outs[k]
                for i in range(STEPS):
                    for f in FIELDS:
                        if not torch.equal(cur[i][f], ref0[k][i][f]):
                            failures.append(f"{k} round {r} step {i} {f} != its replay 0")
            changed = unchanged(inputs, prepared, snap)
            if changed:
                failures.append(f"round {r}: inputs changed {changed[:5]}")
            restore(inputs, prepared, snap)
            if r % 10 == 9:
                log(f"check round {r + 1}/{ROUNDS}: {len(failures)} failures")
        for k in graphs:
            if k == "P":
                continue
            rows, worst = diff_table(ref0[k], P)
            ok = all(w["bitwise"] for w in worst.values()) if k == "P2" else within(worst)
            if not ok:
                failures.append(f"{k} vs P outside {'bitwise' if k == 'P2' else 'tolerance'}")
            summary[k] = {"worst": worst, "per_step_max_abs": rows, "passed": ok}
        # fp32-free sanity: per-step P matches an eager per-step request (no graph) bitwise.
        restore(inputs, prepared, snap)
        eager, _ = with_env("0", req)
        torch.cuda.synchronize()
        eager_ok = all(torch.equal(eager[i][f], P[i][f]) for i in range(STEPS) for f in FIELDS)
        if not eager_ok:
            failures.append("graph P != eager per-step request")
        log(f"failures: {len(failures)} {failures[:10]}")
        print(json.dumps({
            "passed": not failures,
            "compared": "B (step-batched) and P2 (A/A) vs P (TRTLLM_FLUX3_COND_STEP_BATCH=0, main's "
                        "per-step i2i path), every replay vs replay 0, inputs unchanged",
            "tolerance": {"rel_l2": TOL_REL_L2, "max_frac": TOL_MAX_FRAC},
            "rounds": ROUNDS, "arms": list(graphs), "graph_P_equals_eager": eager_ok,
            "summary": {k: {"passed": v["passed"], "worst": v["worst"]} for k, v in summary.items()},
            "failures": failures[:20],
            "output": {"class": "reordered"},
        }), flush=True)


if __name__ == "__main__":
    main()
