"""Idea 228 STEP 1: fused BF16 text QKV GEMM (cat([wq,wk,wv]) + split views) for txt_mode_blocks.

modes (argv[1]):
  sweep : op level, synthetic weights [3072,3072] x3, x [1, 16*T, 3072] for T = 80..960 step 80
          (every pad_multiple bucket below the step-batch limit 1024). fused vs 3 separate F.linear,
          bitwise. Untimed repeated sequence (same warmup/iters as timing) vs reference and vs
          iteration 0, inputs unchanged. Timing at M=1280 and 2560, interleaved, CUDA events.
  model : synthetic Flux3Model (flow_map, hidden 3072, 24 heads, 5 txt blocks, every module
          excluded from NVFP4 -> the served BF16 SEPARATE_QKV text path), txt blocks torch.compile'd
          like the pipeline. prepare_flow_map_text with the fused weight off (main's path) / on:
          _flux3_text_steps_tokens bitwise (T=80 and 160), 2 consecutive requests bitwise, A/B/A +
          A/A interleaved timing of the whole P-span, a kernel table per request (NET COST row) and
          the fused copy's memory.
  pair  : for run_gpu_pair: this tree's default P-span (prepare_flow_map_text, T=80) kernel_ms.
"""

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

MODE = sys.argv[1] if len(sys.argv) > 1 else "pair"
STEPS = 16
D = 3072
WARMUP = 5


def log(*a):
    print(*a, flush=True)


def header():
    try:
        head = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True).stdout
    except Exception as e:  # noqa: BLE001
        head = repr(e)
    import tensorrt_llm

    log("HEAD:", head.strip().replace("\n", " | "))
    log("tensorrt_llm:", tensorrt_llm.__file__)


def stats(ts):
    s = sorted(ts)
    n = len(s)
    med = s[n // 2]
    p10, p90 = s[int(0.1 * (n - 1))], s[int(0.9 * (n - 1))]
    return {"median": med, "p10": p10, "p90": p90, "noise": (p90 - p10) / abs(med) if med else None, "n": n}


def ev_time(fn):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


# ----------------------------------------------------------------------------- sweep
def sweep():
    g = torch.Generator(device="cuda").manual_seed(1234)
    ws = [(torch.randn(D, D, generator=g, device="cuda") / D**0.5).to(torch.bfloat16) for _ in range(3)]
    wcat = torch.cat(ws, 0).contiguous()
    res = {"bitwise": {}, "timing_us": {}}
    ok = True
    for t in range(80, 1024, 80):
        m = STEPS * t
        x = torch.randn(1, m, D, generator=g, device="cuda").to(torch.bfloat16)
        x0 = x.clone()
        ref = [F.linear(x, w) for w in ws]
        iters = 20
        for _ in range(WARMUP):
            F.linear(x, wcat)
        good = True
        first = None
        for it in range(iters):
            out = F.linear(x, wcat).split([D, D, D], dim=-1)
            if first is None:
                first = [o.clone() for o in out]
            good &= all(torch.equal(o, r) for o, r in zip(out, ref))
            good &= all(torch.equal(o, f) for o, f in zip(out, first))
        good &= torch.equal(x, x0)
        res["bitwise"][str(t)] = bool(good)
        ok &= bool(good)
        log(f"T={t} M={m} bitwise={good}")
        if t in (80, 160):
            ROUNDS = 100

            def sep():
                for w in ws:
                    F.linear(x, w)

            def fus():
                F.linear(x, wcat).split([D, D, D], dim=-1)

            for _ in range(10):
                sep(), fus()
            torch.cuda.synchronize()
            a, b, a2 = [], [], []
            for _ in range(ROUNDS):
                a.append(ev_time(sep) * 1e3)
                b.append(ev_time(fus) * 1e3)
                a2.append(ev_time(sep) * 1e3)
            res["timing_us"][str(t)] = {
                "separate": stats(a),
                "fused": stats(b),
                "fused_minus_separate": stats([y - x_ for x_, y in zip(a, b)]),
                "aa": stats([y - x_ for x_, y in zip(a, a2)]),
            }
            log(f"T={t} timing", json.dumps(res["timing_us"][str(t)]))
    res["passed"] = ok
    print(json.dumps(res))


# ----------------------------------------------------------------------------- model
def build():
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3Model
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    pretrained = SimpleNamespace(
        flow_map=True, hidden_size=3072, num_heads=24, depth=5, depth_single_blocks=1,
        context_in_dim=20480, mlp_ratio=3.0,
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
        [torch.zeros(n_img), torch.arange(n_img) // 8, torch.arange(n_img) % 8, torch.zeros(n_img)], dim=1
    ).to("cuda", torch.long)
    packed = pack([{"text": (ctx, text_ids), "image": (img, img_ids)}], {"text": model.context_in_dim, **model.in_channels})
    inputs = packed.kwargs()
    inputs["_flux3_uniform_timesteps"] = True
    with torch.inference_mode():
        prepared = model.prepare_static_inputs(**inputs)
    return inputs, prepared


def targets_for(inputs):
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.scheduler import Flux3FlowMatchScheduler

    s = Flux3FlowMatchScheduler(3.0)
    s.set_timesteps(STEPS)
    ts = inputs["x_text_timesteps"]
    return s.sigmas[1 : len(s.timesteps) + 1].to(device=ts.device, dtype=ts.dtype)


def request(model, inputs, prepared, targets):
    return model.prepare_flow_map_text(targets, **inputs, **prepared)["_flux3_text_steps_tokens"]


def set_flag(model, on):
    """on=False: main's path (attribute None == TRTLLM_FLUX3_P_TEXT_GEMM=0)."""
    if not hasattr(model, "prepare_text_qkv_weights"):
        assert not on, "base tree has no fused text QKV path"
        return []
    os.environ["TRTLLM_FLUX3_P_TEXT_GEMM"] = "1" if on else "0"
    return model.prepare_text_qkv_weights()


def snapshot(inputs, prepared):
    return {k: v.clone() for k, v in {**inputs, **prepared}.items() if isinstance(v, torch.Tensor)}


def unchanged(inputs, prepared, snap):
    cur = {**inputs, **prepared}
    return all(torch.equal(cur[k], v) for k, v in snap.items())


def kernel_table(fn, reps=3):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    k = {}
    for ev in prof.events():
        if ev.device_type != DeviceType.CUDA:
            continue
        e = k.setdefault(ev.name[:100], [0.0, 0])
        e[0] += (ev.time_range.end - ev.time_range.start) / reps
        e[1] += 1 / reps
    return {n: {"us": round(v[0], 2), "calls": round(v[1], 2)} for n, v in sorted(k.items(), key=lambda kv: -kv[1][0])}


def model_mode():
    header()
    torch.manual_seed(0)
    t0 = time.time()
    model = build()
    log(f"built in {time.time() - t0:.1f}s")
    out = {"bitwise": {}, "passed": True}
    with torch.inference_mode():
        for T in (80, 160):
            inputs, prepared = make_inputs(model, T, 7 + T)
            snap = snapshot(inputs, prepared)
            tg = targets_for(inputs)
            set_flag(model, False)
            for _ in range(3):
                request(model, inputs, prepared, tg)
            ref0 = request(model, inputs, prepared, tg).clone()
            ref0b = request(model, inputs, prepared, tg).clone()
            torch.cuda.synchronize()
            mem0 = torch.cuda.memory_allocated()
            fused = set_flag(model, True)
            mem1 = torch.cuda.memory_allocated()
            for _ in range(3):
                request(model, inputs, prepared, tg)
            on1 = request(model, inputs, prepared, tg).clone()
            on2 = request(model, inputs, prepared, tg).clone()
            set_flag(model, False)
            off_again = request(model, inputs, prepared, tg).clone()
            r = {
                "fused_blocks": fused,
                "fused_copy_mb": round((mem1 - mem0) / 2**20, 1),
                "on_vs_off": torch.equal(on1, ref0),
                "on_repeat": torch.equal(on1, on2),
                "off_repeat": torch.equal(ref0, ref0b),
                "off_after_on_vs_off": torch.equal(off_again, ref0),
                "inputs_unchanged": unchanged(inputs, prepared, snap),
                "max_abs_on_vs_off": (on1.float() - ref0.float()).abs().max().item(),
            }
            ok = all(r[k] for k in ("on_vs_off", "on_repeat", "off_repeat", "off_after_on_vs_off", "inputs_unchanged"))
            r["passed"] = ok and len(fused) == 5
            out["bitwise"][str(T)] = r
            out["passed"] &= r["passed"]
            log(f"T={T}", json.dumps(r))

        # A/B/A + A/A timing of the P-span at T=80 (served length).
        T = 80
        inputs, prepared = make_inputs(model, T, 87)
        tg = targets_for(inputs)
        fn = lambda: request(model, inputs, prepared, tg)  # noqa: E731
        for on in (False, True):
            set_flag(model, on)
            for _ in range(WARMUP):
                fn()
        torch.cuda.synchronize()
        rounds = int(os.environ.get("P_ROUNDS", "40"))
        a, b, a2 = [], [], []
        for i in range(rounds):
            set_flag(model, False)
            fn()
            a.append(ev_time(fn))
            set_flag(model, True)
            fn()
            b.append(ev_time(fn))
            set_flag(model, False)
            fn()
            a2.append(ev_time(fn))
            if i % 10 == 0:
                log(f"round {i}: off {a[-1]:.3f} on {b[-1]:.3f} off {a2[-1]:.3f} ms")
        timing = {
            "flag0_ms": stats(a),
            "flag1_ms": stats(b),
            "flag0_again_ms": stats(a2),
            "delta_1_minus_0_ms": stats([y - x for x, y in zip(a, b)]),
            "delta_1_minus_mean0_ms": stats([y - (x + z) / 2 for x, y, z in zip(a, b, a2)]),
            "aa_ms": stats([z - x for x, z in zip(a, a2)]),
            "rounds": rounds,
            "unit": "ms per request (one prepare_flow_map_text = whole P-span, T=80, M=1280)",
        }
        log("timing", json.dumps(timing))
        out["timing"] = timing
        set_flag(model, False)
        k0 = kernel_table(fn)
        set_flag(model, True)
        k1 = kernel_table(fn)
        tot0 = sum(v["us"] for v in k0.values())
        tot1 = sum(v["us"] for v in k1.values())
        diff = {}
        for n in set(k0) | set(k1):
            u0 = k0.get(n, {"us": 0.0, "calls": 0})
            u1 = k1.get(n, {"us": 0.0, "calls": 0})
            if abs(u1["us"] - u0["us"]) > 2.0 or u1["calls"] != u0["calls"]:
                diff[n] = {"flag0": u0, "flag1": u1}
        out["net_cost"] = {
            "kernel_sum_us_flag0": round(tot0, 1),
            "kernel_sum_us_flag1": round(tot1, 1),
            "saving_us": round(tot0 - tot1, 1),
            "changed_kernels": diff,
        }
        log("net_cost", json.dumps(out["net_cost"]))
    print(json.dumps(out))


def pair_mode():
    header()
    torch.manual_seed(0)
    model = build()
    with torch.inference_mode():
        fused = set_flag(model, True) if hasattr(model, "prepare_text_qkv_weights") else []
        log("fused blocks:", fused)
        inputs, prepared = make_inputs(model, 80, 87)
        tg = targets_for(inputs)
        fn = lambda: request(model, inputs, prepared, tg)  # noqa: E731
        for _ in range(WARMUP + 3):
            fn()
        torch.cuda.synchronize()
        ts = [ev_time(fn) for _ in range(int(os.environ.get("P_ITERS", "80")))]
        st = stats(ts)
        log("P-span", json.dumps(st))
    print(json.dumps({
        "batch": {"1": {"kernel_ms": st["median"], "noise_floor": st["noise"]}},
        "output": {"class": "exact"},
        "path": "fused" if fused else "separate",
        "p10_ms": st["p10"], "p90_ms": st["p90"],
    }))


if __name__ == "__main__":
    {"sweep": sweep, "model": model_mode, "pair": pair_mode}[MODE]()
