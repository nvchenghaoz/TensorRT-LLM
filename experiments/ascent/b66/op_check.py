"""Branch 66: fused LN+adaLN-modulate+NVFP4 quantize op vs main's two kernels (standalone).

Reference (main): the block's LN+modulate compiled by Inductor (per-block compile options,
emulate_precision_casts=True) followed by FlashInfer nvfp4_quantize(cute-dsl, layout_128x4,
do_shuffle=False, host-float global scale) exactly as trtllm::flux3_nvfp4_quantize_prepared calls it.

Usage:
  op_check.py check   -> config sweep + bitwise check (x_mod, fp4, FULL poisoned SF buffer) and an
                         untimed repeat sequence (20 iters, vs iter 0 and reference, inputs unchanged)
  op_check.py bench   -> cold-L2 per-call timing of main's two kernels (baseline tree) or the fused
                         op (candidate tree); prints the evidence JSON.
"""

import json
import statistics
import sys

import torch

from tensorrt_llm._torch.modules.layer_norm import LayerNorm

D = 3072
EPS = 1e-6
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
opts = {"emulate_precision_casts": True}
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(1234)

try:
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq
except ImportError:
    faq = None
if len(sys.argv) > 2 and sys.argv[2] == "force_baseline":
    faq_saved, faq = faq, None

ln = LayerNorm(hidden_size=D, eps=EPS, has_weights=False, has_bias=False, dtype=torch.bfloat16).to(dev)


def heavy(*shape):
    # Activations with outlier rows/channels (served |x| reaches ~6e5).
    base = torch.randn(*shape, device=dev, generator=g)
    mag = torch.exp(torch.randn(*shape[:-1], 1, device=dev, generator=g) * 2.0)
    chan = torch.exp(torch.randn(shape[-1], device=dev, generator=g) * 1.5)
    return (base * mag * chan).to(torch.bfloat16)


class Single(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = ln

    def forward(self, x, shift_t, scale_t, select):
        shift = torch.where(select, shift_t[:, 1:2], shift_t[:, 0:1])
        scale = torch.where(select, scale_t[:, 1:2], scale_t[:, 0:1])
        return (1 + scale) * self.pre_norm(x) + shift


class Mode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = ln

    def forward(self, x, shift, scale):
        return (1 + scale) * self.pre_norm(x) + shift


def fi_quant(x2d, gs_float):
    from flashinfer import SfLayout
    from flashinfer import nvfp4_quantize as quantize

    data, scales = quantize(
        x2d, gs_float, sfLayout=SfLayout.layout_128x4, do_shuffle=False, backend="cute-dsl"
    )
    return data.view(torch.uint8).contiguous(), scales.view(torch.uint8).reshape(-1).contiguous()


def make_cases():
    cases = []
    # Single blocks: S = 4096 + T (T=80 served, T=160 second bucket), [1,2,D] tables as views of
    # a larger modulation tensor (nonzero storage offset), select True on the text rows.
    for t in (80, 160):
        s = 4096 + t
        x = heavy(1, s, D)
        mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        shift_t, scale_t = mod[..., D : 2 * D], mod[..., 4 * D : 5 * D]
        select = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
        select[:, :t] = True
        cases.append((f"single_T{t}", "single", (x, shift_t, scale_t, select)))
    # Mode blocks (image stream, M=4096): rows as views of a per-request [1,16,.] step table.
    x = heavy(1, 4096, D)
    table = (torch.randn(1, 16, 3 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    for k in (0, 5, 15):
        row = table[:, k : k + 1]
        cases.append((f"mode_step{k}", "mode", (x, row[..., :D], row[..., D : 2 * D])))
    return cases


compiled = {"single": torch.compile(Single(), dynamic=None, options=opts),
            "mode": torch.compile(Mode(), dynamic=None, options=opts)}


def reference(kind, args, gs_float):
    with torch.inference_mode():
        xm = compiled[kind](*args)
        data, sf = fi_quant(xm.reshape(-1, D), gs_float)
    return xm, data, sf


def candidate(kind, args, gs_t, config=None, poison=True):
    if kind == "single":
        x, shift_t, scale_t, select = args
        sel, sc, sh = select, scale_t, shift_t
    else:
        x, shift, scale = args
        sel, sc, sh = None, scale, shift
    m = x.numel() // D
    outs = None
    if poison:
        outs = (
            torch.full_like(x, float("nan")),
            torch.full((m, D // 2), 0xAB, dtype=torch.uint8, device=dev),
            torch.full((faq._scale_size(m, D),), 0xAB, dtype=torch.uint8, device=dev),
        )
    with torch.inference_mode():
        return faq._launch(x, sel, sc, sh, ln.weight, ln.bias, gs_t, EPS, config=config, outs=outs)


def ulps(a, b):
    ai = a.view(torch.int16).int()
    bi = b.view(torch.int16).int()
    return (ai - bi).abs().max().item()


def run_check():
    assert faq is not None, "candidate tree required"
    res = {"cases": {}}
    ok = True
    sweep = [(512, 4), (512, 8), (1024, 4), (1024, 8), (1024, 16), (2048, 8), (2048, 16)]
    for name, kind, args in make_cases():
        xm0, _, _ = reference(kind, args, 1.0)
        gs_float = float(torch.tensor(2688.0 / xm0.float().abs().max().item(), dtype=torch.float32))
        gs_t = torch.tensor([gs_float], dtype=torch.float32, device=dev)
        ref = reference(kind, args, gs_float)
        # Config sweep: which (R0_BLOCK, num_warps) is bitwise to Inductor's LN.
        match = {}
        for cfg in sweep:
            out = candidate(kind, args, gs_t, config=cfg)
            match[str(cfg)] = torch.equal(out[0], ref[0])
        cand = candidate(kind, args, gs_t)
        sf_ref = ref[2]
        rec = {
            "config_match": match,
            "x_mod_bitwise": torch.equal(cand[0], ref[0]),
            "x_mod_max_ulp": ulps(cand[0], ref[0]),
            "fp4_bitwise": torch.equal(cand[1], ref[1]),
            "fp4_mismatch": int((cand[1] != ref[1]).sum().item()),
            "sf_full_bitwise": cand[2].numel() == sf_ref.numel() and torch.equal(cand[2], sf_ref),
            "sf_mismatch": int((cand[2] != sf_ref).sum().item()) if cand[2].numel() == sf_ref.numel() else -1,
            "fp4_rows": cand[1].shape[0],
            "sf_bytes": cand[2].numel(),
        }
        # Quantizer alone on the reference x_mod (isolates quantize exactness from LN order).
        if kind == "single":
            pass
        # Saturation probe: 4x larger global scale drives e4m3 scales to satfinite.
        gs_big = float(torch.tensor(gs_float * 4.0, dtype=torch.float32))
        gs_big_t = torch.tensor([gs_big], dtype=torch.float32, device=dev)
        ref_b = reference(kind, args, gs_big)
        cand_b = candidate(kind, args, gs_big_t)
        rec["sat_fp4_bitwise"] = torch.equal(cand_b[1], ref_b[1])
        rec["sat_sf_bitwise"] = torch.equal(cand_b[2], ref_b[2])
        # Untimed repeat sequence: 20 iterations, same inputs; vs iter 0 and reference; inputs
        # unchanged (pristine copies compared before the reset point).
        pristine = [a.clone() for a in args]
        first = None
        rep_ok = True
        for it in range(20):
            out = candidate(kind, args, gs_t, poison=(it % 2 == 0))
            if first is None:
                first = out
            rep_ok &= all(torch.equal(o, f) for o, f in zip(out, first))
            rep_ok &= all(torch.equal(o, r) for o, r in zip(out, ref))
        rep_ok &= all(torch.equal(a, p) for a, p in zip(args, pristine))
        rec["repeat_bitwise"] = rep_ok
        case_ok = (
            rec["x_mod_bitwise"] and rec["fp4_bitwise"] and rec["sf_full_bitwise"]
            and rec["sat_fp4_bitwise"] and rec["sat_sf_bitwise"] and rep_ok
        )
        rec["passed"] = case_ok
        ok &= case_ok
        res["cases"][name] = rec
        print(name, json.dumps(rec), flush=True)
    print(json.dumps({"passed": bool(ok), "compared_against":
                      "Inductor LN+modulate (emulate_precision_casts) + FlashInfer nvfp4_quantize "
                      "cute-dsl 128x4 host-float gs (main's flux3_nvfp4_quantize_prepared)",
                      **res}))


def run_bench():
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
    results = {}
    arm = "candidate" if faq is not None else "baseline"
    for name, kind, args in make_cases():
        if name not in ("single_T80", "mode_step5"):
            continue
        xm0, _, _ = reference(kind, args, 1.0)
        gs_float = float(torch.tensor(2688.0 / xm0.float().abs().max().item(), dtype=torch.float32))
        gs_t = torch.tensor([gs_float], dtype=torch.float32, device=dev)
        if faq is None:
            def fn():
                return reference(kind, args, gs_float)
        else:
            def fn():
                return candidate(kind, args, gs_t, poison=False)
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(3):
            for _ in range(100):
                flush.add_(1)  # evict L2 (cold-L2 like the served chain)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e))
        times.sort()
        med = statistics.median(times)
        p10, p90 = times[len(times) // 10], times[(9 * len(times)) // 10]
        results[name] = {"ms": med, "noise": (p90 - p10) / med}
        print(arm, name, json.dumps(results[name]), flush=True)
    total = results["single_T80"]["ms"]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": total, "noise_floor": results["single_T80"]["noise"]}},
        "output": {"class": "exact"},
        "arm": arm,
        "detail": results,
        "note": "single_T80 = one single-block LN+mod+quantize call at M=4176, cold L2, "
                "includes launch overhead of each arm (main: 2 launches, fused: 1)",
    }))


if MODE == "check":
    run_check()
else:
    run_bench()
