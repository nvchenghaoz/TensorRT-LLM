"""Branch 81 (idea 134): NVFP4 LN+adaLN+quant kernel skipping the dead bf16 x_mod store.

  op_check.py check -> skip (store_x_mod=False) vs store kernel vs main's pre-66 reference
                       (Inductor LN+mod, emulate_precision_casts + FlashInfer nvfp4_quantize
                       cute-dsl 128x4): fp4 data + FULL poisoned SF buffer bitwise at M=4176/4256
                       single tables and M=4096 mode row views; skip leaves the poisoned x_mod
                       untouched; untimed 20-iteration repeat bitwise, inputs unchanged.
  op_check.py bench -> cold-L2 per-call timing of the fused op inside a CUDA graph. Baseline tree
                       (no store_x_mod kwarg) = store kernel; candidate tree = skip kernel.
"""

import inspect
import json
import statistics
import sys

import torch

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq

D = 3072
EPS = 1e-6
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(1234)
HAS_SKIP = "store_x_mod" in inspect.signature(faq._launch).parameters
ln = LayerNorm(hidden_size=D, eps=EPS, has_weights=False, has_bias=False, dtype=torch.bfloat16).to(dev)
with torch.no_grad():
    ln.weight.copy_((1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16))
    ln.bias.copy_((0.05 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16))


def heavy(*shape):
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


def make_cases():
    cases = []
    for t in (80, 160):
        s = 4096 + t
        x = heavy(1, s, D)
        mod = (torch.randn(1, 2, 9 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        shift_t, scale_t = mod[..., D : 2 * D], mod[..., 4 * D : 5 * D]
        select = torch.zeros(1, s, 1, dtype=torch.bool, device=dev)
        select[:, :t] = True
        cases.append((f"single_T{t}", "single", (x, shift_t, scale_t, select)))
    x = heavy(1, 4096, D)
    table = (torch.randn(1, 16, 3 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    for k in (0, 5, 15):
        row = table[:, k : k + 1]
        cases.append((f"mode_step{k}", "mode", (x, row[..., :D], row[..., D : 2 * D])))
    return cases


_compiled = {}


def reference(kind, args, gs_float):
    from flashinfer import SfLayout
    from flashinfer import nvfp4_quantize as quantize

    if kind not in _compiled:
        mod = Single() if kind == "single" else Mode()
        _compiled[kind] = torch.compile(mod, dynamic=None, options={"emulate_precision_casts": True})
    with torch.inference_mode():
        xm = _compiled[kind](*args)
        data, sf = quantize(
            xm.reshape(-1, D), gs_float, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
            backend="cute-dsl",
        )
    return xm, data.view(torch.uint8).contiguous(), sf.view(torch.uint8).reshape(-1).contiguous()


def fused(kind, args, gs_t, store, poison=True):
    if kind == "single":
        x, sh, sc, sel = args
    else:
        x, sh, sc = args
        sel = None
    m = x.numel() // D
    outs = None
    if poison:
        outs = (
            torch.full_like(x, float("nan")),
            torch.full((m, D // 2), 0xAB, dtype=torch.uint8, device=dev),
            torch.full((faq._scale_size(m, D),), 0xAB, dtype=torch.uint8, device=dev),
        )
    kw = {"store_x_mod": store} if HAS_SKIP else {}
    with torch.inference_mode():
        return faq._launch(x, sel, sc, sh, ln.weight, ln.bias, gs_t, EPS, outs=outs, **kw)


def gs_for(kind, args):
    xm0, _, _ = reference(kind, args, 1.0)
    gs_float = float(torch.tensor(2688.0 / xm0.float().abs().max().item(), dtype=torch.float32))
    return gs_float, torch.tensor([gs_float], dtype=torch.float32, device=dev)


def run_check():
    assert HAS_SKIP, "candidate tree required"
    ok = True
    res = {}
    for name, kind, args in make_cases():
        gs_float, gs_t = gs_for(kind, args)
        ref = reference(kind, args, gs_float)
        store = fused(kind, args, gs_t, True)
        skip = fused(kind, args, gs_t, False)
        rec = {
            "store_x_mod_bitwise_ref": torch.equal(store[0], ref[0]),
            "store_fp4_bitwise_ref": torch.equal(store[1], ref[1]),
            "store_sf_bitwise_ref": torch.equal(store[2], ref[2]),
            "skip_fp4_bitwise_store": torch.equal(skip[1], store[1]),
            "skip_sf_full_bitwise_store": torch.equal(skip[2], store[2]),
            "skip_fp4_bitwise_ref": torch.equal(skip[1], ref[1]),
            "skip_sf_full_bitwise_ref": skip[2].numel() == ref[2].numel() and torch.equal(skip[2], ref[2]),
            # Skip must not write x_mod: the poisoned NaN buffer stays all-NaN.
            "skip_x_mod_untouched": bool(torch.isnan(skip[0]).all().item()),
        }
        gs_big = float(torch.tensor(gs_float * 4.0, dtype=torch.float32))
        gs_big_t = torch.tensor([gs_big], dtype=torch.float32, device=dev)
        ref_b = reference(kind, args, gs_big)
        skip_b = fused(kind, args, gs_big_t, False)
        rec["sat_skip_fp4_bitwise_ref"] = torch.equal(skip_b[1], ref_b[1])
        rec["sat_skip_sf_bitwise_ref"] = torch.equal(skip_b[2], ref_b[2])
        # Untimed repeat sequence: 20 iterations (same as the timed graph's per-input reuse),
        # every iteration vs iter 0 and the reference; inputs unchanged vs pristine copies.
        pristine = [a.clone() for a in args]
        first = None
        rep = True
        for it in range(20):
            out = fused(kind, args, gs_t, False, poison=(it % 2 == 0))
            if first is None:
                first = out
            rep &= torch.equal(out[1], first[1]) and torch.equal(out[2], first[2])
            rep &= torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2])
        rep &= all(torch.equal(a, p) for a, p in zip(args, pristine))
        rec["repeat_bitwise"] = bool(rep)
        rec["passed"] = all(bool(v) for v in rec.values())
        ok &= rec["passed"]
        res[name] = rec
        print(name, json.dumps(rec), flush=True)
    print(json.dumps({
        "passed": bool(ok),
        "compared_against": "store kernel (branch 66) and main's pre-66 Inductor LN+modulate "
        "(emulate_precision_casts) + FlashInfer nvfp4_quantize cute-dsl 128x4; fp4 + full "
        "poisoned SF buffer bitwise",
        "cases": res,
    }))


def run_bench():
    arm = "candidate" if HAS_SKIP else "baseline"
    results = {}
    nsets = 8
    for name, kind, args in make_cases():
        if name not in ("single_T80", "mode_step5"):
            continue
        _, gs_t = gs_for(kind, args)
        sets = [(args[0].roll(i * 7, dims=1).contiguous(),) + tuple(args[1:]) for i in range(nsets)]

        def fn(a):
            return fused(kind, a, gs_t, False, poison=False)

        with torch.inference_mode():
            for a in sets:
                fn(a)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for a in sets:
                    fn(a)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(4):
                    for a in sets:
                        fn(a)
            calls = 4 * nsets
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            times = []
            for _ in range(60):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                graph.replay()
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e) / calls)
        times.sort()
        med = statistics.median(times)
        p10, p90 = times[len(times) // 10], times[(9 * len(times)) // 10]
        results[name] = {"ms": med, "noise": (p90 - p10) / med}
        print(arm, name, json.dumps(results[name]), flush=True)
    print(json.dumps({
        "batch": {"1": {"kernel_ms": results["single_T80"]["ms"],
                        "noise_floor": results["single_T80"]["noise"]}},
        "output": {"class": "exact"},
        "arm": arm,
        "detail": results,
        "note": "single_T80 = one _adaln_nvfp4_kernel call at M=4176 in a CUDA graph (32 calls over "
        "8 rotating 25.7 MB inputs, cold L2), median of 60 replays; baseline stores bf16 x_mod, "
        "candidate skips it (fp4/SF bitwise)",
    }))


if MODE == "check":
    run_check()
else:
    run_bench()
