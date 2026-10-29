"""Branch 132 (idea 153): _adaln_nvfp4_kernel keeps the row's x chunks in registers (no 2nd x read).

  op_check.py check -> in-process: fused op (store / skip x_mod) vs main's contract reference
                       (Inductor LN+mod with emulate_precision_casts + FlashInfer nvfp4_quantize
                       cute-dsl 128x4): fp4 + FULL poisoned SF buffer bitwise at M=4176, 4256
                       (single tables), 4096 and 80 (mode rows), 4x saturating global scale;
                       forced-store x_mod bitwise; untimed repeat sequence (same 5 warmup + 20
                       iterations as one timed graph's per-case calls) every iteration == iter 0
                       == reference; inputs == pristine copies. Also prints sha256 of outputs so
                       the pair's two arms can be compared (bitwise vs main's kernel).
  op_check.py bench -> served mix in ONE CUDA graph: 448 calls at M=4176 (single, select) +
                       64 at M=4096 (mode rows) = 512 calls, x L2-hot as in-model; >=10 s soak,
                       200 timed replays. kernel_ms = graph time / 512 (mean us/call of the mix).
"""

import hashlib
import json
import statistics
import sys
import time

import torch

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as faq

D = 3072
EPS = 1e-6
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(1234)
CAND = hasattr(faq, "_row_cached")
ARM = "candidate" if CAND else "baseline"
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
        cases.append((f"single_M{s}", "single", (x, shift_t, scale_t, select)))
    x = heavy(1, 4096, D)
    table = (torch.randn(1, 16, 3 * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    for k in (0, 15):
        row = table[:, k : k + 1]
        cases.append((f"mode_M4096_step{k}", "mode", (x, row[..., :D], row[..., D : 2 * D])))
    x80 = heavy(1, 80, D)
    row = table[:, 5:6]
    cases.append(("mode_M80", "mode", (x80, row[..., :D], row[..., D : 2 * D])))
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


def fused(kind, args, gs_t, store, poison=True, outs=None):
    if kind == "single":
        x, sh, sc, sel = args
    else:
        x, sh, sc = args
        sel = None
    m = x.numel() // D
    if poison:
        outs = (
            torch.full_like(x, float("nan")),
            torch.full((m, D // 2), 0xAB, dtype=torch.uint8, device=dev),
            torch.full((faq._scale_size(m, D),), 0xAB, dtype=torch.uint8, device=dev),
        )
    with torch.inference_mode():
        return faq._launch(x, sel, sc, sh, ln.weight, ln.bias, gs_t, EPS, outs=outs, store_x_mod=store)


def gs_for(kind, args):
    xm0, _, _ = reference(kind, args, 1.0)
    gs_float = float(torch.tensor(2688.0 / xm0.float().abs().max().item(), dtype=torch.float32))
    return gs_float, torch.tensor([gs_float], dtype=torch.float32, device=dev)


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def run_check():
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
            "store_sf_full_bitwise_ref": torch.equal(store[2], ref[2]),
            "skip_fp4_bitwise_ref": torch.equal(skip[1], ref[1]),
            "skip_sf_full_bitwise_ref": skip[2].numel() == ref[2].numel() and torch.equal(skip[2], ref[2]),
            "skip_x_mod_untouched": bool(torch.isnan(skip[0]).all().item()),
        }
        gs_big = float(torch.tensor(gs_float * 4.0, dtype=torch.float32))
        gs_big_t = torch.tensor([gs_big], dtype=torch.float32, device=dev)
        ref_b = reference(kind, args, gs_big)
        skip_b = fused(kind, args, gs_big_t, False)
        rec["sat4x_fp4_bitwise_ref"] = torch.equal(skip_b[1], ref_b[1])
        rec["sat4x_sf_full_bitwise_ref"] = torch.equal(skip_b[2], ref_b[2])
        # Untimed repeat sequence: 5 warmup + 20 iterations, poisoned outputs re-made on even
        # iterations (reset points), each iteration vs iteration 0 and the reference.
        pristine = [a.clone() for a in args]
        first = None
        rep = True
        for it in range(25):
            out = fused(kind, args, gs_t, it % 3 == 0, poison=(it % 2 == 0),
                        outs=None if it % 2 == 0 else out)
            if first is None:
                first = out
                first = tuple(t.clone() for t in out)
            rep &= torch.equal(out[1], first[1]) and torch.equal(out[2], first[2])
            rep &= torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2])
            if it % 3 == 0:
                rep &= torch.equal(out[0], ref[0])
        rep &= all(torch.equal(a, p) for a, p in zip(args, pristine))
        rec["repeat_bitwise"] = bool(rep)
        rec["passed"] = all(bool(v) for v in rec.values())
        rec["sha_x_mod"] = h(store[0])
        rec["sha_fp4"] = h(store[1])
        rec["sha_sf"] = h(store[2])
        rec["sha_sat_fp4"] = h(skip_b[1])
        rec["sha_sat_sf"] = h(skip_b[2])
        ok &= rec["passed"]
        res[name] = rec
        print(ARM, name, json.dumps(rec), flush=True)
    print(json.dumps({
        "passed": bool(ok),
        "arm": ARM,
        "compared_against": "main's contract reference: Inductor LN+modulate (emulate_precision_casts) "
        "+ FlashInfer nvfp4_quantize cute-dsl 128x4; fp4 + full poisoned SF buffer + forced-store "
        "x_mod bitwise; sha256 prefixes printed for cross-arm comparison with main's kernel",
        "cases": res,
    }))


def run_bench():
    cases = {n: (k, a) for n, k, a in make_cases()}
    k1, a1 = cases["single_M4176"]
    k2, a2 = cases["mode_M4096_step0"]
    _, gs1 = gs_for(k1, a1)
    _, gs2 = gs_for(k2, a2)
    # Hash the served-mix outputs so the two arms can be compared bitwise.
    o1 = fused(k1, a1, gs1, False)
    o2 = fused(k2, a2, gs2, False)
    hashes = {"M4176_fp4": h(o1[1]), "M4176_sf": h(o1[2]), "M4096_fp4": h(o2[1]), "M4096_sf": h(o2[2])}
    print(ARM, "hashes", json.dumps(hashes), flush=True)
    outs1 = tuple(t.clone() for t in o1)
    outs2 = tuple(t.clone() for t in o2)
    n1, n2 = 448, 64

    def body():
        # 16 steps x (28 single at M=4176 + 4 mode at M=4096) = 512 calls.
        for _ in range(16):
            for _ in range(n1 // 16):
                fused(k1, a1, gs1, False, poison=False, outs=outs1)
            for _ in range(n2 // 16):
                fused(k2, a2, gs2, False, poison=False, outs=outs2)

    with torch.inference_mode():
        body()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        calls = n1 + n2
        t0 = time.time()
        soak = 0
        while time.time() - t0 < 12.0:
            graph.replay()
            soak += 1
            if soak % 200 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        print(ARM, "soak replays", soak, flush=True)
        times = []
        for i in range(200):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            graph.replay()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e) / calls)
            if i % 50 == 0:
                print(ARM, "sample", i, times[-1] * 1000, "us/call", flush=True)
        # Per-M-class detail (diagnostic): one graph per class, 64 calls each, 100 replays.
        detail = {}
        for cname, kk, aa, gg, oo in (("M4176_single", k1, a1, gs1, outs1),
                                      ("M4096_mode", k2, a2, gs2, outs2)):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(64):
                    fused(kk, aa, gg, False, poison=False, outs=oo)
            for _ in range(20):
                gr.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(100):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                gr.replay()
                e.record()
                e.synchronize()
                ts.append(s.elapsed_time(e) / 64 * 1000)
            ts.sort()
            detail[cname] = {"us_median": statistics.median(ts), "us_p10": ts[10], "us_p90": ts[90]}
            print(ARM, cname, json.dumps(detail[cname]), flush=True)
    # Post-timing outputs must still equal the pre-timing hash (same inputs, same op).
    stable = h(outs1[1]) == hashes["M4176_fp4"] and h(outs2[2]) == hashes["M4096_sf"]
    st = sorted(times)
    med = statistics.median(st)
    p10, p90 = st[len(st) // 10], st[(9 * len(st)) // 10]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": (p90 - p10) / med}},
        "output": {"class": "exact"},
        "arm": ARM,
        "samples": len(st),
        "p10_ms": p10,
        "p90_ms": p90,
        "hashes": hashes,
        "per_class": detail,
        "post_timing_outputs_stable": bool(stable),
        "note": "served mix in one CUDA graph: 448 calls M=4176 single-table + 64 calls M=4096 "
        "mode-row (store_x_mod=False, as served), x L2-hot; kernel_ms = graph time / 512; "
        "12 s soak, 200 replays, no clock lock",
    }))


if MODE == "check":
    run_check()
else:
    run_bench()
