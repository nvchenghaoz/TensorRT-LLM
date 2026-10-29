"""Idea 17 run_gpu_pair unit: one denoise step's routed GEMMs at the served 2048^2 shapes.

Same script in both trees (base = main: Pingpong 128x128x128 QKV3, 128x128x128 LPT pair, FP8R
arm 1; branch = large-M routes). Calls go through the served entry points only:
flux3_sm120_strided_gemm.gemm_qkv3, flux3_sm120_lpt_pair.run, flux3_fp8r_gemm.gemm (default arm).
SwiGLU gate_up is not routed (identical in both trees) and not in the unit.

Cases (one CUDA graph each = one denoise step's routed GEMM calls, NSETS=2 input sets per M
alternating, so A + W + out per call exceed the 128 MB L2):
  t2i-2048: 28 single blocks @ M 16896 + 4 image mode blocks @ M 16384 (QKV3 + tail pair each)
            + image.0 FP8R (QKV, gate_up, down, attn_out) @ M 16384.
  i2i-2048: 28 single blocks @ M 33280 + 8 mode blocks (image 1-4, image_cond 1-4) @ M 16384
            + image.0 FP8R @ M 16384.
Inputs: NVFP4 quantized from BF16 normals (flashinfer nvfp4_quantize, 128x4 scales, per-plane
global scales), FP8R from quantize_fp8r. Fixed seeds; pristine copies checked after timing.
Timing: 5 warm replays, then ROUNDS replays per case, cases interleaved; kernel_ms = median per
replay (one step), noise_floor = (p90 - p10) / median.
"""

import json
import os
import sys
import time

import torch

ROUNDS = int(os.environ.get("BENCH_ROUNDS", "30"))
NSETS = 2
H, FF = 3072, 9216


def log(*a):
    print(*a, flush=True)


def _quant(x):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    gs = float(x.abs().amax().float() / (448.0 * 6.0))
    inv = torch.tensor([1.0 / gs], device="cuda", dtype=torch.float32)
    q, sf = fp.nvfp4_quantize(x, inv)
    return q, sf, gs


def nvfp4_inputs(m, g):
    """QKV3 (3 planes, own global scales) + down + attn_out operands at M rows."""
    x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16, generator=g)
    a, a_sf, a_gs = _quant(x)
    planes = [_quant(torch.randn(H, H, device="cuda", dtype=torch.bfloat16, generator=g) * s)
              for s in (0.02, 0.035, 0.012)]
    wqkv = torch.cat([p[0] for p in planes], 0).contiguous()
    sqkv = torch.cat([p[1] for p in planes], 0).contiguous()
    aqkv = torch.tensor([a_gs * p[2] for p in planes], device="cuda", dtype=torch.float32)
    xd = torch.randn(m, FF, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
    d, d_sf, d_gs = _quant(xd)
    wd, swd, wd_gs = _quant(torch.randn(H, FF, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
    wo, swo, wo_gs = _quant(torch.randn(H, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
    return {
        "m": m, "a": a, "a_sf": a_sf, "wqkv": wqkv, "sqkv": sqkv, "aqkv": aqkv,
        "qkv_out": torch.empty(3 * m * H, device="cuda", dtype=torch.bfloat16),
        "d": d, "d_sf": d_sf, "wd": wd, "swd": swd,
        "alpha_d": torch.tensor([d_gs * wd_gs], device="cuda", dtype=torch.float32),
        "wo": wo, "swo": swo,
        "alpha_o": torch.tensor([a_gs * wo_gs], device="cuda", dtype=torch.float32),
        "down_out": torch.empty(m, H, device="cuda", dtype=torch.bfloat16),
        "attn_out": torch.empty(m, H, device="cuda", dtype=torch.bfloat16),
    }


def fp8r_inputs(m, g):
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import quantize_fp8r_weight

    res = {}
    for name, (n, k) in {"qkv": (3 * H, H), "gate_up": (2 * FF, H), "down": (H, FF),
                         "attn_out": (H, H)}.items():
        x = (torch.randn((m, k), device="cuda", generator=g) * 1.5).to(torch.bfloat16)
        x[:, :8] *= 20
        w = (torch.randn((n, k), device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        data, scales = quantize_fp8r(x)
        wd, ws = quantize_fp8r_weight(w)
        res[name] = (data.contiguous(), wd, scales.float().reshape(-1).contiguous(), ws,
                     torch.empty((m, n), device="cuda", dtype=torch.bfloat16))
    return res


def qkv3(c):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    strided.gemm_qkv3(c["a"], c["wqkv"], c["a_sf"], c["sqkv"], c["aqkv"], c["qkv_out"])


def pair(c):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

    lpt.run(c["d"], c["wd"], c["d_sf"], c["swd"], c["alpha_d"], c["down_out"],
            c["a"], c["wo"], c["a_sf"], c["swo"], c["alpha_o"], c["attn_out"])


def fp8r(f):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg

    for name in ("qkv", "gate_up", "down", "attn_out"):
        a, w, sa, sw, out = f[name]
        fg.gemm(a, w, sa, sw, out=out)


def preload():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda", torch.cuda.current_device())
    fp.sm120_linear_preload(dev)
    strided.preload(dev)
    lpt.preload(dev)
    fg.preload(dev)


def case_calls(sets, f8, singles_m, n_single, mode_m, n_mode):
    calls = []
    for i in range(n_single):
        c = sets[singles_m][i % NSETS]
        calls += [lambda c=c: qkv3(c), lambda c=c: pair(c)]
    for i in range(n_mode):
        c = sets[mode_m][i % NSETS]
        calls += [lambda c=c: qkv3(c), lambda c=c: pair(c)]
    calls.append(lambda: fp8r(f8))
    return calls


def capture(calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in calls:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    torch.cuda.synchronize()
    return g


def tensors_of(sets, f8):
    out = []
    for lst in sets.values():
        for c in lst:
            out += [v for k, v in c.items() if isinstance(v, torch.Tensor) and not k.endswith("_out")]
    for v in f8.values():
        out += list(v[:4])
    return out


def main():
    import tensorrt_llm  # noqa: F401

    t0 = time.time()
    preload()
    try:
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_large_m as lm

        log(f"tree: branch (large-M routes {sorted(lm._enabled_routes())}, min M {lm.LARGE_M_MIN})")
    except ImportError:
        log("tree: base (no flux3_sm120_large_m)")
    log(f"preload {time.time() - t0:.0f}s")
    g = torch.Generator(device="cuda")
    g.manual_seed(2048)
    sets = {m: [nvfp4_inputs(m, g) for _ in range(NSETS)] for m in (16384, 16896, 33280)}
    f8 = fp8r_inputs(16384, g)
    pristine = [t.clone() for t in tensors_of(sets, f8)]
    cases = {
        "t2i-2048": case_calls(sets, f8, 16896, 28, 16384, 4),
        "i2i-2048": case_calls(sets, f8, 33280, 28, 16384, 8),
    }
    graphs = {k: capture(v) for k, v in cases.items()}
    log(f"captured {list(graphs)} ({time.time() - t0:.0f}s)")
    for _ in range(5):
        for gr in graphs.values():
            gr.replay()
    torch.cuda.synchronize()
    times = {k: [] for k in graphs}
    names = list(graphs)
    for r in range(ROUNDS):
        order = names if r % 2 == 0 else names[::-1]
        for k in order:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[k].replay()
            e1.record()
            e1.synchronize()
            times[k].append(e0.elapsed_time(e1))
        if r % 10 == 0:
            log(f"round {r}: " + " ".join(f"{k}={times[k][-1]:.3f}ms" for k in names))
    ok = all(torch.equal(a, b) for a, b in zip(pristine, tensors_of(sets, f8)))
    batch = {}
    for k, v in times.items():
        v = sorted(v)
        med = v[len(v) // 2]
        p10, p90 = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1))]
        batch[k] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med}
        log(f"{k}: median {med:.3f} ms/step p10 {p10:.3f} p90 {p90:.3f}")
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "inputs_unchanged": ok,
                      "unit": "one CUDA graph replay = one denoise step's QKV3 + tail-pair calls "
                              "(28 single + 4/8 mode blocks) + image.0's 4 FP8R GEMMs"}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
