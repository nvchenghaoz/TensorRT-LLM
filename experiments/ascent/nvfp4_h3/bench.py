"""Per-rank H3 DiT GEMM microbenchmark: BF16 F.linear vs dynamic NVFP4 (W4A4).

The NVFP4 path mirrors NVFP4LinearMethod._input_prepare + apply under
force_dynamic_quantization with use_tunable_quantize (the visual-gen setting):
  amax = torch.amax(torch.abs(x)).float(); scale = 448*6/amax
  act_fp4, act_sf = trtllm.tunable_fp4_quantize(x, scale, 16, False)
  out = trtllm.nvfp4_gemm(act_fp4, w_fp4, act_sf, w_sf, alpha, bf16, bias=...)
Weights are quantized like visual_gen/quantization/ops.quantize_nvfp4 (one
global scale per weight; swizzled block scales as after loading).

Shapes (TP2 rank, M = 37966 packed rows at 768x1344x124f; census knowledge 275):
  qkv 10752x5376, out 5376x3584, gate_up 14336x5376, down 5376x7168,
  adaln 96768x2688 (+bias) at M=6.

Usage:
  bench.py --mode bf16|nvfp4            timed run, prints evidence JSON
  bench.py --mode nvfp4 --check         untimed correctness/repeat check
  bench.py --amax                       amax(abs) eager vs torch.compile vs aminmax
"""
import argparse
import json
import sys

import torch
import torch.nn.functional as F

M = 37966
SHAPES = [  # name, M, N, K, bias
    ("qkv", M, 10752, 5376, False),
    ("out", M, 5376, 3584, False),
    ("gate_up", M, 14336, 5376, False),
    ("down", M, 5376, 7168, False),
    ("adaln", 6, 96768, 2688, True),
]
WARMUP, ITERS = 5, 20
FP8_MAX, E2M1_MAX = 448.0, 6.0


def log(*a):
    print(*a, flush=True)


def make_inputs(name, m, n, k, bias, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    if name == "down":
        a = torch.randn(m, k, device=dev, generator=g)
        b = torch.randn(m, k, device=dev, generator=g)
        x = F.silu(a) * b
    elif name == "adaln":
        x = F.silu(torch.randn(m, k, device=dev, generator=g) * 2.0)
    else:
        col_scale = torch.exp(torch.randn(k, device=dev, generator=g) * 0.5)
        col_shift = torch.randn(k, device=dev, generator=g) * 0.3
        x = torch.randn(m, k, device=dev, generator=g) * col_scale + col_shift
    x = x.to(torch.bfloat16).contiguous()
    w = (torch.randn(n, k, device=dev, generator=g) * (1.0 / k**0.5)).to(torch.bfloat16)
    bvec = (torch.randn(n, device=dev, generator=g) * 0.02).to(torch.bfloat16) if bias else None
    return x, w, bvec


class Nvfp4Linear:
    def __init__(self, w, bias):
        import tensorrt_llm  # noqa: F401  (registers trtllm ops)

        amax_w = w.float().abs().max()
        gsf = (FP8_MAX * E2M1_MAX / amax_w).float().reshape(1)
        self.w_fp4, self.w_sf = torch.ops.trtllm.fp4_quantize(w, gsf, 16, False)
        self.weight_scale_2 = (amax_w / (FP8_MAX * E2M1_MAX)).float()
        self.bias = bias

    def __call__(self, x):
        amax_input = torch.amax(torch.abs(x)).float()
        input_scale = FP8_MAX * E2M1_MAX / amax_input
        alpha = (amax_input / (FP8_MAX * E2M1_MAX)) * self.weight_scale_2
        act_fp4, act_sf = torch.ops.trtllm.tunable_fp4_quantize(x, input_scale, 16, False)
        return torch.ops.trtllm.nvfp4_gemm(
            act_fp4, self.w_fp4, act_sf, self.w_sf, alpha, torch.bfloat16,
            allowed_backends="cutlass,cublaslt,cutedsl,cuda_core", bias=self.bias)


def build(mode):
    ops = []
    for i, (name, m, n, k, bias) in enumerate(SHAPES):
        x, w, b = make_inputs(name, m, n, k, bias, 1234 + i)
        if mode == "bf16":
            fn = (lambda x, w=w, b=b: F.linear(x, w, b))
        else:
            fn = Nvfp4Linear(w, b)
        ops.append((name, x, w, b, fn))
        log(f"built {name} M={m} N={n} K={k}")
    if mode == "nvfp4":
        from tensorrt_llm._torch.autotuner import autotune
        with autotune(skip_dynamic_tuning_buckets=True):
            for name, x, _, _, fn in ops:
                fn(x)
        torch.cuda.synchronize()
        log("autotune done")
    return ops


def timed(ops):
    for _ in range(WARMUP):
        for _, x, _, _, fn in ops:
            fn(x)
    torch.cuda.synchronize()
    per = {name: [] for name, *_ in ops}
    totals = []
    for it in range(ITERS):
        evs = []
        for name, x, _, _, fn in ops:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn(x)
            e.record()
            evs.append((name, s, e))
        torch.cuda.synchronize()
        tot = 0.0
        for name, s, e in evs:
            t = s.elapsed_time(e)
            per[name].append(t)
            tot += t
        totals.append(tot)
    return per, totals


def q(v, p):
    v = sorted(v)
    i = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
    return v[i]


def run_timed(mode):
    ops = build(mode)
    per, totals = timed(ops)
    med = q(totals, 0.5)
    noise = (q(totals, 0.9) - q(totals, 0.1)) / med
    detail = {n: {"median_ms": round(q(v, 0.5), 4),
                  "spread": round((q(v, 0.9) - q(v, 0.1)) / q(v, 0.5), 4)} for n, v in per.items()}
    for n, d in detail.items():
        log(f"{mode} {n}: {d}")
    log(f"{mode} layer GEMM total median {med:.3f} ms, noise {noise:.4f}")
    cls = "exact" if mode == "bf16" else "mode_switch"
    out = {"batch": {"1": {"step_ms": round(med, 4), "noise_floor": round(noise, 4)}},
           "output": {"class": cls}, "mode": mode, "per_gemm": detail,
           "note": "one H3 DiT layer's GEMMs (+AdaLN) per TP2 rank; nvfp4 includes amax+quantize"}
    print(json.dumps(out))


def rel_l2(a, ref):
    return ((a.float() - ref).norm() / ref.norm()).item()


def run_check():
    torch.backends.cuda.matmul.allow_tf32 = False
    ops = build("nvfp4")
    results = {}
    passed = True
    for name, x, w, b, fn in ops:
        x0 = x.clone()
        ref = torch.matmul(x.float(), w.float().t())
        if b is not None:
            ref = ref + b.float()
        bf = F.linear(x, w, b)
        first = fn(x)
        torch.cuda.synchronize()
        # Same warmup/iteration sequence as timing, untimed; every output must
        # equal the first (deterministic) and the input must stay unchanged.
        bad = []
        for it in range(WARMUP + ITERS):
            o = fn(x)
            if not torch.equal(o, first):
                bad.append(it)
            if not torch.equal(x, x0):
                bad.append(f"input_changed@{it}")
        r = {
            "nvfp4_rel_l2": rel_l2(first, ref),
            "nvfp4_max_abs": (first.float() - ref).abs().max().item(),
            "bf16_rel_l2": rel_l2(bf, ref),
            "bf16_max_abs": (bf.float() - ref).abs().max().item(),
            "nvfp4_vs_bf16_rel_l2": rel_l2(first, bf.float()),
            "repeat_mismatch_iters": bad,
            "finite": bool(torch.isfinite(first).all().item()),
        }
        # Sanity bound: NVFP4 per-GEMM error expected ~0.13 (census 275).
        ok = (not bad) and r["finite"] and r["nvfp4_rel_l2"] < 0.25
        passed = passed and ok
        results[name] = r
        log(name, json.dumps(r))
        del ref, bf, first
        torch.cuda.empty_cache()
    print(json.dumps({"passed": passed, "output_class": "mode_switch",
                      "compared": "dynamic NVFP4 Linear vs FP32 torch.matmul of same BF16 inputs; BF16 F.linear error alongside; 25 untimed reruns bitwise vs first",
                      "results": results}))


def run_amax():
    """Is torch.amax(torch.abs(x)) one reduction under torch.compile?"""
    res = {}
    for name, m, n, k, bias in SHAPES[:4]:
        x, _, _ = make_inputs(name, m, n, k, bias, 7)

        def eager(x):
            return torch.amax(torch.abs(x)).float()

        comp = torch.compile(eager, dynamic=False)

        def mm(x):
            lo, hi = torch.aminmax(x)
            return torch.maximum(-lo, hi).float()

        r = {}
        for lbl, f in (("eager_abs_amax", eager), ("compiled_abs_amax", comp), ("aminmax", mm)):
            for _ in range(WARMUP):
                f(x)
            torch.cuda.synchronize()
            ts = []
            for _ in range(ITERS):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); f(x); e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            r[lbl] = round(q(ts, 0.5), 4)
        assert torch.equal(eager(x), comp(x)) and torch.equal(eager(x), mm(x))
        res[name] = r
        log(name, r)
    print(json.dumps({"amax_ms": res}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="nvfp4", choices=["bf16", "nvfp4"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--amax", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_device(0)
    log("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    if a.amax:
        run_amax()
    elif a.check:
        run_check()
    else:
        run_timed(a.mode)
    sys.stdout.flush()
