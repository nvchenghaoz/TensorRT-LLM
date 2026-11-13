"""Idea 141: Sage quantizer JIT variants per packed length S, and bitwise vs the pinned kernels.

The overlay's flashinfer/cute_dsl/sparse/bsa_utils/sage_quant_sm120.py marks the S-derived int
arguments do_not_specialize. The pinned module is rebuilt here from the overlay's own text with those
decorators reverted to a bare @triton.jit (so the comparison is the exact pinned kernel source), and
loaded as a separate module. For each S (served layout: Q/K contiguous NHD, V a strided view of a
packed QKV buffer, all passed as HND views): all six quantizer outputs bitwise equal, overlay vs
pinned; compiled-variant count per kernel for both; quantize time at S=37966, interleaved.
Last line: JSON {passed, ...}.
"""
import importlib.util
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import torch

H, D = 28, 128
S_LIST = [37966, 37769, 37760, 38016, 37888, 37633, 37841, 37905, 37999, 38017, 4097, 12032, 1870]
dev = torch.device("cuda")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_pinned():
    import flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 as ov

    text = Path(ov.__file__).read_text()
    n_dec = len(re.findall(r"@triton\.jit\(do_not_specialize=\[[^\]]*\]\)", text))
    pinned_text = re.sub(r"@triton\.jit\(do_not_specialize=\[[^\]]*\]\)", "@triton.jit", text)
    tmp = Path(tempfile.mkdtemp()) / "sage_quant_pinned.py"
    tmp.write_text(pinned_text)
    spec = importlib.util.spec_from_file_location("sage_quant_pinned", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sage_quant_pinned"] = mod
    spec.loader.exec_module(mod)
    return ov, mod, n_dec


KERNELS = ["_quantize_sage_q_kernel", "_sage_kv_stats_partial_kernel", "_sage_kv_stats_finalize_kernel",
           "_quantize_sage_kv_kernel"]


def variants(mod):
    out = {}
    for name in KERNELS:
        fn = getattr(mod, name)
        n = None
        try:
            caches = fn.device_caches
            n = sum(len(v[0]) for v in caches.values())
        except Exception:  # noqa: BLE001
            try:
                n = sum(len(v) for v in fn.cache.values())
            except Exception:  # noqa: BLE001
                n = None
        out[name] = n
    return out


def inputs(s, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(1, s, H, D, generator=g, device=dev, dtype=torch.bfloat16) * 2
    k = torch.randn(1, s, H, D, generator=g, device=dev, dtype=torch.bfloat16) * 3 + 0.5
    qkv = torch.randn(1, s, 3, H, D, generator=g, device=dev, dtype=torch.bfloat16)
    v = qkv[:, :, 2]
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def main():
    ov, pin, n_dec = load_pinned()
    log("overlay decorators with do_not_specialize:", n_dec)
    res = {"per_S": {}, "overlay_decorators": n_dec}
    ok = n_dec == 4
    for i, s in enumerate(S_LIST):
        q, k, v = inputs(s, 100 + i)
        a = ov.quantize_sage_qkv_sm120(q, k, v)
        b = pin.quantize_sage_qkv_sm120(q, k, v)
        torch.cuda.synchronize()
        eq = [bool(torch.equal(x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x,
                               y.view(torch.uint8) if y.dtype == torch.float8_e4m3fn else y))
              for x, y in zip(a, b)]
        # repeat the overlay call: deterministic
        a2 = ov.quantize_sage_qkv_sm120(q, k, v)
        rep = all(bool(torch.equal(x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x,
                                   y.view(torch.uint8) if y.dtype == torch.float8_e4m3fn else y))
                  for x, y in zip(a, a2))
        res["per_S"][s] = {"bitwise_vs_pinned": eq, "repeat_bitwise": rep,
                           "variants_overlay": variants(ov), "variants_pinned": variants(pin)}
        ok = ok and all(eq) and rep
        log(s, res["per_S"][s])
        del a, b, a2, q, k, v
    res["variants_overlay_final"] = variants(ov)
    res["variants_pinned_final"] = variants(pin)
    one_variant = all(n == 1 for n in res["variants_overlay_final"].values())
    # timing at the protocol shape, interleaved
    q, k, v = inputs(37966, 7)
    times = {"overlay": [], "pinned": []}
    for _ in range(5):
        ov.quantize_sage_qkv_sm120(q, k, v)
        pin.quantize_sage_qkv_sm120(q, k, v)
    torch.cuda.synchronize()
    for r in range(40):
        for name, m in (("overlay", ov), ("pinned", pin)) if r % 2 == 0 else (("pinned", pin), ("overlay", ov)):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            m.quantize_sage_qkv_sm120(q, k, v)
            e1.record()
            e1.synchronize()
            times[name].append(e0.elapsed_time(e1))
    st = {}
    for n, t in times.items():
        t = sorted(t)
        st[n] = {"median_ms": t[len(t) // 2], "p10": t[4], "p90": t[35]}
    res["quantize_time_S37966"] = st
    log("timing", st)
    print(json.dumps({"passed": bool(ok and one_variant), "one_variant_per_kernel": one_variant,
                      "compared": "overlay (do_not_specialize) vs pinned Sage quantizers, all 6 outputs bitwise",
                      **res}, default=str))


if __name__ == "__main__":
    main()
