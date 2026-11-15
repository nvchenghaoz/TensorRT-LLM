"""Post-attention junction: rmsnorm_residual_add + pre-ffn rmsnorm_fp4_quant
(main's SM120 chain) vs the fused rmsnorm_residual_add_fp4_quant.

usage: mf2_bench.py time|check|divexp <checkpoint>
time: CUDA-event timing of whichever path this tree has (fused if present).
check: candidate tree only; bitwise/agreement vs main's chain, FP32 ref,
       repeated-execution sequence, input immutability.
"""
import json
import os
import sys

import torch

MODE = sys.argv[1]
CK = sys.argv[2] if len(sys.argv) > 2 else None

H, EPS = 5376, 1e-6
if CK and os.path.exists(os.path.join(CK, "config.json")):
    cfg = json.load(open(os.path.join(CK, "config.json")))
    tc = cfg.get("text_config", cfg)
    H = int(tc["hidden_size"])
    EPS = float(tc.get("rms_norm_eps", EPS))
print(f"hidden={H} eps={EPS}", flush=True)

from tensorrt_llm._torch.kernels.fused_ops import rmsnorm_residual_add as rra  # noqa: E402
from tensorrt_llm._torch.kernels.fused_ops.rmsnorm_fp4_quant import (  # noqa: E402
    rmsnorm_fp4_quant,
    rmsnorm_fp4_quant_available,
)

FUSED = getattr(rra, "rmsnorm_residual_add_fp4_quant", None)
print("fused available:", FUSED is not None, "cute available:", rmsnorm_fp4_quant_available(), flush=True)

CASES = [
    ("300", 300), ("1000", 1000), ("1200", 1200), ("2000", 2000), ("4000", 4000),
    ("4800", 4800), ("8000", 8000), ("8100", 8100), ("8192", 8192),
]
# The kernel is one program per row with no sequence boundaries, so packed
# mixes are their row totals: 16x300 = 4800, 4x2000 = 8000, 16x512 = 8192.
WARMUP, ITERS = 5, 30
dev = "cuda"


REAL = {}


def _real_weights():
    """Layer norm weights and gate_up input scale from the checkpoint."""
    if REAL or not CK:
        return REAL
    try:
        from safetensors import safe_open
        idx = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]

        def load(sub):
            k = [k for k in idx if k.endswith(sub)][0]
            with safe_open(os.path.join(CK, idx[k]), framework="pt", device="cpu") as f:
                return f.get_tensor(k)
        nl = int(tc["num_hidden_layers"])
        for li in sorted({0, 5, nl // 2, nl - 1}):
            wpa = load(f"layers.{li}.post_attention_layernorm.weight").to(torch.bfloat16).to(dev)
            wpf = load(f"layers.{li}.pre_feedforward_layernorm.weight").to(torch.bfloat16).to(dev)
            iv = load(f"layers.{li}.mlp.gate_proj.input_scale").float().item()
            REAL[li] = (wpa, wpf, torch.tensor([1.0 / iv], device=dev, dtype=torch.float32))
        print("real norm weights from layers", list(REAL), flush=True)
    except Exception as e:  # noqa: BLE001
        print("real weights unavailable:", repr(e), flush=True)
    return REAL


def make_inputs(m, seed=1234, layer=None):
    g = torch.Generator(device=dev).manual_seed(seed + m)
    real = _real_weights()
    # heavy-tailed channels like a real residual stream
    chs = torch.exp(torch.randn(H, device=dev, generator=g))
    x = (torch.randn(m, H, device=dev, generator=g) * 4.0 * chs).to(torch.bfloat16)
    r = (torch.randn(m, H, device=dev, generator=g) * 20.0 * chs.flip(0)).to(torch.bfloat16)
    if real:
        w, w2, gs = real[layer if layer in real else sorted(real)[len(real) // 2]]
        return x, r, w, w2, gs
    w = (torch.rand(H, device=dev, generator=g) * 2.0 + 0.05).to(torch.bfloat16)
    w2 = (torch.rand(H, device=dev, generator=g) * 2.0 + 0.05).to(torch.bfloat16)
    gs = torch.tensor([448.0 * 6.0 / 40.0], device=dev, dtype=torch.float32)
    return x, r, w, w2, gs


def chain(x, r, w, w2, gs):
    h = rra.rmsnorm_residual_add(x, r, w, EPS)
    fp4, sf = rmsnorm_fp4_quant(h, w2, EPS, gs)
    return h, fp4, sf


def fused(x, r, w, w2, gs):
    return FUSED(x, r, w, EPS, w2, EPS, gs)


flush_buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)


def time_case(fn, args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        flush_buf.fill_(1.0)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1))]
    return med, p10, p90


if MODE == "time":
    fn = fused if FUSED is not None else chain
    print("path:", "fused" if FUSED is not None else "chain", flush=True)
    out = {}
    for label, m in CASES:
        args = make_inputs(m)
        med, p10, p90 = time_case(fn, args)
        print(f"M={label:>7} median={med*1000:8.1f}us p10={p10*1000:8.1f} p90={p90*1000:8.1f}", flush=True)
        out[label] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med, "p10_ms": p10, "p90_ms": p90}
        del args
    print(json.dumps({"batch": out, "output": {"class": "exact"}}))
    sys.exit(0)

# ---------------- correctness ----------------
from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import sf_swizzled_offsets  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=dev)


def dequant(fp4, sf, gs, m):
    lo = (fp4 & 0xF).long()
    hi = (fp4 >> 4).long()
    vals = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(m, H)
    off = sf_swizzled_offsets(m, H // 16, fp4.device)
    s = sf[off].view(torch.float8_e4m3fn).float().reshape(m, H // 16)
    return (vals.reshape(m, H // 16, 16) * s[:, :, None]).reshape(m, H) / gs.item()


def ref_fp32(h, w2):
    hf = h.float()
    rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS)
    return hf * rstd * w2.float()


def errs(a, ref):
    d = (a - ref)
    return d.abs().max().item(), (d.norm() / ref.norm()).item()


if MODE == "divexp":
    res = {}
    for flag in (False, True):
        rra._FP4_DIV_RN = flag
        tot = 0
        for label, m in CASES[:9]:
            args = make_inputs(m)
            hb, qb, sb = chain(*args)
            hc, qc, sc = fused(*args)
            torch.cuda.synchronize()
            tot += (qb != qc).sum().item()
        res[str(flag)] = tot
        print("DIV_RN", flag, "payload byte mismatches over cases:", tot, flush=True)
    print(json.dumps({"passed": True, "divexp": res}))
    sys.exit(0)

assert FUSED is not None
allok = True
report = {}
CHECK_CASES = [("1", 1), ("27", 27)] + CASES
LAYERS = sorted(_real_weights()) or [None]
for label, m in CHECK_CASES:
    # Byte identity on every real-weight layer, A (fused), B (chain), A again.
    aba = {}
    for li in LAYERS:
        a = make_inputs(m, seed=77 + (li or 0), layer=li)
        o1 = [t.clone() for t in fused(*a)]
        ob = chain(*a)
        o2 = fused(*a)
        torch.cuda.synchronize()
        aba[str(li)] = {
            "h_diff_elems": (o1[0].view(torch.int16) != ob[0].view(torch.int16)).sum().item(),
            "fp4_diff_bytes": (o1[1] != ob[1]).sum().item(),
            "sf_full_diff_bytes": (o1[2] != ob[2]).sum().item() if o1[2].shape == ob[2].shape else -1,
            "sf_bytes": o1[2].numel(),
            "A_A2_bitwise": all(torch.equal(x, y) for x, y in zip(o1, o2)),
        }
        del a, o1, ob, o2
    aba_ok = all(v["h_diff_elems"] == 0 and v["fp4_diff_bytes"] == 0 and v["sf_full_diff_bytes"] == 0
                 and v["A_A2_bitwise"] for v in aba.values())
    args = make_inputs(m)
    snap = [a.clone() for a in args]
    hb, qb, sb = chain(*args)
    torch.cuda.synchronize()
    # repeated-execution sequence identical to the timing run
    first = None
    rep_ok = True
    for i in range(WARMUP + ITERS):
        flush_buf.fill_(1.0)
        hc, qc, sc = fused(*args)
        if first is None:
            first = (hc.clone(), qc.clone(), sc.clone())
        else:
            rep_ok &= torch.equal(hc, first[0]) and torch.equal(qc, first[1]) and torch.equal(sc, first[2])
    torch.cuda.synchronize()
    inputs_ok = all(torch.equal(a, b) for a, b in zip(args, snap))
    hc, qc, sc = first
    h_bit = torch.equal(hc, hb)
    off = sf_swizzled_offsets(m, H // 16, hc.device)
    sf_valid_eq = torch.equal(sc[off], sb[off])
    sf_full_eq = sc.shape == sb.shape and torch.equal(sc, sb)
    qbytes_diff = (qc != qb).sum().item()
    nib_diff = (((qc & 0xF) != (qb & 0xF)).sum() + ((qc >> 4) != (qb >> 4)).sum()).item()
    nib_rate = nib_diff / (m * H)
    sf_diff = (sc[off] != sb[off]).sum().item()
    rows_diff = ((qc != qb).any(-1)).sum().item()
    # FP32 reference built from the same (bitwise-equal) h
    ref = ref_fp32(hb, args[3])
    dq_c = dequant(qc, sc, args[4], m)
    dq_b = dequant(qb, sb, args[4], m)
    ma_c, rf_c = errs(dq_c, ref)
    ma_b, rf_b = errs(dq_b, ref)
    finite = bool(torch.isfinite(dq_c).all().item())
    cand_vs_main_max = (dq_c - dq_b).abs().max().item()
    ulp = (ref.abs().max().item()) * 2 ** -8
    ref_ok = finite and ma_c <= 1.10 * ma_b + ulp and rf_c <= 1.10 * rf_b + 2 ** -8
    ok = (h_bit and rep_ok and inputs_ok and ref_ok and (qc.shape == qb.shape) and (sc.shape == sb.shape)
          and qbytes_diff == 0 and sf_full_eq and aba_ok)
    allok &= ok
    report[label] = dict(ok=ok, aba_real_layers=aba, h_bitwise=h_bit, sf_valid_bitwise=sf_valid_eq,
                         sf_full_bitwise=sf_full_eq,
                         payload_bitwise=qbytes_diff == 0, payload_nibble_diff=nib_diff,
                         payload_nibble_diff_rate=nib_rate, rows_with_diff=rows_diff, sf_diff=sf_diff,
                         repeated_ok=rep_ok, inputs_unchanged=inputs_ok,
                         cand_err_maxabs=ma_c, main_err_maxabs=ma_b, cand_err_relfro=rf_c,
                         main_err_relfro=rf_b, cand_vs_main_maxabs=cand_vs_main_max, ref_ok=ref_ok)
    print(label, json.dumps(report[label]), flush=True)
    del args, snap, hb, qb, sb, hc, qc, sc, first, ref, dq_c, dq_b
    torch.cuda.empty_cache()

print(json.dumps({"passed": bool(allok), "class": "exact",
                  "compared": "fused vs main chain (rmsnorm_residual_add + flashinfer CuTe rmsnorm_fp4quant): "
                              "h, FP4 payload and full swizzled SF (padding included) byte-identical on "
                              "real checkpoint norm weights/input scales, A,B,A bitwise, 35-call repeat "
                              "sequence bitwise, inputs unchanged, FP32 torch ref sanity",
                  "cases": report}))
