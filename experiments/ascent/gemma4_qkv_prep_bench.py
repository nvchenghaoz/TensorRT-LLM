"""Gemma4 QKV prep microbenchmark / correctness check on SM120 FA2 (FP8 KV).

Modes:
  bench  : times the tree's production QKV prep chain per layer shape and
           token mix. In a tree whose gemma4_fused_qkv_norm_rope_quant accepts
           ``kv_out_fp8`` (branch), the FA2 chain is the fused kernel with BF16
           Q + FP8 K/V; otherwise (main) it is main's unfused chain:
           split -> v_norm/q_norm/k_norm (RMSNorm.forward -> flashinfer_rmsnorm)
           -> flashinfer rope in place -> k/v .to(float8_e4m3fn).
  check  : (branch only) untimed repeated-execution sequence matching the
           timing run, comparing every call with the first call, with main's
           unfused chain, and with an independent FP32 torch reference.

Shapes are read from <checkpoint>/config.json. Synthetic tensors only.
"""

import argparse
import inspect
import json
import math
import os
import sys

import torch

WARMUP = 10
ITERS = 50

CASES = [
    ("1", [1]),
    ("27", [27]),
    ("300", [300]),
    ("1200", [1200]),
    ("4800", [4800]),
    ("3x333+7", [333, 333, 333, 7]),
    ("1000", [1000]),
    ("2000", [2000]),
    ("4000", [4000]),
    ("8000", [8000]),
    ("8192", [8192]),
    ("4x300", [300] * 4),
    ("16x300", [300] * 16),
    ("27x300", [300] * 27),
    ("4x1000", [1000] * 4),
    ("8x1000", [1000] * 8),
    ("4x2000", [2000] * 4),
    ("2x4000", [4000] * 2),
    ("16x512", [512] * 16),
]


def log(*a):
    print(*a, flush=True)


def load_shapes(ckpt):
    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    rp = tc.get("rope_parameters") or {}
    sl = rp.get("sliding_attention", {})
    fu = rp.get("full_attention", {})
    nq = tc["num_attention_heads"]
    nkv = tc["num_key_value_heads"]
    hd = tc["head_dim"]
    ghd = tc.get("global_head_dim") or hd
    k_eq_v = tc.get("attention_k_eq_v", False)
    gnkv = (tc.get("num_global_key_value_heads") or nkv) if k_eq_v else nkv
    shapes = {
        "sliding": dict(nq=nq, nkv=nkv, hd=hd, theta=sl.get("rope_theta", 10000.0),
                        partial=1.0, k_eq_v=False),
        "global": dict(nq=nq, nkv=gnkv, hd=ghd, theta=fu.get("rope_theta", 1e6),
                       partial=fu.get("partial_rotary_factor", 0.25), k_eq_v=k_eq_v),
    }
    eps = tc.get("rms_norm_eps", 1e-6)
    log("config:", json.dumps(dict(hidden=tc.get("hidden_size"), layers=tc.get("num_hidden_layers"),
                                   sliding_window=tc.get("sliding_window"), eps=eps, shapes=shapes)))
    return shapes, eps


def make_cos_sin(hd, theta, partial, max_pos, device):
    # Matches modeling_gemma4: sliding = full rotation; global = proportional
    # RoPE with zero-padded frequencies, cos_sin [max_pos, 2, hd//2] fp32.
    half = hd // 2
    rope_angles = int(partial * hd // 2)
    inv = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float64) / hd))
    inv = torch.cat([inv, torch.zeros(half - rope_angles, dtype=torch.float64)])
    pos = torch.arange(max_pos, dtype=torch.float64)
    s = pos[:, None] * inv[None, :]
    cs = torch.stack([torch.cos(s), torch.sin(s)], dim=1).to(torch.float32)
    return cs.contiguous().to(device)


class Layer:
    def __init__(self, sh, eps, device, seed):
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.nq, self.nkv, self.hd = sh["nq"], sh["nkv"], sh["hd"]
        self.k_eq_v = sh["k_eq_v"]
        self.q_size = self.nq * self.hd
        self.kv_size = self.nkv * self.hd
        self.eps = eps
        self.q_norm = RMSNorm(hidden_size=self.hd, eps=eps, dtype=torch.bfloat16, device=device)
        self.k_norm = RMSNorm(hidden_size=self.hd, eps=eps, dtype=torch.bfloat16, device=device)
        self.v_norm = RMSNorm(hidden_size=self.hd, eps=eps, dtype=torch.bfloat16, device=device,
                              has_weights=False)
        with torch.no_grad():
            self.q_norm.weight.copy_((1.0 + 0.2 * torch.randn(self.hd, generator=g)).to(torch.bfloat16))
            self.k_norm.weight.copy_((1.0 + 0.2 * torch.randn(self.hd, generator=g)).to(torch.bfloat16))
            if sh.get("real_qw") is not None:
                self.q_norm.weight.copy_(sh["real_qw"])
                self.k_norm.weight.copy_(sh["real_kw"])
        self.cos_sin = make_cos_sin(self.hd, sh["theta"], sh["partial"], 16384, device)

    def make_inputs(self, seqs, device, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = sum(seqs)
        w = self.q_size + 2 * self.kv_size
        qkv = (torch.randn(n, w, generator=g) * 2.0).to(torch.bfloat16)
        if self.k_eq_v:  # weight mapper duplicates k_proj into v_proj
            qkv[:, self.q_size + self.kv_size:] = qkv[:, self.q_size:self.q_size + self.kv_size]
        pos = torch.cat([torch.arange(s, dtype=torch.int32) for s in seqs])
        return qkv.to(device), pos.to(device)

    # --- main's unfused FA2 chain (modeling_gemma4.apply_rope + flashinfer forward_impl casts)
    def main_chain(self, qkv, pos):
        from tensorrt_llm._torch.custom_ops import flashinfer_apply_rope_with_cos_sin_cache_inplace
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        v = self.v_norm(v.reshape(-1, self.hd)).reshape(-1, self.kv_size)
        q = self.q_norm(q.reshape(-1, self.hd)).reshape(-1, self.q_size)
        k = self.k_norm(k.reshape(-1, self.hd)).reshape(-1, self.kv_size)
        flashinfer_apply_rope_with_cos_sin_cache_inplace(
            pos.view(-1), q, k, self.hd, self.cos_sin.view(self.cos_sin.shape[0], -1), True)
        k = k.view(-1, self.nkv, self.hd).to(torch.float8_e4m3fn)
        v = v.view(-1, self.nkv, self.hd).to(torch.float8_e4m3fn)
        return q, k, v

    def fused_chain(self, qkv, pos, cute=True):
        from tensorrt_llm._torch.modules.gemma4.fused_qkv import gemma4_fused_qkv_norm_rope_quant
        kw = {"cute_norm_order": cute} if has_cute_mode() else {}
        q, k, v = gemma4_fused_qkv_norm_rope_quant(
            qkv, pos, self.cos_sin, self.q_norm.weight, self.k_norm.weight, self.eps,
            self.nq, self.nkv, self.hd, out_fp8=False, packed_output=False, kv_out_fp8=True, **kw)
        # forward_impl view + (now no-op) cast
        k = k.view(-1, self.nkv, self.hd).to(torch.float8_e4m3fn)
        v = v.view(-1, self.nkv, self.hd).to(torch.float8_e4m3fn)
        return q, k, v

    def fp32_ref(self, qkv, pos):
        x = qkv.float()
        q, k, v = x.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        def nrm(t, w):
            t = t.reshape(-1, self.hd)
            t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)
            return t * w.float() if w is not None else t

        half = self.hd // 2
        cs = self.cos_sin[pos.long()]
        cos = cs[:, 0, :].repeat_interleave(1, 0)
        sin = cs[:, 1, :]

        def rope(t, nh):
            t = t.view(-1, nh, self.hd)
            a, b = t[..., :half], t[..., half:]
            c, s = cos[:, None, :], sin[:, None, :]
            return torch.cat([a * c - b * s, b * c + a * s], dim=-1)

        qn = rope(nrm(q, self.q_norm.weight), self.nq)
        kn = rope(nrm(k, self.k_norm.weight), self.nkv)
        vn = nrm(v, None).view(-1, self.nkv, self.hd)
        return qn.reshape(-1, self.q_size), kn, vn


def has_mixed_mode():
    from tensorrt_llm._torch.modules.gemma4.fused_qkv import gemma4_fused_qkv_norm_rope_quant
    return "kv_out_fp8" in inspect.signature(gemma4_fused_qkv_norm_rope_quant).parameters


def has_cute_mode():
    from tensorrt_llm._torch.modules.gemma4.fused_qkv import gemma4_fused_qkv_norm_rope_quant
    return "cute_norm_order" in inspect.signature(gemma4_fused_qkv_norm_rope_quant).parameters


def bench(ckpt, variant="auto", sliding_only=True):
    dev = torch.device("cuda")
    log("device:", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    shapes, eps = load_shapes(ckpt)
    fused = has_mixed_mode() and variant != "unfused"
    cute = variant != "old"
    log("chain:", (f"fused (branch, cute_norm_order={cute})" if fused else "unfused (main)"))
    out = {}
    for li, (lname, sh) in enumerate(shapes.items()):
        if sliding_only and lname != "sliding":
            continue  # global K=V layers run the unfused chain in both trees
        layer = Layer(sh, eps, dev, seed=100 + li)
        fn = (lambda a, b, _l=layer: _l.fused_chain(a, b, cute=cute)) if fused else layer.main_chain
        for cname, seqs in CASES:
            qkv, pos = layer.make_inputs(seqs, dev, seed=1)
            for _ in range(WARMUP):
                fn(qkv, pos)
            torch.cuda.synchronize()
            ts = []
            for _ in range(ITERS):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn(qkv, pos)
                e.record()
                e.synchronize()
                ts.append(s.elapsed_time(e))
            ts.sort()
            med = ts[len(ts) // 2]
            p10 = ts[int(0.1 * len(ts))]
            p90 = ts[int(0.9 * len(ts))]
            key = f"{lname}_{cname}"
            out[key] = {"kernel_ms": med, "p10_ms": p10, "p90_ms": p90,
                        "noise_floor": (p90 - p10) / med}
            n = sum(seqs)
            log(f"{key}: n={n} median={med*1e3:.1f}us p10={p10*1e3:.1f} p90={p90*1e3:.1f}")
            if fused and cute and has_cute_mode():
                # diagnostic: the previous (tl.sum order) fused kernel on the same inputs
                for _ in range(WARMUP):
                    layer.fused_chain(qkv, pos, cute=False)
                torch.cuda.synchronize()
                to = []
                for _ in range(ITERS):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    layer.fused_chain(qkv, pos, cute=False)
                    e.record()
                    e.synchronize()
                    to.append(s.elapsed_time(e))
                to.sort()
                out[key]["old_fused_kernel_ms"] = to[len(to) // 2]
                log(f"{key}: old fused median={to[len(to) // 2]*1e3:.1f}us")
    print(json.dumps({"batch": out, "output": {"class": "exact"}}))


def stats(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    return dict(max_abs=d.max().item(),
                rel_fro=(torch.linalg.norm(d) / torch.linalg.norm(b).clamp_min(1e-30)).item())


def load_real_norm_weights(ckpt, shapes):
    """Attach one sliding and one global layer's real q_norm/k_norm weights."""
    from safetensors import safe_open
    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    types = tc["layer_types"]
    want = {"sliding": types.index("sliding_attention"), "global": types.index("full_attention")}
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    for lname, li in want.items():
        for nm in ("q_norm", "k_norm"):
            key = next(k for k in wmap if f"layers.{li}.self_attn.{nm}.weight" in k)
            with safe_open(os.path.join(ckpt, wmap[key]), "pt") as sf:
                w = sf.get_tensor(key).to(torch.bfloat16)
            assert w.numel() == shapes[lname]["hd"], (key, w.shape)
            shapes[lname]["real_" + nm[0] + "w"] = w
            log(f"{lname} layer {li} {key}: min={w.float().min():.4f} max={w.float().max():.4f} "
                f"mean={w.float().mean():.4f}")


def check(ckpt, real=False):
    dev = torch.device("cuda")
    shapes, eps = load_shapes(ckpt)
    if real:
        load_real_norm_weights(ckpt, shapes)
    assert has_mixed_mode(), "check needs the branch's kv_out_fp8 mode"
    passed = True
    report = {}
    for li, (lname, sh) in enumerate(shapes.items()):
        layer = Layer(sh, eps, dev, seed=100 + li)
        for cname, seqs in CASES:
            qkv, pos = layer.make_inputs(seqs, dev, seed=1)
            qkv_copy = qkv.clone()
            # Untimed sequence matching timing: WARMUP + ITERS calls, no resets.
            first = None
            rep_ok = True
            for i in range(WARMUP + ITERS):
                o = layer.fused_chain(qkv, pos)
                torch.cuda.synchronize()
                if first is None:
                    first = [t.clone() for t in o]
                else:
                    for t0, t1 in zip(first, o):
                        if not torch.equal(t0.view(torch.uint8), t1.view(torch.uint8)):
                            rep_ok = False
            inp_ok = torch.equal(qkv, qkv_copy)
            q, k, v = first
            assert q.dtype == torch.bfloat16 and k.dtype == v.dtype == torch.float8_e4m3fn
            mq, mk, mv = layer.main_chain(qkv, pos)
            oq, ok_, ov = layer.fused_chain(qkv, pos, cute=False)
            rq, rk, rv = layer.fp32_ref(qkv, pos)
            finite = all(torch.isfinite(t.float()).all().item() for t in (q, k, v))
            shapes_ok = (q.shape == mq.shape and k.shape == mk.shape and v.shape == mv.shape)
            r = {}
            for nm, c, m, ref, old in (("q", q, mq, rq, oq), ("k", k, mk, rk, ok_), ("v", v, mv, rv, ov)):
                cb, mb = c.contiguous().view(torch.uint8), m.contiguous().view(torch.uint8)
                n_mism = int((cb.view(-1) != mb.view(-1)).sum().item())
                mism = n_mism / cb.numel()
                old_mism = int((old.contiguous().view(torch.uint8).view(-1) != mb.view(-1)).sum().item())
                cand = stats(c, ref)
                base = stats(m, ref)
                ulp = 2.0 ** (math.floor(math.log2(max(ref.abs().max().item(), 1e-30))) - 7)
                if c.dtype == torch.float8_e4m3fn:
                    ulp = 2.0 ** (math.floor(math.log2(max(ref.abs().max().item(), 1e-30))) - 3)
                del ulp
                ok = n_mism == 0  # bitwise gate (byte view, -0.0 counts)
                r[nm] = dict(mismatch_frac_vs_main=mism, mismatch_bytes_vs_main=n_mism,
                             old_kernel_mismatch_bytes_vs_main=old_mism, cand_vs_ref=cand, main_vs_ref=base,
                             cand_vs_main_max_abs=stats(c, m)["max_abs"],
                             max_abs_value=c.float().abs().max().item(), ok=ok)
                passed &= ok
            passed &= rep_ok and inp_ok and finite and shapes_ok
            key = f"{lname}_{cname}"
            report[key] = dict(repeat_bitwise=rep_ok, input_unchanged=inp_ok, finite=finite,
                               shapes_ok=shapes_ok, **r)
            log(key, json.dumps(report[key]))
    print(json.dumps({"passed": bool(passed),
                      "compared": "fused BF16-Q/FP8-KV prep (cute_norm_order) vs main unfused chain "
                                  "(flashinfer rmsnorm+rope+.to(fp8)), byte-wise; FP32 torch reference as "
                                  f"sanity; {WARMUP + ITERS} repeated calls bitwise vs first",
                      "mismatch_bytes": {k: {t: report[k][t]["mismatch_bytes_vs_main"] for t in "qkv"}
                                         for k in report},
                      "old_kernel_mismatch_bytes": {k: {t: report[k][t]["old_kernel_mismatch_bytes_vs_main"]
                                                        for t in "qkv"} for k in report},
                      "m8000": {k: report[k] for k in report if k.endswith("_8000")}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "check"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--real-norm-weights", action="store_true")
    ap.add_argument("--variant", choices=["auto", "old", "unfused"], default="auto")
    a = ap.parse_args()
    with torch.inference_mode():
        if a.mode == "bench":
            bench(a.ckpt, variant=a.variant)
        else:
            check(a.ckpt, real=a.real_norm_weights)
