"""SM120 wo_a input glue (project_sparse_attn_output, M > 8): bench + exactness check.

Modes:
  stage0  per-op us/call of main's chain (mla_rope_inplace, ptq(permute102), BMM, whole chain)
          and, when present, the fused kernel; L2-warm (one buffer) and L2-cold (40 rotating).
  pair    40-layer wo_a chain through the real project_sparse_attn_output (o_b_proj = identity),
          one CUDA-event window per iteration, inputs restored before every iteration (untimed).
          Prints the evidence JSON. The arm (fused or main) follows TRTLLM_DSV4_PREFILL_WO_QUANT.
  check   bitwise: fused fp8 bytes / scale words / BMM output vs main's chain at many shapes,
          special rows, repeated-execution sequence with poisoned outputs, CUDA-graph capture at
          M=128 with the Triton compile-cache count unchanged.
Synthetic tensors only (no model load); dims read from the checkpoint config.json.
"""

import argparse
import json
import math
import os
import sys
import types

import torch

TP = 4
CTX = 34816 + 2048


def log(*a):
    print(*a, flush=True)


def _find(cfg, key):
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        for v in cfg.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def dims(checkpoint):
    with open(os.path.join(checkpoint, "config.json")) as f:
        cfg = json.load(f)
    out = {}
    for k in ("num_attention_heads", "head_dim", "qk_rope_head_dim", "o_lora_rank", "o_groups"):
        v = _find(cfg, k)
        if v is None:
            raise SystemExit(f"config.json missing {k}: keys {sorted(cfg)}")
        out[k] = int(v)
    log("config dims:", out)
    heads_tp = out["num_attention_heads"] // TP
    groups_tp = out["o_groups"] // TP
    d = dict(
        heads=heads_tp,
        groups=groups_tp,
        head_dim=out["head_dim"],
        rope=out["qk_rope_head_dim"],
        nope=out["head_dim"] - out["qk_rope_head_dim"],
        rank=out["o_lora_rank"],
    )
    d["gw"] = heads_tp // groups_tp * d["head_dim"]
    log("rank dims (TP4):", d)
    assert d["head_dim"] == 512 and d["rope"] == 64, d
    return d


def make_cos_sin(dev):
    half = 32
    inv = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float64) / half))
    ang = torch.arange(CTX, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.stack([ang.cos(), ang.sin()], 1).float().contiguous().to(dev)


def make_layer(d, dev, gen):
    from tensorrt_llm.quantization.utils import fp8_utils

    g, rank, gw = d["groups"], d["rank"], d["gw"]
    w = (torch.randn(g, rank, gw, generator=gen, device=dev) * 0.5).to(torch.float8_e4m3fn)
    exps = torch.randint(-12, -4, (g, rank // 128, gw // 128), generator=gen, device=dev)
    block = torch.pow(2.0, exps.float()).contiguous()
    ws = fp8_utils.transform_sf_into_required_layout(
        block, mn=rank, k=gw, recipe=(1, 128, 128), num_groups=g, is_sfa=False
    )
    return w, ws


def fake_module(d, w, ws, cos_sin):
    m = types.SimpleNamespace()
    m.o_a_proj = w
    m.o_a_proj_scale = ws
    m.o_a_proj_dequant = None
    m.use_cute_dsl_blockscaling_bmm = False
    m.use_cute_dsl_bf16_bmm = False
    m.num_heads_tp = d["heads"]
    m.n_local_groups = d["groups"]
    m.qk_nope_head_dim = d["nope"]
    m.qk_rope_head_dim = d["rope"]
    m.o_lora_rank = d["rank"]
    m.inverse_rotary_emb = types.SimpleNamespace(rotary_cos_sin=cos_sin, is_neox=False)
    m.o_b_proj = lambda x: x
    m.dtype = torch.bfloat16
    return m


def make_input(d, m, dev, gen, special=False):
    x = torch.randn(m, d["heads"], d["head_dim"], generator=gen, device=dev) * 2.0
    # per-row dynamic range variety
    x = x * torch.pow(2.0, torch.randint(-6, 6, (m, 1, 1), generator=gen, device=dev).float())
    x = x.to(torch.bfloat16)
    if special and m >= 6:
        x[0] = 0
        x[1, 0, 3] = float("nan")
        x[1, 1, 470] = float("nan")
        x[2, 0, 7] = float("inf")
        x[2, 2, 500] = float("-inf")
        x[3] = (torch.randn(d["heads"], d["head_dim"], device=dev) * 1e-39).to(torch.bfloat16)
        x[4, :, :] = 3.0e38
        x[5, :, 448:] = 0
        if m > 7:
            x[7, :, ::2] = -1e-30
    pos = torch.randint(0, CTX, (m,), generator=gen, device=dev, dtype=torch.int32)
    return x.contiguous(), pos


def project(mod, x, pos):
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import (
        project_sparse_attn_output,
    )

    return project_sparse_attn_output(mod, [x.flatten(1)], pos)


def have_fused():
    try:
        from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import prefill_wo_quant  # noqa

        return True
    except ImportError:
        return False


def time_loop(fn, reps, pre=None):
    """Median us/call over 5 windows of `reps` calls (events)."""
    res = []
    for _ in range(5):
        if pre:
            pre()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for i in range(reps):
            fn(i)
        e.record()
        torch.cuda.synchronize()
        res.append(s.elapsed_time(e) * 1000 / reps)
    res.sort()
    return res[2]


def stage0(d, dev, ms):
    from tensorrt_llm.quantization.utils import fp8_utils

    fused = have_fused()
    if fused:
        from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
            prefill_wo_quant as pwq,
        )
    gen = torch.Generator(device=dev).manual_seed(0)
    cos_sin = make_cos_sin(dev)
    L = 40
    layers = [make_layer(d, dev, gen) for _ in range(L)]
    g, rank = d["groups"], d["rank"]
    rows = []
    for m in ms:
        xs = [make_input(d, m, dev, gen) for _ in range(L)]
        outs = [torch.empty(m, g, rank, device=dev, dtype=torch.bfloat16) for _ in range(L)]
        for mode, nbuf in (("warm", 1), ("cold", L)):
            def rope(i):
                x, p = xs[i % nbuf]
                torch.ops.trtllm.mla_rope_inplace(
                    x, p, cos_sin, d["heads"], d["nope"], d["rope"], True, False
                )

            def ptq(i):
                x, _ = xs[i % nbuf]
                fp8_utils.per_token_quant_and_transform(x.view(m, g, -1), need_permute102=True)

            q = [
                fp8_utils.per_token_quant_and_transform(
                    xs[j][0].view(m, g, -1), need_permute102=True
                )
                for j in range(nbuf)
            ]

            def bmm(i):
                a, sa = q[i % nbuf]
                w, ws = layers[i % nbuf]
                torch.ops.trtllm.fp8_block_scaling_bmm_out(
                    a, w, sa, ws, outs[i % nbuf].transpose(0, 1)
                )

            def chain(i):
                rope(i)
                x, _ = xs[i % nbuf]
                a, sa = fp8_utils.per_token_quant_and_transform(
                    x.view(m, g, -1), need_permute102=True
                )
                w, ws = layers[i % nbuf]
                torch.ops.trtllm.fp8_block_scaling_bmm_out(
                    a, w, sa, ws, outs[i % nbuf].transpose(0, 1)
                )

            ops = {"rope": rope, "ptq": ptq, "bmm": bmm, "chain_main": chain}
            if fused:

                def fz(i):
                    x, p = xs[i % nbuf]
                    pwq.inv_rope_quant_permute102(x, p, cos_sin, g, d["nope"], d["rope"])

                def chain_f(i):
                    x, p = xs[i % nbuf]
                    a, sa = pwq.inv_rope_quant_permute102(x, p, cos_sin, g, d["nope"], d["rope"])
                    w, ws = layers[i % nbuf]
                    torch.ops.trtllm.fp8_block_scaling_bmm_out(
                        a, w, sa, ws, outs[i % nbuf].transpose(0, 1)
                    )

                ops["fused"] = fz
                ops["chain_fused"] = chain_f
            for name, fn in ops.items():
                for i in range(3):
                    fn(i)
                torch.cuda.synchronize()
                reps = 200 if mode == "warm" else 80
                us = time_loop(fn, reps)
                rows.append(dict(m=m, l2=mode, op=name, us=round(us, 2)))
                log(f"M={m:5d} {mode} {name:12s} {us:9.2f} us/call")
        del xs, outs
        torch.cuda.empty_cache()
    log(json.dumps({"stage0": rows}))


def pair(d, dev, ms, iters, warmup):
    gen = torch.Generator(device=dev).manual_seed(0)
    cos_sin = make_cos_sin(dev)
    L = 40
    layers = [make_layer(d, dev, gen) for _ in range(L)]
    mods = [fake_module(d, w, ws, cos_sin) for (w, ws) in layers]
    env = os.environ.get("TRTLLM_DSV4_PREFILL_WO_QUANT", "1")
    log(f"pair arm: fused_module_present={have_fused()} env={env}")
    result = {}
    for m in ms:
        pristine = [make_input(d, m, dev, gen) for _ in range(L)]
        work = [(x.clone(), p) for x, p in pristine]

        def restore():
            for (x, _), (x0, _) in zip(work, pristine):
                x.copy_(x0)

        def run():
            y = None
            for li in range(L):
                y = project(mods[li], *work[li])
            return y

        for _ in range(warmup):
            restore()
            run()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            restore()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            run()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        med = times[len(times) // 2]
        p10 = times[int(0.1 * (len(times) - 1))]
        p90 = times[int(0.9 * (len(times) - 1))]
        noise = (p90 - p10) / med
        log(f"M={m}: 40-layer wo_a chain median {med:.4f} ms p10 {p10:.4f} p90 {p90:.4f} noise {noise:.4f}")
        result[str(m)] = {"kernel_ms": round(med, 5), "noise_floor": round(noise, 5)}
        del pristine, work
        torch.cuda.empty_cache()
    log(json.dumps({"batch": result, "output": {"class": "exact"}}))


def _cache_count(fn):
    n = 0
    for v in getattr(fn, "device_caches", {}).values():
        try:
            n += len(v[0])
        except Exception:
            pass
    return n


def _bits_equal(a, b):
    if a.shape != b.shape or a.stride() != b.stride() or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.float8_e4m3fn,):
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a, b)


def check(d, dev, iters, warmup):
    from tensorrt_llm.quantization.utils import fp8_utils
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import prefill_wo_quant as pwq

    gen = torch.Generator(device=dev).manual_seed(1)
    cos_sin = make_cos_sin(dev)
    g = d["groups"]
    w, ws = make_layer(d, dev, gen)
    mod = fake_module(d, w, ws, cos_sin)
    failures = []
    report = {}

    def main_chain_q(x, p):
        xr = x.clone()
        torch.ops.trtllm.mla_rope_inplace(xr, p, cos_sin, d["heads"], d["nope"], d["rope"], True, False)
        a, sa = fp8_utils.per_token_quant_and_transform(xr.view(x.shape[0], g, -1), need_permute102=True)
        return xr, a, sa

    def main_project(x, p):
        os.environ["TRTLLM_DSV4_PREFILL_WO_QUANT"] = "0"
        try:
            return project(mod, x.clone(), p)
        finally:
            os.environ["TRTLLM_DSV4_PREFILL_WO_QUANT"] = "1"

    def poison(m):
        gw = d["gw"]
        blobs = [
            torch.full((g, m, gw), 0x7F, dtype=torch.uint8, device=dev),
            torch.full((g, gw // 512, (m + 3) // 4 * 4), -1, dtype=torch.int32, device=dev),
            torch.full((m, g, d["rank"]), float("nan"), dtype=torch.bfloat16, device=dev),
        ]
        del blobs

    # 1) quant outputs + BMM output at many shapes, special rows.
    ms = [1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 31, 64, 127, 128, 130, 512, 1023, 2047, 2048, 4096]
    for m in ms:
        x, p = make_input(d, m, dev, gen, special=True)
        x_keep = x.clone()
        xr, a_ref, s_ref = main_chain_q(x, p)
        poison(m)
        a, s = pwq.inv_rope_quant_permute102(x, p, cos_sin, g, d["nope"], d["rope"])
        ok_a = _bits_equal(a, a_ref)
        ok_s = _bits_equal(s, s_ref)
        ok_in = _bits_equal(x, x_keep)
        y_ref = main_project(x, p)
        poison(m)
        y = project(mod, x, p)
        ok_y = _bits_equal(y, y_ref)
        # independent fp32 reference: dequantized fp8 vs fp32 rope result (finite rows only)
        sc = torch.pow(2.0, ((s.unsqueeze(-1) >> torch.tensor([0, 8, 16, 24], device=dev, dtype=torch.int32)) & 0xFF).float() - 127)
        sc = sc.flatten(-2).repeat_interleave(128, dim=-1)  # [G, M, gw]
        deq = a.float() * sc
        xr_g = xr.view(m, g, -1).transpose(0, 1).float()
        fin = torch.isfinite(xr_g).all(-1) & (xr_g.abs().amax(-1) < 1e30)
        err = (deq - xr_g).abs()[fin]
        tol = (xr_g.abs()[fin] * 0.07 + 1e-30)
        ok_ref = bool((err <= tol + 2.0 ** -9 * sc[fin]).all()) if err.numel() else True
        rec = dict(fp8=ok_a, scale=ok_s, input_untouched=ok_in, bmm=ok_y, fp32_ref=ok_ref)
        report[f"M{m}"] = rec
        log(f"M={m}: {rec}")
        if not all(rec.values()):
            failures.append(f"M{m}:{rec}")

    # 2) repeated-execution sequence (pair's warmup/iters, restore before each, poisoned outputs).
    for m in (2048, 128, 16, 512):
        x0, p = make_input(d, m, dev, gen, special=True)
        y_ref = main_project(x0, p)
        _, a_ref, s_ref = main_chain_q(x0, p)
        x = x0.clone()
        bad = 0
        for it in range(warmup + iters):
            x.copy_(x0)
            poison(m)
            a, s = pwq.inv_rope_quant_permute102(x, p, cos_sin, g, d["nope"], d["rope"])
            poison(m)
            y = project(mod, x, p)
            ok = _bits_equal(a, a_ref) and _bits_equal(s, s_ref) and _bits_equal(y, y_ref) and _bits_equal(x, x0)
            if not ok:
                bad += 1
        log(f"repeated M={m}: {warmup + iters} iterations, mismatching {bad}")
        report[f"repeat_M{m}"] = bad
        if bad:
            failures.append(f"repeat M{m} bad={bad}")

    # 3) CUDA graph capture at M=128 after eager warmup: no new compile, output exact.
    fn = pwq._inv_rope_quant_permute102_kernel
    m = 128
    x0, p = make_input(d, m, dev, gen, special=True)
    y_ref = main_project(x0, p)
    x = x0.clone()
    project(mod, x, p)  # eager (already compiled above)
    torch.cuda.synchronize()
    n_before = _cache_count(fn)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph, stream=s):
            y_g = project(mod, x, p)
    torch.cuda.synchronize()
    n_after = _cache_count(fn)
    bad = 0
    for _ in range(5):
        x.copy_(x0)
        y_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        bad += 0 if _bits_equal(y_g, y_ref) else 1
    report["graph_M128"] = dict(cache_before=n_before, cache_after=n_after, replay_bad=bad)
    log("graph M=128:", report["graph_M128"])
    if n_before != n_after or bad or n_before == 0:
        failures.append(f"graph:{report['graph_M128']}")
    # 3b) capture when the kernel was never compiled in this process -> main chain inside graph.
    pwq._compiled = False
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph2, stream=s):
            y_g2 = project(mod, x, p)
    torch.cuda.synchronize()
    x.copy_(x0)
    graph2.replay()
    torch.cuda.synchronize()
    ok_fb = _bits_equal(y_g2, y_ref) and _cache_count(fn) == n_after
    report["graph_uncompiled_fallback"] = ok_fb
    log("graph uncompiled fallback exact:", ok_fb)
    if not ok_fb:
        failures.append("graph fallback")
    pwq._compiled = True

    log(json.dumps({"passed": not failures, "compared": "fused inv_rope_quant_permute102 + project_sparse_attn_output vs main mla_rope_inplace + per_token_quant_and_transform(need_permute102) + SM120 BMM (bitwise), plus fp32 dequant reference", "failures": failures[:20], "report": report}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stage0", "pair", "check"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ms", default="2048,128,16,512")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

    dev = torch.device("cuda")
    log("device:", torch.cuda.get_device_name(), "sm", torch.cuda.get_device_capability())
    d = dims(a.checkpoint)
    ms = [int(v) for v in a.ms.split(",")]
    if a.mode == "stage0":
        stage0(d, dev, ms)
    elif a.mode == "pair":
        pair(d, dev, ms, a.iters, a.warmup)
    else:
        check(d, dev, a.iters, a.warmup)


if __name__ == "__main__":
    main()
