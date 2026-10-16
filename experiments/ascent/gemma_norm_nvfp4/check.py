"""Untimed bitwise checks for branch 57 (idea 129).

A. Fused gemma add+RMSNorm+NVFP4 (Triton) vs main's FlashInfer
   gemma_fused_add_rmsnorm + trtllm.fp4_quantize: residual, FP4 and full SF
   buffer (padding rows included) bitwise, many M / input families / scales.
B. Repeated execution, same warmup/iteration counts as the bench (10 + 50):
   residual restored from a pristine copy and outputs poisoned (0xFF) before
   every eager call and every CUDA-graph replay; graph replays interleaved with
   eager calls; each compared to iteration 1 and to the reference.
C. PDL chain in one CUDA graph: producer nvfp4_gemm (N=5120,K=6144) writes x
   -> norm+quant -> gate_up nvfp4_gemm (34816,5120); fused vs main chain,
   gate_up output + residual bitwise over 60 replays.
D. SwiGLU+NVFP4 PDL on vs off (TRTLLM_FUSED_SILU_NVFP4_PDL kill switch):
   eager bytes and a gate_up -> SwiGLU -> down graph chain, 60 replays.
E. Decoder wiring: _post_attention_norm + _DenseMlpAdapter(GatedMLP NVFP4)
   with the fused path vs TRTLLM_GEMMA_NORM_NVFP4_FUSE=0 (main path),
   MLP output + residual bitwise, eager and graph.
"""
import json
import os
import subprocess
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401
from tensorrt_llm._torch.modules import fused_gemma_add_rmsnorm_nvfp4 as fg
from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4 as fs

H, I = 5120, 17408
dev = "cuda"
EPS = 1e-6
WARM, ITERS = 10, 50


def tree_commit():
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True).stdout
        return c, len([ln for ln in d.splitlines() if ln.strip()])
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})", -1


def make_inputs(m, seed, family):
    g = torch.Generator(device=dev).manual_seed(seed)
    if family == "plain":
        x = torch.randn((m, H), generator=g, device=dev) * 0.7
        r = torch.randn((m, H), generator=g, device=dev) * 3.0
    elif family == "rowscale":
        # Rows from 1e-4 to 1e3 magnitude, outliers, zeros, -0.0.
        sc = torch.exp(torch.empty((m, 1), device=dev).uniform_(
            -9.0, 7.0, generator=g))
        x = torch.randn((m, H), generator=g, device=dev) * sc
        r = torch.randn((m, H), generator=g, device=dev) * sc * 4.0
        u = torch.rand((m, H), generator=g, device=dev)
        x = torch.where(u < 0.002, x * 150.0, x)
        x = torch.where((u >= 0.002) & (u < 0.01), torch.zeros_like(x), x)
        r = torch.where((u >= 0.005) & (u < 0.01), torch.zeros_like(r), r)
        x = torch.where((u >= 0.01) & (u < 0.015), torch.full_like(x, -0.0),
                        x)
    else:  # "blocks": whole zero / -0 16-blocks, one all-zero row, tiny rows
        x = torch.randn((m, H), generator=g, device=dev)
        r = torch.randn((m, H), generator=g, device=dev) * 2.0
        nb = H // 16
        sel = torch.rand((m, nb), generator=g, device=dev)
        xb, rb = x.view(m, nb, 16), r.view(m, nb, 16)
        xb[sel < 0.03] = 0.0
        rb[sel < 0.03] = 0.0
        xb[(sel >= 0.03) & (sel < 0.05)] = -0.0
        rb[(sel >= 0.03) & (sel < 0.05)] = -0.0
        if m >= 2:
            x[1] = 0.0
            r[1] = 0.0
        if m >= 3:
            x[2] *= 1e-20
            r[2] *= 1e-20
        if m >= 4:
            x[3] *= 1e-36
            r[3] *= 1e-36
    return x.to(torch.bfloat16), r.to(torch.bfloat16)


def make_weight(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn((H, ), generator=g, device=dev) * 0.3
    u = torch.rand((H, ), generator=g, device=dev)
    w = torch.where(u < 0.01, torch.full_like(w, -1.0), w)  # w + 1 == 0
    w = torch.where((u >= 0.01) & (u < 0.02), w * 20.0, w)
    return w.to(torch.bfloat16)


def reference(x, r, w, gs):
    x2, r2 = x.clone(), r.clone()
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x2, r2, w, EPS)
    q, sf = torch.ops.trtllm.fp4_quantize(x2, gs, 16, False)
    return r2, q.view(torch.uint8), sf.view(torch.uint8)


def fused(x, r, w, gs):
    r2 = r.clone()
    q, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(x, r2, w, EPS, gs)
    return r2, q, sf


def cmp(a, b):
    ra, qa, sa = a
    rb, qb, sb = b
    return {
        "res_eq": torch.equal(ra, rb),
        "fp4_eq": torch.equal(qa, qb),
        "sf_eq": torch.equal(sa, sb),
        "n_fp4_diff": int((qa != qb).sum()),
        "n_sf_diff": int((sa != sb).sum()) if sa.shape == sb.shape else -1,
    }


def ok(c):
    return c["res_eq"] and c["fp4_eq"] and c["sf_eq"]


def part_a():
    out, passed, nblocks = {}, True, 0
    ms = [1, 2, 3, 4, 8, 16, 64, 127, 128, 129, 2048, 8192]
    scales = [100.0, 448.0 * 6.0 / 20.0, 1.0, 0.05]
    for m in ms:
        for fi, fam in enumerate(["plain", "rowscale", "blocks"]):
            for si, sv in enumerate(scales):
                if m >= 2048 and si >= 2:
                    continue
                gs = torch.tensor([sv], dtype=torch.float32, device=dev)
                x, r = make_inputs(m, 7919 * m + 31 * fi + si, fam)
                w = make_weight(17 * m + fi)
                x0 = x.clone()
                ref = reference(x, r, w, gs)
                fu = fused(x, r, w, gs)
                c = cmp(fu, ref)
                c["x_unchanged"] = torch.equal(x, x0)
                nblocks += m * (H // 16)
                good = ok(c) and c["x_unchanged"]
                passed = passed and good
                key = f"M{m}_{fam}_s{si}"
                if not good or m in (1, 4, 2048, 8192):
                    out[key] = c
                    print("A", key, c, flush=True)
    out["blocks_compared"] = nblocks
    return passed, out


def part_b():
    out, passed = {}, True
    gs = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device=dev)
    for m in [1, 2, 4, 16, 2048, 8192]:
        x, r_pristine = make_inputs(m, 4242 + m, "rowscale")
        w = make_weight(99 + m)
        x0 = x.clone()
        ref = reference(x, r_pristine, w, gs)
        r = r_pristine.clone()
        q = torch.empty_like(ref[1])
        sf = torch.empty_like(ref[2])
        first, bad, maxdiff = None, 0, 0
        for it in range(WARM + ITERS):
            r.copy_(r_pristine)
            q.fill_(0xFF)
            sf.fill_(0xFF)
            fg._launch(x, r, w, EPS, gs, q, sf)
            torch.cuda.synchronize()
            cur = (r.clone(), q.clone(), sf.clone())
            if first is None:
                first = cur
            c1, c2 = cmp(cur, first), cmp(cur, ref)
            maxdiff = max(maxdiff, c2["n_fp4_diff"], c2["n_sf_diff"])
            if not (ok(c1) and ok(c2)):
                bad += 1
        # Graph replays interleaved with eager (allocating) calls.
        gr = r_pristine.clone()
        gq = torch.empty_like(ref[1])
        gsf = torch.empty_like(ref[2])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fg._launch(x, gr, w, EPS, gs, gq, gsf)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fg._launch(x, gr, w, EPS, gs, gq, gsf)
        gbad, ebad = 0, 0
        for it in range(WARM + ITERS):
            gr.copy_(r_pristine)
            gq.fill_(0xFF)
            gsf.fill_(0xFF)
            g.replay()
            torch.cuda.synchronize()
            cur = (gr.clone(), gq.clone(), gsf.clone())
            if not (ok(cmp(cur, first)) and ok(cmp(cur, ref))):
                gbad += 1
            if not ok(cmp(fused(x, r_pristine, w, gs), ref)):
                ebad += 1
        del g
        inputs_ok = torch.equal(x, x0)
        out[f"M{m}"] = {"eager_bad": bad, "graph_bad": gbad,
                        "interleaved_eager_bad": ebad,
                        "max_diff_vs_ref": maxdiff,
                        "x_unchanged": inputs_ok}
        passed = passed and bad == 0 and gbad == 0 and ebad == 0 and inputs_ok
        print("B", m, out[f"M{m}"], flush=True)
    return passed, out


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = fn()
    return g, o


def rand_w(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev,
                      generator=g)
    wsf = torch.randint(0x28, 0x38, (n * k // 16, ), dtype=torch.uint8,
                        device=dev, generator=g)
    return w, wsf


def part_c():
    out, passed = {}, True
    from tensorrt_llm._torch.autotuner import autotune
    wp, wpsf = rand_w(H, 6144, 1)
    wg, wgsf = rand_w(2 * I, H, 2)
    alpha = torch.tensor([0.004], dtype=torch.float32, device=dev)
    gs_in = torch.tensor([20.0], dtype=torch.float32, device=dev)
    gs = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device=dev)
    w = make_weight(5)
    for m in [1, 2, 4, 16]:
        a = (torch.randn(m, 6144, device=dev) * 0.5).to(torch.bfloat16)
        aq, asf = torch.ops.trtllm.fp4_quantize(a, gs_in, 16, False)
        r_pristine = (torch.randn(m, H, device=dev) * 3).to(torch.bfloat16)
        r = r_pristine.clone()
        with autotune():
            xx = torch.ops.trtllm.nvfp4_gemm(aq, wp, asf, wpsf, alpha,
                                             torch.bfloat16)
            q0, s0 = torch.ops.trtllm.fp4_quantize(xx, gs, 16, False)
            torch.ops.trtllm.nvfp4_gemm(q0, wg, s0, wgsf, alpha,
                                        torch.bfloat16)

        def chain(use_fused):
            def fn():
                x = torch.ops.trtllm.nvfp4_gemm(aq, wp, asf, wpsf, alpha,
                                                torch.bfloat16)
                if use_fused:
                    q, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(
                        x, r, w, EPS, gs)
                else:
                    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(
                        x, r, w, EPS)
                    q, sf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)
                return torch.ops.trtllm.nvfp4_gemm(q, wg, sf, wgsf, alpha,
                                                   torch.bfloat16)
            return fn

        res = {}
        for name in ("main", "fused"):
            r.copy_(r_pristine)
            gr, o = capture(chain(name == "fused"))
            outs, bad = None, 0
            for it in range(WARM + ITERS):
                r.copy_(r_pristine)
                o.fill_(float("nan"))
                gr.replay()
                torch.cuda.synchronize()
                cur = (o.clone(), r.clone())
                if outs is None:
                    outs = cur
                elif not (torch.equal(cur[0], outs[0])
                          and torch.equal(cur[1], outs[1])):
                    bad += 1
            res[name] = (outs, bad)
            del gr
        eq = (torch.equal(res["main"][0][0], res["fused"][0][0])
              and torch.equal(res["main"][0][1], res["fused"][0][1]))
        nan = bool(torch.isnan(res["main"][0][0].float()).any())
        out[f"M{m}"] = {"gate_up_out_and_residual_eq": eq,
                        "main_replay_bad": res["main"][1],
                        "fused_replay_bad": res["fused"][1], "nan": nan}
        passed = (passed and eq and res["main"][1] == 0
                  and res["fused"][1] == 0 and not nan)
        print("C", m, out[f"M{m}"], flush=True)
    return passed, out


def part_d():
    out, passed = {}, True
    from tensorrt_llm._torch.autotuner import autotune
    wg, wgsf = rand_w(2 * I, H, 3)
    wd, wdsf = rand_w(H, I, 4)
    alpha = torch.tensor([0.004], dtype=torch.float32, device=dev)
    gs_x = torch.tensor([30.0], dtype=torch.float32, device=dev)
    gs_h = torch.tensor([30.0], dtype=torch.float32, device=dev)
    orig = fs._PDL_ENABLED
    for m in [1, 2, 4, 8]:
        h1 = (torch.randn(m, 2 * I, device=dev) * 2).to(torch.bfloat16)
        x = (torch.randn(m, H, device=dev)).to(torch.bfloat16)
        xq, xsf = torch.ops.trtllm.fp4_quantize(x, gs_x, 16, False)
        eager = {}
        for pdl in (False, True):
            fs._PDL_ENABLED = pdl
            eager[pdl] = fs._fused_silu_mul_nvfp4_impl(h1, gs_h)
        e_eq = (torch.equal(eager[False][0], eager[True][0])
                and torch.equal(eager[False][1], eager[True][1]))
        with autotune():
            hh = torch.ops.trtllm.nvfp4_gemm(xq, wg, xsf, wgsf, alpha,
                                             torch.bfloat16)
            q, sf = fs._fused_silu_mul_nvfp4_impl(hh, gs_h)
            torch.ops.trtllm.nvfp4_gemm(q, wd, sf, wdsf, alpha,
                                        torch.bfloat16)

        def fn():
            hh = torch.ops.trtllm.nvfp4_gemm(xq, wg, xsf, wgsf, alpha,
                                             torch.bfloat16)
            q, sf = torch.ops.trtllm.fused_silu_mul_nvfp4(hh, gs_h)
            return torch.ops.trtllm.nvfp4_gemm(q, wd, sf, wdsf, alpha,
                                               torch.bfloat16)

        res = {}
        for pdl in (False, True):
            fs._PDL_ENABLED = pdl
            g, o = capture(fn)
            first, bad = None, 0
            for it in range(WARM + ITERS):
                o.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                if first is None:
                    first = o.clone()
                elif not torch.equal(o, first):
                    bad += 1
            res[pdl] = (first, bad)
            del g
        fs._PDL_ENABLED = orig
        g_eq = torch.equal(res[False][0], res[True][0])
        out[f"M{m}"] = {"eager_bytes_eq": e_eq, "graph_down_out_eq": g_eq,
                        "replay_bad_off": res[False][1],
                        "replay_bad_on": res[True][1],
                        "pdl_launch_at_m": int(
                            (m + 1) * (I // 512) < fs._num_sms(h1.device))}
        passed = (passed and e_eq and g_eq and res[False][1] == 0
                  and res[True][1] == 0)
        print("D", m, out[f"M{m}"], flush=True)
    return passed, out


def part_e():
    out, passed = {}, True
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_qwen3_next as mq
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    torch.manual_seed(0)
    gm = GatedMLP(hidden_size=H, intermediate_size=I, bias=False,
                  dtype=torch.bfloat16, config=ModelConfig(quant_config=qc),
                  layer_idx=0).to(dev)
    with torch.no_grad():
        for lin in (gm.gate_up_proj, gm.down_proj):
            lin.weight.data.copy_(torch.randint(0, 256, lin.weight.shape,
                                                dtype=torch.uint8))
            lin.weight_scale.data.fill_(0x30)
            lin.input_scale.data.fill_(30.0 if lin is gm.down_proj else
                                       448.0 * 6.0 / 20.0)
            lin.alpha.data.fill_(0.002)
            lin.weight_scale_2.data.fill_(1.0)

    class Layer(torch.nn.Module):
        pass

    layer = Layer()
    layer.post_attention_layernorm = RMSNorm(hidden_size=H, eps=EPS,
                                             dtype=torch.bfloat16,
                                             use_gemma=True).to(dev)
    with torch.no_grad():
        layer.post_attention_layernorm.weight.copy_(make_weight(11))
    layer.mlp = mq._DenseMlpAdapter(gm, Mapping())

    def run(x, r, fuse):
        os.environ["TRTLLM_GEMMA_NORM_NVFP4_FUSE"] = "1" if fuse else "0"
        layer.__dict__.pop("_post_attn_nvfp4_ok", None)
        hs, res = mq._post_attention_norm(layer, x, r, None)
        kind = type(hs).__name__
        return layer.mlp(hs), res, kind

    default_min = mq._POST_ATTN_NVFP4_MIN_ROWS
    print("E default min rows", default_min, flush=True)
    cases = [(m, True) for m in [1, 2, 4, 16, 2048]]
    cases += [(m, False) for m in [1, 4, 16, 2048, 4096]]
    for m, forced in cases:
        # forced: threshold 1, so the fused wiring runs at every M;
        # default: decode sizes must stay on main's path.
        mq._POST_ATTN_NVFP4_MIN_ROWS = 1 if forced else default_min
        expect_fused = forced or m >= default_min
        x, r_pristine = make_inputs(m, 555 + m, "plain")
        r = r_pristine.clone()
        ref, rref, kref = run(x.clone(), r, False)
        r = r_pristine.clone()
        o, ro, kfu = run(x.clone(), r, True)
        eq = torch.equal(o, ref) and torch.equal(ro, rref)
        gbad = None
        if m <= 16:
            gx, gr = x.clone(), r_pristine.clone()
            os.environ["TRTLLM_GEMMA_NORM_NVFP4_FUSE"] = "1"
            layer.__dict__.pop("_post_attn_nvfp4_ok", None)

            def fn():
                hs, _ = mq._post_attention_norm(layer, gx, gr, None)
                return layer.mlp(hs)

            g, go = capture(fn)
            gbad = 0
            for _ in range(WARM + ITERS):
                gr.copy_(r_pristine)
                go.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                if not (torch.equal(go, ref) and torch.equal(gr, rref)):
                    gbad += 1
            del g
        key = f"M{m}_{'forced' if forced else 'default'}"
        out[key] = {"mlp_out_and_residual_eq": eq, "kind_fused": kfu,
                    "expect_fused": expect_fused,
                    "kind_killswitch": kref, "graph_bad": gbad,
                    "nan": bool(torch.isnan(ref.float()).any())}
        passed = (passed and eq
                  and (kfu == "Fp4QuantizedTensor") == expect_fused
                  and kref == "Tensor" and gbad in (None, 0))
        print("E", key, out[key], flush=True)
    mq._POST_ATTN_NVFP4_MIN_ROWS = default_min
    os.environ["TRTLLM_GEMMA_NORM_NVFP4_FUSE"] = "1"
    return passed, out


def main():
    c, dirty = tree_commit()
    print(f"TREE_COMMIT={c} dirty={dirty}", flush=True)
    from flashinfer.norm import _use_cuda_norm
    print("flashinfer _use_cuda_norm:", _use_cuda_norm(torch.device(dev)),
          flush=True)
    results, passed = {"tree_commit": c}, True
    parts = sys.argv[1:] or ["a", "b", "c", "d", "e"]
    for p in parts:
        ok_p, res = {"a": part_a, "b": part_b, "c": part_c, "d": part_d,
                     "e": part_e}[p]()
        results[p] = {"passed": ok_p, **res}
        passed = passed and ok_p
        print(f"part {p} passed={ok_p}", flush=True)
    print(json.dumps({
        "passed": passed,
        "compared": "fused gemma add+RMSNorm+NVFP4 residual/FP4/SF bytes vs "
                    "main's flashinfer_gemma_fused_add_rmsnorm + "
                    "trtllm.fp4_quantize (bitwise), repeated execution, "
                    "PDL graph chains, SwiGLU PDL on/off, decoder wiring vs "
                    "kill switch",
        "results": results
    }))


if __name__ == "__main__":
    main()
