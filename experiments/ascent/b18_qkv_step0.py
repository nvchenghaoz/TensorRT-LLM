"""Branch 18 step 0: SM120 NVFP4 QKV (N=9216, K=3072) - fused CuTe op vs CUTLASS GEMM(s) + norm/RoPE pass.

Synthetic activations / q,k,v weights quantized with flashinfer nvfp4_quantize (layout_128x4,
cute-dsl) like flux3_primsts; q, k, v each carry their own weight global scale (as in the
checkpoint), so qkv_scale has three different alphas. Real q/k norm weights from the checkpoint.
"""

import glob
import json
import os
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope import (
    qknorm_rope,
)
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import (
    _sm120_fp4_gemm_runner,
    nvfp4_quantize,
)

ITERS = 50
Ms = [int(x) for x in os.environ.get("B18_MS", "4096,4176,4256").split(",")]
H, D, K = 24, 128, 3072
W = H * D


def log(*a):
    print(*a, flush=True)


def quant(x):
    amax = x.abs().amax().float()
    inv = (448.0 * 6.0 / amax).reshape(1).cuda()
    data, sf = nvfp4_quantize(x, inv)
    return data, sf, float(1.0 / inv.item())


def norm_weights(ckpt):
    from safetensors import safe_open

    for f in sorted(glob.glob(os.path.join(ckpt, "variants", "**", "*.safetensors"), recursive=True)):
        with safe_open(f, "pt") as h:
            keys = list(h.keys())
            qk = "single_blocks.0.norm.query_norm.scale"
            kk = "single_blocks.0.norm.key_norm.scale"
            if qk in keys:
                return h.get_tensor(qk).cuda(), h.get_tensor(kk).cuda(), f
    raise RuntimeError("no norm weights")


def time_fn(fn, iters=ITERS):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    med = t[len(t) // 2]
    return round(med, 1), round((t[(len(t) * 9) // 10] - t[len(t) // 10]) / med, 4)


def main():
    ckpt = sys.argv[1]
    torch.manual_seed(0)
    q_norm, k_norm, src = norm_weights(ckpt)
    log("norm weights from", src)
    runner = _sm120_fp4_gemm_runner()
    side = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = {}
    for M in Ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ws = [torch.randn(W, K, device="cuda", dtype=torch.bfloat16) * s for s in (0.02, 0.035, 0.012)]
        a, a_sf, a_gs = quant(x)
        qw = [quant(w) for w in ws]
        weight = torch.cat([q[0] for q in qw], 0).contiguous()
        w_sf = torch.cat([q[1] for q in qw], 0).contiguous()
        seg_sf = w_sf.numel() // 3
        qkv_scale = torch.tensor([a_gs * q[2] for q in qw], device="cuda", dtype=torch.float32)
        theta = torch.rand(M, 64, device="cuda") * 6.28
        cos_sin = torch.cat([theta.cos(), theta.sin()], -1).float().contiguous()
        positions = torch.arange(M, device="cuda", dtype=torch.int64)

        def cute_fused():
            return flux3_sm120_nvfp4.run(
                "qkv", a, weight, a_sf, w_sf, a_gs, qw[0][2], q_norm=q_norm, k_norm=k_norm,
                cos_sin=cos_sin, positions=positions, qkv_scale=qkv_scale,
            )

        ref = cute_fused().clone()
        row = {"cute_fused": time_fn(cute_fused)}

        def one(i, tactic=0):
            return runner.run_gemm(
                a, weight[i * W:(i + 1) * W], a_sf, w_sf[i * seg_sf:(i + 1) * seg_sf],
                qkv_scale[i:i + 1], 0, tactic, None, None,
            )

        def gemm3():
            return [one(i) for i in range(3)]

        def gemm3_ms():
            cur = torch.cuda.current_stream()
            outs = [None, None, None]
            for i, s in enumerate(side):
                s.wait_stream(cur)
            outs[0] = one(0)
            for i, s in enumerate(side):
                with torch.cuda.stream(s):
                    outs[i + 1] = one(i + 1)
            for s in side:
                cur.wait_stream(s)
            for o in outs[1:]:
                o.record_stream(cur)
            return outs

        def full():
            return runner.run_gemm(a, weight, a_sf, w_sf, qkv_scale[0:1], 0, 0, None, None)

        row["gemm3_t0"] = time_fn(gemm3)
        row["gemm3_ms_t0"] = time_fn(gemm3_ms)
        row["gemm9216_t0"] = time_fn(full)
        log(M, row)

        def cand(gemms):
            q, k, v = gemms()
            out = torch.empty(M, 3 * W, device="cuda", dtype=torch.bfloat16)
            return qknorm_rope(q, k, v, out, q_norm, k_norm, cos_sin, positions)

        for name, g in (("cand3", gemm3), ("cand3_ms", gemm3_ms)):
            out = cand(g)
            torch.cuda.synchronize()
            row[name + "_bitwise_qkv"] = [
                bool(torch.equal(out[:, i * W:(i + 1) * W], ref[:, i * W:(i + 1) * W])) for i in range(3)
            ]
            row[name + "_nmismatch"] = int((out != ref).sum())
            row[name + "_maxabs"] = float((out.float() - ref.float()).abs().max())
            row[name] = time_fn(lambda: cand(g))
            log(M, name, row[name], row[name + "_bitwise_qkv"])

        q, k, v = gemm3()
        out2 = torch.empty(M, 3 * W, device="cuda", dtype=torch.bfloat16)
        row["pass_copy_v"] = time_fn(lambda: qknorm_rope(q, k, v, out2, q_norm, k_norm, cos_sin, positions))
        # in place on the single-GEMM output layout (q|k|v planes of one buffer)
        buf = full()
        pristine = buf.clone()
        row["pass_inplace"] = time_fn(
            lambda: qknorm_rope(buf[:, :W], buf[:, W:2 * W], buf[:, 2 * W:], buf, q_norm, k_norm, cos_sin, positions)
        )
        # in-place correctness vs the separate-source result for the same GEMM output (q segment alpha only)
        buf.copy_(pristine)
        qknorm_rope(buf[:, :W], buf[:, W:2 * W], buf[:, 2 * W:], buf, q_norm, k_norm, cos_sin, positions)
        sep = torch.empty_like(buf)
        qknorm_rope(pristine[:, :W], pristine[:, W:2 * W], pristine[:, 2 * W:], sep, q_norm, k_norm, cos_sin, positions)
        row["inplace_eq_separate"] = bool(torch.equal(buf, sep))
        log(M, json.dumps(row))
        results[M] = row
    print(json.dumps({"passed": True, "table": results}))


if __name__ == "__main__":
    main()
