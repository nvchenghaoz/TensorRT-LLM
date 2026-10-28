"""Branch 39 chain pair: the LINEAR GEMMs inside a realistic per-block kernel sequence.

Per NVFP4 single block (M=4176) the denoiser runs, in order on one stream:
  nvfp4 quantize(attn) -> attn_out GEMM -> gated residual add -> LN-like elementwise ->
  nvfp4 quantize(mlp) -> CuTe PACKED gate_up (SwiGLU -> NVFP4) -> down GEMM -> gated residual add.
This chain repeats the sequence BLOCKS times per iteration (so L2 holds the neighbours' data as in
the model, and the GEMMs follow their real producers), timed with CUDA events around each chain.
Prints per-chain ms and the per-block share. Same script on both arms; only the op's route differs.
"""

import json
import sys

import torch

sys.path.insert(0, "experiments/ascent")
import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402

M, H, FF = 4176, 3072, 9216
BLOCKS = 8
ITERS = 40
WARMUP = 3


def log(*a):
    print(*a, flush=True)


def stat(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med


def main():
    g = torch.Generator(device="cuda").manual_seed(0)
    dev = "cuda"
    blocks = []
    for _ in range(BLOCKS):
        w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
        w_up = torch.randn(2 * FF, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
        w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
        s = [float(t.abs().amax().float() / 2688.0) for t in (w_out, w_up, w_down)]
        q = [fp.nvfp4_quantize(t, torch.tensor([1 / si], device=dev)) for t, si in
             zip((w_out, w_up, w_down), s)]
        blocks.append((q, s))
    x0 = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    attn = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    gate = torch.randn(1, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.1
    s_x = float(x0.abs().amax().float() / 2688.0) * 4
    inv_x = torch.tensor([1 / s_x], device=dev)
    s_down_in = 0.05
    inv_down = torch.tensor([1 / s_down_in], device=dev)

    import os as _os

    tail_pair = _os.environ.get("B39_CHAIN_TAIL_PAIR", "1") == "1"

    def chain():
        # Single-block shape as on main (landing 20): gate_up from LN(x), then the tail pair runs
        # down on the current stream || attn_out (+ its quantize) on the tail-pair side stream.
        x = x0.clone()
        for (q, s) in blocks:
            (qo, so), (qu, su), (qd, sd) = q
            h = torch.nn.functional.layer_norm(x, (H,)) * 1.1 + 0.1
            hq, hs = fp.nvfp4_quantize(h, inv_x)
            packed, psf = flux3_sm120_nvfp4.run(
                "packed", hq, qu, hs, su, s_x, s[1], output_inverse_scale=inv_down)
            pk, pkf = packed.contiguous(), psf.reshape(-1).contiguous()

            def launch_down():
                return torch.ops.trtllm.flux3_primsts_nvfp4_linear(pk, qd, pkf, sd, s_down_in, s[2])

            def launch_attn():
                a, sa = fp.nvfp4_quantize(attn, inv_x)
                return torch.ops.trtllm.flux3_primsts_nvfp4_linear(a, qo, sa, so, s_x, s[0])

            if tail_pair:
                o, d = fp._sm120_tail_pair(x.device, launch_down, launch_attn)
            else:
                d = launch_down()
                o = launch_attn()
            x = x + gate * (o + d)
        return x

    if len(sys.argv) > 1 and sys.argv[1] == "variants":
        import os

        def set_route(r):
            if r == "t0":
                os.environ["TRTLLM_FLUX3_SM120_LINEAR"] = "trtllm"
            else:
                os.environ["TRTLLM_FLUX3_SM120_LINEAR"] = "cutlass"
                os.environ["TRTLLM_FLUX3_SM120_LINEAR_FI_TACTIC"] = r
            fp._sm120_linear_backend.cache_clear()
            fp._sm120_linear_fi_tactic.cache_clear()

        routes = ["t0", "17", "18", "27", "16"]
        evs = {r: [] for r in routes}
        refs = {}
        for r in routes:
            set_route(r)
            for _ in range(WARMUP):
                chain()
            refs[r] = chain().clone()
        torch.cuda.synchronize()
        for i in range(ITERS):
            for r in routes:
                set_route(r)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                chain()
                b.record()
                evs[r].append((a, b))
            torch.cuda.synchronize()
        res = {}
        for r in routes:
            med, noise = stat([a.elapsed_time(b) for a, b in evs[r]])
            res[r] = {"per_block_us": round(med / BLOCKS * 1e3, 2), "noise": round(noise, 4),
                      "bitwise_vs_t0_chain": bool(torch.equal(refs[r], refs["t0"]))}
            log(f"route {r}: {res[r]}")
        print(json.dumps({"passed": True, "variants": res}))
        return
    for _ in range(WARMUP):
        chain()
    torch.cuda.synchronize()
    ev = []
    for i in range(ITERS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = chain()
        b.record()
        ev.append((a, b))
    torch.cuda.synchronize()
    ms = [a.elapsed_time(b) for a, b in ev]
    med, noise = stat(ms)
    ref = chain().clone()
    rep = chain()
    log(f"chain of {BLOCKS} blocks: {med:.3f} ms (noise {noise:.3f}); per block {med / BLOCKS * 1e3:.1f} us; "
        f"repeat bitwise {torch.equal(ref, rep)}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med / BLOCKS, "noise_floor": noise}},
        "chain_ms": med, "blocks": BLOCKS, "repeat_bitwise": bool(torch.equal(ref, rep)),
        "output": {"class": "reordered"},
    }))


if __name__ == "__main__":
    main()
