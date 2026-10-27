"""Branch 11 pair microbenchmark and correctness check for the SM120 NVFP4 LINEAR route.

Calls torch.ops.trtllm.flux3_primsts_nvfp4_linear (what PreparedNVFP4 dispatches to) at the
FLUX.3 Klein single-block LINEAR shapes: attn_out (M,3072,K=3072) and down (M,3072,K=9216), with the
down input produced by the real PACKED_SWIGLU kernel (so its 128x4 scales are the kernel's own).
On main this runs the CuTe kernel; on the branch it runs CUTLASS SM120 fp4_gemm config 0.

  bench : CUDA events after warmup, median of ITERS, per-iteration sum of attn_out+down.
  check : untimed sequence with the same warmup/iteration count; every iteration compared bitwise
          with the protected reference (the unmodified CuTe kernel, flux3_sm120_nvfp4.run('linear'))
          and with iteration 0; inputs checked against pristine copies; several padded M; CUDA graph
          capture/replay; AutoTuner cache must stay free of GEMM entries (no tuning at all).
"""

import json
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

WARMUP = 3
ITERS = 60
H, FF = 3072, 9216


def log(*a):
    print(*a, flush=True)


def quant(x, inv):
    return fp.nvfp4_quantize(x, inv)


def make_case(M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x_attn = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    x_mlp = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_up = torch.randn(2 * FF, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02

    def gs(t):
        return float(t.abs().amax().float() / (448.0 * 6.0))

    def inv(s):
        return torch.tensor([1.0 / s], device=dev, dtype=torch.float32)

    s_attn, s_mlp, s_wout, s_wup, s_wdown = map(gs, (x_attn, x_mlp, w_out, w_up, w_down))
    s_down_in = 0.05  # static input scale of down (checkpoint-style constant)
    a_attn, sf_attn = quant(x_attn, inv(s_attn))
    a_mlp, sf_mlp = quant(x_mlp, inv(s_mlp))
    q_out, sf_out = quant(w_out, inv(s_wout))
    q_up, sf_up = quant(w_up, inv(s_wup))
    q_down, sf_down = quant(w_down, inv(s_wdown))
    # Real packed-SwiGLU producer for the down input and its scales.
    packed, packed_sf = flux3_sm120_nvfp4.run(
        "packed", a_mlp, q_up, sf_mlp, sf_up, s_mlp, s_wup,
        output_inverse_scale=inv(s_down_in),
    )
    packed = packed.contiguous()
    attn_args = (a_attn, q_out, sf_attn, sf_out, s_attn, s_wout)
    down_args = (packed, q_down, packed_sf, sf_down, s_down_in, s_wdown)
    return attn_args, down_args


def op(args):
    return torch.ops.trtllm.flux3_primsts_nvfp4_linear(*args)


def ref(args):
    return flux3_sm120_nvfp4.run("linear", *args)


def tensors(args):
    return [a for a in args if isinstance(a, torch.Tensor)]


def bench():
    M = 4176
    attn_args, down_args = make_case(M)
    pristine = [t.clone() for t in tensors(attn_args) + tensors(down_args)]
    log(f"bench M={M} sm120_linear_backend={getattr(fp, '_sm120_linear_backend', lambda: 'cute')()}")
    for _ in range(WARMUP):
        op(attn_args)
        op(down_args)
    torch.cuda.synchronize()
    for p, t in zip(pristine, tensors(attn_args) + tensors(down_args)):
        assert torch.equal(p, t)
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(ITERS)]
    for i in range(ITERS):
        ev[i][0].record()
        op(attn_args)
        ev[i][1].record()
        op(down_args)
        ev[i][2].record()
    torch.cuda.synchronize()
    a = [e[0].elapsed_time(e[1]) for e in ev]
    d = [e[1].elapsed_time(e[2]) for e in ev]
    s = [x + y for x, y in zip(a, d)]

    def stat(v):
        v = sorted(v)
        med = v[len(v) // 2]
        return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med

    (sm, sn), (am, an), (dm, dn) = stat(s), stat(a), stat(d)
    log(f"attn_out {am*1e3:.1f} us (noise {an:.3f})  down {dm*1e3:.1f} us (noise {dn:.3f})  "
        f"sum {sm*1e3:.1f} us (noise {sn:.3f})")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": sm, "noise_floor": sn}},
        "per_shape": {"attn_out_ms": [am, an], "down_ms": [dm, dn]},
        "M": M,
        "output": {"class": "exact"},
    }))


def check():
    from tensorrt_llm._torch.autotuner import AutoTuner

    results = {}
    passed = True
    backend = fp._sm120_linear_backend()
    log(f"check sm120_linear_backend={backend}")
    AutoTuner.get().clear_cache()
    for M in (4176, 4256, 4336, 4416, 4096, 4496):
        attn_args, down_args = make_case(M, seed=M)
        for name, args in (("attn_out", attn_args), ("down", down_args)):
            pristine = [t.clone() for t in tensors(args)]
            reference = ref(args).clone()
            # protocol-matching untimed sequence: warmup, then ITERS compared iterations
            for _ in range(WARMUP):
                op(args)
            for p, t in zip(pristine, tensors(args)):
                assert torch.equal(p, t), "input mutated during warmup"
            first = None
            n_ref_ok = n_first_ok = 0
            max_abs = 0.0
            for i in range(ITERS):
                out = op(args)
                if first is None:
                    first = out.clone()
                n_ref_ok += bool(torch.equal(out, reference))
                n_first_ok += bool(torch.equal(out, first))
                max_abs = max(max_abs, float((out.float() - reference.float()).abs().max()))
            inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, tensors(args)))
            ok = (n_ref_ok == ITERS and n_first_ok == ITERS and inputs_ok
                  and tuple(first.shape) == tuple(reference.shape))
            passed &= ok
            results[f"M{M}_{name}"] = {
                "bitwise_vs_cute_ref": f"{n_ref_ok}/{ITERS}",
                "bitwise_vs_iter0": f"{n_first_ok}/{ITERS}",
                "max_abs": max_abs,
                "inputs_unchanged": inputs_ok,
                "ok": ok,
            }
            log(f"M={M} {name}: {results[f'M{M}_{name}']}")
    # CUDA graph capture / replay (twice) at M=4176
    attn_args, down_args = make_case(4176, seed=7)
    r_attn, r_down = ref(attn_args).clone(), ref(down_args).clone()
    op(attn_args), op(down_args)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            g_attn = op(attn_args)
            g_down = op(down_args)
    graph_ok = True
    for rep in range(3):
        graph.replay()
        torch.cuda.synchronize()
        graph_ok &= torch.equal(g_attn, r_attn) and torch.equal(g_down, r_down)
    results["cuda_graph_replay_x3_bitwise"] = graph_ok
    passed &= graph_ok
    log(f"cuda graph replay bitwise: {graph_ok}")
    gemm_entries = [str(k) for k in AutoTuner.get().profiling_cache.cache.keys()
                    if "gemm" in str(k).lower()]
    results["autotuner_gemm_cache_entries"] = gemm_entries
    no_tuning = len(gemm_entries) == 0
    results["no_tuning_launched"] = no_tuning
    passed &= no_tuning
    print(json.dumps({"passed": bool(passed), "backend": backend,
                      "reference": "unmodified CuTe kernel flux3_sm120_nvfp4.run('linear'), bitwise",
                      "results": results}))


if __name__ == "__main__":
    torch.manual_seed(0)
    (bench if sys.argv[1:] == ["bench"] else check)()
