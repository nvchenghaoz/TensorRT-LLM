"""Branch 19: SM120 NVFP4 CuTe PACKED gate_up (SwiGLU -> FP4) epilogue, pair bench + bitwise check.

Shapes (FLUX.3 Klein single blocks): gate_up N=18432, K=3072, M=4176 (single 0-23) / 4096 (image.1-4).

  bench : CUDA events, WARMUP then ITERS back-to-back calls, median and (p90-p10)/median.
          Times flux3_sm120_nvfp4.run('packed') (the tree's kernel), main's kernel loaded from git
          (REF_COMMIT, an in-process A/A-style yardstick) and TRT-LLM CUTLASS FP4GemmRunner t0/t2
          (BF16 out, same N/K) in the same job.
  check : untimed sequence with the same WARMUP/ITERS; every iteration's PACKED data bytes
          (rows < M), the full scale buffer and the down GEMM fed by them compared bitwise with the
          protected reference (main's unmodified kernel source at REF_COMMIT, run in-process) and
          with iteration 0; inputs checked against pristine copies; CUDA graph replay x3.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4 as m
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

REF_COMMIT = "c9db0ddadb232cbde7c4d90211b1f762c90f44b9"
KERNEL_PATH = "tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/sm120/flux3_sm120_nvfp4_kernel.py"
WARMUP = 10
ITERS = 100
H, FF = 3072, 9216


def log(*a):
    print(*a, flush=True)


def load_ref_kernel_cls():
    src = subprocess.run(
        ["git", "show", f"{REF_COMMIT}:{KERNEL_PATH}"], capture_output=True, text=True, check=True
    ).stdout
    d = tempfile.mkdtemp(prefix="b19ref_")
    path = os.path.join(d, "flux3_ref_kernel.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("flux3_ref_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Flux3Sm120Nvfp4Kernel


_REF = {}


def run_ref(*args, **kwargs):
    if "cls" not in _REF:
        _REF["cls"] = load_ref_kernel_cls()
        _REF["cache"] = {}
    saved = (m.Flux3Sm120Nvfp4Kernel, m._COMPILED)
    m.Flux3Sm120Nvfp4Kernel, m._COMPILED = _REF["cls"], _REF["cache"]
    try:
        return m.run(*args, **kwargs)
    finally:
        m.Flux3Sm120Nvfp4Kernel, m._COMPILED = saved


def make_case(M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x_mlp = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    w_up = torch.randn(2 * FF, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02

    def gs(t):
        return float(t.abs().amax().float() / (448.0 * 6.0))

    def inv(s):
        return torch.tensor([1.0 / s], device=dev, dtype=torch.float32)

    s_mlp, s_wup, s_wdown = map(gs, (x_mlp, w_up, w_down))
    s_down_in = 0.05
    a_mlp, sf_mlp = fp.nvfp4_quantize(x_mlp, inv(s_mlp))
    q_up, sf_up = fp.nvfp4_quantize(w_up, inv(s_wup))
    q_down, sf_down = fp.nvfp4_quantize(w_down, inv(s_wdown))
    packed_args = (a_mlp, q_up, sf_mlp, sf_up, s_mlp, s_wup)
    packed_kw = {"output_inverse_scale": inv(s_down_in)}
    down_rest = (q_down, sf_down, s_down_in, s_wdown)
    return packed_args, packed_kw, down_rest


def tensors(args, kw):
    return [a for a in list(args) + list(kw.values()) if isinstance(a, torch.Tensor)]


def packed(args, kw):
    return m.run("packed", *args, **kw)


def down(data, scale, rest):
    q_down, sf_down, s_in, s_w = rest
    return torch.ops.trtllm.flux3_primsts_nvfp4_linear(data, q_down, scale, sf_down, s_in, s_w)


def stat(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med


def time_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(ITERS)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return stat([s.elapsed_time(e) for s, e in ev])


def bench():
    out = {}
    runner = fp._sm120_fp4_gemm_runner()
    for M in (4176, 4096):
        args, kw, _ = make_case(M)
        pristine = [t.clone() for t in tensors(args, kw)]
        a, w, sfa, sfw, s_in, s_w = args
        alpha = m._scalar(a.device, s_in * s_w)

        def cut(tactic):
            return lambda: runner.run_gemm(a, w, sfa, sfw, alpha, 0, tactic, None, None)

        rows = {}
        # Two rounds, interleaving the kernels, to expose drift.
        for rnd in range(2):
            for name, fn in (("packed", lambda: packed(args, kw)),
                             ("packed_main_ref", lambda: run_ref("packed", *args, **kw)),
                             ("cutlass_t0", cut(0)), ("cutlass_t2", cut(2))):
                med, noise = time_fn(fn)
                rows.setdefault(name, []).append((med, noise))
                log(f"M={M} round {rnd} {name}: {med * 1e3:.1f} us (noise {noise:.3f})")
        for p, t in zip(pristine, tensors(args, kw)):
            assert torch.equal(p, t), "input mutated"
        out[M] = rows
    med, noise = out[4176]["packed"][-1]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
        "table_us": {str(M): {k: [round(x[0] * 1e3, 1) for x in v] for k, v in r.items()}
                     for M, r in out.items()},
        "noise": {str(M): {k: [round(x[1], 4) for x in v] for k, v in r.items()}
                  for M, r in out.items()},
        "metric": "run('packed') N=18432 K=3072 M=4176, CUDA events median",
        "output": {"class": "exact"},
    }))


def check():
    results = {}
    passed = True
    for M in (4096, 4176, 4256):
        args, kw, rest = make_case(M, seed=M)
        pristine = [t.clone() for t in tensors(args, kw)]
        ref_data, ref_scale = run_ref("packed", *args, **kw)
        ref_data, ref_scale = ref_data.clone(), ref_scale.clone()
        ref_down = down(ref_data.contiguous(), ref_scale, rest).clone()
        for _ in range(WARMUP):
            packed(args, kw)
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, tensors(args, kw)))
        first = None
        n_ref = n_first = n_down = 0
        mism_bytes = 0
        for i in range(ITERS):
            data, scale = packed(args, kw)
            if first is None:
                first = (data.clone(), scale.clone())
            same = torch.equal(data, ref_data) and torch.equal(scale, ref_scale)
            n_ref += same
            if not same:
                mism_bytes = max(mism_bytes, int((data != ref_data).sum()) + int((scale != ref_scale).sum()))
            n_first += torch.equal(data, first[0]) and torch.equal(scale, first[1])
            if i < 3 or i == ITERS - 1:
                n_down += torch.equal(down(data.contiguous(), scale, rest), ref_down)
        inputs_ok &= all(torch.equal(p, t) for p, t in zip(pristine, tensors(args, kw)))
        # A/B/A: the reference again after the candidate calls.
        ref2 = run_ref("packed", *args, **kw)
        ref_stable = torch.equal(ref2[0], ref_data) and torch.equal(ref2[1], ref_scale)
        ok = (n_ref == ITERS and n_first == ITERS and n_down == 4 and inputs_ok and ref_stable
              and tuple(first[0].shape) == tuple(ref_data.shape))
        passed &= ok
        results[f"M{M}"] = {
            "packed_bitwise_vs_main_ref": f"{n_ref}/{ITERS}",
            "packed_bitwise_vs_iter0": f"{n_first}/{ITERS}",
            "down_bitwise_vs_main_ref": f"{n_down}/4",
            "max_mismatched_bytes": mism_bytes,
            "inputs_unchanged": inputs_ok,
            "ref_stable_ABA": ref_stable,
            "data_shape": list(first[0].shape),
            "scale_numel": int(first[1].numel()),
            "ok": ok,
        }
        log(f"M={M}: {results[f'M{M}']}")
    # CUDA graph capture / replay x3 at M=4176
    args, kw, rest = make_case(4176, seed=7)
    rd, rs = run_ref("packed", *args, **kw)
    rd, rs = rd.clone(), rs.clone()
    packed(args, kw)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            gd, gs_ = packed(args, kw)
    graph_ok = True
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        graph_ok &= torch.equal(gd, rd) and torch.equal(gs_, rs)
    results["cuda_graph_replay_x3_bitwise"] = graph_ok
    passed &= graph_ok
    log(f"cuda graph replay bitwise: {graph_ok}")
    print(json.dumps({
        "passed": bool(passed),
        "reference": f"main's unmodified kernel ({REF_COMMIT[:12]}:{KERNEL_PATH}) run in-process, "
                     "bitwise on PACKED data rows<M, full scale buffer and the down GEMM output",
        "results": results,
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    (bench if sys.argv[1:] == ["bench"] else check)()
