"""Branch 36 / idea 51 STEP 0 + lever sizing (one process, one GPU, graph replays interleaved).

Per served M (16384, 16464, 32848), synthetic NVFP4 inputs at the real shapes (N = K = 3072):
  op_main   main's fused op (fused QKV GEMM + plain v_quant), keep_v=False, all levers off
  op_rev / op_evl / op_pdl / op_all   the same op with VQ_REVERSE / V_EVICT_LAST / VQ_PDL / all
  gemm      the fused GEMM launch alone;  vq / vq_rev  v_quant alone (after one GEMM)
  noop      the fused GEMM built with FLUX3_EPI_DIAG_NOOP (mainloop floor; diagnostic only)
  t256      landing 6's bare 256x128 QKV3 (flux3_sm120_large_m.gemm_qkv3, BF16 out)
  b128      main's bare 128x128 QKV3 (flux3_sm120_strided_gemm.gemm_qkv3, BF16 out)
Each graph holds R calls; a timing = one replay / R; median over ITERS, noise (p90-p10)/median.
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "idea14_epi"))
from op import SM, N, make_inputs  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def stats(t):
    t = sorted(t)
    med = t[len(t) // 2]
    return med, (t[int(0.9 * len(t)) - 1] - t[int(0.1 * len(t))]) / med


def build_noop():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        "flux3_qkv_epi_b36_noop",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_qkv_epi_fp8_sm120.cu"],
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16", "-DENABLE_FP4",
                                        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                                        "-DFLUX3_EPI_DIAG_NOOP"],
        extra_cflags=["-DFAST_BUILD"],
    )
    return spec.build_and_load()


def main():
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import _MIN_SCALING_FACTOR
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_large_m as large_m
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    log("X_POLICY env:", os.environ.get("TRTLLM_FLUX3_ADALN_WARP_X_POLICY", "<unset: default>"),
        "(no adaLN in this microbench)")
    dev = torch.device("cuda")
    strided.preload(dev)
    epi.preload(dev)
    mod = epi.module()
    t0 = time.time()
    noop = build_noop()
    log(f"noop built {time.time() - t0:.0f}s")
    ms = [int(x) for x in os.environ.get("S0_MS", "16384,16464,32848").split(",")]
    R = int(os.environ.get("S0_R", "4"))
    ITERS = int(os.environ.get("S0_ITERS", "40"))
    # Job 461 also timed V_EVICT_LAST and VQ_PDL arms (removed after: inside the noise).
    levers = {"op_main": False, "op_rev": True}
    result = {}
    for m in ms:
        a, w, sfa, sfb, alpha, qn, kn, cs, pos = make_inputs(m, dev)
        qkv8 = torch.empty((3, m, N), dtype=torch.uint8, device=dev)
        gout = torch.empty((m, 3 * N), dtype=torch.bfloat16, device=dev)
        v = gout.view(3, m, N)[2]
        scales = torch.empty(8, dtype=torch.float32, device=dev)
        n_part = ((m + 127) // 128) * (N // 128)
        parts = torch.zeros(n_part + 8, dtype=torch.float32, device=dev)
        ws = strided.workspace(dev, 0)
        grid = max(1, min(2 * 188, (m * N // 8 + 511) // 512))
        minsf = float(_MIN_SCALING_FACTOR)

        def op(lv):
            def f():
                epi.VQ_REVERSE = lv
                epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, False)
            return f

        fns = {name: op(lv) for name, lv in levers.items()}
        fns["gemm"] = lambda: mod.qkv3_epi_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, qkv8, v, parts,
                                               ws, minsf)
        fns["vq"] = lambda: mod.v_quant(v, qkv8[2], parts, n_part, qn, kn, scales, SM, minsf,
                                        grid, 0, 0)
        fns["vq_rev"] = lambda: mod.v_quant(v, qkv8[2], parts, n_part, qn, kn, scales, SM, minsf,
                                            grid, 0, 1)
        fns["noop"] = lambda: noop.qkv3_epi_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, qkv8, v,
                                                parts, ws, minsf)
        fns["b128"] = lambda: strided.gemm_qkv3(a, w, sfa, sfb, alpha, gout)
        if large_m.enabled("qkv3"):
            large_m.preload_qkv3(strided._WORKSPACE_BYTES)
            fns["t256"] = lambda: large_m.gemm_qkv3(a, w, sfa, sfb, alpha, gout, ws)
        for f in fns.values():
            f()
            f()
        torch.cuda.synchronize()
        graphs = {}
        for name, f in fns.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(R):
                    f()
            graphs[name] = g
        epi.VQ_REVERSE = False
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        times = {n: [] for n in graphs}
        for it in range(ITERS):
            for name, g in graphs.items():
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                times[name].append(e0.elapsed_time(e1) / R)
        row = {}
        for name, t in times.items():
            med, nf = stats(t)
            row[name] = {"ms": round(med, 4), "nf": round(nf, 4)}
        row["gap_ms"] = round(row["op_main"]["ms"] - row["gemm"]["ms"] - row["vq"]["ms"], 4)
        log(f"M{m}", json.dumps(row))
        result[str(m)] = row
        del graphs
        torch.cuda.empty_cache()
    print(json.dumps({"step0": result}))


if __name__ == "__main__":
    sys.exit(main())
