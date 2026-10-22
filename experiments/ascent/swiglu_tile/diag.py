"""Branch 11 STEP 0: SwiGLU-epilogue gate_up dispatch arms on the real MLP chain.

Arms per M (same process, rotated order each iteration, A/A copy of 'unf'):
  unf : nvfp4_gemm (autotuned) + fused_silu_mul_nvfp4_permuted  (main below 17 rows)
  t0  : nvfp4_gate_up_swiglu_nvfp4 tile 0 (256x128x128)
  t1  : nvfp4_gate_up_swiglu_nvfp4 tile 1 (128x128x256)
  main: main's rule (unf below 17, t1 below 1024, t0 from 1024)
Chain per layer: (M>=1024) fused_gemma_add_rmsnorm_nvfp4 -> GatedMLP, else BF16
input -> GatedMLP. L=4 distinct layers (~570 MB), 256 MB L2 scrub before each
timed run. Decode rows: CUDA graph (PDL on). Prefill rows: eager.
Prints per-arm median/p10/p90 us per layer and bitwise agreement of outputs.
"""
import json
import os
import statistics
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "swiglu_epi"))
from mlp_common import EPS, GS_X, H, build_mlp, dev  # noqa: E402

from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP  # noqa: E402
from tensorrt_llm._torch.utils import Fp4QuantizedTensor  # noqa: E402

L = int(os.environ.get("BENCH_L", "4"))
MS = [int(v) for v in os.environ.get(
    "BENCH_MS", "8,16,24,32,48,64,1024,2048,2055,2063,2079,4096,4127,6175,8192").split(",")]
WARM, ITERS = int(os.environ.get("BENCH_WARM", "10")), int(os.environ.get("BENCH_ITERS", "50"))
ARMS = os.environ.get("BENCH_ARMS", "unf,t0,t1,unf2").split(",")
ORIG_TILE = GatedMLP._swiglu_epi_tile


def set_arm(arm):
    if arm.startswith("unf"):
        GatedMLP._SWIGLU_EPI_MIN_ROWS = 1 << 30
        GatedMLP._swiglu_epi_tile = staticmethod(ORIG_TILE)
    elif arm == "main":
        GatedMLP._SWIGLU_EPI_MIN_ROWS = 17
        GatedMLP._swiglu_epi_tile = staticmethod(lambda m: 0 if m >= 1024 else 1)
    else:
        t = int(arm[1:2])
        GatedMLP._SWIGLU_EPI_MIN_ROWS = 1
        GatedMLP._swiglu_epi_tile = staticmethod(lambda m, t=t: t)


def main():
    mlps = [build_mlp(100 + i, 1.0) for i in range(L)]
    print("permuted", [bool(getattr(m, "_swiglu_epi_permuted", False)) for m in mlps], flush=True)
    gammas = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
    gs = torch.tensor([GS_X], dtype=torch.float32, device=dev)
    scrub = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)
    res = {}
    for m in MS:
        gen = torch.Generator(device=dev).manual_seed(7 + m)
        x0 = (torch.randn((L, m, H), generator=gen, device=dev) * 0.7).to(torch.bfloat16)
        r0 = (torch.randn((L, m, H), generator=gen, device=dev) * 3.0).to(torch.bfloat16)
        xp, rp = x0.clone(), r0.clone()
        prefill = m >= 1024

        def layer(i):
            if prefill:
                q, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(xp[i], rp[i], gammas[i], EPS, gs)
                return mlps[i](Fp4QuantizedTensor(q, sf))
            return mlps[i](xp[i])

        set_arm("unf")
        with autotune():
            xp.copy_(x0)
            rp.copy_(r0)
            layer(0)
        torch.cuda.synchronize()
        runners, outs = {}, {}
        for arm in ARMS:
            set_arm(arm)
            if prefill:
                def run(arm=arm):
                    set_arm(arm)
                    return [layer(i) for i in range(L)]
                xp.copy_(x0)
                rp.copy_(r0)
                o = run()
                torch.cuda.synchronize()
                outs[arm] = torch.stack(o).clone()
                runners[arm] = run
            else:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        xp.copy_(x0)
                        rp.copy_(r0)
                        [layer(i) for i in range(L)]
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    go = [layer(i) for i in range(L)]
                xp.copy_(x0)
                rp.copy_(r0)
                g.replay()
                torch.cuda.synchronize()
                outs[arm] = torch.stack(go).clone()
                runners[arm] = (lambda g=g: g.replay())
        ref = outs[ARMS[0]]
        exact = {a: bool(torch.equal(outs[a], ref)) for a in ARMS}
        ts = {a: [] for a in ARMS}
        for it in range(WARM + ITERS):
            order = ARMS[it % len(ARMS):] + ARMS[:it % len(ARMS)]
            for arm in order:
                xp.copy_(x0)
                rp.copy_(r0)
                scrub.zero_()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                runners[arm]()
                e1.record()
                e1.synchronize()
                if it >= WARM:
                    ts[arm].append(e0.elapsed_time(e1) * 1e3 / L)
        row = {}
        for a in ARMS:
            v = sorted(ts[a])
            n = len(v)
            med = statistics.median(v)
            row[a] = {"med": round(med, 2), "p10": round(v[int(0.1 * (n - 1))], 2),
                      "p90": round(v[int(0.9 * (n - 1))], 2)}
        row["exact_vs_" + ARMS[0]] = exact
        res[str(m)] = row
        print("M", m, json.dumps(row), flush=True)
        del runners, outs
        torch.cuda.empty_cache()
    set_arm("main")
    print(json.dumps({"diag": res}))


if __name__ == "__main__":
    main()
