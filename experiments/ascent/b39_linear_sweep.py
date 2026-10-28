"""Branch 39 step 1: FLUX.3 LINEAR (attn_out, down) TRT-LLM CUTLASS t0 vs FlashInfer stock SM120 tactics.

One job, no build beyond the stock FlashInfer JIT module (gen_gemm_sm120_module_cutlass_fp4).
Shapes: attn_out (N=3072, K=3072) and down (N=3072, K=9216) at M = 4096, 4176, 4256.
Inputs: NVFP4 via flux3_primsts.nvfp4_quantize (flashinfer layout_128x4, no shuffle); down's
input from the real CuTe run('packed') producer; alpha = main's cached device scalar.
Arms (stock index = tile_idx*4 + {0 swap DP, 1 DP, 2 swap SK, 3 SK}; tiles by element TileShape):
  t0 = main's _sm120_cutlass_linear (TRT-LLM FP4GemmRunner tactic 0, 128x128x128 DP).
Timing: CUDA events per call, arms interleaved inside each iteration, median of ITERS, warm L2.
Correctness: pristine restore before timing and before checks; each arm called NREP times from
pristine inputs, compared bitwise to t0 and to its own first call; fp32 reference (dequantized
NVFP4 operands, fp32 matmul, * alpha) gives t0's own error for scale; ulp distance vs t0.
"""

import json
import os
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

ITERS = int(os.environ.get("B39_ITERS", "80"))
WARMUP = 3
NREP = 4
H, FF = 3072, 9216
Ms = [int(x) for x in os.environ.get("B39_MS", "4096,4176,4256").split(",")]
TILES = ["128x32x128", "128x32x256", "128x64x128", "128x64x256",
         "128x128x128", "128x128x256", "256x128x128", "128x256x128"]
VARIANT = ["swapDP", "DP", "swapSK", "SK"]
DEFAULT_ARMS = [-1, 21, 17, 16, 13, 12, 9, 25, 24, 29, 28, 19, 18, 15, 27, 23, 31, 11]
ARMS = [int(x) for x in os.environ.get("B39_ARMS", ",".join(map(str, DEFAULT_ARMS))).split(",")]


def log(*a):
    print(*a, flush=True)


def arm_name(idx):
    if idx < 0:
        return "fi[-1]"
    return f"fi[{idx}]{TILES[idx // 4]}_{VARIANT[idx % 4]}"


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def dequant(data_u8, sf_u8, rows, k):
    """NVFP4 (packed e2m1, low nibble first) with 128x4-swizzled E4M3 scales -> fp32 [rows, k]."""
    lut = E2M1.to(data_u8.device)
    lo = (data_u8 & 0xF).long()
    hi = (data_u8 >> 4).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(rows, k)
    nkb = k // 16
    ncol_tiles = (nkb + 3) // 4
    r = torch.arange(rows, device=data_u8.device).view(-1, 1)
    kb = torch.arange(nkb, device=data_u8.device).view(1, -1)
    off = ((r // 128) * (ncol_tiles * 512) + (kb // 4) * 512 + (r % 32) * 16
           + ((r % 128) // 32) * 4 + (kb % 4))
    sf = sf_u8.view(torch.float8_e4m3fn).float()[off.reshape(-1)].reshape(rows, nkb)
    return (vals.reshape(rows, nkb, 16) * sf.unsqueeze(-1)).reshape(rows, k)


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
    s_down_in = 0.05
    a_attn, sf_attn = fp.nvfp4_quantize(x_attn, inv(s_attn))
    a_mlp, sf_mlp = fp.nvfp4_quantize(x_mlp, inv(s_mlp))
    q_out, sf_out = fp.nvfp4_quantize(w_out, inv(s_wout))
    q_up, sf_up = fp.nvfp4_quantize(w_up, inv(s_wup))
    q_down, sf_down = fp.nvfp4_quantize(w_down, inv(s_wdown))
    packed, packed_sf = flux3_sm120_nvfp4.run(
        "packed", a_mlp, q_up, sf_mlp, sf_up, s_mlp, s_wup, output_inverse_scale=inv(s_down_in))
    packed = packed.contiguous()
    return {
        "attn_out": (a_attn, q_out, sf_attn, sf_out, s_attn, s_wout),
        "down": (packed, q_down, packed_sf.reshape(-1).contiguous(), sf_down, s_down_in, s_wdown),
    }


def stat(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med


def ordered(x_bf16):
    s = x_bf16.view(torch.int16).int()
    return torch.where(s >= 0, s, -32768 - s)


def main():
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_fp4
    import time

    t = time.time()
    mod = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
    log(f"stock module load {time.time() - t:.1f}s; tactic_num={mod.fp4_gemm_tactic_num()}")
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    table = {}
    passed = True
    for M in Ms:
        case = make_case(M, seed=M)
        for shape, args in case.items():
            a, w, sfa, sfw, gsa, gsw = args
            N, K = w.shape[0], w.shape[1] * 2
            alpha = _scalar(a.device, gsa * gsw)
            tensors = [a, w, sfa, sfw, alpha]
            pristine = [x.clone() for x in tensors]

            def t0():  # TRT-LLM FP4GemmRunner tactic 0 (main's route), called directly
                return fp._sm120_fp4_gemm_runner().run_gemm(a, w, sfa, sfw, alpha, 0, 0, None, None)

            outs = {idx: torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for idx in ARMS}

            def mk(idx):
                o = outs[idx]

                def f():
                    mod.fp4_gemm(a, w, sfa, sfw, alpha, o, ws, idx)
                    return o
                return f

            fns = {"t0": t0}
            for idx in ARMS:
                f = mk(idx)
                try:
                    f()
                    torch.cuda.synchronize()
                    fns[arm_name(idx)] = f
                except Exception as e:  # noqa: BLE001
                    log(f"  {arm_name(idx)} failed: {str(e)[:120]}")
            for p, x in zip(pristine, tensors):
                x.copy_(p)
            # ---- timing (interleaved) ----
            for _ in range(WARMUP):
                for f in fns.values():
                    f()
            torch.cuda.synchronize()
            ev = {k: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(ITERS)] for k in fns}
            for i in range(ITERS):
                for k, f in fns.items():
                    ev[k][i][0].record()
                    f()
                    ev[k][i][1].record()
            torch.cuda.synchronize()
            times = {k: stat([s.elapsed_time(e) * 1e3 for s, e in v]) for k, v in ev.items()}
            # ---- correctness (untimed, from pristine) ----
            for p, x in zip(pristine, tensors):
                x.copy_(p)
            ref_t0 = t0().clone()
            ad = dequant(a, sfa, M, K)
            wd = dequant(w, sfw, N, K)
            ref32 = (ad @ wd.t()) * (gsa * gsw)
            del ad, wd
            denom = ref32.abs().clamp_min(1e-6)
            t0_err = (ref_t0.float() - ref32).abs()
            t0_stats = {"max_abs_vs_fp32": float(t0_err.max()),
                        "max_rel_vs_fp32": float((t0_err / denom).max()),
                        "rel_l2_vs_fp32": float(t0_err.norm() / ref32.norm())}
            rows = {}
            for k, f in fns.items():
                firsts = None
                rep_ok = True
                for r in range(NREP):
                    o = f().clone()
                    if firsts is None:
                        firsts = o
                    else:
                        rep_ok &= torch.equal(o, firsts)
                inputs_ok = all(torch.equal(p, x) for p, x in zip(pristine, tensors))
                o = firsts
                bitwise = torch.equal(o, ref_t0)
                d = (o.float() - ref_t0.float()).abs()
                ulp = (ordered(o) - ordered(ref_t0)).abs()
                e32 = (o.float() - ref32).abs()
                rows[k] = {
                    "us": round(times[k][0], 2), "noise": round(times[k][1], 4),
                    "vs_t0": round(times[k][0] / times["t0"][0], 4),
                    "bitwise_t0": bitwise, "repeat_bitwise": rep_ok, "inputs_ok": inputs_ok,
                    "max_abs_vs_t0": float(d.max()), "max_ulp_vs_t0": int(ulp.max()),
                    "diff_frac": float((d > 0).float().mean()),
                    "max_rel_vs_t0": float((d / ref_t0.float().abs().clamp_min(1e-6)).max()),
                    "rel_l2_vs_fp32": float(e32.norm() / ref32.norm()),
                }
                passed &= rep_ok and inputs_ok
            key = f"M{M}_{shape}"
            table[key] = {"t0_fp32": t0_stats, "arms": rows}
            log(f"== {key}  t0 vs fp32: {t0_stats}")
            for k, r in sorted(rows.items(), key=lambda kv: kv[1]["us"]):
                log(f"  {k:28s} {r['us']:8.2f} us  x{r['vs_t0']:.4f}  noise {r['noise']:.3f}  "
                    f"bitwise_t0={r['bitwise_t0']} rep={r['repeat_bitwise']} "
                    f"max_ulp={r['max_ulp_vs_t0']} diff_frac={r['diff_frac']:.2e} "
                    f"relL2_fp32={r['rel_l2_vs_fp32']:.3e}")
            del outs, fns, ref32
            torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(passed), "table": table}))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
