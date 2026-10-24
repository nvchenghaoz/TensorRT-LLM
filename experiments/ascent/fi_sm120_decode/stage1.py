# Idea 43 stage 1: decode M=64/128 NVFP4 W4A4 GEMMs on under-filled grids.
# Model-free, synthetic NVFP4 operands of the real Qwen3.8-27B W4A4 shapes.
# Per (shape, M): main's AutoTuner pick (this tree's tuning_config, cold L2), every valid
# main tactic (cutlass / cublaslt incl. its splitKreduce), and all 32 FlashInfer 0.6.18 SM120
# CUTLASS NVFP4 tactics (8 tiles x {swapDP, DP, swapSK, SK}, PDL on). Timing: one CUDA graph
# of nb back-to-back calls over distinct weight copies (>= 400 MB, cold L2), CUDA events.
# Also: error of every tactic vs an FP32 reference from the same FP4 operands, bitwise
# equality with main's pick.
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import nvfp4_coldtune as ct  # noqa: E402

SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o_out": (5120, 6144),
          "down": (5120, 17408)}
LAYERS = {"qkvz": 48, "qkv": 16, "o_out": 64, "down": 64}
MS = (64, 128)
WARM, ITERS = 10, 50
DRAM = 1508.57e9
log = ct.log
FI_NAMES = ["128x32x128", "128x32x256", "128x64x128", "128x64x256", "128x128x128",
            "128x128x256", "256x128x128", "128x256x128"]
FI_SCHED = ["swapDP", "DP", "swapSK", "SK"]
_WS = {}


def fi_name(t):
    return "fi_%s_%s" % (FI_NAMES[t // 4], FI_SCHED[t % 4])


class FiRunner:
    """Callable with the NVFP4GemmUnifiedRunner signature: r(inputs, tactic=t)."""

    def __init__(self, mod):
        self.mod = mod
        if "ws" not in _WS:
            _WS["ws"] = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        self.ws = _WS["ws"]

    def __call__(self, inputs, tactic=0):
        a, w, asf, wsf, alpha = inputs
        out = torch.empty(a.shape[0], w.shape[0], dtype=torch.bfloat16, device="cuda")
        self.mod.fp4_gemm(a.view(torch.uint8), w.view(torch.uint8), asf.view(torch.uint8),
                          wsf.view(torch.uint8), alpha, out, self.ws, int(tactic))
        return out


def heavy_acts(s, m):
    """Activations with a wide dynamic range so FP32 accumulation order matters."""
    g = torch.Generator(device="cuda").manual_seed(1234 + m + s.n)
    a = torch.randn(m, s.k, generator=g, device="cuda")
    sc = torch.exp(torch.randn(m, s.k // 16, 1, generator=g, device="cuda") * 2.0)
    a = (a.view(m, s.k // 16, 16) * sc).reshape(m, s.k).to(torch.bfloat16)
    gs_a = (448.0 * 6.0 / a.float().abs().max()).reshape(1).float()
    af, asf = ct.quant(a, gs_a)
    _, asf_lin = ct.quant(a, gs_a, swizzled=False)
    alpha = (1.0 / (gs_a * s.gs_w)).float()
    return dict(a=af, sf=asf, sf_lin=asf_lin, alpha=alpha)


def time_with(r, s, m, tac):
    g, _ = ct.build_graph(s, m, tac, r)
    st = ct.stats(ct.time_graph(g, s.nb, WARM, ITERS))
    del g
    return st


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, autotune
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    from flashinfer.jit.gemm.core import gen_gemm_sm120_module_cutlass_fp4
    torch.backends.cuda.matmul.allow_tf32 = False
    log("building/loading FlashInfer SM120 CUTLASS FP4 module ...")
    mod = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
    nfi = int(mod.fp4_gemm_tactic_num())
    log("FlashInfer tactics: %d" % nfi)
    fi = FiRunner(mod)
    ct.MS = tuple(sorted(set(MS) | {8192}))
    shapes = [ct.Shape(nm, n, k) for nm, (n, k) in SHAPES.items()]
    for s in shapes:
        for m in MS:
            s.acts[m] = heavy_acts(s, m)
    r = ct.runner()
    tuner = AutoTuner.get()
    tuner.clear_cache()
    for s in shapes:
        with autotune():
            torch.ops.trtllm.nvfp4_gemm(*s.inputs(8192), torch.bfloat16,
                                        allowed_backends=ct.BACKENDS)
        torch.cuda.synchronize()
        log("tuned %s" % s.name)
    out = {"picks": {}, "rows": {}}
    for s in shapes:
        for m in MS:
            hit, rid, tac, t = tuner.profiling_cache.search_cache(
                ct.OP, [r], tuple(tuner._get_input_sizes(s.inputs(m))),
                NVFP4GemmUnifiedRunner.tuning_config, apply_map_to_tuning_buckets=True)
            out["picks"]["%s/%d" % (s.name, m)] = list(tac)
    log("main picks %s" % json.dumps(out["picks"]))
    step = {m: {"main": 0.0, "best_main": 0.0, "best_fi": 0.0, "best_all": 0.0, "floor": 0.0}
            for m in MS}
    for s in shapes:
        for m in MS:
            key = "%s/%d" % (s.name, m)
            ref = s.ref(m)
            ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)
            pick = tuple(out["picks"][key])
            main_out = r(s.inputs(m), tactic=pick)
            torch.cuda.synchronize()

            def err(o):
                d = (o.float() - ref).abs()
                return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
                        "bitwise_main": bool(torch.equal(o, main_out))}

            res = {}
            for tac in r.get_valid_tactics(s.inputs(m), OptimizationProfile()):
                nm = "%s%d" % tac
                try:
                    st = time_with(r, s, m, tac)
                    o = r(s.inputs(m), tactic=tac)
                    res[nm] = dict(us=st["median_ms"] * 1e3, p90_us=st["p90_ms"] * 1e3,
                                   noise=st["noise"], **err(o))
                except Exception as e:  # noqa: BLE001
                    log("  fail %s %s: %s" % (key, nm, str(e)[:120]))
            for t in range(nfi):
                nm = fi_name(t)
                try:
                    st = time_with(fi, s, m, t)
                    o = fi(s.inputs(m), tactic=t)
                    torch.cuda.synchronize()
                    res[nm] = dict(us=st["median_ms"] * 1e3, p90_us=st["p90_ms"] * 1e3,
                                   noise=st["noise"], **err(o))
                except Exception as e:  # noqa: BLE001
                    log("  fail %s %s: %s" % (key, nm, str(e)[:120]))
            pk = "%s%d" % pick
            mains = {k: v for k, v in res.items() if not k.startswith("fi_")}
            fis = {k: v for k, v in res.items() if k.startswith("fi_")}
            bm = min(mains, key=lambda k: mains[k]["us"])
            bf = min(fis, key=lambda k: fis[k]["us"]) if fis else None
            floor_us = (s.wbytes + m * s.k // 2) / DRAM * 1e6
            row = {"floor_us": floor_us, "bf16_ulp": ulp, "main_pick": pk,
                   "main_pick_us": res[pk]["us"], "main_pick_err": res[pk]["max_abs"],
                   "best_main": bm, "best_main_us": mains[bm]["us"],
                   "best_fi": bf, "best_fi_us": fis[bf]["us"] if bf else None,
                   "best_fi_err": fis[bf]["max_abs"] if bf else None,
                   "best_fi_bitwise_main": fis[bf]["bitwise_main"] if bf else None,
                   "all": res}
            try:
                row["main_pick_kernels"] = ct.kernel_names(s, m, pick)
            except Exception as e:  # noqa: BLE001
                row["main_pick_kernels"] = ["err %s" % e]
            out["rows"][key] = row
            L = LAYERS[s.name]
            step[m]["main"] += L * row["main_pick_us"]
            step[m]["best_main"] += L * row["best_main_us"]
            step[m]["best_fi"] += L * min(row["best_fi_us"] or 1e9, row["main_pick_us"])
            step[m]["best_all"] += L * min(v["us"] for v in res.values())
            step[m]["floor"] += L * floor_us
            log("%s: floor %.1f | main pick %s %.2f us (err %.3g) | best main %s %.2f | "
                "best FI %s %.2f (err %.3g, bitwise %s)" % (
                    key, floor_us, pk, row["main_pick_us"], row["main_pick_err"], bm,
                    row["best_main_us"], bf, row["best_fi_us"] or -1, row["best_fi_err"] or -1,
                    row["best_fi_bitwise_main"]))
            top = sorted(res.items(), key=lambda kv: kv[1]["us"])[:8]
            log("   top8: " + ", ".join("%s %.2f" % (k, v["us"]) for k, v in top))
            log("   main pick kernels: %s" % row["main_pick_kernels"])
            del ref
            torch.cuda.empty_cache()
    out["step_us"] = step
    log("per-step (x layers) us: %s" % json.dumps(step))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
