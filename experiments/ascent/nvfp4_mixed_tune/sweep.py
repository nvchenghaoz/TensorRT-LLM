# STEP 0 for idea 13 (nvfp4_gemm tactics at mixed prefill rows 2048+d / 4096+d).
# Model-free; synthetic NVFP4 operands of the real Qwen3.8-27B W4A4 prefill shapes.
#   1. Per (shape, M): time EVERY valid NVFP4GemmUnifiedRunner tactic (cutlass configs +
#      cuBLASLt heuristic indices for that M), cold L2 (>= 400 MB of rotated weight copies in
#      one CUDA graph), CUDA events.
#   2. Main's AutoTuner picks (this tree's tuning_config), two fresh tunes in one process.
#   3. Kernel names behind each cuBLASLt index at 2048 vs 2079 (positional identity check).
# Reports: main pick cost vs per-M oracle, per shape, per M.
import json
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import nvfp4_coldtune as ct  # noqa: E402

SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o_out": (5120, 6144),
          "down": (5120, 17408)}
MS = tuple(int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else
                            "2048,2055,2063,2079,4096,4127,6175,8192".split(",")))
WARM, ITERS = 3, 15
log = ct.log


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, autotune
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    ct.MS = tuple(sorted(set(MS) | {8192}))
    shapes = [ct.Shape(nm, n, k) for nm, (n, k) in SHAPES.items()]
    r = ct.runner()
    out = {"tactics": {}, "picks": {}, "kernels": {}}
    # 2. main's tuner picks, two fresh tunes
    tuner = AutoTuner.get()
    for rep in range(2):
        tuner.clear_cache()
        for s in shapes:
            with autotune():
                torch.ops.trtllm.nvfp4_gemm(*s.inputs(8192), torch.bfloat16,
                                            allowed_backends=ct.BACKENDS)
        torch.cuda.synchronize()
        p = {}
        for s in shapes:
            p[s.name] = {}
            for m in MS:
                hit, rid, tac, t = tuner.profiling_cache.search_cache(
                    ct.OP, [r], tuple(tuner._get_input_sizes(s.inputs(m))),
                    NVFP4GemmUnifiedRunner.tuning_config, apply_map_to_tuning_buckets=True)
                p[s.name][m] = list(tac) if isinstance(tac, tuple) else tac
        out["picks"]["tune%d" % rep] = p
        log("tune%d picks %s" % (rep, json.dumps(p)))
    # 1. all tactics
    for s in shapes:
        out["tactics"][s.name] = {}
        for m in MS:
            tacs = r.get_valid_tactics(s.inputs(m), OptimizationProfile())
            res = {}
            for tac in tacs:
                try:
                    st = ct.time_tactic(s, m, tac, WARM, ITERS)
                    res["%s%d" % tac] = round(st["median_ms"], 5)
                except Exception as e:  # noqa: BLE001
                    res["%s%d" % tac] = None
                    log("  fail %s M=%d %s: %s" % (s.name, m, tac, str(e)[:100]))
            out["tactics"][s.name][m] = res
            best = min((v, k) for k, v in res.items() if v is not None)
            log("%s M=%d best %s %.4f ms | %s" % (s.name, m, best[1], best[0],
                                                  json.dumps(res)))
    # 3. kernel names behind cuBLASLt indices
    for s in shapes:
        for m in (2048, 2079, 4096, 4127):
            if m not in MS:
                continue
            tacs = [t for t in r.get_valid_tactics(s.inputs(m), OptimizationProfile())
                    if t[0] == "cublaslt"]
            for tac in tacs:
                key = "%s/M%d/%s%d" % (s.name, m, tac[0], tac[1])
                try:
                    out["kernels"][key] = ct.kernel_names(s, m, tac)
                except Exception as e:  # noqa: BLE001
                    out["kernels"][key] = ["err %s" % e]
                log("  kernels %s: %s" % (key, out["kernels"][key]))
    # summary: main pick vs oracle
    summ = {}
    for rep in out["picks"]:
        tot_main, tot_best = {}, {}
        for s in shapes:
            for m in MS:
                res = out["tactics"][s.name][m]
                pk = out["picks"][rep][s.name][m]
                key = "%s%d" % tuple(pk)
                bestv = min(v for v in res.values() if v is not None)
                mv = res.get(key)
                tot_main[m] = tot_main.get(m, 0) + (mv or float("nan")) * \
                    {"qkvz": 48, "qkv": 16, "o_out": 64, "down": 64}[s.name]
                tot_best[m] = tot_best.get(m, 0) + bestv * \
                    {"qkvz": 48, "qkv": 16, "o_out": 64, "down": 64}[s.name]
        summ[rep] = {m: {"main_step_ms": round(tot_main[m], 3),
                         "oracle_step_ms": round(tot_best[m], 3),
                         "loss_pct": round(100 * (tot_main[m] / tot_best[m] - 1), 2)}
                     for m in MS}
        log("%s per-step (x layers) main pick vs oracle: %s" % (rep, json.dumps(summ[rep])))
    out["summary"] = summ
    print(json.dumps(out))


if __name__ == "__main__":
    main()
