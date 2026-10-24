# Idea 43 paired microbenchmark + untimed check. Identical in base and candidate trees:
# run inline (python -c "<this file>" <mode>) so the base tree needs no copy. Uses the
# tree's own trtllm::nvfp4_gemm (AutoTuner, this tree's tactic space, cold-L2 tuning).
# Decode rows M=64/128, the four W4A4 shapes of Qwen3.8-27B-NVFP4 per decode step:
# qkvz (16384,5120) x48, qkv (14336,5120) x16, o_out (5120,6144) x64, down (5120,17408) x64.
# Timed unit per call: fp4_quantize (the real producer of the GEMM's FP4 input + SF) then
# nvfp4_gemm, nb back-to-back calls over distinct weight copies (>= 400 MB, cold L2) in one
# CUDA graph, PDL on. 10 warmup + 50 timed replays. kernel_ms = per-step layer-weighted
# sum of the per-call medians.
# mode check: the tree's picks vs an FP32 reference (<= 1.25x the base kernel's error,
# cuBLASLt/CUTLASS pick of the base tree recomputed in-process with fi tactics disabled,
# + 1 bf16 ulp), and an untimed sequence with the same warmup/iters as timing: every
# replay's outputs bitwise vs replay 1 and the inputs unchanged.
import json
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import nvfp4_coldtune as ct  # noqa: E402

SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o_out": (5120, 6144),
          "down": (5120, 17408)}
LAYERS = {"qkvz": 48, "qkv": 16, "o_out": 64, "down": 64}
MS = (64, 128)
WARM, ITERS = 10, 50
log = ct.log
mode = sys.argv[1] if len(sys.argv) > 1 else "bench"


def picks_of(tuner, r, shapes):
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    p = {}
    for s in shapes:
        for m in MS:
            hit, rid, tac, t = tuner.profiling_cache.search_cache(
                ct.OP, [r], tuple(tuner._get_input_sizes(s.inputs(m))),
                NVFP4GemmUnifiedRunner.tuning_config, apply_map_to_tuning_buckets=True)
            p["%s/%d" % (s.name, m)] = list(tac) if isinstance(tac, tuple) else tac
    return p


def tune(shapes):
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    tuner = AutoTuner.get()
    tuner.clear_cache()
    for s in shapes:
        with autotune():  # profiles every M bucket <= the input's rows
            torch.ops.trtllm.nvfp4_gemm(*s.inputs(MS[-1]), torch.bfloat16,
                                        allowed_backends=ct.BACKENDS)
        torch.cuda.synchronize()
    return picks_of(tuner, ct.runner(), shapes)


class Chain:
    """x (bf16) -> fp4_quantize -> nvfp4_gemm over weight copy i, for i in range(nb)."""

    def __init__(self, s, m):
        self.s, self.m = s, m
        g = torch.Generator(device="cuda").manual_seed(4321 + m + s.n)
        x = torch.randn(m, s.k, generator=g, device="cuda")
        sc = torch.exp(torch.randn(m, s.k // 16, 1, generator=g, device="cuda") * 2.0)
        self.x = (x.view(m, s.k // 16, 16) * sc).reshape(m, s.k).to(torch.bfloat16)
        self.x0 = self.x.clone()
        self.gs = (448.0 * 6.0 / self.x.float().abs().max()).reshape(1).float()
        self.alpha = (1.0 / (self.gs * s.gs_w)).float()

    def call(self, i, tac=None):
        a, sf = ct.quant(self.x, self.gs)
        w, wsf = self.s.copies[i % self.s.nb]
        if tac is not None:
            return ct.runner()([a, w, sf, wsf, self.alpha], tactic=tac)
        return torch.ops.trtllm.nvfp4_gemm(a, w, sf, wsf, self.alpha, torch.bfloat16,
                                           allowed_backends=ct.BACKENDS)

    def graph(self, tac=None):
        for i in range(self.s.nb):
            self.call(i, tac)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for i in range(self.s.nb):
                self.call(i, tac)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = [self.call(i, tac) for i in range(self.s.nb)]
        return g, outs

    def ref(self):
        a, sf_lin = ct.quant(self.x, self.gs, swizzled=False)
        _, sf = ct.quant(self.x, self.gs)
        self.s.acts[self.m] = dict(a=a, sf=sf, sf_lin=sf_lin, alpha=self.alpha)
        return self.s.ref(self.m)


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
    torch.backends.cuda.matmul.allow_tf32 = False
    has_fi = hasattr(tco, "FlashInferCutlassNVFP4Runner")
    log("tree has fi_cutlass tactics: %s" % has_fi)
    ct.MS = MS
    shapes = [ct.Shape(nm, n, k) for nm, (n, k) in SHAPES.items()]
    picks = tune(shapes)
    log("picks %s" % json.dumps(picks))
    chains = {(s.name, m): Chain(s, m) for s in shapes for m in MS}

    if mode == "bench":
        batch, detail = {}, {}
        for m in MS:
            tot, noise = 0.0, 0.0
            for s in shapes:
                c = chains[(s.name, m)]
                g, _ = c.graph()
                st = ct.stats(ct.time_graph(g, s.nb, WARM, ITERS))
                del g
                d = dict(st, tactic=picks["%s/%d" % (s.name, m)], us=st["median_ms"] * 1e3)
                detail["%s/%d" % (s.name, m)] = d
                tot += LAYERS[s.name] * st["median_ms"]
                noise = max(noise, st["noise"])
                log("  M=%d %s pick %s: %.2f us/call (p90 %.2f) noise %.3f" % (
                    m, s.name, d["tactic"], d["us"], st["p90_ms"] * 1e3, st["noise"]))
            batch[str(m)] = {"kernel_ms": round(tot, 5), "noise_floor": round(noise, 4)}
            log("M=%d per-step W4A4 GEMM chain total %.4f ms" % (m, tot))
        print(json.dumps({"batch": batch, "output": {"class": "reordered" if has_fi else "exact"},
                          "picks": picks, "detail": detail}))
        return

    if mode == "oracle":
        from tensorrt_llm._torch.autotuner import OptimizationProfile
        r = ct.runner()
        res = {}
        for m in MS:
            for s in shapes:
                c = chains[(s.name, m)]
                t = {}
                for rep in range(2):
                    for tac in r.get_valid_tactics(s.inputs(m), OptimizationProfile()):
                        if tac[0] == "cublaslt" and tac[1] > 4:
                            continue
                        g, _ = c.graph(tac)
                        st = ct.stats(ct.time_graph(g, s.nb, WARM, ITERS))
                        del g
                        t.setdefault("%s%d" % tac, []).append(st["median_ms"] * 1e3)
                med = {k: min(v) for k, v in t.items()}
                top = sorted(med.items(), key=lambda kv: kv[1])[:8]
                key = "%s/%d" % (s.name, m)
                res[key] = {"pick": picks[key], "top8": top,
                            "pick_us": med.get("%s%d" % tuple(picks[key]))
                            if isinstance(picks[key], list) else None}
                log("%s pick %s %.2f | chain top8: %s" % (
                    key, picks[key], res[key]["pick_us"] or -1,
                    ", ".join("%s %.2f" % kv for kv in top)))
        print(json.dumps(res))
        return

    # check
    base_picks = None
    if has_fi:
        os.environ["TRTLLM_NVFP4_FI_CUTLASS"] = "0"
        base_picks = tune(shapes)
        os.environ["TRTLLM_NVFP4_FI_CUTLASS"] = "1"
        log("base (fi disabled) picks %s" % json.dumps(base_picks))
        picks = tune(shapes)
        log("tree picks (re-tuned) %s" % json.dumps(picks))
    r = ct.runner()
    ok, res = True, {}
    for s in shapes:
        for m in MS:
            key = "%s/%d" % (s.name, m)
            c = chains[(s.name, m)]
            ref = c.ref()
            ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)
            a, sf = ct.quant(c.x, c.gs)
            inp = [a, s.w, sf, s.w_sf, c.alpha]
            tree_out = r(inp, tactic=ct.as_tac(picks[key]))
            bt = base_picks[key] if base_picks else picks[key]
            base_out = r(inp, tactic=ct.as_tac(bt))
            torch.cuda.synchronize()
            e_t = (tree_out.float() - ref).abs()
            e_b = (base_out.float() - ref).abs()
            tol_ok = bool(torch.isfinite(tree_out).all()) and \
                e_t.max().item() <= 1.25 * e_b.max().item() + ulp and \
                e_t.mean().item() <= 1.25 * e_b.mean().item() + ulp
            # untimed repeated graph replays, same warmup + iters as timing
            g, outs = c.graph()
            first, nmis, per_it = None, 0, []
            for it in range(WARM + ITERS):
                g.replay()
                torch.cuda.synchronize()
                cur = torch.stack([o.clone() for o in outs])
                if first is None:
                    first = cur[0].clone()
                nmis += int(sum(not torch.equal(cur[j], first) for j in range(s.nb)))
                per_it.append((cur[0].float() - ref).abs().max().item())
            inputs_ok = bool(torch.equal(c.x, c.x0))
            del g, outs
            graph_eq_eager = bool(torch.equal(first, tree_out))
            row = {"tree_pick": picks[key], "base_pick": bt,
                   "tree_err_max": e_t.max().item(), "tree_err_mean": e_t.mean().item(),
                   "base_err_max": e_b.max().item(), "base_err_mean": e_b.mean().item(),
                   "bitwise_vs_base": bool(torch.equal(tree_out, base_out)),
                   "bf16_ulp": ulp, "tol_ok": tol_ok, "replays": WARM + ITERS,
                   "calls_per_replay": s.nb, "mismatch_vs_replay1": nmis,
                   "graph_eq_eager": graph_eq_eager, "inputs_unchanged": inputs_ok,
                   "per_replay_max_err": [min(per_it), max(per_it)]}
            res[key] = row
            ok = ok and tol_ok and nmis == 0 and inputs_ok and graph_eq_eager
            log("  %s %s" % (key, json.dumps(row)))
            del ref
            torch.cuda.empty_cache()
    print(json.dumps({"passed": ok, "compared": (
        "tree's nvfp4_gemm pick at M=64/128 vs FP32 reference from the same FP4 operands "
        "(<= 1.25x base-tactic error + 1 bf16 ulp, max and mean); CUDA-graph replays "
        "(fp4_quantize->nvfp4_gemm x nb weight copies, 60 replays) bitwise vs replay 1, "
        "graph == eager, inputs unchanged"), "results": res}))


main()
