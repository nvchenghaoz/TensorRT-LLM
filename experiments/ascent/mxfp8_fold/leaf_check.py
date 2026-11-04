"""Branch 63 (idea 125) on main 18027035: untimed checks for acceptance items 4, 5 and 6.

One EP4 rank of DeepSeek-V4.1-Flash (96 of 384 experts, top-6, hidden 5120, inter 2304, SwiGLU
limit 10); synthetic MXFP4 weights (experiments/ascent/w4a8_fi_chain/common.py layout); no model load.

Item 6: torch.ops.trtllm.fused_moe with Half x + BF16 output under MXFP4 weights / MXFP8 act
        scaling (the in-kernel-quant runner) must raise the moeOp.cpp TORCH_CHECK.
Item 4: the real TrtllmCutlassW4a8Mxfp4Mxfp8Impl.quantize_input/run_moe (landing 21's leaf plus the
        fold) at M=1/2/4/8/32/33/128/2048/6144, seeds 0/1 + skewed routing (+ a <=2-local-picks
        case at M<=32), job-617 protocol: 3+20 eager calls, then a graph of 20 calls replayed 45
        times, every output compared with torch.equal against main's own path computed in-process:
          M<=32: Sm12xW4a8DecodeState.run (landing 21's FI chain, what main's leaf runs)
          M>32 : mxfp8_quantize + fused_moe(fp8 x, input_sf) (main's CUTLASS path)
        with the deterministic unfused finalize (use_fused_finalize=False; the production atomic
        epilogue is run-to-run nondeterministic on main, knowledge 990/1332). Routing is
        instrumented: M<=32 -> FI chain, no mxfp8_quantize, no fused_moe; M>32 -> fused_moe with
        BF16 x and no input_sf (fold), no mxfp8_quantize. run_moe's quantizing fallback
        (do_finalize=False at M<=32) -> mxfp8_quantize + fused_moe(fp8), bitwise = main.
Item 5: from an empty AutoTuner cache, tune fused_moe with BF16 x (the in-kernel-quant runner) under
        autotune() at M=33/2048/6144, for both finalize modes: every tactic of both GEMMs is
        profiled, no failed tactic, no CUDA error; every cache entry is keyed by the FP8 runner's
        unique_id (x dtype float8_e4m3fn); the FP8 runner then hits that cache and picks the same
        tactics; tuned BF16 call == FP8 reference (torch.equal, unfused finalize; within main's own
        spread with the fused finalize).
Last stdout line: {"passed": bool, ...}.
"""

import json
import os
import sys
import time
import traceback

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))

import torch  # noqa: E402

T0 = time.time()
H, I, E, E_GLOBAL, TOPK, LIMIT = 5120, 2304, 96, 384, 6, 10.0
WARM, REPS, REPLAYS = 3, 20, 45
MS = [1, 2, 4, 8, 32, 33, 128, 2048, 6144]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def make_inputs(m, seed=0, skew=False, max_local=None):
    g = torch.Generator(device="cuda").manual_seed(99 + 7 * m + 1000 * seed)
    x = torch.randn(m, H, device="cuda", dtype=torch.float32, generator=g)
    r = torch.arange(m, device="cuda")
    x = (x * torch.pow(2.0, ((r * 5) % 9 - 4).float())[:, None]).to(torch.bfloat16).contiguous()
    gc = torch.Generator().manual_seed(5 + 13 * m + 1000 * seed)
    sel = torch.empty(m, TOPK, dtype=torch.int32)
    for t in range(m):
        if max_local is not None:
            nl = int(torch.randint(1, max_local + 1, (1,), generator=gc))
            loc = torch.randperm(E, generator=gc)[:nl]
            rem = torch.randperm(E_GLOBAL - E, generator=gc)[: TOPK - nl] + E
            perm = torch.cat([loc, rem])[torch.randperm(TOPK, generator=gc)]
        else:
            perm = torch.randperm(E_GLOBAL, generator=gc)[:TOPK]
            if skew:
                perm = perm[perm != 3][: TOPK - 1]
                perm = torch.cat([torch.tensor([3]), perm])
            elif not (perm < E).any():
                perm[0] = int(torch.randint(0, E, (1,), generator=gc))
        sel[t] = perm.to(torch.int32)
    scales = torch.softmax(torch.randn(m, TOPK, generator=gc), dim=-1).float()
    return x, sel.cuda(), scales.cuda()


class Main:
    """main's CUTLASS path: mxfp8_quantize + fused_moe(fp8 x, input_sf), same args as the leaf."""

    def __init__(self, w31, s31, w2v, s2s, ff):
        from tensorrt_llm._torch.utils import ActivationType
        self.w31, self.w2v, self.ff = w31, w2v, ff
        one = torch.ones(E, dtype=torch.float32, device="cuda")
        self.qs = [s31, one, s2s, one.clone()]
        self.limit = torch.full((E,), LIMIT, dtype=torch.float32, device="cuda")
        self.act = int(ActivationType.Swiglu)

    def op(self, xin, xsf, ids, w, out_dtype=torch.bfloat16):
        return torch.ops.trtllm.fused_moe(
            xin, ids, w, self.w31, None, self.w2v, None, out_dtype,
            quant_scales=self.qs, input_sf=xsf, swizzled_input_sf=True,
            swiglu_alpha=None, swiglu_beta=None, swiglu_limit=self.limit, swiglu_clamp_after_silu=False,
            tp_size=1, tp_rank=0, ep_size=4, ep_rank=0, cluster_size=1, cluster_rank=0,
            enable_alltoall=False, use_deepseek_fp8_block_scale=False, use_w4_group_scaling=False,
            use_int8_woq_per_channel=False, use_mxfp8_act_scaling=True, min_latency_mode=False,
            use_fused_finalize=self.ff, tune_max_num_tokens=8192, activation_type=self.act,
            unpadded_hidden_size=H,
        )[0]

    def __call__(self, x, ids, w):
        xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True, alignment=128)
        return self.op(xq, xsf, ids, w)

    def fold(self, x, ids, w):
        return self.op(x, None, ids, w)


def make_leaf(st, w31, s31, w2v, s2s, ff):
    from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8 import TrtllmCutlassW4a8Mxfp4Mxfp8Impl as Leaf
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8CutlassFusedMoEMethod
    from tensorrt_llm._torch.utils import ActivationType

    leaf = Leaf.__new__(Leaf)
    torch.nn.Module.__init__(leaf)
    one = torch.ones(E, dtype=torch.float32, device="cuda")
    attrs = dict(
        _fi_decode=st, quant_method=W4A8MXFP4MXFP8CutlassFusedMoEMethod(),
        w3_w1_weight=w31, w2_weight=w2v, w3_w1_bias=None, w2_bias=None,
        quant_scales=[s31, one, s2s, one.clone()], act_alpha=None, act_beta=None,
        act_clamp=torch.full((E,), LIMIT, dtype=torch.float32, device="cuda"),
        act_clamp_after_silu=False, tp_size=1, tp_rank=0, ep_size=4, ep_rank=0,
        cluster_size=1, cluster_rank=0, use_fused_finalize=ff, tune_max_num_tokens=8192,
        activation_type=ActivationType.Swiglu, unpadded_hidden_size=H, hidden_size=H,
    )
    for k, v in attrs.items():
        object.__setattr__(leaf, k, v)
    return leaf


class Probe:
    """Counts what the leaf's call ran (eager calls; capture also runs the Python once)."""

    def __init__(self, leaf, st, leaf_mod):
        self.reset()
        orig_q = leaf_mod.quantize_input_mxfp8
        orig_fm = leaf._run_fused_moe_op
        orig_run = st.run

        def q(*a, **k):
            self.n_quant += 1
            return orig_q(*a, **k)

        def fm(ctx, **k):
            self.fused.append((ctx.x.dtype, ctx.x_sf is None))
            return orig_fm(ctx, **k)

        def run(*a, **k):
            self.n_fi += 1
            return orig_run(*a, **k)

        leaf_mod.quantize_input_mxfp8 = q
        object.__setattr__(leaf, "_run_fused_moe_op", fm)
        st.run = run
        self.direct_fi = orig_run

    def reset(self):
        self.n_quant, self.n_fi, self.fused = 0, 0, []

    def summary(self):
        return {"mxfp8_quantize_calls": self.n_quant, "fi_chain_calls": self.n_fi,
                "fused_moe_calls": [(str(d), nosf) for d, nosf in self.fused]}


def leaf_call(leaf, x, ids, w, do_finalize=True):
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoECommPlan, MoERunContext
    xq, xsf = leaf.quantize_input(x, post_quant_comm=False)
    plan = MoECommPlan(input_sf_swizzled=True, enable_alltoall=False, moe_output=None,
                       payload_in_workspace=False)
    ctx = MoERunContext(token_selected_experts=ids, token_final_scales=w, x=xq, x_sf=xsf,
                        output_dtype=torch.bfloat16, comm_plan=plan, do_finalize=do_finalize)
    return leaf.run_moe(ctx)


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def sequence(fn, cmp):
    """Job-617 protocol: WARM+REPS eager calls, then a graph of REPS calls replayed REPLAYS times."""
    for _ in range(WARM + REPS):
        cmp(fn())
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(WARM):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [fn() for _ in range(REPS)]
    for _ in range(REPLAYS):
        g.replay()
        torch.cuda.synchronize()
        for o in outs:
            cmp(o)
    del g, outs


def fresh_runners():
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()  # pitfall 1332: the C++ runner keeps the finalize mode it was built with
    torch.cuda.synchronize()


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from common import make_weights, trtllm_layout
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8 as leaf_mod
    from tensorrt_llm._torch.moe.fused_moe.cutlass.w4a8_mxfp4_mxfp8_sm12x_decode import (
        FI_DECODE_MAX_TOKENS, Sm12xW4a8DecodeState, fi_decode_available)

    res = {"items": {"4": {}, "5": {}, "6": {}}}
    ok = {"4": True, "5": True, "6": True}
    has_fold = hasattr(leaf_mod.TrtllmCutlassW4a8Mxfp4Mxfp8Impl, "_folds_quant_into_cutlass")
    avail, why = fi_decode_available()
    log("gpu", torch.cuda.get_device_name(), "fold in tree:", has_fold, "fi avail:", avail, why,
        "FI_DECODE_MAX_TOKENS", FI_DECODE_MAX_TOKENS, "fold max", getattr(leaf_mod, "_MXFP8_FOLD_MAX_TOKENS", None))
    res.update(fold_in_tree=has_fold, fi_available=avail, fi_max_tokens=FI_DECODE_MAX_TOKENS)
    if not (avail and has_fold):
        print(json.dumps({"passed": False, "error": f"fi_available={avail} ({why}) fold={has_fold}"}))
        sys.exit(1)

    torch.manual_seed(0)
    w1, s1, w2, s2 = make_weights()
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)

    # ---------------- item 6 ----------------
    fresh_runners()
    main_det = Main(w31, s31, w2v, s2s, ff=False)
    x, ids, w = make_inputs(64)
    try:
        main_det.op(x.half(), None, ids, w, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        res["items"]["6"] = {"raised": False}
        ok["6"] = False
    except Exception as ex:  # noqa: BLE001
        msg = str(ex)
        hit = "requires the input dtype to equal the output dtype" in msg
        res["items"]["6"] = {"raised": True, "type": type(ex).__name__, "clear_message": hit,
                             "message": msg.splitlines()[0][:400] if msg else ""}
        ok["6"] = hit
    # The rejected construction must not leave a runner behind, and BF16->BF16 must still work.
    bad_left = [k for k in MoERunner.runner_dict if k[0] == torch.float16]
    torch.cuda.synchronize()
    o = main_det.fold(x, ids, w)
    torch.cuda.synchronize()
    res["items"]["6"].update(half_runner_cached=bool(bad_left), bf16_after_ok=bool(torch.isfinite(o.float()).all()))
    ok["6"] = ok["6"] and not bad_left and res["items"]["6"]["bf16_after_ok"]
    log("item6", ok["6"], json.dumps(res["items"]["6"]))

    # ---------------- item 4 ----------------
    st = Sm12xW4a8DecodeState(w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
                              num_local_experts=E, slot_start=0, swiglu_limit=LIMIT,
                              enable_pdl=os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1")
    st.warmup(TOPK)
    leaf = make_leaf(st, w31, s31, w2v, s2s, ff=False)
    probe = Probe(leaf, st, leaf_mod)
    fresh_runners()
    AutoTuner.get().clear_cache()
    xt, it, wt = make_inputs(max(MS))
    with autotune():  # production order: main's FP8 runner tunes; the fold runner shares its id
        main_det(xt, it, wt)
    torch.cuda.synchronize()
    log("item4: tuned (unfused finalize) at M", max(MS))

    cases = []
    for m in MS:
        cases += [(m, 0, False, None), (m, 1, False, None), (m, 0, True, None)]
        if m <= FI_DECODE_MAX_TOKENS:
            cases.append((m, 2, False, 2))
    for (m, seed, skew, maxl) in cases:
        key = f"M{m}_seed{seed}" + ("_skew" if skew else "") + (f"_le{maxl}local" if maxl else "")
        try:
            x, ids, w = make_inputs(m, seed=seed, skew=skew, max_local=maxl)
            x0 = x.clone()
            fi_path = m <= FI_DECODE_MAX_TOKENS
            ref_fn = (lambda: probe.direct_fi(x, ids, w)) if fi_path else (lambda: main_det(x, ids, w))
            ref = ref_fn().clone()
            reps = [ref_fn() for _ in range(5)]
            ref_det = all(torch.equal(r, ref) for r in reps)
            env = max(rel_err(r, ref) for r in reps)
            probe.reset()
            o1 = leaf_call(leaf, x, ids, w)
            route = probe.summary()
            if fi_path:
                route_ok = route["fi_chain_calls"] == 1 and route["mxfp8_quantize_calls"] == 0 and not route["fused_moe_calls"]
            else:
                route_ok = (route["fi_chain_calls"] == 0 and route["mxfp8_quantize_calls"] == 0
                            and route["fused_moe_calls"] == [("torch.bfloat16", True)])
            st_ = {"iters": 0, "mismatch": 0, "max_rel": 0.0}

            def cmp(o):
                st_["iters"] += 1
                if not torch.equal(o, ref):
                    st_["mismatch"] += 1
                    st_["max_rel"] = max(st_["max_rel"], rel_err(o, ref))

            cmp(o1)
            sequence(lambda: leaf_call(leaf, x, ids, w), cmp)
            nloc = (ids < E).sum(dim=1)
            r = {"fi_path": fi_path, "route": route, "route_ok": route_ok, "iters": st_["iters"],
                 "mismatch": st_["mismatch"], "mismatch_max_rel": st_["max_rel"], "ref_deterministic": ref_det,
                 "ref_self_envelope_rel": env, "tokens_ge3_local": int((nloc >= 3).sum()),
                 "max_local_picks": int(nloc.max()), "inputs_unchanged": bool(torch.equal(x, x0)),
                 "finite": bool(torch.isfinite(ref.float()).all()), "nonzero_frac": float((ref != 0).float().mean())}
            # Bitwise required wherever main's own path is deterministic (all CUTLASS cases with the unfused
            # finalize; FI cases whose tokens have <= 2 local picks). FI cases with >= 3 local picks per token
            # scatter-add in completion order on main too: then within main's own repeat envelope.
            case_ok = route_ok and r["inputs_unchanged"] and r["finite"] and r["nonzero_frac"] > 0.5
            if ref_det:
                case_ok = case_ok and st_["mismatch"] == 0
            else:
                case_ok = case_ok and fi_path and st_["max_rel"] <= max(2 * env, 1e-2)
            if maxl is not None or not fi_path:
                case_ok = case_ok and ref_det and st_["mismatch"] == 0
            if fi_path:  # run_moe's quantizing fallback (a call the FI chain does not serve)
                probe.reset()
                fb = leaf_call(leaf, x, ids, w, do_finalize=False)
                fr = probe.summary()
                fref = main_det(x, ids, w)
                fb_ok = (fr["mxfp8_quantize_calls"] == 1 and fr["fi_chain_calls"] == 0
                         and fr["fused_moe_calls"] == [("torch.float8_e4m3fn", False)] and torch.equal(fb, fref))
                n_fb = 0
                for _ in range(10):
                    n_fb += int(not torch.equal(leaf_call(leaf, x, ids, w, do_finalize=False), fref))
                r.update(fallback_route=fr, fallback_bitwise=bool(fb_ok), fallback_repeat_mismatch=n_fb)
                case_ok = case_ok and fb_ok and n_fb == 0
            r["ok"] = bool(case_ok)
            ok["4"] = ok["4"] and case_ok
            res["items"]["4"][key] = r
            log(key, "OK" if case_ok else "FAIL", json.dumps(r))
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            res["items"]["4"][key] = {"ok": False, "error": repr(ex)}
            ok["4"] = False
        torch.cuda.empty_cache()

    # ---------------- item 5 ----------------
    tuner = AutoTuner.get()
    calls = []
    orig_choose = tuner.choose_one

    def choose(custom_op, runners, tuning_config, inputs, **kw):
        out = orig_choose(custom_op, runners, tuning_config, inputs, **kw)
        if custom_op.startswith("trtllm::fused_moe"):
            hit = tuner.profiling_cache.search_cache(custom_op, runners, tuple(tuner._get_input_sizes(inputs)),
                                                     tuning_config)[0]
            calls.append({"op": custom_op, "x": str(runners[0].x_dtype), "tuning": tuner.is_tuning_mode,
                          "hit": bool(hit), "tactic": int(out[1])})
        return out

    tuner.choose_one = choose
    profiled = {}
    orig_fwd = MoERunner.forward

    def fwd(self, inputs, gemm_idx=0, tactic=-1, do_preparation=False):
        if tuner.is_tuning_mode and not do_preparation and tactic >= 0:
            profiled.setdefault((str(self.x_dtype), gemm_idx), set()).add(int(tactic))
        return orig_fwd(self, inputs, gemm_idx=gemm_idx, tactic=tactic, do_preparation=do_preparation)

    MoERunner.forward = fwd
    for ff in (False, True):
        for m in (33, 2048, 6144):
            key = f"M{m}_fused_finalize{int(ff)}"
            try:
                fresh_runners()
                tuner.clear_cache()
                tuner.stats = type(tuner.stats)()
                calls.clear()
                profiled.clear()
                mm = Main(w31, s31, w2v, s2s, ff=ff)
                x, ids, w = make_inputs(m)
                with autotune():
                    tuned = mm.fold(x, ids, w)  # BF16 x: the in-kernel-quant runner tunes from an empty cache
                torch.cuda.synchronize()
                tune_calls = list(calls)
                failed = {k: len(v) for k, v in tuner.stats.failed_profiling_count.items() if v}
                keys = list(tuner.profiling_cache.cache.keys())
                ids_ok = bool(keys) and all("torch.float8_e4m3fn" in k[2] and "torch.bfloat16, torch.int64" not in k[2]
                                            for k in keys)
                bf16_rt = [v for k, v in MoERunner.runner_dict.items() if k[0] == torch.bfloat16]
                n_tac = {g: bf16_rt[0].get_tactic_num(g) for g in (1, 2)} if bf16_rt else {}
                prof = {g: len(profiled.get(("torch.bfloat16", g), ())) for g in (1, 2)}
                all_prof = bool(n_tac) and all(prof[g] == n_tac[g] for g in (1, 2))
                calls.clear()
                fp8_out = mm(x, ids, w)  # FP8 runner, inference mode: must hit the shared entries
                bf16_out = mm.fold(x, ids, w)
                torch.cuda.synchronize()
                infer_calls = list(calls)
                fp8_calls = [c for c in infer_calls if c["x"] == "torch.float8_e4m3fn"]
                bf_calls = [c for c in infer_calls if c["x"] == "torch.bfloat16"]
                tac_tuned = [c["tactic"] for c in tune_calls]
                same_tac = (len(fp8_calls) == 2 and all(c["hit"] for c in fp8_calls)
                            and [c["tactic"] for c in fp8_calls] == [c["tactic"] for c in bf_calls] == tac_tuned)
                if not ff:
                    eq = bool(torch.equal(tuned, fp8_out) and torch.equal(bf16_out, fp8_out))
                    num_ok, num = eq, {"bitwise_tuned_vs_fp8": eq}
                else:
                    spread = max(rel_err(mm(x, ids, w), fp8_out) for _ in range(5))
                    e_fold = max(rel_err(mm.fold(x, ids, w), fp8_out) for _ in range(5))
                    num_ok = e_fold <= max(2 * spread, 1e-2)
                    num = {"fold_vs_fp8_max_rel": e_fold, "fp8_self_spread_rel": spread}
                r = {"tune_calls": tune_calls, "failed_profiling": failed, "cache_entries": len(keys),
                     "cache_keys_fp8_id": ids_ok, "tactic_num": n_tac, "profiled_distinct_tactics": prof,
                     "all_tactics_profiled": all_prof, "infer_calls": infer_calls, "fp8_hits_same_tactics": same_tac,
                     **num, "example_key": str(keys[0])[:300] if keys else None}
                case_ok = (not failed) and ids_ok and all_prof and same_tac and num_ok and bool(
                    torch.isfinite(tuned.float()).all())
                r["ok"] = bool(case_ok)
                ok["5"] = ok["5"] and case_ok
                res["items"]["5"][key] = r
                log(key, "OK" if case_ok else "FAIL", json.dumps(r)[:3000])
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                res["items"]["5"][key] = {"ok": False, "error": repr(ex)}
                ok["5"] = False
            torch.cuda.empty_cache()
    MoERunner.forward = orig_fwd
    tuner.choose_one = orig_choose

    res["ok"] = ok
    res["passed"] = all(ok.values())
    res["compared"] = ("item 4: leaf (quantize_input + run_moe) vs main's path in-process (FI chain direct at "
                       "M<=32, mxfp8_quantize + fused_moe(fp8) above), torch.equal every iteration, unfused "
                       "finalize; item 5: tuned BF16-input call vs FP8-input call; item 6: Half x + BF16 out raises")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
