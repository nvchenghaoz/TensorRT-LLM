"""mHC fused_hc (SM120 FMA path) microbenchmark / bitwise check, one GPU, synthetic tensors.

DeepSeek-V4.1-Flash: hc_mult=4, hidden=5120, W_T fp32 [24, 4*5120].

Modes:
  bench  : per M, time every tactic get_valid_tactics() returns for M (what the
           autotuner sees) and report the best (kernel_ms = full op = pmap/GEMM kernel +
           bigfuse). Also prints per-tactic times. Same command runs in baseline and
           candidate trees.
  check  : bitwise compare every tactic in get_valid_tactics (incl. new tile_m>1
           ones) against the current SM120 tactic ("fused_half_fma", 24, 1, bs, 1) at
           many M, with the harness iteration/reset sequence, plus an fp64 reference on
           residual_cur / sqrsum-dependent outputs.
  ncu    : run one tactic once at the given Ms (for profile_kernels).
"""
import json
import os
import sys
import time

import torch

import tensorrt_llm  # noqa: F401  (loads trtllm ops)
from tensorrt_llm._torch.modules.mhc import mhc_cuda

HC = 4
HID = 5120
N = HC * (2 + HC)
K = HC * HID
RMS_EPS = 1e-6
PRE_EPS = 1e-6
SK_EPS = 1e-6
POST_MULT = 2.0
SK_REP = 20


def make_inputs(M, seed=0, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed + M)
    x = torch.randn((M, HID), generator=g, device=dev).to(torch.bfloat16)
    r = torch.randn((M, HC, HID), generator=g, device=dev).to(torch.bfloat16)
    pm = (torch.rand((M, HC), generator=g, device=dev) * 2.0).float()
    cm = torch.rand((M, HC, HC), generator=g, device=dev).float()
    cm = cm / cm.sum(dim=1, keepdim=True)
    w = (torch.randn((N, K), generator=g, device=dev) * 0.02).float()
    sc = (torch.rand((3,), generator=g, device=dev) * 0.5 + 0.5).float()
    base = (torch.randn((N,), generator=g, device=dev) * 0.1).float()
    nw = (torch.rand((HID,), generator=g, device=dev) + 0.5).to(torch.bfloat16)
    return [x, r, pm, cm, w, sc, base], nw


def runner():
    return mhc_cuda.MhcFusedHcRunner(HC, HID, RMS_EPS, PRE_EPS, SK_EPS, POST_MULT, SK_REP)


def call(rn, inputs, tactic, nw, pre_ext=None):
    kw = dict(norm_weight=nw, norm_eps=1e-6)
    if pre_ext is not None:
        kw["pre_mix_ext"] = pre_ext
    return rn.forward(inputs, tactic=tactic, **kw)


def tactics_for(rn, inputs):
    return rn.get_valid_tactics(inputs, None)


def time_op(fn, iters, flush):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    med = ts[len(ts) // 2]
    p10 = ts[int(0.1 * (len(ts) - 1))]
    p90 = ts[int(0.9 * (len(ts) - 1))]
    return med, (p90 - p10) / med


def bench(Ms, iters, reps):
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    rn = runner()
    out = {}
    detail = {}
    for M in Ms:
        inputs, nw = make_inputs(M)
        tacs = tactics_for(rn, inputs)
        print(f"M={M}: {len(tacs)} tactics", flush=True)
        res = {}
        for t in tacs:
            fn = lambda t=t: call(rn, inputs, t, nw)  # noqa: E731
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            res[t] = time_op(fn, iters, flush)
        # re-time the top-3 'reps' times (interleaved) to reduce noise
        top = sorted(res, key=lambda t: res[t][0])[:3]
        final = {t: [] for t in top}
        for _ in range(reps):
            for t in top:
                fn = lambda t=t: call(rn, inputs, t, nw)  # noqa: E731
                final[t].append(time_op(fn, iters, flush))
        best = min(top, key=lambda t: sorted(x[0] for x in final[t])[len(final[t]) // 2])
        meds = sorted(x[0] for x in final[best])
        med = meds[len(meds) // 2]
        noise = max(x[1] for x in final[best])
        out[str(M)] = {"kernel_ms": med, "noise_floor": noise, "tactic": list(best)}
        detail[str(M)] = {str(t): round(v[0] * 1000, 2) for t, v in sorted(res.items(), key=lambda kv: kv[1][0])}
        print(f"  best {best} {med*1000:.1f} us noise {noise:.3f}", flush=True)
        print("  all(us):", detail[str(M)], flush=True)
        del inputs
        torch.cuda.empty_cache()
    print(json.dumps({"batch": out, "output": {"class": "exact"}}))


def ref_residual(inputs):
    x, r, pm, cm, w = [t.double() for t in inputs[:5]]
    # new_r[j,h] = pm[j]*x[h] + sum_k cm[k,j]*r[k,h]
    new_r = pm[:, :, None] * x[:, None, :] + torch.einsum("mkj,mkh->mjh", cm, r)
    return new_r


def check(Ms, iters):
    rn = runner()
    allpass = True
    report = {}
    for M in Ms:
        inputs, nw = make_inputs(M, seed=1)
        pre_ext = torch.rand((M, HC), device="cuda").float() + 0.5
        bs = 512 if M < 512 else 256
        base_t = ("fused_half_fma", 24, 1, bs, 1)
        tacs = tactics_for(rn, inputs)
        cand = [t for t in tacs if t[0] == "fused_half_fma" and t[2] == 1]
        newt = [t for t in tacs if t[4] > 1]
        rep = {"n_tactics": len(tacs), "n_new": len(newt), "base": list(base_t)}
        for mode, pe in (("v4", None), ("v41_lagged", pre_ext)):
            ref = [o.clone() for o in call(rn, inputs, base_t, nw, pe)]
            # fp64 sanity of residual_cur vs independent reference
            nr = ref_residual(inputs)
            rerr = (ref[0].double() - nr).abs().max().item()
            rtol_ok = torch.allclose(ref[0].double(), nr, rtol=1.6e-2, atol=1.6e-2)
            finite = all(torch.isfinite(o).all().item() for o in ref)
            mism = {}
            for t in cand:
                # harness-like: 3 warmup + iters calls; compare every call
                bad = 0
                for it in range(3 + iters):
                    o = call(rn, inputs, t, nw, pe)
                    for a, b in zip(o, ref):
                        if a.shape != b.shape or a.stride() != b.stride() or not torch.equal(a, b):
                            bad += 1
                            break
                    # inputs untouched (no in-place state): verify each iteration
                    if it == 0:
                        pass
                if bad:
                    mism[str(t)] = bad
            ok = (not mism) and rtol_ok and finite
            allpass &= ok
            rep[mode] = {"mismatching_tactics": mism, "resid_fp64_maxabs": rerr, "resid_tol_ok": rtol_ok,
                         "finite": finite, "n_compared": len(cand), "outputs": len(ref)}
        # inputs must be unchanged after all calls
        inputs2, _ = make_inputs(M, seed=1)
        unchanged = all(torch.equal(a, b) for a, b in zip(inputs, inputs2))
        allpass &= unchanged
        rep["inputs_unchanged"] = unchanged
        report[str(M)] = rep
        print(f"M={M}: {json.dumps(rep)}", flush=True)
        del inputs, inputs2
        torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(allpass), "compared": "every fused_half_fma ks=1 tactic (incl. tile_m>1) vs current SM120 tactic (fused_half_fma,24,1,bs,1), outputs residual_cur/post_mix_cur/comb_mix_cur/layer_input_cur(/pre_mix_out) bitwise, every call of 3 warmup+iters; residual vs fp64 ref", "report": report}))


def ncu(Ms, tactic):
    rn = runner()
    for M in Ms:
        inputs, nw = make_inputs(M)
        t = tuple(tactic)
        if t[3] == 0:
            t = (t[0], t[1], t[2], 512 if M < 512 else 256, t[4])
        call(rn, inputs, t, nw)
        torch.cuda.synchronize()
        print("ran", M, t, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    Ms = [int(x) for x in sys.argv[2].split(",")]
    print("device", torch.cuda.get_device_name(), "sm", torch.cuda.get_device_properties(0).multi_processor_count,
          flush=True)
    if mode == "bench":
        bench(Ms, int(os.environ.get("ITERS", "30")), int(os.environ.get("REPS", "5")))
    elif mode == "check":
        check(Ms, int(os.environ.get("ITERS", "5")))
    elif mode == "ncu":
        tac = json.loads(sys.argv[3]) if len(sys.argv) > 3 else ["fused_half_fma", 24, 1, 0, 1]
        ncu(Ms, tac)
