# Branch 114 / idea 200 correctness check (untimed), on the candidate tree.
# 1. Kill switch: cuda_core_nvfp4_gemm with TRTLLM_CUDA_CORE_NVFP4_M1_PREWAIT=0 (the previous plain loop at
#    M=1) vs default, eager, M=1..4, both shapes (gate_up 34816x5120, down 5120x17408), real producers
#    (fp4_quantize for gate_up, fused_silu_mul_nvfp4 for down). Must be bitwise equal.
# 2. Repeated execution: the served chain fp4_quantize -> gate_up -> fused_silu_mul_nvfp4 -> down for L
#    layers in one CUDA graph (PDL on), 10 warmup + 50 replays, every output NaN-poisoned before each replay;
#    every replay must equal replay 1 and the eager chain; inputs/weights unchanged (hashes) at the end.
#    Same for a kill-switch graph. Also compares against an independent FP32 reference (advisory).
# 3. Two fresh cold-L2 AutoTuner tunes (cache cleared each time) of trtllm::nvfp4_gemm at M=1 (bucket 1)
#    on both shapes: the pick must be cuda_core.
# 4. Output hashes of the eager chain at M=1..4 (compare with the base arm of the run_gpu_pair job).
import hashlib
import json
import os
import subprocess
import time

os.environ["TRTLLM_ENABLE_PDL"] = "1"

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402
from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner  # noqa: E402
from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4 as _fsm  # noqa: E402,F401

KILL = "TRTLLM_CUDA_CORE_NVFP4_M1_PREWAIT"
dev = torch.device("cuda")
L = 12
WARM, ITERS = 10, 50


def log(*a):
    print(*a, flush=True)


def tree():
    try:
        wt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        h = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD", "HEAD^{tree}"], capture_output=True, text=True,
                           timeout=20).stdout.split()
        d = subprocess.run(["git", "-C", wt, "status", "--porcelain"], capture_output=True, text=True,
                           timeout=20).stdout.strip().splitlines()
        log(f"TREE_COMMIT={h[0] if h else '?'} TREE={h[1] if len(h) > 1 else '?'} DIRTY={len(d)}")
    except Exception as e:  # noqa: BLE001
        log(f"TREE_COMMIT=? {e!r}")


def sha(ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:20]


def q4(x, gs):
    return torch.ops.trtllm.fp4_quantize(x, gs, 16, False, True)


def gemm(aq, wq, asf, wsf, alpha):
    return torch.ops.trtllm.cuda_core_nvfp4_gemm(aq, wq, asf, wsf, alpha, None, torch.bfloat16, act_sf_swizzled=True)


def mkw(n, k, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    w = torch.randn(n, k, device=dev, generator=g).to(torch.bfloat16)
    w.view(n, k // 16, 16)[::5, ::11, :] = 0
    w.view(n, k // 16, 16)[1::97, 2::13, :] *= 40.0
    w.view(n, k // 16, 16)[2::7, 5::9, 3] = -0.0
    gs = (448.0 * 6.0 / w.float().abs().amax()).reshape(1).float()
    wq, wsf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return wq, wsf, gs


def dequant(qv, sf, gs, rows, k):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev,
                       dtype=torch.float32)
    lo = lut[(qv & 0xF).long()]
    hi = lut[(qv >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(rows, k)
    ncol = k // 16
    ntk = (ncol + 3) // 4
    r = torch.arange(rows, device=dev).view(-1, 1)
    c = torch.arange(ncol, device=dev).view(1, -1)
    idx = ((r // 128) * ntk + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    s = sf.view(-1)[idx].view(torch.float8_e4m3fn).float()
    return v * s.repeat_interleave(16, dim=1) / gs


def with_kill(fn):
    os.environ[KILL] = "0"
    try:
        return fn()
    finally:
        os.environ.pop(KILL, None)


def main():
    tree()
    t0 = time.time()
    GU = mkw(34816, 5120, 17)
    DN = mkw(5120, 17408, 29)
    guw = [GU[0]] + [GU[0].clone() for _ in range(L - 1)]
    gusf = [GU[1]] + [GU[1].clone() for _ in range(L - 1)]
    dnw = [DN[0]] + [DN[0].clone() for _ in range(L - 1)]
    dnsf = [DN[1]] + [DN[1].clone() for _ in range(L - 1)]
    wsnap = sha(guw + gusf + dnw + dnsf)
    log(f"weights ready ({time.time() - t0:.0f}s) wsha={wsnap}")
    report = {"kill_switch": {}, "replays": {}, "tune": {}, "hashes": {}, "ref": {}}
    ok = True

    def acts(m):
        A = dict(x=[], gin=[], alg=[], gmid=[], ald=[])
        for i in range(L):
            g = torch.Generator(device=dev)
            g.manual_seed(1000 * m + i)
            x = (torch.randn(m, 5120, device=dev, generator=g) * 2.0).to(torch.bfloat16)
            x.view(m, 320, 16)[:, 3::7, :] = 0
            gs = (448.0 * 6.0 / x.float().abs().amax()).reshape(1).float()
            alg = (1.0 / (gs * GU[2])).float()
            aq, asf = q4(x, gs)
            h = gemm(aq, guw[i], asf, gusf[i], alg)
            a = h[:, :17408].float()
            s = (a * torch.sigmoid(a) * h[:, 17408:].float()).to(torch.bfloat16).float()
            gm = (448.0 * 6.0 / s.abs().amax().clamp_min(1e-6)).reshape(1).float()
            ald = (1.0 / (gm * DN[2])).float()
            for kk, vv in (("x", x), ("gin", gs), ("alg", alg), ("gmid", gm), ("ald", ald)):
                A[kk].append(vv)
        torch.cuda.synchronize()
        return A

    def layer(A, i, keep=None):
        aq, asf = q4(A["x"][i], A["gin"][i])
        h = gemm(aq, guw[i], asf, gusf[i], A["alg"][i])
        dq, dsf = torch.ops.trtllm.fused_silu_mul_nvfp4(h, A["gmid"][i])
        o = gemm(dq, dnw[i], dsf, dnsf[i], A["ald"][i])
        if keep is not None:
            keep.append((aq, asf, h, dq, dsf))
        return h, o

    for m in (1, 2, 3, 4):
        A = acts(m)
        xs = sha(A["x"])
        keep = []
        eager = [layer(A, i, keep) for i in range(L)]
        eager_k = with_kill(lambda: [layer(A, i) for i in range(L)])
        torch.cuda.synchronize()
        eq_gu = all(torch.equal(a[0], b[0]) for a, b in zip(eager, eager_k))
        eq_dn = all(torch.equal(a[1], b[1]) for a, b in zip(eager, eager_k))
        # isolated op on identical inputs (down with the silu producer's output), default vs kill
        iso_dn = [gemm(kp[3], dnw[i], kp[4], dnsf[i], A["ald"][i]) for i, kp in enumerate(keep)]
        iso_dn_k = with_kill(lambda: [gemm(kp[3], dnw[i], kp[4], dnsf[i], A["ald"][i]) for i, kp in enumerate(keep)])
        iso_gu = [gemm(kp[0], guw[i], kp[1], gusf[i], A["alg"][i]) for i, kp in enumerate(keep)]
        iso_gu_k = with_kill(lambda: [gemm(kp[0], guw[i], kp[1], gusf[i], A["alg"][i]) for i, kp in enumerate(keep)])
        eq_iso = (all(torch.equal(a, b) for a, b in zip(iso_dn, iso_dn_k))
                  and all(torch.equal(a, b) for a, b in zip(iso_gu, iso_gu_k))
                  and all(torch.equal(a, e[1]) for a, e in zip(iso_dn, eager))
                  and all(torch.equal(a, e[0]) for a, e in zip(iso_gu, eager)))
        finite = all(bool(torch.isfinite(e[1].float()).all()) and bool(torch.isfinite(e[0].float()).all())
                     for e in eager)
        rec = {"gate_up_eq_kill": eq_gu, "down_eq_kill": eq_dn, "isolated_eq": eq_iso, "finite": finite}
        ok = ok and eq_gu and eq_dn and eq_iso and finite
        report["kill_switch"][f"M{m}"] = rec
        report["hashes"][f"M{m}"] = sha([e[1] for e in eager])
        report["hashes"][f"M{m}_gate_up"] = sha([e[0] for e in eager])
        # FP32 reference for layer 0 (advisory; identically quantized operands)
        kp = keep[0]
        ref_gu = dequant(kp[0], kp[1], A["gin"][0], m, 5120) @ dequant(GU[0], GU[1], GU[2], 34816, 5120).t()
        ref_dn = dequant(kp[3], kp[4], A["gmid"][0], m, 17408) @ dequant(DN[0], DN[1], DN[2], 5120, 17408).t()
        report["ref"][f"M{m}"] = {
            "gate_up_max_abs": round((eager[0][0].float() - ref_gu).abs().max().item(), 5),
            "down_max_abs": round((eager[0][1].float() - ref_dn).abs().max().item(), 5),
            "down_out_mean_abs": round(eager[0][1].float().abs().mean().item(), 4)}
        del ref_gu, ref_dn
        log(f"[check] M={m} kill switch: {json.dumps(rec)} ref {json.dumps(report['ref'][f'M{m}'])} "
            f"({time.time() - t0:.0f}s)")
        if m in (1, 2):
            for arm in ("default", "kill"):
                def fn():
                    return [layer(A, i) for i in range(L)]

                def cap():
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        fn()
                    torch.cuda.current_stream().wait_stream(s)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        outs = fn()
                    torch.cuda.synchronize()
                    return g, outs
                g, outs = with_kill(cap) if arm == "kill" else cap()
                flat = [t for pair in outs for t in pair]
                ref_flat = [t for pair in eager for t in pair]
                first = None
                rep_ok, eq_eager, maxdev = True, True, 0.0
                for it in range(WARM + ITERS):
                    for t in flat:
                        t.fill_(float("nan"))
                    g.replay()
                    torch.cuda.synchronize()
                    cur = [t.clone() for t in flat]
                    if first is None:
                        first = cur
                    rep_ok = rep_ok and all(torch.equal(a, b) for a, b in zip(cur, first))
                    e_ok = all(torch.equal(a, b) for a, b in zip(cur, ref_flat))
                    eq_eager = eq_eager and e_ok
                    if not e_ok:
                        maxdev = max(maxdev, max((a.float() - b.float()).abs().max().item()
                                                 for a, b in zip(cur, ref_flat)))
                inputs_ok = sha(A["x"]) == xs and sha(guw + gusf + dnw + dnsf) == wsnap
                r2 = {"replays": WARM + ITERS, "nan_poisoned": True, "eq_replay1": rep_ok, "eq_eager": eq_eager,
                      "max_dev_vs_eager": maxdev, "inputs_unchanged": inputs_ok}
                ok = ok and rep_ok and eq_eager and inputs_ok
                report["replays"][f"M{m}/{arm}"] = r2
                log(f"[check] M={m} graph {arm}: {json.dumps(r2)} ({time.time() - t0:.0f}s)")
                del g, outs, flat
        del A, keep, eager, eager_k
        torch.cuda.empty_cache()

    # fresh cold-L2 tunes at bucket 1
    tuner = AutoTuner.get()
    for rep in range(2):
        for name, (W, n, k) in (("gate_up", (GU, 34816, 5120)), ("down", (DN, 5120, 17408))):
            tuner.clear_cache()
            g = torch.Generator(device=dev)
            g.manual_seed(70 + rep)
            x = (torch.randn(1, k, device=dev, generator=g) * 2.0).to(torch.bfloat16)
            gs = (448.0 * 6.0 / x.float().abs().amax()).reshape(1).float()
            aq, asf = q4(x, gs)
            alpha = (1.0 / (gs * W[2])).float()
            with autotune(skip_dynamic_tuning_buckets=True):
                torch.ops.trtllm.nvfp4_gemm(aq, W[0], asf, W[1], alpha, torch.bfloat16)
            torch.cuda.synchronize()
            r = NVFP4GemmUnifiedRunner(0, torch.bfloat16, "cutlass,cublaslt,cuda_core".split(","))
            inputs = [aq, W[0], asf, W[1], alpha]
            hit, rid, tac, tms = tuner.profiling_cache.search_cache(
                "trtllm::nvfp4_gemm::gemm", [r], tuple(tuner._get_input_sizes(inputs)),
                NVFP4GemmUnifiedRunner.tuning_config, apply_map_to_tuning_buckets=True)
            pick_cc = isinstance(tac, (tuple, list)) and len(tac) > 0 and tac[0] == "cuda_core"
            ok = ok and bool(hit) and pick_cc
            rec = {"hit": bool(hit), "tactic": str(tac), "tuner_ms": None if tms is None else float(tms),
                   "cuda_core": pick_cc}
            report["tune"][f"tune{rep}/{name}/M1"] = rec
            log(f"[check] tune {rep} {name} M=1: {json.dumps(rec)} ({time.time() - t0:.0f}s)")
    tuner.clear_cache()
    report["passed"] = bool(ok)
    report["compared_against"] = ("kill switch TRTLLM_CUDA_CORE_NVFP4_M1_PREWAIT=0 (the previous wait-first loop "
                                  "at M=1, bitwise main's kernel); eager chain for NaN-poisoned graph replays; "
                                  "hashes to compare with the base arm of the run_gpu_pair job")
    log(json.dumps(report))


main()
