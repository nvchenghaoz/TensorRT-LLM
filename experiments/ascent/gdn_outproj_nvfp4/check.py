# Untimed correctness for the GDN out_proj decode NVFP4 copy (candidate tree).
# A = out_proj without the flag (main's path: gated norm -> FP8 -> cudaCoreGemmV2 /
#     cublas), B = flagged as GDN __init__ flags it (rows <= 16: BF16 norm -> Marlin).
# 1) decode rows 1/2/4/8/16: same C=15 rotated copies, warmup 10 + 60 replays as the
#    timed bench; after EVERY replay all 15 outputs must be bitwise equal to eager
#    call 1; per-iteration max error vs the FP32 references is recorded.
#    B is checked against (i) the FP32 reference of main's math (FP8 weight * scale)
#    and (ii) an independent FP32 reference from the same quantized operands
#    (dequantized NVFP4 weight copy recipe x the BF16 gated-norm output).
# 2) rows > 16 (17/32/256/2048/8192): B's gated-norm output and out_proj output
#    bitwise equal to A's (both FP8 path).
# 3) kill switches: TRTLLM_GDN_OUTPROJ_NVFP4_DECODE=0 and TRTLLM_GDN_INPROJ_NVFP4_DECODE=0
#    -> no copy, outputs bitwise equal to A at rows 1/4/16/2048.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import common as cm  # noqa: E402

C, WARM, ITERS = 15, 10, 60
DEC = [1, 2, 4, 8, 16]
PRE = [17, 32, 256, 2048, 8192]
log = cm.log
failures = []


def fail(msg):
    failures.append(msg)
    log("FAIL", msg)


def capture_out_proj_input(gm):
    store = {}

    def hook(mod, args, kwargs=None):
        store["x"] = args[0].detach().clone()
        return None

    h = gm.out_proj.register_forward_pre_hook(hook)
    return store, h


def nvfp4_dequant_reference(w8, s):
    """Independent FP32 dequant of the copy recipe (linear.py
    _build_fp8_nvfp4_decode_copy): E4M3 values as BF16, g = 2^ceil(log2(amax/2688)),
    fp4_quantize with global scale s/g, dequant = e2m1 * SF * g."""
    n, k = w8.shape
    w_vals = w8.to(torch.bfloat16)
    amax = torch.clamp(w_vals.abs().amax().float() * s, min=torch.finfo(torch.float32).tiny)
    g = torch.exp2(torch.ceil(torch.log2(amax / (448.0 * 6.0))))
    qg = (s / g).reshape(1).to(torch.float32)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w_vals, qg, 16, False, True)
    sf = torch.ops.trtllm.block_scale_interleave_reverse(w_sf.view(n, -1))[:, :k // 16]
    sf_f = sf.contiguous().view(torch.float8_e4m3fn).float()
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=w8.device)
    p = w_fp4.view(torch.uint8)
    vals = torch.stack([lut[(p & 0xF).long()], lut[(p >> 4).long()]], dim=-1).reshape(n, k)
    wq = (vals.view(n, k // 16, 16) * sf_f.unsqueeze(-1)).reshape(n, k) * g
    wref = w8.float() * s
    rel = ((wq - wref).norm() / wref.norm()).item()
    return wq, rel


def main():
    flag_default = cm.candidate_flag()
    log("candidate flag (env default):", flag_default)
    if not flag_default:
        fail("GDN out_proj flag is not on by default in this tree")
    A = [cm.make_gdn(1000 + i, False) for i in range(C)]
    B = [cm.make_gdn(1000 + i, True) for i in range(C)]
    if getattr(B[0][0].out_proj, "_nvfp4_decode_weight", None) is None:
        fail("B has no NVFP4 decode copy")
    if getattr(A[0][0].out_proj, "_nvfp4_decode_weight", None) is not None:
        fail("A unexpectedly has a copy")
    wq, w_rel = nvfp4_dequant_reference(B[0][1][0], B[0][1][1])
    log("NVFP4 copy weight rel-RMS vs FP8*scale: %.5f" % w_rel)
    report = {"weight_rel_rms_nvfp4_vs_fp8": w_rel, "decode": {}, "prefill": {}, "kill_switch": {}}

    # 1) decode rows <= 16: repeated execution with the timed bench's counts.
    for m in DEC:
        a, z, _ = cm.make_inputs(m)
        st, h = capture_out_proj_input(B[0][0])
        eB = [gm(a, z).clone() for gm, _ in B]
        h.remove()
        xb = st["x"]
        stA, hA = capture_out_proj_input(A[0][0])
        eA = [gm(a, z).clone() for gm, _ in A]
        hA.remove()
        xa = stA["x"]
        if xb.dtype != torch.bfloat16:
            fail(f"M={m}: B out_proj input dtype {xb.dtype}, expected bf16 (Marlin path)")
        if xa.dtype != torch.float8_e4m3fn:
            fail(f"M={m}: A out_proj input dtype {xa.dtype}, expected fp8")
        # eager call 2 == call 1
        e2 = [gm(a, z) for gm, _ in B]
        if not all(torch.equal(x, y) for x, y in zip(e2, eB)):
            fail(f"M={m}: eager call 2 != call 1")
        ref = cm.fp32_reference(a, z, B[0][1])
        ref_q = xb.float() @ wq.t()  # same quantized operands as Marlin
        # FP8 activation rounding consistency: A's FP8 input is B's BF16 rounded
        # by the static scale (same kernel math up to the final cast).
        s_in = A[0][0].norm.fp8_scale.float()
        xa_from_b = (xb.float() * torch.reciprocal(s_in)).to(torch.float8_e4m3fn)
        fp8_same = bool(torch.equal(xa_from_b.view(torch.uint8), xa.view(torch.uint8)))
        # graph capture + replays
        s0 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            for _ in range(3):
                o = [gm(a, z) for gm, _ in B]
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            o = [gm(a, z) for gm, _ in B]
        mism, per_it = 0, []
        for it in range(WARM + ITERS):
            for t in o:
                t.fill_(float("nan")) if it % 7 == 0 else None
            gr.replay()
            torch.cuda.synchronize()
            ok = all(torch.equal(x, y) for x, y in zip(o, eB))
            if not ok:
                mism += 1
            per_it.append((o[0].float() - ref).abs().max().item())
        del gr
        if mism:
            fail(f"M={m}: {mism} replays differ from eager call 1")
        eb, ea = cm.err(eB[0], ref), cm.err(eA[0], ref)
        eq = cm.err(eB[0], ref_q)
        bva = cm.err(eB[0], eA[0].float())
        if not eb["finite"]:
            fail(f"M={m}: non-finite B output")
        if eq["rel_rms"] > 0.01:
            fail(f"M={m}: Marlin vs same-operand FP32 reference rel_rms {eq['rel_rms']:.4f} > 0.01")
        report["decode"][str(m)] = {
            "out_proj_input_dtype_B": str(xb.dtype), "out_proj_input_dtype_A": str(xa.dtype),
            "replays": WARM + ITERS, "replay_mismatch_vs_call1": mism,
            "per_iter_max_abs_vs_fp32_ref": {"min": min(per_it), "max": max(per_it)},
            "B_vs_fp32_fp8weight_ref": eb, "A_main_vs_fp32_fp8weight_ref": ea,
            "B_vs_fp32_same_nvfp4_operands_ref": eq, "B_vs_A_main": bva,
            "A_fp8_input_equals_fp8_of_B_bf16_norm": fp8_same,
        }
        log(f"M={m}: replays ok={mism == 0} B-vs-ref {eb['rel_rms']:.4f} A-vs-ref {ea['rel_rms']:.4f} "
            f"B-vs-sameoperand {eq['rel_rms']:.5f} B-vs-A {bva['rel_rms']:.4f} fp8_same={fp8_same}")

    # 2) rows > 16: bitwise main (both norm output and out_proj output).
    for m in PRE:
        a, z, _ = cm.make_inputs(m)
        stB, hB = capture_out_proj_input(B[0][0])
        ob = B[0][0](a, z)
        hB.remove()
        stA, hA = capture_out_proj_input(A[0][0])
        oa = A[0][0](a, z)
        hA.remove()
        norm_eq = (stB["x"].dtype == stA["x"].dtype == torch.float8_e4m3fn
                   and torch.equal(stB["x"].view(torch.uint8), stA["x"].view(torch.uint8)))
        out_eq = torch.equal(ob, oa)
        report["prefill"][str(m)] = {"norm_out_dtype": str(stB["x"].dtype),
                                     "norm_bitwise": bool(norm_eq), "out_proj_bitwise": bool(out_eq)}
        log(f"prefill M={m}: norm bitwise {norm_eq} out_proj bitwise {out_eq}")
        if not (norm_eq and out_eq):
            fail(f"prefill M={m}: not bitwise equal to main")

    # 3) kill switches
    for env in ("TRTLLM_GDN_OUTPROJ_NVFP4_DECODE", "TRTLLM_GDN_INPROJ_NVFP4_DECODE"):
        os.environ[env] = "0"
        try:
            flag = cm.candidate_flag()
            K = cm.make_gdn(1000, flag)[0]
            has = getattr(K.out_proj, "_nvfp4_decode_weight", None) is not None
            res = {"flag": flag, "copy_built": has}
            for m in (1, 4, 16, 2048):
                a, z, _ = cm.make_inputs(m)
                eq = bool(torch.equal(K(a, z), A[0][0](a, z)))
                res[str(m)] = eq
                if not eq:
                    fail(f"kill switch {env}=0: M={m} differs from main")
            if has:
                fail(f"kill switch {env}=0 still built a copy")
            report["kill_switch"][env] = res
            log(f"kill switch {env}=0: {res}")
            del K
        finally:
            del os.environ[env]

    report["passed"] = not failures
    report["failures"] = failures
    report["compared_against"] = ("main's path (same tree, out_proj unflagged) bitwise for rows>16 and "
                                  "kill switches; eager call 1 bitwise for every replay; FP32 references "
                                  "(FP8 weight*scale and same-operand NVFP4 dequant) for rows<=16")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
