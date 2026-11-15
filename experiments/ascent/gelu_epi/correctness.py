"""Untimed correctness sequence for the fused gate_up + gelu + NVFP4-quant chain (no model load).

Mirrors bench.py's timing structure, with the tactic PINNED (no tuner):
  M order MS (ascending, as bench.py), per M 5 warmup + 30 calls, call i uses weight copy i % 3
  (3 rotating gate_up weight copies, as bench.py). Then an A,B,A revisit: pass A = MS, pass B =
  MS reversed, pass A' = MS again, each pass 35 calls per M, no reset anywhere.
  MS = every targeted M (1000, 1200, 2000, 4000, 4800, 8000, 8100, 8192) plus the full-tile
  M 512, 1024, 4096 (no SF padding rows).
  reference (a), main's chain: FP4GemmRunner tactic ('cutlass', 2) on the ORIGINAL rows ->
    gelu_tanh_mul_fp4_quant, once per (M, copy).
  candidate: nvfp4_gelu_gemm_fp4 (fused kernel, tactic-2 mainloop) on the permuted rows, every
    call into fresh 0xA5-prefilled payload + SF buffers.
  Every call's payload and FULL SF buffer must be byte-equal to that (M, copy)'s call 1 and to
  reference (a). Inputs (activation FP4/SF/alpha, all weight copies original + permuted) must be
  bitwise unchanged after the whole sequence. The fused module entry is wrapped with a counter:
  the fused dispatch count per M is reported and must equal the number of candidate calls; a
  separate post-sequence probe records the CUDA kernel names one fused call launches per M.
  reference (b): FP32 dequantized GEMM -> gelu_tanh*up in FP32 at M=1000 (copy 0); the dequantized
  FP4 output's max-abs / rel-Frobenius error vs (b) is reported for main and candidate.
Data sets: randn seed 0, heavy-tailed randn*2^U(-8,8) per row.
usage: correctness.py <checkpoint>
"""
import json
import os
import sys
import threading
import time

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _hb():
    while True:
        time.sleep(45)
        log("heartbeat")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import FLUSH, ITERS, WARMUP  # noqa: E402  (the timed run's counts, not re-declared)

MS = [512, 1000, 1024, 1200, 2000, 4000, 4096, 4800, 8000, 8100, 8192]
PASSES = [("A", MS), ("B", MS[::-1]), ("A2", MS)]
TACTIC = ("cutlass", 2)
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def unswizzle_sf(sf, rows, nvec, torch):
    nkt = (nvec + 3) // 4
    mm = torch.arange(rows, device=sf.device)[:, None]
    kk = torch.arange(nvec, device=sf.device)[None, :]
    off = (mm // 128) * (nkt * 512) + (kk // 4) * 512 + (mm % 32) * 16 + ((mm % 128) // 32) * 4 + (kk % 4)
    return sf.view(torch.uint8)[off.reshape(-1)].reshape(rows, nvec)


def dequant(fp4, sf, gs, rows, cols, torch):
    lut = torch.tensor(E2M1, device=fp4.device)
    b = fp4.view(torch.uint8)[:rows].reshape(rows, cols // 2).long()
    vals = torch.stack([lut[b & 15], lut[b >> 4]], dim=-1).reshape(rows, cols)
    s = unswizzle_sf(sf, rows, cols // 16, torch).view(torch.float8_e4m3fn).float()
    return vals * s.repeat_interleave(16, dim=1) / gs


class CountingModule:
    """Wraps the fused JIT module; counts nvfp4_gelu_gemm launches per current label."""

    def __init__(self, mod):
        self._mod = mod
        self.label = None
        self.counts = {}

    def nvfp4_gelu_gemm(self, *args):
        self.counts[self.label] = self.counts.get(self.label, 0) + 1
        return self._mod.nvfp4_gelu_gemm(*args)

    def __getattr__(self, name):
        return getattr(self._mod, name)


def main():
    threading.Thread(target=_hb, daemon=True).start()
    ck = sys.argv[1]
    import torch
    cfg = json.load(open(os.path.join(ck, "config.json")))
    tc = cfg.get("text_config", cfg)
    hidden, inter = tc["hidden_size"], tc["intermediate_size"]
    N, K = 2 * inter, hidden
    log(f"config hidden={hidden} intermediate={inter} -> gate_up N={N} K={K}")
    print(f"[protocol] bench WARMUP={WARMUP} ITERS={ITERS} flush={FLUSH} | correctness calls/key/pass="
          f"{WARMUP + ITERS} passes={[p[0] for p in PASSES]} flush={FLUSH} order(A)={MS}", flush=True)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    from tensorrt_llm._torch.kernels.fused_ops import nvfp4_gelu_gemm_sm120 as ge
    from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant
    assert ge.gelu_gemm_epi_supported(), "kill switch set or not SM120"
    mod = ge.get_gelu_gemm_module()
    assert mod is not None, ge._module_error
    counter = CountingModule(mod)
    ge._module = counter  # nvfp4_gelu_gemm_fp4 -> get_gelu_gemm_module() -> counter
    dev = torch.device("cuda")
    runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, ["cutlass"])
    report = {"datasets": {}, "M_order": MS, "passes": [p[0] for p in PASSES], "tactic": list(TACTIC),
              "calls_per_M_per_pass": WARMUP + ITERS}
    all_ok = True
    for ds in ("seed0", "heavy"):
        counter.counts = {}
        g = torch.Generator(device=dev)
        g.manual_seed({"seed0": 11, "heavy": 12}[ds])
        copies = []
        for c in range(3):
            w = torch.randn(N, K, device=dev, generator=g)
            if ds == "heavy":
                w = w * torch.exp2(torch.empty(N, 1, device=dev).uniform_(-8, 8, generator=g))
            w = w.to(torch.bfloat16)
            gs_w = (448 * 6 / w.float().abs().max()).reshape(1).float()
            w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs_w, 16, False)
            del w
            wp, wsp = ge.permute_gate_up_rows(w_fp4, w_sf, K)
            copies.append((w_fp4, w_sf, wp, wsp, gs_w))
        torch.cuda.empty_cache()
        acts = {}
        for M in MS:
            a = torch.randn(M, K, device=dev, generator=g)
            if ds == "heavy":
                a = a * torch.exp2(torch.empty(M, 1, device=dev).uniform_(-8, 8, generator=g))
            a = a.to(torch.bfloat16)
            gs_a = (448 * 6 / a.float().abs().max()).reshape(1).float()
            a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a, gs_a, 16, False)
            acts[M] = (a_fp4, a_sf, [(1.0 / (gs_a * c[4])).float() for c in copies], gs_a)
        # down_proj input global scale from the true act amax at M=8000, copy 0
        a_fp4, a_sf, alphas, _ = acts[8000]
        y = runner(inputs=[a_fp4, copies[0][0], a_sf, copies[0][1], alphas[0]], tactic=TACTIC)
        amax = (torch.nn.functional.gelu(y[:, :inter].float(), approximate="tanh") *
                y[:, inter:].float()).abs().max().clamp_min(1e-30)
        gs_d = (448 * 6 / amax).reshape(1).float()
        del y
        snap = {M: (v[0].clone(), v[1].clone(), [x.clone() for x in v[2]]) for M, v in acts.items()}
        wsnap = [tuple(x.clone() for x in c[:4]) for c in copies]

        # reference (a): main's chain pinned to tactic 2, once per (M, copy)
        refs = {}
        for M in MS:
            a_fp4, a_sf, alphas, _ = acts[M]
            for ci, c in enumerate(copies):
                y = runner(inputs=[a_fp4, c[0], a_sf, c[1], alphas[ci]], tactic=TACTIC)
                refs[(M, ci)] = gelu_tanh_mul_fp4_quant(y, gs_d)
                del y
        torch.cuda.synchronize()
        log(f"{ds}: references built ({len(refs)})")

        def cand(M, i):
            a_fp4, a_sf, alphas, _ = acts[M]
            c = copies[i % 3]
            ref_sf = refs[(M, i % 3)][1]
            out = torch.full((M, inter // 2), 0xA5, dtype=torch.uint8, device=dev)
            sf = torch.full_like(ref_sf, 0xA5)
            return ge.nvfp4_gelu_gemm_fp4(a_fp4, a_sf, c[2], c[3], alphas[i % 3], gs_d, out=out, sf=sf)

        firsts, rows = {}, {}
        for pname, order in PASSES:
            for M in order:
                counter.label = M
                st = rows.setdefault(str(M), {"calls": 0, "bad_vs_first": 0, "bad_vs_ref": 0,
                                              "first_mismatch": None, "per_pass_bad": {}})
                bad_here = 0
                for i in range(WARMUP + ITERS):
                    key = (M, i % 3)
                    fp4, sf = cand(M, i)
                    if key not in firsts:
                        firsts[key] = (fp4.clone(), sf.clone())
                    st["calls"] += 1
                    r = refs[key]
                    eq_first = torch.equal(fp4, firsts[key][0]) and torch.equal(sf, firsts[key][1])
                    eq_ref = (fp4.shape == r[0].shape and sf.shape == r[1].shape and
                              torch.equal(fp4, r[0]) and torch.equal(sf, r[1]))
                    st["bad_vs_first"] += int(not eq_first)
                    st["bad_vs_ref"] += int(not eq_ref)
                    bad_here += int(not (eq_first and eq_ref))
                    if not eq_ref and st["first_mismatch"] is None:
                        for name, x, rr in (("fp4", fp4, r[0]), ("sf", sf, r[1])):
                            if x.shape != rr.shape:
                                st["first_mismatch"] = {"buf": name, "shape": [list(x.shape), list(rr.shape)]}
                                break
                            ne = (x.reshape(-1) != rr.reshape(-1)).nonzero()
                            if ne.numel():
                                j = int(ne[0])
                                st["first_mismatch"] = {"pass": pname, "call": i, "copy": i % 3, "buf": name,
                                                        "byte": j, "cand": int(x.reshape(-1)[j]),
                                                        "ref": int(rr.reshape(-1)[j]), "count": int(ne.numel())}
                                break
                st["per_pass_bad"][pname] = bad_here
            torch.cuda.synchronize()
            log(f"{ds}: pass {pname} done, bad={[(k, v['per_pass_bad'][pname]) for k, v in rows.items() if v['per_pass_bad'][pname]]}")
        counter.label = None
        inputs_ok = all(torch.equal(acts[M][0], snap[M][0]) and torch.equal(acts[M][1], snap[M][1]) and
                        all(torch.equal(x, y) for x, y in zip(acts[M][2], snap[M][2])) for M in MS)
        inputs_ok = inputs_ok and all(torch.equal(x, y) for c, s in zip(copies, wsnap) for x, y in zip(c[:4], s))
        dispatch = {str(M): counter.counts.get(M, 0) for M in MS}
        dispatch_ok = all(dispatch[str(M)] == rows[str(M)]["calls"] == len(PASSES) * (WARMUP + ITERS) for M in MS)

        # kernel-name probe (outside the counted sequence): what one fused call launches
        kernels = {}
        if ds == "seed0":
            from torch.profiler import ProfilerActivity, profile
            for M in MS:
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    cand(M, 0)
                    torch.cuda.synchronize()
                names = sorted({e.name for e in prof.events()
                                if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type)
                                and "fill" not in e.name.lower() and "Memcpy" not in e.name
                                and "Memset" not in e.name})
                kernels[str(M)] = [n[:160] for n in names]
            log(f"{ds}: fused-call kernels {kernels}")

        # reference (b): FP32 dequantized GEMM -> gelu_tanh * up in FP32, at M=1000, copy 0
        errs = {}
        M = 1000
        a_fp4, a_sf, alphas, gs_a = acts[M]
        A = dequant(a_fp4, a_sf, gs_a, M, K, torch)
        W = dequant(copies[0][0], copies[0][1], copies[0][4], N, K, torch)
        Y = A @ W.t()
        ref_b = torch.nn.functional.gelu(Y[:, :inter], approximate="tanh") * Y[:, inter:]
        del A, W, Y
        for name, (fp4, sf) in (("main", refs[(M, 0)]), ("candidate", firsts[(M, 0)])):
            d = dequant(fp4, sf, gs_d, M, inter, torch)
            errs[name] = {"max_abs": float((d - ref_b).abs().max()),
                          "rel_fro": float((d - ref_b).norm() / ref_b.norm())}
        finite = all(bool(torch.isfinite(torch.tensor(v["max_abs"]))) for v in errs.values())
        ok = inputs_ok and dispatch_ok and finite and all(
            r["bad_vs_first"] == 0 and r["bad_vs_ref"] == 0 for r in rows.values())
        ok = ok and errs["candidate"]["max_abs"] <= 1.10 * errs["main"]["max_abs"] + 1e-6
        all_ok &= ok
        report["datasets"][ds] = {"ok": ok, "inputs_unchanged": inputs_ok, "fused_dispatch_per_M": dispatch,
                                  "dispatch_ok": dispatch_ok, "rows": rows, "fused_call_kernels": kernels,
                                  "ref_b_err_M1000": errs}
        log(f"{ds}: ok={ok} inputs_unchanged={inputs_ok} dispatch={dispatch} errs={errs}")
        del copies, acts, snap, wsnap, refs, firsts
        torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(all_ok),
                      "compared": "fused nvfp4_gelu_gemm_fp4 (permuted rows) vs main's chain pinned to "
                                  "('cutlass', 2) -> gelu_tanh_mul_fp4_quant; payload + full SF bytes, every "
                                  "call vs its (M, copy) call 1 and vs main; bench.py structure (M order, "
                                  "5+30 calls per M, 3 rotating weight copies) then A,B,A revisit; fresh "
                                  "0xA5-prefilled outputs; inputs unchanged; fused dispatch count per M; "
                                  "FP32 dequant reference error",
                      **report}), flush=True)


if __name__ == "__main__":
    main()
