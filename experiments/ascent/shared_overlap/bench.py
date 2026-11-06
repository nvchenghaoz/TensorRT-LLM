"""Branch 118 (idea 129): eager-prefill fork/join of the DSV4 shared expert.

One TP4 rank (rank 0) of DeepSeek-V4.1-Flash under the live MoE layout
(engine.json at main 2fa8e034 has no moe_* keys -> EP4: 96 of 384 routed
experts, inter 2304, top-6, hidden 5120). The REAL DeepseekV4MoE.forward of
whichever tree runs it drives:
  routed: torch.ops.trtllm.fused_moe W4A8 MXFP4 x MXFP8 (synthetic weights,
          fixed routing; the gate is not run), via mxfp8_fold/moe_fold_bench.Problem
          (MXFP8 quant fold on at M >= _MXFP8_FOLD_MIN_TOKENS, as the leaf does)
  shared: the checkpoint-configured shared GatedMLP (shared_tp 2, row split)
  all-reduce: identity.
BASE tree: serial routed -> shared on one stream. CAND tree: at M >= 1024 the
routed MoE on the calling stream and the shared rows on the MoeShared aux stream.

Modes:
  bench: (1) deterministic unfused finalize: per-M sha256 of forward's output
         (must match across arms -> bitwise equal); (2) production fused
         finalize: block time per M (eager, GPU-side events, host queued ahead
         behind a sleep kernel so host launch time is hidden as in prefill).
  check: candidate tree only, untimed, same warmup/iters/repeats as bench, with
         the deterministic finalize: every fork/join output bitwise equal to the
         serial path (same process, overlap disabled) and to an independent
         composition (routed + shared(rows) slice-added); counts which path ran
         at each M (M=512 must stay serial).
Usage: python bench.py --ckpt <ckpt> --mode bench|check [--ms 512,2048,6144]
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared_row_split"))
sys.path.insert(0, os.path.join(HERE, "..", "mxfp8_fold"))

import torch  # noqa: E402

import moe_fold_bench as mfb  # noqa: E402  (sets TRTLLM_ENABLE_PDL default, imports trtllm)
import common as srs  # noqa: E402

LAYOUT = "EP4"
WARMUP, ITERS, REPEATS = 5, 30, 5
SLEEP_CYCLES = 6_000_000  # ~2.5 ms at 2.4 GHz: lets the host enqueue the whole forward


def log(*a):
    print(*a, flush=True)


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


class Rig:
    def __init__(self, ckpt):
        os.environ.setdefault("CKPT", ckpt)
        srs.CKPT = ckpt
        from tensorrt_llm._torch.models import modeling_deepseekv4 as m4

        self.m4 = m4
        self.p = mfb.Problem(*mfb.read_dims(ckpt), ep=4)
        mlp, shared_tp, scale, info = srs.build_shared_mlp(0)
        moe, _stub, has_split = srs.build_moe(0, mlp, shared_tp, scale, top_k=self.p.topk)
        assert has_split and moe._shared_row_split_parts == 2, "row split must be on (rank 0)"
        moe.aux_stream = torch.cuda.Stream()
        self.moe, self.info = moe, info
        self.has_overlap = hasattr(m4.DeepseekV4MoE, "_run_routed_main_shared_aux")
        self.arm = "CAND" if self.has_overlap else "BASE"
        self.cur = None
        self.calls = {"overlap": 0}
        moe.compute_routed_output = self._routed
        if self.has_overlap:
            orig = moe._run_routed_main_shared_aux

            def counted(a, b):
                self.calls["overlap"] += 1
                return orig(a, b)

            moe._run_routed_main_shared_aux = counted

    def _routed(self, *a, **k):
        x, sel, sc = self.cur
        fold = mfb.FOLD_MIN <= x.shape[0] <= mfb.FOLD_MAX
        return self.p.run(x, sel, sc, fold)

    def set_m(self, m, seed=0):
        self.cur = self.p.inputs(m, seed=seed)
        return self.cur[0]

    def forward(self, hs):
        return self.moe.forward(hs)


def time_m(rig, hs):
    rep_meds, all_t = [], []
    for rep in range(REPEATS):
        ts = []
        for it in range(WARMUP + ITERS):
            torch.cuda.synchronize()
            torch.cuda._sleep(SLEEP_CYCLES)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = rig.forward(hs)
            e1.record()
            e1.synchronize()
            if it >= WARMUP:
                ts.append(e0.elapsed_time(e1))
        ts.sort()
        rep_meds.append(ts[len(ts) // 2])
        all_t += ts
        log(f"{rig.arm} layout={LAYOUT} M={hs.shape[0]} rep={rep} median_ms={rep_meds[-1]:.5f} "
            f"p10={ts[int(0.1 * (len(ts) - 1))]:.5f} p90={ts[int(0.9 * (len(ts) - 1))]:.5f}")
    all_t.sort()
    n = len(all_t)
    rep_meds.sort()
    med = rep_meds[len(rep_meds) // 2]
    return {"kernel_ms": med,
            "noise_floor": (all_t[int(0.9 * (n - 1))] - all_t[int(0.1 * (n - 1))]) / all_t[n // 2],
            "repeat_spread": (rep_meds[-1] - rep_meds[0]) / med, "repeat_medians": rep_meds,
            "sm_clock_mhz": srs.sm_clock_mhz(), "out_shape": list(out.shape)}


def warm_shared(rig, ms):
    for m in ms:
        hs = rig.set_m(m)
        for _ in range(2):
            rig.forward(hs)
    torch.cuda.synchronize()


def mode_bench(rig, ms):
    log(f"{rig.arm} layout={LAYOUT} info={json.dumps(rig.info)} fold_min={mfb.FOLD_MIN} "
        f"overlap_min={getattr(rig.m4, 'SHARED_OVERLAP_MIN_TOKENS', None)}")
    mfb.phase(rig.p, ms, False)
    warm_shared(rig, ms)
    shas = {}
    for m in ms:
        hs = rig.set_m(m)
        a = rig.forward(hs).clone()
        b = rig.forward(hs).clone()
        assert torch.equal(a, b), f"deterministic finalize: repeated forward differs at M={m}"
        assert torch.isfinite(a.float()).all()
        shas[str(m)] = sha(a)
        log(f"{rig.arm} layout={LAYOUT} M={m} det_sha={shas[str(m)]}")
    mfb.phase(rig.p, ms, True)
    warm_shared(rig, ms)
    res = {}
    for m in ms:
        hs = rig.set_m(m)
        before = rig.calls["overlap"]
        res[str(m)] = time_m(rig, hs)
        res[str(m)]["overlap_calls"] = rig.calls["overlap"] - before
        log(f"{rig.arm} layout={LAYOUT} M={m} RESULT kernel_ms={res[str(m)]['kernel_ms']:.5f} "
            f"noise={res[str(m)]['noise_floor']:.4f} spread={res[str(m)]['repeat_spread']:.4f} "
            f"overlap_calls={res[str(m)]['overlap_calls']} sm_clock={res[str(m)]['sm_clock_mhz']}")
        torch.cuda.empty_cache()
    print(json.dumps({"arm": rig.arm, "layout": LAYOUT, "batch": res,
                      "output": {"class": "exact", "sha256_16_unfused_finalize": shas}}))


def mode_check(rig, ms):
    if not rig.has_overlap:
        print(json.dumps({"passed": False, "error": "tree has no shared overlap"}))
        sys.exit(1)
    m4 = rig.m4
    mfb.phase(rig.p, ms, False)
    warm_shared(rig, ms)
    passed, results = True, {}
    thr = m4.SHARED_OVERLAP_MIN_TOKENS
    for m in ms:
        for seed in (0, 1):
            key = f"M{m}_seed{seed}"
            hs = rig.set_m(m, seed=seed)
            hs0 = hs.clone()
            # Serial reference in this process (overlap disabled), and an independent composition.
            m4.SHARED_OVERLAP_ENABLED = False
            before = rig.calls["overlap"]
            ref = rig.forward(hs).clone()
            serial_ok = rig.calls["overlap"] == before
            m4.SHARED_OVERLAP_ENABLED = True
            routed = rig._routed()
            s0, s1 = m4.shared_expert_row_range(m, rig.moe._shared_row_split_parts,
                                                rig.moe._shared_row_split_index)
            routed[s0:s1] += rig.moe.shared_experts(hs[s0:s1])
            indep_ok = torch.equal(routed, ref)
            st = {"iters": 0, "mismatch": 0, "max_abs": 0.0}
            before = rig.calls["overlap"]
            for rep in range(REPEATS):
                for it in range(WARMUP + ITERS):
                    torch.cuda._sleep(SLEEP_CYCLES)
                    out = rig.forward(hs)
                    st["iters"] += 1
                    if not torch.equal(out, ref):
                        st["mismatch"] += 1
                        st["max_abs"] = max(st["max_abs"], (out.float() - ref.float()).abs().max().item())
            n_overlap = rig.calls["overlap"] - before
            want_overlap = m >= thr
            path_ok = n_overlap == (st["iters"] if want_overlap else 0)
            inputs_ok = torch.equal(hs, hs0)
            finite = bool(torch.isfinite(ref.float()).all())
            nz = float((ref != 0).float().mean())
            ok = serial_ok and indep_ok and st["mismatch"] == 0 and path_ok and inputs_ok and finite and nz > 0.5
            passed &= ok
            results[key] = {"iters": st["iters"], "mismatch": st["mismatch"], "max_abs": st["max_abs"],
                            "overlap_calls": n_overlap, "expected_path": "fork_join" if want_overlap else "serial",
                            "path_ok": path_ok, "serial_ref_ok": serial_ok, "indep_compose_equal": indep_ok,
                            "inputs_unchanged": inputs_ok, "finite": finite, "nonzero_frac": nz,
                            "sha": sha(ref), "ok": ok}
            log(f"CAND layout={LAYOUT} {key}", json.dumps(results[key]))
            torch.cuda.empty_cache()
    print(json.dumps({"passed": passed, "layout": LAYOUT, "threshold": thr,
                      "compared": "DeepseekV4MoE.forward fork/join (routed main, shared rows on MoeShared aux) "
                                  "vs the serial path (same process, TRTLLM shared overlap disabled) and vs "
                                  "routed + shared_experts(rows) composed by hand; torch.equal on every one of "
                                  f"{REPEATS}x({WARMUP}+{ITERS}) iterations per case, deterministic unfused finalize",
                      "cases": results}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="bench", choices=["bench", "check"])
    ap.add_argument("--ms", default="512,2048,6144")
    args = ap.parse_args()
    ms = [int(v) for v in args.ms.split(",")]
    torch.manual_seed(0)
    rig = Rig(args.ckpt)
    log(f"arm={rig.arm} layout={LAYOUT} gpu={torch.cuda.get_device_name()} ms={ms}")
    if args.mode == "bench":
        mode_bench(rig, ms)
    else:
        mode_check(rig, ms)


if __name__ == "__main__":
    main()
