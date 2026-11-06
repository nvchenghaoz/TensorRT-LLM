"""Branch 118 (idea 129): eager-prefill fork/join of the DSV4 shared expert.

One TP4 rank (rank 0) of DeepSeek-V4.1-Flash under the live MoE layout
(engine.json at main 8e291e60: moe_tensor_parallel_size 2, moe_expert_parallel_size 2
-> EP2xTP2: 192 of 384 routed experts x inter 1152, top-6, hidden 5120). The REAL
DeepseekV4MoE.forward of whichever tree runs it drives:
  routed: torch.ops.trtllm.fused_moe W4A8 MXFP4 x MXFP8 (synthetic weights,
          fixed uniform routing; the gate is not run), tp_size=2 ep_size=2 rank 0
          (class Routed, copied from b100_moe_prefill/bench.py Rank), MXFP8 quant
          fold on at M >= _MXFP8_FOLD_MIN_TOKENS, as the leaf does
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

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402

import common as srs  # noqa: E402

LAYOUT = "EP2xTP2"
WARMUP, ITERS, REPEATS = 5, 30, 5
ALIGN = 128
try:
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8 as _leaf

    FOLD_MIN = getattr(_leaf, "_MXFP8_FOLD_MIN_TOKENS", 1024)
    FOLD_MAX = getattr(_leaf, "_MXFP8_FOLD_MAX_TOKENS", -1)
except Exception:  # pragma: no cover
    FOLD_MIN, FOLD_MAX = 1024, -1
if FOLD_MAX < 0:
    FOLD_MAX = 1 << 30


def read_dims(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    if "text_config" in cfg:
        cfg = {**cfg, **cfg["text_config"]}
    return cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"], cfg["num_experts_per_tok"]


class Routed:
    """EP2xTP2 rank 0 (moe_ep_rank 0, moe_tp_rank 0) of the routed W4A8 MXFP4 x MXFP8 fused_moe, as
    experiments/ascent/b100_moe_prefill/bench.py Rank (192 local experts x inter 1152, tp_size=2,
    ep_size=2, uniform top-k over all experts, production MXFP8 quant fold at FOLD_MIN <= M), plus a
    finalize-mode flag (ff) switched only through phase()."""

    def __init__(self, hidden, inter, n_exp, topk, tp=2, ep=2):
        self.hidden, self.n_exp, self.topk, self.tp, self.ep = hidden, n_exp, topk, tp, ep
        E, H, I = n_exp // ep, hidden, inter // tp
        assert I % ALIGN == 0
        self.e_local, self.inter_local = E, I
        g = torch.Generator(device="cuda").manual_seed(1234)
        dev = "cuda"
        self.w31 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev,
                                 generator=g).view(torch.int64)
        self.w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev,
                                generator=g).view(torch.int64)
        sf31 = torch.randint(117, 125, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
        sf2 = torch.randint(117, 125, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g)
        ones = torch.ones(E, dtype=torch.float32, device=dev)
        self.quant_scales = [sf31.view(torch.int32).contiguous(), ones, sf2.view(torch.int32).contiguous(),
                             ones.clone()]
        self.ff = True

    def inputs(self, m, seed=0):
        g = torch.Generator(device="cuda").manual_seed(99 + 7 * m + 1000 * seed)
        x = torch.randn(m, self.hidden, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
        gc = torch.Generator().manual_seed(5 + 13 * m + 1000 * seed)
        sel = torch.stack([torch.randperm(self.n_exp, generator=gc)[: self.topk] for _ in range(m)]).to(torch.int32)
        scales = torch.softmax(torch.randn(m, self.topk, generator=gc), dim=-1).float()
        return x.contiguous(), sel.cuda(), scales.cuda()

    def run(self, x, sel, scales):
        if FOLD_MIN <= x.shape[0] <= FOLD_MAX:  # production: BF16 in, MXFP8 quant folded into expandInputRows
            xin, xsf = x, None
        else:
            xin, xsf = torch.ops.trtllm.mxfp8_quantize(x, True, alignment=ALIGN)
        return torch.ops.trtllm.fused_moe(
            xin, sel, scales, self.w31, None, self.w2, None, torch.bfloat16,
            quant_scales=self.quant_scales, input_sf=xsf, swizzled_input_sf=True,
            tp_size=self.tp, tp_rank=0, ep_size=self.ep, ep_rank=0,
            use_mxfp8_act_scaling=True, use_fused_finalize=self.ff,
        )[0]


def phase(p, ms, ff, tune):
    """New finalize-mode phase: MoERunner.runner_dict's key omits use_fused_finalize (pitfall 1332), so clear
    it (and the tuning cache). tune=False keeps the default tactic, identical in both arms' processes, so the
    deterministic-finalize sha compares across arms independent of autotuner picks."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    torch.cuda.synchronize()
    MoERunner.runner_dict.clear()
    AutoTuner.get().profiling_cache.cache.clear()
    p.ff = ff
    log(f"layout={LAYOUT} phase fused_finalize={ff} tune={tune}")
    if tune:
        x, sel, sc = p.inputs(max(ms))
        with autotune():
            p.run(x, sel, sc)
        torch.cuda.synchronize()
    picks = []
    for k, v in AutoTuner.get().profiling_cache.cache.items():
        if str(k[0]).startswith("trtllm::fused_moe"):
            try:
                picks.append((str(k[0]).rsplit(":", 1)[-1], str(k[3][0][0]), str(v[1])))
            except Exception:  # noqa: BLE001
                picks.append((str(k)[:160], "?", str(v)[:80]))
    picks.sort()
    log(f"layout={LAYOUT} tuned tactics (gemm, M bucket, tactic): {picks}")


SLEEP_CYCLES = 6_000_000  # ~2.5 ms at 2.4 GHz: lets the host enqueue the whole forward


ARM = "?"


def log(*a):
    s = " ".join(str(x) for x in a)
    if not s.startswith("ARM="):
        s = f"ARM={ARM} layout={LAYOUT} " + s
    print(s, flush=True)


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


class Rig:
    def __init__(self, ckpt):
        os.environ.setdefault("CKPT", ckpt)
        srs.CKPT = ckpt
        from tensorrt_llm._torch.models import modeling_deepseekv4 as m4

        self.m4 = m4
        self.p = Routed(*read_dims(ckpt), tp=2, ep=2)
        mlp, shared_tp, scale, info = srs.build_shared_mlp(0)
        moe, _stub, has_split = srs.build_moe(0, mlp, shared_tp, scale, top_k=self.p.topk)
        assert has_split and moe._shared_row_split_parts == 2, "row split must be on (rank 0)"
        moe.aux_stream = torch.cuda.Stream()
        self.moe, self.info = moe, info
        self.has_overlap = hasattr(m4.DeepseekV4MoE, "_run_routed_main_shared_aux")
        self.arm = "CAND" if self.has_overlap else "BASE"
        global ARM
        ARM = self.arm
        if not hasattr(moe, "_shared_decode_split"):
            moe._shared_decode_split = False  # decode split not set up here (bench covers M >= 256 eager)
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
        return self.p.run(x, sel, sc)

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
        log(f"ARM={rig.arm} layout={LAYOUT} M={hs.shape[0]} rep={rep} median_ms={rep_meds[-1]:.5f} "
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
    log(f"ARM={rig.arm} layout={LAYOUT} info={json.dumps(rig.info)} fold_min={FOLD_MIN} "
        f"local_experts={rig.p.e_local} inter_local={rig.p.inter_local} "
        f"overlap_min={getattr(rig.m4, 'SHARED_OVERLAP_MIN_TOKENS', None)}")
    phase(rig.p, ms, False, tune=False)
    warm_shared(rig, ms)
    shas = {}
    for m in ms:
        hs = rig.set_m(m)
        a = rig.forward(hs).clone()
        b = rig.forward(hs).clone()
        assert torch.equal(a, b), f"deterministic finalize: repeated forward differs at M={m}"
        assert torch.isfinite(a.float()).all()
        shas[str(m)] = sha(a)
        log(f"ARM={rig.arm} layout={LAYOUT} M={m} det_sha={shas[str(m)]}")
    phase(rig.p, ms, True, tune=True)
    warm_shared(rig, ms)
    res = {}
    for m in ms:
        hs = rig.set_m(m)
        before = rig.calls["overlap"]
        res[str(m)] = time_m(rig, hs)
        res[str(m)]["overlap_calls"] = rig.calls["overlap"] - before
        log(f"ARM={rig.arm} layout={LAYOUT} M={m} RESULT kernel_ms={res[str(m)]['kernel_ms']:.5f} "
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
    from tensorrt_llm.logger import logger as trt_logger
    trt_logger.set_level("info")  # the overlap's info_once receipt (names the threshold T)
    phase(rig.p, ms, False, tune=False)
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
            log(f"ARM=CAND layout={LAYOUT} {key}", json.dumps(results[key]))
            torch.cuda.empty_cache()
    # CUDA-graph capture (decode graphs, and any captured window): the row split and so the overlap are
    # unreachable while capturing; the captured graph replays bitwise equal to the same tree's eager path.
    from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
    graph = {}
    for m in (4, 2048):
        hs = rig.set_m(m)
        before = rig.calls["overlap"]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                rig.forward(hs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        before_cap = rig.calls["overlap"]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gout = rig.forward(hs)
        cap_calls = rig.calls["overlap"] - before_cap
        outs_ok, n_rep = True, 10
        ref_cap = None
        for _ in range(n_rep):
            g.replay()
            torch.cuda.synchronize()
            if ref_cap is None:
                ref_cap = gout.clone()
            outs_ok &= torch.equal(gout, ref_cap)
        if m < thr:
            eager = rig.forward(hs)
            eager_eq = torch.equal(eager, ref_cap)
        else:
            eager_eq = None  # eager at M>=256 takes the row split (different, reordered sum) by design
        ok = cap_calls == 0 and outs_ok and (eager_eq is not False) and bool(torch.isfinite(ref_cap.float()).all())
        passed &= ok
        graph[f"M{m}"] = {"capture_overlap_calls": cap_calls, "warm_overlap_calls_side_stream": before_cap - before,
                          "replays": n_rep, "replays_equal": outs_ok, "replay_eq_eager": eager_eq, "ok": ok}
        log(f"ARM=CAND layout={LAYOUT} graph_M{m}", json.dumps(graph[f"M{m}"]))
        del g, gout
        torch.cuda.empty_cache()
    dms = bool(do_multi_stream())
    passed &= not dms
    print(json.dumps({"passed": passed, "layout": LAYOUT, "threshold": thr, "do_multi_stream_after": dms,
                      "graph_capture": graph,
                      "compared": "DeepseekV4MoE.forward fork/join (routed main, shared rows on MoeShared aux) "
                                  "vs the serial path (same process, TRTLLM shared overlap disabled) and vs "
                                  "routed + shared_experts(rows) composed by hand; torch.equal on every one of "
                                  f"{REPEATS}x({WARMUP}+{ITERS}) iterations per case, deterministic unfused finalize",
                      "cases": results}))


def slot_guard(limit_gib=4.0, wait_s=180, poll_s=10):
    """Top-lead rule (pitfall 3067/3072): wait-only guard. If another process still holds > limit_gib on a
    visible GPU after wait_s, print DIRTY SLOT and exit 3 before allocating. Never kills anything."""
    import time
    t0 = time.time()
    while True:
        used = []
        for d in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(d)
            used.append((total - free) / 2**30)
        print(f"slot_guard used_gib={[round(u, 2) for u in used]} t={time.time() - t0:.0f}s", flush=True)
        if all(u <= limit_gib for u in used):
            return
        if time.time() - t0 >= wait_s:
            print(f"DIRTY SLOT {[round(u, 2) for u in used]}", flush=True)
            print(json.dumps({"passed": False, "error": "DIRTY SLOT", "used_gib": used}))
            sys.exit(3)
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="bench", choices=["bench", "check"])
    ap.add_argument("--ms", default="512,2048,6144")
    args = ap.parse_args()
    ms = [int(v) for v in args.ms.split(",")]
    slot_guard()
    torch.manual_seed(0)
    rig = Rig(args.ckpt)
    log(f"ARM={rig.arm} layout={LAYOUT} gpu={torch.cuda.get_device_name()} ms={ms}")
    if args.mode == "bench":
        mode_bench(rig, ms)
    else:
        mode_check(rig, ms)


if __name__ == "__main__":
    main()
