"""Routed-MoE MXFP8 activation-quant fold: microbench + bitwise check (idea 125).

One EP4 rank of DeepSeek-V4.1-Flash: hidden, moe_intermediate_size, top-k and
n_routed_experts / 4 local experts read from the checkpoint config.json.
Synthetic MXFP4 weights / UE8M0 scales; no model load.

Paths:
  standalone: torch.ops.trtllm.mxfp8_quantize(x, swizzled) + fused_moe(fp8 x, input_sf)
              (what main runs at every M)
  fold:       fused_moe(bf16 x, input_sf=None): MXFP8 quant inside expandInputRows
              (candidate, M <= TRTLLM_MOE_MXFP8_QUANT_FOLD_MAX_TOKENS)

Modes:
  bench  : per M, CUDA-graph timing of the path this tree runs in production
           (base tree: standalone everywhere; candidate: fold where gated on).
           Prints evidence JSON with a sha256 of every M's output (must match
           across arms -> bitwise equal to main).
  check  : candidate tree only. Untimed repeated sequence (same warmup / iters /
           graph replays as bench) comparing fold vs standalone bitwise at every
           iteration, plus skewed routing and per-row power-of-two scaling.
  diag   : candidate tree only. Times fold vs standalone at every M (gate study).
"""

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402

try:
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8 as _leaf

    FOLD_MAX = getattr(_leaf, "_MXFP8_FOLD_MAX_TOKENS", 0)
except Exception:  # pragma: no cover
    FOLD_MAX = 0
if FOLD_MAX < 0:
    FOLD_MAX = 1 << 30  # fold at every M

ALIGN = 128


def log(*a):
    print(*a, flush=True)


def read_dims(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    if "text_config" in cfg:
        cfg = {**cfg, **cfg["text_config"]}
    hidden = cfg["hidden_size"]
    inter = cfg["moe_intermediate_size"]
    n_exp = cfg["n_routed_experts"]
    topk = cfg["num_experts_per_tok"]
    return hidden, inter, n_exp, topk


class Problem:
    def __init__(self, hidden, inter, n_exp, topk, ep=4, seed=0):
        self.hidden, self.inter, self.n_exp, self.topk, self.ep = hidden, inter, n_exp, topk, ep
        self.e_local = n_exp // ep
        g = torch.Generator(device="cuda").manual_seed(1234 + seed)
        dev = "cuda"
        E, H, I = self.e_local, hidden, inter
        self.w31 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev,
                                 generator=g).view(torch.int64)
        self.w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev,
                                generator=g).view(torch.int64)
        # UE8M0 weight scales over 8 exponents (2^-10 .. 2^-3), int32-packed.
        sf31 = torch.randint(117, 125, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
        sf2 = torch.randint(117, 125, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g)
        self.sf31 = sf31.view(torch.int32).contiguous()
        self.sf2 = sf2.view(torch.int32).contiguous()
        ones = torch.ones(E, dtype=torch.float32, device=dev)
        self.quant_scales = [self.sf31, ones, self.sf2, ones.clone()]

    def inputs(self, m, seed=0, skew=False):
        g = torch.Generator(device="cuda").manual_seed(99 + 7 * m + 1000 * seed)
        x = torch.randn(m, self.hidden, device="cuda", dtype=torch.float32, generator=g)
        # Distinct per-row power-of-two scale so every row gets its own SF exponents.
        r = torch.arange(m, device="cuda")
        x = (x * torch.pow(2.0, ((r * 5) % 9 - 4).float())[:, None]).to(torch.bfloat16)
        gc = torch.Generator().manual_seed(5 + 13 * m + 1000 * seed)
        sel = torch.empty(m, self.topk, dtype=torch.int32)
        for t in range(m):
            perm = torch.randperm(self.n_exp, generator=gc)[: self.topk]
            if skew:
                # every token's first expert is local expert 3 -> one expert holds all M rows
                perm = perm[perm != 3][: self.topk - 1]
                perm = torch.cat([torch.tensor([3]), perm])
            elif not (perm < self.e_local).any():
                perm[0] = int(torch.randint(0, self.e_local, (1,), generator=gc))
            sel[t] = perm.to(torch.int32)
        scales = torch.softmax(torch.randn(m, self.topk, generator=gc), dim=-1).float()
        return x.contiguous(), sel.cuda(), scales.cuda()

    ff = True

    def run(self, x, sel, scales, fold, fused_finalize=None):
        if fused_finalize is None:
            fused_finalize = self.ff
        assert fused_finalize == self.ff, "switch finalize mode only via phase()"
        # fused_finalize=True is production (FC2 epilogue scatters with red.global atomics, so the
        # top-k sum order -- and the bf16 output -- varies run to run even on main).
        # fused_finalize=False runs the separate finalize kernel (fixed sum order, deterministic):
        # every bitwise comparison uses it; the fold only changes bytes upstream of FC1.
        if fold:
            xin, xsf = x, None
        else:
            xin, xsf = torch.ops.trtllm.mxfp8_quantize(x, True, alignment=ALIGN)
        out = torch.ops.trtllm.fused_moe(
            xin, sel, scales, self.w31, None, self.w2, None, torch.bfloat16,
            quant_scales=self.quant_scales, input_sf=xsf, swizzled_input_sf=True,
            tp_size=1, tp_rank=0, ep_size=self.ep, ep_rank=0,
            use_mxfp8_act_scaling=True, use_fused_finalize=fused_finalize,
        )
        return out[0]


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def phase(p, ms, ff):
    """Start a finalize-mode phase. MoERunner.runner_dict's instance_key omits use_fused_finalize, so the C++
    FusedMoeRunner keeps the use_fused_finalize_ it was built with: a later call with the other flag silently
    runs the first mode (and its tactic list). Clear the dict so this phase's runners are built with ff, then
    tune (main's standalone path; the fold runner shares that runner's tuned tactics in the candidate)."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
    MoERunner.runner_dict.clear()
    torch.cuda.synchronize()
    p.ff = ff
    log(f"phase fused_finalize={ff}: autotune over M={max(ms)}")
    x, sel, sc = p.inputs(max(ms))
    with autotune():
        p.run(x, sel, sc, fold=False)
    torch.cuda.synchronize()
    # Tuned tactic per (gemm, M bucket): cross-process timing differences at small M can come from a
    # different tactic pick here (2 GEMM_1 / 2-4 GEMM_2 tactics with close times), not from the fold.
    picks = sorted((k[0].rsplit(":", 1)[-1], k[3][0][0], v[1]) for k, v in AutoTuner.get().profiling_cache.cache.items()
                   if k[0].startswith("trtllm::fused_moe"))
    log("tuned tactics (gemm, M bucket, tactic):", picks)


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def graph_time(fn, reps=20, trials=40):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            out = fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / reps)
    ts.sort()
    med = ts[len(ts) // 2]
    p10, p90 = ts[int(0.1 * (len(ts) - 1))], ts[int(0.9 * (len(ts) - 1))]
    return med, (p90 - p10) / med, g, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="bench", choices=["bench", "check", "diag", "det"])
    ap.add_argument("--ms", default="1,2,4,8,16,128,2048,6144")
    args = ap.parse_args()
    ms = [int(v) for v in args.ms.split(",")]
    dims = read_dims(args.ckpt)
    log("dims hidden/inter/n_exp/topk:", dims, "fold_max:", FOLD_MAX, "gpu:", torch.cuda.get_device_name())
    p = Problem(*dims)

    if args.mode == "bench":
        # Phase 1, deterministic finalize: per-M output sha (must match across arms -> bitwise equal to main),
        # eager twice + graph replay all bitwise equal.
        phase(p, ms, False)
        dets, shas = {}, {}
        for m in ms:
            fold = m <= FOLD_MAX
            x, sel, sc = p.inputs(m)
            det = p.run(x, sel, sc, fold).clone()
            assert torch.equal(det, p.run(x, sel, sc, fold)), f"unfused finalize not deterministic at M={m}"
            _, _, g, out = graph_time(lambda: p.run(x, sel, sc, fold), reps=2, trials=3)
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, det), f"unfused finalize: graph replay differs from eager at M={m}"
            assert torch.isfinite(det.float()).all()
            dets[m], shas[str(m)] = det, sha(det)
            log(f"M={m} fold={fold} det_sha={shas[str(m)]}")
            del g, out
        # Phase 2, production (atomic fused finalize): timing; output within tolerance of phase 1's.
        phase(p, ms, True)
        res = {}
        for m in ms:
            fold = m <= FOLD_MAX
            x, sel, sc = p.inputs(m)
            ref = p.run(x, sel, sc, fold).clone()
            med, noise, g, out = graph_time(lambda: p.run(x, sel, sc, fold))
            g.replay()
            torch.cuda.synchronize()
            e_graph, e_eager = rel_err(out, dets[m]), rel_err(ref, dets[m])
            assert e_graph < 2e-2 and e_eager < 2e-2, f"fused-finalize output off at M={m}: {e_graph} {e_eager}"
            res[str(m)] = {"kernel_ms": med, "noise_floor": noise}
            log(f"M={m} fold={fold} kernel_ms={med:.5f} noise={noise:.4f} fused_vs_det_rel={e_graph:.2e}/{e_eager:.2e}")
            del g, out
            torch.cuda.empty_cache()
        print(json.dumps({"batch": res, "output": {"class": "exact", "sha256_16_unfused_finalize": shas},
                          "fold_max_tokens": FOLD_MAX}))
        return

    if args.mode == "det":
        # main's path only: how does repeated output vary (which rows, how much, NaN?)
        res = {}
        for ff in (False, True):
            phase(p, ms, ff)
            for m in ms:
                x, sel, sc = p.inputs(m)
                for fold in ((False, True) if FOLD_MAX > 0 else (False,)):
                    outs = [p.run(x, sel, sc, fold=fold).clone() for _ in range(8)]
                    r0 = outs[0]
                    nan = int(torch.isnan(r0.float()).sum())
                    inf = int(torch.isinf(r0.float()).sum())
                    diffs = []
                    for o in outs[1:]:
                        d = (o.float() - r0.float()).abs()
                        neq = (o.view(torch.int16) != r0.view(torch.int16))
                        rows = neq.any(dim=1).nonzero().flatten()
                        diffs.append({"neq_elems": int(neq.sum()), "neq_rows": int(rows.numel()),
                                      "first_rows": rows[:6].tolist(),
                                      "max_abs": float(torch.nan_to_num(d, nan=-1.0).max()),
                                      "rel": rel_err(torch.nan_to_num(o), torch.nan_to_num(r0))})
                    key = f"M{m}_ff{int(ff)}_fold{int(fold)}"
                    # local experts per token, to see if variation follows multi-local-expert tokens
                    nloc = (sel < p.e_local).sum(dim=1)
                    res[key] = {"nan": nan, "inf": inf, "absmax": float(torch.nan_to_num(r0.float()).abs().max()),
                                "diffs": diffs[:3], "n_tokens_multi_local": int((nloc > 1).sum())}
                    log(key, json.dumps(res[key]))
                torch.cuda.empty_cache()
        print(json.dumps({"det": res}))
        return

    if FOLD_MAX <= 0:
        print(json.dumps({"passed": False, "error": "tree has no fold"}))
        sys.exit(1)

    if args.mode == "diag":
        phase(p, ms, True)
        res = {}
        for m in ms:
            x, sel, sc = p.inputs(m)
            t_s, n_s, _, _ = graph_time(lambda: p.run(x, sel, sc, False))
            t_f, n_f, _, _ = graph_time(lambda: p.run(x, sel, sc, True))
            res[str(m)] = {"standalone_ms": t_s, "fold_ms": t_f, "delta_us": (t_f - t_s) * 1e3,
                           "noise": max(n_s, n_f)}
            log(f"M={m} standalone={t_s:.5f} fold={t_f:.5f} delta={1e3 * (t_f - t_s):+.2f}us noise={max(n_s, n_f):.4f}")
        print(json.dumps({"diag": res}))
        return

    # check: untimed repeated sequence, same warmup (3) / graph reps (20) / replays (5+40) as bench.
    WARM, REPS, REPLAYS = 3, 20, 45
    results = {}
    passed = True
    cases = []
    for m in ms:
        for seed in (0, 1):
            cases.append((m, seed, False))
        cases.append((m, 0, True))
    def sequence(x, sel, sc, cmp):
        """Eager WARM+REPS calls, then a graph of REPS calls replayed REPLAYS times; cmp on every output."""
        for _ in range(WARM + REPS):
            cmp(p.run(x, sel, sc, fold=True))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(WARM):
                p.run(x, sel, sc, fold=True)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = [p.run(x, sel, sc, fold=True) for _ in range(REPS)]
        for _ in range(REPLAYS):
            g.replay()
            torch.cuda.synchronize()
            for o in outs:
                cmp(o)
        del g, outs

    # Pass 1, deterministic finalize: fold vs main's standalone path, bitwise on every iteration.
    phase(p, ms, False)
    refs = {}
    for (m, seed, skew) in cases:
        key = f"M{m}_seed{seed}{'_skew' if skew else ''}"
        x, sel, sc = p.inputs(m, seed=seed, skew=skew)
        x0 = x.clone()
        ref = p.run(x, sel, sc, fold=False).clone()  # main's path (standalone mxfp8_quantize)
        ref_det = torch.equal(ref, p.run(x, sel, sc, fold=False))
        st = {"iters": 0, "mismatch": 0, "max_abs": 0.0}

        def cmp_exact(o):
            st["iters"] += 1
            if not torch.equal(o, ref):
                st["mismatch"] += 1
                st["max_abs"] = max(st["max_abs"], (o.float() - ref.float()).abs().max().item())

        sequence(x, sel, sc, cmp_exact)
        inputs_ok = torch.equal(x, x0)
        finite = bool(torch.isfinite(ref.float()).all())
        nz = float((ref != 0).float().mean())
        nloc = (sel < p.e_local).sum(dim=1)
        refs[key] = ref
        results[key] = {"exact_iters": st["iters"], "exact_mismatch": st["mismatch"], "exact_max_abs": st["max_abs"],
                        "ref_deterministic": ref_det, "inputs_unchanged": inputs_ok, "finite": finite,
                        "nonzero_frac": nz, "tokens_ge3_local": int((nloc >= 3).sum()), "sha": sha(ref)}
        passed &= ref_det and st["mismatch"] == 0 and inputs_ok and finite and nz > 0.5
        log(key, "exact", results[key])
        torch.cuda.empty_cache()

    # Pass 2, production atomic fused finalize (nondeterministic on main too): fold within main's own spread.
    phase(p, ms, True)
    for (m, seed, skew) in cases:
        key = f"M{m}_seed{seed}{'_skew' if skew else ''}"
        x, sel, sc = p.inputs(m, seed=seed, skew=skew)
        ref = refs[key]
        main_fused_rel = max(rel_err(p.run(x, sel, sc, fold=False), ref) for _ in range(5))
        fz = {"iters": 0, "max_rel": 0.0}

        def cmp_fused(o):
            fz["iters"] += 1
            fz["max_rel"] = max(fz["max_rel"], rel_err(o, ref))

        sequence(x, sel, sc, cmp_fused)
        fused_ok = fz["max_rel"] <= max(2 * main_fused_rel, 1e-2)
        passed &= fused_ok
        results[key].update({"fused_iters": fz["iters"], "fused_fold_max_rel_vs_ref": fz["max_rel"],
                             "fused_main_max_rel_vs_ref": main_fused_rel, "fused_ok": fused_ok})
        log(key, "fused", results[key])
        torch.cuda.empty_cache()
    print(json.dumps({"passed": passed, "compared": "fused_moe(bf16 x, in-kernel MXFP8 quant) vs "
                      "mxfp8_quantize + fused_moe(fp8 x, input_sf) (main path). Bitwise (torch.equal) on every "
                      "eager iteration and graph replay with the deterministic unfused finalize; with production's "
                      "atomic fused finalize (nondeterministic on main too) max rel err vs the same reference, "
                      "bounded by main's own fused spread", "cases": results}))


if __name__ == "__main__":
    main()
