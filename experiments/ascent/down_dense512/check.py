"""Idea 112 / branch 51: untimed correctness for the down_proj CUTLASS-only dense band (M 513-1023).

One process, branch tree. Tuned like the engine warmup (one M=8192 call per MLP shape under
autotune()). Main's output in the same process = the module switches turned off at run time:
  off_dp  : torch_custom_ops.NVFP4_DENSE_DP = False   (the new band off)
  off_all : torch_custom_ops.NVFP4_DENSE_TUNING = False (TRTLLM_NVFP4_DENSE_TUNING=0)
Before every op call a block of the output's size is filled with 0xA5 and freed, so the op's
output allocation starts from 0xA5 bytes (unwritten elements would read 0xA5A5).
  1. every M 513..1023 x {randn seed 1, seed 2, heavy-tailed randn*2^U(-8,8) rows; real layer
     0/30/59 down inputs captured at M=1000 (rows sliced, or wrapped mod 1000 above 1000)}:
     band output int16-equal to off_dp and off_all; band dispatch counted.
  2. repeated A,B,A = (1000, 600, 1000), 25 calls each (the timing run's 5 warmup + 20 timed),
     every call vs call 1 and vs off_dp; seed 1 and real L30.
  3. M 300/512/1024/1200/2000/4000/8000: on vs off_dp vs off_all int16-equal, no band dispatch,
     picks logged.
  4. FP32 dequant reference at 600/900/1000 (seed1, real L30): error within 1.10x main + 1 ulp.
  5. inputs unchanged; autotuner cache unchanged by the checks.

usage: check.py <checkpoint> <capture_dir>
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nvfp4_dense"))
import bench  # noqa: E402

log = bench.log
A5 = -23131  # int16 view of 0xA5A5


def main():
    ck, cap = sys.argv[1], sys.argv[2]
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    t = bench.tco()
    dims = bench.read_dims(ck)
    n, k = dims["down"]
    rng = t._NVFP4_DENSE_DP_SHAPES[(n, k)]
    log(f"[policy] dp_band={rng} dense={t._NVFP4_DENSE_SHAPES} sm_ok={t._nvfp4_dense_sm_ok()}")
    counts = {"dp": 0, "dense": 0, "main": 0}
    _fwd = t.NVFP4GemmUnifiedRunner.forward

    def fwd(self, inputs, tactic=-1, bias=None, **kw):
        key = "dp" if self.dense_cutlass_only else ("dense" if self.dense_min_m is not None else "main")
        counts[key] += 1
        return _fwd(self, inputs, tactic=tactic, bias=bias, **kw)

    t.NVFP4GemmUnifiedRunner.forward = fwd

    def mode(name):
        t.NVFP4_DENSE_TUNING = name != "off_all"
        t.NVFP4_DENSE_DP = name != "off_dp"

    def call(xq, xsf, wq, wsf, alpha, m):
        s = torch.empty(m * n * 2, dtype=torch.uint8, device="cuda").fill_(0xA5)
        del s
        return bench.gemm(xq, xsf, wq, wsf, alpha)

    weights = {name: bench.make_weight(*dims[name], 100 + i) for i, name in enumerate(("gate_up", "down"))}
    t0 = time.time()
    with autotune():
        for name, (wq, wsf, wgs) in weights.items():
            xq, xsf, xgs = bench.make_act(8192, dims[name][1], 7)
            bench.gemm(xq, xsf, wq, wsf, (1.0 / (xgs * wgs)).float())
    torch.cuda.synchronize()
    log(f"[tune] {time.time() - t0:.1f}s")
    cache_before = dict(AutoTuner.get().profiling_cache.cache)
    swq, swsf, swgs = weights["down"]

    # real captures
    with open(os.path.join(cap, "manifest.json")) as f:
        man = json.load(f)
    reals = {}
    for e in man["entries"]:
        if e["shape"] == "down" and e["M"] == 1000 and e["prompt"] == "cap_isl1k@1":
            tt = torch.load(os.path.join(cap, e["file"]))
            wt = torch.load(os.path.join(cap, man["weights"][f"L{e['layer']}_down"]["file"]))
            xq = tt["act_fp4"].cuda()
            rows_pad = (xq.shape[0] + 127) // 128 * 128
            lin = torch.ops.trtllm.block_scale_interleave_reverse(
                tt["act_sf"].cuda().view(torch.uint8).reshape(rows_pad, -1))[:xq.shape[0]].contiguous()
            reals[e["layer"]] = {"xq": xq, "sf_lin": lin, "wq": wt["weight"].cuda(),
                                 "wsf": wt["weight_scale"].cuda(),
                                 "alpha": tt["alpha"].cuda().float().reshape(-1)[:1],
                                 "out": tt.get("out")}
    log(f"[capture] real down layers={sorted(reals)} commit={man['commit']}")

    def real_inputs(layer, m):
        r = reals[layer]
        idx = torch.arange(m, device="cuda") % r["xq"].shape[0]
        xq = r["xq"][idx].contiguous()
        sf = torch.ops.trtllm.block_scale_interleave(r["sf_lin"][idx].contiguous()).reshape(-1)
        return xq, sf, r["wq"], r["wsf"], r["alpha"]

    def syn_inputs(variant, m):
        seed, heavy = {"seed1": (1, False), "seed2": (2, False), "heavy": (3, True)}[variant]
        xq, xsf, xgs = bench.make_act(m, k, seed * 7919 + m, heavy)
        return xq, xsf, swq, swsf, (1.0 / (xgs * swgs)).float()

    variants = ["seed1", "seed2", "heavy"] + [f"real_L{L}" for L in sorted(reals)]

    def inputs_for(v, m):
        return real_inputs(int(v.split("L")[1]), m) if v.startswith("real") else syn_inputs(v, m)

    passed = True
    fails = []
    # in-model sanity: main's op on the captured M=1000 input equals the in-model output
    inmodel = {}
    for L, r in reals.items():
        if r["out"] is not None:
            mode("off_dp")
            o = call(*real_inputs(L, 1000), 1000)
            inmodel[L] = bench.int16_mismatch(o, r["out"].cuda())[0]
    log(f"[inmodel] main(off_dp) vs captured in-model output at M=1000: {inmodel}")

    # 1. every M in the band
    band = {"cases": 0, "mismatch_vs_off_dp": 0, "mismatch_vs_off_all": 0, "a5_elems": 0, "first_fail": None}
    for m in range(rng[0], rng[1] + 1):
        for v in variants:
            ins = inputs_for(v, m)
            snap = [x.clone() for x in ins]
            mode("on")
            c0 = counts["dp"]
            o_on = call(*ins, m)
            dp_used = counts["dp"] - c0
            mode("off_dp")
            o_off = call(*ins, m)
            mode("off_all")
            o_all = call(*ins, m)
            mode("on")
            torch.cuda.synchronize()
            a, fa = bench.int16_mismatch(o_on, o_off)
            b, _ = bench.int16_mismatch(o_on, o_all)
            a5 = int((o_on.view(torch.int16) == A5).sum().item())
            unchanged = all(torch.equal(x.view(torch.uint8) if x.dtype != torch.float32 else x,
                                        y.view(torch.uint8) if y.dtype != torch.float32 else y)
                            for x, y in zip(snap, ins))
            ok = a == 0 and b == 0 and dp_used == 1 and unchanged and bool(torch.isfinite(o_on.float()).all()) \
                and o_on.shape == (m, n) and o_on.dtype == torch.bfloat16
            band["cases"] += 1
            band["mismatch_vs_off_dp"] += a
            band["mismatch_vs_off_all"] += b
            band["a5_elems"] += a5
            if not ok:
                passed = False
                fails.append(f"band|{v}|M{m}|vs_off_dp={a}|vs_off_all={b}|dp_used={dp_used}|unchanged={unchanged}")
                if band["first_fail"] is None:
                    band["first_fail"] = {"v": v, "m": m, "first": fa}
            del o_on, o_off, o_all, ins, snap
        if m % 64 == 0 or m in (rng[0], rng[1]):
            log(f"[band] through M={m}: {band} counts={counts}")
    log(f"[band] done {band}")

    # 2. repeated A,B,A
    seq = {}
    for v in ("seed1", "real_L30"):
        for step, m in enumerate((1000, 600, 1000)):
            ins = inputs_for(v, m)
            mode("off_dp")
            ref = call(*ins, m)
            mode("on")
            first = call(*ins, m)
            bad = 0
            for _ in range(24):
                o = call(*ins, m)
                bad += bench.int16_mismatch(o, first)[0]
                del o
            vs = bench.int16_mismatch(first, ref)[0]
            seq[f"{v}|{step}|M{m}"] = {"repeat_mismatch": bad, "vs_off_dp": vs}
            if bad or vs:
                passed = False
                fails.append(f"seq|{v}|{step}|M{m}")
    log(f"[seq] {seq}")

    # 3. out-of-band M
    outb = {}
    for m in (300, 512, 1024, 1200, 2000, 4000, 8000):
        ins = syn_inputs("seed1", m)
        mode("on")
        c0 = counts["dp"]
        o_on = call(*ins, m)
        used = counts["dp"] - c0
        mode("off_dp")
        o_off = call(*ins, m)
        mode("off_all")
        o_all = call(*ins, m)
        mode("on")
        r = {"vs_off_dp": bench.int16_mismatch(o_on, o_off)[0], "off_all_vs_off_dp": None,
             "dp_dispatch": used, "picks": bench.picks_for("down", m, n, k)}
        if m < 4096:  # off_all and on agree there by construction (no dense band); compare anyway
            r["off_all_vs_off_dp"] = bench.int16_mismatch(o_all, o_off)[0]
        outb[m] = r
        if r["vs_off_dp"] or used or (r["off_all_vs_off_dp"] or 0):
            passed = False
            fails.append(f"outband|M{m}")
        del o_on, o_off, o_all
    log(f"[outband] {outb}")

    # 4. FP32 reference
    refs = {}
    for v in ("seed1", "real_L30"):
        for m in (600, 900, 1000):
            xq, xsf, wq, wsf, alpha = inputs_for(v, m)
            if v == "seed1":
                xgs = None
            mode("on")
            o_on = call(xq, xsf, wq, wsf, alpha, m)
            mode("off_dp")
            o_off = call(xq, xsf, wq, wsf, alpha, m)
            mode("on")
            # dequant with gs=1: alpha carries the global scales
            xr = bench.dequant(xq, xsf, torch.ones(1, device="cuda"), m, k)
            wr = bench.dequant(wq, wsf, torch.ones(1, device="cuda"), n, k)
            ref = (xr @ wr.t()) * alpha
            e_c = o_on.float() - ref
            e_m = o_off.float() - ref
            ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)
            r = {"maxabs": e_c.abs().max().item(), "relfro": (e_c.norm() / ref.norm()).item(),
                 "main_maxabs": e_m.abs().max().item(), "main_relfro": (e_m.norm() / ref.norm()).item(),
                 "cand_vs_main_maxabs": (o_on.float() - o_off.float()).abs().max().item()}
            r["tol_ok"] = r["maxabs"] <= 1.10 * r["main_maxabs"] + ulp and r["relfro"] <= 1.10 * r["main_relfro"]
            refs[f"{v}|M{m}"] = r
            if not r["tol_ok"]:
                passed = False
                fails.append(f"ref|{v}|M{m}")
            del xr, wr, ref, e_c, e_m
    log(f"[ref] {refs}")

    cache_after = dict(AutoTuner.get().profiling_cache.cache)
    mutated = cache_before != cache_after
    passed &= not mutated
    passed &= all(v == 0 for v in inmodel.values())
    out = {"passed": bool(passed),
           "compared": "band pick (CUTLASS-only gemm_dense_dp) vs main in the same process with the band off "
                       "(NVFP4_DENSE_DP=False) and with all dense tuning off (TRTLLM_NVFP4_DENSE_TUNING=0), int16, "
                       "outputs allocated over 0xA5 bytes, every M 513-1023 x seed1/seed2/heavy + real L0/30/59; "
                       "A,B,A (1000,600,1000) 25 calls each vs call 1; out-of-band M unchanged; FP32 dequant ref",
           "band": band, "seq": seq, "outband": outb, "ref": refs, "inmodel_main_vs_captured": inmodel,
           "autotuner_cache_mutated": mutated, "dispatch_counts": counts, "fails": fails[:50]}
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(art, "check.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
