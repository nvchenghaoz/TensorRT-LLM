"""Branch 184 (idea 312): fp32 split partials from the h16 fold render (`sa+lq+el` / `sa+pw+lq+el`).

Imports experiments/ascent/cake_q1/chain.py's builders (World, Chain, capture, time_arms) and never edits it.

Arms (candidate tree: CakeDecode has _f32_variant_for; the base tree runs only `main`):
  main        : base tree, CakeDecode() as on main (fp32 partials, pinned `default` body).
  default     : candidate, TRTLLM_CSA2_CAKE_F32_VARIANT=default (must equal main bitwise).
  sa+lq+el    : candidate, pin the fold render (generic page resolve) on dual calls.
  sa+pw+lq+el : candidate, pin the fold render (pow2 page resolve) on dual calls.
  auto        : candidate production route (count buckets in cake_decode._F32_FOLD_BUCKETS).

Modes
  time  : per count: (a) per-launch eager CUPTI durations (prep / body / merge) of one width-640 call,
          256 MB L2 flush before every call, 50 iters; (b) 40-layer chain CUDA-graph replays, arms
          interleaved in rounds (5 x (2 warmup + 20)), cold L2: 'attn' = prep+body+merge only (all counts),
          'full' = + K2'/K3' WO projection (counts <= 8, pdl_consumer from _lowm_wo_follows);
          (c) Kineto per-kernel spans inside the 'full' (or 'attn') graph. Last line = evidence JSON:
          kernel_ms = 'full' chain for counts <= 8, 'attn' chain for 16, of 'auto' (candidate) / 'main' (base).
  check : candidate only. (1) direct CakeDecode calls on NON-STRESS pools (build_pools(large=False)),
          widths 640 / 512+128 (top-k 384 < 512) / 128, empty SWA rows, counts 1/2/4/8/16, pdl_consumer
          on and off: eager 10 warmup + 100 iters and graph replay 10 + 100, output, out_lse poisoned
          (finite sentinels, plus mid_out / mid_lse) before every call and compared bitwise every iteration
          against the `default` arm; 8 advancing decode steps (new q / selections copied into the captured
          graph's inputs) vs eager `default` per step; FP64 reference error reported.
          (2) 40-layer chain (attention + WO, counts 1/2/4/8): auto graph, 110 replays, every replay's 40
          attention + 40 projection outputs bitwise vs the eager `default` chain.
"""

import argparse
import importlib.util
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cake_q1_chain", os.path.join(HERE, "chain.py"))
CH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CH)
CC = CH.CC
import torch  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2 import cake_decode as CD  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention  # noqa: E402

CAND = hasattr(CD.CakeDecode, "_f32_variant_for")
ARMS = ["default", "sa+lq+el", "sa+pw+lq+el", "auto"] if CAND else ["main"]
EVID = "auto" if CAND else "main"
ART = CH.ART
K2, K3 = CH.K2, CH.K3


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def new_cake(arm):
    if arm == "main":
        return CD.CakeDecode()
    return CD.CakeDecode(f32_variant=arm)


def kind(name):
    if "cake_index_prep" in name:
        return "prep"
    if "CakeMergeF32" in name:
        return "merge"
    if "decode_body" in name or "kernel_sm120_dsv41_mixed_sparse_mla" in name:
        return "body"
    if K2 in name:
        return "K2"
    if K3 in name:
        return "K3"
    return None


def summ(v):
    return CH.summ(v) if v else None


def load_trace(path):
    evs = json.load(open(path))
    evs = evs.get("traceEvents", evs) if isinstance(evs, dict) else evs
    return [e for e in evs if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel"]


def plan_of(cake):
    out = []
    for b in cake._buffers.values():
        launch = b.get("launch")
        if launch:
            p = launch[2]
            out.append(dict(variant=p.variant, splits=p.num_splits, cpb=p.chunks_per_block,
                            decode_body=p.decode_body))
    return out


# --------------------------------------------------------------------------- per-launch (eager)
def per_launch(arm, count, pools, n=50):
    from torch.profiler import ProfilerActivity, profile

    swa_pool, main_pool = pools
    sel = CC.build_selection(count, 900 + count, width=640)
    q, sink = CC.make_q(count, 910 + count)
    out = torch.empty_like(q)
    cake = new_cake(arm)

    def step():
        cake(q, swa_pool, sel["swa"], main_pool, sel["logical"], sink, CC.SCALE, out,
             mapping=sel["mapping"], pdl_consumer=False)

    for _ in range(3):
        CH.flush()
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            CH.flush()
            step()
        torch.cuda.synchronize()
    path = ART / f"eager_{arm.replace('+', '_')}_{count}.json"
    prof.export_chrome_trace(str(path))
    dur, names = {}, {}
    for e in load_trace(path):
        k = kind(e.get("name", ""))
        if k:
            dur.setdefault(k, []).append(float(e.get("dur", 0)))
            names.setdefault(k, set()).add(e.get("name", "")[:110])
    res = {k: summ(v) | {"n": len(v), "names": sorted(names[k])} for k, v in dur.items()}
    res["plan"] = plan_of(cake)
    return res


# --------------------------------------------------------------------------- chain graphs
def set_arm(chain, arm, keep):
    cakes = []
    for L in chain.layers:
        L["fi"]._cake = new_cake(arm)
        cakes.append(L["fi"]._cake)
    keep.append(cakes)  # graphs reference these buffers: keep them alive


def attn_only(chain):
    def run():
        for L in chain.layers:
            L["state"].output_pdl_consumer = False
            L["fi"].forward_generation(L["q"].flatten(1), L["meta"], L["args"])
        return [L["out"] for L in chain.layers]
    return run


def capture_fn(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = fn()
    torch.cuda.synchronize()
    return g, outs


def spans(g, tag, reps=5):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            CH.flush()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
    path = ART / f"spans_{tag}.json"
    prof.export_chrome_trace(str(path))
    dur = {}
    for e in load_trace(path):
        n = e.get("name", "")
        k = kind(n)
        if k == "body":
            k = "body_dual" if ("dual" in n or "decode_body<" in n) else "body_single"
        if k:
            dur.setdefault(k, []).append(float(e.get("dur", 0)))
    return {k: summ(v) | {"n": len(v)} for k, v in dur.items()}


def mode_time(args):
    world = CH.World(args.ckpt)
    pools = CC.build_pools(5, large=False)
    batch, table = {}, {}
    for count in [int(c) for c in args.counts.split(",")]:
        row = {"clock_start": CC.sm_clock()}
        row["per_launch"] = {arm: per_launch(arm, count, pools) for arm in ARMS}
        log(f"M={count} per_launch " + json.dumps(
            {a: {k: (v.get("median") if isinstance(v, dict) and "median" in v else v) for k, v in r.items()}
             for a, r in row["per_launch"].items()}))
        chain = CH.Chain(world, count, seed=count)
        keep, graphs = [], {}
        full = count <= 8
        for arm in ARMS:
            set_arm(chain, arm, keep)
            graphs[("attn", arm)], _ = capture_fn(attn_only(chain))
            if full:
                chain.arm = "default"
                graphs[("full", arm)], _ = capture_fn(chain)
        t = CH.time_arms(graphs)
        row["chain_us"] = {f"{f}|{a}": {k: (v * 1e3 if k != "noise" else v) for k, v in s.items()}
                           for (f, a), s in t.items()}
        tag = "full" if full else "attn"
        row["spans"] = {arm: spans(graphs[(tag, arm)], f"M{count}_{tag}_{arm.replace('+', '_')}")
                        for arm in ARMS}
        row["chain_plans"] = {arm: sorted({json.dumps(p) for c in cs for p in plan_of(c)})
                              for arm, cs in zip(ARMS, keep)}
        row["clock_end"] = CC.sm_clock()
        table[str(count)] = row
        s = t[(tag, EVID)]
        batch[str(count)] = {"kernel_ms": s["median"], "noise_floor": s["noise"], "p10_ms": s["p10"],
                             "p90_ms": s["p90"], "chain": tag, "per_layer_us": s["median"] * 1e3 / 40}
        log(f"M={count} chain " + json.dumps(row["chain_us"]))
        log(f"M={count} spans " + json.dumps({a: {k: v["median"] for k, v in r.items()}
                                               for a, r in row["spans"].items()}))
        log(f"M={count} plans " + json.dumps(row["chain_plans"]))
        del graphs, chain, keep
        torch.cuda.empty_cache()
    (ART / "table_312.json").write_text(json.dumps(table, indent=1))
    print(json.dumps({"batch": batch, "candidate": CAND, "arm": EVID, "lease": CH.lease_info(),
                      "output": {"class": "exact"}}))


# --------------------------------------------------------------------------- check
SEL_KEYS = ("swa", "logical", "table", "requests", "visible")


def case_inputs(count, width, seed, main_w):
    sel = CC.build_selection(count, seed, width=width, empty_rows=(0,) if count > 1 else ())
    if main_w is not None:
        sel["logical"] = sel["logical"][:, :main_w].contiguous()
    sel["mapping"] = (sel["table"], CC.PAGE, sel["max_positions"], sel["requests"], sel["visible"])
    q, sink = CC.make_q(count, seed + 1)
    return sel, q, sink


def call(cake, q, sink, sel, pools, width, out, pdl):
    swa_pool, main_pool = pools
    has_main = width > CC.SWA_W
    cake(q, swa_pool, sel["swa"], main_pool if has_main else None, sel["logical"] if has_main else None,
         sink, CC.SCALE, out, mapping=sel["mapping"] if has_main else None, pdl_consumer=pdl)


def poison(cake, out):
    out.fill_(3.0)
    for b in cake._buffers.values():
        b["mid_out"].fill_(5.0)
        b["mid_lse"].fill_(-3.0)
        b["lse"].fill_(-3.0)


def lse_of(cake):
    return next(iter(cake._buffers.values()))["lse"]


def load_case(dst_sel, q, sink, src):
    sel2, q2, sink2 = src
    for key in SEL_KEYS:
        dst_sel[key].copy_(sel2[key])
    q.copy_(q2)
    sink.copy_(sink2)


def check_direct(args, pools):
    ok, rep = True, {}
    shapes = [(640, None), (640, 384), (128, None)]
    for count in [int(c) for c in args.counts.split(",")]:
        for width, main_w in shapes:
            for pdl in (True, False):
                tag = f"M{count}_w{width}{'' if main_w is None else '_k' + str(main_w)}_pdl{int(pdl)}"
                sel, q, sink = case_inputs(count, width, 40 + count, main_w)
                ref_c = new_cake("default")
                ref_out = torch.empty_like(q)
                call(ref_c, q, sink, sel, pools, width, ref_out, pdl)
                torch.cuda.synchronize()
                ref_lse = lse_of(ref_c).clone()
                ref_out = ref_out.clone()
                ref64 = CC.reference(q, sink, pools[0], pools[1], dict(sel, width=width))
                steps = [case_inputs(count, width, 1000 + 17 * s + count, main_w) for s in range(8)]
                r = {"default_max_abs_vs_fp64": (ref_out.double() - ref64).abs().max().item()}
                for arm in ("sa+lq+el", "sa+pw+lq+el", "auto"):
                    cake = new_cake(arm)
                    out = torch.empty_like(q)
                    # eager: same warmup / iteration / reset sequence as the timing harness
                    bad, worst, fin = 0, 0.0, True
                    for it in range(args.warmup + args.iters):
                        if it:
                            poison(cake, out)
                        call(cake, q, sink, sel, pools, width, out, pdl)
                        torch.cuda.synchronize()
                        lse = lse_of(cake)
                        fin &= bool(torch.isfinite(out).all() and torch.isfinite(lse).all())
                        if not (torch.equal(out, ref_out) and torch.equal(lse, ref_lse)):
                            bad += 1
                            worst = max(worst, (out.float() - ref_out.float()).abs().max().item(),
                                        (lse - ref_lse).abs().max().item())
                    eager = dict(bad=bad, max_abs=worst, finite=fin, plan=plan_of(cake))
                    err64 = (out.double() - ref64).abs().max().item()
                    # graph replay of the warmed-up shape
                    g, _ = capture_fn(lambda: call(cake, q, sink, sel, pools, width, out, pdl))
                    gbad, gworst = 0, 0.0
                    for it in range(args.warmup + args.iters):
                        poison(cake, out)
                        g.replay()
                        torch.cuda.synchronize()
                        lse = lse_of(cake)
                        fin &= bool(torch.isfinite(out).all() and torch.isfinite(lse).all())
                        if not (torch.equal(out, ref_out) and torch.equal(lse, ref_lse)):
                            gbad += 1
                            gworst = max(gworst, (out.float() - ref_out.float()).abs().max().item())
                    # 8 advancing decode steps through the captured graph (new inputs every step)
                    sbad = 0
                    for src in steps:
                        load_case(sel, q, sink, src)
                        poison(cake, out)
                        g.replay()
                        ro = torch.empty_like(q)
                        rc = new_cake("default")
                        call(rc, q, sink, sel, pools, width, ro, pdl)
                        torch.cuda.synchronize()
                        if not (torch.equal(out, ro) and torch.equal(lse_of(cake), lse_of(rc))):
                            sbad += 1
                    load_case(sel, q, sink, case_inputs(count, width, 40 + count, main_w))
                    del g
                    passed = bad == 0 and gbad == 0 and sbad == 0 and fin
                    ok &= passed
                    r[arm] = dict(eager=eager, graph=dict(bad=gbad, max_abs=gworst), steps8_bad=sbad,
                                  max_abs_vs_fp64=err64, passed=passed)
                rep[tag] = r
                log(tag, json.dumps(r))
    return ok, rep


def check_chain(args, world):
    ok, rep = True, {}
    for count in [int(c) for c in args.chain_counts.split(",") if c]:
        chain = CH.Chain(world, count, seed=20 + count)
        keep = []
        set_arm(chain, "default", keep)
        chain.arm = "default"
        chain()
        torch.cuda.synchronize()
        ref_res = [r.clone() for r in chain()]
        ref_att = [L["out"].clone() for L in chain.layers]
        torch.cuda.synchronize()
        set_arm(chain, "auto", keep)
        g, outs = CH.capture(chain, "default")
        bad, worst = 0, 0.0
        for it in range(args.warmup + args.iters):
            chain.poison(outs)
            CH.flush()
            g.replay()
            torch.cuda.synchronize()
            d = 0
            for o, r, L, a in zip(outs, ref_res, chain.layers, ref_att):
                if not (torch.equal(o, r) and torch.equal(L["out"], a)):
                    d += 1
                    worst = max(worst, (L["out"].float() - a.float()).abs().max().item())
            bad += int(d > 0)
        variants = sorted({p["variant"] for L in chain.layers for p in plan_of(L["fi"]._cake)})
        flags = sum(bool(DeepseekV41Attention._lowm_wo_follows(L["mod"], L["out"].view(count, -1)))
                    for L in chain.layers)
        passed = bad == 0
        ok &= passed
        rep[str(count)] = dict(iterations=args.warmup + args.iters, bad=bad, max_abs=worst, variants=variants,
                               pdl_flag_layers=flags, passed=passed)
        log(f"chain M={count}", json.dumps(rep[str(count)]))
        del g, outs, chain, keep
        torch.cuda.empty_cache()
    return ok, rep


def mode_check(args):
    assert CAND, "check runs in the candidate tree"
    pools = CC.build_pools(5, large=False)
    ok1, rep1 = check_direct(args, pools)
    ok2, rep2 = (True, {})
    if args.chain_counts:
        ok2, rep2 = check_chain(args, CH.World(args.ckpt))
    (ART / "check_312.json").write_text(json.dumps({"direct": rep1, "chain": rep2}, indent=1))
    summary = {t: {a: (v["passed"] if isinstance(v, dict) else v) for a, v in r.items()} for t, r in rep1.items()}
    print(json.dumps({"passed": bool(ok1 and ok2), "direct_passed": bool(ok1), "chain_passed": bool(ok2),
                      "compared": "fold-render fp32 partials (sa+lq+el, sa+pw+lq+el, auto) output + out_lse "
                                  "bitwise vs CakeDecode fp32 `default` (main's route) on non-stress pools; eager "
                                  "and graph, 110 iterations each with finite sentinels, 8 advancing steps; "
                                  "40-layer attention+WO chain auto graph vs eager default",
                      "direct": summary, "chain": rep2, "lease": CH.lease_info()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["time", "check"])
    ap.add_argument("ckpt", nargs="?", default="")
    ap.add_argument("--counts", default="1,2,4,8,16")
    ap.add_argument("--chain-counts", default="1,2,4,8")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    mode_time(a) if a.mode == "time" else mode_check(a)
