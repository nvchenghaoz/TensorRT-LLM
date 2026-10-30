"""Idea 62 (denoiser_gemm, branch 42): lift serves()'s M>=24576 decline for landing 7's fused-128
pair + gated residual (trtllm::flux3_sm120_nvfp4_out_pair_residual) at raster 2 (AlongN).

Base (main 7ab1b650: L15 flux3_tres one-launch tail, L16 fused-128 AlongN at 8192<=M<24576, L18
block-27 cut). Main's served route at M=32848 ([1,3,D] table + u8 index, the i2i single blocks):
transformer_flux3.py Flux3SingleStreamBlock.forward tries sm120_tail_pair_residual_forward first
(L1265; declined by serves() at M>=24576), then sm120_tail_one_launch_forward (L1274; branch 3's
flux3_tres, op trtllm::flux3_sm120_nvfp4_tail_one_launch, gate_mode 1), then the bare pair.

Shapes reused from experiments/ascent/pair_residual_raster/bench.py (synthetic NVFP4, fixed seeds).

  table : STEP 0, one process. 20-call CUDA graphs at M=32848 u8x3, 3 rotated input sets:
          A   = main's served route (flux3_tres tail op), A2 = twin of A,
          B   = fused-128 raster 2 (AlongN) past serves(), Bp = fused-128 raster 0 (AlongM).
          Seeded random arm order per sample (balanced predecessors), L2 flushed and x restored
          from a pristine copy before every replay; median us/call, p10/p90, median paired diffs.
  bench : run_gpu_pair arm (both trees). One CUDA graph per case = one denoise step's block tails
          as served by the tree's route (gate_form -> fused op, else flux3_tres at M>=24576):
            t2i-2048: 27 x (16464, select) + 4 x (16384, row) + 1 x (16384, select sliced)
            i2i-2048: 27 x (32848, u8x3) + 9 x (16384, row) + 1 x (16384, u8x3 sliced)
  check : untimed. Bitwise at 32848 (fused r2 / routed vs flux3_tres vs bare pair + table residual
          vs f32 emulation), eager and graph, NaN-poisoned outputs, epoch +1; 16384/16464 forms
          (incl. both block-27 forms) routed == main's pair + compiled residual; kill switches;
          dynamo fullgraph trace of gate_form; repeat sequence (t2i->i2i->t2i rounds).
"""

import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "pair_residual_raster"))
import bench as rb  # noqa: E402  (branch 21's shapes and helpers)

H = rb.H
SETS = 3
WARMUP = 3
ITERS = 20
log = rb.log


def lptr_mod():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair_residual as lptr
    return lptr


def make(M, seed, form):
    """rb.make_case plus the block-27 sliced forms (index = prefix of the full-S index)."""
    if form == "u8x3_slice":  # i2i block 27 at 16384: [1,3,D] + mod[3][:, :16384]
        c = rb.make_case(M, seed, "u8x3")
        g = torch.Generator(device="cuda").manual_seed(seed + 5)
        full = torch.randint(0, 3, (1, 32848, 1), device="cuda", dtype=torch.uint8, generator=g)
        c["full_index"] = full
        c["index"] = full[:, :M]
        c["gate_tok"] = c["gate"][0][c["index"].reshape(-1).long()].unsqueeze(0).contiguous()
        return c
    if form == "select_slice":  # t2i block 27 at 16384: [1,2,D] + mod[3][:, :16384] (bool)
        c = rb.make_case(M, seed, "select")
        full = (torch.arange(16464, device="cuda") >= 16384).view(1, 16464, 1)
        c["full_index"] = full
        c["index"] = full[:, :M]
        return c
    return rb.make_case(M, seed, form)


def run_tol(c):
    """Main's M>=24576 route: branch 3's flux3_tres one-launch tail, gate_mode 1 (table+index)."""
    return torch.ops.trtllm.flux3_sm120_nvfp4_tail_one_launch(
        c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
        c["attn_q"], c["attn_sf"], c["out_ad"].weight, c["out_ad"].scale,
        c["x"], c["gate"], c["index"], 1)


def served(c):
    """The tree's tail route order (Flux3SingleStreamBlock.forward): fused op if gate_form
    takes the call, else flux3_tres at M >= 24576, else the bare pair + compiled residual."""
    lptr = lptr_mod()
    if lptr.gate_form(c["gate"], c["index"], c["M"], H):
        return rb.run_fused(c)
    if c["M"] >= 24576:
        return run_tol(c)
    return rb.run_base(c)


def preload():
    lptr = rb.preload()
    has = hasattr(lptr, "large_m_enabled")
    log(f"tree: {'branch' if has else 'base'}; serves(32848)={lptr.serves(32848)} "
        f"serves(16464)={lptr.serves(16464)} serves(16384)={lptr.serves(16384)} "
        f"raster_for(32848)={lptr.raster_for(32848)}")
    return lptr


def stat(v):
    return rb.stat(v)


def time_arms(graphs, sets, pristine_x, samples, warm, calls, seed=1234):
    names = list(graphs)
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def prep():
        for c, px in zip(sets, pristine_x):
            c["x"].copy_(px)
        flush.zero_()

    for _ in range(warm):
        for n in names:
            prep()
            graphs[n].replay()
    torch.cuda.synchronize()
    rng = random.Random(seed)
    out = {n: [] for n in names}
    for i in range(samples):
        order = names[:]
        rng.shuffle(order)
        evs = []
        for n in order:
            prep()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            b.record()
            evs.append((n, a, b))
        torch.cuda.synchronize()
        for n, a, b in evs:
            out[n].append(a.elapsed_time(b) * 1e3 / calls)
        if i % 10 == 0:
            log(f"  sample {i} order {order}: " +
                " ".join(f"{n}={out[n][-1]:.1f}us" for n in names))
    return out


def table():
    lptr = preload()
    M, calls, samples = 32848, 20, int(os.environ.get("SAMPLES", "60"))
    cs = [make(M, 7 * M + i, "u8x3") for i in range(SETS)]
    pristine_x = [c["x"].clone() for c in cs]
    pristine = [[t.clone() for t in rb.inputs(c)] for c in cs]
    s = torch.cuda.Stream()

    def seq(f):
        return [lambda c=cs[i % SETS]: f(c) for i in range(calls)]

    graphs = {}
    graphs["A_main_tres"], _ = rb.capture(seq(run_tol), s)
    graphs["B_fused_r2"], _ = rb.capture(seq(lambda c: rb.run_fused_r(c, 2)), s)
    graphs["Bp_fused_r0"], _ = rb.capture(seq(lambda c: rb.run_fused_r(c, 0)), s)
    r0_res = int(lptr.module().lpt_pair_residual_last_raster())
    graphs["A2_main_tres"], _ = rb.capture(seq(run_tol), s)
    log(f"captured; raster 0 resolves to {r0_res}")
    res = time_arms(graphs, cs, pristine_x, samples=samples, warm=5, calls=calls)
    row = {"M": M, "form": "u8x3 [1,3,D] table + uint8 index 0..2", "calls_per_replay": calls,
           "samples": samples, "raster0_resolves_to": r0_res}
    for n, v in res.items():
        med, p10, p90 = stat(v)
        row[n] = {"us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
                  "noise": round((p90 - p10) / med, 4)}

    def pdiff(a, b):
        d = sorted(x - y for x, y in zip(res[a], res[b]))
        return round(d[len(d) // 2], 2)

    a = row["A_main_tres"]["us"]
    row["paired_median_diff_us"] = {"A2-A": pdiff("A2_main_tres", "A_main_tres"),
                                    "B-A": pdiff("B_fused_r2", "A_main_tres"),
                                    "Bp-A": pdiff("Bp_fused_r0", "A_main_tres")}
    row["AA_pct"] = round(100 * (row["A2_main_tres"]["us"] - a) / a, 2)
    row["B_vs_A_pct"] = round(100 * (row["B_fused_r2"]["us"] - a) / a, 2)
    row["Bp_vs_A_pct"] = round(100 * (row["Bp_fused_r0"]["us"] - a) / a, 2)
    for p, c in zip(pristine, cs):
        assert all(torch.equal(x, y) for x, y in zip(p, rb.inputs(c))), "inputs changed"
    log(json.dumps(row))
    print(json.dumps({"table_us_per_call": row}))


CASES = {
    "t2i-2048": [(16464, "select", 27), (16384, "row", 4), (16384, "select_slice", 1)],
    "i2i-2048": [(32848, "u8x3", 27), (16384, "row", 9), (16384, "u8x3_slice", 1)],
}


def bench():
    preload()
    pool = {}
    for parts in CASES.values():
        for M, form, _ in parts:
            if (M, form) not in pool:
                pool[(M, form)] = [make(M, 1000 * M + 17 * i + len(form), form)
                                   for i in range(SETS)]
    pristine = {k: [[t.clone() for t in rb.inputs(c)] for c in cs] for k, cs in pool.items()}
    graphs = {}
    s = torch.cuda.Stream()
    for case, parts in CASES.items():
        sq = []
        for M, form, calls in parts:
            cs = pool[(M, form)]
            sq += [lambda c=cs[i % SETS]: served(c) for i in range(calls)]
        graphs[case], _ = rb.capture(sq, s)
    all_sets = [c for cs in pool.values() for c in cs]
    px = [c["x"].clone() for c in all_sets]
    res = time_arms(graphs, all_sets, px, samples=60, warm=10, calls=1, seed=99)
    result = {}
    for case, v in res.items():
        med, p10, p90 = stat(v)
        result[case] = {"kernel_ms": med / 1e3, "noise_floor": (p90 - p10) / med,
                        "p10_ms": p10 / 1e3, "p90_ms": p90 / 1e3}
        log(f"{case}: median {med/1e3:.4f} ms p10 {p10/1e3:.4f} p90 {p90/1e3:.4f}")
    for k, cs in pool.items():
        for p, c in zip(pristine[k], cs):
            assert all(torch.equal(x, y) for x, y in zip(p, rb.inputs(c))), "inputs changed"
    print(json.dumps({
        "batch": {k: {"kernel_ms": v["kernel_ms"], "noise_floor": v["noise_floor"]}
                  for k, v in result.items()},
        "detail": result,
        "unit": "one CUDA graph replay = one denoise step of the case's block tails as routed "
                "by the tree (post-L18 counts); 3 input sets per (M, form); L2 flushed",
        "output": {"class": "exact"},
    }))


def check():
    lptr = preload()
    ws = lptr.workspace(torch.device("cuda", torch.cuda.current_device()))
    epoch_word = 128 // 4

    def epoch():
        return int(ws[:256].view(torch.int32)[epoch_word].item())

    passed = True
    results = {}
    configs = [(32848, "u8x3", 0), (32848, "u8x3", 1), (32848, "u8x3", 2),
               (16464, "select", 0), (16384, "row", 0), (16384, "u8x3_slice", 0),
               (16384, "select_slice", 0), (4176, "select", 0)]
    pool = {}
    for M, form, k in configs:
        c = make(M, 100 + M + 31 * k + len(form), form)
        key = f"M{M}_{form}_{k}"
        pool[key] = c
        pristine = [t.clone() for t in rb.inputs(c)]
        attn_y, mlp_y = rb.run_pair(c)
        ref = rb.run_base(c)  # main's bare pair + compiled residual (per-token gate at u8)
        emu = rb.emulate_ref(c, attn_y, mlp_y)  # f32 emulation of pair + table residual
        r = {"bare_vs_f32_emulation_bitwise": bool(torch.equal(ref, emu))}
        ok_all = r["bare_vs_f32_emulation_bitwise"]
        arms = [("routed", lambda: served(c))]
        if M >= 24576:
            arms += [("tres", lambda: run_tol(c)), ("fused_r2", lambda: rb.run_fused_r(c, 2)),
                     ("fused_r0", lambda: rb.run_fused_r(c, 0))]
        for arm, fn in arms:
            for _ in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            n_ref = n_emu = n_it0 = 0
            epochs = []
            it0 = None
            for i in range(ITERS):
                e0 = epoch()
                out = fn().reshape(c["x"].shape)
                torch.cuda.synchronize()
                epochs.append(epoch() - e0)
                n_ref += bool(torch.equal(out, ref))
                n_emu += bool(torch.equal(out, emu))
                if it0 is None:
                    it0 = out.clone()
                n_it0 += bool(torch.equal(out, it0))
                out.fill_(float("nan"))
                del out
            fused_arm = arm.startswith("fused") or (arm == "routed" and bool(lptr.gate_form(
                c["gate"], c["index"], M, H)))
            exp_ep = {1} if fused_arm else {0}
            ok = n_ref == ITERS and n_emu == ITERS and n_it0 == ITERS and set(epochs) == exp_ep
            ok_all &= ok
            r[arm] = {"bitwise_vs_bare_pair_residual": f"{n_ref}/{ITERS}",
                      "bitwise_vs_f32_emulation": f"{n_emu}/{ITERS}",
                      "bitwise_vs_iter0": f"{n_it0}/{ITERS}",
                      "epoch_step": sorted(set(epochs)), "ok": ok}
        if M >= 24576:  # the 80-row partial tile (M % 128 rows), explicitly
            tail = M % 128
            o2 = rb.run_fused_r(c, 2).reshape(c["x"].shape)
            o3 = run_tol(c).reshape(c["x"].shape)
            torch.cuda.synchronize()
            r["partial_tile_rows"] = tail
            r["partial_tile_fused_r2_eq_tres"] = bool(torch.equal(o2[:, -tail:], o3[:, -tail:]))
            r["partial_tile_fused_r2_eq_bare"] = bool(torch.equal(o2[:, -tail:], ref[:, -tail:]))
            r["full_fused_r2_eq_tres"] = bool(torch.equal(o2, o3))
            ok_all &= (r["partial_tile_fused_r2_eq_tres"] and r["partial_tile_fused_r2_eq_bare"]
                       and r["full_fused_r2_eq_tres"])
            del o2, o3
        r["gate_form"] = bool(lptr.gate_form(c["gate"], c["index"], M, H))
        r["inputs_unchanged"] = all(torch.equal(p, t) for p, t in zip(pristine, rb.inputs(c)))
        ok_all &= r["inputs_unchanged"]
        r["ok"] = ok_all
        passed &= ok_all
        c["ref"] = ref
        results[key] = r
        log(key, json.dumps(r))
        del attn_y, mlp_y, emu
        torch.cuda.synchronize()

    # Graph replay at 32848: fused r2, routed and tres inside CUDA graphs vs refs.
    s = torch.cuda.Stream()
    big = [pool[f"M32848_u8x3_{k}"] for k in range(3)]
    g_arms = {}
    for name, f in (("graph_fused_r2", lambda c: rb.run_fused_r(c, 2)),
                    ("graph_routed", served), ("graph_tres", run_tol)):
        g, outs = rb.capture([lambda c=c, f=f: f(c) for c in big], s)
        mism = 0
        for _ in range(10):
            for o in outs:
                o.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            mism += sum(not torch.equal(o.reshape(c["x"].shape), c["ref"])
                        for o, c in zip(outs, big))
        g_arms[name] = {"replays": 10, "mismatches": int(mism)}
        passed &= mism == 0
        del g, outs
    log("graph replay at 32848:", json.dumps(g_arms))

    # Kill switches and route census (serves / raster_for by M range).
    probe = (4176, 8191, 8192, 16384, 16464, 24575, 24576, 32848, 65536)

    def clear():
        lptr.raster_enabled.cache_clear()
        if hasattr(lptr, "large_m_enabled"):
            lptr.large_m_enabled.cache_clear()

    def census():
        return {M: [bool(lptr.serves(M)), int(lptr.raster_for(M))] for M in probe}

    kill = {"default": census()}
    c = pool["M32848_u8x3_0"]
    for env in ("TRTLLM_FLUX3_PAIR_RESIDUAL_LARGE_M", "TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_RASTER"):
        os.environ[env] = "0"
        clear()
        cen = census()
        kill[env + "=0"] = cen
        gf = bool(lptr.gate_form(c["gate"], c["index"], 32848, H))
        e0 = epoch()
        out = served(c).reshape(c["x"].shape)
        torch.cuda.synchronize()
        kill[env + "=0_gate_form_32848"] = gf
        kill[env + "=0_routed_is_tres_epoch_unchanged"] = epoch() == e0
        kill[env + "=0_routed_eq_ref"] = bool(torch.equal(out, c["ref"]))
        ok = ((not gf) and kill[env + "=0_routed_eq_ref"] and epoch() == e0
              and all(not v[0] for M, v in cen.items() if M >= 24576)
              and all(v[0] == (0 < M) for M, v in cen.items() if M < 24576))
        if env.endswith("LARGE_M"):
            ok = ok and all(v[1] == kill["default"][M][1] for M, v in cen.items())
        kill[env + "=0_ok"] = ok
        passed &= ok
        del out, os.environ[env]
        clear()
    kill["restored"] = census()
    log("kill switches:", json.dumps(kill))

    # Dynamo: gate_form/serves trace in a fullgraph compile (the blocks are torch.compiled).
    try:
        def fn_d(x, gate, index):
            if lptr.gate_form(gate, index, 32848, H):
                return x * 2
            return x * 3

        cf = torch.compile(fn_d, fullgraph=True, dynamic=False)
        xs = torch.ones(4, device="cuda")
        y = cf(xs, c["gate"], c["index"])
        dyn = {"traced": True, "compiled_value": float(y[0]),
               "eager_value": float(fn_d(xs, c["gate"], c["index"])[0])}
        dyn["ok"] = dyn["compiled_value"] == dyn["eager_value"]
    except Exception as e:  # noqa: BLE001
        dyn = {"traced": False, "error": repr(e)[:600], "ok": False}
    passed &= dyn["ok"]
    log("dynamo:", json.dumps(dyn))

    # Repeat sequence: t2i graph and i2i graph (routed), eager 4176 after each, shared workspace.
    t2i_seq = ([pool["M16464_select_0"]] * 27 + [pool["M16384_row_0"]] * 4 +
               [pool["M16384_select_slice_0"]])
    i2i_seq = ([big[i % 3] for i in range(27)] + [pool["M16384_row_0"]] * 9 +
               [pool["M16384_u8x3_slice_0"]])
    small = pool["M4176_select_0"]
    seqs = {"t2i": t2i_seq, "i2i": i2i_seq}
    uniq = {id(c): c for c in t2i_seq + i2i_seq + [small]}
    pristine_all = {k: [t.clone() for t in rb.inputs(c)] for k, c in uniq.items()}
    graphs, outs = {}, {}
    for name, sq in seqs.items():
        graphs[name], outs[name] = rb.capture([lambda c=c: served(c) for c in sq], s)
    rounds = int(os.environ.get("ROUNDS", "50"))
    mism_ref = mism_it0 = checked = ep_bad = 0
    it0 = {}
    n_fused = {n: sum(bool(lptr.gate_form(c["gate"], c["index"], c["M"], H)) for c in sq)
               for n, sq in seqs.items()}
    for r_ in range(rounds):
        for name in ("t2i", "i2i", "t2i"):
            for o in outs[name]:
                o.fill_(float("nan"))
            e0 = epoch()
            graphs[name].replay()
            torch.cuda.synchronize()
            ep_bad += (epoch() - e0) != n_fused[name]
            for j, (o, c) in enumerate(zip(outs[name], seqs[name])):
                checked += 1
                o2 = o.reshape(c["x"].shape)
                mism_ref += not bool(torch.equal(o2, c["ref"]))
                if (name, j) not in it0:
                    it0[(name, j)] = o2.clone()
                mism_it0 += not bool(torch.equal(o2, it0[(name, j)]))
            e0 = epoch()
            so = served(small).reshape(small["x"].shape)
            torch.cuda.synchronize()
            ep_bad += (epoch() - e0) != 1
            checked += 1
            mism_ref += not bool(torch.equal(so, small["ref"]))
            del so
        if r_ % 10 == 0:
            log(f"  round {r_}: mismatches ref {mism_ref} it0 {mism_it0} epoch_bad {ep_bad}")
    inputs_final = all(all(torch.equal(p, t) for p, t in zip(pristine_all[k], rb.inputs(c)))
                       for k, c in uniq.items())
    rep = {"rounds": rounds, "order": "t2i->i2i->t2i per round (+ eager M=4176 after each graph)",
           "outputs_checked": checked, "mismatches_vs_kill_switch_route_ref": mism_ref,
           "mismatches_vs_iter0": mism_it0, "epoch_step_bad": ep_bad,
           "fused_calls_per_graph": n_fused, "inputs_unchanged": inputs_final}
    passed &= mism_ref == 0 and mism_it0 == 0 and ep_bad == 0 and inputs_final
    log("repeat:", json.dumps(rep))
    print(json.dumps({
        "passed": bool(passed),
        "compared_against": "main's bare LPT pair (flux3_sm120_nvfp4_out_pair_prequant) + compiled "
                            "emulate_precision_casts residual (per-token gate = table[index]), "
                            "branch 3's flux3_tres tail (main's served 32848 route), and an f32 "
                            "emulation of pair + table residual",
        "per_case": results, "graph_32848": g_arms, "kill_switch": kill, "dynamo": dyn,
        "repeat_sequence": rep,
    }))


if __name__ == "__main__":
    {"bench": bench, "table": table, "check": check}[sys.argv[1]]()
