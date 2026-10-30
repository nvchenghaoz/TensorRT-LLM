"""Idea 89 served route: v2w (FLUX3_EPI_V2=2) = EPI_VEC's MMA permutation + lane-local epilogue on
gate_up weights row-permuted at preparation (flux3_swiglu_fp4_cutlass.maybe_permute_for_v2w,
called from primsts._prepare_nvfp4; owner flag _flux3_v2w routes the module).

  check : untimed. Per M in 16464 (80-row tail), 16384, 32848, eager 4176: reference = main's
          _dyn_ev module on the ORIGINAL weights (0xFF-poisoned whole padded out/out_sf). The
          candidate runs the tree's route: maybe_permute_for_v2w -> PreparedNVFP4 owner ->
          (a) swc.swiglu_packed(module=None) and (b) torch.ops.trtllm.flux3_primsts_nvfp4_swiglu_packed;
          warmup 2 + 5 iterations vs reference and vs iteration 0, inputs unchanged; CuTe fallback
          (CUTLASS disabled) on permuted weights; unpermute round trip; graph t2i -> i2i -> t2i
          -> i2i -> t2i (16464+16384 / 32848+16384) with eager 4176 between, poisoned before
          every replay; env TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2=0/1/unset/bogus in subprocesses.
  pair  : evidence. The tree's served route at 16464 / 16384 / 32848: main = get_module() on
          original weights; branch = maybe_permute_for_v2w + its module. us/call median of
          interleaved graph replays (5 calls/graph), A/A arm in the same process.
"""

import json
import os
import random
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
from bench import BASE, KEYS, N, SERVED_MS, call, capture, git_log, log, make_case, stats  # noqa: E402

swc = bench.swc


def tree_route(c):
    """(case, module, permuted) the tree serves for this case's weights."""
    if hasattr(swc, "maybe_permute_for_v2w"):
        w, s, permuted = swc.maybe_permute_for_v2w(c["b"], c["sfb"])
        return dict(c, b=w, sfb=s), swc.get_module(permuted), permuted
    return c, swc.get_module(), False


def check():
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

    gl = git_log()
    bench.prebuild([("_dyn_ev", BASE), ("_dyn_ev2w", BASE + ("-DFLUX3_EPI_V2=2",))])
    ref_mod = swc.get_module_dynamic(True, False)
    assert swc.v2w_prepare_enabled(), "check runs with the default env (EPI_V2 on)"
    passed, results, cases, owners = True, {}, {}, {}
    WARM, ITERS = 2, 5
    for M in SERVED_MS + (4176,):
        c = make_case(M, seed=M + 11)
        s_x = float(c["alpha"].item())  # alpha = s_x * s_w; the op takes the two floats
        bufs = swc.allocate_outputs(M, N, c["a"].device)
        ref = [t.clone() for t in call(c, ref_mod, bufs)]
        torch.cuda.synchronize()
        w2, s2, permuted = swc.maybe_permute_for_v2w(c["b"], c["sfb"])
        owner = PreparedNVFP4(w2, s2, 1.0, s_x, swiglu=True)
        owner._flux3_v2w = permuted
        owners[M] = owner
        cw = dict(c, b=owner.weight, sfb=owner.scale)
        cases[M] = (c, cw, ref)
        pristine = {k: cw[k].clone() for k in KEYS}
        row = {"permuted": permuted, "weight_is_v2w": swc.weight_is_v2w(owner.weight),
               "module_is_v2w": swc.get_module(swc.weight_is_v2w(owner.weight))
               is swc.get_module_dynamic(True, True)}
        uw, us = swc.unpermute_weight_v2w(owner.weight, owner.scale)
        row["unpermute_roundtrip"] = bool(torch.equal(uw.view(torch.uint8), c["b"].view(torch.uint8))
                                          and torch.equal(us.view(torch.uint8).reshape(-1),
                                                          c["sfb"].view(torch.uint8).reshape(-1)))

        def via_swc():
            return call(cw, None, bufs)

        def via_op():
            p, s = torch.ops.trtllm.flux3_primsts_nvfp4_swiglu_packed(
                cw["a"], cw["b"], cw["sfa"], cw["sfb"], 1.0, s_x, cw["inv"])
            return p, s

        for name, fn, full in (("swc_route", via_swc, True), ("custom_op", via_op, False)):
            for _ in range(WARM):
                fn()
            first, n_ref, n_first = None, 0, 0
            for _ in range(ITERS):
                if full:
                    out = [t.clone() for t in fn()]
                else:
                    p, s = fn()
                    out = [p.clone(), s.clone()]
                torch.cuda.synchronize()
                r = ref if full else [ref[0][:M], ref[1]]
                if first is None:
                    first = out
                n_ref += all(torch.equal(x, y) for x, y in zip(out, r))
                n_first += all(torch.equal(x, y) for x, y in zip(out, first))
            row[name] = {"bitwise_vs_main_dyn_ev": f"{n_ref}/{ITERS}", "vs_call0": f"{n_first}/{ITERS}"}
            passed &= n_ref == ITERS and n_first == ITERS
        # CuTe fallback with the permuted weights (CUTLASS route disabled for this call).
        orig_enabled = swc.enabled
        swc.enabled = lambda: False
        try:
            p, s = via_op()
            torch.cuda.synchronize()
            rows_p = ((M + 127) // 128) * 128
            idx = swc._sf_index(rows_p, N // 32, p.device)[:M].reshape(-1)
            s_flat, r_flat = s.view(torch.uint8).reshape(-1), ref[1].view(torch.uint8).reshape(-1)
            row["cute_fallback_bitwise"] = bool(
                torch.equal(p[:M].reshape(-1), ref[0][:M].reshape(-1))
                and s_flat.numel() >= int(idx.max()) + 1
                and torch.equal(s_flat[idx], r_flat[idx]))
        except Exception as e:  # noqa: BLE001
            row["cute_fallback_bitwise"] = f"n/a: {type(e).__name__}: {str(e)[:160]}"
        finally:
            swc.enabled = orig_enabled
        row["inputs_unchanged"] = all(torch.equal(pristine[k], cw[k]) for k in KEYS)
        passed &= (permuted and row["weight_is_v2w"] and row["module_is_v2w"]
                   and row["unpermute_roundtrip"] and row["inputs_unchanged"]
                   and row["cute_fallback_bitwise"] in (True,) )
        results[f"M{M}"] = row
        log(f"M={M}: {json.dumps(row)}")

    # Serving-like graph sequence on the prepared (permuted) weights, module chosen by the flag.
    gbufs = {M: swc.allocate_outputs(M, N, cases[M][0]["a"].device) for M in cases}
    pristine = {M: {k: cases[M][1][k].clone() for k in KEYS} for M in cases}

    def run_graph_fn(Ms):
        return lambda: [call(cases[M][1], None, gbufs[M], poison=False) for M in Ms]

    graphs = {"t2i": (capture(run_graph_fn((16464, 16384))), (16464, 16384)),
              "i2i": (capture(run_graph_fn((32848, 16384))), (32848, 16384))}
    # >= 50 rounds after warmup of t2i -> i2i (then the next round's t2i), eager 4176 between,
    # outputs 0xFF-poisoned before every call, each output bitwise vs round 0 AND vs the
    # kill-switch arm (main's _dyn_ev on the original weights); pristine inputs checked and
    # restored every round.
    rounds = int(os.environ.get("EPI_V2_SEQ_ROUNDS", "50"))
    for name in ("t2i", "i2i"):  # warmup
        graphs[name][0].replay()
    torch.cuda.synchronize()
    mism, mism_r0, n_cmp, first = 0, 0, 0, {}
    for rnd in range(rounds):
        for name in ("t2i", "i2i"):
            g, Ms = graphs[name]
            for M in Ms:
                gbufs[M][0].fill_(0xFF)
                gbufs[M][1].fill_(0xFF)
            g.replay()
            torch.cuda.synchronize()
            for M in Ms:
                n_cmp += 1
                key = (name, M)
                if key not in first:
                    first[key] = [t.clone() for t in gbufs[M]]
                mism += 0 if all(torch.equal(x, y) for x, y in zip(gbufs[M], cases[M][2])) else 1
                mism_r0 += 0 if all(torch.equal(x, y) for x, y in zip(gbufs[M], first[key])) else 1
            e = call(cases[4176][1], None, gbufs[4176])
            torch.cuda.synchronize()
            n_cmp += 1
            mism += 0 if all(torch.equal(x, y) for x, y in zip(e, cases[4176][2])) else 1
        for M in cases:
            if not all(torch.equal(pristine[M][k], cases[M][1][k]) for k in KEYS):
                mism += 1
                log(f"inputs changed at round {rnd} M={M}")
                for k in KEYS:
                    cases[M][1][k].copy_(pristine[M][k])
        if rnd % 10 == 0:
            log(f"sequence round {rnd}: comparisons {n_cmp} mismatches {mism} vs-round0 {mism_r0}")
    del graphs
    seq = ["t2i", "i2i"] * rounds
    results["graph_sequence"] = {"rounds": rounds, "order": "t2i(16464+16384) -> eager 4176 -> "
                                 "i2i(32848+16384) -> eager 4176, repeated",
                                 "comparisons": n_cmp, "mismatches_vs_kill_switch_ref": mism,
                                 "mismatches_vs_round0": mism_r0}
    mism += mism_r0
    passed &= mism == 0
    log(f"graph sequence: {results['graph_sequence']}")

    env_rows = {}
    code = ("import json,torch,tensorrt_llm\n"
            "from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as s\n"
            "try:\n"
            " w=torch.zeros(18432,1536,dtype=torch.uint8,device='cuda'); sf=torch.zeros(18432*192,dtype=torch.uint8,device='cuda')\n"
            " _,_,p=s.maybe_permute_for_v2w(w,sf)\n"
            " m=s.get_module(p)\n"
            " names={n: s._build(*a) is m for n,a in {'dyn_ev':('_dyn_ev',('-DFLUX3_TILESCHED=1','-DFLUX3_EPI_VEC=1')),"
            "'dyn_ev2w':('_dyn_ev2w',('-DFLUX3_TILESCHED=1','-DFLUX3_EPI_VEC=1','-DFLUX3_EPI_V2=2'))}.items()}\n"
            " print(json.dumps({'permuted': p, 'module': [n for n,v in names.items() if v]}))\n"
            "except ValueError as e:\n print(json.dumps({'error': str(e)[:120]}))\n")
    for v2 in ("0", "1", None, "bogus"):
        env = dict(os.environ)
        for k in ("TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2", "TRTLLM_FLUX3_SM120_SWIGLU_EPI_VEC",
                  "TRTLLM_FLUX3_SM120_SWIGLU_SCHED", "TRTLLM_FLUX3_SM120_SWIGLU_CUTLASS"):
            env.pop(k, None)
        if v2 is not None:
            env["TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2"] = v2
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        line = (r.stdout.strip().splitlines() or ["{}"])[-1]
        env_rows[str(v2)] = json.loads(line) if line.startswith("{") else {"raw": r.stderr[-300:]}
        log(f"env EPI_V2={v2}: {env_rows[str(v2)]}")
    env_ok = (env_rows["0"] == {"permuted": False, "module": ["dyn_ev"]}
              and env_rows["1"] == {"permuted": True, "module": ["dyn_ev2w"]}
              and env_rows["None"] == {"permuted": True, "module": ["dyn_ev2w"]}
              and "error" in env_rows["bogus"])
    results["env_selection"] = {"rows": env_rows, "ok": env_ok}
    passed &= env_ok
    ev = bench.resources(swc.gen_spec("_dyn_ev", BASE))
    results["dyn_ev_sass_sha256"] = ev.get("sass_sha256")
    print(json.dumps({"passed": bool(passed), "git": gl,
                      "reference": "main's served _dyn_ev module (same name/flags as main) on the "
                                   "original weights, whole padded out/out_sf after 0xFF poison, "
                                   "bitwise; candidate = this tree's prepared route",
                      "results": results}))


def pair():
    gl = git_log()
    ev = None
    batch = {}
    for M in SERVED_MS:
        c = make_case(M)
        cw, mod, permuted = tree_route(c)
        if ev is None:
            ev = bench.resources(swc.gen_spec("_dyn_ev", BASE))
            log(f"route permuted={permuted}; _dyn_ev sass {ev.get('sass_sha256')} REG {ev.get('REG')}")
        graphs, keep, t = {}, [], {}
        for arm in ("route", "route_aa"):
            b2 = swc.allocate_outputs(M, N, c["a"].device)
            keep.append(b2)
            graphs[arm] = capture(lambda b2=b2: [call(cw, mod, b2, poison=False)
                                                 for _ in range(bench.CALLS)])
            t[arm] = []
        rng = random.Random(M)
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        for _ in range(bench.ROUNDS * 2):
            order = list(graphs)
            rng.shuffle(order)
            for arm in order:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[arm].replay()
                e1.record()
                e1.synchronize()
                t[arm].append(e0.elapsed_time(e1) * 1000.0 / bench.CALLS)
        del graphs
        s, s2 = stats(t["route"]), stats(t["route_aa"])
        log(f"M={M}: {s} A/A {s2} clocks {bench.clocks()}")
        batch[str(M)] = {"kernel_ms": round(s["median_us"] / 1000.0, 5), "noise_floor": s["noise"],
                         "aa_kernel_ms": round(s2["median_us"] / 1000.0, 5)}
        del c, cw
        torch.cuda.empty_cache()
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "git": gl,
                      "dyn_ev_sass_sha256": ev.get("sass_sha256") if ev else None}))


if __name__ == "__main__":
    {"check": check, "pair": pair}[sys.argv[1]]()
