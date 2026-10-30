"""Idea 95 (branch 61) untimed correctness: t256 fused pair+residual at M >= 24576.

1. Routing: t256 at every M >= 8192; TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_T256_LARGE=0 -> t256 only at
   8192 <= M < 24576 (main's table); T256=0 -> 128 everywhere. raster_for unchanged (2 at >= 8192).
2. Per shape (32848 u8x3 incl. the 80-row last tile, 24576 u8x3, 24577 u8x3, 16464 select, 16384
   row, 4176 select): routed output over WARMUP + ITERS calls bitwise vs pair + Inductor residual
   (pair + per-token table gate), vs the f32 emulation, and vs the 128 instance at main's raster
   (fused-128 r2 at M >= 24576); NaN-poisoned mlp_y scratch via run_raw; inputs unchanged.
3. >= 5000 consecutive routed 32848 calls: a 50-call graph replayed 100x, outputs NaN-poisoned
   before every replay, every output bitwise vs the reference; epoch +50 per replay.
4. Kill switch T256_LARGE=0: the routed op is byte for byte the 128 instance (main) at M >= 24576.
5. Repeat sequence (NaN-poisoned outputs, pristine inputs restored before each round): t2i and
   i2i step graphs alternating t2i->i2i->i2i->t2i..., one warmup + ROUNDS rounds, eager M=4176
   call after every graph; vs round 0, vs kill-switch graphs (T256_LARGE=0) and vs pair +
   Inductor; epoch advance == launches and tile counter >= last launch's tiles.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "idea55_t256"))
import bench as b55  # noqa: E402

H = b55.H
log = b55.log
WARMUP, ITERS = 3, 20
KILL = "TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_T256_LARGE"


def set_env(lptr, name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    lptr.t256_enabled.cache_clear()
    if hasattr(lptr, "t256_large_enabled"):
        lptr.t256_large_enabled.cache_clear()


def main():
    lptr = b55.preload()
    dev = torch.device("cuda", torch.cuda.current_device())
    ws = lptr.workspace(dev)
    passed = True
    results = {}

    def epoch():
        return int(ws[:256].view(torch.int32)[128 // 4].item())

    def counter():
        return int(ws[:4].view(torch.int32)[0].item())

    # 1. Routing tables.
    probe = (1, 4176, 8191, 8192, 16384, 16464, 24575, 24576, 24577, 32848, 33280, 65536)
    got = {M: b55.route_of(lptr, M) for M in probe}
    ok_r = all(got[M] == ("t256" if M >= 8192 else "128") for M in probe)
    set_env(lptr, KILL, "0")
    got_k = {M: b55.route_of(lptr, M) for M in probe}
    ok_k = all(got_k[M] == ("t256" if 8192 <= M < 24576 else "128") for M in probe)
    set_env(lptr, KILL, None)
    set_env(lptr, "TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_T256", "0")
    got_0 = {M: b55.route_of(lptr, M) for M in probe}
    ok_0 = all(v == "128" for v in got_0.values())
    set_env(lptr, "TRTLLM_FLUX3_SM120_PAIR_RESIDUAL_T256", None)
    rasters = {M: lptr.raster_for(M) for M in probe}
    ok_ras = all(rasters[M] == (2 if M >= 8192 else 0) for M in probe)
    results["routing"] = {"served": got, "t256_large_0": got_k, "t256_0": got_0,
                          "raster_for": rasters, "ok": ok_r and ok_k and ok_0 and ok_ras}
    passed &= results["routing"]["ok"]
    log(f"routing: {results['routing']}")

    # 2. Per-shape bitwise.
    configs = [(32848, "u8x3", False), (24576, "u8x3", False), (24577, "u8x3", False),
               (16464, "select", False), (16384, "row", False), (16384, "row", True),
               (16384, "sel", False), (16384, "u8s", False), (4176, "select", False)]
    pool = {}
    for M, form, nfo0 in configs:
        c = b55.make_case(M, 9500 + M + len(form) + int(nfo0), form, nfo0)
        key = f"M{M}_{form}" + ("_nfo0" if nfo0 else "")
        pool[key] = c
        pristine = [t.clone() for t in b55.inputs(c)]
        attn_y, mlp_y = b55.run_pair(c)
        ref = b55.run_base(c)
        emu = b55.emulate_ref(c, attn_y, mlp_y)
        r = {"ref_vs_f32_emulation": bool(torch.equal(ref, emu))}
        ok_all = r["ref_vs_f32_emulation"]
        r128 = b55.run_forced(c, "128", 2 if M >= 8192 else 0).reshape(c["x"].shape).clone()
        r["f128_main_raster_vs_ref"] = bool(torch.equal(r128, ref))
        ok_all &= r["f128_main_raster_vs_ref"]
        for _ in range(WARMUP):
            b55.run_fused(c)
        torch.cuda.synchronize()
        n_ref = n_128 = n_tail = 0
        eps = set()
        for i in range(ITERS):
            e0 = epoch()
            out = b55.run_fused(c).reshape(c["x"].shape)
            torch.cuda.synchronize()
            eps.add(epoch() - e0)
            n_ref += bool(torch.equal(out, ref))
            n_128 += bool(torch.equal(out, r128))
            tail = M % 256 or 256
            n_tail += bool(torch.equal(out[:, -tail:], ref[:, -tail:])) and \
                not bool(torch.isnan(out).any())
            out.fill_(float("nan"))
            del out
        ok = n_ref == ITERS and n_128 == ITERS and n_tail == ITERS and eps == {1}
        r["routed"] = {"bitwise_vs_ref": f"{n_ref}/{ITERS}", "bitwise_vs_f128": f"{n_128}/{ITERS}",
                       "last_tile_rows": M % 256 or 256, "last_tile_bitwise_no_nan":
                       f"{n_tail}/{ITERS}", "epoch_step": sorted(eps), "ok": ok}
        ok_all &= ok
        if not c["nfo0"]:
            down_alpha = torch.tensor([c["down_ad"].input_global_scale *
                                       c["down_ad"].weight_global_scale], device=dev,
                                      dtype=torch.float32)
            attn_alpha = torch.tensor([c["out_ad"].input_global_scale *
                                       c["out_ad"].weight_global_scale], device=dev,
                                      dtype=torch.float32)
            mlp_y_s = torch.empty((M, H), device=dev, dtype=torch.bfloat16)
            n_ok = 0
            for i in range(WARMUP + ITERS):
                mlp_y_s.fill_(float("nan"))
                out = lptr.run_raw(
                    c["down_in"], c["down_ad"].weight, c["down_sf"], c["down_ad"].scale,
                    down_alpha, c["attn_q"], c["out_ad"].weight, c["attn_sf"],
                    c["out_ad"].scale, attn_alpha, c["x"], c["gate"], c["index"],
                    mlp_y=mlp_y_s).reshape(c["x"].shape)
                torch.cuda.synchronize()
                n_ok += bool(torch.equal(out, ref))
                del out
            r["run_raw_nan_mlp_y"] = f"{n_ok}/{WARMUP + ITERS}"
            ok_all &= n_ok == WARMUP + ITERS
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, b55.inputs(c)))
        r["inputs_unchanged"] = inputs_ok
        ok_all &= inputs_ok
        r["ok"] = bool(ok_all)
        passed &= ok_all
        c["ref"], c["r128"] = ref, r128
        results[key] = r
        log(key, json.dumps(r))
        del attn_y, mlp_y, emu
        torch.cuda.synchronize()

    s = torch.cuda.Stream()

    # 3. >= 5000 consecutive routed calls at 32848.
    big = pool["M32848_u8x3"]
    n_graph, n_rep = 50, int(os.environ.get("IDEA95_REPLAYS", "100"))
    g_big, o_big = b55.capture([lambda: b55.run_fused(big) for _ in range(n_graph)], s)
    bad = 0
    bad_ep = 0
    for rep in range(n_rep):
        for o in o_big:
            o.fill_(float("nan"))
        e0 = epoch()
        g_big.replay()
        torch.cuda.synchronize()
        bad_ep += epoch() - e0 != n_graph
        for o in o_big:
            bad += not torch.equal(o.reshape(big["x"].shape), big["ref"])
        if rep % 20 == 0:
            log(f"  consecutive replay {rep}: mismatches {bad} epoch_bad {bad_ep}")
    ok5k = bad == 0 and bad_ep == 0
    results["consecutive_32848"] = {"calls": n_graph * n_rep, "mismatches": bad,
                                    "epoch_bad_replays": bad_ep, "ok": ok5k}
    passed &= ok5k
    log(f"consecutive: {results['consecutive_32848']}")
    del g_big, o_big
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 4. Kill switch T256_LARGE=0 -> main's 128 instance, byte for byte.
    set_env(lptr, KILL, "0")
    kill_ok = True
    for key in ("M32848_u8x3", "M24576_u8x3", "M24577_u8x3", "M16464_select", "M16384_row",
                "M4176_select"):
        c = pool[key]
        out = b55.run_fused(c).reshape(c["x"].shape)
        torch.cuda.synchronize()
        want = c["r128"] if c["M"] >= 24576 or c["M"] < 8192 else c["ref"]
        kill_ok &= bool(torch.equal(out, want))
        del out
    set_env(lptr, KILL, None)
    results["kill_switch_t256_large_0"] = {"outputs_eq_main": kill_ok}
    passed &= kill_ok
    log(f"kill switch: {kill_ok}")

    # 5. Repeat sequence.
    rounds = int(os.environ.get("IDEA95_ROUNDS", "50"))
    t2i_seq = [pool["M16464_select"]] * 27 + [pool["M16384_row"]] * 4 + [pool["M16384_sel"]]
    i2i_seq = ([pool["M16384_row"]] * 4 + [pool["M16384_row_nfo0"]] * 5 + [pool["M16384_u8s"]]
               + [pool["M32848_u8x3"]] * 4)
    g_t2i, o_t2i = b55.capture([lambda c=c: b55.run_fused(c) for c in t2i_seq], s)
    g_i2i, o_i2i = b55.capture([lambda c=c: b55.run_fused(c) for c in i2i_seq], s)
    set_env(lptr, KILL, "0")
    g_t2i_k, o_t2i_k = b55.capture([lambda c=c: b55.run_fused(c) for c in t2i_seq], s)
    g_i2i_k, o_i2i_k = b55.capture([lambda c=c: b55.run_fused(c) for c in i2i_seq], s)
    set_env(lptr, KILL, None)
    small = pool["M4176_select"]
    pristine_all = {k: [t.clone() for t in b55.inputs(c)] for k, c in pool.items()}

    def tiles(M):
        bm = 256 if b55.route_of(lptr, M) == "t256" else 128
        return 2 * ((M + bm - 1) // bm) * (H // 128)

    kill = {}
    for name, g, outs in (("t2i", g_t2i_k, o_t2i_k), ("i2i", g_i2i_k, o_i2i_k)):
        for o in outs:
            o.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        kill[name] = [o.clone() for o in outs]
    kill_vs_ref = sum(not torch.equal(k.reshape(c["x"].shape), c["ref"])
                      for name, cs in (("t2i", t2i_seq), ("i2i", i2i_seq))
                      for k, c in zip(kill[name], cs))
    mism = {"vs_round0": 0, "vs_kill_switch": 0, "vs_pair_inductor": 0}
    state_bad = total = 0
    seq_log, round0, counters = [], {}, {}
    for rnd in range(-1, rounds):
        for k, c in pool.items():
            for p, t in zip(pristine_all[k], b55.inputs(c)):
                t.copy_(p)
        torch.cuda.synchronize()
        order = [("t2i", g_t2i, o_t2i, t2i_seq), ("i2i", g_i2i, o_i2i, i2i_seq)]
        if rnd % 2:
            order.reverse()
        bad_rnd = 0
        for name, g, outs, cs in order:
            for o in outs:
                o.fill_(float("nan"))
            e0 = epoch()
            g.replay()
            torch.cuda.synchronize()
            d_ep, cnt = epoch() - e0, counter()
            ok_state = d_ep == len(cs) and cnt >= tiles(cs[-1]["M"])
            counters.setdefault(name, set()).add(cnt)
            if rnd == 0:
                round0[name] = [o.clone() for o in outs]
            for i, (o, c) in enumerate(zip(outs, cs)):
                o = o.reshape(c["x"].shape)
                b_ref = not torch.equal(o, c["ref"])
                b_kill = not torch.equal(o, kill[name][i].reshape(c["x"].shape))
                b_r0 = rnd > 0 and not torch.equal(o, round0[name][i].reshape(c["x"].shape))
                mism["vs_pair_inductor"] += b_ref
                mism["vs_kill_switch"] += b_kill
                mism["vs_round0"] += b_r0
                bad_rnd += b_ref or b_kill or b_r0
            total += len(outs)
            e0 = epoch()
            o = b55.run_fused(small).reshape(small["x"].shape)
            torch.cuda.synchronize()
            d_ep_s, cnt_s = epoch() - e0, counter()
            ok_state &= d_ep_s == 1 and cnt_s >= tiles(small["M"])
            counters.setdefault("eager4176", set()).add(cnt_s)
            b_s = not torch.equal(o, small["ref"])
            mism["vs_pair_inductor"] += b_s
            bad_rnd += b_s
            total += 1
            del o
            state_bad += not ok_state
            seq_log.append(f"{name}:{bad_rnd}{'' if ok_state else '!state'}")
        if rnd % 10 == 0:
            log(f"  repeat round {rnd}: mismatches {mism} state_bad {state_bad}")
    inputs_ok = all(all(torch.equal(p, t) for p, t in zip(pristine_all[k], b55.inputs(c)))
                    for k, c in pool.items())
    rep_ok = sum(mism.values()) == 0 and kill_vs_ref == 0 and state_bad == 0 and inputs_ok
    results["repeat_sequence"] = {
        "rounds_after_warmup": rounds, "mismatches": mism, "kill_switch_vs_pair_inductor":
        kill_vs_ref, "compared": total, "state_checks_failed": state_bad,
        "tile_counter_values": {k: sorted(v) for k, v in counters.items()},
        "tile_counts_last_launch": {"t2i": tiles(t2i_seq[-1]["M"]), "i2i": tiles(i2i_seq[-1]["M"]),
                                    "eager4176": tiles(small["M"])},
        "inputs_unchanged": inputs_ok, "ok": rep_ok, "order": " ".join(seq_log)}
    passed &= rep_ok
    log(f"repeat sequence: {results['repeat_sequence']}")
    print(json.dumps({"passed": bool(passed), "compared_against":
                      "pair (flux3_sm120_nvfp4_out_pair_prequant) + Inductor emulate_precision_casts "
                      "residual, f32 emulation, the 128 instance at main's raster (fused-128 r2 at "
                      "M >= 24576), and T256_LARGE=0 kill-switch graphs", "results": results}))


if __name__ == "__main__":
    main()
