"""Idea 17 repeated-execution check (plan 366 recipe), real checkpoint weights, served Ms.

State shared across calls: the one LPT-pair workspace per device (args + atomic tile counter),
the QKV3 slot workspaces (CLC slot 0) and the FP8R workspace slots 0/1. This walks it the way
serving does: two step-shaped CUDA graphs plus an eager main-route M 4176 call between them.

  t2i graph: 28 x (QKV3 + pair) at M 16464 (served t2i single M, partial tiles; weights of
             single_blocks.0 / .13 alternating) + 4 x (QKV3 + pair) at M 16384 (image.1 / .2)
             + image.0's 4 FP8R GEMMs at M 16384.
  i2i graph: the same with the singles at M 32848 (served i2i single M, partial tiles).
  eager    : QKV3 + pair at M 4176 (single_blocks.0), must stay on main's 128-tile route.
Sequence: W warm rounds, then ROUNDS rounds of t2i -> eager4176 -> i2i -> eager4176 (so
t2i -> i2i -> t2i ... with the 4176 call between graphs at a fixed cadence). Timed first; pristine
copies of inputs, weights and workspaces are taken before it and restored after it. Then the
UNTIMED identical sequence: every output NaN-filled (0xFFFF) before every replay/call, compared
bitwise with (a) main's modules called directly on the same inputs (gemm_qkv3_pingpong with main's
raster/swizzle, lpt.module().lpt_pair with sched_mode on its own workspace, FP8R arm 1) and (b)
iteration 0 of the same case; tile counter checked after every graph replay (its last pair call)
and every eager pair call; inputs/weights vs pristine before the reset and after every round.
Last stdout line: {"passed": ...}.
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check as ck  # noqa: E402

ROUNDS = int(os.environ.get("REPEAT_ROUNDS", "50"))
WARM = 5
H = 3072
M_T2I, M_I2I, M_MODE, M_EAGER = 16464, 32848, 16384, 4176
SINGLES = ["single_blocks.0", "single_blocks.13"]
MODES = ["content_mode_blocks.image.1", "content_mode_blocks.image.2"]


def log(*a):
    print(*a, flush=True)


def bits(t):
    return t.view(torch.int16)


def make_set(h, blk, m, g):
    inp = ck.block_inputs(h, blk, m, g)
    a, w, sa, sw, al = inp["qkv"]
    d, wd, dsf, swd, ald, ao, wo, aosf, swo, alo = inp["pair"]
    return {
        "name": f"{blk}/M{m}", "m": m, "qkv": (a, w, sa, sw, al),
        "pair": (d, wd, dsf, swd, ald, ao, wo, aosf, swo, alo),
        "qkv_out": torch.empty(3 * m * H, device="cuda", dtype=torch.bfloat16),
        "down_out": torch.empty(m, H, device="cuda", dtype=torch.bfloat16),
        "attn_out": torch.empty(m, H, device="cuda", dtype=torch.bfloat16),
    }


def fp8r_set(h, g, m):
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import quantize_fp8r_weight

    blk = "content_mode_blocks.image.0"
    ws = {
        "qkv": torch.cat([h.get_tensor(f"{blk}.{p}_proj.weight") for p in "qkv"], 0),
        "gate_up": h.get_tensor(f"{blk}.mlp_in.weight"),
        "down": h.get_tensor(f"{blk}.mlp_out.weight"),
        "attn_out": h.get_tensor(f"{blk}.attn_out.weight"),
    }
    res = {}
    for name, w in ws.items():
        w = w.cuda().to(torch.bfloat16)
        wd, wsc = quantize_fp8r_weight(w)
        x = (torch.randn((m, w.shape[1]), device="cuda", generator=g) * 1.5).to(torch.bfloat16)
        x[:, :8] *= 20
        a, sa = quantize_fp8r(x)
        res[name] = (a.contiguous(), wd, sa.float().reshape(-1).contiguous(), wsc,
                     torch.empty((m, w.shape[0]), device="cuda", dtype=torch.bfloat16))
    return res


def outs_of(s):
    return [s["qkv_out"], s["down_out"], s["attn_out"]]


def inputs_of(sets, f8):
    out = []
    for s in sets:
        out += list(s["qkv"]) + list(s["pair"])
    for v in f8.values():
        out += list(v[:4])
    return out


def main():
    from safetensors import safe_open

    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_large_m as lm
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    t0 = time.time()
    ckpt = sys.argv[1]
    dev = torch.device("cuda", torch.cuda.current_device())
    fp.sm120_linear_preload(dev)
    strided.preload(dev)
    lpt.preload(dev)
    fg.preload(dev)
    routes = sorted(lm._enabled_routes())
    log(f"routes {routes} min M {lm.LARGE_M_MIN} sched_mode {lpt.sched_mode()} "
        f"({time.time() - t0:.0f}s)")
    g = torch.Generator(device="cuda")
    g.manual_seed(1717)
    with safe_open(str(ck.ckpt_file(ckpt)), "pt", device="cpu") as h:
        t2i_s = [make_set(h, b, M_T2I, g) for b in SINGLES]
        i2i_s = [make_set(h, b, M_I2I, g) for b in SINGLES]
        mode_s = [make_set(h, b, M_MODE, g) for b in MODES]
        eager = make_set(h, SINGLES[0], M_EAGER, g)
        f8 = fp8r_set(h, g, M_MODE)
    all_sets = t2i_s + i2i_s + mode_s + [eager]
    log(f"inputs built ({time.time() - t0:.0f}s)")

    # References: main's modules directly, own workspaces.
    qws = torch.zeros(strided._WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    pws = torch.zeros(lpt._WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    main_qkv = strided.qkv_pingpong_module()
    main_pair = lpt.module()
    refs = {}
    for s in all_sets:
        a, w, sa, sw, al = s["qkv"]
        rq = torch.empty_like(s["qkv_out"])
        main_qkv.gemm_qkv3_pingpong(a, w, sa, sw, al, rq, qws, int(strided.QKV3_RASTER),
                                    int(strided.QKV3_SWIZZLE))
        d, wd, dsf, swd, ald, ao, wo, aosf, swo, alo = s["pair"]
        rd = torch.empty_like(s["down_out"])
        ra = torch.empty_like(s["attn_out"])
        main_pair.lpt_pair(d, wd, dsf, swd, ald, rd, ao, wo, aosf, swo, alo, ra, pws,
                           lpt.sched_mode(), 0)
        refs[id(s["qkv_out"])], refs[id(s["down_out"])], refs[id(s["attn_out"])] = rq, rd, ra
    for v in f8.values():
        a, w, sa, sw, out = v
        refs[id(out)] = fg.gemm(a, w, sa, sw, arm=1, slot=1)
    torch.cuda.synchronize()
    log(f"references: {len(refs)} buffers ({time.time() - t0:.0f}s)")

    def qkv3(s):
        a, w, sa, sw, al = s["qkv"]
        strided.gemm_qkv3(a, w, sa, sw, al, s["qkv_out"])

    def pair(s):
        d, wd, dsf, swd, ald, ao, wo, aosf, swo, alo = s["pair"]
        lpt.run(d, wd, dsf, swd, ald, s["down_out"], ao, wo, aosf, swo, alo, s["attn_out"])

    def step_calls(singles):
        calls = []
        for i in range(28):
            s = singles[i % 2]
            calls += [lambda s=s: qkv3(s), lambda s=s: pair(s)]
        for i in range(4):
            s = mode_s[i % 2]
            calls += [lambda s=s: qkv3(s), lambda s=s: pair(s)]
        for v in f8.values():
            calls.append(lambda v=v: fg.gemm(v[0], v[1], v[2], v[3], out=v[4]))
        return calls

    def capture(calls):
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for f in calls:
                f()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for f in calls:
                f()
        torch.cuda.synchronize()
        return gr

    graphs = {"t2i-2048": capture(step_calls(t2i_s)), "i2i-2048": capture(step_calls(i2i_s))}
    outs = {"t2i-2048": [o for s in t2i_s + mode_s for o in outs_of(s)] + [v[4] for v in f8.values()],
            "i2i-2048": [o for s in i2i_s + mode_s for o in outs_of(s)] + [v[4] for v in f8.values()],
            "eager-4176": outs_of(eager)}
    log(f"captured ({time.time() - t0:.0f}s)")

    def run_eager():
        qkv3(eager)
        pair(eager)

    def tiles(m):
        tm = 256 if lm.use("pair", m) else 128
        return 2 * ((m + tm - 1) // tm) * (H // 128)

    expect_counter = {"t2i-2048": tiles(M_MODE), "i2i-2048": tiles(M_MODE), "eager-4176": tiles(M_EAGER)}
    route = {
        "t2i-2048": {"single_M": M_T2I, "qkv3": "t256" if lm.use("qkv3", M_T2I) else "main",
                     "pair": "t256" if lm.use("pair", M_T2I) else "main"},
        "i2i-2048": {"single_M": M_I2I, "qkv3": "t256" if lm.use("qkv3", M_I2I) else "main",
                     "pair": "t256" if lm.use("pair", M_I2I) else "main"},
        "mode-16384": {"qkv3": "t256" if lm.use("qkv3", M_MODE) else "main",
                       "pair": "t256" if lm.use("pair", M_MODE) else "main",
                       "fp8r_arms": {k: (lm.fp8r_arm(v[1].shape[0]) if lm.use("fp8r", M_MODE)
                                         else fg.ARM) for k, v in f8.items()}},
        "eager-4176": {"qkv3": "t256" if lm.use("qkv3", M_EAGER) else "main",
                       "pair": "t256" if lm.use("pair", M_EAGER) else "main"},
    }
    seq_round = ["t2i-2048", "eager-4176", "i2i-2048", "eager-4176"]

    def do(k):
        if k == "eager-4176":
            run_eager()
        else:
            graphs[k].replay()

    wsp = [lpt.workspace(dev)] + [strided.workspace(dev, i) for i in range(3)] + \
          [fg.workspace(dev, i) for i in range(2)]
    ins = inputs_of(all_sets, f8)
    pristine_in = [t.clone() for t in ins]
    pristine_ws = [t.clone() for t in wsp]

    # Timed sequence.
    times = {k: [] for k in seq_round}
    for r in range(WARM + ROUNDS):
        for k in seq_round:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            do(k)
            e1.record()
            e1.synchronize()
            if r >= WARM:
                times[k].append(e0.elapsed_time(e1))
        if r % 10 == 0:
            log(f"timed round {r}")
    fails = []
    inputs_ok_timed = all(torch.equal(a, b) for a, b in zip(pristine_in, ins))
    if not inputs_ok_timed:
        fails.append("inputs/weights changed during timed sequence")
    # Reset from the pristine copy.
    for t, p in zip(wsp, pristine_ws):
        t.copy_(p)
    for t, p in zip(ins, pristine_in):
        t.copy_(p)
    torch.cuda.synchronize()

    # Untimed identical sequence.
    stat = {k: {"iterations": 0, "mismatch_vs_main": 0, "mismatch_vs_iter0": 0,
                "max_abs_diff": 0.0, "counter_ok": 0, "counter_checks": 0} for k in outs}
    first = {}
    inputs_ok = True
    for r in range(WARM + ROUNDS):
        for k in seq_round:
            for o in outs[k]:
                bits(o).fill_(-1)
            do(k)
            torch.cuda.synchronize()
            st = stat[k]
            st["iterations"] += 1
            bad_ref = [o for o in outs[k] if not torch.equal(bits(o), bits(refs[id(o)]))]
            if bad_ref:
                st["mismatch_vs_main"] += 1
                for o in bad_ref:
                    dd = (o.float() - refs[id(o)].float()).abs().nan_to_num(float("inf")).max()
                    st["max_abs_diff"] = max(st["max_abs_diff"], float(dd))
            if k not in first:
                first[k] = [o.clone() for o in outs[k]]
            if not all(torch.equal(bits(o), bits(f)) for o, f in zip(outs[k], first[k])):
                st["mismatch_vs_iter0"] += 1
            if lpt.sched_mode() >= 1:
                c = int(lpt.workspace(dev)[:4].view(torch.int32).item())
                st["counter_checks"] += 1
                st["counter_ok"] += c == expect_counter[k]
        if not all(torch.equal(a, b) for a, b in zip(pristine_in, ins)):
            inputs_ok = False
        if r % 10 == 0:
            log(f"untimed round {r}: " + " ".join(
                f"{k}={s['iterations'] - s['mismatch_vs_main']}/{s['iterations']}"
                for k, s in stat.items()))
    for k, st in stat.items():
        if st["mismatch_vs_main"] or st["mismatch_vs_iter0"]:
            fails.append(f"{k}: {st}")
        if st["counter_ok"] != st["counter_checks"]:
            fails.append(f"{k}: tile counter {st['counter_ok']}/{st['counter_checks']}")
    if not inputs_ok:
        fails.append("inputs/weights changed during untimed sequence")
    if route["eager-4176"] != {"qkv3": "main", "pair": "main"}:
        fails.append("M4176 took a large-M route")

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else None

    print(json.dumps({
        "passed": not fails, "fails": fails[:20], "routes_enabled": routes,
        "route_per_call": route, "per_case": stat, "expected_counter": expect_counter,
        "inputs_unchanged": inputs_ok and inputs_ok_timed,
        "sequence": f"{WARM} warm + {ROUNDS} rounds of " + " -> ".join(seq_round),
        "timed_median_ms": {k: med(v) for k, v in times.items()},
        "compared": "bitwise vs main's modules called directly (gemm_qkv3_pingpong main "
                    "raster/swizzle, lpt.module().lpt_pair sched_mode own workspace, FP8R arm 1) "
                    "and vs iteration 0; NaN-poisoned outputs; real checkpoint weights",
    }), flush=True)


if __name__ == "__main__":
    main()
