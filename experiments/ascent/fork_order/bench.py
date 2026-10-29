"""Idea 234 STEP 0(b)/2: block graph replay of the served mode-3 fork op, gate_up_first vs
fmha_first (TRTLLM_FLUX3_SIDE_GATE_UP_ORDER), synthetic tensors at served shapes (no model load).

Unit: one CUDA graph of one denoise step's NVFP4 block fronts, 28 x S4176 + 4 x S4096 (QKV op
-> trtllm::flux3_sm120_fp8_prequant_attention_side_gate_up -> down||attn_out pair -> residual),
built with experiments/ascent/idea161/step0.py's block_front_direct.

  table : arms S (main's sequential ops), A and A' (mode 3, gate_up_first: two captures = A/A),
          B and B' (mode 3, fmha_first), ROUNDS interleaved rounds behind a spin, ms/step median.
  pair  : evidence JSON for run_gpu_pair: mode 3 with the tree's default order (main: gate_up
          first; this branch: fmha_first). Delegates to step0.pair('m3').
  check : untimed bitwise check: attn / packed / scales / block output of both orders vs main's
          sequential ops and vs call 0 (2 warmups, 20 eager iterations), graph capture + 5
          replays, inputs unchanged; counters issued == joined, sync_fallback == 0.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "idea161"))
os.environ.setdefault("IDEA161_PATH", "direct")

import torch  # noqa: E402

import step0  # noqa: E402

fp = step0.fp
MAIN_AT_START = "1185e8f9"
ROUNDS = int(os.environ.get("FORK_ORDER_ROUNDS", "120"))
ORDER_ENV = "TRTLLM_FLUX3_SIDE_GATE_UP_ORDER"


def log(*a):
    print(*a, flush=True)


def probe():
    import tensorrt_llm

    def sh(*cmd):
        p = subprocess.run(list(cmd), capture_output=True, text=True)
        return p.returncode, p.stdout.strip()

    head = sh("git", "rev-parse", "HEAD")[1]
    rc = sh("git", "merge-base", "--is-ancestor", MAIN_AT_START, "HEAD")[0]
    log(f"PROBE HEAD={head} MAIN_AT_START={MAIN_AT_START} "
        f"tensorrt_llm={tensorrt_llm.__file__} merge_base_is_ancestor rc={rc} "
        f"has_order_switch={hasattr(fp, 'side_gate_up_order')}")
    return {"head": head, "ancestor_rc": rc}


def with_order(order, fn):
    old = os.environ.get(ORDER_ENV)
    if order is None:
        os.environ.pop(ORDER_ENV, None)
    else:
        os.environ[ORDER_ENV] = order
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop(ORDER_ENV, None)
        else:
            os.environ[ORDER_ENV] = old


ARMS = {"S": ("off", None), "A": ("m3", "gate_up_first"), "A'": ("m3", "gate_up_first"),
        "B": ("m3", "fmha_first"), "B'": ("m3", "fmha_first")}


def table():
    info = probe()
    cases = [(step0.make_case(4176, 1), 28), (step0.make_case(4096, 2), 4)]
    graphs = {}
    c0 = dict(fp.SIDE_GATE_UP_COUNTERS)
    for name, (arm, order) in ARMS.items():
        def step(arm=arm):
            outs = []
            for c, n in cases:
                for _ in range(n):
                    outs.append(step0.block_front(c, arm)[0])
            return outs[-1]

        graphs[name], _ = with_order(order, lambda: step0.capture(step))
        log(f"captured {name} ({arm}, {order})")
    counters = {k: fp.SIDE_GATE_UP_COUNTERS[k] - c0[k] for k in c0}
    log(f"capture counters {counters}; soak")
    t0 = time.time()
    while time.time() - t0 < 20.0:
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
    keys = list(graphs)
    times = {k: [] for k in keys}
    for r in range(ROUNDS):
        rot = r % len(keys)
        order = keys[rot:] + keys[:rot]
        if r % 2:
            order = order[::-1]
        evs = []
        for k in order:
            torch.cuda.synchronize()
            evs.append((k, *step0.time_graph(graphs[k])))
        torch.cuda.synchronize()
        for k, a, b in evs:
            times[k].append(a.elapsed_time(b))
        if r % 20 == 0:
            log(f"round {r}: " + " ".join(f"{k}={times[k][-1]:.3f}" for k in keys))
    row = {}
    for k in keys:
        m, noise = step0.stat(times[k])
        row[k] = {"ms_per_step": round(m, 4), "noise": round(noise, 4)}
    a = (row["A"]["ms_per_step"] + row["A'"]["ms_per_step"]) / 2
    b = (row["B"]["ms_per_step"] + row["B'"]["ms_per_step"]) / 2
    s = row["S"]["ms_per_step"]
    summary = {
        "A_mean": round(a, 4), "B_mean": round(b, 4), "S": s,
        "AA_spread_ms": round(abs(row["A"]["ms_per_step"] - row["A'"]["ms_per_step"]), 4),
        "BB_spread_ms": round(abs(row["B"]["ms_per_step"] - row["B'"]["ms_per_step"]), 4),
        "B_minus_A_ms_per_step": round(b - a, 4),
        "B_minus_A_us_per_pair": round((b - a) * 1e3 / 32, 2),
        "A_minus_S_us_per_pair": round((a - s) * 1e3 / 32, 2),
        "B_minus_S_us_per_pair": round((b - s) * 1e3 / 32, 2),
        "B_minus_A_ms_per_request_raw": round((b - a) * 16, 3),
    }
    print(json.dumps({"table": row, "summary": summary, "rounds": ROUNDS,
                      "unit": "ms per graph replay = 28 x S4176 + 4 x S4096 block fronts",
                      "capture_counters": counters, "probe": info,
                      "issued_eq_joined": counters["issued"] == counters["joined"],
                      "sync_fallback": counters["sync_fallback"]}), flush=True)


def check():
    info = probe()
    results, passed = {}, True
    c0 = dict(fp.SIDE_GATE_UP_COUNTERS)
    for S, seed in ((4176, 11), (4096, 12)):
        c = step0.make_case(S, seed)
        pristine = [t.clone() for t in step0.inputs(c)]
        ref = [t.clone() for t in step0.block_front(c, "off")]
        torch.cuda.synchronize()
        orders_out = {}
        for order in ("gate_up_first", "fmha_first"):
            def run():
                for p, t in zip(pristine, step0.inputs(c)):
                    t.copy_(p)
                for _ in range(2):
                    step0.block_front(c, "m3")
                n_ref = n_first = 0
                first = None
                iters = 20
                for _ in range(iters):
                    out = step0.block_front(c, "m3")
                    if first is None:
                        first = [t.clone() for t in out]
                    n_ref += all(torch.equal(x, y) for x, y in zip(out, ref))
                    n_first += all(torch.equal(x, y) for x, y in zip(out, first))
                inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, step0.inputs(c)))
                g, gout = step0.capture(lambda: step0.block_front(c, "m3"))
                g_ok = 0
                for _ in range(5):
                    for t in gout:
                        t.zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    g_ok += all(torch.equal(x, y) for x, y in zip(gout, ref))
                del g
                inputs_ok &= all(torch.equal(p, t) for p, t in zip(pristine, step0.inputs(c)))
                return first, {"eager_bitwise_vs_main_ops": f"{n_ref}/{iters}",
                               "eager_vs_call0": f"{n_first}/{iters}",
                               "graph_replay_bitwise_vs_main_ops": f"{g_ok}/5",
                               "inputs_unchanged": inputs_ok,
                               "ok": n_ref == iters and n_first == iters and g_ok == 5
                               and inputs_ok}

            first, r = with_order(order, run)
            orders_out[order] = first
            results[f"S{S}_{order}"] = r
            passed &= r["ok"]
            log(f"S={S} {order}: {json.dumps(r)}")
        same = all(torch.equal(x, y) for x, y in
                   zip(orders_out["gate_up_first"], orders_out["fmha_first"]))
        results[f"S{S}_fmha_first_vs_gate_up_first_bitwise"] = same
        passed &= same
    counters = {k: fp.SIDE_GATE_UP_COUNTERS[k] - c0[k] for k in c0}
    passed &= counters["issued"] == counters["joined"] > 0 and counters["sync_fallback"] == 0
    passed &= info["ancestor_rc"] == 0
    print(json.dumps({
        "passed": bool(passed),
        "reference": "main's sequential ops (visual_gen_fp8_attention_sm120_prequant then "
                     "flux3_nvfp4_swiglu_prepared) on pristine inputs; bitwise on block output, "
                     "attention output, packed gate_up bytes and scales",
        "results": results, "counters": counters, "probe": info,
    }), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    if cmd == "table":
        table()
    elif cmd == "pair":
        probe()
        step0.pair("m3")
    elif cmd == "check":
        check()
    else:
        raise SystemExit(f"unknown command {cmd}")
