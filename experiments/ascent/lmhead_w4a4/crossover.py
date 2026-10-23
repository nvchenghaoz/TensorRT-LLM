# Crossover table for the SM120 lm_head (N=248320, K=5120): both arms in one process,
# interleaved per iteration, built by this tree's own code:
#   marlin: TRTLLM_LMHEAD_W4A4_MIN_ROWS=0 (main's Marlin W4A16 path), timed as lm_head(x).float()
#   w4a4:   TRTLLM_LMHEAD_W4A4_MIN_ROWS=1 (W4A4 at every row, FP32 out; .float() is a no-op)
# Autotune emulated as in serving (one 1-row lm_head call under autotune()).
# 512 MiB read flush before every call; eager calls are preceded by a sync so host launch
# overhead (fp4_quantize + nvfp4_gemm dispatch) is timed. 10 warmup + ITERS timed.
# usage: python crossover.py <checkpoint>
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import check as ck  # noqa: E402

EAGER_MS = [int(v) for v in os.environ.get("EAGER_MS", "33,36,40,44,48,52,56,60,64").split(",")]
GRAPH_MS = [int(v) for v in os.environ.get("GRAPH_MS", "32,48,56,64,65,68,96,115,128,132").split(",")]
WARM, ITERS = 10, int(os.environ.get("ITERS", "80"))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def main():
    ckpt = sys.argv[1]
    t = ck.lb.load_lm_head(ckpt)
    heads = {"marlin": ck.build(t, min_rows=0)[0], "w4a4": ck.build(t, min_rows=1)[0]}
    ck.log("arms %s" % {k: (type(h.quant_method).__name__, h.__dict__.get("_w4a4_min_rows"))
                        for k, h in heads.items()})
    sigma = float(t["input_scale"].float()) * 2688.0 / 6.0
    g = torch.Generator(device="cuda").manual_seed(4242)
    xs = {m: (torch.randn(m, ck.K, generator=g, device="cuda") * sigma).to(torch.bfloat16)
          for m in sorted(set(EAGER_MS + GRAPH_MS))}
    for h in heads.values():
        ck.tune(h, xs[min(xs)])
    flush = torch.empty(512 << 18, dtype=torch.float32, device="cuda")
    fsum = torch.empty((), dtype=torch.float32, device="cuda")
    fns = {k: (lambda y, h=h: h(y).float()) for k, h in heads.items()}
    table = {}
    for mode, ms in (("eager", EAGER_MS), ("graph", GRAPH_MS)):
        for m in ms:
            x = xs[m]
            graphs = {k: ck.lb.capture(f, x)[0] for k, f in fns.items()} if mode == "graph" else None
            ev = {k: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(WARM + ITERS)] for k in fns}
            with torch.inference_mode():
                for i in range(WARM + ITERS):
                    for k, f in fns.items():
                        torch.sum(flush, dim=0, out=fsum)
                        if mode == "eager":
                            torch.cuda.synchronize()
                        ev[k][i][0].record()
                        if mode == "eager":
                            f(x)
                        else:
                            graphs[k].replay()
                        ev[k][i][1].record()
            torch.cuda.synchronize()
            row = {}
            for k in fns:
                ts = [a.elapsed_time(b) for a, b in ev[k][WARM:]]
                row[k] = {"median_ms": round(pct(ts, 0.5), 5), "p10_ms": round(pct(ts, 0.1), 5),
                          "p90_ms": round(pct(ts, 0.9), 5)}
            row["w4a4_wins_beyond_spread"] = row["w4a4"]["p90_ms"] < row["marlin"]["p10_ms"]
            row["delta_ms"] = round(row["w4a4"]["median_ms"] - row["marlin"]["median_ms"], 5)
            table["%s_M%d" % (mode, m)] = row
            ck.log("%s M=%d marlin %.4f [%.4f,%.4f] w4a4 %.4f [%.4f,%.4f] win=%s" % (
                mode, m, row["marlin"]["median_ms"], row["marlin"]["p10_ms"],
                row["marlin"]["p90_ms"], row["w4a4"]["median_ms"], row["w4a4"]["p10_ms"],
                row["w4a4"]["p90_ms"], row["w4a4_wins_beyond_spread"]))
            del graphs
            torch.cuda.empty_cache()
    print(json.dumps({"crossover": table}))


if __name__ == "__main__":
    main()
