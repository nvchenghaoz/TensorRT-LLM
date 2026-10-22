# STEP 0b: upper bound of arm F (gated RMSNorm folded into T1).
# Chain of LAYERS GDN layers in one CUDA graph, cold state (restore + 256 MiB
# L2 evict before every replay, outside timing), interleaved rounds.
#   main+norm : main T1 config, then rms_norm_gated_token_major (as serving)
#   bv8+norm  : BV=8 1-warp T1 (exact arm), then the norm
#   main      : main T1 alone (norm cost = main+norm - main)
#   F-ub      : BV=128 8-warp T1 alone, NO norm (deliberately wrong output;
#               upper bound for any F arm with one 8-warp program per (n, hv))
#   main+norm_aa : A/A copy
# Also prints the norm kernel's TTGIR layouts (to emulate its sum order).
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from tensorrt_llm._torch.modules.mamba import layernorm_gated  # noqa: E402

LAYERS = int(os.environ.get("LAYERS", "48"))
WARMUP, ITERS, ROUNDS = 10, 50, 3


def log(*a):
    print(*a, flush=True)


def norm(lay):
    y = layernorm_gated.rms_norm_gated_token_major(
        lay["out"].reshape(-1, common.V), lay["z"], lay["norm_w"], 1e-6)
    lay["y"] = y


def with_norm(call):

    def f(lay):
        call(lay)
        norm(lay)

    return f


def capture(layers, call):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for lay in layers:
            call(lay)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for lay in layers:
            call(lay)
    torch.cuda.synchronize()
    return g


def time_graph(layers, g):
    evs = [(torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
           for _ in range(WARMUP + ITERS)]
    for e0, e1 in evs:
        common.reset(layers)
        e0.record()
        g.replay()
        e1.record()
    torch.cuda.synchronize()
    return [e0.elapsed_time(e1) * 1e3 / len(layers)
            for e0, e1 in evs[WARMUP:]]


def stats(xs):
    xs = sorted(xs)
    m = xs[len(xs) // 2]
    return dict(med=m, p10=xs[int(0.1 * len(xs))], p90=xs[int(0.9 * len(xs))],
                nf=(xs[int(0.9 * len(xs))] - xs[int(0.1 * len(xs))]) / m)


def main():
    batches = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else
                                "8,16,32").split(",")]
    main_call = common.make_call(None, None)
    arms = {
        "main+norm": with_norm(main_call),
        "bv8+norm": with_norm(common.make_call(8, 1)),
        "main": main_call,
        "F-ub": common.make_call(128, 8),
        "main+norm_aa": with_norm(main_call),
    }
    result = {}
    for n in batches:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(2000 + n)
        layers = [common.make_layer(n, gen) for _ in range(LAYERS)]
        graphs = {k: capture(layers, f) for k, f in arms.items()}
        samples = {k: [] for k in arms}
        names = list(arms)
        for r in range(ROUNDS):
            for k in (names if r % 2 == 0 else names[::-1]):
                samples[k] += time_graph(layers, graphs[k])
            log(f"N={n} round {r} done")
        base = stats(samples["main+norm"])["med"]
        res = {}
        for k in names:
            st = stats(samples[k])
            st["gain_vs_main_norm"] = (base - st["med"]) / base
            res[k] = st
            log(f"N={n} {k:13s} med {st['med']:7.3f} us/layer p10 "
                f"{st['p10']:7.3f} p90 {st['p90']:7.3f} nf {st['nf']:.3f} "
                f"gain vs main+norm {st['gain_vs_main_norm']*100:+.2f}%")
        result[str(n)] = res
        del graphs, layers
        torch.cuda.empty_cache()
    try:
        fn = layernorm_gated._rms_norm_gated_fwd_multirow_kernel
        fn = getattr(fn, "fn", fn)
        while hasattr(fn, "fn"):
            fn = fn.fn
        for dkey, val in fn.device_caches.items():
            cache = val[0] if isinstance(val, tuple) else val
            for key, ker in cache.items():
                ttgir = ker.asm.get("ttgir", "")
                log("norm kernel warps", ker.metadata.num_warps, "regs",
                    ker.n_regs)
                for ln in ttgir.splitlines():
                    if ln.startswith("#blocked") or "tt.reduce" in ln or \
                            "convert_layout" in ln:
                        log("  ", ln.strip()[:300])
    except Exception as e:  # noqa
        log("norm ttgir error", repr(e))
    print(json.dumps({"step0b": result}), flush=True)


main()
