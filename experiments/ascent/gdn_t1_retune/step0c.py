# Arm F timing: chain of LAYERS GDN decode layers (CUDA graph, cold state:
# restore + 256 MiB L2 evict before every replay, outside timing), arms
# interleaved over rounds. main+norm = main T1 config then
# rms_norm_gated_token_major (as serving); F{w} = fused T1 + gated norm with
# w warps; main+norm_aa = A/A copy.
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from tensorrt_llm._torch.modules.fla import gdn_decode_t1 as t1mod  # noqa
from tensorrt_llm._torch.modules.mamba import layernorm_gated  # noqa: E402

LAYERS = int(os.environ.get("LAYERS", "48"))
WARMUP, ITERS, ROUNDS = 10, 50, 3
EPS = 1e-6


def log(*a):
    print(*a, flush=True)


def main_norm(lay):
    common.make_call(None, None)(lay)
    lay["y"] = layernorm_gated.rms_norm_gated_token_major(
        lay["out"].reshape(-1, common.V), lay["z"], lay["norm_w"], EPS)


def make_f(warps):

    def f(lay):
        lay["y"] = t1mod.fused_sigmoid_gating_delta_rule_update_t1_norm(
            A_log=lay["A_log"], a=lay["a"], dt_bias=lay["dt_bias"],
            softplus_beta=1.0, softplus_threshold=20.0, q=lay["q"],
            k=lay["k"], v=lay["v"], b=lay["b"], z=lay["z"],
            norm_weight=lay["norm_w"], eps=EPS,
            initial_state_source=lay["pool"],
            initial_state_indices=lay["idx"], scale=common.K**-0.5,
            use_qk_l2norm_in_kernel=True, num_warps=warps)

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
                                "1,2,3,4,6,8,16,32").split(",")]
    warps = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else
                              "8").split(",")]
    arms = {"main+norm": main_norm}
    for w in warps:
        arms[f"F{w}"] = make_f(w)
    arms["main+norm_aa"] = main_norm
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
        base = stats(samples["main+norm"])["med"]
        res = {}
        for k in names:
            st = stats(samples[k])
            st["gain"] = (base - st["med"]) / base
            res[k] = st
            log(f"N={n} {k:13s} med {st['med']:7.3f} us/layer p10 "
                f"{st['p10']:7.3f} p90 {st['p90']:7.3f} nf {st['nf']:.3f} "
                f"gain {st['gain']*100:+.2f}%")
        result[str(n)] = res
        del graphs, layers
        torch.cuda.empty_cache()
    print(json.dumps({"step0c": result}), flush=True)


main()
