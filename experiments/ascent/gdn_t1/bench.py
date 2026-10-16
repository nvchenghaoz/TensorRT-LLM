# GDN decode (fused_sigmoid_gating_delta_rule_update) microbenchmark.
# Self-contained: runs unchanged against main (generic Triton kernel) and the
# candidate (T == 1 kernel), so it can be passed inline with `python -c`.
# Serving shapes of Qwen3.8-27B GDN layers at TP=1: H=16, HV=48, K=V=128;
# q/k/v are strided views of the in_proj_qkvz output (row stride 16384), a/b are
# column slices of the in_proj_ba output; bf16 state pool [slots, HV, V, K]
# with non-identity state indices. One CUDA graph = 30 GDN layers (distinct
# pools/inputs), like one decode step. Before every iteration (outside the
# timed region, identical in both arms): restore all pools and outputs from a
# pristine copy, then evict L2 with a 256 MiB read. 20 warmup + 100 timed.
# Optional argv: "sweep" -> time (BV, num_warps) configs of the T == 1 kernel.
import json
import sys

import torch

from tensorrt_llm._torch.modules.fla import \
    fused_sigmoid_gating_recurrent as fsgr

try:
    from tensorrt_llm._torch.modules.fla import gdn_decode_t1 as t1mod
except ImportError:
    t1mod = None


def log(*args):
    print(*args, flush=True)


H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VAL_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VAL_DIM
QKVZ_DIM = CONV_DIM + VAL_DIM
LAYERS, SLOTS, WARMUP, ITERS = 30, 16, 20, 100
BATCHES = [1, 2, 4, 8]
REPORT = ["1", "2", "4"]
dev = torch.device("cuda")
flush_buf = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def make_layer(n, gen):
    qkvz = (torch.randn(n, QKVZ_DIM, device=dev, generator=gen) *
            0.5).to(torch.bfloat16)
    ba = torch.randn(n, 2 * HV, device=dev, generator=gen).to(torch.bfloat16)
    a_log = torch.log(
        torch.rand(HV, device=dev, generator=gen) * 15.0 + 1.0).float()
    dt_bias = (torch.randn(HV, device=dev, generator=gen) * 0.5).float()
    pool = (torch.randn(SLOTS, HV, V, K, device=dev, generator=gen) *
            0.1).to(torch.bfloat16)
    idx = torch.randperm(SLOTS, device=dev, generator=gen)[:n].to(torch.int32)
    mixed = qkvz[:, :CONV_DIM]
    return dict(
        q=mixed[:, :KEY_DIM].view(1, n, H, K),
        k=mixed[:, KEY_DIM:2 * KEY_DIM].view(1, n, H, K),
        v=mixed[:, 2 * KEY_DIM:].view(1, n, HV, V),
        b=ba[:, :HV],
        a=ba[:, HV:],
        A_log=a_log,
        dt_bias=dt_bias,
        pool=pool,
        pool0=pool.clone(),
        idx=idx,
        cu=torch.arange(n + 1, dtype=torch.long, device=dev),
        out=torch.zeros(1, n, HV, V, dtype=torch.bfloat16, device=dev),
    )


def call_public(lay):
    fsgr.fused_sigmoid_gating_delta_rule_update(
        A_log=lay["A_log"],
        dt_bias=lay["dt_bias"],
        q=lay["q"],
        k=lay["k"],
        v=lay["v"],
        a=lay["a"],
        b=lay["b"],
        initial_state_source=lay["pool"],
        initial_state_indices=lay["idx"],
        cu_seqlens=lay["cu"],
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        output=lay["out"],
    )


def make_call_t1(bv, warps):

    def call(lay):
        t1mod.fused_sigmoid_gating_delta_rule_update_t1(
            A_log=lay["A_log"],
            a=lay["a"],
            dt_bias=lay["dt_bias"],
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=lay["q"],
            k=lay["k"],
            v=lay["v"],
            b=lay["b"],
            initial_state_source=lay["pool"],
            initial_state_indices=lay["idx"],
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            output=lay["out"],
            BV=bv,
            num_warps=warps,
        )

    return call


def reset(layers):
    for lay in layers:
        lay["pool"].copy_(lay["pool0"])
        lay["out"].zero_()
    flush_buf.sum()  # read-evict L2 (clean lines), outside the timed region


def time_graph(layers, call):
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
    evs = [(torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
           for _ in range(WARMUP + ITERS)]
    for e0, e1 in evs:
        reset(layers)
        e0.record()
        g.replay()
        e1.record()
    torch.cuda.synchronize()
    ms = sorted(e0.elapsed_time(e1) / len(layers) for e0, e1 in evs[WARMUP:])
    med = ms[len(ms) // 2]
    p10, p90 = ms[int(0.1 * len(ms))], ms[int(0.9 * len(ms))]
    del g
    return med, p10, p90


def main():
    sweep = len(sys.argv) > 1 and sys.argv[1] == "sweep"
    log("module", fsgr.__file__, "t1_available", t1mod is not None,
        "gpu", torch.cuda.get_device_name())
    result = {}
    for n in BATCHES:
        gen = torch.Generator(device=dev)
        gen.manual_seed(1000 + n)
        layers = [make_layer(n, gen) for _ in range(LAYERS)]
        med, p10, p90 = time_graph(layers, call_public)
        log(f"N={n} public: median {med*1e3:.3f} us p10 {p10*1e3:.3f} "
            f"p90 {p90*1e3:.3f} us per call")
        result[str(n)] = {
            "kernel_ms": med,
            "noise_floor": (p90 - p10) / med,
            "p90_ms": p90
        }
        if sweep and t1mod is not None:
            t1mod.GDN_T1_DECODE_ENABLED = False
            med_o, _, p90_o = time_graph(layers, call_public)
            t1mod.GDN_T1_DECODE_ENABLED = True
            log(f"N={n} original kernel: median {med_o*1e3:.3f} us "
                f"p90 {p90_o*1e3:.3f}")
            for bv in (8, 16, 32, 64):
                for warps in (1, 2, 4, 8):
                    med_c, _, p90_c = time_graph(layers,
                                                 make_call_t1(bv, warps))
                    log(f"N={n} t1 BV={bv} warps={warps}: median "
                        f"{med_c*1e3:.3f} us p90 {p90_c*1e3:.3f} "
                        f"speedup {med_o/med_c:.3f}x")
        del layers
        torch.cuda.empty_cache()
    out = {
        "batch": {
            b: {
                "kernel_ms": result[b]["kernel_ms"],
                "noise_floor": result[b]["noise_floor"]
            }
            for b in REPORT
        },
        "output": {
            "class": "exact"
        },
        "extra": {
            "N8": result["8"],
            "p90_ms": {b: result[b]["p90_ms"]
                       for b in REPORT}
        },
    }
    print(json.dumps(out), flush=True)


main()
