# STEP 0 for idea 38: GDN decode T1 launch-shape sweep at N=8/16/32 (+ N=1..4).
# Synthetic tensors of the serving shapes (Qwen3.8-27B GDN, H=16, HV=48,
# K=V=128): q/k/v are views of one [N, 16384] in_proj_qkvz row block, a/b column
# slices of [N, 96] in_proj_ba, bf16 state pool with the serving slot stride
# (817152 elements, [HV, V, K] K-innermost inside a slot), permuted indices.
# Chain = one CUDA graph of LAYERS T1 calls on distinct layers (cold state,
# each layer's dirty write-back drains into the next layer's reads, as in a
# decode step). Before every replay: restore all pools from pristine copies and
# evict L2 with a 256 MiB read (outside the timed region). Arms are timed in
# interleaved rounds; "main" is timed twice (A/A floor).
# Every arm is also run once from the pristine state and compared bitwise with
# main's kernel (o and the whole pool).
import json
import os
import sys
import time

import torch

from tensorrt_llm._torch.modules.fla import gdn_decode_t1 as t1mod

H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VAL_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VAL_DIM
QKVZ_DIM = CONV_DIM + VAL_DIM
SLOT_STRIDE = 817152
LAYERS = int(os.environ.get("LAYERS", "48"))
WARMUP, ITERS = 10, 50
ROUNDS = int(os.environ.get("ROUNDS", "3"))
dev = torch.device("cuda")
flush_buf = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def log(*a):
    print(*a, flush=True)


def make_layer(n, gen):
    slots = n + 4
    qkvz = (torch.randn(n, QKVZ_DIM, device=dev, generator=gen) * 0.7).to(
        torch.bfloat16)
    ba = (torch.randn(n, 2 * HV, device=dev, generator=gen) * 1.5).to(
        torch.bfloat16)
    A = torch.exp(torch.rand(HV, device=dev, generator=gen) * 6.0 - 5.3)
    flat = (torch.randn(slots * SLOT_STRIDE, device=dev, generator=gen) *
            0.2).to(torch.bfloat16)
    pool = flat.as_strided((slots, HV, V, K), (SLOT_STRIDE, V * K, K, 1))
    idx = torch.randperm(slots, device=dev, generator=gen)[:n].to(torch.int32)
    mixed = qkvz[:, :CONV_DIM]
    return dict(
        q=mixed[:, :KEY_DIM].view(1, n, H, K),
        k=mixed[:, KEY_DIM:2 * KEY_DIM].view(1, n, H, K),
        v=mixed[:, 2 * KEY_DIM:].view(1, n, HV, V),
        b=ba[:, :HV],
        a=ba[:, HV:],
        A_log=torch.log(A).float(),
        dt_bias=(torch.randn(HV, device=dev, generator=gen) * 0.5 -
                 1.0).float(),
        flat=flat,
        flat0=flat.clone(),
        pool=pool,
        idx=idx,
        out=torch.zeros(n, HV, V, dtype=torch.bfloat16, device=dev),
    )


def make_call(bv, warps):

    def call(lay):
        t1mod.fused_sigmoid_gating_delta_rule_update_t1(
            A_log=lay["A_log"], a=lay["a"], dt_bias=lay["dt_bias"],
            softplus_beta=1.0, softplus_threshold=20.0, q=lay["q"],
            k=lay["k"], v=lay["v"], b=lay["b"],
            initial_state_source=lay["pool"],
            initial_state_indices=lay["idx"], scale=K**-0.5,
            use_qk_l2norm_in_kernel=True, output=lay["out"], BV=bv,
            num_warps=warps)

    return call


def reset(layers):
    for lay in layers:
        lay["flat"].copy_(lay["flat0"])
        lay["out"].zero_()
    flush_buf.sum()


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
        reset(layers)
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


def kernel_info():
    out = []
    try:
        fn = t1mod.fused_sigmoid_gating_delta_rule_update_t1_kernel
        for dkey, val in fn.device_caches.items():
            cache = val[0] if isinstance(val, tuple) else val
            for key, ker in cache.items():
                md = ker.metadata
                ttgir = ker.asm.get("ttgir", "")
                layouts = [ln.strip() for ln in ttgir.splitlines()
                           if ln.startswith("#")]
                out.append(dict(num_warps=md.num_warps, n_regs=ker.n_regs,
                                n_spills=ker.n_spills, layouts=layouts,
                                shared=md.shared))
    except Exception as e:  # noqa
        out.append(dict(error=repr(e)))
    return out


def main():
    batches = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else
                                "8,16,32").split(",")]
    arm_spec = sys.argv[2] if len(sys.argv) > 2 else (
        "8x1,16x1,32x1,64x1,16x2,16x4,32x2,32x4,64x4,128x4,128x8,128x16")
    arms = [("main", None)] + [(f"BV{s.split('x')[0]}w{s.split('x')[1]}",
                                (int(s.split("x")[0]), int(s.split("x")[1])))
                               for s in arm_spec.split(",")]
    arms.append(("main_aa", None))
    log("gpu", torch.cuda.get_device_name(), "layers", LAYERS, "arms",
        [a for a, _ in arms])
    result = {}
    for n in batches:
        gen = torch.Generator(device=dev)
        gen.manual_seed(1000 + n)
        layers = [make_layer(n, gen) for _ in range(LAYERS)]
        dflt = t1mod.gdn_t1_decode_config(n * HV, V)
        log(f"N={n}: main config BV={dflt[0]} warps={dflt[1]}")
        calls = {}
        for name, cfg in arms:
            calls[name] = make_call(None, None) if cfg is None else make_call(
                *cfg)
        # Bitwise vs main, one call per layer from pristine state.
        reset(layers)
        for lay in layers:
            calls["main"](lay)
        torch.cuda.synchronize()
        ref = [(lay["out"].clone(), lay["flat"].clone()) for lay in layers]
        bit = {}
        for name, _ in arms:
            reset(layers)
            for lay in layers:
                calls[name](lay)
            torch.cuda.synchronize()
            same = all(
                torch.equal(lay["out"].view(torch.int16), r[0].view(
                    torch.int16)) and torch.equal(
                        lay["flat"].view(torch.int16), r[1].view(torch.int16))
                for lay, r in zip(layers, ref))
            bit[name] = same
        log(f"N={n} bitwise vs main: {bit}")
        graphs = {name: capture(layers, calls[name]) for name, _ in arms}
        samples = {name: [] for name, _ in arms}
        for r in range(ROUNDS):
            order = arms if r % 2 == 0 else list(reversed(arms))
            for name, _ in order:
                samples[name] += time_graph(layers, graphs[name])
            log(f"N={n} round {r} done")
        res = {}
        base = stats(samples["main"])["med"]
        for name, _ in arms:
            st = stats(samples[name])
            st["bitwise"] = bit[name]
            st["gain_vs_main"] = (base - st["med"]) / base
            res[name] = st
            log(f"N={n} {name:10s} med {st['med']:7.3f} us/layer p10 "
                f"{st['p10']:7.3f} p90 {st['p90']:7.3f} nf {st['nf']:.3f} "
                f"gain {st['gain_vs_main']*100:+.2f}% bitwise {bit[name]}")
        result[str(n)] = res
        del graphs, layers
        torch.cuda.empty_cache()
    info = kernel_info()
    for i in info:
        log("kernel", json.dumps(i))
    print(json.dumps({"sweep": result, "kernels": [
        {k: v for k, v in i.items() if k != "layouts"} for i in info]}),
          flush=True)


main()
