# Kernel A/B for idea 37 (8-bit GDN recurrent state pool). BRACE-FREE so it
# can be passed inline with python -c to run_gpu_pair and run unchanged in
# the base tree (bf16 pool only) and in the candidate tree (bf16 pool = main's
# kernel code, unchanged, plus the int8 pool).
# Shapes: Qwen3.8-27B GDN (H=16, HV=48, K=V=128). q/k/v/z are views of one
# [N, 16384] in_proj_qkvz block, a/b slices of [N, 96]. bf16 pool slot stride
# 817152 elements (V2 [ssm | conv] slot); int8 pool [slots, 48, 132, 128] with
# slot stride 872448 bytes. Permuted slot indices.
# Decode: Qwen3NextGatedDeltaNet.forward_decode (conv_fused, no norm fusion ->
# T1; N=32 with norm fusion -> T1-norm) for LAYERS distinct layers in one CUDA
# graph; before every replay (outside timing) restore every pool from a
# pristine copy and evict L2 with a 256 MiB read. Mixed: forward_extend with
# 4 context requests [2048, 2048, 2048, 1920] from a snapshot state + 124
# decode riders, eager, same reset + flush per iteration.
# 2 rounds x (10 warmup + 50 timed) per case; median, p10, p90 per layer.
import json
import os
import types

import torch

from tensorrt_llm._torch.modules.mamba import gdn_mixer

try:
    from tensorrt_llm._torch.modules.fla import gdn_state_q8 as q8mod
except ImportError:
    q8mod = None

H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VAL_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VAL_DIM
QKVZ_DIM = CONV_DIM + VAL_DIM
SLOT_BF16 = 817152
ROWS8 = V + 4
SLOT_Q8 = HV * ROWS8 * K + 61440
LAYERS = int(os.environ.get("GDN_AB_LAYERS", "8"))
WARMUP, ITERS, ROUNDS = 10, 50, 2
EPS = 1e-6
DECODE_N = [64, 97, 104, 112, 127, 128]
NORM_N = [32]
MIXED_P = [2048, 2048, 2048, 1920]
MIXED_R = 124
dev = torch.device("cuda")
flush_buf = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=dev)
ARMS = ["bf16"] + (["q8"] if q8mod is not None else [])
ONLY = os.environ.get("GDN_AB_ONLY", "")


def log(*a):
    print(*a, flush=True)


def make_pool(arm, slots, gen):
    if arm == "bf16":
        flat = (torch.randn(slots * SLOT_BF16, device=dev, generator=gen) *
                0.2).to(torch.bfloat16)
        pool = flat.as_strided((slots, HV, V, K), (SLOT_BF16, V * K, K, 1))
        return flat, pool
    flat = torch.zeros(slots * SLOT_Q8, dtype=torch.int8, device=dev)
    pool = flat.as_strided((slots, HV, ROWS8, K), (SLOT_Q8, ROWS8 * K, K, 1))
    for s0 in range(0, slots, 16):
        s1 = min(slots, s0 + 16)
        src = torch.randn(s1 - s0, HV, V, K, device=dev, generator=gen) * 0.2
        q8mod.q8_quant_scatter(
            src, pool, torch.arange(s0, s1, dtype=torch.int32, device=dev))
    return flat, pool


def make_fake(gen):
    A = torch.exp(torch.rand(HV, device=dev, generator=gen) * 6.0 - 5.3)
    fake = types.SimpleNamespace(
        key_dim_per_tp=KEY_DIM, num_k_heads_per_tp=H, head_k_dim=K,
        num_v_heads_per_tp=HV, head_v_dim=V,
        A_log=torch.log(A).float(),
        dt_bias=(torch.randn(HV, device=dev, generator=gen) * 0.5 -
                 1.0).float(),
        conv1d=types.SimpleNamespace(
            weight=(torch.randn(CONV_DIM, 4, device=dev, generator=gen) *
                    0.3).to(torch.bfloat16),
            bias=None),
        activation="silu")
    cls = gdn_mixer.Qwen3NextGatedDeltaNet
    if hasattr(cls, "_forward_extend_q8"):
        fake._forward_extend_q8 = types.MethodType(cls._forward_extend_q8,
                                                   fake)
    return fake


def make_decode_layer(arm, n, gen):
    slots = n + 4
    qkvz = (torch.randn(n, QKVZ_DIM, device=dev, generator=gen) * 0.7).to(
        torch.bfloat16)
    ba = (torch.randn(n, 2 * HV, device=dev, generator=gen) * 1.5).to(
        torch.bfloat16)
    flat, pool = make_pool(arm, slots, gen)
    return dict(
        n=n, qkvz=qkvz, a=ba[:, HV:], b=ba[:, :HV],
        z=qkvz[:, CONV_DIM:].view(n, HV, V),
        norm_w=(1.0 + 0.1 * torch.randn(V, device=dev, generator=gen)).to(
            torch.bfloat16),
        flat=flat, flat0=flat.clone(), pool=pool,
        idx=torch.randperm(slots, device=dev,
                           generator=gen)[:n].to(torch.int32),
        cu=torch.arange(n + 1, dtype=torch.long, device=dev),
        fake=make_fake(gen))


def decode_step(lay, norm):
    kw = dict(mixed_qkv=lay["qkvz"][:, :CONV_DIM], a=lay["a"], b=lay["b"],
              cache_indices=lay["idx"], num_decodes=lay["n"],
              conv_fused=True)
    if norm:
        kw["norm_fusion"] = (lay["z"], lay["norm_w"], EPS)
    return gdn_mixer.Qwen3NextGatedDeltaNet.forward_decode(
        lay["fake"], None, lay["pool"], lay["cu"], **kw)


def stats(xs):
    xs = sorted(xs)
    med = xs[len(xs) // 2]
    p10, p90 = xs[int(0.1 * len(xs))], xs[int(0.9 * len(xs))]
    return dict(kernel_ms=med, noise_floor=(p90 - p10) / med, p10_ms=p10,
                p90_ms=p90)


def time_decode(arm, n, norm):
    gen = torch.Generator(device=dev)
    gen.manual_seed(4000 + n)
    layers = [make_decode_layer(arm, n, gen) for _ in range(LAYERS)]

    def reset():
        for lay in layers:
            lay["flat"].copy_(lay["flat0"])
        flush_buf.sum()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for lay in layers:
            decode_step(lay, norm)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for lay in layers:
            decode_step(lay, norm)
    torch.cuda.synchronize()
    xs = []
    for _ in range(ROUNDS):
        evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))
               for _ in range(WARMUP + ITERS)]
        for e0, e1 in evs:
            reset()
            e0.record()
            g.replay()
            e1.record()
        torch.cuda.synchronize()
        xs += [e0.elapsed_time(e1) / LAYERS for e0, e1 in evs[WARMUP:]]
    del g, layers
    torch.cuda.empty_cache()
    return stats(xs)


def make_mixed(arm, gen):
    nseq = len(MIXED_P) + MIXED_R
    slots = nseq + 8
    lens = MIXED_P + [1] * MIXED_R
    T = sum(lens)
    qkvz = (torch.randn(T, QKVZ_DIM, device=dev, generator=gen) * 0.7).to(
        torch.bfloat16)
    ba = (torch.randn(T, 2 * HV, device=dev, generator=gen) * 1.5).to(
        torch.bfloat16)
    flat, pool = make_pool(arm, slots, gen)
    conv = (torch.randn(slots, CONV_DIM, 3, device=dev, generator=gen) *
            0.5).to(torch.bfloat16)
    qsl = [0]
    for ln in lens:
        qsl.append(qsl[-1] + ln)
    idx = torch.randperm(slots, device=dev, generator=gen)[:nseq].to(
        torch.int32)
    return dict(
        qkvz=qkvz, qkvz0=qkvz.clone(), ba=ba, flat=flat, flat0=flat.clone(),
        pool=pool, conv=conv,
        conv0=conv.clone(), idx=idx,
        qsl=torch.tensor(qsl, dtype=torch.int32, device=dev),
        qsl_long=torch.tensor(qsl, dtype=torch.long, device=dev),
        has_init=torch.ones(nseq, dtype=torch.bool, device=dev),
        T=T, nseq=nseq, fake=make_fake(gen))


def mixed_step(m):
    P = len(MIXED_P)
    kw = dict(
        mixed_qkv=m["qkvz"][:, :CONV_DIM], a=m["ba"][:, HV:],
        b=m["ba"][:, :HV], batch_size=m["nseq"],
        has_initial_states=m["has_init"], cache_indices=m["idx"],
        query_start_loc=m["qsl"], query_start_loc_long=m["qsl_long"],
        num_prefill_tokens=sum(MIXED_P), num_decode_tokens=MIXED_R,
        state_indices_p=m["idx"][:P], state_indices_d=m["idx"][P:],
        num_prefill=P, num_decodes=MIXED_R, state_workspace=None,
        intermediate_state_indices=None, use_replay=False)
    return gdn_mixer.Qwen3NextGatedDeltaNet.forward_extend(
        m["fake"], m["conv"], m["pool"], **kw)


def time_mixed(arm):
    gen = torch.Generator(device=dev)
    gen.manual_seed(777)
    m = make_mixed(arm, gen)

    def reset():
        m["flat"].copy_(m["flat0"])
        m["conv"].copy_(m["conv0"])
        # causal_conv1d_update convolves the rider rows in place.
        m["qkvz"].copy_(m["qkvz0"])
        flush_buf.sum()

    xs = []
    for _ in range(ROUNDS):
        evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))
               for _ in range(WARMUP + ITERS)]
        for e0, e1 in evs:
            reset()
            e0.record()
            mixed_step(m)
            e1.record()
        torch.cuda.synchronize()
        xs += [e0.elapsed_time(e1) for e0, e1 in evs[WARMUP:]]
    del m
    torch.cuda.empty_cache()
    return stats(xs)


def main():
    log("gpu", torch.cuda.get_device_name(), "arms", ARMS, "layers", LAYERS)
    res = dict()
    # bf16 is timed twice (A/A) in every tree.
    order = ["bf16", "bf16_aa"] + ARMS[1:]
    for arm_name in order:
        arm = "bf16" if arm_name.startswith("bf16") else arm_name
        r = dict()
        if ONLY in ("", "decode"):
            for n in DECODE_N:
                r["t1_" + str(n)] = time_decode(arm, n, False)
                log(arm_name, "T1 N=%d median %.2f us/layer p10 %.2f p90 %.2f"
                    % (n, r["t1_" + str(n)]["kernel_ms"] * 1e3,
                       r["t1_" + str(n)]["p10_ms"] * 1e3,
                       r["t1_" + str(n)]["p90_ms"] * 1e3))
            for n in NORM_N:
                r["t1norm_" + str(n)] = time_decode(arm, n, True)
                log(arm_name, "T1norm N=%d median %.2f us/layer" %
                    (n, r["t1norm_" + str(n)]["kernel_ms"] * 1e3))
        if ONLY in ("", "mixed"):
            r["mixed"] = time_mixed(arm)
            log(arm_name, "mixed [2048x3,1920]+124 median %.1f us p10 %.1f "
                "p90 %.1f" % (r["mixed"]["kernel_ms"] * 1e3,
                              r["mixed"]["p10_ms"] * 1e3,
                              r["mixed"]["p90_ms"] * 1e3))
        res[arm_name] = r
    rep = res[order[-1]]
    batch = dict()
    if "t1_64" in rep:
        batch["64"] = dict(kernel_ms=rep["t1_64"]["kernel_ms"],
                           noise_floor=rep["t1_64"]["noise_floor"])
        batch["128"] = dict(kernel_ms=rep["t1_128"]["kernel_ms"],
                            noise_floor=rep["t1_128"]["noise_floor"])
    clazz = "precision" if order[-1] == "q8" else "exact"
    print(json.dumps(dict(
        batch=batch, output=dict(clazz=clazz),
        extra=dict(reported_arm=order[-1], arms=res, layers=LAYERS,
                   unit="ms per GDN layer (decode, CUDA graph of LAYERS "
                   "cold-state layers) or ms per mixed-step layer (eager)")
    )).replace('"clazz"', '"class"'), flush=True)


main()
