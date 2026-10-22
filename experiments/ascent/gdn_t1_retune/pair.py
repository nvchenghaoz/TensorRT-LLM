# Paired kernel-time benchmark for idea 38 (GDN decode T1 + gated RMSNorm).
# Self-contained and BRACE-FREE (passed inline with python -c to run_gpu_pair)
# so it runs unchanged in the base tree and in the candidate tree. Each tree
# runs ITS OWN production path from Qwen3NextGatedDeltaNet.forward_decode to
# the gated-norm output y:
#   base: forward_decode -> o, then rms_norm_gated_token_major(o, z, w, eps)
#   candidate: same, except at gdn_mixer's routed rows (5..32) it calls
#              forward_decode(..., norm_fusion=(z, w, eps)) -> y (one kernel).
# Shapes: Qwen3.8-27B GDN (H=16, HV=48, K=V=128); q/k/v/z are views of one
# [N, 16384] in_proj_qkvz block, a/b slices of [N, 96]; bf16 state pool with
# the serving slot stride 817152 and permuted slot indices; conv already
# applied (conv_fused=True; unchanged by this branch).
# Timing: one CUDA graph = 48 layers (distinct pools/inputs) like one decode
# step. Before every replay, outside the timed region: restore every pool
# from a pristine copy, then evict L2 with a 256 MiB read. 3 rounds x
# (10 warmup + 50 timed) CUDA-event replays; median and p10/p90 per layer.
import json
import types

import torch

from tensorrt_llm._torch.modules.mamba import gdn_mixer
from tensorrt_llm._torch.modules.mamba.layernorm_gated import \
    rms_norm_gated_token_major

H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VAL_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VAL_DIM
QKVZ_DIM = CONV_DIM + VAL_DIM
SLOT_STRIDE = 817152
LAYERS, WARMUP, ITERS, ROUNDS = 48, 10, 50, 3
EPS = 1e-6
BATCHES = [1, 2, 3, 4, 8, 16, 32]
REPORT = ["8", "16", "32"]
dev = torch.device("cuda")
flush_buf = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=dev)
FUSION = getattr(gdn_mixer, "_T1_NORM_FUSION", False)


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
    return dict(
        n=n, qkvz=qkvz, a=ba[:, HV:], b=ba[:, :HV],
        z=qkvz[:, CONV_DIM:].view(n, HV, V),
        A_log=torch.log(A).float(),
        dt_bias=(torch.randn(HV, device=dev, generator=gen) * 0.5 -
                 1.0).float(),
        norm_w=(1.0 + 0.1 * torch.randn(V, device=dev, generator=gen)).to(
            torch.bfloat16),
        flat=flat, flat0=flat.clone(),
        pool=flat.as_strided((slots, HV, V, K), (SLOT_STRIDE, V * K, K, 1)),
        idx=torch.randperm(slots, device=dev,
                           generator=gen)[:n].to(torch.int32),
        cu=torch.arange(n + 1, dtype=torch.long, device=dev),
        fake=types.SimpleNamespace(key_dim_per_tp=H * K,
                                   num_k_heads_per_tp=H, head_k_dim=K,
                                   num_v_heads_per_tp=HV, head_v_dim=V,
                                   A_log=None, dt_bias=None),
    )


def routed(n):
    return FUSION and (gdn_mixer._T1_NORM_FUSION_MIN_ROWS <= n <=
                       gdn_mixer._T1_NORM_FUSION_MAX_ROWS)


def step(lay):
    fake = lay["fake"]
    fake.A_log, fake.dt_bias = lay["A_log"], lay["dt_bias"]
    kw = dict(mixed_qkv=lay["qkvz"][:, :CONV_DIM], a=lay["a"], b=lay["b"],
              cache_indices=lay["idx"], num_decodes=lay["n"],
              conv_fused=True)
    if routed(lay["n"]):
        kw["norm_fusion"] = (lay["z"], lay["norm_w"], EPS)
        return gdn_mixer.Qwen3NextGatedDeltaNet.forward_decode(
            fake, None, lay["pool"], lay["cu"], **kw)
    o = gdn_mixer.Qwen3NextGatedDeltaNet.forward_decode(
        fake, None, lay["pool"], lay["cu"], **kw)
    return rms_norm_gated_token_major(o.reshape(-1, V), lay["z"],
                                      lay["norm_w"], EPS)


def reset(layers):
    for lay in layers:
        lay["flat"].copy_(lay["flat0"])
    flush_buf.sum()


def main():
    log("gpu", torch.cuda.get_device_name(), "fusion_in_tree", FUSION)
    out = dict()
    for n in BATCHES:
        gen = torch.Generator(device=dev)
        gen.manual_seed(4000 + n)
        layers = [make_layer(n, gen) for _ in range(LAYERS)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for lay in layers:
                step(lay)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for lay in layers:
                step(lay)
        torch.cuda.synchronize()
        xs = []
        for r in range(ROUNDS):
            evs = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
                   for _ in range(WARMUP + ITERS)]
            for e0, e1 in evs:
                reset(layers)
                e0.record()
                g.replay()
                e1.record()
            torch.cuda.synchronize()
            xs += [e0.elapsed_time(e1) / LAYERS for e0, e1 in evs[WARMUP:]]
        xs.sort()
        med = xs[len(xs) // 2]
        p10, p90 = xs[int(0.1 * len(xs))], xs[int(0.9 * len(xs))]
        out[str(n)] = dict(kernel_ms=med, noise_floor=(p90 - p10) / med,
                           p10_ms=p10, p90_ms=p90, routed=routed(n))
        log("N=%d routed=%s: median %.3f us/layer p10 %.3f p90 %.3f" %
            (n, routed(n), med * 1e3, p10 * 1e3, p90 * 1e3))
        del g, layers
        torch.cuda.empty_cache()
    batch = dict()
    for b in REPORT:
        batch[b] = dict(kernel_ms=out[b]["kernel_ms"],
                        noise_floor=out[b]["noise_floor"])
    print(json.dumps(dict(
        batch=batch, output=dict(clazz="exact"),
        extra=dict(per_layer_ms_all_N=out, fusion_in_tree=FUSION,
                   unit="ms per GDN layer (T1 + gated norm), 48-layer "
                   "cold-state CUDA-graph chain"))).replace(
                       '"clazz"', '"class"'), flush=True)


main()
