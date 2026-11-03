"""Shared helpers for the SM120 W4A8 MXFP4xMXFP8 FlashInfer decode-chain study (idea 35).

One EP4 rank of DeepSeek-V4.1-Flash: 96 local of 384 experts, top-6, hidden 5120,
moe_intermediate 2304, swiglu_limit 10.  Synthetic MXFP4 weights (random e2m1 codes,
random ue8m0 block exponents) laid out both the TRT-LLM CUTLASS way (swizzled SF) and the
FlashInfer SM12x way (MN-major int32-packed SF); the FP4 code bytes are shared.
"""

import time

import torch

H = 5120
I = 2304
E = 96
E_GLOBAL = 384
TOPK = 6
LIMIT = 10.0
EP_RANK = 0
EP_SIZE = 4

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def make_weights(seed=0, device="cuda"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    w1 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device, generator=g)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=device, generator=g)
    # ue8m0 exponents 119..122 -> scales 2^-8 .. 2^-5
    s1 = torch.randint(119, 123, (E, 2 * I, H // 32), dtype=torch.uint8, device=device, generator=g)
    s2 = torch.randint(119, 123, (E, H, I // 32), dtype=torch.uint8, device=device, generator=g)
    return w1, s1, w2, s2


def trtllm_layout(w1, s1, w2, s2):
    """What W4A8MXFP4MXFP8CutlassFusedMoEMethod holds after loading."""
    w31 = w1.view(torch.int64)  # (E, 2I, H/16)
    w2v = w2.view(torch.int64)  # (E, H, I/16)
    s31 = (
        torch.ops.trtllm.block_scale_interleave(s1.contiguous())
        .view(torch.int32)
        .reshape(E, 2 * I, H // 128)
    )
    s2s = (
        torch.ops.trtllm.block_scale_interleave(s2.contiguous())
        .view(torch.int32)
        .reshape(E, H, I // 128)
    )
    return w31, s31, w2v, s2s


def fi_sf_from_raw(raw):
    """raw ue8m0 bytes (E, N, K/32) -> FlashInfer SM12x MN-major SF (E, K/128, 4N) uint8."""
    e, n, kb = raw.shape
    assert n % 4 == 0 and kb % 4 == 0
    return raw.contiguous().view(torch.int32).transpose(1, 2).contiguous().view(torch.uint8)


def dequant_expert(codes, sf_raw):
    """codes (N, K/2) uint8, sf_raw (N, K/32) uint8 -> fp32 (N, K)."""
    n, kh = codes.shape
    k = kh * 2
    lut = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32, device=codes.device)
    lo = (codes & 0x0F).long()
    hi = (codes >> 4).long()
    vals = torch.empty(n, k, dtype=torch.float32, device=codes.device)
    vals[:, 0::2] = lut[lo]
    vals[:, 1::2] = lut[hi]
    scale = torch.pow(2.0, sf_raw.float() - 127.0)
    return (vals.view(n, k // 32, 32) * scale[:, :, None]).view(n, k)


def make_routing(m, n_local, seed, device="cuda", empty_token=None):
    """Global top-6 ids with exactly n_local picks on rank 0's experts [0, 96)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(1000 + seed)
    ids = torch.empty(m, TOPK, dtype=torch.int32)
    for t in range(m):
        nl = 0 if (empty_token is not None and t == empty_token) else n_local
        loc = torch.randperm(E, generator=g)[:nl]
        rem = torch.randperm(E_GLOBAL - E, generator=g)[: TOPK - nl] + E
        row = torch.cat([loc, rem])
        row = row[torch.randperm(TOPK, generator=g)]
        ids[t] = row.to(torch.int32)
    w = torch.rand(m, TOPK, generator=g) + 0.1
    w = (w / w.sum(-1, keepdim=True) * 1.5).float()
    return ids.to(device), w.to(device)


def reference(x, ids, w, w1, s1, w2, s2):
    """FP32 dequantized reference of the rank-0 routed-MoE output (BF16 x, no activation quant)."""
    m = x.shape[0]
    out = torch.zeros(m, H, dtype=torch.float32, device=x.device)
    xf = x.float()
    cache = {}
    for t in range(m):
        for j in range(TOPK):
            e = int(ids[t, j])
            if not (0 <= e - EP_RANK * E < E):
                continue
            le = e - EP_RANK * E
            if le not in cache:
                cache[le] = (dequant_expert(w1[le], s1[le]), dequant_expert(w2[le], s2[le]))
            W1, W2 = cache[le]
            h = W1 @ xf[t]
            lin, gate = h[:I], h[I:]
            gate = gate.clamp(max=LIMIT)
            lin = lin.clamp(-LIMIT, LIMIT)
            act = gate * torch.sigmoid(gate) * lin
            out[t] += float(w[t, j]) * (W2 @ act)
    return out


def err_stats(out, ref):
    o = out.float()
    d = (o - ref).abs()
    amax = ref.abs().max().item()
    denom = (o.double() ** 2 + ref.double() ** 2).sum().item()
    cdiff = 0.0 if denom == 0 else 1.0 - 2.0 * (o.double() * ref.double()).sum().item() / denom
    rel = d / ref.abs().clamp_min(0.05 * max(amax, 1e-6))
    return {
        "max_abs": d.max().item(),
        "max_abs_over_ref_absmax": d.max().item() / max(amax, 1e-12),
        "max_rel_floor5pct": rel.max().item(),
        "mean_rel_floor5pct": rel.mean().item(),
        "calc_diff": cdiff,
        "ref_absmax": amax,
        "finite": bool(torch.isfinite(o).all().item()),
    }


class CutlassChain:
    """main's decode chain: mxfp8_quantize + torch.ops.trtllm.fused_moe (EP4 rank 0)."""

    def __init__(self, w31, s31, w2v, s2s):
        from tensorrt_llm._torch.utils import ActivationType

        self.w31, self.w2v = w31, w2v
        fake = torch.ones(E, dtype=torch.float32, device="cuda")
        self.qs = [s31, fake, s2s, fake]
        self.limit = torch.full((E,), LIMIT, dtype=torch.float32, device="cuda")
        self.act = int(ActivationType.Swiglu)

    def __call__(self, x, ids, w):
        xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True, alignment=128)
        return torch.ops.trtllm.fused_moe(
            xq, ids, w, self.w31, None, self.w2v, None, torch.bfloat16,
            quant_scales=self.qs, input_sf=xsf, swizzled_input_sf=True,
            swiglu_alpha=None, swiglu_beta=None, swiglu_limit=self.limit,
            tp_size=1, tp_rank=0, ep_size=EP_SIZE, ep_rank=EP_RANK,
            cluster_size=1, cluster_rank=0, enable_alltoall=False,
            use_deepseek_fp8_block_scale=False, use_w4_group_scaling=False,
            use_int8_woq_per_channel=False, use_mxfp8_act_scaling=True,
            min_latency_mode=False, use_fused_finalize=True,
            tune_max_num_tokens=8192, activation_type=self.act,
            unpadded_hidden_size=H,
        )[0]


class FiChain:
    """FlashInfer SM12x chain called op by op (runners.py SM12xMxfp8Mxfp4Runner.forward order)."""

    def __init__(self, w1, w1_sf, w2, w2_sf, pdl=False, zero_sf1=True):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1 import (
            cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4, out_sf_shape)
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize import (
            cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4)
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_q0_route_triton import (
            make_mxfp8_q0_route_workspace, mxfp8_q0_route_triton)
        from flashinfer.tllm_enums import ActivationType as FiAct

        self.fc1, self.fc2, self.route = (cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4,
                                          cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4,
                                          mxfp8_q0_route_triton)
        self.out_sf_shape, self.make_ws = out_sf_shape, make_mxfp8_q0_route_workspace
        self.act = FiAct.Swiglu
        self.w1, self.w1_sf, self.w2, self.w2_sf = w1, w1_sf, w2, w2_sf
        self.pdl, self.zero_sf1 = pdl, zero_sf1
        self.bufs = {}

    def buffers(self, x, ids):
        m = x.shape[0]
        if m not in self.bufs:
            pairs = m * TOPK
            ws = self.make_ws(x, ids, E)
            q1 = torch.empty(pairs, I, dtype=torch.float8_e4m3fn, device=x.device)
            sf1 = torch.zeros(self.out_sf_shape(pairs, I, E), dtype=torch.int32, device=x.device)
            out = torch.zeros(m, H, dtype=torch.bfloat16, device=x.device)
            self.bufs[m] = (ws, q1, sf1, out)
        return self.bufs[m]

    def __call__(self, x, ids, w):
        ws, q1, sf1, out = self.buffers(x, ids)
        local = ids - EP_RANK * E
        local = torch.where((local >= 0) & (local < E), local, torch.full_like(local, -1))
        out.zero_()
        if self.zero_sf1:
            sf1.zero_()
        offsets, token_map, rw, a_q, a_sf = self.route(x, local, w, E, workspace=ws,
                                                       enable_pdl=self.pdl)
        self.fc1(a_q, a_sf, self.w1, self.w1_sf, offsets, tune=False, enable_pdl=self.pdl,
                 out_q=q1, out_sf=sf1, activation=self.act, swiglu_limit=LIMIT)
        self.fc2(q1, sf1, self.w2, self.w2_sf, offsets, token_map, rw, x.shape[0], tune=False,
                 enable_pdl=self.pdl, out=out)
        return out


def time_graph(fn, args, iters=60, warmup=10, flush_mb=512):
    """Capture fn(*args) in a CUDA graph, replay with an L2 flush before each timed replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*args)
    torch.cuda.synchronize()
    flush = torch.empty(flush_mb * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    ts = []
    for i in range(warmup + iters):
        flush.zero_()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if i >= warmup:
            ts.append(a.elapsed_time(b))
    ts.sort()
    med = ts[len(ts) // 2]
    p10, p90 = ts[len(ts) // 10], ts[(9 * len(ts)) // 10]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "noise": (p90 - p10) / med}, g, out
