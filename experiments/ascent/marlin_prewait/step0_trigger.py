# Idea 28 stage 2 STEP 0 (upper-bound probe, candidate tree only; numerics of the producer are NOT the
# served norm's, timing only). Per layer in one CUDA graph, cold L2 (C weight copies), PDL on:
#   producer (Triton gemma add-RMSNorm stand-in, PDL launch; trigger LATE = after its stores, as the FlashInfer
#   norm does today, or EARLY = right after its global loads, before the reduction) -> Marlin -> FlashInfer
#   gemma add-RMSNorm.
# Arms (rotated each iteration): LK (late trigger, Marlin wait-first = main), LP (late, Marlin pre-wait L2 prefetch
# of (stages-1) k-tiles; _d2/_d4 = 2x/4x that depth), EK (early, wait-first), EP (early, pre-wait), LP2 (= LP, A/A).
import json
import os
import time

os.environ["TRTLLM_ENABLE_PDL"] = "1"
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin  # noqa: E402

dev = torch.device("cuda")
EPS, WARM, ITERS = 1e-6, 10, 60
SHAPES = [("qkvz", 16384, 5120, 6), ("qkv", 14336, 5120, 7), ("outproj", 5120, 6144, 16)]
MS = [4, 8, 12, 16]
ARMS = [("LK", False, False, 1), ("LP", False, True, 1), ("LP_d2", False, True, 2), ("LP_d4", False, True, 4),
        ("EK", True, False, 1), ("EP", True, True, 1), ("EP_d2", True, True, 2), ("EP_d4", True, True, 4),
        ("LP2", False, True, 1)]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


@triton.jit
def _add_rmsnorm(x_ptr, r_ptr, w_ptr, H, eps, EARLY: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    if EARLY:
        # make the trigger depend on the loaded data (it issues after the loads land)
        if tl.sum(h, axis=0) != 12345.678:
            tl.extra.cuda.gdc_launch_dependents()
    tl.store(r_ptr + row * H + offs, h.to(tl.bfloat16), mask=mask)
    ms = tl.sum(h * h, axis=0) / H
    y = h * tl.rsqrt(ms + eps) * (1.0 + w)
    tl.store(x_ptr + row * H + offs, y.to(tl.bfloat16), mask=mask)
    if not EARLY:
        tl.extra.cuda.gdc_launch_dependents()


def tnorm(x, r, w, early):
    m, H = x.shape
    _add_rmsnorm[(m,)](x, r, w, H, EPS, EARLY=early, BLOCK=triton.next_power_of_2(H), num_warps=8,
                       launch_pdl=True)


def build(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    gs = (448.0 * 6.0 / w.float().abs().amax()).float().reshape(1)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, (1.0 / gs).float(), n, k, 16)


def marlin(x, wt, n, k):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None, out_dtype=torch.bfloat16,
                                              size_n=n, size_k=k)


class Chain:
    def __init__(self, n, k, m, ws, wn1, wn2, seed, early):
        self.n, self.k, self.m, self.ws, self.wn1, self.wn2, self.early = n, k, m, ws, wn1, wn2, early
        c = len(ws)
        per = m * k * 2 + m * n
        g = torch.Generator(device=dev).manual_seed(seed)
        st0 = torch.zeros(c * per, device=dev, dtype=torch.bfloat16)
        for i in range(c):
            st0[i * per: i * per + 2 * m * k] = (torch.randn(2 * m * k, generator=g, device=dev) * 0.5).to(
                torch.bfloat16)
        self.st0 = st0
        self.st = st0.clone()
        self.x, self.r1, self.r2 = [], [], []
        for i in range(c):
            b = i * per
            self.x.append(self.st[b: b + m * k].view(m, k))
            self.r1.append(self.st[b + m * k: b + 2 * m * k].view(m, k))
            self.r2.append(self.st[b + 2 * m * k: b + per].view(m, n))

    def run(self):
        self.st.copy_(self.st0)
        ys = []
        for i in range(len(self.ws)):
            tnorm(self.x[i], self.r1[i], self.wn1, self.early)
            y = marlin(self.x[i], self.ws[i], self.n, self.k)
            torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(y, self.r2[i], self.wn2, EPS)
            ys.append(y)
        return ys

    def state(self, ys):
        return torch.cat([self.st.clone()] + [y.reshape(-1).clone() for y in ys])


def capture(obj):
    s0 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for _ in range(3):
            obj.run()
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        o = obj.run()
    return gr, o


def main():
    detail = {}
    ok = True
    for sname, n, k, c in SHAPES:
        base = build(n, k, 11 + n + k)
        ws = [tuple(t.clone() for t in base) for _ in range(c)]
        del base
        g = torch.Generator(device=dev).manual_seed(5)
        wn1 = (0.1 * torch.randn(k, generator=g, device=dev)).to(torch.bfloat16)
        wn2 = (0.1 * torch.randn(n, generator=g, device=dev)).to(torch.bfloat16)
        for m in MS:
            objs, graphs, outs, eager = {}, {}, {}, {}
            for name, early, prewait, depth in ARMS:
                os.environ["TRTLLM_MARLIN_PREWAIT_B"] = "1" if prewait else "0"
                os.environ["TRTLLM_MARLIN_PREWAIT_DEPTH"] = str(depth)
                obj = Chain(n, k, m, ws, wn1, wn2, 100 + m, early)
                eager[name] = obj.state(obj.run())
                gr, o = capture(obj)
                objs[name], graphs[name], outs[name] = obj, gr, o
                for _ in range(WARM):
                    gr.replay()
            torch.cuda.synchronize()
            evs = {a[0]: [] for a in ARMS}
            names = [a[0] for a in ARMS]
            for it in range(ITERS):
                order = names[it % len(names):] + names[:it % len(names)]
                for a in order:
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[a].replay()
                    e1.record()
                    evs[a].append((e0, e1))
            torch.cuda.synchronize()
            key = f"{sname}_M{m}"
            rec = {}
            for a in names:
                t = sorted(e0.elapsed_time(e1) * 1e3 / c for e0, e1 in evs[a])
                med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
                st = objs[a].state(outs[a])
                eq = bool(torch.equal(st, eager["LK"]))
                ok = ok and eq
                rec[a] = {"us_med": round(med, 3), "us_p10": round(p10, 3), "us_p90": round(p90, 3), "eq_LK": eq}
            base_us = rec["LK"]["us_med"]
            deltas = {a: round(rec[a]["us_med"] - base_us, 3) for a in names}
            rec["delta_vs_LK_us"] = deltas
            detail[key] = rec
            log(f"{key}: med us " + " ".join(f"{a} {rec[a]['us_med']:.2f}" for a in names) +
                " | delta vs LK " + json.dumps(deltas) + f" | bitwise {all(rec[a]['eq_LK'] for a in names)}")
            del graphs, outs, objs
            torch.cuda.synchronize()
        del ws
        torch.cuda.empty_cache()
    print(json.dumps({"detail": detail, "bitwise_all_arms": ok,
                      "note": "STEP 0 probe: Triton stand-in producer, timing only"}))


main()
