# Idea 28 stage 2: real-producer chains per site (candidate tree), CUDA graph + PDL, cold L2 (C weight copies).
#   site qkvz / qkv : FlashInfer gemma fused add-RMSNorm (next_layer/input norm, H=5120) -> Marlin -> trailing norm
#   site outproj    : Triton gated RMSNorm (rms_norm_gated_token_major, 48 heads x 128) -> Marlin (5120,6144)
#                     -> trailing gemma fused add-RMSNorm
# Producer: S = stock trigger (after the stores; main), E = early trigger (after the loads land: FlashInfer overlay
# module fused_add_rmsnorm_early_trigger.py / TRTLLM_GATED_NORM_EARLY_TRIGGER=1).
# Marlin: K = TRTLLM_MARLIN_PREWAIT_B=0 (main's wait-first prologue), P = pre-wait L2 prefetch (stage 1).
# Arms rotated every iteration: SK (= main), SP (stage 1 only), EK, EP (stage 1 + 2), SK2 and EP2 (A/A copies).
# Every arm's chain state must be bitwise equal to SK's.
import hashlib
import json
import os
import time

os.environ["TRTLLM_ENABLE_PDL"] = "1"
import torch  # noqa: E402

import flashinfer.norm.kernels.fused_add_rmsnorm as fi_pinned  # noqa: E402
import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
import tensorrt_llm._torch.modules.mamba.layernorm_gated  # noqa: E402,F401
from flashinfer.norm.kernels.fused_add_rmsnorm_early_trigger import (  # noqa: E402
    FusedAddRMSNormEarlyTriggerKernel, gemma_fused_add_rmsnorm_early_trigger)
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin  # noqa: E402

dev = torch.device("cuda")
EPS, WARM, ITERS = 1e-6, 10, 60
HEADS, HD = 48, 128
SITES = [("qkvz", 16384, 5120, 6, [4, 8, 12, 16]), ("qkv", 14336, 5120, 7, [4, 8, 12, 16, 24, 32]),
         ("outproj", 5120, 6144, 16, [4, 8, 12, 16, 24, 32])]
ARMS = [("SK", False, False), ("SP", False, True), ("EK", True, False), ("EP", True, True), ("SK2", False, False),
        ("EP2", True, True)]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


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
                                              size_n=n, size_k=k, workspace_mode=2 if x.shape[0] > 16 else 0)


def norm_stock(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)


def norm_early(x, r, w):
    gemma_fused_add_rmsnorm_early_trigger(x, r, w, EPS, enable_pdl=True)


class Chain:
    def __init__(self, site, n, k, m, ws, wn1, wn2, seed, early):
        self.site, self.n, self.k, self.m, self.ws, self.wn1, self.wn2, self.early = (site, n, k, m, ws, wn1, wn2,
                                                                                      early)
        c = len(ws)
        # per copy: x (m*k), r1 (m*k), z (m*k, gated site), r2 (m*n)
        per = 3 * m * k + m * n
        self.per = per
        g = torch.Generator(device=dev).manual_seed(seed)
        st0 = torch.zeros(c * per, device=dev, dtype=torch.bfloat16)
        for i in range(c):
            st0[i * per: i * per + 3 * m * k] = (torch.randn(3 * m * k, generator=g, device=dev) * 0.5).to(
                torch.bfloat16)
        self.st0 = st0
        self.st = st0.clone()
        self.x, self.r1, self.z, self.r2 = [], [], [], []
        for i in range(c):
            b = i * per
            self.x.append(self.st[b: b + m * k].view(m, k))
            self.r1.append(self.st[b + m * k: b + 2 * m * k].view(m, k))
            self.z.append(self.st[b + 2 * m * k: b + 3 * m * k].view(m, k))
            self.r2.append(self.st[b + 3 * m * k: b + per].view(m, n))

    def run(self):
        self.st.copy_(self.st0)
        ys = []
        for i in range(len(self.ws)):
            if self.site == "outproj":
                os.environ["TRTLLM_GATED_NORM_EARLY_TRIGGER"] = "1" if self.early else "0"
                a = torch.ops.trtllm.rms_norm_gated_token_major(
                    self.x[i].view(self.m * HEADS, HD), self.z[i].view(self.m, HEADS, HD), self.wn1, EPS)
                a = a.view(self.m, self.k)
            else:
                (norm_early if self.early else norm_stock)(self.x[i], self.r1[i], self.wn1)
                a = self.x[i]
            y = marlin(a, self.ws[i], self.n, self.k)
            norm_stock(y, self.r2[i], self.wn2)
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
    pinned_src = open(fi_pinned.__file__).read()
    prov = {"pinned_module_file": fi_pinned.__file__,
            "pinned_has_overlay_mark": ("early" in pinned_src.lower() and "EarlyTrigger" in pinned_src),
            "pinned_sha": hashlib.sha256(pinned_src.encode()).hexdigest()[:16],
            "overlay_is_subclass": issubclass(FusedAddRMSNormEarlyTriggerKernel, fi_pinned.FusedAddRMSNormKernel)}
    log("provenance", json.dumps(prov))
    detail = {}
    ok = True
    for site, n, k, c, rows in SITES:
        base = build(n, k, 11 + n + k)
        ws = [tuple(t.clone() for t in base) for _ in range(c)]
        del base
        g = torch.Generator(device=dev).manual_seed(5)
        wn1 = (0.1 * torch.randn(HD if site == "outproj" else k, generator=g, device=dev)).to(torch.bfloat16)
        wn2 = (0.1 * torch.randn(n, generator=g, device=dev)).to(torch.bfloat16)
        for m in rows:
            objs, graphs, outs, eager = {}, {}, {}, {}
            for name, early, prewait in ARMS:
                os.environ["TRTLLM_MARLIN_PREWAIT_B"] = "1" if prewait else "0"
                obj = Chain(site, n, k, m, ws, wn1, wn2, 100 + m, early)
                eager[name] = obj.state(obj.run())
                gr, o = capture(obj)
                objs[name], graphs[name], outs[name] = obj, gr, o
                for _ in range(WARM):
                    gr.replay()
            torch.cuda.synchronize()
            names = [a[0] for a in ARMS]
            evs = {a: [] for a in names}
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
            key = f"{site}_M{m}"
            rec = {}
            for a in names:
                t = sorted(e0.elapsed_time(e1) * 1e3 / c for e0, e1 in evs[a])
                med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
                st = objs[a].state(outs[a])
                eq = bool(torch.equal(st, eager["SK"])) and bool(torch.equal(eager[a], eager["SK"]))
                ok = ok and eq
                rec[a] = {"us_med": round(med, 3), "us_p10": round(p10, 3), "us_p90": round(p90, 3), "eq_SK": eq,
                          "hash": sha(st)}
            base_us = rec["SK"]["us_med"]
            rec["delta_vs_SK_us"] = {a: round(rec[a]["us_med"] - base_us, 3) for a in names}
            rec["aa_SK2_minus_SK"] = round(rec["SK2"]["us_med"] - base_us, 3)
            rec["aa_EP2_minus_EP"] = round(rec["EP2"]["us_med"] - rec["EP"]["us_med"], 3)
            detail[key] = rec
            log(f"{key}: delta vs SK {json.dumps(rec['delta_vs_SK_us'])} | SK med {base_us:.2f} us | "
                f"bitwise {all(rec[a]['eq_SK'] for a in names)}")
            del graphs, outs, objs
            torch.cuda.synchronize()
        del ws
        torch.cuda.empty_cache()
    os.environ.pop("TRTLLM_GATED_NORM_EARLY_TRIGGER", None)
    print(json.dumps({"provenance": prov, "detail": detail, "bitwise_all_arms": ok,
                      "note": "stage-2 per-site chain A/B, real producers; delta = arm median - SK median (us/layer)"}))


main()
