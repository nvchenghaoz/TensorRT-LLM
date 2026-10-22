"""Idea 50 step 0a: exposed time of the two FusedAddRMSNorm sites per decode layer.

One-tree, one-process chain on synthetic tensors of the real shapes (no model):
per layer (M rows), L distinct layers in one CUDA graph (cold weights, ~207 MB/layer):

  out_proj Marlin W4A16 (5120 x 6144, BF16 in; workspace_mode as main:
      0 at M <= 16, 2 at M > 16)
  -> post_attention FlashInfer gemma add+RMSNorm (in place, H=5120)
  -> fp4_quantize -> gate_up nvfp4_gemm (34816 x 5120) -> fused_silu_mul_nvfp4
  -> down nvfp4_gemm (5120 x 17408)
  -> next_layer FlashInfer gemma add+RMSNorm
  -> in_proj_qkvz (16384 x 5120): Marlin W4A16 at M <= 16, else fp4_quantize +
     nvfp4_gemm (main's cap 16 for this shape).

Arms (each its own captured graph, replayed in rotating order, residual restored
from a pristine copy and a 256 MB READ flush before every replay, outside timing):
  A   full chain                   A2  full chain again (A/A)
  NP  without the post_attention norm (Marlin -> fp4_quantize directly)
  NN  without the next_layer norm (down -> qkvz directly)
exposed_post = A - NP, exposed_next = A - NN (per layer, median of per-iteration
paired differences); excess = exposed_post - exposed_next; A/A = A2 - A.
"""
import json
import os
import statistics
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4 as _fs  # noqa: E402,F401
from tensorrt_llm._torch.modules import linear as LIN  # noqa: E402

dev = torch.device("cuda")
H, I, VD, QKVZ = 5120, 17408, 6144, 16384
L = int(os.environ.get("S0_LAYERS", "10"))
WARM = int(os.environ.get("S0_WARM", "10"))
ITERS = int(os.environ.get("S0_ITERS", "150"))
EPS = 1e-6
ARMS = ("A", "NP", "NN", "A2")


def log(*a):
    print(*a, flush=True)


def marlin_copy(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=dev) * 0.02
    s = (w.abs().amax() / 448.0).float()
    wv = (w / s).to(torch.float8_e4m3fn).to(torch.bfloat16)
    del w
    amax = wv.abs().amax().float() * s
    gg = torch.exp2(torch.ceil(torch.log2(amax / 2688.0)))
    fp4, sf = torch.ops.trtllm.fp4_quantize(wv, (s / gg).reshape(1), 16, False, True)
    del wv
    return LIN._pack_nvfp4_weight_for_marlin(fp4, sf, gg.reshape(1).to(torch.float32), n, k, 16)


def nvfp4_w(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=g)
    wsf = torch.randint(0x28, 0x38, (n * k // 16, ), dtype=torch.uint8, device=dev, generator=g)
    return w, wsf


def marlin(x, c, n, k, mode):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, c[0], scale_a=None, scale_b=c[1], alpha=None,
                                              weight_global_scale=c[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=n, size_k=k,
                                              workspace_mode=mode)


def build_weights():
    W = dict(
        alpha=torch.tensor([0.002], dtype=torch.float32, device=dev),
        gs_x=torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device=dev),
        gs_h=torch.tensor([30.0], dtype=torch.float32, device=dev),
    )
    W["wo"] = [marlin_copy(H, VD, 1000 + i) for i in range(L)]
    W["wq"] = [marlin_copy(QKVZ, H, 2000 + i) for i in range(L)]
    W["wq4"] = [nvfp4_w(QKVZ, H, 2500 + i) for i in range(L)]
    W["wg"] = [nvfp4_w(2 * I, H, 3000 + i) for i in range(L)]
    W["wd"] = [nvfp4_w(H, I, 4000 + i) for i in range(L)]
    W["g1"] = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
    W["g2"] = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
    W["scrub"] = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev).normal_()
    W["sink"] = torch.empty((), dtype=torch.float32, device=dev)
    log(f"weights built (L={L})")
    return W


class Chain:

    def __init__(self, m, W):
        self.m, self.W = m, W
        self.mode = 2 if m > LIN._FP8_NVFP4_DECODE_MAX_ROWS else 0
        self.qkvz_marlin = m <= LIN._FP8_NVFP4_DECODE_MAX_ROWS
        gen = torch.Generator(device=dev).manual_seed(7 + m)
        self.xa = (torch.randn((L, m, VD), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        self.r0 = (torch.randn((m, H), generator=gen, device=dev) * 3.0).to(torch.bfloat16)
        self.res = self.r0.clone()
        with autotune():
            self.reset()
            self.layer(0, True, True)
        torch.cuda.synchronize()

    def reset(self):
        self.res.copy_(self.r0)

    def layer(self, i, post, nxt):
        W = self.W
        y = marlin(self.xa[i], W["wo"][i], H, VD, self.mode)
        if post:
            torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(y, self.res, W["g1"][i], EPS)
        q, sf = torch.ops.trtllm.fp4_quantize(y, W["gs_x"], 16, False)
        h = torch.ops.trtllm.nvfp4_gemm(q, W["wg"][i][0], sf, W["wg"][i][1], W["alpha"], torch.bfloat16)
        hq, hsf = torch.ops.trtllm.fused_silu_mul_nvfp4(h, W["gs_h"])
        d = torch.ops.trtllm.nvfp4_gemm(hq, W["wd"][i][0], hsf, W["wd"][i][1], W["alpha"], torch.bfloat16)
        if nxt:
            torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(d, self.res, W["g2"][i], EPS)
        if self.qkvz_marlin:
            return y, marlin(d, W["wq"][i], QKVZ, H, 0)
        dq, dsf = torch.ops.trtllm.fp4_quantize(d, W["gs_x"], 16, False)
        return y, torch.ops.trtllm.nvfp4_gemm(dq, W["wq4"][i][0], dsf, W["wq4"][i][1], W["alpha"],
                                              torch.bfloat16)

    def capture(self, post, nxt):

        def body():
            return [self.layer(i, post, nxt) for i in range(L)]

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.reset()
                body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = body()
        torch.cuda.synchronize()
        return gr, outs


def stats(v):
    v = sorted(v)
    n = len(v)
    med = statistics.median(v)
    return {"med": round(med, 3), "p10": round(v[int(0.1 * (n - 1))], 3),
            "p90": round(v[int(0.9 * (n - 1))], 3),
            "noise": round((v[int(0.9 * (n - 1))] - v[int(0.1 * (n - 1))]) / med, 4)}


def time_arms(ch, graphs, warm=WARM, iters=ITERS):
    arms = tuple(graphs)
    W = ch.W
    ts = {a: [] for a in arms}
    for it in range(warm + iters):
        r = it % len(arms)
        order = list(arms[r:] + arms[:r])
        if it % 2:
            order = order[::-1]
        evs = {}
        for arm in order:
            ch.reset()
            torch.sum(W["scrub"], dim=0, out=W["sink"])
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[arm][0].replay()
            e1.record()
            evs[arm] = (e0, e1)
        torch.cuda.synchronize()
        if it >= warm:
            for arm in arms:
                ts[arm].append(evs[arm][0].elapsed_time(evs[arm][1]) * 1e3 / L)
    return ts


def main():
    ms = [int(a) for a in sys.argv[1:]] or [4, 8, 12, 16, 24, 32]
    log(f"step0a L={L} warm={WARM} iters={ITERS} ms={ms} pdl={os.environ.get('TRTLLM_ENABLE_PDL')}")
    W = build_weights()
    results = {}
    for m in ms:
        ch = Chain(m, W)
        cfg = {"A": (True, True), "A2": (True, True), "NP": (False, True), "NN": (True, False)}
        graphs = {arm: ch.capture(*cfg[arm]) for arm in ARMS}
        log(f"M={m} captured (out_proj workspace_mode={ch.mode}, qkvz {'marlin' if ch.qkvz_marlin else 'w4a4'})")
        ts = time_arms(ch, graphs)
        outs = {}
        for arm in ("A", "A2"):
            ch.reset()
            graphs[arm][0].replay()
            torch.cuda.synchronize()
            outs[arm] = ([t.clone() for pair in graphs[arm][1] for t in pair], ch.res.clone())
        aa_bitwise = (all(torch.equal(a, b) for a, b in zip(outs["A"][0], outs["A2"][0]))
                      and torch.equal(outs["A"][1], outs["A2"][1]))

        def pdiff(a, b):
            return statistics.median([x - y for x, y in zip(ts[a], ts[b])])

        row = {a: stats(ts[a]) for a in ARMS}
        row["exposed_post_us"] = round(pdiff("A", "NP"), 3)
        row["exposed_post_us_A2"] = round(pdiff("A2", "NP"), 3)
        row["exposed_next_us"] = round(pdiff("A", "NN"), 3)
        row["exposed_next_us_A2"] = round(pdiff("A2", "NN"), 3)
        row["excess_us"] = round(row["exposed_post_us"] - row["exposed_next_us"], 3)
        row["excess_us_A2"] = round(row["exposed_post_us_A2"] - row["exposed_next_us_A2"], 3)
        row["aa_us"] = round(pdiff("A2", "A"), 3)
        row["aa_bitwise"] = aa_bitwise
        row["family"] = ("full_k" if m <= 8 else "kM16" if m <= 16 else "wide")
        results[str(m)] = row
        log("M", m, json.dumps(row))
        del graphs, outs, ch
        torch.cuda.empty_cache()
    log(json.dumps({
        "batch": {k: {"kernel_ms": v["A"]["med"] / 1e3, "noise_floor": v["A"]["noise"]}
                  for k, v in results.items()},
        "output": {"class": "exact"},
        "step0a": results,
    }))


if __name__ == "__main__":
    main()
