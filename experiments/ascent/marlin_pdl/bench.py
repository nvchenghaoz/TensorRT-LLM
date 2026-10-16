import os, sys, json, hashlib, time
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
import tensorrt_llm._torch.custom_ops
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin
# Marlin PDL + persistent workspace STEP 1 bench (brace-free: passed verbatim via python -c to both trees).
# Per layer, one CUDA graph: gemma fused add RMSNorm (FlashInfer, PDL) writing Marlin's BF16 input ->
# trtllm::marlin_nvfp4_gemm -> gemma fused add RMSNorm reading Marlin's output (in place, residual r2).
# Cold L2: C distinct Marlin-packed copies per shape (>= 256 MB) chained in one graph. All mutable
# state lives in one flat buffer restored by a single copy at the start of every replay.
# Arms captured in one process, replayed alternately: A = both switches 0, W = workspace only, B = both on.
# In the base tree the switches do not exist, so A = W = B = main.
dev = torch.device("cuda")
EPS, WARM, ITERS = 1e-6, 10, 60
SHAPES = [("qkvz", 16384, 5120, 6), ("outproj", 5120, 6144, 15), ("qkv", 14336, 5120, 7)]
MS = [1, 2, 4]
ARMS = [("A", "0", "0"), ("W", "1", "0"), ("B", "1", "1")]
HAS = hasattr(torch.ops.trtllm, "marlin_nvfp4_persistent_workspace_snapshot")
def log(*a):
    print(*a, flush=True)
def set_arm(a):
    os.environ["TRTLLM_MARLIN_PERSISTENT_WORKSPACE"] = a[1]
    os.environ["TRTLLM_MARLIN_PDL"] = a[2]
def h(t):
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
        weight_global_scale=wt[2], bias=None, out_dtype=torch.bfloat16, size_n=n, size_k=k)
def norm(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)
class Chain:
    def __init__(self, n, k, m, ws, wn1, wn2, seed):
        self.n, self.k, self.m, self.ws, self.wn1, self.wn2 = n, k, m, ws, wn1, wn2
        c = len(ws)
        per = m * k * 2 + m * n
        g = torch.Generator(device=dev).manual_seed(seed)
        st0 = torch.zeros(c * per, device=dev, dtype=torch.bfloat16)
        for i in range(c):
            st0[i * per: i * per + 2 * m * k] = (torch.randn(2 * m * k, generator=g, device=dev) * 0.5).to(torch.bfloat16)
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
            norm(self.x[i], self.r1[i], self.wn1)
            y = marlin(self.x[i], self.ws[i], self.n, self.k)
            norm(y, self.r2[i], self.wn2)
            ys.append(y)
        return ys
torch.manual_seed(0)
log("has_switches", HAS, "pdl_env", os.environ.get("TRTLLM_ENABLE_PDL"), "gpu", torch.cuda.get_device_name())
res = dict()
detail = dict()
hashes = dict()
bitwise_ok = True
t_start = time.time()
for sname, n, k, c in SHAPES:
    base = build(n, k, 11 + n + k)
    ws = [tuple(t.clone() for t in base) for _ in range(c)]
    del base
    mb = sum(sum(t.numel() * t.element_size() for t in w) for w in ws) / 1e6
    g = torch.Generator(device=dev).manual_seed(5)
    wn1 = (0.1 * torch.randn(k, generator=g, device=dev)).to(torch.bfloat16)
    wn2 = (0.1 * torch.randn(n, generator=g, device=dev)).to(torch.bfloat16)
    log("shape", sname, n, k, "copies", c, "weight MB %.1f" % mb, "t=%.0fs" % (time.time() - t_start))
    for m in MS:
        chains, graphs, outs, eager = dict(), dict(), dict(), dict()
        for a in ARMS:
            set_arm(a)
            ch = Chain(n, k, m, ws, wn1, wn2, 100 + m)
            ys = ch.run()
            eager[a[0]] = (torch.cat([r.clone() for r in ch.r2]), torch.cat([y.clone() for y in ys]))
            s0 = torch.cuda.Stream()
            s0.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s0):
                for _ in range(3):
                    ch.run()
            torch.cuda.current_stream().wait_stream(s0)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                o = ch.run()
            chains[a[0]] = ch
            graphs[a[0]] = gr
            outs[a[0]] = o
            for _ in range(WARM):
                gr.replay()
        torch.cuda.synchronize()
        evs = dict((a[0], []) for a in ARMS)
        for it in range(ITERS):
            for a in ARMS:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[a[0]].replay()
                e1.record()
                evs[a[0]].append((e0, e1))
        torch.cuda.synchronize()
        key = "%s_M%d" % (sname, m)
        detail[key] = dict()
        for a in ARMS:
            t = sorted(e0.elapsed_time(e1) / c for e0, e1 in evs[a[0]])
            med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
            ch = chains[a[0]]
            r2 = torch.cat([r.clone() for r in ch.r2])
            yy = torch.cat([y.clone() for y in outs[a[0]]])
            eq_eager = bool(torch.equal(r2, eager[a[0]][0]) and torch.equal(yy, eager[a[0]][1]))
            eq_a = bool(torch.equal(r2, eager["A"][0]) and torch.equal(yy, eager["A"][1]))
            bitwise_ok = bitwise_ok and eq_eager and eq_a
            detail[key][a[0]] = dict(us_med=med * 1e3, us_p10=p10 * 1e3, us_p90=p90 * 1e3,
                                     noise=(p90 - p10) / med, replay_eq_eager=eq_eager, eq_armA=eq_a)
            hashes[key + "_" + a[0]] = h(r2) + ":" + h(yy)
        dA = detail[key]["A"]["us_med"]
        detail[key]["B_minus_A_us"] = detail[key]["B"]["us_med"] - dA
        detail[key]["W_minus_A_us"] = detail[key]["W"]["us_med"] - dA
        log("%s M=%d per-layer us med A %.2f W %.2f B %.2f | p90 A %.2f W %.2f B %.2f | B-A %.2f W-A %.2f | bitwise %s" % (
            sname, m, dA, detail[key]["W"]["us_med"], detail[key]["B"]["us_med"], detail[key]["A"]["us_p90"],
            detail[key]["W"]["us_p90"], detail[key]["B"]["us_p90"], detail[key]["B_minus_A_us"],
            detail[key]["W_minus_A_us"], all(detail[key][x[0]]["replay_eq_eager"] and detail[key][x[0]]["eq_armA"] for x in ARMS)))
        if sname == "qkvz":
            b = detail[key]["B"]
            res[str(m)] = dict(kernel_ms=b["us_med"] / 1e3, noise_floor=b["noise"], p90_ms=b["us_p90"] / 1e3)
        del graphs, outs, chains
        torch.cuda.synchronize()
    del ws
    torch.cuda.empty_cache()
for m in [64, 129]:
    for sname, n, k, c in SHAPES[:2]:
        base = build(n, k, 11 + n + k)
        x = (torch.randn(m, k, device=dev, generator=torch.Generator(device=dev).manual_seed(m)) * 0.5).to(torch.bfloat16)
        outs_e = dict()
        for a in ARMS:
            set_arm(a)
            outs_e[a[0]] = [marlin(x, base, n, k).clone() for _ in range(3)]
        torch.cuda.synchronize()
        ok = all(torch.equal(o, outs_e["A"][0]) for a in ARMS for o in outs_e[a[0]])
        bitwise_ok = bitwise_ok and ok
        hashes["eager_%s_M%d" % (sname, m)] = h(outs_e["A"][0]) + ":" + h(outs_e["B"][0])
        log("eager %s M=%d arms bitwise equal %s" % (sname, m, ok))
ws_zero = None
if HAS:
    torch.cuda.synchronize()
    snap = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device=dev))
    ws_zero = bool((snap == 0).all().item())
    log("persistent workspace bound ints", snap.numel(), "all zero", ws_zero)
out = dict(batch=res)
out["output"] = dict(shape_primary="qkvz (16384,5120) per-layer chain, arm B", detail=detail, hashes=hashes,
                     bitwise_arms_and_eager=bitwise_ok, workspace_all_zero=ws_zero, has_switches=HAS,
                     warmup=WARM, iters=ITERS)
out["output"]["class"] = "exact"
print(json.dumps(out))
