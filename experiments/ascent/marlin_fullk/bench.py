import os, sys, json, time, subprocess
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
import tensorrt_llm._torch.custom_ops
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin
# Full-K dense Marlin (idea 136) STEP 0 / kernel pair bench. Brace-free: passed verbatim via python -c to both trees.
# Shape (5120,6144) = GDN out_proj / attn o_proj decode copy. Two graph variants per arm, cold L2 (16 distinct
# Marlin-packed weight copies, ~283 MB, chained in one graph), PDL on:
#   mm    : 16 back-to-back marlin_nvfp4_gemm calls (Marlin -> Marlin PDL edges)
#   chain : 16 x (gemma fused add RMSNorm -> Marlin -> gemma fused add RMSNorm), state restored by one copy
# Arms captured in one process and replayed alternately: A = TRTLLM_MARLIN_FULLK=0 (main's 188-CTA split-K launch),
# F = TRTLLM_MARLIN_FULLK=1 (80-CTA full-K launch). In the base tree the switch does not exist, so A = F = main.
dev = torch.device("cuda")
EPS, WARM, ITERS, COPIES = 1e-6, 10, 60, 16
N, K = 5120, 6144
MS = [1, 2, 4, 8]
ARMS = ["A", "F"]
T0 = time.time()
def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)
def git(args):
    try:
        return subprocess.run(["git"] + args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return "n/a " + str(e)
head = git(["rev-parse", "HEAD"])
dirty = git(["status", "--porcelain"])
print("TREE_COMMIT=" + head, "dirty_files=%d" % (len(dirty.splitlines()) if dirty else 0), flush=True)
def set_arm(a):
    os.environ["TRTLLM_MARLIN_FULLK"] = "0" if a == "A" else "1"
def build(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    gs = (448.0 * 6.0 / w.float().abs().amax()).float().reshape(1)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, (1.0 / gs).float(), n, k, 16)
def marlin(x, wt):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
        weight_global_scale=wt[2], bias=None, out_dtype=torch.bfloat16, size_n=N, size_k=K)
def norm(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)
class MM:
    def __init__(self, m, ws, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        self.x = (torch.randn(m, K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        self.ws = ws
    def run(self):
        return [marlin(self.x, w) for w in self.ws]
    def state(self, ys):
        return torch.cat([y.reshape(-1).clone() for y in ys])
class Chain:
    def __init__(self, m, ws, wn1, wn2, seed):
        self.m, self.ws, self.wn1, self.wn2 = m, ws, wn1, wn2
        c = len(ws)
        per = m * K * 2 + m * N
        g = torch.Generator(device=dev).manual_seed(seed)
        st0 = torch.zeros(c * per, device=dev, dtype=torch.bfloat16)
        for i in range(c):
            st0[i * per: i * per + 2 * m * K] = (torch.randn(2 * m * K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        self.st0 = st0
        self.st = st0.clone()
        self.x, self.r1, self.r2 = [], [], []
        for i in range(c):
            b = i * per
            self.x.append(self.st[b: b + m * K].view(m, K))
            self.r1.append(self.st[b + m * K: b + 2 * m * K].view(m, K))
            self.r2.append(self.st[b + 2 * m * K: b + per].view(m, N))
    def run(self):
        self.st.copy_(self.st0)
        ys = []
        for i in range(len(self.ws)):
            norm(self.x[i], self.r1[i], self.wn1)
            y = marlin(self.x[i], self.ws[i])
            norm(y, self.r2[i], self.wn2)
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
log("gpu", torch.cuda.get_device_name(), "pdl_env", os.environ.get("TRTLLM_ENABLE_PDL"))
base = build(N, K, 11 + N + K)
ws = [tuple(t.clone() for t in base) for _ in range(COPIES)]
del base
mb = sum(sum(t.numel() * t.element_size() for t in w) for w in ws) / 1e6
g = torch.Generator(device=dev).manual_seed(5)
wn1 = (0.1 * torch.randn(K, generator=g, device=dev)).to(torch.bfloat16)
wn2 = (0.1 * torch.randn(N, generator=g, device=dev)).to(torch.bfloat16)
log("copies", COPIES, "weight MB %.1f" % mb)
res = dict()
detail = dict()
bitwise_ok = True
for variant in ["mm", "chain"]:
    for m in MS:
        objs, graphs, outs, eager = dict(), dict(), dict(), dict()
        for a in ARMS:
            set_arm(a)
            obj = MM(m, ws, 100 + m) if variant == "mm" else Chain(m, ws, wn1, wn2, 100 + m)
            eager[a] = obj.state(obj.run())
            gr, o = capture(obj)
            objs[a], graphs[a], outs[a] = obj, gr, o
            for _ in range(WARM):
                gr.replay()
        torch.cuda.synchronize()
        evs = dict((a, []) for a in ARMS)
        for it in range(ITERS):
            for a in (ARMS if it % 2 == 0 else ARMS[::-1]):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[a].replay()
                e1.record()
                evs[a].append((e0, e1))
        torch.cuda.synchronize()
        key = "%s_M%d" % (variant, m)
        detail[key] = dict()
        for a in ARMS:
            t = sorted(e0.elapsed_time(e1) * 1e3 / COPIES for e0, e1 in evs[a])
            med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
            st = objs[a].state(outs[a])
            eq_eager = bool(torch.equal(st, eager[a]))
            eq_a = bool(torch.equal(st, eager["A"]))
            bitwise_ok = bitwise_ok and eq_eager and eq_a
            detail[key][a] = dict(us_med=med, us_p10=p10, us_p90=p90, noise=(p90 - p10) / med,
                                  replay_eq_eager=eq_eager, eq_armA=eq_a)
        dA, dF = detail[key]["A"], detail[key]["F"]
        gain = (dA["us_med"] - dF["us_med"]) / dA["us_med"]
        beyond = dF["us_p90"] < dA["us_p10"]
        detail[key]["F_vs_A_pct"] = gain * 100.0
        detail[key]["F_p90_below_A_p10"] = beyond
        log("%-5s M=%d per-call us  A med %.2f p10 %.2f p90 %.2f | F med %.2f p10 %.2f p90 %.2f | F vs A %+.2f%% | "
            "F.p90<A.p10 %s | bitwise F==A %s" % (variant, m, dA["us_med"], dA["us_p10"], dA["us_p90"], dF["us_med"],
            dF["us_p10"], dF["us_p90"], -gain * 100.0, beyond, dF["eq_armA"] and dF["replay_eq_eager"]))
        if variant == "mm":
            res[str(m)] = dict(kernel_ms=dF["us_med"] / 1e3, noise_floor=dF["noise"], p90_ms=dF["us_p90"] / 1e3,
                               mainarm_kernel_ms=dA["us_med"] / 1e3)
        del graphs, outs, objs
        torch.cuda.synchronize()
out = dict(batch=res)
out["output"] = dict(shape="(5120,6144) Marlin W4A16, 16-copy cold-L2 graph, PDL; kernel_ms = arm F per call (mm)",
                     detail=detail, bitwise_F_eq_A_and_eager=bitwise_ok, warmup=WARM, iters=ITERS, tree=head)
out["output"]["class"] = "exact"
print(json.dumps(out))
