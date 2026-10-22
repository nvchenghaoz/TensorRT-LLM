import os, sys, json, hashlib, time, subprocess
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
import tensorrt_llm._torch.custom_ops
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin
# Idea 28 stage 1 pair bench (brace-free: passed verbatim via python -c to both trees).
# Per layer, one CUDA graph node chain: gemma fused add RMSNorm (FlashInfer, PDL) writing Marlin's BF16 input ->
# trtllm::marlin_nvfp4_gemm (PDL) -> gemma fused add RMSNorm reading Marlin's output. Cold L2: C distinct
# Marlin-packed weight copies per shape (>= 250 MB) chained in one graph; all mutable state restored by one copy.
# Arms captured in one process and replayed alternately (order flipped every iteration):
#   P = default (candidate: B / weight-scale cp.async issued before griddepcontrol.wait)
#   K = TRTLLM_MARLIN_PREWAIT_B=0 (candidate: main's wait-first prologue).
# In the base tree the switch does not exist, so P = K = main (an in-process A/A).
# batch[M].kernel_ms = arm P chain time per decode step, weighted by Marlin calls per step:
#   qkvz (16384,5120) x48 (rows <= 16 only) + qkv (14336,5120) x16 + out_proj/o_proj (5120,6144) x64.
dev = torch.device("cuda")
EPS, WARM, ITERS = 1e-6, 10, 60
SHAPES = [("qkvz", 16384, 5120, 6, 48), ("qkv", 14336, 5120, 7, 16), ("outproj", 5120, 6144, 16, 64)]
MS = [4, 8, 12, 16, 24, 32]
# Rows 17-32: only qkv and out_proj/o_proj run Marlin (workspace_mode 2, FP32 split-K reduce); qkvz runs W4A4 there.
def rows_for(sname):
    return MS if sname != "qkvz" else [m for m in MS if m <= 16]
def mode_for(m):
    return 2 if m > 16 else 0
ARMS = ["P", "K"]
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
    os.environ["TRTLLM_MARLIN_PREWAIT_B"] = "1" if a == "P" else "0"
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
        weight_global_scale=wt[2], bias=None, out_dtype=torch.bfloat16, size_n=n, size_k=k,
        workspace_mode=mode_for(x.shape[0]))
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
detail = dict()
hashes = dict()
per_step = dict((str(m), dict(P=0.0, K=0.0)) for m in MS)
noise = dict((str(m), dict(P=0.0, K=0.0)) for m in MS)
p90s = dict((str(m), dict(P=0.0, K=0.0)) for m in MS)
bitwise_ok = True
for sname, n, k, c, calls in SHAPES:
    base = build(n, k, 11 + n + k)
    ws = [tuple(t.clone() for t in base) for _ in range(c)]
    del base
    mb = sum(sum(t.numel() * t.element_size() for t in w) for w in ws) / 1e6
    g = torch.Generator(device=dev).manual_seed(5)
    wn1 = (0.1 * torch.randn(k, generator=g, device=dev)).to(torch.bfloat16)
    wn2 = (0.1 * torch.randn(n, generator=g, device=dev)).to(torch.bfloat16)
    log("shape", sname, n, k, "copies", c, "weight MB %.1f" % mb)
    for m in rows_for(sname):
        objs, graphs, outs, eager = dict(), dict(), dict(), dict()
        for a in ARMS:
            set_arm(a)
            obj = Chain(n, k, m, ws, wn1, wn2, 100 + m)
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
        key = "%s_M%d" % (sname, m)
        detail[key] = dict()
        for a in ARMS:
            t = sorted(e0.elapsed_time(e1) * 1e3 / c for e0, e1 in evs[a])
            med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
            st = objs[a].state(outs[a])
            eq_eager = bool(torch.equal(st, eager[a]))
            eq_k = bool(torch.equal(st, eager["K"]))
            bitwise_ok = bitwise_ok and eq_eager and eq_k
            detail[key][a] = dict(us_med=med, us_p10=p10, us_p90=p90, noise=(p90 - p10) / med,
                                  replay_eq_eager=eq_eager, eq_armK=eq_k)
            hashes[key + "_" + a] = h(st)
            per_step[str(m)][a] += med * calls / 1e3
            p90s[str(m)][a] += p90 * calls / 1e3
            noise[str(m)][a] = max(noise[str(m)][a], (p90 - p10) / med)
        dP, dK = detail[key]["P"], detail[key]["K"]
        detail[key]["P_minus_K_us"] = dP["us_med"] - dK["us_med"]
        log("%-7s M=%2d per-layer us  P med %.2f p10 %.2f p90 %.2f | K med %.2f p10 %.2f p90 %.2f | P-K %+.2f us | "
            "bitwise P==K==eager %s | hash %s" % (sname, m, dP["us_med"], dP["us_p10"], dP["us_p90"], dK["us_med"],
            dK["us_p10"], dK["us_p90"], dP["us_med"] - dK["us_med"], dP["eq_armK"] and dP["replay_eq_eager"]
            and dK["replay_eq_eager"], hashes[key + "_P"]))
        del graphs, outs, objs
        torch.cuda.synchronize()
        # Cross-tree bitwise evidence: eager Marlin output hashes, served launches (full_k / kM16Table) and the
        # default split-K launch, for both arms (P and K must match each other and the base tree's hashes).
        xg = torch.Generator(device=dev).manual_seed(7 + m)
        xin = (torch.randn(m, k, generator=xg, device=dev) * 0.5).to(torch.bfloat16)
        for fam in ["served", "splitk"]:
            for flag in ["TRTLLM_MARLIN_FULLK", "TRTLLM_MARLIN_INPROJ_FULLK", "TRTLLM_MARLIN_M16_TABLE"]:
                os.environ[flag] = "1" if fam == "served" else "0"
            for a in ARMS:
                set_arm(a)
                o1 = marlin(xin, ws[0], n, k).clone()
                o2 = marlin(xin, ws[1], n, k).clone()
                torch.cuda.synchronize()
                bitwise_ok = bitwise_ok and bool(torch.equal(o1, o2))
                hashes["eager_%s_M%d_%s_%s" % (sname, m, fam, a)] = h(o1)
            same = hashes["eager_%s_M%d_%s_P" % (sname, m, fam)] == hashes["eager_%s_M%d_%s_K" % (sname, m, fam)]
            bitwise_ok = bitwise_ok and same
        for flag in ["TRTLLM_MARLIN_FULLK", "TRTLLM_MARLIN_INPROJ_FULLK", "TRTLLM_MARLIN_M16_TABLE"]:
            os.environ.pop(flag, None)
    del ws
    torch.cuda.empty_cache()
res = dict()
for m in MS:
    s = str(m)
    res[s] = dict(kernel_ms=per_step[s]["P"], noise_floor=noise[s]["P"], p90_ms=p90s[s]["P"],
                  armK_kernel_ms=per_step[s]["K"], armK_noise=noise[s]["K"])
    log("M=%2d chain ms/step  P %.4f  K %.4f  P-K %+.4f ms (%+.2f%%)" % (m, per_step[s]["P"], per_step[s]["K"],
        per_step[s]["P"] - per_step[s]["K"], 100.0 * (per_step[s]["P"] - per_step[s]["K"]) / per_step[s]["K"]))
out = dict(batch=res)
out["output"] = dict(measure="norm->Marlin->norm chain per decode step (48 qkvz + 16 qkv + 64 out_proj), "
                     "cold L2, CUDA graph, PDL; kernel_ms = arm P", detail=detail, hashes=hashes,
                     bitwise_arms_and_eager=bitwise_ok, warmup=WARM, iters=ITERS, tree=head)
out["output"]["class"] = "exact"
print(json.dumps(out))
