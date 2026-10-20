import hashlib, json, os, subprocess, sys, time
os.environ['TRTLLM_ENABLE_PDL'] = '1'
import torch
import tensorrt_llm
from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4 as fsm
# Branch 114 / idea 200 pair bench (BRACE-FREE: passed inline to run_gpu_pair, base lacks branch files).
# Per layer = the dense-MLP decode chain as served: fp4_quantize -> gate_up (34816,5120) cuda_core ->
# fused_silu_mul_nvfp4 -> down (5120,17408) cuda_core; L layers with distinct weights per graph (1.8 GB),
# 512 MiB READ flush before every replay, PDL on, arms rotated per iteration.
# Arms: mlp (default), mlp_aa (A/A copy), mlp_kill (TRTLLM_CUDA_CORE_NVFP4_M1_PREWAIT=0 at capture; on the
# base it is unknown, so it equals mlp), sd / sd_kill (silu -> down only).
def log(*a):
    print(*a, flush=True)
dev = torch.device('cuda')
L = 12
WARM = 10
ITERS = 150
ROUNDS = 3
MS = [1, 2, 4]
HASH_MS = [1, 2, 3, 4]
KILL = 'TRTLLM_CUDA_CORE_NVFP4_M1_PREWAIT'
try:
    hh = subprocess.run(['git', 'rev-parse', 'HEAD', 'HEAD^' + chr(123) + 'tree' + chr(125)], capture_output=True, text=True, timeout=20).stdout.split()
    dd = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    log('TREE_COMMIT=%s TREE=%s DIRTY=%d CWD=%s' % (hh[0] if hh else '?', hh[1] if len(hh) > 1 else '?', len(dd), os.getcwd()))
except Exception as e:
    log('TREE_COMMIT=? %r' % (e,))
try:
    log('GPU ' + subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active', '--format=csv,noheader'], capture_output=True, text=True, timeout=20).stdout.strip())
except Exception:
    pass
def sha(ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:20]
def q4(x, gs):
    return torch.ops.trtllm.fp4_quantize(x, gs, 16, False, True)
def gemm(aq, wq, asf, wsf, alpha):
    return torch.ops.trtllm.cuda_core_nvfp4_gemm(aq, wq, asf, wsf, alpha, None, torch.bfloat16, act_sf_swizzled=True)
def mkw(n, k, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    w = torch.randn(n, k, device=dev, generator=g).to(torch.bfloat16)
    w.view(n, k // 16, 16)[::5, ::11, :] = 0
    gs = (448.0 * 6.0 / w.float().abs().amax()).reshape(1).float()
    wq, wsf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return wq, wsf, gs
t0 = time.time()
GU = mkw(34816, 5120, 17)
DN = mkw(5120, 17408, 29)
guw = [GU[0]] + [GU[0].clone() for _ in range(L - 1)]
gusf = [GU[1]] + [GU[1].clone() for _ in range(L - 1)]
dnw = [DN[0]] + [DN[0].clone() for _ in range(L - 1)]
dnsf = [DN[1]] + [DN[1].clone() for _ in range(L - 1)]
WB_GU = GU[0].numel() + GU[1].numel()
WB_DN = DN[0].numel() + DN[1].numel()
flush_buf = torch.randint(0, 1 << 30, ((512 << 20) // 8,), device=dev, dtype=torch.int64)
torch.cuda.synchronize()
log('weights ready: %d layers, gate_up %.1f MB, down %.1f MB per layer (%.0fs)' % (L, WB_GU / 1e6, WB_DN / 1e6, time.time() - t0))
def acts(m):
    A = dict(x=[], gin=[], alg=[], gmid=[], ald=[], h=[])
    for i in range(L):
        g = torch.Generator(device=dev)
        g.manual_seed(1000 * m + i)
        x = (torch.randn(m, 5120, device=dev, generator=g) * 2.0).to(torch.bfloat16)
        x.view(m, 320, 16)[:, 3::7, :] = 0
        gs = (448.0 * 6.0 / x.float().abs().amax()).reshape(1).float()
        alg = (1.0 / (gs * GU[2])).float()
        aq, asf = q4(x, gs)
        h = gemm(aq, guw[i], asf, gusf[i], alg)
        a = h[:, :17408].float()
        s = (a * torch.sigmoid(a) * h[:, 17408:].float()).to(torch.bfloat16).float()
        gm = (448.0 * 6.0 / s.abs().amax().clamp_min(1e-6)).reshape(1).float()
        ald = (1.0 / (gm * DN[2])).float()
        A['x'].append(x); A['gin'].append(gs); A['alg'].append(alg); A['gmid'].append(gm); A['ald'].append(ald); A['h'].append(h.clone())
    torch.cuda.synchronize()
    return A
def make(A, kind):
    def mlp(i):
        aq, asf = q4(A['x'][i], A['gin'][i])
        h = gemm(aq, guw[i], asf, gusf[i], A['alg'][i])
        dq, dsf = torch.ops.trtllm.fused_silu_mul_nvfp4(h, A['gmid'][i])
        return gemm(dq, dnw[i], dsf, dnsf[i], A['ald'][i])
    def sd(i):
        dq, dsf = torch.ops.trtllm.fused_silu_mul_nvfp4(A['h'][i], A['gmid'][i])
        return gemm(dq, dnw[i], dsf, dnsf[i], A['ald'][i])
    return mlp if kind == 'mlp' else sd
def cap(fn, kill):
    if kill:
        os.environ[KILL] = '0'
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(L):
                fn(i)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = [fn(i) for i in range(L)]
        torch.cuda.synchronize()
    finally:
        os.environ.pop(KILL, None)
    return g, outs
def qt(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
hashes = dict()
for m in HASH_MS:
    A = acts(m)
    f = make(A, 'mlp')
    e = [f(i).clone() for i in range(L)]
    os.environ[KILL] = '0'
    ek = [f(i).clone() for i in range(L)]
    os.environ.pop(KILL, None)
    torch.cuda.synchronize()
    hashes['M%d' % m] = sha(e)
    hashes['M%d_kill' % m] = sha(ek)
    del A
log('eager mlp output hashes ' + json.dumps(hashes))
ARMS = ['mlp', 'mlp_aa', 'mlp_kill', 'sd', 'sd_kill']
batch = dict()
detail = dict()
for m in MS:
    A = acts(m)
    eager = dict(mlp=[make(A, 'mlp')(i).clone() for i in range(L)], sd=[make(A, 'sd')(i).clone() for i in range(L)])
    torch.cuda.synchronize()
    ts = dict((a, []) for a in ARMS)
    mism = dict((a, 0) for a in ARMS)
    for rnd in range(ROUNDS):
        graphs = dict()
        outs = dict()
        for a in ARMS:
            graphs[a], outs[a] = cap(make(A, a.split('_')[0]), a.endswith('kill'))
        for it in range(WARM + ITERS):
            sh = (it + rnd) % len(ARMS)
            order = ARMS[sh:] + ARMS[:sh]
            evs = []
            for a in order:
                flush_buf.sum()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[a].replay()
                e1.record()
                evs.append((a, e0, e1))
            torch.cuda.synchronize()
            if it >= WARM:
                for a, e0, e1 in evs:
                    ts[a].append(e0.elapsed_time(e1) / L)
        for a in ARMS:
            ref = eager[a.split('_')[0]]
            mism[a] += sum(int(not torch.equal(outs[a][i], ref[i])) for i in range(L))
        log('M=%d round %d: ' % (m, rnd) + ' '.join('%s=%.2fus' % (a, 1e3 * qt(ts[a][-ITERS:], 0.5)) for a in ARMS) + ' mism=' + json.dumps(mism) + ' (%.0fs)' % (time.time() - t0))
        del graphs, outs
        torch.cuda.synchronize()
    rec = dict()
    for a in ARMS:
        med = qt(ts[a], 0.5)
        rec[a] = dict(med_us=round(1e3 * med, 3), p10_us=round(1e3 * qt(ts[a], 0.1), 3), p90_us=round(1e3 * qt(ts[a], 0.9), 3))
    pk = [(x - y) * 1e3 for x, y in zip(ts['mlp_kill'], ts['mlp'])]
    paa = [(x - y) * 1e3 for x, y in zip(ts['mlp_aa'], ts['mlp'])]
    psd = [(x - y) * 1e3 for x, y in zip(ts['sd_kill'], ts['sd'])]
    rec['paired_kill_minus_default_us'] = dict(mlp=[round(qt(pk, p), 3) for p in (0.1, 0.5, 0.9)], sd=[round(qt(psd, p), 3) for p in (0.1, 0.5, 0.9)], aa=[round(qt(paa, p), 3) for p in (0.1, 0.5, 0.9)])
    rec['bitwise_mismatch_vs_eager'] = mism
    detail['M%d' % m] = rec
    med = qt(ts['mlp'], 0.5)
    batch[str(m)] = dict(kernel_ms=round(med, 6), noise_floor=round((qt(ts['mlp'], 0.9) - qt(ts['mlp'], 0.1)) / med, 4), p90_ms=round(qt(ts['mlp'], 0.9), 6), sd_ms=round(qt(ts['sd'], 0.5), 6), kill_ms=round(qt(ts['mlp_kill'], 0.5), 6))
    log('RESULT M=%d ' % m + json.dumps(rec))
    del A, eager
    torch.cuda.empty_cache()
allok = all(v == 0 for r in detail.values() for v in r['bitwise_mismatch_vs_eager'].values())
print(json.dumps(dict(batch=batch, output=dict([('class', 'exact')]), hashes=hashes, graph_bitwise_ok=allok, detail=detail, layers=L, iters=ITERS * ROUNDS)), flush=True)
