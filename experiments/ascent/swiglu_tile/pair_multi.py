import json, os, statistics, subprocess, sys
BENCH = r"""
import json, os, statistics, sys
os.environ.setdefault('TRTLLM_ENABLE_PDL', '1')
import torch
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments', 'ascent', 'swiglu_epi'))
from mlp_common import EPS, GS_X, H, build_mlp, dev
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
L = 4
MS = [int(v) for v in os.environ.get('BENCH_MS', '8,16,24,32,512,2079,4127').split(',')]
WARM, ITERS = 10, 100
mlps = [build_mlp(100 + i, 1.0) for i in range(L)]
gammas = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
gs = torch.tensor([GS_X], dtype=torch.float32, device=dev)
scrub = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)
out = {}
for m in MS:
    gen = torch.Generator(device=dev).manual_seed(7 + m)
    x0 = (torch.randn((L, m, H), generator=gen, device=dev) * 0.7).to(torch.bfloat16)
    r0 = (torch.randn((L, m, H), generator=gen, device=dev) * 3.0).to(torch.bfloat16)
    xp, rp = x0.clone(), r0.clone()
    prefill = m >= 1024
    def layer(i):
        if prefill:
            q, sf = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(xp[i], rp[i], gammas[i], EPS, gs)
            return mlps[i](Fp4QuantizedTensor(q, sf))
        return mlps[i](xp[i])
    with autotune():
        layer(0)
    torch.cuda.synchronize()
    if prefill:
        def runner():
            return [layer(i) for i in range(L)]
    else:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                [layer(i) for i in range(L)]
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gouts = [layer(i) for i in range(L)]
        def runner(g=g):
            g.replay()
            return gouts
    ts = []
    for it in range(WARM + ITERS):
        xp.copy_(x0)
        rp.copy_(r0)
        scrub.zero_()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        o = runner()
        e1.record()
        e1.synchronize()
        if it >= WARM:
            ts.append(e0.elapsed_time(e1) * 1e3 / L)
    v = sorted(ts)
    n = len(v)
    med = statistics.median(v)
    out[str(m)] = {'med': med, 'p10': v[int(0.1 * (n - 1))], 'p90': v[int(0.9 * (n - 1))]}
    torch.cuda.synchronize()
    if not prefill:
        del g, gouts
    torch.cuda.empty_cache()
print(json.dumps(out))
"""
NPROC = int(os.environ.get('BENCH_NPROC', '5'))
runs = []
for r in range(NPROC):
    p = subprocess.run([sys.executable, '-c', BENCH], capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith('{')]
    if p.returncode != 0 or not lines:
        print(p.stderr[-3000:], flush=True)
        raise SystemExit('bench process failed')
    d = json.loads(lines[-1])
    runs.append(d)
    print('PROC', r, json.dumps({k: round(v['med'], 2) for k, v in d.items()}), flush=True)
batch = {}
for m in runs[0]:
    meds = [r[m]['med'] for r in runs]
    mm = statistics.median(meds)
    within = statistics.median([(r[m]['p90'] - r[m]['p10']) / r[m]['med'] for r in runs])
    across = (max(meds) - min(meds)) / mm
    batch[m] = {'kernel_ms': mm / 1e3, 'noise_floor': round(max(within, across), 4),
                'within_run_p90_p10': round(within, 4), 'across_process_range': round(across, 4),
                'process_medians_us': [round(x, 2) for x in meds]}
print(json.dumps({'batch': batch, 'output': {'class': 'exact'}}))
