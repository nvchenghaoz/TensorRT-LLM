import json, subprocess, hashlib, time
import torch
import tensorrt_llm._torch.modules.fla.flashinfer_chunk as fc
F = torch.nn.functional
NEW = hasattr(fc, 'sm120_pool_io_eligible')
tree = subprocess.run(['git', 'rev-parse', 'HEAD^' + chr(123) + 'tree' + chr(125)], capture_output=True, text=True).stdout.strip()
print('TREE=' + tree, 'NEW_POOL_PATH', NEW, flush=True)
HQ, HV, D = 16, 48, 128
SSM = HV * D * D
STRIDE = 817152
GAP = STRIDE - SSM
NB = 40
dev = 'cuda'
g0 = torch.Generator(device='cuda'); g0.manual_seed(4242)
buf = (torch.randn(NB, STRIDE, generator=g0, device=dev) * 0.05).to(torch.bfloat16)
pool = buf[:, GAP:].view(NB, HV, D, D)
pristine = buf.clone()
ws = (torch.empty(NB, HV, D, D, dtype=torch.float32, device=dev), torch.empty(NB, HV, D, D, dtype=torch.float32, device=dev))
fl = torch.ones(64 * 1024 * 1024, device=dev)
def mk(sl):
    g = torch.Generator(device='cuda'); g.manual_seed(777)
    T = sum(sl); N = len(sl)
    q = F.normalize(torch.randn(1, T, HQ, D, generator=g, device=dev), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, T, HQ, D, generator=g, device=dev), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, T, HV, D, generator=g, device=dev).to(torch.bfloat16)
    glog = torch.log(torch.exp(-F.softplus(torch.randn(1, T, HV, generator=g, device=dev) - 6.0))).contiguous()
    beta = torch.sigmoid(torch.randn(1, T, HV, generator=g, device=dev)).to(torch.bfloat16)
    cu = torch.tensor([0] + [sum(sl[:j + 1]) for j in range(N)], dtype=torch.int64, device=dev)
    gp = torch.Generator(); gp.manual_seed(99 + N)
    idx = torch.randperm(NB, generator=gp)[:N].to(torch.int32).to(dev)
    nl = sum(1 for x in sl if x > 1)
    out = torch.empty(1, T, HV, D, dtype=torch.bfloat16, device=dev)
    def full():
        fc.chunk_gated_delta_rule(q, k, v, glog, beta, scale=D ** -0.5, initial_state=pool, initial_state_indices=idx, inplace_indexed_state_update=True, output_final_state=False, cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=False, output=out, state_workspace=ws, num_long_seqs=nl)
    def gather():
        fc.gather_cast_vk_to_fp32_vk(pool, idx, out_dtype=torch.float32, output=ws[0][:N])
    def scatter():
        fc.cast_scatter_fp32_vk_to_vk(ws[1][:N], pool, idx)
    return full, gather, scatter, out
def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:12]
def q3(xs):
    xs = sorted(xs); n = len(xs)
    return xs[int(round(0.1 * (n - 1)))], xs[int(round(0.5 * (n - 1)))], xs[int(round(0.9 * (n - 1)))]
S = [('8', [2048] + [1] * 7), ('16', [2048] + [1] * 15), ('32', [2048] + [1] * 31), ('1x2048', [2048]), ('2x2048+30', [2048, 2048] + [1] * 30)]
res = dict()
for name, sl in S:
    full, gather, scatter, out = mk(sl)
    buf.copy_(pristine); full(); torch.cuda.synchronize()
    sh_o = sha(out); sh_p = sha(buf)
    arms = [('a', full), ('b', full), ('gather', gather), ('scatter', scatter)]
    ts = dict((a, []) for a, _ in arms)
    for it in range(70):
        order = arms if it % 2 == 0 else list(reversed(arms))
        for a, fn in order:
            buf.copy_(pristine)
            torch.cuda._sleep(2000000); torch.sum(fl)
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); e1.synchronize()
            if it >= 10:
                ts[a].append(e0.elapsed_time(e1))
    r = dict()
    for a in ts:
        p10, med, p90 = q3(ts[a])
        r[a] = dict(median_ms=med, p10_ms=p10, p90_ms=p90)
    pa, ma, qa = q3(ts['a'])
    res[name] = dict(kernel_ms=ma, p10_ms=pa, p90_ms=qa, noise_floor=(qa - pa) / ma, aa_gap=abs(r['b']['median_ms'] / ma - 1.0), arms=r, sha_out=sh_o, sha_pool=sh_p, n=60)
    print(name, 'full a %.1f us  b %.1f us  | gather %.1f us  scatter %.1f us  | noise %.3f  sha out %s pool %s' % (ma * 1e3, r['b']['median_ms'] * 1e3, r['gather']['median_ms'] * 1e3, r['scatter']['median_ms'] * 1e3, (qa - pa) / ma, sh_o, sh_p), flush=True)
batch = dict((k, res[k]) for k in ('8', '16', '32'))
print(json.dumps(dict(batch=batch, shapes=res, new_pool_path=NEW, output=dict([('class', 'exact')]))), flush=True)
