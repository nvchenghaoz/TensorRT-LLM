# Brace-free body of the run_gpu_pair command (fact 1213: the base arm lacks
# branch files, so the pair passes this text inline via python -c). Kept here
# only so the exact timed code is committed; see bench.py for probe/check.
import json, inspect, subprocess, time, os
import torch
import flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 as m
F = torch.nn.functional
tree = subprocess.run(['git', 'rev-parse', 'HEAD^' + chr(123) + 'tree' + chr(125)], capture_output=True, text=True).stdout.strip()
print('TREE=' + tree, flush=True)
acc = 'num_long_seqs' in inspect.signature(m.delta_rule_prefill_dsl).parameters
print('VSPLIT_API', acc, 'KILL', os.environ.get('FLASHINFER_GDN_SM120_VSPLIT'), flush=True)
fl = torch.ones(64 * 1024 * 1024, device='cuda')
def mk(sl):
    g = torch.Generator(device='cuda'); g.manual_seed(777)
    T = sum(sl); N = len(sl)
    q = F.normalize(torch.randn(T, 16, 128, generator=g, device='cuda'), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(T, 16, 128, generator=g, device='cuda'), dim=-1).to(torch.bfloat16)
    v = torch.randn(T, 48, 128, generator=g, device='cuda').to(torch.bfloat16)
    a = torch.exp(-F.softplus(torch.randn(T, 48, generator=g, device='cuda') - 6.0)).contiguous()
    b = torch.sigmoid(torch.randn(T, 48, generator=g, device='cuda')).contiguous()
    i = (torch.randn(N, 48, 128, 128, generator=g, device='cuda') * 0.05).contiguous()
    cu = torch.tensor([0] + [sum(sl[:j + 1]) for j in range(N)], dtype=torch.int64, device='cuda')
    o = torch.empty(T, 48, 128, dtype=torch.bfloat16, device='cuda')
    s = torch.empty(N, 48, 128, 128, dtype=torch.float32, device='cuda')
    kw = dict(num_long_seqs=sum(1 for x in sl if x > 1)) if acc else dict()
    return lambda: m.delta_rule_prefill_dsl(o, s, q, k, v, i, a, b, cu, 128 ** -0.5, **kw)
S = [('1', [2048]), ('2', [2048, 1]), ('4', [2048, 1, 1, 1]), ('1x512', [512]), ('1x1024', [1024]), ('1x3072', [3072]), ('2x2048', [2048, 2048]), ('3x2048', [2048] * 3), ('3x2048+1', [2048, 2048, 2048, 1]), ('4x2048', [2048] * 4)]
res = dict()
for name, sl in S:
    fn = mk(sl)
    for _ in range(10):
        torch.sum(fl); fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(60):
        torch.cuda._sleep(2000000); torch.sum(fl)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); e1.synchronize()
        xs.append(e0.elapsed_time(e1))
    xs.sort()
    med = xs[30]; p10 = xs[6]; p90 = xs[53]
    res[name] = dict(kernel_ms=med, p10_ms=p10, p90_ms=p90, noise_floor=(p90 - p10) / med, n=60)
    print(name, sl, 'median_us %.2f p10 %.2f p90 %.2f' % (med * 1e3, p10 * 1e3, p90 * 1e3), flush=True)
batch = dict((k, res[k]) for k in ('1', '2', '4'))
print(json.dumps(dict(batch=batch, shapes=res, output=dict([('class', 'exact')]))), flush=True)
