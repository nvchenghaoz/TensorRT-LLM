import json, types, torch
import tensorrt_llm._torch.attention.qk_norm_attention as qkmod
import tensorrt_llm._torch.modules.embedding as embmod
from tensorrt_llm._torch.modules.linear import TensorParallelMode as TPM
dev = 'cuda'
torch.manual_seed(0)
HQ, HKV, HD, RD, SEC, EPS = 24, 4, 256, 64, [11, 11, 10], 1e-6
W = 2 * HQ * HD + 2 * HKV * HD
V, D, MAXTOK, L = 248320, 5120, 16384, 16
cs = torch.randn(262144, 2, RD // 2, device=dev)
flush = torch.empty(96 * 2**20, device=dev)  # 384 MB L2 rotation
pre = qkmod.QKNormRoPEAttention.preprocess_qkv
CAND = hasattr(qkmod, '_ROPE_GATE_STRIDED_POS')
print('candidate tree:', CAND, flush=True)
def mk():
    return types.SimpleNamespace(
        _can_use_fused_qk_norm_rope_gate=lambda q, p: True,
        q_norm=types.SimpleNamespace(weight=torch.randn(HD, device=dev, dtype=torch.bfloat16), variance_epsilon=EPS),
        k_norm=types.SimpleNamespace(weight=torch.randn(HD, device=dev, dtype=torch.bfloat16)),
        rotary_emb=types.SimpleNamespace(rotary_cos_sin=cs), num_heads=HQ, num_key_value_heads=HKV,
        head_dim=HD, _get_qk_norm_rotary_dim=lambda: RD, pos_embd_params=types.SimpleNamespace(mrope_section=SEC))
selfs = [mk() for _ in range(L)]
pos = torch.zeros(3, 1, MAXTOK, dtype=torch.int32, device=dev)
for s in range(3):
    pos[s, 0] = torch.arange(34817 + 5 * s, 34817 + 5 * s + MAXTOK, device=dev, dtype=torch.int32)
weight = torch.randn(V, D, device=dev, dtype=torch.bfloat16)
def emb(ids):
    return embmod.pre_comm_embedding_ops(ids, weight, 1, 0, TPM.COLUMN, 0, V, True, 0)
def set_mode(fused):
    if CAND:
        qkmod._ROPE_GATE_STRIDED_POS = fused
        embmod._EMBED_FUSED_GATHER = fused
def make_fns(n):
    ids = torch.randint(0, V, (n,), device=dev, dtype=torch.int32)
    qkvs = [torch.randn(n, W, device=dev, dtype=torch.bfloat16) for _ in range(L)]
    p = pos[:, :, :n]
    def rope():
        return [pre(selfs[l], qkvs[l], p) for l in range(L)]
    def step():
        return [emb(ids)] + rope()
    def emb8():
        return [emb(ids) for _ in range(8)]
    return step, rope, emb8
def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g
def run_timed(runners, warm, iters):
    times = [[] for _ in runners]
    for it in range(warm + iters):
        for i, r in enumerate(runners):
            flush.zero_()
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); r(); b.record(); b.synchronize()
            if it >= warm:
                times[i].append(a.elapsed_time(b))
    out = []
    for t in times:
        q = torch.tensor(t).quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()
        out.append({'p10': q[0], 'median': q[1], 'p90': q[2], 'noise': (q[2] - q[0]) / q[1]})
    return out
modes = [True, False] if CAND else [True]
result = {}
diag = {}
for n in (1, 2, 4):
    step, rope, emb8 = make_fns(n)
    for name, fn, div in (('step', step, 1), ('rope16', rope, L), ('emb8', emb8, 8)):
        graphs = []
        for m in modes:
            set_mode(m)
            graphs.append(capture(fn))
        set_mode(True)
        res = run_timed([g.replay for g in graphs], 10, 100)
        for m, r in zip(modes, res):
            key = f'{name}_N{n}_' + ('default' if m else 'killswitch')
            diag[key] = {k: (v / div if k != 'noise' else v) for k, v in r.items()}
            print(key, 'per-unit ms', json.dumps(diag[key]), flush=True)
        if name == 'step':
            result[str(n)] = {'kernel_ms': res[0]['median'], 'noise_floor': res[0]['noise']}
for t in (2048, 8192):
    step, rope, emb8 = make_fns(t)
    runners = []
    for m in modes:
        runners.append((lambda m=m: (set_mode(m), step())))
    res = run_timed(runners, 10, 50)
    set_mode(True)
    for m, r in zip(modes, res):
        key = f'eager_step_T{t}_' + ('default' if m else 'killswitch')
        diag[key] = r
        print(key, json.dumps(r), flush=True)
    result[str(t)] = {'kernel_ms': res[0]['median'], 'noise_floor': res[0]['noise']}
    # separate eager embedding / rope-layer timing
    ids = torch.randint(0, V, (t,), device=dev, dtype=torch.int32)
    qkv = torch.randn(t, W, device=dev, dtype=torch.bfloat16)
    p = pos[:, :, :t]
    for name, fn in (('emb', lambda: emb(ids)), ('rope1', lambda: pre(selfs[0], qkv, p))):
        rs = run_timed([(lambda m=m, fn=fn: (set_mode(m), fn())) for m in modes], 10, 50)
        set_mode(True)
        for m, r in zip(modes, rs):
            key = f'eager_{name}_T{t}_' + ('default' if m else 'killswitch')
            diag[key] = r
            print(key, json.dumps(r), flush=True)
print('DIAG', json.dumps(diag), flush=True)
print(json.dumps({'batch': result, 'output': {'class': 'exact'}}), flush=True)
