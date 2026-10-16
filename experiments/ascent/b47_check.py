"""Branch 47 correctness: bitwise vs main's code, repeated-execution in CUDA
graphs, kernel counts, STEP 0 position layout. Prints {"passed": ...} last."""
import json, os, sys, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b47_ref_rope_main import main_rope_gate, main_embedding
import tensorrt_llm._torch.attention.qk_norm_attention as qkmod
import tensorrt_llm._torch.modules.embedding as embmod
from tensorrt_llm._torch.modules.linear import TensorParallelMode as TPM

dev = 'cuda'
torch.manual_seed(0)
HQ, HKV, HD, RD, SEC, EPS = 24, 4, 256, 64, (11, 11, 10), 1e-6
W = 2 * HQ * HD + 2 * HKV * HD
V, D, MAXTOK, MAXPOS, L = 248320, 5120, 16384, 262144, 16
cs = torch.randn(MAXPOS, 2, RD // 2, device=dev)
pre = qkmod.QKNormRoPEAttention.preprocess_qkv
fails = []
summary = {}


def beq(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
    return torch.equal(a, b)


def mk():
    return types.SimpleNamespace(
        _can_use_fused_qk_norm_rope_gate=lambda q, p: True,
        q_norm=types.SimpleNamespace(weight=torch.randn(HD, device=dev, dtype=torch.bfloat16), variance_epsilon=EPS),
        k_norm=types.SimpleNamespace(weight=torch.randn(HD, device=dev, dtype=torch.bfloat16)),
        rotary_emb=types.SimpleNamespace(rotary_cos_sin=cs), num_heads=HQ, num_key_value_heads=HKV,
        head_dim=HD, _get_qk_norm_rotary_dim=lambda: RD, pos_embd_params=types.SimpleNamespace(mrope_section=list(SEC)))


def ref_rope(s, qkv, p, mrope=True):
    return main_rope_gate(qkv, s.q_norm.weight, s.k_norm.weight, cs, p, EPS, HQ, HKV, HD, RD,
                          SEC if mrope else None)


# ---- STEP 0: the runner's decode position buffer layout (runner.py L418, L3157)
pos = torch.empty((3, 1, MAXTOK), dtype=torch.int, device=dev)
for s in range(3):  # distinct sections so a wrong section stride is caught
    pos[s, 0] = torch.randint(0, MAXPOS, (MAXTOK,), device=dev, dtype=torch.int32)
pos1 = torch.randint(0, MAXPOS, (MAXTOK,), device=dev, dtype=torch.int32)
for n in (1, 4):
    pid = pos[:, :, :n]
    r = pid.reshape(3, -1)
    summary[f'step0_N{n}'] = dict(position_ids_shape=list(pid.shape), position_ids_stride=list(pid.stride()),
                                 reshaped_stride=list(r.stride()), reshaped_is_contiguous=r.is_contiguous())
print('STEP0', json.dumps({k: v for k, v in summary.items() if k.startswith('step0')}), flush=True)

# ---- Part 1 bitwise vs main
selfs = [mk() for _ in range(L)]
rope_cases = 0
for n in (1, 2, 4, 16, 2048, 8192):
    qkv = torch.randn(n, W, device=dev, dtype=torch.bfloat16)
    s = selfs[n % L]
    variants = {'strided': pos[:, :, :n], 'contig': pos[:, :, :n].contiguous(),
                'int64_strided': pos.to(torch.int64)[:, :, :n]}
    ref_q, ref_g = ref_rope(s, qkv, variants['contig'])
    for vname, p in variants.items():
        for path in ('preprocess', 'op'):
            if path == 'preprocess':
                q, _, _, g = pre(s, qkv, p)
            else:
                from tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate import \
                    fused_qkv_gemma_rmsnorm_rope_gate as op
                q, g = op(qkv, s.q_norm.weight, s.k_norm.weight, cs, p.reshape(3, -1), EPS, HQ, HKV, HD, RD, SEC)
            ok = beq(q, ref_q) and beq(g, ref_g)
            rope_cases += 1
            if not ok:
                fails.append(f'rope mrope N={n} {vname} {path}')
    # plain RoPE (2-D [1, N] position_ids) path unchanged
    if n in (1, 4, 2048):
        pp = pos1[:n].unsqueeze(0)
        rq, rg = ref_rope(s, qkv, pp, mrope=False)
        q, _, _, g = pre(s, qkv, pp)
        rope_cases += 1
        if not (beq(q, rq) and beq(g, rg)):
            fails.append(f'rope plain N={n}')
    torch.cuda.synchronize()
    print(f'rope N={n} done, fails so far {len(fails)}', flush=True)
summary['rope_bitwise_cases'] = rope_cases

# ---- Part 2 bitwise vs main
weight = torch.randn(V, D, device=dev, dtype=torch.bfloat16)


def emb(ids, mode=TPM.COLUMN):
    return embmod.pre_comm_embedding_ops(ids, weight, 1, 0, mode, 0, V, True, 0)


spec = [-1, V, V + 5, 0, V - 1]
id_sets = [[x] for x in spec] + [[12345], [-1, V - 1], [V, 7], [-1, V, V + 5, 0], [0, V - 1, 777, V + 5]]
for t in (2048, 8192):
    r = torch.randint(-3, V + 3, (t,)).tolist()
    r[:5] = spec
    r[-5:] = spec
    id_sets.append(r)
emb_cases = 0
for ids_l in id_sets:
    for dt in (torch.int32, torch.int64):
        ids = torch.tensor(ids_l, device=dev, dtype=dt)
        ref = main_embedding(ids, weight, 0, V)
        for mode in (TPM.COLUMN, None):
            out = emb(ids, mode)
            emb_cases += 1
            if not beq(out, ref):
                fails.append(f'emb n={len(ids_l)} {dt} {mode}')
        embmod._EMBED_FUSED_GATHER = False
        ks = emb(ids)
        embmod._EMBED_FUSED_GATHER = True
        if not beq(ks, ref):
            fails.append(f'emb killswitch n={len(ids_l)} {dt}')
# 2-D ids and empty input
ids2 = torch.tensor([[-1, 5, V], [V - 1, 0, V + 5]], device=dev)
if not beq(emb(ids2), main_embedding(ids2, weight, 0, V)):
    fails.append('emb 2-D')
e = torch.empty(0, dtype=torch.int32, device=dev)
if emb(e).shape != main_embedding(e, weight, 0, V).shape:
    fails.append('emb empty')
summary['emb_bitwise_cases'] = emb_cases + 2
print(f'embedding bitwise done, fails so far {len(fails)}', flush=True)

# ---- kernel counts (eager, one call each)
from torch.profiler import ProfilerActivity, profile


def count_kernels(fn):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    return sum(1 for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA)


qkv1 = torch.randn(1, W, device=dev, dtype=torch.bfloat16)
ids1 = torch.tensor([42], device=dev, dtype=torch.int32)
counts = {}
for m in (True, False):
    qkmod._ROPE_GATE_STRIDED_POS = m
    embmod._EMBED_FUSED_GATHER = m
    tag = 'candidate' if m else 'killswitch_main_path'
    counts[f'rope_layer_N1_{tag}'] = count_kernels(lambda: pre(selfs[0], qkv1, pos[:, :, :1]))
    counts[f'embedding_N1_{tag}'] = count_kernels(lambda: emb(ids1))
qkmod._ROPE_GATE_STRIDED_POS = True
embmod._EMBED_FUSED_GATHER = True
counts['per_step_saved'] = (16 * (counts['rope_layer_N1_killswitch_main_path'] - counts['rope_layer_N1_candidate'])
                            + counts['embedding_N1_killswitch_main_path'] - counts['embedding_N1_candidate'])
summary['kernel_counts'] = counts
print('KERNEL COUNTS', json.dumps(counts), flush=True)

# ---- repeated execution in CUDA graphs: 10 warmup + 50 iterations, NaN-filled outputs
def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = fn()
    torch.cuda.synchronize()
    return g, outs


rep = {}
for n in (1, 2, 4):
    qkvs = [torch.randn(n, W, device=dev, dtype=torch.bfloat16) for _ in range(L)]
    qkvs0 = [x.clone() for x in qkvs]
    pos0 = pos.clone()
    ids_l = ([V - 1], [-1, 7], [-1, V, 0, V - 1])[(1, 2, 4).index(n)]
    ids = torch.tensor(ids_l, device=dev, dtype=torch.int32)
    p = pos[:, :, :n]

    def step():
        outs = [emb(ids)]
        for l in range(L):
            q, _, _, g = pre(selfs[l], qkvs[l], p)
            outs += [q, g]
        return outs

    refs = [main_embedding(ids, weight, 0, V)]
    for l in range(L):
        refs += list(ref_rope(selfs[l], qkvs[l], p.contiguous()))
    g, outs = capture(step)
    first = None
    drift = 0
    mism = 0
    for it in range(60):  # 10 warmup + 50 iterations, same reset points as timing
        for o in outs:
            o.fill_(float('nan'))
        g.replay()
        torch.cuda.synchronize()
        cur = [o.clone() for o in outs]
        if first is None:
            first = cur
        drift += sum(0 if beq(a, b) else 1 for a, b in zip(cur, first))
        mism += sum(0 if beq(a, b) else 1 for a, b in zip(cur, refs))
        if not all(beq(a, b) for a, b in zip(qkvs, qkvs0)) or not beq(pos, pos0):
            fails.append(f'inputs mutated N={n} it={it}')
    rep[f'N{n}'] = {'iterations': 60, 'drift_vs_iter1': drift, 'mismatch_vs_main': mism}
    if drift or mism:
        fails.append(f'repeated N={n} drift={drift} mism={mism}')
summary['repeated_execution'] = rep
print('REPEATED', json.dumps(rep), flush=True)
summary['fails'] = fails
print(json.dumps({'passed': not fails, 'compared_against': 'main 2c795b56 rope-gate kernel (verbatim copy) with positions.contiguous(), and main TP1 masked-embedding torch path', **summary}), flush=True)
