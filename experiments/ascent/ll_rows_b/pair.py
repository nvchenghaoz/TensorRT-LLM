import os, sys, time, json, threading, subprocess
os.environ['TLLM_WORKER_USE_SINGLE_PROCESS'] = '1'
import torch
# Idea 67 (layer-63 Stage B row slice) in-process pair + correctness. Brace-free so the base arm can run it inline
# with python -c. argv: checkpoint, shared out dir (same slot for both arms), timed iterations, warmup iterations.
# Base arm (main tree): check scenarios + timing, saves base.pt. Candidate arm: the same scenarios with the slice ON
# and with the kill switch (OFF), in one process; compares ON vs OFF per step (read rows of the FP4 bytes and
# de-swizzled SFs into o_proj / gate_up / down_proj, final hidden state, logits hash, ids, logprobs) and both against
# base.pt. Timing: per eager mixed step, CUDA events from layer 63's pre-hook to the lm_head output (the whole
# affected op: layer 63 + final norm + lm_head) and the whole model forward.
CK = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ll67'
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 6
NW = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(OUT, exist_ok=True)
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, CudaGraphConfig
import tensorrt_llm._torch.models.modeling_qwen3_next as mq
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
CAND = hasattr(mq, '_LAST_LAYER_ROW_SLICE')
ARM = 'cand' if CAND else 'base'
def _git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return 'unknown:' + repr(e)
LOAD0 = os.getloadavg()
print('TREE_COMMIT', _git_head(), 'cwd', os.getcwd(), 'arm', ARM, flush=True)
AT = os.path.join(OUT, 'autotuner_cache.json')
if not CAND and os.path.exists(AT):
    os.remove(AT)
os.environ['TLLM_AUTOTUNER_CACHE_PATH'] = AT
T0 = time.time()
def log(*a):
    print('[' + ARM + ' %.0fs]' % (time.time() - T0), *a, flush=True)
def hb():
    while True:
        time.sleep(30)
        log('heartbeat')
threading.Thread(target=hb, daemon=True).start()
def set_mode(on):
    if CAND:
        mq._LAST_LAYER_ROW_SLICE = bool(on)
def slice_steps():
    if CAND:
        return int(mq._LAST_LAYER_ROW_SLICE_STATS['steps'])
    return -1
llm = LLM(model=CK, max_seq_len=36864, max_batch_size=128, max_num_tokens=8192,
          enable_chunked_prefill=True,
          cuda_graph_config=CudaGraphConfig(max_batch_size=128),
          kv_cache_config=KvCacheConfig(dtype='fp8', enable_block_reuse=True,
              mamba_state_config=dict(additional_snapshot_offsets_from_start=[32768])))
log('LLM ready')
model = llm._executor.engine.model_engine.model
last = None
lm_head = None
for name, m in model.named_modules():
    if type(m).__name__ == 'Qwen3NextFullAttentionDecoderLayer':
        if last is None or m.layer_idx > last.layer_idx:
            last = m
    if name.endswith('lm_head') and lm_head is None:
        lm_head = m
log('last layer', last.layer_idx, 'lm_head', type(lm_head).__name__)
gm = last.mlp.mlp
o_proj = last.self_attn.o_proj
down = gm.down_proj
# CAP: check capture. TM: timing capture.
CAP = dict(on=False, steps=[], cur=None)
TM = dict(on=False, ev=[], cur=None)
def pad128(n):
    return (n + 127) // 128 * 128
def rows_of(x, sel):
    # FP4 bytes and de-swizzled SFs (or BF16 bytes) of rows sel; sel None = all rows.
    if isinstance(x, Fp4QuantizedTensor):
        fp4 = x.fp4_tensor
        m = fp4.shape[0]
        sfc = fp4.shape[1] * 2 // 16
        sf = x.scaling_factor.view(torch.uint8).reshape(-1)[:pad128(m) * sfc]
        sf2 = torch.ops.trtllm.block_scale_interleave_reverse(sf.view(1, pad128(m), sfc)).view(pad128(m), sfc)[:m]
        if sel is not None:
            fp4 = fp4.index_select(0, sel)
            sf2 = sf2.index_select(0, sel)
        return dict(kind='fp4', m=m, fp4=fp4.view(torch.uint8).cpu(), sf=sf2.contiguous().cpu())
    t = x if sel is None else x.index_select(0, sel)
    return dict(kind=str(t.dtype), m=x.shape[0], raw=t.contiguous().view(torch.uint8).cpu())
def thash(t):
    xi = t.contiguous().view(torch.int32).reshape(-1).to(torch.int64)
    w = torch.arange(xi.numel(), device=xi.device, dtype=torch.int64) % 1000003 + 1
    return [int(xi.sum()), int((xi * w).sum())]
def layer_pre(mod, args, kwargs):
    if torch.cuda.is_current_stream_capturing():
        return
    md = kwargs.get('attn_metadata')
    if md is None or md.num_contexts <= 0 or md.num_tokens < 2048:
        return
    if TM['on']:
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        TM['cur'] = e0
    if not CAP['on']:
        return
    seq = [int(x) for x in md.seq_lens.tolist()]
    nt = int(md.num_tokens)
    idx = []
    s = 0
    for v in seq:
        s += v
        idx.append(s - 1)
    sel = torch.tensor(idx, device='cuda', dtype=torch.long)
    hs = kwargs.get('hidden_states')
    res = kwargs.get('residual')
    st = dict(seq=seq, nc=int(md.num_contexts), nt=nt, sel=sel, n=len(idx))
    # Layer 63's inputs (outputs of layers 0-62): every row.
    st['in_hash'] = thash(hs) + thash(res) if (hs is not None and res is not None) else None
    st['slices0'] = slice_steps()
    CAP['cur'] = st
def sel_for(x):
    st = CAP['cur']
    m = x.fp4_tensor.shape[0] if isinstance(x, Fp4QuantizedTensor) else x.shape[0]
    if m == st['nt']:
        return st['sel']
    return None
def mk_pre(key):
    def pre(mod, args):
        if not CAP['on'] or CAP['cur'] is None or torch.cuda.is_current_stream_capturing():
            return
        x = args[0]
        CAP['cur'][key] = rows_of(x, sel_for(x))
    return pre
def lm_post(mod, args, out):
    if torch.cuda.is_current_stream_capturing():
        return
    if TM['on'] and TM['cur'] is not None:
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        TM['ev'].append([TM['cur'], e1])
        TM['cur'] = None
    if not CAP['on'] or CAP['cur'] is None:
        return
    st = CAP['cur']
    CAP['cur'] = None
    h = args[0]
    st['hid'] = h.detach().contiguous().view(torch.uint8).cpu()
    st['hid_rows'] = h.shape[0]
    st['logit_hash'] = thash(out.float())
    k = min(out.shape[0], st['nc'] + 4)
    st['logits_k'] = out[:k].detach().float().cpu()
    st['sliced'] = slice_steps() - st['slices0']
    del st['sel']
    CAP['steps'].append(st)
FW = dict(on=False, ev=[], cur=None, nt=None)
def fw_pre(mod, args, kwargs):
    if not TM['on'] or torch.cuda.is_current_stream_capturing():
        return
    md = kwargs.get('attn_metadata')
    if md is None or md.num_contexts <= 0 or md.num_tokens < 2048:
        FW['cur'] = None
        return
    e0 = torch.cuda.Event(enable_timing=True)
    e0.record()
    FW['cur'] = [e0, int(md.num_seqs) - int(md.num_contexts), int(md.num_tokens)]
def fw_post(mod, args, out):
    if FW['cur'] is None or torch.cuda.is_current_stream_capturing():
        return
    e1 = torch.cuda.Event(enable_timing=True)
    e1.record()
    FW['ev'].append(FW['cur'] + [e1])
    FW['cur'] = None
last.register_forward_pre_hook(layer_pre, with_kwargs=True)
o_proj.register_forward_pre_hook(mk_pre('o_in'))
last.mlp.register_forward_pre_hook(mk_pre('gu_in'))
down.register_forward_pre_hook(mk_pre('dn_in'))
lm_head.register_forward_hook(lm_post)
# The engine calls the top model's .forward directly (no module hooks), so the forward window runs from the
# decoder model's pre-hook (embedding + 64 layers) to the lm_head output (final norm + logits).
inner = None
for name, m in model.named_modules():
    if type(m).__name__ == 'Qwen3NextModel':
        inner = m
log('decoder model', type(inner).__name__)
inner.register_forward_pre_hook(fw_pre, with_kwargs=True)
lm_head.register_forward_hook(fw_post)
g = torch.Generator().manual_seed(1234)
PREFIX = torch.randint(1000, 200000, (32768,), generator=g).tolist()
def P(i, n=2048):
    gg = torch.Generator().manual_seed(100000 + i)
    return PREFIX + torch.randint(1000, 200000, (n,), generator=gg).tolist()
def outres(o):
    c = o.outputs[0]
    lps = []
    for tid, d in zip(c.token_ids, c.logprobs or []):
        v = d.get(tid) if isinstance(d, dict) else None
        lps.append(float(v.logprob) if v is not None else None)
    return dict(ids=list(c.token_ids), lp=lps, cached=getattr(o, 'cached_tokens', None))
def gen(prompts, mt, lp=True):
    sp = SamplingParams(max_tokens=mt, temperature=0.0, top_k=1, ignore_eos=True,
                        logprobs=1 if lp else None)
    t = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t
    return [outres(o) for o in outs], dt
def scen(prompts, mt):
    CAP['steps'] = []
    CAP['cur'] = None
    CAP['on'] = True
    res, dt = gen(prompts, mt)
    torch.cuda.synchronize()
    CAP['on'] = False
    return dict(res=res, steps=CAP['steps'])
def mixed(seed_a, seed_b):
    sa = SamplingParams(max_tokens=64, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)
    sb = SamplingParams(max_tokens=8, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)
    CAP['steps'] = []
    CAP['cur'] = None
    CAP['on'] = True
    fa = llm.generate_async(P(seed_a), sa, streaming=True)
    fb = None
    for part in fa:
        if fb is None and len(part.outputs[0].token_ids) >= 12:
            fb = llm.generate_async(P(seed_b), sb)
    if fb is None:
        fb = llm.generate_async(P(seed_b), sb)
    ra = fa.result()
    rb = fb.result()
    torch.cuda.synchronize()
    CAP['on'] = False
    return dict(res=[outres(ra), outres(rb)], steps=CAP['steps'])
def eqd(a, b):
    if a is None or b is None:
        return a is None and b is None
    if a['kind'] != b['kind']:
        return False
    if a['kind'] == 'fp4':
        return a['fp4'].shape == b['fp4'].shape and torch.equal(a['fp4'], b['fp4']) and torch.equal(a['sf'], b['sf'])
    return a['raw'].shape == b['raw'].shape and torch.equal(a['raw'], b['raw'])
def cmp(a, b):
    o = dict()
    o['ids_eq'] = len(a['res']) == len(b['res']) and all(x['ids'] == y['ids'] for x, y in zip(a['res'], b['res']))
    lpd = 0.0
    for x, y in zip(a['res'], b['res']):
        for u, v in zip(x['lp'], y['lp']):
            if u is not None and v is not None:
                lpd = max(lpd, abs(u - v))
    o['lp_maxdiff'] = lpd
    sa = [[s['seq'], s['nc']] for s in a['steps']]
    sb = [[s['seq'], s['nc']] for s in b['steps']]
    o['n_steps'] = [len(sa), len(sb)]
    o['aligned'] = sa == sb
    keys = ('o_in', 'gu_in', 'dn_in')
    bad = dict(o_in=0, gu_in=0, dn_in=0, hid=0, logits=0, layer_in=0)
    lmax = 0.0
    nrows = 0
    for x, y in zip(a['steps'], b['steps']):
        for k in keys:
            if not eqd(x.get(k), y.get(k)):
                bad[k] += 1
        if not (x['hid'].shape == y['hid'].shape and torch.equal(x['hid'], y['hid'])):
            bad['hid'] += 1
        if x['logit_hash'] != y['logit_hash']:
            bad['logits'] += 1
        if x['in_hash'] != y['in_hash']:
            bad['layer_in'] += 1
        if x['logits_k'].shape == y['logits_k'].shape:
            lmax = max(lmax, float((x['logits_k'] - y['logits_k']).abs().max()))
        nrows += x['hid_rows']
    o['mismatch_steps'] = bad
    o['read_rows'] = nrows
    o['logits_k_maxabs'] = lmax
    o['bitwise'] = (o['aligned'] and o['ids_eq'] and lpd == 0.0 and all(v == 0 for v in bad.values()))
    return o
def summ(d):
    return [[s['nc'], s['n'] - s['nc'], s['nt'], s['sliced'], s.get('o_in', dict()).get('m'),
             s.get('gu_in', dict()).get('m'), s.get('dn_in', dict()).get('m')] for s in d['steps']]
N1, N2 = 68, 128
def run_checks(off):
    d = dict()
    d['c3'] = scen([P(off + 1), P(off + 2), P(off + 3)], 8)
    d['mixed'] = mixed(off + 4, off + 5)
    d['c1'] = scen([P(off + 100 + j) for j in range(N1)], 24)
    d['c2'] = scen([P(off + 300 + j) for j in range(N2)], 40)
    return d
log('prime')
r, dt = gen([P(0)], 1, False)
log('prime done %.2fs cached=%s' % (dt, r[0]['cached']))
modes = [True, False] if CAND else [None]
for mode in modes:
    set_mode(mode)
    gen([P(10 + j) for j in range(8)], 4, False)
log('warm done')
SC = dict()
for mode in modes:
    set_mode(mode)
    SC['m' + str(mode)] = run_checks(0)
    for k in SC['m' + str(mode)]:
        log('mode', mode, k, 'cached', SC['m' + str(mode)][k]['res'][0]['cached'], 'steps [nc, ng, nt, sliced, m_o, m_gu, m_dn]',
            json.dumps(summ(SC['m' + str(mode)][k])))
# Repeated-execution check: the timed loop's warmup + iteration count on C1 with the SAME prompts every iteration
# (prefix reused; the 2048 new tokens of each request recomputed). Every iteration vs iteration 1 and vs main.
REPC = dict()
for mode in modes:
    set_mode(mode)
    first = None
    eqs = []
    sl = []
    for it in range(NW + NT):
        s0 = slice_steps()
        x = scen([P(500 + j) for j in range(N1)], 24)
        sl.append(slice_steps() - s0)
        if first is None:
            first = x
        else:
            eqs.append(cmp(first, x)['bitwise'])
    REPC['m' + str(mode)] = dict(first=first, eq=eqs, slices=sl)
    log('repeat mode', mode, 'all equal iter1', all(eqs), 'n', len(eqs), 'slice steps per iter', sl)
# Timing: same prompts as the repeat loop; NW warmup + NT timed scenario iterations per mode, modes interleaved.
TIM = dict()
for mode in modes:
    TIM['m' + str(mode)] = dict(c1_win=[], c1_fw=[], c2_win=[], c2_fw=[], c1_ng=[], c2_ng=[])
LOADT = os.getloadavg()
log('LOADAVG_TIMING_START', LOADT)
for case, nreq, mt, base_seed in (('c1', N1, 24, 500), ('c2', N2, 40, 700)):
    for it in range(NW + NT):
        for mode in modes:
            set_mode(mode)
            TM['ev'] = []
            FW['ev'] = []
            TM['on'] = True
            gen([P(base_seed + j) for j in range(nreq)], mt, False)
            torch.cuda.synchronize()
            TM['on'] = False
            if it < NW:
                continue
            win = [a.elapsed_time(b) for a, b in TM['ev']]
            fws = [[ng, nt, a.elapsed_time(b)] for a, ng, nt, b in FW['ev']]
            T = TIM['m' + str(mode)]
            # Steady-state band: mixed steps with >= half the case's concurrency riding (c64: >= 32, c128: >= 64).
            band = 32 if case == 'c1' else 64
            for (ng, nt, ms), w in zip(fws, win):
                if ng >= band:
                    T[case + '_win'].append(w)
                    T[case + '_fw'].append(ms)
                    T[case + '_ng'].append(ng)
        log('timed', case, 'iter', it)
def stats(v):
    v = sorted(v)
    n = len(v)
    if n == 0:
        return dict(n=0)
    med = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    p10 = v[max(0, int(0.1 * (n - 1)))]
    p90 = v[min(n - 1, int(round(0.9 * (n - 1))))]
    return dict(median=med, p10=p10, p90=p90, noise=(p90 - p10) / med, n=n)
TS = dict()
for k in TIM:
    TS[k] = dict()
    for b in TIM[k]:
        if not b.endswith('_ng'):
            TS[k][b] = stats(TIM[k][b])
log('timing ms', json.dumps(TS))
torch.save(dict(SC=SC, REPC=REPC, TIM=TIM), os.path.join(OUT, ARM + '.pt'))
log('saved', os.path.join(OUT, ARM + '.pt'))
REP = dict()
PASS = None
CASES = ('c3', 'mixed', 'c1', 'c2')
if CAND:
    on_off = dict()
    for k in CASES:
        on_off[k] = cmp(SC['mFalse'][k], SC['mTrue'][k])
    REP['cand_on_vs_cand_off'] = on_off
    bp = os.path.join(OUT, 'base.pt')
    PASS = dict()
    for k in CASES:
        PASS['on_vs_off_bitwise_' + k] = on_off[k]['bitwise']
        PASS['sliced_every_step_' + k] = len(SC['mTrue'][k]['steps']) > 0 and all(s['sliced'] == 1 for s in SC['mTrue'][k]['steps'])
        PASS['killswitch_never_slices_' + k] = all(s['sliced'] == 0 for s in SC['mFalse'][k]['steps'])
    PASS['repeat_on_equal_iter1'] = all(REPC['mTrue']['eq']) and len(REPC['mTrue']['eq']) == NW + NT - 1
    PASS['repeat_on_slice_counter_pos'] = all(v > 0 for v in REPC['mTrue']['slices'])
    PASS['repeat_on_vs_off_iter1'] = cmp(REPC['mFalse']['first'], REPC['mTrue']['first'])['bitwise']
    if os.path.exists(bp):
        B = torch.load(bp, weights_only=False)
        vb_on = dict()
        vb_off = dict()
        for k in CASES:
            vb_on[k] = cmp(B['SC']['mNone'][k], SC['mTrue'][k])
            vb_off[k] = cmp(B['SC']['mNone'][k], SC['mFalse'][k])
        REP['base_vs_cand_on'] = vb_on
        REP['base_vs_cand_off'] = vb_off
        REP['base_repeat_equal'] = all(B['REPC']['mNone']['eq'])
        r_on = cmp(B['REPC']['mNone']['first'], REPC['mTrue']['first'])
        r_off = cmp(B['REPC']['mNone']['first'], REPC['mFalse']['first'])
        REP['repeat_first_base_vs_on'] = r_on
        REP['repeat_first_base_vs_off'] = r_off
        TB = dict()
        for b in B['TIM']['mNone']:
            if not b.endswith('_ng'):
                TB[b] = stats(B['TIM']['mNone'][b])
        REP['base_timing'] = TB
        for k in CASES:
            # Cross-process: counts where both trees scheduled the same steps.
            PASS['base_vs_on_bitwise_' + k] = vb_on[k]['bitwise'] or (not vb_on[k]['aligned'])
            PASS['base_vs_off_bitwise_' + k] = vb_off[k]['bitwise'] or (not vb_off[k]['aligned'])
        PASS['base_vs_on_aligned_c1'] = vb_on['c1']['aligned']
        PASS['repeat_iter1_base_vs_on'] = r_on['bitwise']
        PASS['repeat_iter1_base_vs_off'] = r_off['bitwise']
    else:
        REP['error'] = 'base.pt not found at ' + bp
        PASS['base_found'] = False
log('report', json.dumps(REP, default=str))
log('pass', json.dumps(PASS))
key = 'mTrue' if CAND else 'mNone'
batch = dict()
for b, case in (('64', 'c1'), ('128', 'c2')):
    s = TS[key][case + '_win']
    f = TS[key][case + '_fw']
    batch[b] = dict(kernel_ms=s['median'], noise_floor=s['noise'], p10=s['p10'], p90=s['p90'], n=s['n'],
                    step_forward_ms=f['median'], step_forward_noise=f['noise'])
res = dict(batch=batch, output=dict(zip(['class'], ['exact'])), arm=ARM)
if PASS is not None:
    res['passed'] = all(PASS.values())
    res['checks'] = PASS
if CAND:
    res['kill_switch_arm'] = TS['mFalse']
LOAD1 = os.getloadavg()
res['loadavg_1min'] = dict(start=LOAD0[0], timing_start=LOADT[0], end=LOAD1[0])
res['tree_commit'] = _git_head()
llm.shutdown()
print(json.dumps(res, default=str), flush=True)
os._exit(0)
