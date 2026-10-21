import os, sys, time, json, threading
os.environ['TLLM_WORKER_USE_SINGLE_PROCESS'] = '1'
import torch
CK = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ll127'
NT1 = int(sys.argv[3]) if len(sys.argv) > 3 else 50
os.makedirs(OUT, exist_ok=True)
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, CudaGraphConfig
import tensorrt_llm._torch.models.modeling_qwen3_next as mq
CAND = hasattr(mq, '_LAST_LAYER_SKIP_FMHA_A')
print('TREE_COMMIT arm', 'cand' if CAND else 'base', flush=True)
ARM = 'cand' if CAND else 'base'
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
        mq._LAST_LAYER_SKIP_FMHA_A = bool(on)
def lto_count():
    if hasattr(torch.ops.trtllm, 'fmha_set_last_tile_split_skip_a'):
        return int(torch.ops.trtllm.fmha_set_last_tile_split_skip_a(False))
    return -1
llm = LLM(model=CK, max_seq_len=36864, max_batch_size=8, max_num_tokens=8192,
          enable_chunked_prefill=True,
          cuda_graph_config=CudaGraphConfig(max_batch_size=8),
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
LID = last.layer_idx
log('last full-attention layer', LID, 'lm_head', type(lm_head).__name__)
CAP = dict(on=False, kv=[], logits=[], hid=[], pend=False)
def attn_hook(mod, args, kwargs, out):
    if not CAP['on'] or torch.cuda.is_current_stream_capturing():
        return
    md = kwargs.get('attn_metadata')
    if md is None or md.num_contexts <= 0:
        return
    kvm = md.kv_cache_manager
    nc = md.num_contexts
    tpb = int(kvm.tokens_per_block)
    rids = [int(x) for x in list(md.request_ids)[:nc]]
    seq = [int(x) for x in md.seq_lens[:nc].tolist()]
    kvl = [int(x) for x in md.kv_lens[:nc].tolist()]
    cached = [kvl[i] - seq[i] for i in range(nc)]
    idx = kvm.get_batch_cache_indices(rids, LID)
    buf = kvm.get_buffers(LID)
    bu8 = buf.view(torch.uint8)
    pages = []
    for i in range(nc):
        b0 = cached[i] // tpb
        b1 = (cached[i] + seq[i] - 1) // tpb
        pids = torch.tensor(idx[i][b0:b1 + 1], device=buf.device, dtype=torch.long)
        pages.append(bu8.index_select(0, pids).cpu())
    gkv = [int(x) for x in md.kv_lens[nc:nc + int(md.num_generations)].tolist()]
    CAP['kv'].append(dict(seq=seq, cached=cached, ng=int(md.num_generations), gkv=gkv, lto=lto_count(), pages=pages))
    CAP['pend'] = True
def lm_hook(mod, args, out):
    if not CAP['pend']:
        return
    CAP['pend'] = False
    CAP['hid'].append(args[0].detach().float().cpu())
    CAP['logits'].append(out.detach().float().cpu())
last.self_attn.register_forward_hook(attn_hook, with_kwargs=True)
lm_head.register_forward_hook(lm_hook)
g = torch.Generator().manual_seed(1234)
PREFIX = torch.randint(1000, 200000, (32768,), generator=g).tolist()
def P(i, n=2048):
    gg = torch.Generator().manual_seed(100000 + i)
    return PREFIX + torch.randint(1000, 200000, (n,), generator=gg).tolist()
def gen(prompts, mt, lp=True):
    sp = SamplingParams(max_tokens=mt, temperature=0.0, top_k=1, ignore_eos=True,
                        logprobs=1 if lp else None)
    t = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t
    res = []
    for o in outs:
        c = o.outputs[0]
        lps = []
        if lp and c.logprobs:
            for tid, d in zip(c.token_ids, c.logprobs):
                v = d.get(tid) if isinstance(d, dict) else None
                lps.append(float(v.logprob) if v is not None else None)
        res.append(dict(ids=list(c.token_ids), lp=lps, cached=getattr(o, 'cached_tokens', None)))
    return res, dt
def scen(prompts, mt):
    CAP['kv'] = []
    CAP['logits'] = []
    CAP['hid'] = []
    CAP['pend'] = False
    CAP['on'] = True
    res, dt = gen(prompts, mt)
    torch.cuda.synchronize()
    CAP['on'] = False
    return dict(res=res, kv=CAP['kv'], logits=CAP['logits'], hid=CAP['hid'])
def mixed(seed_a, seed_b):
    sa = SamplingParams(max_tokens=96, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)
    sb = SamplingParams(max_tokens=8, temperature=0.0, top_k=1, ignore_eos=True, logprobs=1)
    CAP['kv'] = []
    CAP['logits'] = []
    CAP['hid'] = []
    CAP['pend'] = False
    CAP['on'] = True
    # B joins while A decodes: submit B once A has streamed 16 tokens, so B's context step is a mixed [ctx B + gen A]
    # step at about the same A position in both arms (the A position is recorded and compared).
    fa = llm.generate_async(P(seed_a), sa, streaming=True)
    fb = None
    for part in fa:
        if fb is None and len(part.outputs[0].token_ids) >= 16:
            fb = llm.generate_async(P(seed_b), sb)
    if fb is None:
        fb = llm.generate_async(P(seed_b), sb)
    ra = fa.result()
    rb = fb.result()
    torch.cuda.synchronize()
    CAP['on'] = False
    res = []
    for o in (ra, rb):
        c = o.outputs[0]
        lps = []
        for tid, d in zip(c.token_ids, c.logprobs or []):
            v = d.get(tid) if isinstance(d, dict) else None
            lps.append(float(v.logprob) if v is not None else None)
        res.append(dict(ids=list(c.token_ids), lp=lps, cached=getattr(o, 'cached_tokens', None)))
    return dict(res=res, kv=CAP['kv'], logits=CAP['logits'], hid=CAP['hid'])
log('prime')
r, dt = gen([P(0)], 1, False)
log('prime done %.2fs cached=%s' % (dt, r[0]['cached']))
modes = [False, True] if CAND else [None]
for mode in modes:
    set_mode(mode)
    gen([P(1)], 2, False)
    gen([P(2), P(3)], 2, False)
    gen([P(4), P(5), P(6), P(7)], 2, False)
log('warm done')
SC = dict()
def run_set(off):
    d = dict()
    d['c1a'] = scen([P(off + 10)], 8)
    d['c1b'] = scen([P(off + 11)], 8)
    d['b2'] = scen([P(off + 12), P(off + 13)], 8)
    d['b4'] = scen([P(off + 14), P(off + 15), P(off + 16), P(off + 17)], 8)
    d['rep'] = [scen([P(off + 10)], 8) for k in range(4)]
    d['long'] = scen([P(off + 20)], 64)
    d['mixed'] = mixed(off + 30, off + 31)
    # Guard fallback: 2048 and 1900 new tokens are in different q-tiles (32 and 30 of 64 rows), so the host guard
    # keeps launch A (the whole kernel's rows).
    d['ragged'] = scen([P(off + 32), P(off + 33, 1900)], 8)
    return d
def show(tag, d):
    log(tag, 'cached c1a/rep', d['c1a']['res'][0]['cached'], [x['res'][0]['cached'] for x in d['rep']],
        'lto per ctx step c1a', [x['lto'] for x in d['c1a']['kv']],
        'ng per ctx step mixed', [x['ng'] for x in d['mixed']['kv']])
# The same prompt sequence in both arms: set S (seeds 10..) then set S2 (seeds 50..). Base runs both on main's
# code; the candidate runs S with the change on and S2 with the kill switch (TRTLLM_LAST_LAYER_SKIP_FMHA_A=0 path).
set_mode(True)
SC['S'] = run_set(0)
show('S', SC['S'])
set_mode(False)
SC['S2'] = run_set(40)
show('S2', SC['S2'])
TIM = dict()
for mode in modes:
    TIM['m' + str(mode)] = dict(b1=[], b2=[], b4=[])
seed = 1000
for nb, iters, wu in ((1, NT1, 10), (2, 30, 5), (4, 20, 5)):
    for it in range(iters + wu):
        for mode in modes:
            set_mode(mode)
            ps = [P(seed + j) for j in range(nb)]
            seed += nb
            _, dt = gen(ps, 1, False)
            if it >= wu:
                TIM['m' + str(mode)]['b' + str(nb)].append(dt * 1000.0)
    log('timed nb', nb)
def stats(v):
    v = sorted(v)
    n = len(v)
    med = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    p10 = v[max(0, int(0.1 * (n - 1)))]
    p90 = v[min(n - 1, int(round(0.9 * (n - 1))))]
    return dict(median=med, p10=p10, p90=p90, noise=(p90 - p10) / med, n=n)
TS = dict()
for k in TIM:
    TS[k] = dict()
    for b in TIM[k]:
        TS[k][b] = stats(TIM[k][b])
log('timing ms', json.dumps(TS))
torch.save(dict(SC=SC, TIM=TIM), os.path.join(OUT, ARM + '.pt'))
ART = os.environ.get('AUTO_DEV_ARTIFACTS')
if ART:
    os.makedirs(ART, exist_ok=True)
    torch.save(dict(SC=SC, TIM=TIM), os.path.join(ART, ARM + '.pt'))
log('saved', os.path.join(OUT, ARM + '.pt'), 'host', os.uname()[1], 'gpu', os.environ.get('CUDA_VISIBLE_DEVICES'))
def cmp(a, b):
    o = dict()
    o['ids_eq'] = all(x['ids'] == y['ids'] for x, y in zip(a['res'], b['res']))
    lpd = 0.0
    for x, y in zip(a['res'], b['res']):
        for u, v in zip(x['lp'], y['lp']):
            if u is not None and v is not None:
                lpd = max(lpd, abs(u - v))
    o['lp_maxdiff'] = lpd
    o['ctx_steps'] = [len(a['kv']), len(b['kv'])]
    o['gen_positions'] = [[x['gkv'] for x in a['kv']], [x['gkv'] for x in b['kv']]]
    o['aligned'] = o['gen_positions'][0] == o['gen_positions'][1]
    kv_eq = len(a['kv']) == len(b['kv'])
    npg = 0
    for x, y in zip(a['kv'], b['kv']):
        for p, q in zip(x['pages'], y['pages']):
            npg += p.shape[0]
            kv_eq = kv_eq and p.shape == q.shape and torch.equal(p, q)
    o['kv_bitwise'] = kv_eq
    o['kv_pages'] = npg
    lg_eq = len(a['logits']) == len(b['logits'])
    lg_max = 0.0
    hd_eq = lg_eq
    hd_max = 0.0
    am_eq = lg_eq
    rows = 0
    for x, y in zip(a['logits'], b['logits']):
        if x.shape != y.shape:
            lg_eq = False
            am_eq = False
            continue
        rows += x.shape[0]
        lg_eq = lg_eq and torch.equal(x, y)
        lg_max = max(lg_max, float((x - y).abs().max()))
        am_eq = am_eq and torch.equal(x.argmax(-1), y.argmax(-1))
    for x, y in zip(a['hid'], b['hid']):
        if x.shape != y.shape:
            hd_eq = False
            continue
        hd_eq = hd_eq and torch.equal(x, y)
        hd_max = max(hd_max, float((x - y).abs().max()))
    o['logit_rows'] = rows
    o['logits_bitwise'] = lg_eq
    o['logits_maxabs'] = lg_max
    o['argmax_eq'] = am_eq
    o['final_hidden_bitwise'] = hd_eq
    o['final_hidden_maxabs'] = hd_max
    return o
def cmp_all(A, B):
    o = dict()
    for k in ('c1a', 'c1b', 'b2', 'b4', 'long', 'mixed', 'ragged'):
        o[k] = cmp(A[k], B[k])
    return o
REP = dict()
for tag in ('S', 'S2'):
    # Repeats of one prompt see one cache state (its own blocks reused): each equals repeat 1.
    REP['repeat_' + tag] = [cmp(SC[tag]['rep'][0], x) for x in SC[tag]['rep'][1:]]
REP['lto'] = [[x['lto'] for x in SC['S'][k]['kv']] for k in ('c1a', 'b2', 'b4', 'mixed')]
REP['cached'] = [[x['cached'] for x in SC['S'][k]['res']] for k in ('c1a', 'b2', 'b4', 'mixed')]
REP['mixed_ng'] = [[x['ng'] for x in SC[t]['mixed']['kv']] for t in ('S', 'S2')]
PASS = None
if CAND:
    bp = os.path.join(OUT, 'base.pt')
    if os.path.exists(bp):
        B = torch.load(bp, weights_only=False)
        REP['base_vs_cand_on'] = cmp_all(B['SC']['S'], SC['S'])
        REP['base_vs_cand_killswitch'] = cmp_all(B['SC']['S2'], SC['S2'])
        REP['base_timing'] = dict()
        for b in B['TIM']['mNone']:
            REP['base_timing'][b] = stats(B['TIM']['mNone'][b])
        on = REP['base_vs_cand_on']
        ks = REP['base_vs_cand_killswitch']
        PASS = dict()
        PASS['kv_bitwise_on'] = all(on[k]['kv_bitwise'] for k in on)
        PASS['killswitch_all_bitwise'] = all(ks[k]['kv_bitwise'] and ks[k]['logits_bitwise'] and ks[k]['ids_eq'] for k in ks)
        PASS['greedy_ids_on'] = all(on[k]['ids_eq'] for k in on)
        PASS['repeat_on'] = all(x['logits_bitwise'] and x['ids_eq'] and x['kv_bitwise'] for x in REP['repeat_S'])
        PASS['skip_a_used'] = REP['lto'][0][0] > 0 if REP['lto'][0] else False
        PASS['ragged_fallback_keeps_a'] = SC['S']['ragged']['kv'][0]['lto'] == SC['S']['mixed']['kv'][-1]['lto']
        PASS['mixed_has_gen_rows'] = any(x['ng'] > 0 for x in SC['S']['mixed']['kv'])
        PASS['mixed_aligned'] = on['mixed']['aligned']
        PASS['logits_bitwise_on'] = all(on[k]['logits_bitwise'] and on[k]['final_hidden_bitwise'] for k in on)
    else:
        REP['base_vs_cand_on'] = 'base.pt not found at ' + bp
log('report', json.dumps(REP, default=str))
log('pass', json.dumps(PASS))
key = 'mTrue' if CAND else 'mNone'
batch = dict()
for b in ('b1', 'b2', 'b4'):
    s = TS[key][b]
    batch[b[1:]] = dict(step_ms=s['median'], noise_floor=s['noise'], p10=s['p10'], p90=s['p90'])
res = dict(batch=batch, output=dict(**dict(zip(['class'], ['exact']))), arm=ARM)
if PASS is not None:
    res['passed'] = all(PASS.values())
    res['checks'] = PASS
if CAND:
    res['kill_switch_arm'] = TS['mFalse']
llm.shutdown()
print(json.dumps(res, default=str), flush=True)
os._exit(0)
