# Paired microbench for idea 13 (nvfp4_gemm mixed-step buckets), interleaved rounds.
# Each arm tunes once in a fresh process (this tree's tuning_config), then times the cached
# pick for each (shape, M) on synthetic NVFP4 operands of the real Qwen3.8-27B W4A4 shapes,
# cold L2 (rotated weight copies in one CUDA graph), CUDA events.
# ROUNDS interleaved rounds over all (M, shape); per shape the time is the median of all
# rounds' samples. kernel_ms = layer-weighted sum (qkvz x48, qkv x16, o_out x64, down x64).
# noise_floor per M = max of (p90-p10)/median of the per-round layer-weighted sums and the
# A/A gap between the odd and even rounds' sums.
import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments', 'ascent'))
import tensorrt_llm  # noqa
import nvfp4_coldtune as ct
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner as U
SH = {'qkvz': (16384, 5120, 48), 'qkv': (14336, 5120, 16), 'o_out': (5120, 6144, 64), 'down': (5120, 17408, 64)}
MS = [int(v) for v in os.environ.get('BENCH_MS', '8,16,32,2048,2055,2063,2079,4096,4103,4111,4127,6152,6175,8168,8191,8192').split(',')]
ROUNDS = int(os.environ.get('BENCH_ROUNDS', '8'))
WARM, ITERS = 10, int(os.environ.get('BENCH_ITERS', '25'))
ct.MS = tuple(sorted(set(MS) | {8192}))
shapes = [ct.Shape(k, n, kk) for k, (n, kk, _) in SH.items()]
tuner = AutoTuner.get()
tuner.clear_cache()
for s in shapes:
    with autotune():
        torch.ops.trtllm.nvfp4_gemm(*s.inputs(8192), torch.bfloat16, allowed_backends=ct.BACKENDS)
torch.cuda.synchronize()
r = ct.runner()
picks = {}
for m in MS:
    for s in shapes:
        hit, rid, tac, _ = tuner.profiling_cache.search_cache(ct.OP, [r], tuple(tuner._get_input_sizes(s.inputs(m))), U.tuning_config, apply_map_to_tuning_buckets=True)
        picks[(m, s.name)] = (bool(hit), tac)
samples = {k: [] for k in picks}          # all per-iteration times
round_med = {k: [] for k in picks}        # per-round medians
for rnd in range(ROUNDS):
    for m in MS:
        for s in shapes:
            hit, tac = picks[(m, s.name)]
            g, _ = ct.build_graph(s, m, ct.as_tac(tac), r)
            ts = ct.time_graph(g, s.nb, WARM, ITERS)
            del g, _
            samples[(m, s.name)] += ts
            round_med[(m, s.name)].append(ct.pct(ts, 0.5))
    torch.cuda.empty_cache()
    ct.log('round %d/%d done' % (rnd + 1, ROUNDS))
batch, detail = {}, {}
for m in MS:
    tot, p90, d = 0.0, 0.0, {}
    per_round = [0.0] * ROUNDS
    for s in shapes:
        w = SH[s.name][2]
        st = ct.stats(samples[(m, s.name)])
        tot += w * st['median_ms']
        p90 += w * st['p90_ms']
        for i, v in enumerate(round_med[(m, s.name)]):
            per_round[i] += w * v
        hit, tac = picks[(m, s.name)]
        d[s.name] = {'tactic': list(tac) if isinstance(tac, tuple) else tac, 'hit': hit, 'median_ms': st['median_ms'],
                     'p90_ms': st['p90_ms'], 'noise': st['noise'], 'round_medians_ms': [round(v, 5) for v in round_med[(m, s.name)]]}
    rs = ct.stats(per_round)
    even = sorted(per_round[0::2])[len(per_round[0::2]) // 2]
    odd = sorted(per_round[1::2])[len(per_round[1::2]) // 2]
    aa = abs(even - odd) / rs['median_ms']
    batch[str(m)] = {'kernel_ms': round(tot, 4), 'p90_ms': round(p90, 4), 'round_spread': round(rs['noise'], 4),
                     'aa_rel': round(aa, 4), 'noise_floor': round(max(rs['noise'], aa), 4)}
    detail[str(m)] = d
    ct.log('M=%d layer-weighted W4A4 GEMM sum %.3f ms (p90 %.3f) round-spread %.4f A/A %.4f %s' % (
        m, tot, p90, rs['noise'], aa, json.dumps({k: (v['tactic'], round(v['median_ms'], 4)) for k, v in d.items()})))
print(json.dumps({'batch': batch, 'output': {'class': 'reordered'}, 'detail': detail}))
