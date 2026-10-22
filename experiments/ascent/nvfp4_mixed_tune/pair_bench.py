import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments', 'ascent'))
import tensorrt_llm  # noqa
import nvfp4_coldtune as ct
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner as U
SH = {'qkvz': (16384, 5120, 48), 'qkv': (14336, 5120, 16), 'o_out': (5120, 6144, 64), 'down': (5120, 17408, 64)}
MS = [int(v) for v in os.environ.get('BENCH_MS', '8,16,32,2048,2055,2063,2079,4096,4103,4111,4127,6152,6175,8168,8191,8192').split(',')]
ct.MS = tuple(sorted(set(MS) | {8192}))
shapes = [ct.Shape(k, n, kk) for k, (n, kk, _) in SH.items()]
tuner = AutoTuner.get()
tuner.clear_cache()
for s in shapes:
    with autotune():
        torch.ops.trtllm.nvfp4_gemm(*s.inputs(8192), torch.bfloat16, allowed_backends=ct.BACKENDS)
torch.cuda.synchronize()
r = ct.runner()
batch, detail = {}, {}
for m in MS:
    tot, tot2, noise, p90, d = 0.0, 0.0, 0.0, 0.0, {}
    for s in shapes:
        hit, rid, tac, _ = tuner.profiling_cache.search_cache(ct.OP, [r], tuple(tuner._get_input_sizes(s.inputs(m))), U.tuning_config, apply_map_to_tuning_buckets=True)
        st = ct.time_tactic(s, m, tac, 10, 50)
        st2 = ct.time_tactic(s, m, tac, 10, 50)  # A/A: same pick, second timing pass
        w = SH[s.name][2]
        tot += w * st['median_ms']
        tot2 += w * st2['median_ms']
        p90 += w * st['p90_ms']
        noise = max(noise, st['noise'], st2['noise'])
        d[s.name] = {'tactic': list(tac) if isinstance(tac, tuple) else tac, 'hit': bool(hit), 'median_ms': st['median_ms'], 'p90_ms': st['p90_ms'], 'aa_median_ms': st2['median_ms'], 'noise': st['noise']}
    aa = abs(tot2 - tot) / tot
    batch[str(m)] = {'kernel_ms': round(tot, 4), 'p90_ms': round(p90, 4), 'aa_ms': round(tot2, 4), 'aa_rel': round(aa, 4), 'noise_floor': round(max(noise, aa), 4)}
    detail[str(m)] = d
    ct.log('M=%d layer-weighted W4A4 GEMM sum %.3f ms (p90 %.3f, A/A %.3f) noise %.4f %s' % (m, tot, p90, tot2, noise, json.dumps({k: (v['tactic'], round(v['median_ms'], 4)) for k, v in d.items()})))
print(json.dumps({'batch': batch, 'output': {'class': 'reordered'}, 'detail': detail}))
