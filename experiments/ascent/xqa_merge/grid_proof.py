# Grid proof for the XQA one-wave cap (branch 29): one profiled eager TrtllmAttention decode per B in {1,2,3,4}
# through tests/microbenchmarks/attention_perf/attention_perf_harness.py (unmodified, imported read-only), then the
# kernel_mha launch grid from the Kineto trace. Expected on the candidate build: {47,4,1}/{23,4,2}/{15,4,3}/{11,4,4}.
import importlib.util
import json
import os
import sys
import tempfile

import torch

assert "TRTLLM_XQA_BLOCKS_PER_SEQUENCE" not in os.environ, "env override must be unset"
path = "tests/microbenchmarks/attention_perf/attention_perf_harness.py"
spec = importlib.util.spec_from_file_location("attention_perf_harness", path)
h = importlib.util.module_from_spec(spec)
sys.modules["attention_perf_harness"] = h
spec.loader.exec_module(h)

EXPECT = {1: [47, 4, 1], 2: [23, 4, 2], 3: [15, 4, 3], 4: [11, 4, 4]}
grids = {}
ok = True
for b in (1, 2, 3, 4):
    case = h.AttnCase(case_id=f"grid_b{b}", phase="generation", num_heads=24, num_kv_heads=4, head_dim=256,
                      page_size=32, seq_len=1, num_cached_tokens=34816, batch_size=b)
    print(f"B={b}: running eager decode under torch.profiler", flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        sig = h._collect_signals_inproc(case, warmup=1, iters=1)
    torch.cuda.synchronize()
    fn = os.path.join(tempfile.mkdtemp(), f"trace_b{b}.json")
    prof.export_chrome_trace(fn)
    with open(fn) as f:
        tr = json.load(f)
    evs = [e for e in tr.get("traceEvents", []) if e.get("cat") == "kernel" and "kernel_mha" in e.get("name", "")]
    seen = sorted({json.dumps(e.get("args", {}).get("grid")) for e in evs})
    blocks = sorted({json.dumps(e.get("args", {}).get("block")) for e in evs})
    grids[str(b)] = dict(launches=len(evs), grids=[json.loads(s) for s in seen], blocks=[json.loads(s) for s in blocks],
                         gpu_time_median_ms=getattr(sig, "gpu_time_median_ms", None))
    good = len(evs) > 0 and all(json.loads(s) == EXPECT[b] for s in seen)
    ok = ok and good
    print(f"B={b}: kernel_mha launches {len(evs)} grids {seen} blocks {blocks} expect {EXPECT[b]} -> "
          f"{'PASS' if good else 'FAIL'}", flush=True)
    if not evs:
        names = sorted({e.get('name', '')[:120] for e in tr.get('traceEvents', []) if e.get('cat') == 'kernel'})
        print("  kernels seen:", names[:40], flush=True)
print(json.dumps({"passed": ok, "check": "kernel_mha grid per B (eager TrtllmAttention decode, 34816 cached tokens, "
                  "TRTLLM_XQA_BLOCKS_PER_SEQUENCE unset)", "expected": EXPECT, "observed": grids}), flush=True)
