"""Runs pair.py (BENCH_MS from env) then dumps the AutoTuner cache entries, to
see whether per-process timing modes follow the tuned tactic choice."""
import os
import sys

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pair.py")).read()
exec(compile(src, "pair.py", "exec"))
from tensorrt_llm._torch.autotuner import AutoTuner  # noqa: E402

for k, v in AutoTuner.get().profiling_cache.cache.items():
    ks = str(k)
    if "fp4" in ks.lower() or "nvfp4" in ks.lower():
        print("TUNED", ks[:220], "->", str(v)[:200], flush=True)
sys.stdout.flush()
