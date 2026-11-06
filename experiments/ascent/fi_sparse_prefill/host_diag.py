"""Diagnostic for idea 372: where the host time of one FI shared-prefill call goes.

Prints Triton's JITFunction.run source (launcher ABI for a cached-handle launch), then
perf_counter host us per call (500 back-to-back calls, one sync at the end) for: the full
_forward_selected, prepare_shared_prefill_indices, wrapper.run, PreparedCall.execute.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pair import Case  # noqa: E402
from stage0 import HEADS, SCALE, Scenario, log  # noqa: E402


def host_us(fn, n=500):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    h = (time.perf_counter() - t) / n * 1e6
    torch.cuda.synchronize()
    return h


def main():
    import triton
    from triton.runtime import jit

    log("triton", triton.__version__, jit.__file__)
    src = inspect.getsource(jit.JITFunction.run)
    log("=== JITFunction.run ===")
    for line in src.splitlines():
        log("|", line)
    try:
        from triton.compiler import compiler as cc

        log("=== CompiledKernel.__getitem__/run ===")
        for name in ("__getitem__", "_init_handles", "launch_metadata"):
            fn = getattr(cc.CompiledKernel, name, None)
            if fn is not None:
                for line in inspect.getsource(fn).splitlines():
                    log("|", line)
    except Exception as e:  # noqa: BLE001
        log("compiler src failed", e)
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import fi_sparse_prefill as fsp

    res = {}
    for i, (r, t, hm) in enumerate([(1, 128, True), (1, 128, False)]):
        s = Scenario(r, t, hm, seed=1000 + i)
        c = Case(s)
        c()
        torch.cuda.synchronize()
        fi = c.helper._fi_prefill
        dev = s.q.device
        bufs = fi._scratch(s.count, HEADS, 128, 512 if hm else 0, dev)
        wrapper = fi._wrapper(hm, dev)
        r_ = {}
        r_["forward_selected"] = host_us(c)
        r_["index_prep"] = host_us(
            lambda: fsp.prepare_shared_prefill_indices(
                s.swa_pool, s.swa_slots, s.main_pool if hm else None, s.main_logical if hm else None,
                s.mapping, s.plan, bufs["swa"], bufs["main"] if hm else None,
            )
        )
        cache = s.swa_pool.view(-1, 1, 528)
        extra = s.main_pool.view(-1, 1, 288) if hm else None
        out = s.out_fi

        def wr():
            wrapper.run(q=s.q, kv_cache=cache, indices=bufs["swa"], output=out, sm_scale=SCALE, attn_sink=s.sink,
                        extra_kv_cache=extra, extra_indices=bufs["main"] if hm else None, out_lse=bufs["lse"])

        r_["wrapper_run"] = host_us(wr)
        calls = list(wrapper._prepared_calls.values())
        log("prepared calls", len(calls), [c_.workspace for c_ in calls])
        pc = calls[-1]
        mid = pc.mid if pc.mid is not None else out
        mlse = pc.mlse if pc.mlse is not None else out.view(torch.float32)

        def ex():
            pc.execute(s.q, cache, bufs["swa"], out, SCALE, None, s.sink, extra, bufs["main"] if hm else None, None,
                       mid, mlse, bufs["lse"], 1.0)

        r_["execute"] = host_us(ex)
        from flashinfer.mla._sparse_mla_sm120 import _calibration as cal

        r_["refresh_store"] = host_us(cal.refresh_store)
        def ctx():
            with torch.cuda.device(dev):
                pass

        r_["device_ctx"] = host_us(ctx)
        res[s.name] = r_
        log(s.name, json.dumps(r_))
    print(json.dumps({"passed": True, "host_us": res}))


if __name__ == "__main__":
    sys.exit(main())
