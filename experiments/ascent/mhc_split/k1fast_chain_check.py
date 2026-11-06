"""chain.py --mode check with the fast K1 (default), plus a no-Triton-compile-inside-capture audit.

chain.check graphs REF (split off: mhc_fused_hc's layer_input), SPLIT, SLEEP (aux sleep before
every K2) and NOPDL (TRTLLM_ENABLE_PDL=0) at M=1/2/4/8, replays 8 steps with new inputs and
NaN-poisoned outputs, and compares every boundary output bitwise. This wrapper replaces
chain.capture with the same warmup + capture and counts K1 compiled variants (fast and main
kernels) across the torch.cuda.graph block: any growth is a compile inside capture.
Usage: k1fast_chain_check.py <ckpt> [extra chain.py args]
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain  # noqa: E402

from tensorrt_llm._torch.modules.mhc import mhc_lagged_split as _split  # noqa: E402
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream  # noqa: E402


def n_compiled():
    n = 0
    for k in (_split._mhc_lagged_layer_input_kernel, _split._mhc_lagged_layer_input_fast_kernel):
        for dc in (getattr(k, "device_caches", None) or {}).values():
            kc = dc[0] if isinstance(dc, tuple) else dc
            try:
                n += len(kc)
            except TypeError:
                pass
    return n


AUDIT = {"captures": 0, "compiles_in_capture": 0}


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), with_multi_stream(True):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    n0 = n_compiled()
    g = torch.cuda.CUDAGraph()
    with with_multi_stream(True):
        with torch.cuda.graph(g):
            out = fn()
    torch.cuda.synchronize()
    AUDIT["captures"] += 1
    AUDIT["compiles_in_capture"] += n_compiled() - n0
    return g, out


LAST = {}


def _print(*a, **k):
    if len(a) == 1 and isinstance(a[0], str) and a[0].startswith("{"):
        try:
            LAST["check"] = json.loads(a[0])
        except ValueError:
            pass
    print(*a, **k)


def main():
    chain.capture = capture
    chain.print = _print
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.argv = [sys.argv[0], "--mode", "check", "--ckpt", ckpt] + sys.argv[2:]
    chain.main()
    chk = LAST.get("check", {})
    passed = bool(chk.get("passed")) and AUDIT["compiles_in_capture"] == 0 and _split.K1_FAST
    print(json.dumps({"passed": passed, "k1_fast": _split.K1_FAST, "chain_check_passed": chk.get("passed"),
                      "audit": AUDIT, "compared": chk.get("compared"), "summary": chk.get("summary")}))


if __name__ == "__main__":
    main()
