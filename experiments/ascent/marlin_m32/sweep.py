# Sweep: dense Marlin NVFP4 launch shape (thread cfg x grid x stages) for decode rows 17-32
# (thread_m_blocks=2, FP32 split-K reduce = workspace_mode 2) on (5120,6144), (16384,5120),
# (14336,5120). Cold L2 (>= 400 MB of rotated copies in one graph), CUDA graph + PDL; all configs
# of one (shape, M) are captured and replayed alternately in one process. "main"
# (TRTLLM_MARLIN_M32_TABLE=0) is branch 18's generic launch and the in-process reference, "mainA"
# an A/A copy, "table" this tree's kM32Table default.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from common import (SHAPES, capture, eq, err, gemm, log, make_copy, nbytes,  # noqa: E402
                    set_cfg, time_graphs)

WARM, ITERS = 10, int(os.environ.get("SWEEP_ITERS", "60"))
MS = [int(v) for v in os.environ.get("SWEEP_MS", "32,24,17").split(",")]

CFGS = {
    (5120, 6144): [
        "128,64,128,80,4", "128,64,128,80,6",
        "128,64,128,160,4", "128,64,128,160,6",
        "128,64,128,120,6", "128,64,128,188,6",
        "64,128,128,40,8", "64,128,128,80,6", "64,128,128,80,8", "64,128,128,120,8",
        "64,128,128,160,8", "64,128,128,188,8",
        "64,256,256,80,6", "64,256,256,160,6",
    ],
    (16384, 5120): [
        "64,128,128,128,4", "64,128,128,128,6", "64,128,128,128,8",
        "64,128,128,188,8",
        "128,64,128,188,6", "128,64,128,128,6",
        "64,256,256,64,6", "64,256,256,128,6", "64,256,256,188,6",
    ],
    (14336, 5120): [
        "64,128,128,112,4", "64,128,128,112,6", "64,128,128,112,8",
        "64,128,128,188,8",
        "128,64,128,188,6", "128,64,128,112,6",
        "64,256,256,56,6", "64,256,256,112,6", "64,256,256,168,6",
    ],
}

only = os.environ.get("SWEEP_SHAPES")
p = torch.cuda.get_device_properties(0)
log("device", p.name, "sms", p.multi_processor_count)
result = dict()
for (n, k), C in SHAPES.items():
    if only and "%d,%d" % (n, k) not in only.split(";"):
        continue
    copies, refs = [], None
    for i in range(C):
        c, r = make_copy(n, k, 1000 + i, want_ref=(i == 0))
        copies.append(c)
        if r is not None:
            refs = r
    torch.cuda.synchronize()
    log("shape", (n, k), "copies", C, "MB/copy %.1f" % (nbytes(n, k) / 1e6),
        "mem GB %.2f" % (torch.cuda.memory_allocated() / 1e9))
    cfgs = ["main", "mainA", "table"] + CFGS[(n, k)]
    shape_res = dict()
    for m in MS:
        g = torch.Generator(device="cuda").manual_seed(7 + m)
        x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        ref = x.float() @ refs["w_ref"].t()
        graphs, outs, eager = dict(), dict(), dict()
        for cfg in cfgs:
            set_cfg("main" if cfg == "mainA" else cfg)
            eager[cfg] = gemm(x, copies[0], n, k).clone()
            graphs[cfg], outs[cfg] = capture(x, copies, n, k)
        set_cfg("main")
        t = time_graphs(graphs, WARM, ITERS, C)
        mres = dict()
        for cfg in cfgs:
            o0 = outs[cfg][0]
            e = err(o0, ref)
            mres[cfg] = dict(t[cfg], tbs=nbytes(n, k) / (t[cfg]["us"] * 1e-6) / 1e12,
                             bitwise_vs_main=eq(o0, outs["main"][0]), replay_eq_eager=eq(o0, eager[cfg]),
                             max_abs=e["max_abs"], mean_abs=e["mean_abs"], ulp=e["bf16_ulp"], finite=e["finite"])
        base = mres["main"]["us"]
        for cfg in sorted(cfgs, key=lambda c: mres[c]["us"]):
            r = mres[cfg]
            log("  (%d,%d) M=%d %-20s med %6.2f us p10 %6.2f p90 %6.2f %.3f TB/s d=%+6.2f us bitwise=%s "
                "rep==eager=%s max_abs %.4g mean_abs %.4g (ulp %.3g)" % (
                    n, k, m, cfg, r["us"], r["p10"], r["p90"], r["tbs"], r["us"] - base, r["bitwise_vs_main"],
                    r["replay_eq_eager"], r["max_abs"], r["mean_abs"], r["ulp"]))
        shape_res[str(m)] = mres
        del graphs, outs
        torch.cuda.synchronize()
    result["%d,%d" % (n, k)] = shape_res
    del copies, refs
    torch.cuda.empty_cache()
print(json.dumps(dict(sweep=result, warmup=WARM, iters=ITERS)))
