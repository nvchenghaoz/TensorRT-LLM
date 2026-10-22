# Correctness check for idea 3 (GDN in_proj_ba BF16 [M,5120]x[96,5120]^T on the Triton small-N GEMV at M=5-16).
# Production path: a real BF16 Linear (UnquantizedLinearMethod.apply -> maybe_small_n_gemv).
# Per M (1-16 routed, 17/24/32 must stay on cuBLAS), per input seed:
#  1. routing: maybe_small_n_gemv returns a tensor for M <= 16 and None above; the Linear output equals it bitwise.
#  2. repeated-call: eager output vs CUDA-graph replays (10 warmup + 100 = 110 replays), every replay bitwise equal to
#     the first replay and to eager.
#  3. aux-stream capture: the same call captured on a side stream inside a fork/join graph (maybe_execute_in_parallel
#     order) replays bitwise equal.
#  4. reordered tolerance vs an FP32 reference built from the same BF16 operands: no NaN/Inf, max-abs and mean-abs
#     error <= 1.25x cuBLAS's error + one BF16 ulp of the output magnitude.
# Last stdout line: JSON with passed.
import json
import os
import time

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules import small_n_gemv as SNG
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode

T0 = time.time()


def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)


dev = torch.device("cuda")
K, N = 5120, 96
MS = [int(s) for s in os.environ.get("C_MS", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,24,32").split(",")]
SEEDS = [int(s) for s in os.environ.get("C_SEEDS", "0,1,2").split(",")]
REPLAYS = 110
NEW_MAX = int(os.environ.get("C_NEW_MAX", "16"))

g = torch.Generator(device=dev).manual_seed(123)
wb = (torch.randn(N, K, generator=g, device=dev) * 0.02).to(torch.bfloat16)
lin = Linear(K, N, bias=False, dtype=torch.bfloat16, tensor_parallel_mode=TensorParallelMode.COLUMN,
             reduce_output=False).to(dev)
lin.load_weights([dict(weight=wb)])
lin.post_load_weights()
w = lin.weight
log("route table", SNG._MAX_M_BY_N, "enable_cuda_core", getattr(lin, "enable_cuda_core", None))
aux = torch.cuda.Stream()
side = torch.cuda.Stream()

results = dict()
ok_all = True
for m in MS:
    for seed in SEEDS:
        gx = torch.Generator(device=dev).manual_seed(1000 * m + seed)
        scale = (0.5, 1.0, 4.0)[seed % 3]
        x = (torch.randn(m, K, generator=gx, device=dev) * scale).to(torch.bfloat16)
        r = dict()
        routed = SNG.maybe_small_n_gemv(x, w)
        r["routed"] = routed is not None
        r["route_expected"] = m <= NEW_MAX
        eager = lin(x)
        r["linear_equals_gemv"] = bool(routed is not None and torch.equal(eager, routed)) if routed is not None else None
        cub = F.linear(x, w)  # cuBLAS (main's path at these M)
        ref = x.float() @ w.float().t()
        err = (eager.float() - ref).abs()
        cerr = (cub.float() - ref).abs()
        ulp = float(ref.abs().max()) * 2.0 ** -7
        r["finite"] = bool(torch.isfinite(eager).all())
        r["max_err"] = float(err.max())
        r["mean_err"] = float(err.mean())
        r["cublas_max_err"] = float(cerr.max())
        r["cublas_mean_err"] = float(cerr.mean())
        r["tol_ok"] = bool(r["finite"] and r["max_err"] <= 1.25 * r["cublas_max_err"] + ulp
                           and r["mean_err"] <= 1.25 * r["cublas_mean_err"] + ulp)
        r["bitwise_vs_cublas"] = bool(torch.equal(eager, cub))
        # graph replay, single stream
        xs = x.clone()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                lin(xs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            yg = lin(xs)
        gr.replay()
        torch.cuda.synchronize()
        first = yg.clone()
        bad = 0
        for it in range(REPLAYS - 1):
            gr.replay()
            torch.cuda.synchronize()
            if not torch.equal(yg, first):
                bad += 1
        r["graph_first_equals_eager"] = bool(torch.equal(first, eager))
        r["graph_replays_differing"] = bad
        # fork/join capture: ba on aux beside a main-stream kernel
        ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()
        dummy = torch.zeros(1 << 20, device=dev)
        gr2 = torch.cuda.CUDAGraph()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            lin(xs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(gr2):
            ev0.record()
            dummy.add_(1.0)
            with torch.cuda.stream(aux):
                ev0.wait()
                ya = lin(xs)
                ev1.record()
            ev1.wait()
            yb = ya + 0
        bad2 = 0
        for it in range(REPLAYS):
            gr2.replay()
            torch.cuda.synchronize()
            if not torch.equal(yb, eager):
                bad2 += 1
        r["aux_graph_replays_differing"] = bad2
        del gr, gr2
        ok = (r["routed"] == r["route_expected"] and r["tol_ok"] and r["graph_first_equals_eager"]
              and bad == 0 and bad2 == 0 and (r["linear_equals_gemv"] in (None, True)))
        r["ok"] = bool(ok)
        ok_all = ok_all and ok
        results["M%d_s%d" % (m, seed)] = r
        log("M=%2d seed %d routed %s(exp %s) maxerr %.3e (cublas %.3e) meanerr %.3e (cublas %.3e) tol %s "
            "eq_cublas %s graph first==eager %s replays diff %d aux diff %d -> %s" % (
                m, seed, r["routed"], r["route_expected"], r["max_err"], r["cublas_max_err"], r["mean_err"],
                r["cublas_mean_err"], r["tol_ok"], r["bitwise_vs_cublas"], r["graph_first_equals_eager"], bad,
                bad2, "OK" if ok else "FAIL"))
print(json.dumps(dict(passed=bool(ok_all), compared="Linear(in_proj_ba shape) output vs FP32 reference (tolerance "
                      "1.25x cuBLAS + 1 ulp), eager vs graph replay vs first replay bitwise over 110 replays, "
                      "fork/join aux-stream capture bitwise over 110 replays", results=results)))
