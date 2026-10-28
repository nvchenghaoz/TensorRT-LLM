"""Idea 79 Stage 2Q: the composed SM120 QKV op (_sm120_cutlass_qkv, per-plane alpha, planar).

Synthetic NVFP4 inputs at FLUX.3 klein QKV shapes (K=3072, 3 x 3072 columns, 24 heads of 128).
  time  : graph replay (10 calls per replay, interleaved over M), median of samples, per M.
          Runs whatever schedule the tree defaults to (base: Cooperative strided; branch: Pingpong).
  check : untimed. Pingpong route vs TRTLLM_FLUX3_SM120_QKV_SCHED=cooperative (main's case-1 op)
          and vs TRTLLM_FLUX3_SM120_QKV_INPLACE=0 (pre-57 path), bitwise, >=4 calls vs call 0,
          freed 0xFFFF garbage under every output, inputs unchanged; then CUDA graph capture with
          the side streams + 3 replays bitwise to eager; no workspace allocation during capture.
"""

import json
import os
import statistics
import sys

import torch

DEV = torch.device("cuda", 0)
MS = (4096, 4176, 4256)
K = 3072
W = 3072
REPS = 10
SAMPLES = int(os.environ.get("IDEA79_SAMPLES", "80"))


def log(*a):
    print(*a, flush=True)


def rup(x, m):
    return (x + m - 1) // m * m


def make_inputs():
    g = torch.Generator(device=DEV)
    g.manual_seed(79)
    w = torch.randint(0, 256, (3 * W, K // 2), dtype=torch.uint8, device=DEV, generator=g)
    w_sf = torch.randint(0x30, 0x40, (3 * W * (K // 16),), dtype=torch.uint8, device=DEV, generator=g)
    qkv_scale = torch.tensor([0.011, 0.013, 0.017], dtype=torch.float32, device=DEV)
    q_norm = (torch.rand(128, device=DEV, generator=g) + 0.5).to(torch.bfloat16)
    k_norm = (torch.rand(128, device=DEV, generator=g) + 0.5).to(torch.bfloat16)
    ang = torch.rand(8192, 64, device=DEV, generator=g) * 6.28
    cos_sin = torch.cat([ang.cos(), ang.sin()], dim=1).contiguous()
    per_m = {}
    for M in MS:
        a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=DEV, generator=g)
        a_sf = torch.randint(0x30, 0x40, (rup(M, 128) * (K // 16),), dtype=torch.uint8,
                             device=DEV, generator=g)
        pos = torch.randint(0, 8192, (M,), dtype=torch.int64, device=DEV, generator=g)
        per_m[M] = (a, a_sf, pos)
    return w, w_sf, qkv_scale, q_norm, k_norm, cos_sin, per_m


def op(fp, ins, M):
    w, w_sf, qkv_scale, q_norm, k_norm, cos_sin, per_m = ins
    a, a_sf, pos = per_m[M]
    return fp._sm120_cutlass_qkv(a, w, a_sf, w_sf, 1.0, 1.0, q_norm, k_norm, cos_sin, pos,
                                 qkv_scale, planar=True)


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    med = statistics.median(xs)
    return {"med": round(med, 2), "p10": round(xs[int(0.1 * (n - 1))], 2),
            "p90": round(xs[int(0.9 * (n - 1))], 2),
            "noise": round((xs[int(0.9 * (n - 1))] - xs[int(0.1 * (n - 1))]) / med, 4)}


def setup():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as sg

    sched = sg.qkv_schedule() if hasattr(sg, "qkv_schedule") else "cooperative(main)"
    log("schedule:", sched)
    sg.preload(DEV)
    return fp, sg, sched


def run_time():
    fp, sg, sched = setup()
    ins = make_inputs()
    graphs = {}
    for M in MS:
        for _ in range(3):
            op(fp, ins, M)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(REPS):
                op(fp, ins, M)
        torch.cuda.synchronize()
        graphs[M] = g
    for _ in range(5):
        for g in graphs.values():
            g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    times = {M: [] for M in MS}
    for s in range(SAMPLES):
        for M in (MS if s % 2 == 0 else MS[::-1]):
            e0.record()
            graphs[M].replay()
            e1.record()
            e1.synchronize()
            times[M].append(e0.elapsed_time(e1) * 1000.0 / REPS)
        if s % 20 == 0:
            log(f"sample {s}")
    res = {f"M{M}": stats(times[M]) for M in MS}
    log(json.dumps(res))
    m = res["M4176"]
    print(json.dumps({"batch": {"1": {"kernel_ms": m["med"] / 1000.0, "noise_floor": m["noise"]}},
                      "output": {"class": "exact"}, "schedule": sched, "op_us": res}))


def run_check():
    fp, sg, sched = setup()
    assert sched == "pingpong", sched
    ins = make_inputs()
    pristine = [t.clone() for t in ins[:6]] + [{M: tuple(x.clone() for x in v)
                                                 for M, v in ins[6].items()}]

    def inputs_unchanged():
        ok = all(torch.equal(x, y) for x, y in zip(ins[:6], pristine[:6]))
        for M in MS:
            ok &= all(torch.equal(x, y) for x, y in zip(ins[6][M], pristine[6][M]))
        return bool(ok)

    def garbage(M):
        t = torch.empty((M, 3 * W), dtype=torch.bfloat16, device=DEV)
        t.view(torch.int16).fill_(-1)
        del t

    def with_env(name, value, fn):
        old = os.environ.get(name)
        os.environ[name] = value
        sg.qkv_schedule.cache_clear()
        sg.inplace_enabled.cache_clear()
        try:
            return fn()
        finally:
            if old is None:
                del os.environ[name]
            else:
                os.environ[name] = old
            sg.qkv_schedule.cache_clear()
            sg.inplace_enabled.cache_clear()

    res = {"schedule": sched}
    passed = True
    for M in MS:
        ref_coop = with_env("TRTLLM_FLUX3_SM120_QKV_SCHED", "cooperative",
                            lambda: (sg.preload(DEV), op(fp, ins, M).clone())[1])
        ref_old = with_env("TRTLLM_FLUX3_SM120_QKV_INPLACE", "0", lambda: op(fp, ins, M).clone())
        torch.cuda.synchronize()
        first = None
        ok_coop = ok_old = ok_rep = True
        for call in range(5):
            garbage(M)
            o = op(fp, ins, M)
            torch.cuda.synchronize()
            ok_coop &= torch.equal(o.view(torch.int16), ref_coop.view(torch.int16))
            ok_old &= torch.equal(o.view(torch.int16), ref_old.view(torch.int16))
            if first is None:
                first = o.clone()
            else:
                ok_rep &= torch.equal(o.view(torch.int16), first.view(torch.int16))
        nan = int(torch.isnan(first.float()).sum())
        inp = inputs_unchanged()
        r = {"bitwise_vs_cooperative_main": bool(ok_coop), "bitwise_vs_pre57_path": bool(ok_old),
             "calls_vs_call0": bool(ok_rep), "inputs_unchanged": inp, "nan": nan,
             "absmax": float(first.float().abs().max())}
        passed &= bool(ok_coop and ok_old and ok_rep and inp and nan == 0)

        # CUDA graph capture with side streams; replay 3x bitwise to eager; no new workspaces.
        n_ws = len(sg._WORKSPACES)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            go = op(fp, ins, M)
        torch.cuda.synchronize()
        ws_ok = len(sg._WORKSPACES) == n_ws
        g_ok = True
        for _ in range(3):
            go.view(torch.int16).fill_(-1)
            g.replay()
            torch.cuda.synchronize()
            g_ok &= torch.equal(go.view(torch.int16), first.view(torch.int16))
        r.update({"graph_replay3_bitwise_eager": bool(g_ok), "no_ws_alloc_in_capture": bool(ws_ok),
                  "inputs_unchanged_after_graph": inputs_unchanged()})
        passed &= bool(g_ok and ws_ok and r["inputs_unchanged_after_graph"])
        del g, go
        res[f"M{M}"] = r
        log(json.dumps({f"M{M}": r}))
    res["passed"] = bool(passed)
    print(json.dumps(res))


if __name__ == "__main__":
    {"time": run_time, "check": run_check}[sys.argv[1]]()
