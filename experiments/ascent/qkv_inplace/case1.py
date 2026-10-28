"""Branch 57 (idea 103): FLUX.3 SM120 NVFP4 QKV in place, case 1 (planar).

Op under test: torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope with per-plane alphas (the
checkpoint's to_q/to_k/to_v route), planar layout (main >= landing 18).
  main      : 3 TRT-LLM FP4GemmRunner t0 GEMMs into 3 fresh [M,3072] buffers, then qknorm_rope
              writes a 4th buffer (and copies v).
  candidate : 3 FlashInfer strided SM120 GEMMs write planes i*M*3072 of one [M,9216] buffer, then
              qknorm_rope runs in place on planes 0/1 (copy_v=False).

Modes:
  bench : runs in BOTH trees (run_gpu_pair). One sample = CUDA-event-timed replay of a CUDA graph
          of K=10 back-to-back op calls, reported per call. >=10 s warm soak, M=4096/4176/4256
          interleaved, 60 samples each, median; noise = (p90-p10)/median. Headline M=4176.
  table : candidate tree only. Same timing method; main composed (in place switched off in the
          same process) vs candidate composed, candidate GEMMs alone, in-place pass alone, main's
          GEMMs alone and main's pass alone; cold build seconds of the strided module.
  check : candidate tree only, untimed. Bitwise vs main's planar op (in place switched off) at
          M=4096/4176/4256: warmup 5 + 50 calls, every call compared with the reference and with
          call 0, the op's output block prefilled with 0xFFFF before every call (block reuse
          counted), inputs compared with pristine copies before every reset. CUDA graph capture
          of the op (side streams) + 3 replays bitwise vs eager; no workspace allocation and
          fewer allocations inside capture; pass compile keys (no compile per M).
"""

import json
import statistics
import sys
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_qknorm_rope as qr
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

try:
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided
except ImportError:  # base tree
    strided = None

H, K = 3072, 3072
MS = (4096, 4176, 4256)
WARMUP = 5
ITERS = 50
GRAPH_K = 10
SAMPLES = 60
SOAK_S = 10.0

op = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope


def log(*a):
    print(*a, flush=True)


def mk(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    ws = [torch.randn(H, K, device="cuda", dtype=torch.bfloat16, generator=g) * s
          for s in (0.02, 0.035, 0.012)]

    def q(t):
        s = float(t.abs().amax().float() / 2688.0)
        d, sf = fp.nvfp4_quantize(t, torch.tensor([1.0 / s], device="cuda"))
        return d, sf, s

    a, asf, ags = q(x)
    qw = [q(w) for w in ws]
    w = torch.cat([t[0] for t in qw]).contiguous()
    wsf = torch.cat([t[1] for t in qw]).contiguous()
    qs = torch.tensor([ags * t[2] for t in qw], device="cuda", dtype=torch.float32)
    qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
    cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
    pos = torch.arange(M, device="cuda", dtype=torch.int64)
    return (a, w, asf, wsf, ags, qw[0][2], qn, kn, cs, pos, qs)


def set_inplace(on):
    if strided is not None:
        strided.inplace_enabled = (lambda: True) if on else (lambda: False)


def route():
    return "inplace_case1" if (strided is not None and strided.inplace_enabled()) else "main_planar"


# --------------------------------------------------------------------------- timing helpers
def graph_of(fn):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_K):
            fn()
    torch.cuda.synchronize()
    return graph


def time_graphs(graphs):
    """graphs: dict name -> CUDAGraph. Warm soak, then interleaved samples."""
    t0 = time.time()
    while time.time() - t0 < SOAK_S:
        for gr in graphs.values():
            gr.replay()
        torch.cuda.synchronize()
    ev = {k: [] for k in graphs}
    for it in range(SAMPLES):
        for name, gr in graphs.items():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
            ev[name].append((s, e))
        if it % 20 == 0:
            log("sample", it)
    torch.cuda.synchronize()
    res = {}
    for name, lst in ev.items():
        t = sorted(s.elapsed_time(e) / GRAPH_K for s, e in lst)
        med = statistics.median(t)
        p10, p90 = t[len(t) // 10], t[(len(t) * 9) // 10]
        res[name] = {"ms": round(med, 5), "noise": round((p90 - p10) / med, 5)}
        log(f"{name}: {med * 1000:.1f} us noise {(p90 - p10) / med:.4f}")
    return res


# --------------------------------------------------------------------------- bench (pair)
def bench():
    log("route", route())
    graphs = {}
    keep = []
    for M in MS:
        args = mk(M, M)
        keep.append(args)
        graphs[f"op_M{M}"] = graph_of(lambda args=args: op(*args))
    res = time_graphs(graphs)
    head = res["op_M4176"]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": head["ms"], "noise_floor": head["noise"]}},
        "output": {"class": "exact"},
        "route": route(),
        "what": "trtllm::flux3_primsts_nvfp4_qkv_rope per-plane alpha planar, ms per call "
                "(CUDA graph of 10 calls per sample), M=4176 headline",
        "detail": res,
    }))


# --------------------------------------------------------------------------- table
def gemms_main(args):
    a, w, asf, wsf, ags, wgs, qn, kn, cs, pos, qs = args
    runner = fp._sm120_fp4_gemm_runner()
    seg = wsf.numel() // 3
    current = torch.cuda.current_stream()
    side = fp._sm120_qkv_side_streams(a.device.index)
    for st in side:
        st.wait_stream(current)

    def g(i):
        return runner.run_gemm(a, w[i * H:(i + 1) * H], asf, wsf[i * seg:(i + 1) * seg],
                               qs[i:i + 1], 0, 0, None, None)
    q = g(0)
    planes = []
    for i, st in enumerate(side):
        with torch.cuda.stream(st):
            planes.append(g(i + 1))
    for st in side:
        current.wait_stream(st)
    return q, planes


def gemms_cand(args, out):
    a, w, asf, wsf, ags, wgs, qn, kn, cs, pos, qs = args
    M = a.shape[0]
    seg = wsf.numel() // 3
    current = torch.cuda.current_stream()
    side = fp._sm120_qkv_side_streams(a.device.index)
    for st in side:
        st.wait_stream(current)

    def g(i):
        strided.gemm_into(a, w[i * H:(i + 1) * H], asf, wsf[i * seg:(i + 1) * seg],
                          qs[i:i + 1], out, i * M * H, H, i)
    g(0)
    for i, st in enumerate(side):
        with torch.cuda.stream(st):
            g(i + 1)
    for st in side:
        current.wait_stream(st)


def table():
    t0 = time.time()
    strided.module()
    build_s = time.time() - t0
    log(f"strided module load/build {build_s:.2f} s")
    graphs = {}
    keep = []
    for M in MS:
        args = mk(M, M)
        a, w, asf, wsf, ags, wgs, qn, kn, cs, pos, qs = args
        out = torch.empty(M, 3 * H, device="cuda", dtype=torch.bfloat16)
        q0, (k0, v0) = gemms_main(args)
        q0, k0, v0 = q0.clone(), k0.clone(), v0.clone()
        dst = torch.empty_like(out)
        keep.append((args, out, q0, k0, v0, dst))
        set_inplace(False)
        graphs[f"main_composed_M{M}"] = graph_of(lambda args=args: op(*args))
        set_inplace(True)
        graphs[f"cand_composed_M{M}"] = graph_of(lambda args=args: op(*args))
        graphs[f"main_gemms_M{M}"] = graph_of(lambda args=args: gemms_main(args))
        graphs[f"cand_gemms_M{M}"] = graph_of(lambda args=args, out=out: gemms_cand(args, out))
        graphs[f"main_pass_M{M}"] = graph_of(
            lambda q0=q0, k0=k0, v0=v0, dst=dst, args=args: qr.qknorm_rope(
                q0, k0, v0, dst, args[6], args[7], args[8], args[9], planar=True))
        pv = out.view(3, M, H)
        graphs[f"cand_pass_inplace_M{M}"] = graph_of(
            lambda pv=pv, out=out, args=args: qr.qknorm_rope(
                pv[0], pv[1], pv[2], out, args[6], args[7], args[8], args[9], planar=True))
    res = time_graphs(graphs)
    rows = {}
    for M in MS:
        rows[str(M)] = {k.rsplit("_M", 1)[0] + "_us": round(v["ms"] * 1000, 1)
                        for k, v in res.items() if k.endswith(f"_M{M}")}
        log("M", M, rows[str(M)])
    head = res["cand_composed_M4176"]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": head["ms"], "noise_floor": head["noise"]}},
        "output": {"class": "exact"},
        "strided_module_load_s": round(build_s, 2),
        "table_us": rows,
        "detail": res,
    }))


# --------------------------------------------------------------------------- check
def tensors(args):
    return [t for t in args if isinstance(t, torch.Tensor)]


def check_op(M, results):
    args = mk(M, 3 * M + 1)
    ins = tensors(args)
    pristine = [t.clone() for t in ins]
    set_inplace(False)
    ref = op(*args).clone()
    set_inplace(True)
    nbytes = M * 3 * H * 2
    first = None
    n_ref = n_first = n_inputs = n_reuse = 0
    ok_layout = True
    for it in range(WARMUP + ITERS):
        n_inputs += all(torch.equal(t, p) for t, p in zip(ins, pristine))
        for t, p in zip(ins, pristine):
            t.copy_(p)
        # Prefill the block the op's output will reuse with 0xFFFF (bf16 NaN), then free it.
        junk = torch.empty(M, 3 * H, device="cuda", dtype=torch.bfloat16)
        junk.view(torch.int16).fill_(-1)
        junk_ptr = junk.data_ptr()
        del junk
        o = op(*args)
        n_reuse += o.data_ptr() == junk_ptr
        ok_layout &= tuple(o.shape) == (M, 3 * H) and o.is_contiguous()
        if it < WARMUP:
            continue
        if first is None:
            first = o.clone()
        n_ref += bool(torch.equal(o, ref))
        n_first += bool(torch.equal(o, first))
    n_inputs_final = all(torch.equal(t, p) for t, p in zip(ins, pristine))
    ok = (n_ref == ITERS and n_first == ITERS and n_inputs == WARMUP + ITERS and n_inputs_final
          and ok_layout and not torch.isnan(ref.float()).any().item())
    results[f"op_M{M}"] = {
        "bitwise_vs_main_planar": f"{n_ref}/{ITERS}",
        "bitwise_vs_call0": f"{n_first}/{ITERS}",
        "inputs_unchanged_before_reset": f"{n_inputs}/{WARMUP + ITERS}",
        "prefill_0xFFFF_block_reused": f"{n_reuse}/{WARMUP + ITERS}",
        "ok": ok,
    }
    log(f"op_M{M}", results[f"op_M{M}"])
    return ok


def check():
    results, passed = {}, True
    log("route", route(), "planar", fp.SM120_QKV_PLANAR)
    t0 = time.time()
    strided.module()
    results["strided_module_load_s"] = round(time.time() - t0, 2)
    for M in MS:
        args = mk(M, 0)
        strided.preload(args[0].device)
        break
    n_ws = len(strided._WORKSPACES)
    for M in MS:
        passed &= check_op(M, results)
    keys_after = [str(k) for k in qr._COMPILED]
    results["pass_compile_keys"] = keys_after
    # one pass compile per (src_ld, copy_v, planar) layout, not per M
    passed &= len(qr._COMPILED) <= 2
    # CUDA graph: capture op (side streams), replay 3x, bitwise vs eager
    for M in (4176, 4096):
        args = mk(M, 11 + M)
        set_inplace(False)
        main_ref = op(*args).clone()
        set_inplace(True)
        eager = op(*args).clone()
        torch.cuda.synchronize()
        stats0 = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_out = op(*args)
        torch.cuda.synchronize()
        cand_allocs = torch.cuda.memory_stats()["allocation.all.allocated"] - stats0
        set_inplace(False)
        stats1 = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph_main = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_main):
            op(*args)
        torch.cuda.synchronize()
        main_allocs = torch.cuda.memory_stats()["allocation.all.allocated"] - stats1
        set_inplace(True)
        gok = True
        for _ in range(3):
            g_out.view(torch.int16).fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            gok &= bool(torch.equal(g_out, eager)) and bool(torch.equal(g_out, main_ref))
        ws_same = len(strided._WORKSPACES) == n_ws
        results[f"cuda_graph_M{M}"] = {
            "replay_x3_bitwise_vs_eager_and_main": gok,
            "workspace_dict_unchanged": ws_same,
            "allocations_in_capture_candidate": cand_allocs,
            "allocations_in_capture_main": main_allocs,
        }
        # Same capture overhead in both arms; the candidate drops main's 3 fresh GEMM planes.
        passed &= gok and ws_same and main_allocs - cand_allocs == 3
        log(f"cuda_graph_M{M}", results[f"cuda_graph_M{M}"])
    results["workspaces"] = len(strided._WORKSPACES)
    print(json.dumps({
        "passed": bool(passed),
        "reference": "main's planar per-plane _sm120_cutlass_qkv (TRT-LLM FP4GemmRunner t0 x3 + "
                     "qknorm_rope into a fresh buffer with v copy), same process with "
                     "TRTLLM_FLUX3_SM120_QKV_INPLACE switched off; bitwise",
        "results": results,
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"bench": bench, "table": table, "check": check}[mode]()
