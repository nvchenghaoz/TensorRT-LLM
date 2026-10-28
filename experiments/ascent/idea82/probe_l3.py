"""Idea 82: QKV planes as ONE batched L=3 CUTLASS NVFP4 Pingpong launch (gemm_qkv3_pingpong).

Synthetic NVFP4 tensors at FLUX.3 klein QKV shapes (no model load): M in {4096, 4176, 4256},
K=3072, weight [9216, K/2] (3 planes of N=3072), distinct per-plane alphas.

Correctness (untimed, bitwise):
  - each L=3 plane vs TRT-LLM FP4GemmRunner t0 (run_gemm) per plane; L=3 vs main's 3-stream
    per-plane Pingpong gemm_into; out prefilled with BF16 0xFFFF before every call; >= 4 calls,
    call k vs call 0; inputs vs pristine copies after every call.
  - composed op flux3_primsts._sm120_cutlass_qkv(planar=True): candidate (batched) vs main's
    per-plane 3-stream path (the same code as main, selected by strided.batched=False), 4 calls,
    the allocator block the op will reuse prefilled with 0xFFFF before each call.
  - CUDA graph: candidate composed op captured once, replayed 3x, bitwise vs eager; workspace
    dict unchanged by capture; graph node census (memset / kernel nodes) when available.
Timing: CUDA graphs of REPS calls per replay, arms interleaved (order alternates), median of
SAMPLES replays, with an A/A row.
"""

import json
import os
import statistics
import sys
import threading
import time

import torch

REPS = 10
SAMPLES = int(os.environ.get("IDEA82_SAMPLES", "100"))
DEV = torch.device("cuda", 0)
MS = [int(x) for x in os.environ.get("IDEA82_MS", "4096,4176,4256").split(",")]


def log(*a):
    print(*a, flush=True)


def heartbeat(tag):
    stop = threading.Event()
    t0 = time.time()

    def run():
        while not stop.wait(30):
            log(f"hb {tag} {time.time() - t0:.0f}s")

    threading.Thread(target=run, daemon=True).start()
    return stop


def rand_sf(n):
    # UE4M3 bytes 0x30..0x3F: 0.5 .. ~0.94, never NaN.
    return torch.randint(0x30, 0x40, (n,), dtype=torch.uint8, device=DEV)


def rup(x, m):
    return (x + m - 1) // m * m


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    med = statistics.median(xs)
    p10 = xs[int(0.1 * (n - 1))]
    p90 = xs[int(0.9 * (n - 1))]
    return {"med": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
            "noise": round((p90 - p10) / med, 4)}


def time_graphs(arms):
    graphs = {}
    for name, fn in arms.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(REPS):
                fn()
        torch.cuda.synchronize()
        graphs[name] = g
    for _ in range(5):
        for g in graphs.values():
            g.replay()
    torch.cuda.synchronize()
    times = {n: [] for n in graphs}
    order = list(graphs)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for s in range(SAMPLES):
        for n in (order if s % 2 == 0 else order[::-1]):
            e0.record()
            graphs[n].replay()
            e1.record()
            e1.synchronize()
            times[n].append(e0.elapsed_time(e1) * 1000.0 / REPS)
    del graphs
    return {n: stats(v) for n, v in times.items()}


def graph_census(g):
    try:
        from cuda.bindings import driver as cu  # type: ignore

        raw = g.raw_cuda_graph()
        err, _, n = cu.cuGraphGetNodes(raw, 0)
        err, nodes, n = cu.cuGraphGetNodes(raw, n)
        kinds = {}
        for nd in nodes:
            err, t = cu.cuGraphNodeGetType(nd)
            kinds[str(t).split(".")[-1]] = kinds.get(str(t).split(".")[-1], 0) + 1
        return {"nodes": int(n), "kinds": kinds}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


def main():
    torch.manual_seed(1234)
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    log("torch", torch.__version__, torch.cuda.get_device_name(0))
    log("sched", strided.qkv_schedule())
    stop = heartbeat("build")
    t0 = time.time()
    try:
        mod = strided.pingpong_module()
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        log("BUILD FAILED")
        log(msg[-15000:])
        log(json.dumps({"passed": False, "build": "failed", "error_tail": msg[-3000:]}))
        sys.exit(1)
    finally:
        stop.set()
    build_s = round(time.time() - t0, 1)
    log(f"pingpong_module build_and_load {build_s}s")
    strided.preload(DEV)
    ws_need = int(mod.qkv3_workspace_size(1 << 16, 3072, 3072))
    log("qkv3 workspace bytes", ws_need)
    runner = fp._sm120_fp4_gemm_runner()
    t0_tactic = fp._SM120_LINEAR_CUTLASS_TACTIC
    side = fp._sm120_qkv_side_streams(DEV.index)
    ws = [strided.workspace(DEV, i) for i in range(3)]

    K = 3072
    W = 3072
    wq = torch.randint(0, 256, (3 * W, K // 2), dtype=torch.uint8, device=DEV)
    wq_sf = rand_sf(3 * W * (K // 16))
    seg = wq_sf.numel() // 3
    qkv_scale = torch.tensor([0.011, 0.013, 0.017], dtype=torch.float32, device=DEV)
    q_norm = (torch.rand(128, device=DEV) + 0.5).to(torch.bfloat16)
    k_norm = (torch.rand(128, device=DEV) + 0.5).to(torch.bfloat16)
    pw = (wq.clone(), wq_sf.clone(), qkv_scale.clone(), q_norm.clone(), k_norm.clone())

    results = {"build_s": build_s, "qkv3_workspace_bytes": ws_need, "timing": {},
               "bitwise": {}, "graph": {}}
    all_ok = True
    orig_batched = strided.batched

    for M in MS:
        log(f"=== M={M}")
        a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=DEV)
        a_sf = rand_sf(rup(M, 128) * (K // 16))
        cos_sin = torch.randn(M, 128, device=DEV, dtype=torch.float32)
        positions = torch.randperm(M, device=DEV).to(torch.int64)
        pa = (a.clone(), a_sf.clone(), cos_sin.clone(), positions.clone())
        out = torch.empty((3, M, W), dtype=torch.bfloat16, device=DEV)
        out_p1 = torch.empty((M, 3 * W), dtype=torch.bfloat16, device=DEV)

        def inputs_ok():
            return bool(torch.equal(a, pa[0]) and torch.equal(a_sf, pa[1])
                        and torch.equal(cos_sin, pa[2]) and torch.equal(positions, pa[3])
                        and torch.equal(wq, pw[0]) and torch.equal(wq_sf, pw[1])
                        and torch.equal(qkv_scale, pw[2]) and torch.equal(q_norm, pw[3])
                        and torch.equal(k_norm, pw[4]))

        def t0_plane(i):
            return runner.run_gemm(a, wq[i * W:(i + 1) * W], a_sf, wq_sf[i * seg:(i + 1) * seg],
                                   qkv_scale[i:i + 1], 0, t0_tactic, None, None)

        def plane_pp(i, slot):
            mod.gemm_into_pingpong(a, wq[i * W:(i + 1) * W], a_sf, wq_sf[i * seg:(i + 1) * seg],
                                   qkv_scale[i:i + 1], out, i * M * W, W, ws[slot])

        def main3():
            cur = torch.cuda.current_stream()
            for s in side:
                s.wait_stream(cur)
            plane_pp(0, 0)
            for j, s in enumerate(side):
                with torch.cuda.stream(s):
                    plane_pp(j + 1, j + 1)
            for s in side:
                cur.wait_stream(s)

        def l3(raster=0, swizzle=1):
            def run():
                mod.gemm_qkv3_pingpong(a, wq, a_sf, wq_sf, qkv_scale, out, ws[0], raster,
                                       swizzle)
            return run

        def p1():
            mod.gemm_into_pingpong(a, wq, a_sf, wq_sf, qkv_scale[0:1], out_p1, 0, 3 * W, ws[0])

        def composed(batched):
            def run():
                strided.batched = (lambda: True) if batched else (lambda: False)
                try:
                    return fp._sm120_cutlass_qkv(a, wq, a_sf, wq_sf, 1.0, 1.0, q_norm, k_norm,
                                                 cos_sin, positions, qkv_scale, planar=True)
                finally:
                    strided.batched = orig_batched
            return run

        # ---------------- correctness
        bw = {}
        refs = [t0_plane(i).clone() for i in range(3)]
        torch.cuda.synchronize()
        out.view(torch.int16).fill_(-1)
        main3()
        torch.cuda.synchronize()
        ref_main = out.clone()
        main3_vs_t0 = all(torch.equal(ref_main[i].view(torch.int16), refs[i].view(torch.int16))
                          for i in range(3))
        bw["main3_vs_t0"] = bool(main3_vs_t0)
        for name, (r, sw) in {"l3": (0, 1), "l3_alongM": (1, 1), "l3_alongN": (2, 1),
                              "l3_swz2": (0, 2)}.items():
            fn = l3(r, sw)
            first = None
            ok_t0 = ok_main = ok_first = ok_in = True
            nan = 0
            for call in range(4):
                out.view(torch.int16).fill_(-1)
                fn()
                torch.cuda.synchronize()
                got = out.clone()
                for i in range(3):
                    ok_t0 &= torch.equal(got[i].view(torch.int16), refs[i].view(torch.int16))
                ok_main &= torch.equal(got.view(torch.int16), ref_main.view(torch.int16))
                if first is None:
                    first = got
                else:
                    ok_first &= torch.equal(got.view(torch.int16), first.view(torch.int16))
                ok_in &= inputs_ok()
                nan += int(torch.isnan(got.float()).sum())
            bw[name] = {"planes_bitwise_vs_t0": bool(ok_t0), "bitwise_vs_main3": bool(ok_main),
                        "calls_vs_call0": bool(ok_first), "inputs_unchanged": bool(ok_in),
                        "nan_count": nan, "calls": 4}
            all_ok &= bool(ok_t0 and ok_main and ok_first and ok_in)
        # P1 per-column-block check (single alpha = plane 0 only; plane 0 block only is exact)
        out_p1.view(torch.int16).fill_(-1)
        p1()
        torch.cuda.synchronize()
        bw["p1_block0_vs_t0"] = bool(torch.equal(out_p1[:, :W].contiguous().view(torch.int16),
                                                 refs[0].view(torch.int16)))

        # composed op: main (per-plane 3-stream) vs candidate (batched)
        nbytes = 3 * M * W
        ref_c = composed(False)().clone()
        torch.cuda.synchronize()
        first = None
        ok_c = ok_first = ok_in = True
        same_block = True
        for call in range(4):
            # prefill the block the op's torch.empty will get with BF16 0xFFFF
            junk = torch.empty((M, 3 * W), dtype=torch.bfloat16, device=DEV)
            junk.view(torch.int16).fill_(-1)
            ptr = junk.data_ptr()
            del junk
            got = composed(True)()
            same_block &= got.data_ptr() == ptr
            torch.cuda.synchronize()
            got = got.clone()
            ok_c &= torch.equal(got.view(torch.int16), ref_c.view(torch.int16))
            if first is None:
                first = got
            else:
                ok_first &= torch.equal(got.view(torch.int16), first.view(torch.int16))
            ok_in &= inputs_ok()
        bw["composed"] = {"bitwise_vs_main_composed": bool(ok_c), "calls_vs_call0": bool(ok_first),
                          "inputs_unchanged": bool(ok_in), "prefilled_block_reused": bool(same_block),
                          "calls": 4, "numel": nbytes}
        all_ok &= bool(ok_c and ok_first and ok_in)
        log(json.dumps({f"M{M}": bw}))
        results["bitwise"][f"M{M}"] = bw

        # post-landing-54 served op: nvfp4_qkv_rope_fp8_impl (planar GEMMs -> qknorm_rope_fp8),
        # candidate (batched) vs main (per-plane 3-stream), outputs (qkv8, scales, gemm).
        fp8_ok = hasattr(fp, "nvfp4_qkv_rope_fp8_impl")
        if fp8_ok:
            sm_scale = 1.0 / (128 ** 0.5)

            def fp8op(batched):
                def run():
                    strided.batched = (lambda: True) if batched else (lambda: False)
                    try:
                        return fp.nvfp4_qkv_rope_fp8_impl(a, wq, a_sf, wq_sf, q_norm, k_norm,
                                                          cos_sin, positions, qkv_scale, sm_scale)
                    finally:
                        strided.batched = orig_batched
                return run

            def as_bits(t):
                return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else (
                    t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32))

            ref8 = [t.clone() for t in fp8op(False)()]
            torch.cuda.synchronize()
            first8 = None
            ok8 = okf8 = okin8 = True
            for call in range(4):
                junk = torch.empty((M, 3 * W), dtype=torch.bfloat16, device=DEV)
                junk.view(torch.int16).fill_(-1)
                del junk
                got = [t.clone() for t in fp8op(True)()]
                torch.cuda.synchronize()
                ok8 &= all(torch.equal(as_bits(x), as_bits(y)) for x, y in zip(got, ref8))
                if first8 is None:
                    first8 = got
                else:
                    okf8 &= all(torch.equal(as_bits(x), as_bits(y)) for x, y in zip(got, first8))
                okin8 &= inputs_ok()
            # graph capture + 3 replays of the candidate fp8 op
            c8 = fp8op(True)
            c8()
            torch.cuda.synchronize()
            g8 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g8):
                gout8 = c8()
            torch.cuda.synchronize()
            rep8 = True
            for r in range(3):
                for t in gout8:
                    as_bits(t).fill_(0x7F if t.dtype == torch.float8_e4m3fn else -1)
                g8.replay()
                torch.cuda.synchronize()
                rep8 &= all(torch.equal(as_bits(x), as_bits(y)) for x, y in zip(gout8, ref8))
            del g8, gout8
            bw["fp8_op"] = {"bitwise_vs_main_qkv8_scales_gemm": bool(ok8),
                            "calls_vs_call0": bool(okf8), "inputs_unchanged": bool(okin8),
                            "graph_3_replays_bitwise_vs_eager": bool(rep8), "calls": 4}
            all_ok &= bool(ok8 and okf8 and okin8 and rep8)
            log(json.dumps({f"M{M}_fp8_op": bw["fp8_op"]}))

        # CUDA graph capture + 3 replays (candidate composed op)
        ws_keys = sorted(strided._WORKSPACES.keys())
        ws_ptrs = [strided._WORKSPACES[k].data_ptr() for k in ws_keys]
        cand = composed(True)
        cand()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gout = cand()
        torch.cuda.synchronize()
        census = graph_census(g)
        rep_ok = True
        for r in range(3):
            gout.view(torch.int16).fill_(-1)
            g.replay()
            torch.cuda.synchronize()
            rep_ok &= torch.equal(gout.view(torch.int16), ref_c.view(torch.int16))
        ws_same = (sorted(strided._WORKSPACES.keys()) == ws_keys
                   and [strided._WORKSPACES[k].data_ptr() for k in ws_keys] == ws_ptrs)
        gr = {"replays_bitwise_vs_eager": bool(rep_ok), "replays": 3,
              "workspaces_unchanged_by_capture": bool(ws_same), "inputs_unchanged": inputs_ok(),
              "census": census}
        log(json.dumps({f"M{M}_graph": gr}))
        results["graph"][f"M{M}"] = gr
        all_ok &= bool(rep_ok and ws_same)
        del g, gout

        # ---------------- timing
        arms = {
            "main3_pp": main3, "main3_pp_AA": main3,
            "l3_pp": l3(0, 1), "l3_alongM": l3(1, 1), "l3_alongN": l3(2, 1), "l3_swz2": l3(0, 2),
            "p1_proxy": p1,
            "composed_main": composed(False), "composed_main_AA": composed(False),
            "composed_cand": composed(True),
        }
        if fp8_ok:
            arms.update({"fp8op_main": fp8op(False), "fp8op_main_AA": fp8op(False),
                         "fp8op_cand": fp8op(True)})
        tm = time_graphs(arms)
        tm["derived"] = {
            "main3_minus_l3_us": round(tm["main3_pp"]["med"] - tm["l3_pp"]["med"], 2),
            "l3_minus_p1_us": round(tm["l3_pp"]["med"] - tm["p1_proxy"]["med"], 2),
            "gemm_AA_us": round(tm["main3_pp_AA"]["med"] - tm["main3_pp"]["med"], 2),
            "composed_main_minus_cand_us": round(
                tm["composed_main"]["med"] - tm["composed_cand"]["med"], 2),
            "composed_AA_us": round(tm["composed_main_AA"]["med"] - tm["composed_main"]["med"], 2),
        }
        if fp8_ok:
            tm["derived"]["fp8op_main_minus_cand_us"] = round(
                tm["fp8op_main"]["med"] - tm["fp8op_cand"]["med"], 2)
            tm["derived"]["fp8op_AA_us"] = round(
                tm["fp8op_main_AA"]["med"] - tm["fp8op_main"]["med"], 2)
        log(json.dumps({f"M{M}": tm}))
        results["timing"][f"M{M}"] = tm
        del a, a_sf, out, out_p1, refs, ref_main, ref_c
        torch.cuda.empty_cache()

    results["passed"] = bool(all_ok)
    m = results["timing"].get("M4176") or next(iter(results["timing"].values()))
    key = "fp8op_cand" if "fp8op_cand" in m else "composed_cand"
    results["batch"] = {"1": {"kernel_ms": m[key]["med"] / 1000.0,
                              "noise_floor": m[key]["noise"]}}
    results["output"] = {"class": "exact"}
    log(json.dumps(results))


if __name__ == "__main__":
    main()
