"""Idea 79 Stage 1: SM120 NVFP4 Pingpong strided GEMM vs Cooperative (same code) and TRT-LLM t0.

Synthetic NVFP4 tensors at FLUX.3 klein shapes (no model load):
  QKV planes: M in {4096, 4176, 4256}, N=3072 per plane, K=3072, per-plane alpha.
  Tail pair: down N=3072 K=9216, attn_out N=3072 K=3072 at M in {4096, 4176} (GEMMs only).
Timing: CUDA graphs of REPS calls per replay (graph replay decides, plan 911), arms interleaved,
median of SAMPLES replays. Correctness: bitwise per plane vs TRT-LLM t0 with the output prefilled
with 0xFFFF before every call, >= 4 calls, call k vs call 0, inputs unchanged.
"""

import json
import os
import pathlib
import statistics
import sys
import threading
import time

import torch

REPS = 10
SAMPLES = int(os.environ.get("IDEA79_SAMPLES", "80"))
WS_BYTES = 1 << 20
FI_WS_BYTES = 128 * 1024 * 1024
DEV = torch.device("cuda", 0)


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


def build_module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    name = "flux3_fp4_gemm_sm120_pingpong.cu"
    src = jit_env.FLASHINFER_CSRC_DIR / name
    if not src.exists():
        src = pathlib.Path(os.getcwd()) / "experiments/ascent-flashinfer/csrc" / name
    log("source:", src, src.exists())
    flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        "flux3_fp4_gemm_sm120_pingpong",
        [src],
        extra_cuda_cflags=flags
        + ["-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
    stop = heartbeat("build")
    t0 = time.time()
    try:
        mod = spec.build_and_load()
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        log("BUILD FAILED")
        log(msg[-12000:])
        log(json.dumps({"build": "failed", "error_tail": msg[-3000:]}))
        sys.exit(1)
    finally:
        stop.set()
    log(f"build_and_load {time.time() - t0:.1f}s")
    return mod


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


def main():
    torch.manual_seed(1234)
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    log("torch", torch.__version__, torch.cuda.get_device_name(0))
    mod = build_module()
    info = {
        "smem_pingpong": int(mod.smem_pingpong()),
        "smem_coop": int(mod.smem_coop()),
        "stages_pingpong": int(mod.stages_pingpong()),
        "stages_coop": int(mod.stages_coop()),
    }
    log("kernel info", info)
    runner = fp._sm120_fp4_gemm_runner()
    t0_tactic = fp._SM120_LINEAR_CUTLASS_TACTIC
    log("trtllm tactic", t0_tactic)
    fi = fp._sm120_fi_fp4_module()
    log("stock FI module loaded")

    side = (torch.cuda.Stream(DEV), torch.cuda.Stream(DEV))
    ws = [torch.empty(WS_BYTES, dtype=torch.uint8, device=DEV) for _ in range(3)]
    fi_ws = [torch.empty(FI_WS_BYTES, dtype=torch.uint8, device=DEV) for _ in range(2)]

    K = 3072
    W = 3072
    # QKV weight [9216, K/2], scales [9216 * K/16], per-plane alpha.
    wq = torch.randint(0, 256, (3 * W, K // 2), dtype=torch.uint8, device=DEV)
    wq_sf = rand_sf(3 * W * (K // 16))
    seg = wq_sf.numel() // 3
    qkv_scale = torch.tensor([0.011, 0.013, 0.017], dtype=torch.float32, device=DEV)
    pristine_w = (wq.clone(), wq_sf.clone(), qkv_scale.clone())

    results = {"info": info, "qkv": {}, "tail": {}, "bitwise": {}}
    all_ok = True

    for M in (4096, 4176, 4256):
        log(f"=== QKV M={M}")
        a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=DEV)
        a_sf = rand_sf(rup(M, 128) * (K // 16))
        a_p = (a.clone(), a_sf.clone())
        out = torch.empty((3, M, W), dtype=torch.bfloat16, device=DEV)

        def t0_plane(i):
            return runner.run_gemm(a, wq[i * W:(i + 1) * W], a_sf, wq_sf[i * seg:(i + 1) * seg],
                                   qkv_scale[i:i + 1], 0, t0_tactic, None, None)

        def strided(fn, i, slot):
            fn(a, wq[i * W:(i + 1) * W], a_sf, wq_sf[i * seg:(i + 1) * seg], qkv_scale[i:i + 1],
               out, i * M * W, W, ws[slot])

        # ---- correctness: bitwise per plane vs t0, 0xFFFF prefill, 4 calls
        refs = [t0_plane(i).clone() for i in range(3)]
        torch.cuda.synchronize()
        bw = {}
        for arm, fn in (("coop", mod.gemm_into_coop), ("pingpong", mod.gemm_into_pingpong)):
            for i in range(3):
                first = None
                ok = True
                untouched = True
                for call in range(4):
                    out.view(torch.int16).fill_(-1)
                    strided(fn, i, 0)
                    torch.cuda.synchronize()
                    got = out[i].clone()
                    ok &= torch.equal(got.view(torch.int16), refs[i].view(torch.int16))
                    if first is None:
                        first = got
                    else:
                        ok &= torch.equal(got.view(torch.int16), first.view(torch.int16))
                    for j in range(3):
                        if j != i:
                            untouched &= bool((out[j].view(torch.int16) == -1).all())
                inputs_ok = (torch.equal(a, a_p[0]) and torch.equal(a_sf, a_p[1])
                             and torch.equal(wq, pristine_w[0]) and torch.equal(wq_sf, pristine_w[1]))
                key = f"M{M}_{arm}_plane{i}"
                bw[key] = {"bitwise_vs_t0": bool(ok), "other_planes_untouched": bool(untouched),
                           "inputs_unchanged": bool(inputs_ok),
                           "nan_count": int(torch.isnan(out[i].float()).sum())}
                all_ok &= bool(ok and untouched and inputs_ok)
        # composed 3-stream strided vs refs
        for arm, fn in (("coop", mod.gemm_into_coop), ("pingpong", mod.gemm_into_pingpong)):
            ok = True
            for call in range(3):
                out.view(torch.int16).fill_(-1)
                cur = torch.cuda.current_stream()
                for s in side:
                    s.wait_stream(cur)
                strided(fn, 0, 0)
                for j, s in enumerate(side):
                    with torch.cuda.stream(s):
                        strided(fn, j + 1, j + 1)
                for s in side:
                    cur.wait_stream(s)
                torch.cuda.synchronize()
                for i in range(3):
                    ok &= torch.equal(out[i].view(torch.int16), refs[i].view(torch.int16))
            bw[f"M{M}_{arm}_composed3"] = {"bitwise_vs_t0": bool(ok)}
            all_ok &= bool(ok)
        log(json.dumps(bw))
        results["bitwise"].update(bw)

        # ---- timing
        def plane_t0():
            t0_plane(0)

        def plane_coop():
            strided(mod.gemm_into_coop, 0, 0)

        def plane_pp():
            strided(mod.gemm_into_pingpong, 0, 0)

        def composed_t0():
            cur = torch.cuda.current_stream()
            for s in side:
                s.wait_stream(cur)
            t0_plane(0)
            planes = []
            for j, s in enumerate(side):
                with torch.cuda.stream(s):
                    planes.append(t0_plane(j + 1))
            for s in side:
                cur.wait_stream(s)
            for p in planes:
                p.record_stream(cur)

        def make_composed(fn):
            def run():
                cur = torch.cuda.current_stream()
                for s in side:
                    s.wait_stream(cur)
                strided(fn, 0, 0)
                for j, s in enumerate(side):
                    with torch.cuda.stream(s):
                        strided(fn, j + 1, j + 1)
                for s in side:
                    cur.wait_stream(s)
            return run

        arms = {
            "plane_t0": plane_t0, "plane_t0_AA": plane_t0, "plane_coop": plane_coop,
            "plane_pingpong": plane_pp,
            "qkv3_t0": composed_t0, "qkv3_t0_AA": composed_t0,
            "qkv3_coop": make_composed(mod.gemm_into_coop),
            "qkv3_pingpong": make_composed(mod.gemm_into_pingpong),
        }
        tm = time_graphs(arms)
        tiles_plane = ((M + 127) // 128) * (W // 128)
        tpc_plane = tiles_plane / 188.0
        tpc_3 = 3 * tiles_plane / 188.0
        d_plane = tm["plane_coop"]["med"] - tm["plane_pingpong"]["med"]
        d_3 = tm["qkv3_coop"]["med"] - tm["qkv3_pingpong"]["med"]
        tm["derived"] = {
            "tiles_per_plane": tiles_plane,
            "tiles_per_cta_plane": round(tpc_plane, 2),
            "tiles_per_cta_qkv3": round(tpc_3, 2),
            "plane_coop_minus_pp_us": round(d_plane, 2),
            "qkv3_coop_minus_pp_us": round(d_3, 2),
            "a_hidden_per_tile_us_plane": round(d_plane / max(1, -(-tiles_plane // 188)), 3),
            "a_hidden_per_tile_us_qkv3": round(d_3 / max(1, -(-3 * tiles_plane // 188)), 3),
            "qkv3_pp_vs_t0_pct": round(100 * (tm["qkv3_pingpong"]["med"] / tm["qkv3_t0"]["med"] - 1), 2),
            "qkv3_AA_pct": round(100 * (tm["qkv3_t0_AA"]["med"] / tm["qkv3_t0"]["med"] - 1), 2),
        }
        log(json.dumps({f"M{M}": tm}))
        results["qkv"][f"M{M}"] = tm
        del a, a_sf, out, refs
        torch.cuda.empty_cache()

    # ---- tail pair (GEMMs only): down K=9216 on current, attn_out K=3072 on side
    Kd = 9216
    wd = torch.randint(0, 256, (W, Kd // 2), dtype=torch.uint8, device=DEV)
    wd_sf = rand_sf(W * (Kd // 16))
    wa = torch.randint(0, 256, (W, K // 2), dtype=torch.uint8, device=DEV)
    wa_sf = rand_sf(W * (K // 16))
    alpha_d = torch.tensor([0.003], dtype=torch.float32, device=DEV)
    alpha_a = torch.tensor([0.009], dtype=torch.float32, device=DEV)
    for M in (4096, 4176):
        log(f"=== tail M={M}")
        ad = torch.randint(0, 256, (M, Kd // 2), dtype=torch.uint8, device=DEV)
        ad_sf = rand_sf(rup(M, 128) * (Kd // 16))
        aa = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=DEV)
        aa_sf = rand_sf(rup(M, 128) * (K // 16))
        out_d = torch.empty((M, W), dtype=torch.bfloat16, device=DEV)

        def fi_gemm(x, xs, w, wsf, al, tactic, slot):
            o = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=DEV)
            fi.fp4_gemm(x, w, xs, wsf, al, o, fi_ws[slot], tactic)
            return o

        def down_t(tactic):
            return lambda: fi_gemm(ad, ad_sf, wd, wd_sf, alpha_d, tactic, 0)

        def down_pp():
            mod.gemm_into_pingpong(ad, wd, ad_sf, wd_sf, alpha_d, out_d, 0, W, ws[0])

        def down_coop():
            mod.gemm_into_coop(ad, wd, ad_sf, wd_sf, alpha_d, out_d, 0, W, ws[0])

        # correctness of down: pp/coop vs t17 (bitwise to t0) and vs t18 (reordered)
        r17 = down_t(17)().clone()
        r18 = down_t(18)().clone()
        r0 = runner.run_gemm(ad, wd, ad_sf, wd_sf, alpha_d, 0, t0_tactic, None, None).clone()
        tb = {}
        for arm, fn in (("coop", down_coop), ("pingpong", down_pp)):
            ok17 = True
            ok0 = True
            for call in range(3):
                out_d.view(torch.int16).fill_(-1)
                fn()
                torch.cuda.synchronize()
                ok17 &= torch.equal(out_d.view(torch.int16), r17.view(torch.int16))
                ok0 &= torch.equal(out_d.view(torch.int16), r0.view(torch.int16))
            diff = (out_d.float() - r18.float()).abs()
            tb[f"M{M}_down_{arm}"] = {
                "bitwise_vs_t17": bool(ok17), "bitwise_vs_trtllm_t0": bool(ok0),
                "vs_t18_max_abs": float(diff.max()),
                "vs_t18_frac_diff": float((diff > 0).float().mean()),
                "ref_absmax": float(r18.float().abs().max()),
            }
        log(json.dumps(tb))
        results["bitwise"].update(tb)

        def pair(launch_down, launch_attn):
            def run():
                cur = torch.cuda.current_stream()
                s = side[0]
                s.wait_stream(cur)
                d = launch_down()
                with torch.cuda.stream(s):
                    at = launch_attn()
                cur.wait_stream(s)
                if at is not None:
                    at.record_stream(cur)
            return run

        def attn_t18():
            return fi_gemm(aa, aa_sf, wa, wa_sf, alpha_a, 18, 1)

        out_a = torch.empty((M, W), dtype=torch.bfloat16, device=DEV)

        def attn_pp():
            mod.gemm_into_pingpong(aa, wa, aa_sf, wa_sf, alpha_a, out_a, 0, W, ws[1])
            return None

        arms = {
            "down_t18": down_t(18), "down_t18_AA": down_t(18), "down_t17": down_t(17),
            "down_coop": down_coop, "down_pingpong": down_pp,
            "attn_t18": attn_t18, "attn_pingpong": attn_pp,
            "pair_t18_t18": pair(down_t(18), attn_t18),
            "pair_t18_t18_AA": pair(down_t(18), attn_t18),
            "pair_pp_t18": pair(lambda: down_pp(), attn_t18),
            "pair_pp_pp": pair(lambda: down_pp(), attn_pp),
            "pair_t17_t18": pair(down_t(17), attn_t18),
        }
        tm = time_graphs(arms)
        log(json.dumps({f"M{M}": tm}))
        results["tail"][f"M{M}"] = tm
        del ad, ad_sf, aa, aa_sf, out_d, out_a
        torch.cuda.empty_cache()

    results["passed"] = bool(all_ok)
    m = results["qkv"]["M4176"]
    results["batch"] = {"1": {"kernel_ms": m["qkv3_pingpong"]["med"] / 1000.0,
                              "noise_floor": m["qkv3_pingpong"]["noise"]}}
    results["output"] = {"class": "exact"}
    log(json.dumps(results))


if __name__ == "__main__":
    main()
