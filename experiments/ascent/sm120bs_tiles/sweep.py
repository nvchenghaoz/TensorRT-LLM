"""S0/S1 sweep for SM120BlockScaledKernel tile configs at the DeepSeek-V4.1-Flash TP4 per-rank prefill shapes.

Shapes (per rank, pitfall 1430): shared gate_up [M,5120]x[2304,5120], shared down [M,1152]x[5120,1152] (L=1 GEMM,
contiguous D, stride 0 like gemm_dispatch_sm120), wo_a BMM 2x[M,4096]x[1024,4096] with strided D
(o_lora[M,2,1024].transpose(0,1), like strided_batch_gemm_dispatch_sm120 in the model).

For each shape and M:
  * bitwise: every tile's output (NaN-poisoned before each of 3 calls) == the production tile (<32,128,4> for M<=64,
    <64,128,4> otherwise); production tile's rel err vs an FP32 dequantized reference is printed.
  * timing: CUDA graph over rotating weight copies (> 2x L2 of weights), 5 warmup + ITERS replays, median us/call
    and (p90-p10)/median.
Env: MS (timing Ms), CMS (check-only Ms), ITERS, SETS (A,B).
"""
import glob
import json
import os
import sys
import threading
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "wo_a_tilen"))

SHAPES = {
    "gate_up": dict(L=1, N=2304, K=5120, strided=False),
    "down": dict(L=1, N=5120, K=1152, strided=False),
    "wo_a": dict(L=2, N=1024, K=4096, strided=True),
}


def log(*a):
    print(*a, flush=True)


def build(tileset):
    from torch.utils.cpp_extension import load

    import bench  # wo_a_tilen/bench.py: find_cutlass

    cut = bench.find_cutlass()
    inc = [
        os.path.join(WT, "cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm"),
        cut,
        os.path.join(os.path.dirname(cut), "tools/util/include"),
    ]
    done = [False]

    def hb():
        t0 = time.time()
        while not done[0]:
            time.sleep(30)
            if not done[0]:
                log(f"hb compiling set {tileset} {time.time() - t0:.0f}s")

    threading.Thread(target=hb, daemon=True).start()
    bdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"sm120bs_tiles_{tileset}_{os.getpid()}")
    os.makedirs(bdir, exist_ok=True)
    try:
        mod = load(
            name=f"sm120bs_tiles_{tileset}",
            sources=[os.path.join(HERE, "harness.cu")],
            extra_include_paths=inc,
            extra_cuda_cflags=[
                "-std=c++20", "-O3", "-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr",
                "--expt-extended-lambda", "-DCUTLASS_ENABLE_GDC_FOR_SM90=1", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                f"-DTILESET_{tileset}", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"
            ],
            extra_cflags=["-std=c++20", "-O2", f"-DTILESET_{tileset}"],
            build_directory=bdir,
            verbose=False,
        )
    except Exception as e:  # noqa
        done[0] = True
        log(f"BUILD FAILED set {tileset}: {str(e)[-3000:]}")
        return None
    done[0] = True
    return mod


def make_inputs(L, M, N, K, gen, ncopies):
    a = (torch.randn(L, M, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
    sm, sk, sn = (M + 3) // 4 * 4, (K + 511) // 512, (N + 3) // 4 * 4
    sfa = torch.randint(118, 134, (L, sk, sm, 4), device="cuda", dtype=torch.uint8, generator=gen)
    sfa = sfa.view(torch.int32).reshape(L, sk, sm).contiguous()
    bs, sfbs = [], []
    for _ in range(ncopies):
        bs.append((torch.randn(L, N, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn))
        sfb = torch.randint(118, 134, (L, sk, sn, 4), device="cuda", dtype=torch.uint8, generator=gen)
        sfbs.append(sfb.view(torch.int32).reshape(L, sk, sn).contiguous())
    return a, sfa, bs, sfbs


def out_views(L, M, N, strided):
    if strided:
        base = torch.empty(M, L, N, device="cuda", dtype=torch.bfloat16)
        return base, base.transpose(0, 1)
    base = torch.empty(L, M, N, device="cuda", dtype=torch.bfloat16)
    return base, base


def fp32_ref(a, sfa, b, sfb, L, M, N, K):
    sk = (K + 511) // 512
    ea = (sfa.view(torch.uint8).reshape(L, sk, -1, 4)[:, :, :M, :].float() - 127).exp2()
    ea = ea.permute(0, 2, 1, 3).reshape(L, M, sk * 4).repeat_interleave(128, dim=2)[:, :, :K]
    eb = (sfb.view(torch.uint8).reshape(L, sk, -1, 4)[:, :, :N, :].float() - 127).exp2()
    eb = eb.permute(0, 2, 1, 3).reshape(L, N, sk * 4).repeat_interleave(128, dim=2)[:, :, :K]
    return torch.bmm(a.float() * ea, (b.float() * eb).transpose(1, 2))


def main():
    sets = os.environ.get("SETS", "A,B").split(",")
    mods = {}
    tiles = []
    for s in sets:
        t0 = time.time()
        m = build(s)
        if m is None:
            continue
        mods[s] = m
        for tm, tn, st, smem in m.tiles():
            tiles.append((s, tm, tn, st, smem))
        log(f"built set {s} in {time.time() - t0:.0f}s")
    log("tiles (set, TileM, TileN, stages, smem bytes):", tiles)
    assert "A" in mods, "production tile set must build"
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    iters = int(os.environ.get("ITERS", "30"))
    ms_t = [int(x) for x in os.environ.get("MS", "128,256,512,1024,2048").split(",")]
    ms_c = [int(x) for x in os.environ.get("CMS", "1,4,16,64,96,200").split(",") if x]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(4321)
    allpass = True
    result = {"tiles": tiles, "checks": {}, "timing": {}}

    def run(t, a, b, sfa, sfb, d):
        s, tm, tn, st, _ = t
        mods[s].run(tm, tn, st, a, b, sfa, sfb, d, nsm)

    def prod_tile(M):
        tm = 32 if M <= 64 else 64
        return [t for t in tiles if t[0] == "A" and t[1] == tm and t[2] == 128][0]

    for name, sh in SHAPES.items():
        L, N, K, strided = sh["L"], sh["N"], sh["K"], sh["strided"]
        wbytes = L * N * K
        ncopies = max(4, -(-300 * 2**20 // wbytes))
        for M in sorted(set(ms_c + ms_t)):
            a, sfa, bs, sfbs = make_inputs(L, M, N, K, gen, ncopies if M in ms_t else 1)
            # ---- bitwise ----
            pt = prod_tile(M)
            rb, rd = out_views(L, M, N, strided)
            rb.fill_(float("nan"))
            run(pt, a, bs[0], sfa, sfbs[0], rd)
            torch.cuda.synchronize()
            ref = rd.float()
            fref = fp32_ref(a, sfa, bs[0], sfbs[0], L, M, N, K)
            rel = ((ref - fref).abs() / (fref.abs() + 1.0)).max().item()
            finite = bool(torch.isfinite(ref).all())
            ck = {"prod_tile": f"{pt[1]}x{pt[2]}x{pt[3]}", "prod_rel_err_vs_fp32": rel, "prod_finite": finite}
            ok_all = finite and rel < 2e-2
            for t in tiles:
                ok = True
                for _ in range(3):
                    ob, od = out_views(L, M, N, strided)
                    ob.fill_(float("nan"))
                    run(t, a, bs[0], sfa, sfbs[0], od)
                    torch.cuda.synchronize()
                    ok &= torch.equal(ob.view(torch.int16), rb.view(torch.int16))
                ck[f"{t[1]}x{t[2]}x{t[3]}"] = bool(ok)
                ok_all &= ok
            result["checks"][f"{name}_M{M}"] = ck
            allpass &= ok_all
            log(f"{name} M={M} bitwise all tiles == prod: {ok_all} {ck}")
            if M not in ms_t:
                continue
            # ---- timing ----
            row = {}
            ob, od = out_views(L, M, N, strided)
            for t in tiles:
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    for i in range(ncopies):
                        run(t, a, bs[i], sfa, sfbs[i], od)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for i in range(ncopies):
                        run(t, a, bs[i], sfa, sfbs[i], od)
                for _ in range(5):
                    g.replay()
                torch.cuda.synchronize()
                ts = []
                for _ in range(iters):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1000 / ncopies)
                ts.sort()
                med = ts[len(ts) // 2]
                noise = (ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med
                tflops = 2.0 * L * M * N * K / (med * 1e-6) / 1e12
                row[f"{t[1]}x{t[2]}x{t[3]}"] = {"us": round(med, 2), "noise": round(noise, 4), "tflops": round(tflops, 1)}
                del g
            result["timing"][f"{name}_M{M}"] = row
            best = min(row.items(), key=lambda kv: kv[1]["us"])
            log(f"{name} M={M} ncopies={ncopies} us/call: " +
                " ".join(f"{k}={v['us']}({v['tflops']}T,n{v['noise']})" for k, v in row.items()) +
                f"  BEST {best[0]} {best[1]['us']}")
            del bs, sfbs
            torch.cuda.empty_cache()
    result["passed"] = bool(allpass)
    os.makedirs(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), exist_ok=True)
    with open(os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "sweep.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"passed": bool(allpass), "n_checks": len(result["checks"])}), flush=True)


if __name__ == "__main__":
    main()
