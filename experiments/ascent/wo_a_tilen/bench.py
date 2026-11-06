"""Standalone gate + bitwise check for SM120 FP8 block-scale strided BMM TileN 128/64/32.

Compiles harness.cu (the worktree's sm120 headers) with torch cpp_extension, then:
  * bitwise compares TileN 64/32 vs TileN 128 (unchanged types) at the wo_a shape
    (L=2, N=1024, K=4096) for contiguous D and strided D (o_lora[M,2,1024].transpose(0,1)),
    NaN-poisoning the output before every call;
  * times each tile in a CUDA graph with rotating weight copies (cold L2).
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


def log(*a):
    print(*a, flush=True)


def find_cutlass():
    cands = []
    env = os.environ.get("CUTLASS_INCLUDE")
    if env:
        cands.append(env)
    exe = os.path.realpath(sys.executable)
    parts = exe.split(os.sep)
    if "native" in parts:
        root = os.sep.join(parts[: parts.index("native") + 1])
        for pat in [
            "build/*/_deps/cutlass-src/include",
            "build/*/*/_deps/cutlass-src/include",
            "build/*/*/*/_deps/cutlass-src/include",
            "trtllm/cpp/build/_deps/cutlass-src/include",
            "*/cpp/build/_deps/cutlass-src/include",
            "*/*/cpp/build/_deps/cutlass-src/include",
        ]:
            cands += sorted(glob.glob(os.path.join(root, pat)))
    cands += sorted(glob.glob(os.path.join(WT, "cpp/build*/_deps/cutlass-src/include")))
    try:
        import flashinfer

        fi = os.path.dirname(flashinfer.__file__)
        cands += [os.path.join(fi, "data/cutlass/include"), os.path.join(fi, "../3rdparty/cutlass/include")]
    except Exception as e:  # noqa
        log("no flashinfer", e)
    for c in cands:
        if os.path.isfile(os.path.join(c, "cute/tensor.hpp")):
            return os.path.abspath(c)
    raise RuntimeError(f"cutlass not found in {cands}")


def build():
    from torch.utils.cpp_extension import load

    cut = find_cutlass()
    log("cutlass include:", cut)
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
                log(f"hb compiling {time.time() - t0:.0f}s")

    th = threading.Thread(target=hb, daemon=True)
    th.start()
    bdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "wo_a_tilen_build_" + str(os.getpid()))
    os.makedirs(bdir, exist_ok=True)
    mod = load(
        name="wo_a_tilen_harness",
        sources=[os.path.join(HERE, "harness.cu")],
        extra_include_paths=inc,
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "-gencode=arch=compute_120a,code=sm_120a",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DCUTLASS_ENABLE_GDC_FOR_SM90=1",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_cflags=["-std=c++20", "-O2"],
        build_directory=bdir,
        verbose=False,
    )
    done[0] = True
    return mod


def make_inputs(L, M, N, K, gen, ncopies=1):
    dev = "cuda"
    a = (torch.randn(L, M, K, device=dev, generator=gen) * 2).to(torch.float8_e4m3fn)
    sm = (M + 3) // 4 * 4
    sk = (K + 511) // 512
    sfa = torch.randint(118, 134, (L, sk, sm, 4), device=dev, dtype=torch.uint8, generator=gen).view(torch.int32)
    sfa = sfa.reshape(L, sk, sm).contiguous()
    bs, sfbs = [], []
    sn = (N + 3) // 4 * 4
    for _ in range(ncopies):
        b = (torch.randn(L, N, K, device=dev, generator=gen) * 2).to(torch.float8_e4m3fn)
        sfb = torch.randint(118, 134, (L, sk, sn, 4), device=dev, dtype=torch.uint8, generator=gen).view(torch.int32)
        bs.append(b)
        sfbs.append(sfb.reshape(L, sk, sn).contiguous())
    return a, sfa, bs, sfbs


def out_views(L, M, N, strided):
    if strided:
        base = torch.empty(M, L, N, device="cuda", dtype=torch.bfloat16)
        return base, base.transpose(0, 1)
    base = torch.empty(L, M, N, device="cuda", dtype=torch.bfloat16)
    return base, base


def main():
    tiles = [(32, 128), (32, 64), (32, 32)]
    mod = build()
    log("built")
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    L, N, K = 2, 1024, 4096
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    Ms = [int(x) for x in os.environ.get("MS", "1,2,4,8,16,32,64,128,256,512,2048").split(",")]
    allpass = True
    checks = {}
    # ---- bitwise ----
    for M in Ms:
        a, sfa, bs, sfbs = make_inputs(L, M, N, K, gen)
        b, sfb = bs[0], sfbs[0]
        # reference dequant fp32
        sk = K // 512
        ea = (sfa.view(torch.uint8).reshape(L, sk, -1, 4)[:, :, :M, :].float() - 127).exp2()  # L,sk,M,4
        ea = ea.permute(0, 2, 1, 3).reshape(L, M, sk * 4).repeat_interleave(128, dim=2)
        eb = (sfb.view(torch.uint8).reshape(L, sk, -1, 4)[:, :, :N, :].float() - 127).exp2()
        eb = eb.permute(0, 2, 1, 3).reshape(L, N, sk * 4).repeat_interleave(128, dim=2)
        ref = torch.bmm(a.float() * ea, (b.float() * eb).transpose(1, 2))
        res = {}
        for strided in (False, True):
            for tm, tn in [(t[0] if M <= 64 else 64, t[1]) for t in tiles]:
                outs = []
                for it in range(3):
                    base, d = out_views(L, M, N, strided)
                    base.fill_(float("nan"))
                    mod.run(tm, tn, a, b, sfa, sfb, d, nsm)
                    torch.cuda.synchronize()
                    outs.append(base.clone())
                res[(strided, tn)] = outs
        ok = True
        for strided in (False, True):
            r0 = res[(strided, 128)][0]
            if not torch.isfinite(r0).all():
                ok = False
                log("ref nonfinite", M, strided)
            d = r0.float() if not strided else r0.transpose(0, 1).float()
            err = ((d - ref).abs() / (ref.abs() + 1.0)).max().item()
            for tn in (128, 64, 32):
                for o in res[(strided, tn)]:
                    same = torch.equal(o.view(torch.int16), r0.view(torch.int16))
                    if not same:
                        ok = False
                        log(f"MISMATCH M={M} strided={strided} tn={tn} maxdiff={(o.float()-r0.float()).abs().max().item()}")
            checks[f"M{M}_{'strided' if strided else 'contig'}"] = {"bitwise_all_tiles": ok, "rel_err_vs_fp32": err}
        # strided vs contig equal
        same_sc = torch.equal(res[(True, 128)][0].transpose(0, 1).contiguous().view(torch.int16),
                              res[(False, 128)][0].view(torch.int16))
        ok = ok and same_sc
        log(f"M={M} bitwise(64,32 vs 128, contig+strided, 3 calls each)={ok} strided==contig {same_sc}")
        allpass = allpass and ok
    # ---- timing ----
    timing = {}
    ncopies = 24  # 24 x 8.4 MB = 201 MB > 128 MB L2
    for M in [int(x) for x in os.environ.get("TMS", "1,2,4,8,64,128,256,512,2048").split(",")]:
        a, sfa, bs, sfbs = make_inputs(L, M, N, K, gen, ncopies)
        base, d = out_views(L, M, N, True)
        row = {}
        for tm, tn in tiles:
            tm = tm if M <= 64 else 64
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for i in range(ncopies):
                    mod.run(tm, tn, a, bs[i], sfa, sfbs[i], d, nsm)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(ncopies):
                    mod.run(tm, tn, a, bs[i], sfa, sfbs[i], d, nsm)
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(30):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000 / ncopies)
            ts.sort()
            med = ts[len(ts) // 2]
            row[tn] = {"us": round(med, 2), "noise": round((ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / med, 4)}
        timing[M] = row
        log(f"M={M} timing us/call (cold-L2 graph, strided D): {row}")
    print(json.dumps({"passed": allpass, "checks": checks, "timing": timing}), flush=True)


if __name__ == "__main__":
    main()
