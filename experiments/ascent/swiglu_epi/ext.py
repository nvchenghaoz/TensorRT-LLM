"""Build/load the STEP 0 fused gate_up + SwiGLU + NVFP4 CUTLASS extension."""
import glob
import os
import threading
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _heartbeat(stop):
    t0 = time.time()
    while not stop.is_set():
        stop.wait(30)
        if not stop.is_set():
            print(f"[build heartbeat] {time.time() - t0:.0f}s", flush=True)


def load_ext(check_layout=False):
    from torch.utils.cpp_extension import load
    roots = sorted(
        glob.glob("/home/scratch.chengzhang/trtllm-dev-qwen38-27b-20261015-171031/"
                  "runner/slots/*/native/build/candidate/cmake/_deps/cutlass-src"))
    roots = [r for r in roots if os.path.exists(os.path.join(r, "include/cutlass/cutlass.h"))]
    assert roots, "no CUTLASS source found"
    root = roots[0]
    print("CUTLASS root", root, flush=True)
    name = "swiglu_epi_ext" + ("_chk" if check_layout else "")
    bdir = os.path.join("/tmp", "swiglu_epi_build_" + name)
    os.makedirs(bdir, exist_ok=True)
    flags = [
        "-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-gencode=arch=compute_120a,code=sm_120a", "-DCUTLASS_ENABLE_GDC_FOR_SM90=1",
        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1", "-DNDEBUG", "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-Xptxas=-warn-spills"
    ]
    if check_layout:
        flags.append("-DSWIGLU_EPI_CHECK=1")
    stop = threading.Event()
    th = threading.Thread(target=_heartbeat, args=(stop, ), daemon=True)
    th.start()
    t0 = time.time()
    try:
        mod = load(name=name,
                   sources=[os.path.join(_HERE, "swiglu_gemm.cu")],
                   extra_include_paths=[os.path.join(root, "include"),
                                        os.path.join(root, "tools/util/include")],
                   extra_cuda_cflags=flags,
                   extra_cflags=["-O3", "-std=c++20"],
                   build_directory=bdir,
                   verbose=True)
    finally:
        stop.set()
    print(f"ext built in {time.time() - t0:.1f}s", flush=True)
    return mod


def make_perm(n, device="cpu"):
    """gate_up row permutation: permuted row r = source row perm[r]."""
    import torch
    d = n // 2
    assert n % 128 == 0
    c = torch.arange(128)
    a = c // 32
    nw = (c % 32) // 16
    b = (c % 16) // 8
    lq = (c % 8) // 2
    p = c % 2
    jl = 32 * nw + 4 * lq + 16 * (a // 2) + 2 * (a % 2) + b
    assert sorted(jl[p == 0].tolist()) == list(range(64))
    t = torch.arange(n // 128)[:, None]
    j = 64 * t + jl[None, :]
    perm = torch.where(p[None, :] == 0, j, d + j).reshape(-1)
    assert sorted(perm.tolist()) == list(range(n))
    return perm.to(device)
