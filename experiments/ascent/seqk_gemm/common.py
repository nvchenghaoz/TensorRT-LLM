"""Shared helpers for idea 73 (sequential-K BF16 GEMM): dims from the checkpoint, main's dispatch,
the prototype extension build, cold-L2 timing and GPU receipts."""
import json
import os
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def ckpt_dims(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    t = cfg.get("text_config", cfg)
    hid = t["hidden_size"]
    H = t["num_attention_heads"]
    KVH = t["num_key_value_heads"]
    hd = t["head_dim"]
    ghd = t.get("global_head_dim", hd)
    gkvh = t.get("num_global_key_value_heads", KVH)
    d = dict(hidden=hid, heads=H, kv_heads=KVH, head_dim=hd, global_head_dim=ghd,
             global_kv_heads=gkvh, layers=t["num_hidden_layers"])
    shapes = {
        "sl_qkv": (H * hd + 2 * KVH * hd, hid),
        "gl_qkv": (H * ghd + gkvh * ghd, hid),
        "sl_o": (hid, H * hd),
        "gl_o": (hid, H * ghd),
    }
    print("ckpt dims", d, "shapes (N,K)", shapes, flush=True)
    return d, shapes


def load_main():
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    routes = {k: mg._CublasMmTokenGate(v) for k, v in mg._SM120_BF16_CUBLAS_MM_ROUTES.items()}

    def main_op(x, w):
        gate = routes.get((w.shape[1], w.shape[0]))
        if gate is not None and x.shape[0] in gate:
            return torch.ops.trtllm.cublas_mm(x, w.t(), None, out_dtype=None,
                                              output_buffer_kind=0, group=None)
        return F.linear(x, w)

    return main_op, routes


def build_ext(name="seqk_gemm_proto"):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    from torch.utils.cpp_extension import load
    bd = os.environ.get("SEQK_BUILD_DIR", "/tmp/seqk_build_" + name)
    os.makedirs(bd, exist_ok=True)
    ext = load(name, [os.path.join(HERE, "seqk_gemm.cu")], build_directory=bd,
               extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a",
                                  "--ptxas-options=-v", "-lineinfo"],
               verbose=bool(os.environ.get("SEQK_VERBOSE")))
    print("built", name, flush=True)
    return ext


def build_prod_ext(name="seqk_prod_ext"):
    """Builds experiments/ascent/seqk_gemm/prod_ext.cu, which compiles the production kernel source
    cpp/tensorrt_llm/kernels/sm120Bf16SeqKGemm/sm120Bf16SeqKGemm.cu (test-only binding)."""
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    from torch.utils.cpp_extension import load
    root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    bd = os.environ.get("SEQK_BUILD_DIR", "/tmp/seqk_build_" + name)
    os.makedirs(bd, exist_ok=True)
    ext = load(name, [os.path.join(HERE, "prod_ext.cu")], build_directory=bd,
               extra_include_paths=[os.path.join(root, "cpp"), os.path.join(root, "cpp", "include")],
               extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a",
                                  "--ptxas-options=-v", "-std=c++20"],
               verbose=bool(os.environ.get("SEQK_VERBOSE")))
    print("built", name, flush=True)
    return ext


def gpu_receipt():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,clocks.sm,clocks.mem,temperature.gpu,"
             "power.draw,clocks_throttle_reasons.active", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as ex:  # noqa: BLE001
        out = str(ex)
    host = os.uname().nodename
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return dict(host=host, cuda_visible=vis, smi=out)


def time_cold(fn_list, warm=5, iters=30):
    """fn_list: callables cycling over distinct weight buffers (cold L2). Returns ms list."""
    n = len(fn_list)
    for i in range(warm):
        fn_list[i % n]()
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        f = fn_list[(i + warm) % n]
        s.record()
        f()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


def stats(ts):
    s = sorted(ts)
    n = len(s)
    med = statistics.median(s)
    p10 = s[max(0, int(0.1 * (n - 1)))]
    p90 = s[min(n - 1, int(round(0.9 * (n - 1))))]
    return dict(median=med, p10=p10, p90=p90, noise=(p90 - p10) / med)


def preheat(seconds=1.5):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    import time
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(0)
