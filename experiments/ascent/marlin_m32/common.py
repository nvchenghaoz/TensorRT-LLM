# Shared helpers for the dense Marlin NVFP4 rows 17-32 launch-table (kM32Table) sweep / pair / check.
# Builds the same load-time NVFP4 W4A16 Marlin copies that linear.py's
# _build_fp8_nvfp4_decode_copy builds from an FP8 per-tensor weight, plus an
# independent FP32 reference from the dequantized NVFP4 operands. Rows 17-32 run
# marlin_nvfp4_gemm with workspace_mode=2 (FP32 split-K reduce), as main routes them.
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "marlin_m16_common",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "marlin_m16", "common.py"))
_m16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m16)
LUT, SHAPES, CALLS = _m16.LUT, _m16.SHAPES, _m16.CALLS
eq, err, log, make_copy, nbytes, time_graphs = (_m16.eq, _m16.err, _m16.log, _m16.make_copy, _m16.nbytes,
                                                _m16.time_graphs)


def gemm(x, c, n, k, mode=2):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, c[0], scale_a=None, scale_b=c[1], alpha=None,
                                              weight_global_scale=c[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=n, size_k=k,
                                              workspace_mode=mode)


def set_cfg(cfg):
    """None -> this tree's default; 'main' -> table off; else explicit TRTLLM_MARLIN_M32_CFG."""
    os.environ.pop("TRTLLM_MARLIN_M32_CFG", None)
    os.environ.pop("TRTLLM_MARLIN_M32_TABLE", None)
    if cfg == "main":
        os.environ["TRTLLM_MARLIN_M32_TABLE"] = "0"
    elif cfg is not None and cfg != "table":
        os.environ["TRTLLM_MARLIN_M32_CFG"] = cfg


def capture(x, copies, n, k, mode=2):
    s0 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for _ in range(2):
            [gemm(x, c, n, k, mode) for c in copies]
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        o = [gemm(x, c, n, k, mode) for c in copies]
    return gr, o
