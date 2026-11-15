"""Paired kernel bench for idea 64 (down_proj wave split).

Runs in BOTH trees: builds 5 real NVFP4 Linear modules with the down_proj
shape from config.json (cold weights: 5 x 65 MB rotated > 2x L2), flags them
the way Gemma4's _Gemma4GeluQuantMLP flags down_proj, and times
quant_method.apply (the production dispatch) on a pre-quantized
Fp4QuantizedTensor.  On main the flag is ignored (single nvfp4_gemm); on the
branch M in [4096, 8192] takes the wave split.  main is autotuned once at
M=8192 as the engine warmup does.  CUDA events, 5 warmup + 20 timed, median
p10/p90.  Last stdout line: evidence JSON.
"""
import json
import os
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, make_act, make_weight, read_dims, receipts, time_fn  # noqa

import tensorrt_llm  # noqa
from tensorrt_llm._torch.autotuner import autotune  # noqa
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode  # noqa
from tensorrt_llm._torch.utils import Fp4QuantizedTensor  # noqa
from tensorrt_llm.mapping import Mapping  # noqa
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa

DEFAULT_MS = ([4096, 4160, 4352, 4416, 4480, 4544, 4800, 4801, 4992, 5056, 5632, 5696, 6144, 6208,
               6656, 6720, 6784, 6848] + list(range(7040, 8193, 64)) + [7999, 8001, 8100])


def build_linears(n, k, nw):
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    lins = []
    for i in range(nw):
        lin = Linear(k, n, bias=False, dtype=torch.bfloat16,
                     mapping=Mapping(world_size=1, tp_size=1, rank=0),
                     tensor_parallel_mode=TensorParallelMode.ROW, quant_config=qc,
                     reduce_output=False).cuda()
        w_fp4, w_sf, wgs = make_weight(n, k, 100 + i)
        with torch.no_grad():
            lin.weight.data.copy_(w_fp4.view(lin.weight.dtype).view(lin.weight.shape))
            lin.weight_scale.data.copy_(w_sf.view(-1))
            lin.weight_scale_2.data.fill_(float(1.0 / wgs))
        lin._wgs = wgs
        lin._gemma4_down_wave_split = True
        lins.append(lin)
    return lins


def set_act_scale(lins, ags):
    for lin in lins:
        with torch.no_grad():
            lin.input_scale.data.copy_(ags)
            lin.inv_input_scale.data.copy_(1.0 / ags)
            lin.alpha.data.copy_((1.0 / (ags * lin._wgs)).float())


def call(lin, a_fp4, a_sf):
    return lin.quant_method.apply(lin, Fp4QuantizedTensor(a_fp4, a_sf), None)


def main():
    n, k, _ = read_dims()
    log(f"down_proj N={n} K={k}")
    ms = [int(x) for x in os.environ["MS"].split(",")] if os.environ.get("MS") else DEFAULT_MS
    nw = 5
    lins = build_linears(n, k, nw)
    mmax = max(ms + [8192])
    a_fp4, a_sf, ags = make_act(mmax, k, 1)
    set_act_scale(lins, ags)
    sfk = ((k // 16) + 3) // 4 * 4
    a_sf = a_sf.view(-1)
    try:
        from tensorrt_llm._torch.modules.gemma4 import down_wave_split as dws
        log(f"branch tree: wave split ENABLED={dws.ENABLED}")
    except ImportError:
        dws = None
        log("base tree: no down_wave_split module")
    with autotune():
        call(lins[0], a_fp4[:8192], a_sf[:8192 * sfk])
    torch.cuda.synchronize()
    # first non-tuning call loads FI on the branch (warmup, untimed)
    call(lins[0], a_fp4[:4096], a_sf[:4096 * sfk])
    torch.cuda.synchronize()
    log("tuned + warm")

    batch = {}
    rec = []
    for j, m in enumerate(ms):
        am = a_fp4[:m]
        sm = a_sf[: ((m + 127) // 128) * 128 * sfk]
        if j % 10 == 0:
            # clock / throttle receipt sampled while the GPU is busy
            for i in range(30):
                call(lins[i % nw], am, sm)
            rec.append(dict(m=m, **receipts(f" busy M={m}")))
            torch.cuda.synchronize()
        st = time_fn(lambda i: call(lins[i % nw], am, sm), iters=20, warmup=5)
        batch[str(m)] = {"kernel_ms": st["med"] / 1000.0, "p10_ms": st["p10"] / 1000.0,
                         "p90_ms": st["p90"] / 1000.0, "noise_floor": st["noise"]}
        log(f"M={m}: {st['med']:.1f} us (p10 {st['p10']:.1f} p90 {st['p90']:.1f}) noise {st['noise']:.4f}")
    rec.append(receipts(" end"))
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    json.dump({"batch": batch, "receipts": rec}, open(os.path.join(art, "bench.json"), "w"), indent=1)
    print(json.dumps({"batch": batch, "output": {"class": "exact"}}))


if __name__ == "__main__":
    main()
