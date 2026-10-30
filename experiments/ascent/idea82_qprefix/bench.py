"""Idea 82 timing (run_gpu_pair, same script in both trees): block 27's fork-to-join.

Served route of single_blocks[27] (primsts.qkv_fp8_attention_forward keep_rows=16384 ->
flux3_primsts.fp8_prequant_attention_nvfp4_out_side_gate_up, fmha_first order): fork event, the
G3 nvfp4_out FMHA over all S rows on the current stream, the side-stream CUTLASS SwiGLU gate_up at
M=16384 (N=18432, K=3072, NVFP4) waiting on the event, join. Base tree: the FMHA is
run_nvfp4_out (all S Q rows, grid.x 172 at S=32848). Branch tree: run_nvfp4_out_q_rows(16384)
(grid.x 86), as the served route calls it unless TRTLLM_FLUX3_LAST_BLOCK_QPREFIX=0.
Labels:
  i2i-2048   S=32848 fork-to-join, per pair (4 pairs per CUDA-graph replay)
  t2i-2048   S=16464 fork-to-join (grid.x 86 in both trees; must not move)
Also single-call medians: fmha alone and gate_up alone per S.
kernel_ms = median per pair; noise = (p90 - p10) / median.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "g3nfo"))

import b186_nfo as b  # noqa: E402
from g3nfo.bench import MLP2, gate_up_inputs, stats, time_graph  # noqa: E402

log = b.log
QR = 16384
LABELS = {"i2i-2048": 32848, "t2i-2048": 16464}


def main():
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as swc

    has_q = hasattr(lib, "fmha_v2_flux3_d128_nvfp4_out_q_rows") and \
        os.environ.get("TRTLLM_FLUX3_LAST_BLOCK_QPREFIX", "1").strip() != "0"
    sample = b.make_qkv(4176, 7, False)
    lev = (*lib.load_trim_levers(tuple(lib.ROUTED_LEVERS), sample["k"], sample["v"]), "nvfp4_out")
    t0 = time.time()
    mod = lib.get_flux3_d128_module(tuple(lev))
    log(f"[build] {lev} q_rows_route={has_q} {time.time() - t0:.1f}s")
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(1234)
    a, w, sfa, sfb, alpha, inv = gate_up_inputs(QR, gen)
    go, gsf = swc.allocate_outputs(QR, MLP2, a.device)
    gmod = swc.get_module()
    side = torch.cuda.Stream()

    def gate_up():
        swc.swiglu_packed(a, w, sfa, sfb, alpha, inv, module=gmod, output=go, output_scale=gsf)

    out = {"batch": {}, "single_call_ms": {}, "levers": list(lev), "q_rows_route": has_q}
    for lab in sys.argv[1:] or list(LABELS):
        S = LABELS[lab]
        inp = b.make_qkv(S, 900 + S, False)
        nout = b.NfoOut(S, post=4096)
        sc = inp["scales"]

        def fmha():
            if has_q:
                lib.fmha_v2_flux3_d128_nvfp4_out_q_rows(
                    inp["q"], inp["k"], inp["v"], nout.fp4, nout.sf, gscale, b.H, b.D, S, QR,
                    scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], module=mod)
            else:
                b.call_nfo(lib, mod, inp, nout, gscale)

        gate_up()
        fmha()
        torch.cuda.synchronize()

        def body():
            cur = torch.cuda.current_stream()
            for _ in range(4):
                side.wait_stream(cur)
                fmha()
                with torch.cuda.stream(side):
                    gate_up()
                cur.wait_stream(side)

        iters = 20 if S > 30000 else 50
        allts = []
        for r in range(3):
            ts = [t / 4 for t in time_graph(body, iters)]
            allts += ts
            m, nf = stats(ts)
            tf = stats([t / 4 for t in time_graph(lambda: [fmha() for _ in range(4)], iters)])[0]
            tg = stats([t / 4 for t in time_graph(lambda: [gate_up() for _ in range(4)], iters)])[0]
            out["single_call_ms"].setdefault(f"S{S}_fmha", []).append(round(tf, 4))
            out["single_call_ms"].setdefault(f"S{S}_gate_up_M{QR}", []).append(round(tg, 4))
            log(f"[{lab}] round {r} fork-to-join {m:.4f} ms noise {nf:.4f} "
                f"(fmha alone {tf:.4f}, gate_up alone {tg:.4f})")
        m, nf = stats(allts)
        out["batch"][lab] = {"kernel_ms": round(m, 4), "noise_floor": round(nf, 4), "S": S}
        del inp, nout
        torch.cuda.empty_cache()
    out["output"] = {"class": "exact"}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
