"""Branch 36 (idea 51) untimed op check on synthetic NVFP4 inputs at the served shapes.

In one process, M alternating 16464 -> 32848 -> 16384 -> 32848 -> 16464, keep_v True and False:
references = main's forward v_quant (VQ_REVERSE=False) and the TRTLLM_FLUX3_QKV_EPI_FP8=0 chain
(flux3_primsts.nvfp4_qkv_rope_fp8_impl with the route off). Candidate = the reverse walk:
  - 4 eager calls (pair.py's per-M graph holds 4 calls) vs both refs and vs call 0;
  - a CUDA graph of 4 calls (as pair.py), replayed 3x, each replay vs the refs;
  - inputs (and the per-device V partials' size) unchanged before every reset.
Bitwise: qkv8 bytes, scales[8] bits, the gemm v plane when keep_v. Last line: JSON.
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "idea14_epi"))
from op import SM, N, make_inputs  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda")
    strided.preload(dev)
    epi.preload(dev)
    fails = []
    checks = 0
    inputs = {m: make_inputs(m, dev) for m in (16384, 16464, 32848)}
    pristine = {m: [t.clone() for t in inp] for m, inp in inputs.items()}

    def call(m, keep_v):
        a, w, sfa, sfb, alpha, qn, kn, cs, pos = inputs[m]
        return fp.nvfp4_qkv_rope_fp8_impl(a, w, sfa, sfb, qn, kn, cs, pos, alpha, SM, keep_v)

    def same(got, ref, m, keep_v):
        ok = torch.equal(got[0].view(torch.uint8), ref[0].view(torch.uint8)) and torch.equal(
            got[1].view(torch.int32), ref[1].view(torch.int32))
        if keep_v:
            ok = ok and torch.equal(got[2].view(3, m, N)[2].view(torch.int16),
                                    ref[2].view(3, m, N)[2].view(torch.int16))
        return ok

    for m in (16464, 32848, 16384, 32848, 16464):
        for keep_v in (True, False):
            epi.ENABLED = False
            ref_off = tuple(t.clone() for t in call(m, True))
            epi.ENABLED = True
            epi.VQ_REVERSE = False
            ref_fwd = tuple(t.clone() for t in call(m, True))
            fused = fp._qkv_epi_fp8_route(inputs[m][0], inputs[m][1], inputs[m][5],
                                          inputs[m][6], inputs[m][7])
            if not fused:
                fails.append(f"M{m}: fused route not taken")
            epi.VQ_REVERSE = True
            first = None
            for i in range(4):
                got = call(m, keep_v)
                for name, ref in (("off", ref_off), ("fwd", ref_fwd)):
                    checks += 1
                    if not same(got, ref, m, keep_v):
                        fails.append(f"M{m} keep_v={keep_v} eager{i} vs {name}")
                if first is None:
                    first = tuple(t.clone() for t in got)
                elif not same(got, first, m, keep_v):
                    fails.append(f"M{m} keep_v={keep_v} eager{i} vs call 0")
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                call(m, keep_v)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            outs = []
            with torch.cuda.graph(g):
                for _ in range(4):
                    outs.append(call(m, keep_v))
            for r in range(3):
                g.replay()
                torch.cuda.synchronize()
                for j, got in enumerate(outs):
                    for name, ref in (("off", ref_off), ("fwd", ref_fwd)):
                        checks += 1
                        if not same(got, ref, m, keep_v):
                            fails.append(f"M{m} keep_v={keep_v} replay{r} call{j} vs {name}")
            del g, outs
            for i, (t, p) in enumerate(zip(inputs[m], pristine[m])):
                if not torch.equal(t, p):
                    fails.append(f"M{m} input {i} changed")
                    t.copy_(p)
            log(f"M{m} keep_v={keep_v} done, fails so far {len(fails)}")
    print(json.dumps({"passed": not fails, "failures": fails[:20], "bitwise_comparisons": checks,
                      "compared_against": "main's forward v_quant and the "
                                          "TRTLLM_FLUX3_QKV_EPI_FP8=0 chain, same process"}))


if __name__ == "__main__":
    sys.exit(main())
