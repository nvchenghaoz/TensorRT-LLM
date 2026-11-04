"""Idea 143 stage B upper-bound probe (timing only, NOT the model's math).

Split-K by S of the FI decode GEMMs moves the same weight bytes through S x more CTAs, each
streaming K/S.  Emulate it with the unmodified kernels on reshaped synthetic problems with the
same bytes per expert: fc1 (n = I*S, k = H/S) and fc2 (n = H*S, k = I/S), D experts with work,
M = 4 tokens.  No partial write / reduction cost is included (that is the upper bound).
Cold L2 (512 MB read-only flush) and warm L2, CUDA-event mean per replay.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))

import torch  # noqa: E402

from common import H, I, log  # noqa: E402

FLUSH = None
EX = 16  # experts in the synthetic layer


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_g(g, do_flush, iters=50, warmup=8):
    ts = []
    for i in range(warmup + iters):
        if do_flush:
            FLUSH.sum()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if i >= warmup:
            ts.append(a.elapsed_time(b) * 1000.0)
    return round(sum(ts) / len(ts), 2)


def weights(rows, k, gen):
    w = torch.randint(0, 256, (EX, rows, k // 2), dtype=torch.uint8, device="cuda", generator=gen)
    raw = torch.randint(119, 123, (EX, rows, k // 32), dtype=torch.uint8, device="cuda", generator=gen)
    sf = raw.contiguous().view(torch.int32).transpose(1, 2).contiguous().view(torch.uint8)
    return w, sf


def main():
    global FLUSH
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1 import (
        cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4 as fc1, out_sf_shape)
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize import (
        cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4 as fc2)
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_q0_route_triton import (
        make_mxfp8_q0_route_workspace, mxfp8_q0_route_triton as route)
    from flashinfer.tllm_enums import ActivationType as FiAct

    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    m = 4
    rows = []
    for which, splits in (("fc1", (1, 2, 5, 10)), ("fc2", (1, 2, 3, 6))):
        for S in splits:
            if which == "fc1":
                n1, k1 = I * S, H // S
            else:
                n1, k1 = I // S, H  # fc1 only produces fc2's input
            w1, w1sf = weights(2 * n1, k1, gen)
            if which == "fc2":
                n2, k2 = H * S, I // S
                w2, w2sf = weights(n2, k2, gen)
            for d in (1, 2, 3, 4):
                x = torch.randn(m, k1, dtype=torch.bfloat16, device="cuda")
                # token t -> local experts; d distinct experts over 4 tokens, rest -1
                ids = torch.full((m, 6), -1, dtype=torch.int32, device="cuda")
                for j in range(max(d, m)):
                    ids[j % m, j // m] = j % d
                wts = torch.full((m, 6), 0.25, dtype=torch.float32, device="cuda")
                ws = make_mxfp8_q0_route_workspace(x, ids, EX)
                offsets, token_map, rw, a_q, a_sf = route(x, ids, wts, EX, workspace=ws,
                                                          enable_pdl=True)
                pairs = ids.numel()
                q1 = torch.zeros(pairs, n1, dtype=torch.float8_e4m3fn, device="cuda")
                sf1 = torch.zeros(out_sf_shape(pairs, n1, EX), dtype=torch.int32, device="cuda")
                f1 = lambda: fc1(a_q, a_sf, w1, w1sf, offsets, tune=False, enable_pdl=True,
                                 out_q=q1, out_sf=sf1, activation=FiAct.Swiglu, swiglu_limit=10.0)
                if which == "fc1":
                    g = capture(f1)
                else:
                    f1()
                    out = torch.zeros(m, n2, dtype=torch.bfloat16, device="cuda")
                    g = capture(lambda: fc2(q1, sf1, w2, w2sf, offsets, token_map, rw, m,
                                            tune=False, enable_pdl=True, out=out))
                row = {"kernel": which, "S": S, "D": d, "cold_us": time_g(g, True),
                       "warm_us": time_g(g, False)}
                rows.append(row)
                log(json.dumps(row))
                del g
            del w1, w1sf
            if which == "fc2":
                del w2, w2sf
    print(json.dumps({"rows": rows}))


if __name__ == "__main__":
    main()
