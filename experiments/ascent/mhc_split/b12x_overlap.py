"""b12x MXFP8 GEMM overlap check with the mHC split's K2 as the aggressor (pitfall 1353).

One CUDA graph per (arm, M): the branch's real fork (mHC.fused_hc_lagged_split: K1 on main,
K2 = mhc_fused_hc with the tuner's tactic on the aux stream), then on main the b12x
trtllm::flashinfer_mm_mxfp8 at [N,K] = [1792,5120], [8192,1280], [12288,1280], then the join.
CONTROL arm: the same fork/join with 4x mul_ on a 256 MB tensor as the aux work.
Every replay's b12x outputs are compared bitwise with mm_mxfp8(backend='cutlass') on the
same act/sf. Prints wrong-element share per replay, arm and shape.
"""

import json
import sys

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401  (registers trtllm ops)
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel_deferred_join,
    with_multi_stream,
)

SHAPES = [(1792, 5120), (8192, 1280), (12288, 1280)]


def log(*a):
    print(*a, flush=True)


def main():
    from flashinfer import mm_mxfp8

    replays = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    dev = torch.device("cuda")
    torch.manual_seed(0)
    hidden = 5120
    mhc = mHC(4, hidden, 20, dtype=torch.float32, eps=1e-6, norm_eps=1e-6, sinkhorn_eps=1e-6,
              post_mult_value=2.0).to(dev)
    with torch.no_grad():
        mhc.fn.normal_(0, 1e-2)
        mhc.base.normal_(0, 0.1)
        mhc.scale.copy_(torch.tensor([0.5, 0.7, 0.9]))
    nw = (1 + 0.1 * torch.randn(hidden, device=dev)).to(torch.bfloat16)
    aux = torch.cuda.Stream()
    ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()
    big = torch.randn(64 * 1024 * 1024, device=dev)  # 256 MB fp32

    weights = []
    for n, k in SHAPES:
        w = (torch.randn(n, k, device=dev) / k ** 0.5).to(torch.bfloat16)
        w_q, w_sf = torch.ops.trtllm.mxfp8_quantize(w, True)
        weights.append((w_q, w_sf))

    summary = {}
    all_k2_clean = True
    control_faults = 0
    for M in (1, 4, 8):
        x = torch.randn(M, hidden, device=dev).to(torch.bfloat16)
        res = torch.randn(M, 4, hidden, device=dev).to(torch.bfloat16)
        post = (2 * torch.rand(M, 4, 1, device=dev)).float()
        comb = torch.rand(M, 4, 4, device=dev)
        comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
        pre = (torch.rand(M, 4, 1, device=dev) + 1e-3).float()
        with autotune(skip_dynamic_tuning_buckets=True):
            mhc.fused_hc_lagged(x, res, post, comb, pre, norm_weight=nw, norm_eps=1e-6)
        acts, refs = [], []
        for (n, k), (w_q, w_sf) in zip(SHAPES, weights):
            a = torch.randn(M, k, device=dev).to(torch.bfloat16)
            a_q, a_sf = torch.ops.trtllm.mxfp8_quantize(a, True)
            acts.append((a_q, a_sf))
            ref = mm_mxfp8(a_q, w_q.t(), a_sf, w_sf, out_dtype=torch.bfloat16,
                           use_8x4_sf_layout=False, backend="cutlass")
            iso = torch.ops.trtllm.flashinfer_mm_mxfp8(a_q, a_sf, w_q, w_sf, torch.bfloat16)
            torch.cuda.synchronize()
            iso_bad = int((iso.view(torch.int16) != ref.view(torch.int16)).sum().item())
            log(f"M={M} shape={n}x{k} isolated b12x vs cutlass bad={iso_bad}")
            refs.append(ref)

        def gemms():
            outs = []
            for (a_q, a_sf), (w_q, w_sf) in zip(acts, weights):
                outs.append(torch.ops.trtllm.flashinfer_mm_mxfp8(a_q, a_sf, w_q, w_sf, torch.bfloat16))
            return outs

        def body_k2():
            split = mhc.fused_hc_lagged_split(x, res, post, comb, pre, nw, 1e-6, aux, ev0, ev1)
            assert split is not None, "split path did not engage"
            outs_mhc, join = split
            outs = gemms()
            join.join()
            return outs

        def body_control():
            def fn1():
                for _ in range(4):
                    big.mul_(1.0000001)
                return None

            _, _, ev = maybe_execute_in_parallel_deferred_join(lambda: None, fn1, ev0, ev1, aux)
            outs = gemms()
            if ev is not None:
                ev.wait()
            return outs

        for arm, body in (("K2", body_k2), ("CONTROL", body_control)):
            with with_multi_stream(True):
                body()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    outs = body()
            torch.cuda.synchronize()
            per = {f"{n}x{k}": [] for n, k in SHAPES}
            for _ in range(replays):
                for o in outs:
                    o.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                for (n, k), o, ref in zip(SHAPES, outs, refs):
                    bad = int((o.view(torch.int16) != ref.view(torch.int16)).sum().item())
                    per[f"{n}x{k}"].append(round(bad / o.numel(), 5))
                    if bad and arm == "K2":
                        all_k2_clean = False
                    if bad and arm == "CONTROL":
                        control_faults += 1
            summary[f"M{M}/{arm}"] = per
            log(f"M={M} arm={arm} wrong-share per replay: {json.dumps(per)}")
            del g
    print(json.dumps({"passed": bool(all_k2_clean), "k2_arm_bitwise_all_replays": all_k2_clean,
                      "control_faulty_replays": control_faults, "replays": replays,
                      "summary": summary}))


if __name__ == "__main__":
    main()
