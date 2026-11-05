"""Idea 254 paired microbenchmark: decode shared expert of one TP4 rank, alone.

Runs the REAL DeepseekV4MoE.forward of whichever tree runs it (base: main's GatedMLP
chain on the full 1152-col half x 0.5; candidate: the decode block split, rank 0 takes
blocks [0,5) of its half) with the routed path stubbed (a preallocated buffer) and the
all-reduce as identity (experiments/ascent/shared_row_split/common.py build_moe).
40 layers with distinct weight copies and inputs in one CUDA graph, a read-only 512 MB
flush before every replay (cold L2), CUDA-event time per replay.
Env: RANK (0: the heavier part; 2: the lighter), MS (default 1,2,4,8), NB (blocks, 5),
REPS (replays per block, 20).
Last line: {"batch": {"<M>": {"kernel_ms": ms per 40-layer step, "noise_floor": ...}}, ...}
"""

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared_row_split"))
if len(sys.argv) > 1 and not os.environ.get("CKPT"):
    os.environ["CKPT"] = sys.argv[1]

import torch  # noqa: E402

import common as srs  # noqa: E402

NL = 40
RANK = int(os.environ.get("RANK", "0"))
MS = [int(v) for v in os.environ.get("MS", "1,2,4,8").split(",")]
NB = int(os.environ.get("NB", "5"))
REPS = int(os.environ.get("REPS", "20"))
TAG = os.environ.get("TAG", "ARM")


def log(*a):
    print(*a, flush=True)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main():
    import tensorrt_llm  # noqa: F401

    mlp, shared_tp, scale, info = srs.build_shared_mlp(rank=RANK)
    log(TAG, "shared", info)
    mlps = [mlp] + [copy.deepcopy(mlp) for _ in range(NL - 1)]
    moes = []
    for m_ in mlps:
        moe, stub, _ = srs.build_moe(RANK, m_, shared_tp, scale)
        if hasattr(moe, "_setup_shared_decode_split"):
            moe._setup_shared_decode_split()
        moes.append((moe, stub))
    split = hasattr(moes[0][0], "_setup_shared_decode_split")
    log(TAG, f"tree has decode split: {split}; state {getattr(moes[0][0], '_shared_decode_split', 'n/a')}")
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    hidden = mlp.gate_up_proj.weight.shape[1]
    res = {}
    names = set()
    for m in MS:
        g = torch.Generator(device="cpu").manual_seed(100 + m)
        xs = [(torch.randn(m, hidden, generator=g) * 0.5).to(torch.bfloat16).cuda()
              for _ in range(NL)]
        bufs = [torch.zeros(m, hidden, dtype=torch.bfloat16, device="cuda") for _ in range(NL)]

        def step():
            outs = []
            for (moe, stub), x, b in zip(moes, xs, bufs):
                stub.buf = b.clone()
                outs.append(moe.forward(x, all_rank_num_tokens=[m] * 4))
            return outs

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                ref = step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        ref = [o.clone() for o in ref]
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = step()
        torch.cuda.synchronize()
        for _ in range(30):
            flush.sum()
            gr.replay()
        torch.cuda.synchronize()
        same = all(torch.equal(o, r) for o, r in zip(outs, ref))
        blocks, allv = [], []
        for _b in range(NB):
            evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                   for _ in range(REPS)]
            for a, b in evs:
                flush.sum()
                a.record()
                gr.replay()
                b.record()
            torch.cuda.synchronize()
            v = [a.elapsed_time(b) for a, b in evs]
            allv += v
            blocks.append(pct(v, 0.5))
        same2 = all(torch.equal(o, r) for o, r in zip(outs, ref))
        med = pct(blocks, 0.5)
        noise = (pct(allv, 0.9) - pct(allv, 0.1)) / med
        disp = (max(blocks) - min(blocks)) / med
        res[str(m)] = {"kernel_ms": round(med, 5), "noise_floor": round(noise, 4),
                       "block_dispersion": round(disp, 4), "us_per_layer": round(med * 1000 / NL, 3),
                       "replays_bitwise_vs_eager": bool(same and same2)}
        log(TAG, f"M={m}: {med * 1000 / NL:.2f} us/layer ({med:.4f} ms/step) noise {noise:.4f} "
            f"disp {disp:.4f} blocks {[round(x * 1000 / NL, 2) for x in blocks]} bitwise {same and same2}")
        if m == MS[0]:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                flush.sum()
                gr.replay()
                torch.cuda.synchronize()
            for e in prof.key_averages():
                if e.device_type is not None and "reduce_kernel" not in e.key:
                    names.add(e.key[:90])
            log(TAG, "kernels:", sorted(names))
        del gr, outs, ref
        torch.cuda.empty_cache()
    print(json.dumps({"batch": res, "output": {"class": "reordered"}, "tag": TAG, "rank": RANK,
                      "decode_split_in_tree": split}))


if __name__ == "__main__":
    main()
