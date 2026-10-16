#!/usr/bin/env python3
# Exploration: sweep the multi-block merge ring depth of the candidate kernel_mha (in-memory patch of the requested
# nbTileBuffers in cpp/kernels/xqa/mha.cu, written to temp copies of the XQA dir; the committed file is not modified)
# against the original mha.cu at the base commit. Variants are timed round-robin in one process (same inputs, same
# flush, same graph-replay loop as xqa_evidence.py bench) so they share GPU conditions. Shape from xqa_evidence.py
# (default Qwen3.8-27B: BF16 KV, 4 KV heads, 6 Q heads per KV head).
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xqa_evidence as ev  # noqa: E402

DEPTH_TEXT = "MultiBlockMergeSMemT<selectMultiBlockMergeNbTileBuffers<4, multiBlockMergeChunkSize>(), multiBlockMergeChunkSize>"
CHUNK_TEXT = "constexpr uint32_t multiBlockMergeChunkSize = selectMultiBlockMergeChunkSize<64>();"


def make_tree(src_dir, depth=None, chunk=None):
    d = tempfile.mkdtemp(prefix="xqa_var_")
    for f in os.listdir(src_dir):
        p = os.path.join(src_dir, f)
        if os.path.isfile(p):
            shutil.copy(p, d)
    src = open(os.path.join(d, "mha.cu")).read()
    if depth is not None:
        assert src.count(DEPTH_TEXT) == 1
        src = src.replace(DEPTH_TEXT, DEPTH_TEXT.replace("<4,", f"<{depth},"))
    if chunk is not None:
        assert src.count(CHUNK_TEXT) == 1
        src = src.replace(CHUNK_TEXT, CHUNK_TEXT.replace("<64>", f"<{chunk}>"))
    open(os.path.join(d, "mha.cu"), "w").write(src)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="orig,d2,d4,d6,d8,d12,d16")
    p.add_argument("--configs", default="1x48,2x48,4x48,1x47")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--flush-mb", type=int, default=384)
    a = p.parse_args()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    ev.ck(ev.cu.cuInit(0), "cuInit")
    tree = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    base = ev.extract_base_tree(ev.BASE_COMMIT)
    kerns = {}
    for v in a.variants.split(","):
        if v == "orig":
            kerns[v] = ev.Kernel(base)
        else:
            depth, chunk = None, None
            for part in v.split("_"):
                if part.startswith("d"):
                    depth = int(part[1:])
                elif part.startswith("c"):
                    chunk = int(part[1:])
            kerns[v] = ev.Kernel(make_tree(tree, depth, chunk))
        ev.log(f"variant {v}: smem {kerns[v].smem} regs {kerns[v].regs} local {kerns[v].lmem}")
    configs = ev.parse_configs(a.configs)
    inp = ev.build_inputs(max(b for b, _ in configs))
    flush = ev.Flusher(a.flush_mb)
    res = {}
    for (b, m) in configs:
        runners = {v: ev.Runner(k, inp, b, m) for v, k in kerns.items()}
        ts = {v: [] for v in kerns}
        shas = {}
        for r in range(a.rounds):
            for v, run in runners.items():
                n = a.warmup + a.iters
                st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
                en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
                for i in range(n):
                    inp["out"].fill_(float("nan"))
                    flush()
                    st[i].record()
                    run.graph.replay()
                    en[i].record()
                torch.cuda.synchronize()
                ts[v].append(np.array([st[i].elapsed_time(en[i]) for i in range(a.warmup, n)]))
                shas[v] = hashlib.sha256(ev.tbytes(run.out())).hexdigest()
        res[f"{b}x{m}"] = {}
        for v in kerns:
            s = ev.stats(np.concatenate(ts[v]))
            s["round_trimmed_means_us"] = [ev.stats(x)["kernel_ms"] * 1e3 for x in ts[v]]
            s["bitwise_equal_orig"] = shas[v] == shas[next(iter(kerns))]
            res[f"{b}x{m}"][v] = s
            ev.log(f"B={b} m={m} {v:>8}: {s['kernel_ms'] * 1e3:7.2f} us (median {s['median_ms'] * 1e3:7.2f}, rounds "
                   + ", ".join(f"{x:.2f}" for x in s["round_trimmed_means_us"]) + f") bitwise==orig {s['bitwise_equal_orig']}")
    print(json.dumps({"sweep": res}), flush=True)


if __name__ == "__main__":
    main()
