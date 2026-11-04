"""Untimed correctness check for the memoized M > 32 CUTLASS dispatch in
trtllm::flashinfer_mm_mxfp8 (tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py).

Reference: the unmodified original op's call for M > 32, stock
flashinfer.mm_mxfp8(..., backend="cutlass"), on the same act/sf (bitwise).

1. Every projection shape at M = 33..8192 (incl. tile edges), in three tuner
   states: before tuning (fallback tactic), inside autotune() (memo bypassed),
   and after tuning (memo refilled). Every call bitwise vs the stock call; the
   memoized tactic must equal choose_one's tactic for the stock inputs.
2. Repeated sequence matching bench.py (5 warmup + 200 calls per shape at
   M = 128/2048 and 3 + 60 EXTEND steps): every output compared bitwise.
3. Decode CUDA graph at M = 1 and 8 (b12x path) and M = 48/128 (memo path):
   capture, replay 30x, every replay bitwise vs eager stock cutlass.
4. Fallback: a non-contiguous weight view goes to the stock path unchanged.
Last stdout line: {"passed": ...}.
"""
import json

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops import flashinfer_custom_ops as fco
from flashinfer import autotune, mm_mxfp8
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm.gemm_base import (DEFAULT_WORKSPACE_SIZE,
                                       _MM_MXFP8_TUNING_CONFIG,
                                       get_cutlass_mxfp8_gemm_module)
from flashinfer.utils import _get_cache_buf

dev = torch.device("cuda")
BF16 = torch.bfloat16
op = torch.ops.trtllm.flashinfer_mm_mxfp8
tuner = AutoTuner.get()
assert hasattr(fco, "_cutlass_memo_mm"), "candidate tree expected"

SHAPES = {
    "wq_a": (1280, 5120),
    "wkv": (512, 5120),
    "wq_a_wkv": (1792, 5120),
    "wq_b": (8192, 1280),
    "index_wq_b": (4096, 1280),
    "wq_b_index": (12288, 1280),
    "wo_b": (5120, 2048),
}
MS = [33, 34, 47, 63, 64, 65, 96, 127, 128, 129, 200, 255, 256, 257, 511, 512,
      1000, 1024, 2047, 2048, 2049, 4096, 6144, 6145, 8192]

torch.manual_seed(0)
W = {}
for name, (n, k) in SHAPES.items():
    w = torch.randn(n, k, device=dev, dtype=BF16) * 0.05
    W[name] = torch.ops.trtllm.mxfp8_quantize(w, True)
ACT = {}


def act(m, k):
    if (m, k) not in ACT:
        g = torch.Generator(device=dev).manual_seed(1000 + m * 7 + k)
        x = torch.randn(m, k, device=dev, dtype=BF16, generator=g)
        ACT[(m, k)] = torch.ops.trtllm.mxfp8_quantize(x, True)
    return ACT[(m, k)]


def stock(aq, asf, wq, wsf):
    return mm_mxfp8(aq, wq.t(), asf, wsf, out_dtype=BF16,
                    use_8x4_sf_layout=False, backend="cutlass")


def stock_tactic(aq, asf, wq, wsf):
    out = torch.empty((aq.shape[0], wq.shape[0]), dtype=BF16, device=dev)
    ws = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, dev)
    runner = get_cutlass_mxfp8_gemm_module(12).cutlass_mxfp8_gemm_runner()
    return tuner.choose_one("mxfp8_gemm", [runner], _MM_MXFP8_TUNING_CONFIG,
                            [aq, wq.t(), asf, wsf, BF16, out, ws])[1]


def memo_tactic(aq, asf, wq, wsf):
    for key, entry in fco._cutlass_memo.items():
        if key[1] == aq.shape[0] and key[3] == wq.shape[0] and key[2] == aq.shape[1]:
            return entry[1]
    return "MISSING"


fails = []
stats = {"calls": 0, "bitwise_eq": 0, "tactic_checks": 0}


def cmp(tag, got, ref):
    stats["calls"] += 1
    if got.shape != ref.shape or got.dtype != ref.dtype or got.stride() != ref.stride():
        fails.append(f"{tag}: meta {got.shape} {got.stride()} vs {ref.shape} {ref.stride()}")
        return
    if not torch.isfinite(got).all():
        fails.append(f"{tag}: non-finite")
    if torch.equal(got.view(torch.int16), ref.view(torch.int16)):
        stats["bitwise_eq"] += 1
    else:
        bad = (got.view(torch.int16) != ref.view(torch.int16)).sum().item()
        fails.append(f"{tag}: {bad} elements differ")


def sweep(state, check_tactic):
    for m in MS:
        for name, (n, k) in SHAPES.items():
            aq, asf = act(m, k)
            wq, wsf = W[name]
            ref = stock(aq, asf, wq, wsf)
            for it in range(3):  # 1st call fills the memo, then hits
                cmp(f"{state} {name} M={m} it{it}", op(aq, asf, wq, wsf, BF16), ref)
            if check_tactic:
                st, mt = stock_tactic(aq, asf, wq, wsf), memo_tactic(aq, asf, wq, wsf)
                stats["tactic_checks"] += 1
                if str(st) != str(mt):
                    fails.append(f"{state} {name} M={m}: memo tactic {mt} != stock {st}")
    torch.cuda.synchronize()
    print(f"[{state}] done calls={stats['calls']} eq={stats['bitwise_eq']} "
          f"memo_entries={len(fco._cutlass_memo)} fails={len(fails)}", flush=True)


# 1a. Before any tuning: fallback tactic memoized.
sweep("pre-tune", True)
pre_entries = len(fco._cutlass_memo)
# 1b. Inside autotune(): memo must be bypassed (stock path tunes).
with autotune():
    sweep("in-tune", False)
in_tune_entries = len(fco._cutlass_memo)
# 1c. After tuning: the signature changed, memo refilled with tuned tactics.
sweep("post-tune", True)
tactics = {}
for m in (128, 2048):
    for name, (n, k) in SHAPES.items():
        aq, asf = act(m, k)
        tactics[f"{name}@M{m}"] = str(stock_tactic(aq, asf, *W[name]))
print("tactics after tune:", tactics, flush=True)

# 2. Repeated sequence matching bench.py.
for m in (128, 2048):
    for name, (n, k) in SHAPES.items():
        aq, asf = act(m, k)
        wq, wsf = W[name]
        ref = stock(aq, asf, wq, wsf)
        for it in range(205):
            cmp(f"seq {name} M={m} it{it}", op(aq, asf, wq, wsf, BF16), ref)
for b in (1, 2, 4):
    seq = []
    for layer in range(40):
        m = 2048 * b if layer < 20 else 128 * b
        for name in ["wq_a_wkv", "wq_b", "index_wq_b", "wo_b"]:
            n, k = SHAPES[name]
            aq, asf = act(m, k)
            seq.append((name, aq, asf, *W[name]))
    refs = [stock(aq, asf, wq, wsf) for _, aq, asf, wq, wsf in seq]
    for step in range(63):
        outs = [op(aq, asf, wq, wsf, BF16) for _, aq, asf, wq, wsf in seq]
        for i, (o, r) in enumerate(zip(outs, refs)):
            cmp(f"step b{b} s{step} c{i}", o, r)
torch.cuda.synchronize()
print(f"[repeated] calls={stats['calls']} eq={stats['bitwise_eq']} fails={len(fails)}",
      flush=True)

# 3. CUDA graphs: decode M=1/8 (b12x) and M=48/128 (memo path).
graph_report = {}
for m in (1, 8, 48, 128):
    items = []
    for name, (n, k) in SHAPES.items():
        aq, asf = act(m, k)
        items.append((name, aq, asf, *W[name]))
        op(aq, asf, *W[name], BF16)  # eager warm (b12x runner build / memo fill)
    refs = [stock(aq, asf, wq, wsf) for _, aq, asf, wq, wsf in items]
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _, aq, asf, wq, wsf in items:
            op(aq, asf, wq, wsf, BF16)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [op(aq, asf, wq, wsf, BF16) for _, aq, asf, wq, wsf in items]
    bad = 0
    for rep in range(30):
        for o in outs:
            o.fill_(0)
        g.replay()
        torch.cuda.synchronize()
        for (name, *_), o, r in zip(items, outs, refs):
            before = len(fails)
            cmp(f"graph M={m} {name} rep{rep}", o, r)
            bad += len(fails) - before
    graph_report[str(m)] = {"replays": 30, "shapes": len(items), "bad": bad}
    del g
print("graphs:", graph_report, flush=True)

# 4. Fallbacks to the stock path.
n, k = SHAPES["wq_b"]
aq, asf = act(256, k)
wq, wsf = W["wq_b"]
wq_nc = torch.empty((n, k * 2), dtype=wq.dtype, device=dev)[:, :k]
wq_nc.copy_(wq)
try:
    got = op(aq, asf, wq_nc, wsf, BF16)
    cmp("fallback non-contiguous weight", got, stock(aq, asf, wq, wsf))
    fallback = "ran"
except Exception as e:  # stock raises the same way
    try:
        stock(aq, asf, wq_nc, wsf)
        fails.append(f"non-contiguous: op raised {e!r} but stock did not")
        fallback = "op-raised"
    except Exception as e2:
        fallback = f"both raise ({type(e2).__name__})"
print("fallback non-contiguous:", fallback, flush=True)

passed = not fails and stats["bitwise_eq"] == stats["calls"]
print(json.dumps({
    "passed": passed,
    "compared": "trtllm::flashinfer_mm_mxfp8 (memoized M>32 dispatch) vs stock flashinfer.mm_mxfp8(backend='cutlass') bitwise; memo tactic vs choose_one tactic",
    "calls": stats["calls"], "bitwise_equal": stats["bitwise_eq"],
    "tactic_checks": stats["tactic_checks"],
    "memo_entries": {"pre_tune": pre_entries, "after_in_tune": in_tune_entries,
                     "post_tune": len(fco._cutlass_memo)},
    "graphs": graph_report, "fallback_non_contiguous": fallback,
    "tactics_after_tune": tactics,
    "fails": fails[:20],
}))
