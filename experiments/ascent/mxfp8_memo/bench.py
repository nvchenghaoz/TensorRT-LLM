"""Host-time bench for trtllm::flashinfer_mm_mxfp8 at prefill sizes (M > 32).

Runs identically in the baseline (main) and candidate (memoized dispatch) trees.

Part 1 (stage-1 table): per projection shape at M in {128, 2048}, host us/call
(median of N calls timed with perf_counter_ns; torch.cuda._sleep queued first,
no sync inside the loop) for the arms
  A  torch.ops.trtllm.flashinfer_mm_mxfp8 (this tree's op)
  B  the op's Python impl called directly (no torch.library dispatch)
  B0 stock flashinfer.mm_mxfp8(backend="cutlass")
  C  stock mm_mxfp8(..., skip_check=True)
  D  memoized direct TVM-FFI module.mxfp8_gemm (tactic + workspace cached)
  CH AutoTuner.choose_one alone (with a fresh CUTLASS runner, as stock builds)
and the tactic choose_one picks after the serving-like autotune() warmup.

Part 2 (gain evidence): host ms to enqueue one EXTEND step's M > 32 MXFP8 call
sequence: 40 layers x (wq_a||wkv, wq_b, index_wq_b, wo_b); layers 0-19 at
M_hi rows, 20-39 at M_lo rows. Batch label b: M_hi = 2048*b, M_lo = 128*b.
Last stdout line: {"batch": {"<b>": {"step_ms", "noise_floor"}}, "output": ...}.
"""
import gc
import json
import os
import statistics
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops import flashinfer_custom_ops as fco
from flashinfer import autotune, mm_mxfp8
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm.gemm_base import (DEFAULT_WORKSPACE_SIZE,
                                       _MM_MXFP8_TUNING_CONFIG,
                                       _load_gemm_sm120_mxfp8_module,
                                       get_cutlass_mxfp8_gemm_module)
from flashinfer.utils import _get_cache_buf

dev = torch.device("cuda")
BF16 = torch.bfloat16
op = torch.ops.trtllm.flashinfer_mm_mxfp8
raw_impl = getattr(fco.flashinfer_mm_mxfp8, "_init_fn", None)
HAS_MEMO = hasattr(fco, "_cutlass_memo_mm")
CALLS = int(os.environ.get("CALLS", "200"))
STEPS = int(os.environ.get("STEPS", "60"))
BATCHES = [int(b) for b in os.environ.get("BATCHES", "1,2,4").split(",")]

SHAPES = {
    "wq_a": (1280, 5120),
    "wkv": (512, 5120),
    "wq_a_wkv": (1792, 5120),
    "wq_b": (8192, 1280),
    "index_wq_b": (4096, 1280),
    "wq_b_index": (12288, 1280),
    "wo_b": (5120, 2048),
}
STEP_SHAPES = ["wq_a_wkv", "wq_b", "index_wq_b", "wo_b"]

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


module = _load_gemm_sm120_mxfp8_module()
tuner = AutoTuner.get()


def choose(aq, asf, wq, wsf):
    out = torch.empty((aq.shape[0], wq.shape[0]), dtype=BF16, device=dev)
    ws = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, dev)
    runner = get_cutlass_mxfp8_gemm_module(12).cutlass_mxfp8_gemm_runner()
    inputs = [aq, wq.t(), asf, wsf, BF16, out, ws]
    return tuner.choose_one("mxfp8_gemm", [runner], _MM_MXFP8_TUNING_CONFIG,
                            inputs)


def arms(aq, asf, wq, wsf):
    _, tactic = choose(aq, asf, wq, wsf)
    m, n = aq.shape[0], wq.shape[0]

    def a():
        return op(aq, asf, wq, wsf, BF16)

    def b():
        return raw_impl(aq, asf, wq, wsf, BF16)

    def b0():
        return mm_mxfp8(aq, wq.t(), asf, wsf, out_dtype=BF16,
                        use_8x4_sf_layout=False, backend="cutlass")

    def c():
        return mm_mxfp8(aq, wq.t(), asf, wsf, out_dtype=BF16,
                        use_8x4_sf_layout=False, backend="cutlass",
                        skip_check=True)

    def d():
        out = torch.empty((m, n), dtype=BF16, device=dev)
        ws = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, dev)
        module.mxfp8_gemm(aq, wq, asf, wsf, out, ws, tactic)
        return out

    def ch():
        return choose(aq, asf, wq, wsf)

    res = {"A": a, "B0": b0, "C": c, "D": d, "CH": ch}
    if raw_impl is not None:
        res["B"] = b
    return res, tactic


def host_us(fn, n=CALLS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda._sleep(int(2.4e9 * 0.05))  # ~50 ms of GPU work queued first
    ts = []
    pc = time.perf_counter_ns
    for _ in range(n):
        t0 = pc()
        fn()
        ts.append(pc() - t0)
    torch.cuda.synchronize()
    ts.sort()
    return ts[len(ts) // 2] / 1e3, ts[int(0.9 * len(ts))] / 1e3


# Serving-like warmup: one autotune() pass per (shape, M) used anywhere below,
# then eager calls (fills the memo on the candidate tree).
ms_all = sorted({128, 2048} | {2048 * b for b in BATCHES} | {128 * b for b in BATCHES})
print(f"tree has_memo={HAS_MEMO} raw_impl={raw_impl is not None} ms={ms_all}",
      flush=True)
with autotune():
    for m in ms_all:
        for name, (n, k) in SHAPES.items():
            aq, asf = act(m, k)
            op(aq, asf, *W[name], BF16)
torch.cuda.synchronize()
print("autotune warmup done", flush=True)

table = {}
for m in (128, 2048):
    for name, (n, k) in SHAPES.items():
        aq, asf = act(m, k)
        fns, tactic = arms(aq, asf, *W[name])
        row = {"tactic": str(tactic)}
        for arm, fn in fns.items():
            med, p90 = host_us(fn)
            row[arm] = round(med, 2)
            row[arm + "_p90"] = round(p90, 2)
        table[f"{name}@M{m}"] = row
        print(f"M={m:5d} {name:11s} [{n:5d},{k:5d}] tactic={tactic} " +
              " ".join(f"{a}={row[a]:7.2f}" for a in fns), flush=True)

# Part 2: one EXTEND step's M > 32 call sequence.
batch = {}
for b in BATCHES:
    m_hi, m_lo = 2048 * b, 128 * b
    seq = []
    for layer in range(40):
        m = m_hi if layer < 20 else m_lo
        for name in STEP_SHAPES:
            n, k = SHAPES[name]
            aq, asf = act(m, k)
            wq, wsf = W[name]
            seq.append((aq, asf, wq, wsf))

    def step():
        for aq, asf, wq, wsf in seq:
            op(aq, asf, wq, wsf, BF16)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    ts = []
    pc = time.perf_counter_ns
    for _ in range(STEPS):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda._sleep(int(2.4e9 * 0.1))  # GPU stays busy; no backpressure
        t0 = pc()
        step()
        ts.append((pc() - t0) / 1e6)
        torch.cuda.synchronize()
    ts.sort()
    med = statistics.median(ts)
    p10 = ts[int(0.1 * len(ts))]
    p90 = ts[min(len(ts) - 1, int(0.9 * len(ts)))]
    batch[str(b)] = {"step_ms": round(med, 4), "noise_floor": round((p90 - p10) / med, 4),
                     "p10_ms": round(p10, 4), "p90_ms": round(p90, 4),
                     "calls_per_step": len(seq), "m_hi": m_hi, "m_lo": m_lo}
    print(f"batch {b}: host ms/step median {med:.3f} p10 {p10:.3f} p90 {p90:.3f} "
          f"({len(seq)} calls, {1e3*med/len(seq):.1f} us/call)", flush=True)

print(json.dumps({"batch": batch, "output": {"class": "exact"},
                  "has_memo": HAS_MEMO, "table": table}))
