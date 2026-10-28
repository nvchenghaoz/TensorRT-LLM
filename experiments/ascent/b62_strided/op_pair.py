"""Branch 62 (idea 112): FP8R q/k/v layout into fp8_sm12x attention, strided vs planar.

Runs the tree's own flux3_fp8r_qkv._sm120_fp8r_qkv_forward (stub attn/adapter around the real
trtllm::flux3_sm120_fp8r_qkv_qknorm_rope op) followed by the attention reshape that
Attention._attn_impl does and trtllm::visual_gen_fp8_attention_sm120, i.e. FP8R op + attention
preprocessing (K1 amax, K2 quantize) + fmha_v2.

Variants (module switches read at call time, so all run in one process):
  planar    : _FP8R_QKV_PLANAR=True                     (main 700ddd59 / landing 34)
  strided   : _FP8R_QKV_PLANAR=False, _CONTIGUOUS=False  (packed [M,9216] column views)
  planar_aa : same as planar, its own graph and buffers (A/A row)
Shapes: M=4176 (4 single blocks 24-27) and M=4096 (content_mode_blocks.image.0), per denoise
step 4*chain(4176) + 1*chain(4096) = one "step" sample.

Timing: one sample = CUDA-event-timed replay of a CUDA graph of K=20 back-to-back chain calls,
reported per call. All 6 graphs interleaved per round in rotated order, after a >=10 s
warm-soak; SM clock logged per block of rounds. All samples kept (no trimming).
Correctness (untimed, after timing, same graphs, same replay counts): every replay's outputs vs
the eager reference of the same variant and vs eager planar (bitwise), vs replay 0, and inputs
vs pristine copies. Plus an in-process A(4176)/B(4256)/A sequence, planar vs strided bitwise at
every position (ruling 500).

Last stdout line: {"batch": {"1": {"kernel_ms", "noise_floor"}}, "output": {"class": "exact"}, ...}
with kernel_ms the per-step (4*M4176 + M4096) time of the variant the TREE defaults to.
"""
import json
import statistics
import sys
import time
import types

import torch

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import (  # noqa: E402
    visual_gen_fp8_attention_sm120,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv as fq  # noqa: E402

DEV = "cuda"
H, D = 24, 128
W = H * D
KDIM = 3072
SCALE = 1.0 / D ** 0.5
K_CALLS = 20
ROUNDS = 60
TREE_DEFAULT = ("planar" if getattr(fq, "_FP8R_QKV_PLANAR", False)
                else ("contiguous" if fq._FP8R_QKV_CONTIGUOUS else "strided"))
FLAGS = {"planar": (True, True), "planar_aa": (True, True), "strided": (False, False),
         "contiguous": (False, True)}

try:
    import pynvml

    pynvml.nvmlInit()
    _NV = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    def sm_clock():
        return pynvml.nvmlDeviceGetClockInfo(_NV, pynvml.NVML_CLOCK_SM)
except Exception:  # noqa: BLE001
    def sm_clock():
        try:
            return torch.cuda.clock_rate()
        except Exception:  # noqa: BLE001
            return -1


gw = torch.Generator(device=DEV).manual_seed(1234)
WEIGHT = (torch.randn(3 * W, KDIM, device=DEV, generator=gw) * 2).to(torch.float8_e4m3fn)
WSCALE = (torch.rand(3 * W, device=DEV, generator=gw) * 0.02 + 0.01).float()
QN = (torch.rand(D, device=DEV, generator=gw) + 0.5).to(torch.bfloat16)
KN = (torch.rand(D, device=DEV, generator=gw) + 0.5).to(torch.bfloat16)


class _Adapter:
    weight, scale = WEIGHT, WSCALE

    def __init__(self, data, scales):
        self._data, self._scales = data, scales

    def can_run(self, original):
        return True

    def activation(self, x):
        return self._data, self._scales


def make_case(m, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    act = (torch.randn(m, KDIM, device=DEV, generator=g) * 2).to(torch.float8_e4m3fn)
    act_s = (torch.rand(m, device=DEV, generator=g) * 0.02 + 0.01).float()
    pos = torch.arange(m, device=DEV, dtype=torch.float32)[:, None]
    freq = torch.arange(64, device=DEV, dtype=torch.float32)[None, :]
    ang = pos * (10000.0 ** (-freq / 64))
    cos_sin = torch.cat((ang.cos(), ang.sin()), -1).contiguous()
    cos = torch.repeat_interleave(ang.cos(), 2, -1).reshape(1, m, 128)
    sin = torch.repeat_interleave(ang.sin(), 2, -1).reshape(1, m, 128)
    original = torch.empty(1, m, KDIM, device=DEV, dtype=torch.bfloat16)
    attn = types.SimpleNamespace(
        _flux3_sm120_fp8_qkv=_Adapter(act, act_s),
        norm_q=types.SimpleNamespace(weight=QN), norm_k=types.SimpleNamespace(weight=KN))
    return {"attn": attn, "original": original, "freqs": (cos, sin, cos_sin),
            "pristine": [t.clone() for t in (act, act_s, cos_sin)], "live": [act, act_s, cos_sin]}


def chain(case, variant):
    fq._FP8R_QKV_PLANAR, fq._FP8R_QKV_CONTIGUOUS = FLAGS[variant]
    q, k, v = fq._sm120_fp8r_qkv_forward(case["attn"], case["original"], case["original"],
                                         case["freqs"])
    # Attention._attn_impl (NHD): [B, S, H*D] -> [B, S, H, D] views.
    b = q.shape[0]
    q, k, v = (t.view(b, -1, H, D) for t in (q, k, v))
    return visual_gen_fp8_attention_sm120(q, k, v, SCALE, False)


def capture(case, variant, k_calls):
    for _ in range(2):
        chain(case, variant)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [chain(case, variant) for _ in range(k_calls)]
    return g, outs


def pctl(xs, p):
    s = sorted(xs)
    i = (len(s) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def inputs_unchanged(case):
    return all(torch.equal(a, b) for a, b in zip(case["live"], case["pristine"]))


def aba_check():
    """In-process A(4176)/B(4256)/A: planar vs strided bitwise at each position (ruling 500)."""
    cases = {"A": make_case(4176, 11), "B": make_case(4256, 12)}
    res = []
    for pos, key in enumerate(["A", "B", "A"]):
        outs = {v: chain(cases[key], v).clone() for v in ("planar", "strided", "contiguous")}
        res.append({"pos": pos, "len": key,
                    "strided_eq_planar": torch.equal(outs["strided"], outs["planar"]),
                    "contiguous_eq_planar": torch.equal(outs["contiguous"], outs["planar"])})
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "time"
    log("tree default variant:", TREE_DEFAULT, "torch", torch.__version__,
        torch.cuda.get_device_name(), "sm_clock", sm_clock())
    variants = ["planar", "strided", "planar_aa"]
    ms = [4176, 4096]
    cases = {m: make_case(m, 7 + i) for i, m in enumerate(ms)}
    refs = {}
    for m in ms:
        for v in ("planar", "strided"):
            refs[(m, v)] = chain(cases[m], v).clone()
        log(f"M={m} eager strided == planar:", torch.equal(refs[(m, "strided")], refs[(m, "planar")]))
    graphs = {}
    for m in ms:
        for v in variants:
            graphs[(m, v)] = capture(cases[m], v, K_CALLS)
    torch.cuda.synchronize()
    keys = list(graphs)
    log("captured", len(keys), "graphs; warm-soak")
    t_end = time.time() + 12.0
    n_soak = 0
    while time.time() < t_end:
        for key in keys:
            graphs[key][0].replay()
        torch.cuda.synchronize()
        n_soak += 1
    log(f"warm-soak {n_soak} rounds, sm_clock {sm_clock()}")
    samples = {key: [] for key in keys}
    clocks = []
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in keys]
    for r in range(ROUNDS):
        order = keys[r % len(keys):] + keys[:r % len(keys)]
        for i, key in enumerate(order):
            s, e = ev[i]
            s.record()
            graphs[key][0].replay()
            e.record()
        torch.cuda.synchronize()
        for i, key in enumerate(order):
            samples[key].append(ev[i][0].elapsed_time(ev[i][1]) * 1000.0 / K_CALLS)
        if r % 10 == 0:
            clocks.append(sm_clock())
            log(f"round {r}: " + " ".join(f"{k[0]}/{k[1]}={samples[k][-1]:.1f}us" for k in keys)
                + f" sm_clock {clocks[-1]}")
    table = {}
    for key in keys:
        xs = samples[key]
        med = statistics.median(xs)
        table[f"{key[0]}/{key[1]}"] = {"median_us": round(med, 2), "p10": round(pctl(xs, .1), 2),
                                       "p90": round(pctl(xs, .9), 2),
                                       "spread": round((pctl(xs, .9) - pctl(xs, .1)) / med, 4)}
    step = {}
    for v in variants:
        xs = [4 * a + b for a, b in zip(samples[(4176, v)], samples[(4096, v)])]
        med = statistics.median(xs)
        step[v] = {"step_us": round(med, 2),
                   "noise_floor": round((pctl(xs, .9) - pctl(xs, .1)) / med, 4),
                   "paired_vs_planar_median_us": round(statistics.median(
                       [x - (4 * a + b) for x, a, b in zip(
                           xs, samples[(4176, "planar")], samples[(4096, "planar")])]), 2)}
    log("per-call table:", json.dumps(table))
    log("per-step (4*M4176 + M4096):", json.dumps(step))

    # Untimed replica: same graphs, same replay counts (soak excluded), bitwise checks.
    checks = {"replay_bitwise_vs_eager_ref": True, "replay_bitwise_vs_replay0": True,
              "strided_ref_eq_planar_ref": all(torch.equal(refs[(m, "strided")], refs[(m, "planar")])
                                               for m in ms),
              "inputs_unchanged": True}
    first = {}
    for r in range(ROUNDS):
        for key in keys:
            graphs[key][0].replay()
        torch.cuda.synchronize()
        for key in keys:
            m, v = key
            ref = refs[(m, "planar")]
            outs = graphs[key][1]
            if not all(torch.equal(o, ref) for o in outs):
                checks["replay_bitwise_vs_eager_ref"] = False
            if r == 0:
                first[key] = [o.clone() for o in outs]
            elif not all(torch.equal(o, f) for o, f in zip(outs, first[key])):
                checks["replay_bitwise_vs_replay0"] = False
    checks["inputs_unchanged"] = all(inputs_unchanged(cases[m]) for m in ms)
    checks["aba_planar_vs_strided"] = aba_check()
    aba_ok = all(p["strided_eq_planar"] and p["contiguous_eq_planar"]
                 for p in checks["aba_planar_vs_strided"])
    passed = (checks["replay_bitwise_vs_eager_ref"] and checks["replay_bitwise_vs_replay0"]
              and checks["strided_ref_eq_planar_ref"] and checks["inputs_unchanged"] and aba_ok)
    log("checks:", json.dumps(checks))
    tv = TREE_DEFAULT if TREE_DEFAULT in step else "planar"
    out = {"batch": {"1": {"kernel_ms": step[tv]["step_us"] / 1000.0,
                           "noise_floor": step[tv]["noise_floor"]}},
           "output": {"class": "exact"}, "tree_variant": tv, "step": step, "per_call": table,
           "sm_clock_log": clocks, "checks": checks}
    if mode == "check":
        out = {"passed": passed, **checks, "tree_variant": tv}
    else:
        out["passed"] = passed
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
