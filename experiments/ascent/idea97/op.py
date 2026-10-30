"""Idea 97 op pair / check: the served decoder mid-block attention at 2048^2 (T=65536, c=512).

usage: python op.py time|check
  time : runs on base and branch trees. torch.compile(Flux3ImageAttnBlock._attention_tc) with
         fp8_pv=True and the decoder's compile options, on channels_last bf16 q/k/v
         [1, 512, 256, 256] (the 1x1 conv outputs of the channels_last decoder), cold L2,
         CUDA events. Base = landing 25's bmm + row_softmax_e4m3 + e4m3 _scaled_mm; branch =
         split_attn_triton (kernel A + kernel B). Then an untimed repeat sequence with the
         same warmup/iteration count, bitwise vs iteration 0, inputs unchanged.
  check: branch tree only. (1) split_attn_e4m3 under torch.compile == direct launch, bitwise.
         (2) compiled _attention_tc with TLLM_FLUX3_VAE_SPLIT_ATTN off (A, landing 25's path)
         and on (B), alternating 50 rounds over two inputs with error_on_recompile; B bitwise
         vs its first; inputs unchanged; B vs A within the declared tolerance; kernel census
         of both arms (no _softmax_rows_e4m3 / _scaled_mm in B, kernels A and B once).
DECLARED TOLERANCE (before timing): O (bf16) B vs A: max abs diff <= 2 bf16 ulp of max|O|
  (2 * 2^-7 * 2^floor(log2 max|O|)) and >= 99.9% of elements bitwise; this is the P.V
  accumulation-order and l-order difference (pq bitwise, scale_a <= 1e-5 rel: s1 job 759).
"""

import json
import os
import sys
import time
from collections import Counter

import torch

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv

MODE = sys.argv[1] if len(sys.argv) > 1 else "time"
ITERS = int(os.environ.get("ITERS", "30"))
WARM = int(os.environ.get("WARM", "3"))
C, H = 512, 256
T0 = time.time()
torch._dynamo.config.cache_size_limit = 64
OPTS = {"layout_optimization": False, "emulate_precision_casts": True}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for amp in (1.5, 1.5, 1.0):
        x = (torch.randn(1, H, H, C, device="cuda", generator=g) * amp).to(torch.bfloat16)
        out.append(x.permute(0, 3, 1, 2))  # [1, C, H, W] channels_last
    return out


FLUSH = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)


def timed(fn, *a):
    FLUSH.fill_(1.0)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn(*a)
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": round(med, 4), "p10_ms": round(p10, 4), "p90_ms": round(p90, 4),
            "noise_floor": round((p90 - p10) / med, 5), "n": n}


def attn(q, k, v):
    return iv.Flux3ImageAttnBlock._attention_tc(q, k, v, True)


def census(fn, *a):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*a)
        torch.cuda.synchronize()
    cnt, dur = Counter(), Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name] += 1
            dur[ev.name] += ev.device_time_total / 1e3
    return {"gpu_total_ms": round(sum(dur.values()), 3),
            "kernels": [(k[:100], cnt[k], round(dur[k], 3)) for k, _ in dur.most_common(12)]}


@torch.inference_mode()
def run_time():
    qkv = inputs(0)
    keep = [t.clone() for t in qkv]
    f = torch.compile(attn, dynamic=False, options=OPTS)
    t = time.time()
    f(*qkv)
    torch.cuda.synchronize()
    log(f"compile {time.time() - t:.1f}s")
    for _ in range(WARM):
        f(*qkv)
    ts = []
    for i in range(ITERS):
        ts.append(timed(f, *qkv))
        if i % 10 == 0:
            log(f"iter {i} {ts[-1]:.3f} ms")
    st = stats(ts)
    log("timing", json.dumps(st))
    cen = census(f, *qkv)
    log("census", json.dumps(cen))
    for _ in range(WARM):
        f(*qkv)
    first, rep, inp = None, True, True
    for _ in range(ITERS):
        inp &= all(torch.equal(a, b) for a, b in zip(qkv, keep))
        y = f(*qkv)
        if first is None:
            first = y.clone()
        else:
            rep &= bool(torch.equal(y, first))
    res = {"batch": {"t2i-2048": {"kernel_ms": st["median_ms"], "noise_floor": st["noise_floor"]},
                     "i2i-2048": {"kernel_ms": st["median_ms"], "noise_floor": st["noise_floor"]}},
           "output": {"class": "reordered"}, "timing": st, "census": cen,
           "repeat_bitwise": rep, "inputs_unchanged": inp, "passed": bool(rep and inp),
           "split_attn_present": hasattr(iv, "_sat"),
           "bench": "idea97/op.py time: compiled _attention_tc(fp8_pv=True), channels_last bf16 "
                    "q/k/v [1,512,256,256] (decoder mid-block at 2048^2); both cases decode once"}
    print(json.dumps(res))


def _ord16(t):
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


@torch.inference_mode()
def run_check():
    from torch._dynamo.utils import counters

    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import split_attn_triton as sa

    res, gates = {}, {}
    qa, qb = inputs(0), inputs(1)
    keep = [t.clone() for t in qa + qb]
    # (1) op compiled vs direct
    q, k, v = qa
    vf = v[0].float().reshape(C, -1)
    s_v = (vf.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1e-30)
    vq = (vf / s_v).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    q2, k2 = q.reshape(1, C, -1)[0], k.reshape(1, C, -1)[0]
    res["q_strides"] = list(q2.stride())
    d = sa.split_attn_e4m3_direct(q2, k2, vq, s_v.reshape(C), C**-0.5)
    fop = torch.compile(lambda a, b, c_, s_: sa.split_attn_e4m3(a, b, c_, s_, C**-0.5), dynamic=False,
                        options=OPTS)
    o = fop(q2, k2, vq, s_v.reshape(C))
    gates["op_compiled_eq_direct"] = bool(torch.equal(o, d))
    del o, d, vf, vq
    torch.cuda.empty_cache()
    log("op compiled == direct", gates["op_compiled_eq_direct"])
    # (2) arms
    fns = {}
    for arm, en in (("A", False), ("B", True)):
        sa.ENABLED = en
        gb0 = sum(counters["graph_break"].values())
        f = torch.compile(lambda a, b, c_: attn(a, b, c_), dynamic=False, options=OPTS)
        f(*qa)
        torch.cuda.synchronize()
        fns[arm] = f
        res[f"graph_breaks_{arm}"] = sum(counters["graph_break"].values()) - gb0
        res[f"census_{arm}"] = census(f, *qa)
        log(arm, json.dumps(res[f"census_{arm}"]))
    gates["no_graph_breaks"] = res["graph_breaks_A"] == 0 and res["graph_breaks_B"] == 0
    nb = " ".join(n for n, _, _ in res["census_B"]["kernels"])
    na = " ".join(n for n, _, _ in res["census_A"]["kernels"])
    def count(sub):
        return sum(c for n, c, _ in res["census_B"]["kernels"] if sub in n)

    gates["B_kernels_once"] = count("qk_tilemax") == 1 and count("pv_prologue") == 1
    gates["B_no_softmax_no_scaled_mm"] = "_softmax_rows_e4m3" not in nb and "gemm" not in nb.lower() \
        and "nvjet" not in nb
    gates["A_is_landing25"] = "_softmax_rows_e4m3_kernel" in na and "_qk_tilemax" not in na
    torch._dynamo.config.error_on_recompile = True
    first, rep, err = {}, True, None
    rounds = int(os.environ.get("ROUNDS", "50"))
    try:
        for i in range(rounds):
            arms = ("B",) if i % 10 else (("A", "B") if i % 20 == 0 else ("B", "A"))
            for arm in arms:
                sa.ENABLED = arm == "B"
                for name, x in (("a", qa), ("b", qb)):
                    y = fns[arm](*x)
                    key = (arm, name)
                    if key in first:
                        rep &= bool(torch.equal(y, first[key]))
                    else:
                        first[key] = y.clone()
                    del y
            if i % 10 == 0:
                ok = all(torch.equal(a, b) for a, b in zip(qa + qb, keep))
                gates[f"inputs_unchanged_{i}"] = ok
                log(f"round {i}: repeat so far {rep}, inputs unchanged {ok}")
    except Exception as ex:  # noqa: BLE001
        err = repr(ex)[:500]
        log("FAILED", err)
    torch._dynamo.config.error_on_recompile = False
    gates["no_recompile"] = err is None
    gates["repeat_bitwise"] = rep
    cmp = {}
    for name in ("a", "b"):
        a, b = first[("A", name)], first[("B", name)]
        mx = a.float().abs().max().item()
        import math
        ulp = 2.0 ** (math.floor(math.log2(mx)) - 7)
        dd = (a.float() - b.float()).abs()
        u = (_ord16(a) - _ord16(b)).abs()
        cmp[name] = {"max_abs": dd.max().item(), "bound": 2 * ulp, "max|O|": mx,
                     "bitwise_share": (u == 0).float().mean().item(), "max_ulp": int(u.max().item())}
        gates[f"tol_{name}"] = dd.max().item() <= 2 * ulp and cmp[name]["bitwise_share"] >= 0.999
    res["B_vs_A"] = cmp
    res["recompile_error"] = err
    res["gates"] = gates
    res["passed"] = bool(all(gates.values()))
    res["output"] = {"class": "reordered"}
    log(json.dumps(cmp))
    print(json.dumps(res))


if __name__ == "__main__":
    {"time": run_time, "check": run_check}[MODE]()
