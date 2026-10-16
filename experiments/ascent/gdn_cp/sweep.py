"""GDN prefill (FlashInfer SM120) cp_chunk_len / CP-dispatch sweep.

Shapes: Qwen3.8-27B GDN layer: q/k [T,16,128] bf16 (L2-normalized), v [T,48,128]
bf16, alpha=exp(g_log) fp32 [T,48] with g_log<=0, beta fp32 [T,48], fp32 initial
state [N,48,128,128]; N sequences x 2048 tokens.

Modes
  sweep : time every arm (cold L2, outputs restored from pristine copies each
          iteration, >=10 warmup + >=50 timed, CUDA events), phase split for CP
          arms, numerics vs an FP32 sequential delta-rule reference.
  pair  : time only the default flashinfer.chunk_gated_delta_rule dispatch
          (what TRT-LLM's flashinfer_chunk.py calls) and print the evidence JSON.
  check : untimed repeated-execution sequence with the same warmup/iteration/
          reset points; every iteration compared bitwise with iteration 1 and
          with the FP32 reference (reordered tolerance vs main's dispatch).
"""

import argparse
import json
import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F

H, HV, D = 16, 48, 128
SEQ = 2048
SCALE = 1.0 / math.sqrt(D)


def log(*a):
    print(*a, flush=True)


def make_inputs(n, seed=1234):
    g = torch.Generator(device="cuda").manual_seed(seed + n)
    t = n * SEQ
    q = torch.randn(t, H, D, device="cuda", generator=g)
    k = torch.randn(t, H, D, device="cuda", generator=g)
    q = F.normalize(q, dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
    k = F.normalize(k, dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
    v = torch.randn(t, HV, D, device="cuda", generator=g).to(torch.bfloat16).contiguous()
    # TRT-LLM: g = -exp(A_log) * softplus(a + dt_bias)  (log space, <= 0)
    a_coef = torch.empty(HV, device="cuda").uniform_(1.0, 16.0, generator=g)
    a_in = torch.randn(t, HV, device="cuda", generator=g) - 4.0
    g_log = -a_coef[None, :] * F.softplus(a_in)
    alpha = torch.exp(g_log).contiguous()
    beta = torch.sigmoid(torch.randn(t, HV, device="cuda", generator=g)).contiguous()
    s0 = (0.3 * torch.randn(n, HV, D, D, device="cuda", generator=g)).contiguous()
    cu = torch.arange(0, n + 1, device="cuda", dtype=torch.int64) * SEQ
    return dict(q=q, k=k, v=v, alpha=alpha, beta=beta, s0=s0, cu=cu, n=n, t=t)


@torch.no_grad()
def reference(inp):
    """FP32 sequential gated delta rule. State layout [N, HV, V, K]."""
    torch.backends.cuda.matmul.allow_tf32 = False
    n = inp["n"]
    rep = HV // H
    q = inp["q"].float().repeat_interleave(rep, dim=1).view(n, SEQ, HV, D)
    k = inp["k"].float().repeat_interleave(rep, dim=1).view(n, SEQ, HV, D)
    v = inp["v"].float().view(n, SEQ, HV, D)
    al = inp["alpha"].view(n, SEQ, HV)
    be = inp["beta"].view(n, SEQ, HV)
    s = inp["s0"].clone()  # [n,HV,V,K]
    o = torch.empty(n, SEQ, HV, D, device="cuda", dtype=torch.float32)
    for i in range(SEQ):
        kk = k[:, i]
        s.mul_(al[:, i, :, None, None])
        kvm = torch.matmul(s, kk.unsqueeze(-1)).squeeze(-1)  # [n,HV,V]
        delta = be[:, i, :, None] * (v[:, i] - kvm)
        s.add_(delta.unsqueeze(-1) * kk.unsqueeze(-2))
        o[:, i] = SCALE * torch.matmul(s, q[:, i].unsqueeze(-1)).squeeze(-1)
    return o.view(n * SEQ, HV, D), s


def errs(x, ref):
    d = (x.float() - ref).abs()
    return dict(
        max_abs=float(d.max()),
        mean_abs=float(d.mean()),
        finite=bool(torch.isfinite(x.float()).all()),
    )


def make_arm(name, inp):
    import flashinfer
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm120 import (
        cp_delta_rule_dsl_sm120,
    )

    q, k, v, al, be, s0, cu = (inp[x] for x in ("q", "k", "v", "alpha", "beta", "s0", "cu"))
    t = inp["t"]

    if name in ("default", "nocp", "cp_api"):
        use_cp = {"default": "auto", "nocp": False, "cp_api": True}[name]

        def fn(o, st):
            flashinfer.chunk_gated_delta_rule(
                q=q, k=k, v=v, g=al, beta=be, scale=SCALE, initial_state=s0,
                output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=False,
                output=o, output_state=st, use_cp=use_cp,
            )

        return fn
    if name.startswith("cpL"):
        L = int(name[3:])

        def fn(o, st):
            cp_delta_rule_dsl_sm120(
                o, st, q, k, v, al, be, cu, SCALE, initial_state=s0,
                max_seqlen=SEQ, cp_chunk_len=L,
            )

        return fn
    if name.startswith("mainemu"):  # mainemu / mainemu2 (repeat, bracketing)
        # main's dispatch: CP at L=1024 with max_seqlen=total at N=1, non-CP otherwise
        if inp["n"] == 1:
            def fn(o, st):
                cp_delta_rule_dsl_sm120(
                    o, st, q, k, v, al, be, cu, SCALE, initial_state=s0,
                    max_seqlen=t, cp_chunk_len=1024,
                )
        else:
            def fn(o, st):
                flashinfer.chunk_gated_delta_rule(
                    q=q, k=k, v=v, g=al, beta=be, scale=SCALE, initial_state=s0,
                    output_final_state=True, cu_seqlens=cu, output=o, output_state=st,
                    use_cp=False,
                )
        return fn
    raise ValueError(name)


class Bufs:
    def __init__(self, inp):
        self.o = torch.empty(inp["t"], HV, D, device="cuda", dtype=torch.bfloat16)
        self.st = torch.empty(inp["n"], HV, D, D, device="cuda", dtype=torch.float32)
        self.o_pristine = torch.zeros_like(self.o)
        self.st_pristine = torch.zeros_like(self.st)
        self.s0_pristine = inp["s0"].clone()
        self.s0 = inp["s0"]

    def reset(self):
        self.o.copy_(self.o_pristine)
        self.st.copy_(self.st_pristine)
        self.s0.copy_(self.s0_pristine)


FLUSH = None


def flush_l2():
    global FLUSH
    if FLUSH is None:
        FLUSH = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    FLUSH.zero_()


def time_arm(fn, bufs, warmup, iters):
    for _ in range(warmup):
        bufs.reset()
        fn(bufs.o, bufs.st)
    torch.cuda.synchronize()
    evs = []
    for _ in range(iters):
        bufs.reset()
        flush_l2()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn(bufs.o, bufs.st)
        b.record()
        evs.append((a, b))
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in evs)
    med = statistics.median(ts)
    p90 = ts[min(len(ts) - 1, int(round(0.9 * (len(ts) - 1))))]
    p10 = ts[int(round(0.1 * (len(ts) - 1)))]
    return dict(median_ms=med, p90_ms=p90, p10_ms=p10, noise=(p90 - p10) / med)


def phase_split(inp, L, bufs, iters=30, max_seqlen=SEQ):
    import cuda.bindings.driver as cuda_driver
    from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm120 as m

    q, k, v, al, be, s0, cu = (inp[x] for x in ("q", "k", "v", "alpha", "beta", "s0", "cu"))
    t = inp["t"]
    dev = q.device
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)

    def run(evs=None):
        def mark(i):
            if evs is not None:
                evs[i].record()

        mark(0)
        tt = m.cp_delta_rule_t_precompute_dsl_sm120(
            k, be, cu, t, max_seqlen=max_seqlen, _skip_check=True, _device=dev, _stream=stream)
        mark(1)
        lt, ls = m.cp_delta_rule_mn_precompute_dsl_sm120(
            k, v, tt, al, cu, t, cp_chunk_len=L, max_seqlen=max_seqlen,
            _skip_check=True, _device=dev, _stream=stream)
        mark(2)
        fs = m.cp_delta_rule_fixup_dsl_sm120(
            lt, ls, cu, t, cp_chunk_len=L, initial_state=s0,
            _skip_check=True, _device=dev, _stream=stream)
        mark(3)
        m.cp_delta_rule_prefill_dsl_sm120(
            bufs.o, bufs.st, q, k, v, tt, fs, al, SCALE, cu, t, cp_chunk_len=L,
            max_seqlen=max_seqlen, initial_state=s0, _skip_check=True, _device=dev,
            _stream=stream)
        mark(4)

    for _ in range(5):
        bufs.reset()
        run()
    torch.cuda.synchronize()
    allev = []
    for _ in range(iters):
        bufs.reset()
        flush_l2()
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        run(evs)
        allev.append(evs)
    torch.cuda.synchronize()
    names = ["t_precompute", "mn_precompute", "fixup", "prefill"]
    out = {}
    for j, nm in enumerate(names):
        out[nm] = statistics.median(e[j].elapsed_time(e[j + 1]) for e in allev)
    return out


def default_choice(n):
    """What the stock dispatch does (for the table)."""
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
        choose_cp_chunk_len_host,
        should_use_cp_host,
    )
    from flashinfer.utils import get_compute_capability, get_device_name, get_device_sm_count

    dev = torch.device("cuda")
    sms = get_device_sm_count(dev)
    cap = get_compute_capability(dev)
    name = get_device_name(dev)
    cp = should_use_cp_host(n * HV, sms, name, device_capability=cap)
    L = choose_cp_chunk_len_host(n * SEQ, HV, sms, device_capability=cap,
                                 total_seqlen=n * SEQ, num_seqs=n, device_name=name)
    return dict(sms=sms, cap=list(cap), name=name, auto_cp=bool(cp), api_L_if_cp=L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep", choices=["sweep", "pair", "check"])
    ap.add_argument("--ns", default="1,2,4")
    ap.add_argument("--Ls", default="2048,1024,768,704,640,512,384,256")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--arms", default="")
    ap.add_argument("--phase_Ls", default="1024,704,512")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--no_ref", action="store_true")
    args = ap.parse_args()
    global SEQ
    SEQ = args.seq
    torch.backends.cuda.matmul.allow_tf32 = False
    ns = [int(x) for x in args.ns.split(",")]
    Ls = [int(x) for x in args.Ls.split(",") if x]

    result = {"mode": args.mode, "per_n": {}}
    for n in ns:
        t0 = time.time()
        inp = make_inputs(n)
        log(f"[N={n}] stock choice: {default_choice(n)}")
        if args.no_ref:
            ref_o = torch.zeros(inp["t"], HV, D, device="cuda")
            ref_s = torch.zeros(n, HV, D, D, device="cuda")
        else:
            ref_o, ref_s = reference(inp)
        log(f"[N={n}] T/seq={SEQ} reference done in {time.time() - t0:.1f}s")
        bufs = Bufs(inp)
        if args.mode == "sweep":
            if args.arms:
                arms = args.arms.split(",")
            else:
                arms = ["default", "nocp", "mainemu"]
                if n > 1:
                    arms.append("cp_api")
            arms += [f"cpL{L}" for L in Ls]
        elif args.mode == "pair":
            # default = this tree's dispatch; mainemu = main's dispatch, same process
            arms = ["mainemu", "default", "mainemu2"]
        else:
            arms = args.arms.split(",") if args.arms else ["default", "mainemu"]
        per = {"stock": default_choice(n)}

        def bf16_ulp(x):
            m = float(x.abs().max())
            return 2.0 ** (math.floor(math.log2(m)) - 7) if m > 0 else 0.0

        per["ulp"] = {"o": bf16_ulp(ref_o), "state": bf16_ulp(ref_s)}
        for arm in arms:
            ta = time.time()
            fn = make_arm(arm, inp)
            # numerics (first call)
            bufs.reset()
            fn(bufs.o, bufs.st)
            torch.cuda.synchronize()
            num = dict(o=errs(bufs.o, ref_o), state=errs(bufs.st, ref_s))
            rec = {"numerics": num}
            if args.mode in ("sweep", "pair"):
                rec.update(time_arm(fn, bufs, args.warmup, args.iters))
            else:
                # repeated-execution: same warmup/iters/reset points as timing
                first_o = first_s = None
                max_drift = 0.0
                worst = dict(o_max=0.0, o_mean=0.0, s_max=0.0, s_mean=0.0)
                bitwise = True
                total = args.warmup + args.iters
                for it in range(total):
                    # before reset: initial state must be untouched
                    if not torch.equal(bufs.s0, bufs.s0_pristine):
                        bitwise = False
                        log(f"  initial state mutated before iter {it}")
                    bufs.reset()
                    flush_l2()
                    fn(bufs.o, bufs.st)
                    torch.cuda.synchronize()
                    if first_o is None:
                        first_o = bufs.o.clone()
                        first_s = bufs.st.clone()
                    else:
                        eq = torch.equal(bufs.o, first_o) and torch.equal(bufs.st, first_s)
                        if not eq:
                            bitwise = False
                            d = max(float((bufs.o.float() - first_o.float()).abs().max()),
                                    float((bufs.st - first_s).abs().max()))
                            max_drift = max(max_drift, d)
                    eo = errs(bufs.o, ref_o)
                    es = errs(bufs.st, ref_s)
                    worst["o_max"] = max(worst["o_max"], eo["max_abs"])
                    worst["o_mean"] = max(worst["o_mean"], eo["mean_abs"])
                    worst["s_max"] = max(worst["s_max"], es["max_abs"])
                    worst["s_mean"] = max(worst["s_mean"], es["mean_abs"])
                    if not (eo["finite"] and es["finite"]):
                        bitwise = False
                rec.update(dict(iterations=total, bitwise_vs_iter1=bitwise,
                                max_drift=max_drift, worst_vs_ref=worst))
            log(f"[N={n}] {arm}: {json.dumps(rec)} ({time.time() - ta:.1f}s)")
            per[arm] = rec
        if args.mode == "sweep":
            split = {}
            for L in [int(x) for x in args.phase_Ls.split(",") if x]:
                split[str(L)] = phase_split(inp, L, bufs)
                log(f"[N={n}] phase split L={L}: {json.dumps(split[str(L)])}")
            per["phase_split"] = split
        result["per_n"][str(n)] = per
        del bufs, inp, ref_o, ref_s
        torch.cuda.empty_cache()

    if args.mode == "pair":
        out = {"batch": {}, "output": {"class": "reordered"}}
        for n, per in result["per_n"].items():
            d = per["default"]
            m1, m2 = per["mainemu"], per["mainemu2"]
            out["batch"][n] = {"kernel_ms": d["median_ms"], "noise_floor": d["noise"],
                               "p90_ms": d["p90_ms"], "numerics": d["numerics"],
                               "main_dispatch_same_process_ms": [m1["median_ms"], m2["median_ms"]],
                               "main_dispatch_p90_ms": [m1["p90_ms"], m2["p90_ms"]]}
        print(json.dumps(out))
    elif args.mode == "check":
        # reordered tolerance: candidate err <= 1.25 * main err + 1 bf16 ulp
        passed = True
        summary = {}
        for n, per in result["per_n"].items():
            main_rec = per.get("mainemu")
            for arm, rec in per.items():
                if not isinstance(rec, dict) or "bitwise_vs_iter1" not in rec:
                    continue
                ok = rec["bitwise_vs_iter1"]
                lims = {}
                if main_rec is not None and arm != "mainemu":
                    mw = main_rec["worst_vs_ref"]
                    for key, w in (("o", "o"), ("state", "s")):
                        ulp = per["ulp"][key]
                        for stat in ("max", "mean"):
                            lim = 1.25 * mw[f"{w}_{stat}"] + ulp
                            got = rec["worst_vs_ref"][f"{w}_{stat}"]
                            lims[f"{w}_{stat}"] = lim
                            if not (got <= lim):
                                ok = False
                                log(f"  N={n} {arm} {key} {stat} {got} > {lim}")
                summary[f"N{n}_{arm}"] = dict(ok=ok, worst=rec["worst_vs_ref"], limits=lims,
                                              bitwise_vs_iter1=rec["bitwise_vs_iter1"],
                                              max_drift=rec["max_drift"])
                passed = passed and ok
        print(json.dumps({"passed": passed, "compared": "o and final state vs FP32 sequential "
                          "delta-rule reference; every iteration vs iteration 1 (bitwise)",
                          "summary": summary}))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
