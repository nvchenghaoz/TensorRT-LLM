"""Idea 386: K1FOLD csa2 A (torch.ops.trtllm.csa2_mxfp8_low_m_gemm_k1fold) kernel checks, 1 GPU, synthetic tensors.

Reference = main's chain: fast K1 (mhc_lagged_split.mhc_lagged_layer_input, TRTLLM_MHC_K1_FAST default) -> A
(csa2_mxfp8_low_m_gemm with the chain-A configuration and flags of low_m_gemm.low_m_mm(trigger_at_start=True)).
Compared bitwise: A's rows [M, 1792], layer_input [M, 5120] (written by tile 0's cluster), quantized activation
xq [M, 5120] and its swizzled scales (whole buffer, both pre-filled with the same byte).

  1. bitwise: M = 1, 2; every P of the mhc tactics (96, 224, 256, 480) that the fold admits (k1fold_plan_ok; M=2
     P=96 is refused), with and without the L2 prefetch instance; 64 random seeds per cell, outputs NaN-poisoned
     before every call, plus 8 more poisoned replays.
  2. graph: CUDA-graph replays of [PDL producer (EARLY trigger, 4 us spin) -> fold] vs [producer -> K1 -> A], 8
     iterations with new inputs and poisoned outputs, compared every iteration.
  3. skip-wait control: the fold launched with debug flag bit 0 (no griddepcontrol.wait) behind a slow producer
     (EARLY trigger, 300 us spin) whose output x_prev was NaN-poisoned must differ; the same launch with the wait
     must match. 10 trials each.
  4. stats: cudaOccupancyMaxActiveClusters / blocks per SM / dynamic smem / regs / spills of A fold on vs off and of
     B (the fused q_norm kernel) instances.
"""

import json
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "router_prewait"))

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.attention.backends.sparse.csa2 import low_m_gemm as L  # noqa: E402
from tensorrt_llm._torch.modules.mhc import mhc_lagged_split as S  # noqa: E402

HID, N = 5120, 1792
EPS = 1e-6


def log(*a):
    print(*a, flush=True)


def make_weight(gen):
    w = (torch.randn((N, HID), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    raw = torch.randint(118, 128, (N, HID // 32), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8)
    sw = torch.ops.trtllm.block_scale_interleave(raw.contiguous()).reshape(-1)
    return w, sw


def make_pf(gen):
    # Stand-in for B's weight + scales (the L2 prefetch target; a hint only).
    return (torch.randint(0, 255, (8192 * 1280,), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8),
            torch.randint(0, 255, (8192 * 40,), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8))


def inputs(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scale = [1.0, 1e-3, 30.0][seed % 3]  # small / large magnitudes too (e8m0 exponent range)
    x = (torch.randn((m, HID), generator=g, device="cuda") * scale).to(torch.bfloat16)
    res = (torch.randn((m, 4, HID), generator=g, device="cuda") * scale).to(torch.bfloat16)
    post = torch.rand((m, 4), generator=g, device="cuda") * 2
    comb = torch.rand((m, 4, 4), generator=g, device="cuda") * 0.5
    pre = torch.rand((m, 4), generator=g, device="cuda") + 0.05
    nw = (1.0 + 0.1 * torch.randn(HID, generator=g, device="cuda")).to(torch.bfloat16)
    return [x, res, post, comb, pre, nw]


def flags_for(m, pf_on):
    flags = L._TRIGGER_AT_START
    extra = {}
    if pf_on:
        pos, last, frac = L._L2PF_DEFAULT.get(m, (1, 0, 154))
        flags |= L._L2PF | (pos << 3) | ((1 if last else 0) << 5)
        extra = {"pf_frac": frac}
    return flags, extra


def bufs(m):
    return {"li": torch.empty((m, HID), dtype=torch.bfloat16, device="cuda"),
            "xq": torch.empty((m, HID), dtype=torch.uint8, device="cuda"),
            "xsf": torch.empty((128 * (HID // 32),), dtype=torch.uint8, device="cuda")}


def poison(b):
    b["li"].fill_(float("nan"))
    b["xq"].fill_(0x7F)
    b["xsf"].fill_(0x5A)


def ref_call(inp, P, w, sw, m, pf_on, pft, b):
    x, res, post, comb, pre, nw = inp
    cfg = L.chain_a_config_for(N, HID, m)
    S.mhc_lagged_layer_input(x, res, post, comb.view(m, 4, 4), pre, nw, EPS, P, out=b["li"])
    flags, extra = flags_for(m, pf_on)
    if pf_on:
        extra.update(pf_w=pft[0], pf_sf=pft[1])
    return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(b["li"], w, sw, cfg[0], cfg[1], cfg[2], 0, True, xq_out=b["xq"],
                                                 xsf_out=b["xsf"], debug_flags=flags, **extra)[0]


def fold_call(inp, P, w, sw, m, pf_on, pft, b, skip_wait=False):
    x, res, post, comb, pre, nw = inp
    cfg = L.chain_a_config_for(N, HID, m)
    flags, extra = flags_for(m, pf_on)
    if pf_on:
        extra.update(pf_w=pft[0], pf_sf=pft[1])
    if skip_wait:
        flags |= 1
    return torch.ops.trtllm.csa2_mxfp8_low_m_gemm_k1fold(x, res, post, comb.view(m, 4, 4), pre, nw, EPS, P, b["li"],
                                                         w, sw, cfg[0], cfg[1], cfg[2], True, debug_flags=flags,
                                                         xq_out=b["xq"], xsf_out=b["xsf"], **extra)


def same(a, c):
    return int((a.contiguous().view(torch.uint8) != c.contiguous().view(torch.uint8)).sum().item())


def cmp(r_out, rb, c_out, cb):
    return {"rows": same(r_out, c_out), "li": same(rb["li"], cb["li"]), "xq": same(rb["xq"], cb["xq"]),
            "xsf": same(rb["xsf"], cb["xsf"])}


def main():
    seeds = int(os.environ.get("K1FOLD_SEEDS", "64"))
    gen = torch.Generator(device="cuda").manual_seed(1234)
    w, sw = make_weight(gen)
    pft = make_pf(gen)
    L.k1fold_warmup(w, sw)
    failures = []
    report = {"bitwise": {}, "plan": {}}
    for m in (1, 2):
        for P in (96, 224, 256, 480):
            cfg = L.chain_a_config_for(N, HID, m)
            report["plan"][f"M{m}_P{P}"] = L.k1fold_plan_ok(cfg[0], cfg[1], m, P)
    log(f"plan (admitted by the fold): {report['plan']}")
    # ---- 1. bitwise, for every P the fold admits at this M (the others keep K1 -> A)
    for m in (1, 2):
        for P in (96, 224, 256, 480):
            if not report["plan"][f"M{m}_P{P}"]:
                continue
            for pf_on in (False, True):
                key = f"M{m}_P{P}_pf{int(pf_on)}"
                bad = {"rows": 0, "li": 0, "xq": 0, "xsf": 0}
                nonfin = 0
                rb, cb = bufs(m), bufs(m)
                for s in range(seeds + 8):
                    inp = inputs(m, 1000 * m + P + (s if s < seeds else 0))
                    poison(rb)
                    poison(cb)
                    r = ref_call(inp, P, w, sw, m, pf_on, pft, rb)
                    c = fold_call(inp, P, w, sw, m, pf_on, pft, cb)
                    torch.cuda.synchronize()
                    for k_, v in cmp(r, rb, c, cb).items():
                        bad[k_] += v
                    nonfin += int((~torch.isfinite(c.float())).sum().item()) + int(
                        (~torch.isfinite(cb["li"].float())).sum().item())
                report["bitwise"][key] = {"calls": seeds + 8, "mismatch": bad, "nonfinite": nonfin}
                log(f"bitwise {key}: {report['bitwise'][key]}")
                if any(bad.values()) or nonfin:
                    failures.append(f"bitwise {key} {bad} nonfinite {nonfin}")
    # ---- 2. graphs behind a PDL producer
    import pdl_ext

    ext = pdl_ext.ext()
    report["graph"] = {}
    for m in (1, 2):
        P = 480
        pf_on = m in L._L2PF_DEFAULT
        inp = inputs(m, 7)
        src = inp[0].clone()
        xbuf = torch.empty_like(src)
        rb, cb = bufs(m), bufs(m)
        static = [xbuf] + inp[1:]

        def run_ref():
            ext.producer(src, xbuf, 4000, 4 * m, 160, pdl_ext.empty(), True)
            return ref_call(static, P, w, sw, m, pf_on, pft, rb)

        def run_fold():
            ext.producer(src, xbuf, 4000, 4 * m, 160, pdl_ext.empty(), True)
            return fold_call(static, P, w, sw, m, pf_on, pft, cb)

        graphs = {}
        for name, fn in (("ref", run_ref), ("fold", run_fold)):
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                fn()
                fn()
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st):
                out = fn()
            torch.cuda.synchronize()
            graphs[name] = (g, out)
        mism = []
        for it in range(8):
            new = inputs(m, 500 + it)
            src.copy_(new[0])
            for dst, s_ in zip(static[1:], new[1:]):
                dst.copy_(s_)
            xbuf.fill_(float("nan"))
            poison(rb)
            poison(cb)
            graphs["ref"][0].replay()
            torch.cuda.synchronize()
            xbuf.fill_(float("nan"))
            graphs["fold"][0].replay()
            torch.cuda.synchronize()
            d = cmp(graphs["ref"][1], rb, graphs["fold"][1], cb)
            mism.append(sum(d.values()))
        report["graph"][f"M{m}"] = {"iters": 8, "mismatch_per_iter": mism}
        log(f"graph M={m}: {report['graph'][f'M{m}']}")
        if any(mism):
            failures.append(f"graph M={m} {mism}")
    # ---- 3. skip-wait negative control
    report["skip_wait"] = {}
    for m in (1, 2):
        P = 480
        caught, ok_wait = 0, 0
        for t in range(10):
            inp = inputs(m, 900 + t)
            src = inp[0].clone()
            xbuf = torch.full_like(src, float("nan"))
            rb, cb = bufs(m), bufs(m)
            poison(rb)
            r = ref_call(inp, P, w, sw, m, False, pft, rb)
            torch.cuda.synchronize()
            for skip in (True, False):
                xbuf.fill_(float("nan"))
                poison(cb)
                torch.cuda.synchronize()
                # 300 us: longer than the host's launch of the fold op, so a skipped wait reads the poison.
                ext.producer(src, xbuf, 300000, 4 * m, 160, pdl_ext.empty(), True)
                c = fold_call([xbuf] + inp[1:], P, w, sw, m, False, pft, cb, skip_wait=skip)
                torch.cuda.synchronize()
                d = sum(cmp(r, rb, c, cb).values())
                if skip:
                    caught += int(d > 0)
                else:
                    ok_wait += int(d == 0)
        report["skip_wait"][f"M{m}"] = {"trials": 10, "skip_wait_caught": caught, "with_wait_exact": ok_wait}
        log(f"skip-wait M={m}: {report['skip_wait'][f'M{m}']}")
        if caught < 10 or ok_wait < 10:
            failures.append(f"skip-wait M={m} caught {caught}/10 exact-with-wait {ok_wait}/10")
    # ---- 4. occupancy / attributes
    names = ["max_active_clusters", "blocks_per_sm", "dyn_smem", "regs", "local_bytes", "static_smem", "max_threads"]
    stats = {}
    dev = torch.empty(1, device="cuda")
    for m in (1, 2):
        cfg = L.chain_a_config_for(N, HID, m)
        for pf_on in (False, True):
            for fold in (False, True):
                v = torch.ops.trtllm.csa2_low_m_gemm_instance_stats(dev, cfg[0], cfg[1], cfg[2], m, HID, False, pf_on,
                                                                    fold).tolist()
                stats[f"A_M{m}_{cfg[0]}x{cfg[1]}x{cfg[2]}_pf{int(pf_on)}_fold{int(fold)}"] = dict(zip(names, v))
        for qcfg in ((4, 1, 2), (4, 1, 4), (8, 1, 2)):
            v = torch.ops.trtllm.csa2_low_m_gemm_instance_stats(dev, qcfg[0], qcfg[1], qcfg[2], m, 1280, True, False,
                                                                False).tolist()
            stats[f"B_M{m}_{qcfg[0]}x{qcfg[1]}x{qcfg[2]}"] = dict(zip(names, v))
        for pf_on in (False, True):
            a0 = stats[f"A_M{m}_{cfg[0]}x{cfg[1]}x{cfg[2]}_pf{int(pf_on)}_fold0"]
            a1 = stats[f"A_M{m}_{cfg[0]}x{cfg[1]}x{cfg[2]}_pf{int(pf_on)}_fold1"]
            if a0["max_active_clusters"] != a1["max_active_clusters"] or a0["blocks_per_sm"] != a1["blocks_per_sm"]:
                failures.append(f"occupancy M={m} pf={pf_on}: {a0} vs {a1}")
            if a1["local_bytes"]:
                failures.append(f"spills M={m} pf={pf_on}: {a1['local_bytes']} B")
    report["stats"] = stats
    for k_, v in stats.items():
        log(f"stats {k_}: {v}")
    passed = not failures
    print(json.dumps({"passed": passed, "failures": failures,
                      "compared": "K1FOLD A (csa2_mxfp8_low_m_gemm_k1fold) vs main fast K1 -> csa2_mxfp8_low_m_gemm "
                                  "(rows, layer_input, xq, xsf bitwise)", **report}))


if __name__ == "__main__":
    main()
