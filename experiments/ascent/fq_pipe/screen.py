"""Idea 163: fused-quant producer pipelining in FlashInfer b12x _lib/dense_gemm.py.

Arms are FlashInfer fq_pipeline = (b_first, wide, prefetch) on the two WO
fused-quant GEMMs (per-rank TP4 shapes from the checkpoint config, synthetic
tensors, no model load):
  K2 = wo_a_dense_gemm_fused_quant_mxfp8 (V11 wo_a, grouped, BF16 A)
  K3 = wo_b_dense_gemm_fused_quant_mxfp8 (wo_b, group-major BF16 A)
"off" = (False, False, 0) is the pinned producer, in the same process.

Numerics: every arm bitwise vs "off" at each M on random + adversarial per-32
blocks (zero blocks, single nonzero max/min-normal bf16 both signs, 1.875*2^e
maxima, 448*sc next to E4M3-subnormal neighbours).
Repeated sequence: the timed graph (40 layers) replayed WARM + REPLAYS times,
every layer's output compared to the eager "off" output after every replay,
inputs checked unmodified.
Timing: cold-L2 40-layer CUDA graphs (weights rotated, ~640 MB), CUDA events,
median us/layer and (p90-p10)/median. Chains: K1 (main's inv_rope_quant128,
storing the bf16 rope output) -> K2 arm -> K3 arm; old = K1 -> TMA-A
wo_a_dense_gemm_mxfp8 -> K3 off. Also K2-only and K3-only rows.

Usage: screen.py <ckpt> [MS] [ARMS]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b12x_wo"))
import common as C  # noqa: E402
import torch  # noqa: E402

from b12x.gemm._shared import wo_mxfp8 as W  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2 import b12x_wo as B  # noqa: E402

ckpt = sys.argv[1]
MS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8").split(",")]
ALL_ARMS = {
    "off": (False, False, 0),
    "bf": (True, False, 0),
    "w": (False, True, 0),
    "bfw": (True, True, 0),
    "bfw_p1": (True, True, 1),
    "bfw_p2": (True, True, 2),
    "bfw_p4": (True, True, 4),
    "w_p1": (False, True, 1),
    "w_p2": (False, True, 2),
    "w_p4": (False, True, 4),
}
# K2_FIXED=off: chains keep K2 on the pinned producer and vary only K3 (K3-only landing)
K2_FIXED = os.environ.get("K2_FIXED", "")
ARM_NAMES = (sys.argv[3] if len(sys.argv) > 3 else "off,bf,bfw,bfw_p2,bfw_p4").split(",")
ARMS = {k: ALL_ARMS[k] for k in ARM_NAMES}
assert "off" in ARMS
LAYERS = int(os.environ.get("LAYERS", "40"))
WARM = int(os.environ.get("WARM", "5"))
REPLAYS = int(os.environ.get("REPLAYS", "30"))
SFB_REP_K2 = os.environ.get("SFB_REP_K2", "1") == "1"
DO_TIMING = os.environ.get("TIMING", "1") == "1"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


dims = C.dims_from_ckpt(ckpt)
G, R, GW, H = dims["groups_local"], dims["rank"], dims["group_width"], dims["hidden"]
log("dims", dims)
w0 = C.make_weights(dims)
cs = C.make_cos_sin(dims)
ws = [w0] + [C.clone_weights(w0, dims) for _ in range(LAYERS - 1)]
log(f"{LAYERS} weight copies ready")


def k2(src, w, out, M, arm):
    return W.wo_a_dense_gemm_fused_quant_mxfp8(
        src, w.wa_b12x, out=out, expected_m=M, sfb_k_replicated=SFB_REP_K2, fq_pipeline=arm)


def k3(tmp, w, out, M, arm):
    return W.wo_b_dense_gemm_fused_quant_mxfp8(
        tmp, w.wb_b12x, out=out, expected_m=M, sfb_k_replicated=False, fq_pipeline=arm)


def adversarial(M, shape_last, nrows_view, seed):
    """BF16 [M, X] with random rows and planted per-32 block cases."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, shape_last, device=C.DEV, generator=g)
    nb = shape_last // 32
    xb = x.view(M, nb, 32)
    for r in range(M):
        for j in range(12):
            b = (r * 13 + j * 37) % nb
            blk = torch.zeros(32, device=C.DEV)
            p = (r + j) % 32
            if j == 0:
                pass  # all-zero block
            elif j == 1:
                blk[p] = 3.3895313892515355e38  # max bf16
            elif j == 2:
                blk[p] = -3.3895313892515355e38
            elif j == 3:
                blk[p] = 1.1754943508222875e-38  # min normal bf16
            elif j == 4:
                blk[p] = -1.1754943508222875e-38
            elif j == 5:
                blk = torch.rand(32, device=C.DEV, generator=g) * 200.0 * 2.0 ** -5
                blk[p] = 240.0 * 2.0 ** -5  # 1.875 * 2^2
            elif j == 6:
                blk = -torch.rand(32, device=C.DEV, generator=g) * 2.0 ** 90
                blk[p] = -1.875 * 2.0 ** 97
            elif j == 7:
                # 448*sc next to neighbours in E4M3's subnormal range at sc=2^-3
                blk = (torch.rand(32, device=C.DEV, generator=g) - 0.5) * 2.0 ** -3 * 2.0 ** -7
                blk[p] = 448.0 * 2.0 ** -3
            elif j == 8:
                blk = (torch.rand(32, device=C.DEV, generator=g) - 0.5) * 2.0 ** -12
                blk[p] = -448.0 * 2.0 ** -9
            elif j == 9:
                blk[p] = 1.0
            elif j == 10:
                blk = torch.randn(32, device=C.DEV, generator=g) * 1e-30
            else:
                blk = torch.randn(32, device=C.DEV, generator=g) * 1e4
            xb[r, b] = blk
    return x.to(torch.bfloat16)


def bitdiff(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


results = {"dims": dims, "arms": ARMS, "layers": LAYERS, "warm": WARM, "replays": REPLAYS,
           "sfb_rep_k2": SFB_REP_K2, "numerics": {}, "repeat": {}, "timing": {}}
all_pass = True

# ---------------- numerics (eager) ----------------
for M in MS:
    row = {}
    cases = {
        "randn": torch.randn(M, G * GW, device=C.DEV,
                             generator=torch.Generator(device="cuda").manual_seed(10 + M)).to(torch.bfloat16),
        "adv": adversarial(M, G * GW, M, 20 + M),
    }
    tmp_cases = {}
    for cname, x in cases.items():
        src = x.view(M, G, GW)
        outs = {}
        for an, arm in ARMS.items():
            tmp = W.empty_dense_gemm_mnl_view(M, R, G, device=C.DEV, dtype=torch.bfloat16)
            k2(src, w0, tmp, M, arm)
            tmp2 = W.empty_dense_gemm_mnl_view(M, R, G, device=C.DEV, dtype=torch.bfloat16)
            k2(src, w0, tmp2, M, arm)
            outs[an] = (tmp.clone(), bitdiff(tmp, tmp2))
        torch.cuda.synchronize()
        ref = outs["off"][0]
        tmp_cases[cname] = ref
        r = {}
        for an, (t, det) in outs.items():
            r[an] = {"bad_vs_off": bitdiff(t, ref), "nondet": det,
                     "finite": bool(torch.isfinite(t.float()).all()) if cname == "randn" else None}
            if r[an]["bad_vs_off"] or det:
                all_pass = False
        row[f"K2_{cname}"] = r
        log(f"M={M} K2 {cname}", json.dumps(r))
    # K3 on K2's output (randn) and on an adversarial group-major tmp
    adv_tmp = adversarial(M, G * R, M, 30 + M)  # [M, G*R] logical group-major rows
    tmp_adv = W.empty_dense_gemm_mnl_view(M, R, G, device=C.DEV, dtype=torch.bfloat16)
    tmp_adv.permute(0, 2, 1).copy_(adv_tmp.view(M, G, R))
    for cname, tmp in (("k2out", tmp_cases["randn"]), ("adv", tmp_adv)):
        outs = {}
        for an, arm in ARMS.items():
            o = torch.empty((M, H, 1), device=C.DEV, dtype=torch.bfloat16)
            k3(tmp, w0, o, M, arm)
            o2 = torch.empty((M, H, 1), device=C.DEV, dtype=torch.bfloat16)
            k3(tmp, w0, o2, M, arm)
            outs[an] = (o.clone(), bitdiff(o, o2))
        torch.cuda.synchronize()
        ref = outs["off"][0]
        r = {an: {"bad_vs_off": bitdiff(o, ref), "nondet": det} for an, (o, det) in outs.items()}
        for v in r.values():
            if v["bad_vs_off"] or v["nondet"]:
                all_pass = False
        row[f"K3_{cname}"] = r
        log(f"M={M} K3 {cname}", json.dumps(r))
    # sanity: K2 off vs an fp32 dequant reference (MXFP8 quant of A emulated by mxfp8_quantize)
    try:
        x = cases["randn"].view(M, G, GW).float()
        xb = x.view(M, G, GW // 32, 32)
        amax = xb.abs().amax(-1, keepdim=True)
        sc = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        sc = torch.where(amax == 0, torch.ones_like(sc), sc)
        xq_ = (xb / sc).to(torch.float8_e4m3fn).float() * sc
        xa = xq_.view(M, G, GW)
        wsc = w0.sa_pow2.repeat_interleave(128, 1).repeat_interleave(128, 2)  # [G, R, GW]
        wa = w0.wa_main.float() * wsc
        ref32 = torch.einsum("mgk,grk->mrg", xa, wa)
        row["K2_off_vs_fp32_ref"] = C.err(tmp_cases["randn"].float(), ref32)
    except Exception as e:  # sanity only
        row["K2_off_vs_fp32_ref"] = repr(e)
    log(f"M={M} K2 off vs fp32 ref", json.dumps(row["K2_off_vs_fp32_ref"]))
    results["numerics"][M] = row


# ---------------- graphs ----------------
def capture(fn_list):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_list[:2]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fn_list:
            f()
    torch.cuda.synchronize()
    return g


def time_graph(g, n):
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPLAYS):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / n)
    ts.sort()
    med = ts[len(ts) // 2]
    return {"us_per_layer": round(med, 3),
            "noise": round((ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med, 4)}


WAGG = int(os.environ.get("WAGG_REPLAYS", "0"))  # replays per arm at M>=2; 3x at M=1
AGG_ITERS = int(os.environ.get("AGG_ITERS", "8"))
results["wagg"] = {}
if WAGG:
    agg_streams = [torch.cuda.Stream() for _ in range(4)]
    agg_bufs = [torch.ones(64 * 1024 * 1024, device=C.DEV, dtype=torch.float32) for _ in range(4)]


def wagg_run(g, n, check):
    """Replay g n times, each overlapped by 4 streams of mul_ over 256 MB; check() -> bad elems."""
    bad_reps, worst = 0, 0
    k2_bad_reps, k3_only_bad_reps = 0, 0
    main = torch.cuda.current_stream()
    for _ in range(n):
        check(reset=True)
        ev = torch.cuda.Event()
        ev.record(main)
        for s, b in zip(agg_streams, agg_bufs):
            s.wait_event(ev)
            with torch.cuda.stream(s):
                for _ in range(AGG_ITERS):
                    b.mul_(1.0000001)
        g.replay()
        torch.cuda.synchronize()
        nb2, nb3 = check(reset=False)
        if nb2 + nb3:
            bad_reps += 1
            worst = max(worst, nb2 + nb3)
        if nb2:
            k2_bad_reps += 1
        elif nb3:
            k3_only_bad_reps += 1
    for b in agg_bufs:
        b.fill_(1.0)
    return {"replays": n, "bad_replays": bad_reps, "max_bad_elems": worst,
            "k2_bad_replays": k2_bad_reps, "k3_only_bad_replays": k3_only_bad_reps}


for M in MS:
    o3, pos = C.make_inputs(dims, M, seed=100 + M)
    xq = W.empty_mxfp8_rows_for_dense_gemm(M, GW, num_groups=G, device=C.DEV)
    rope = torch.empty(M, G * GW, device=C.DEV, dtype=torch.bfloat16)
    src = rope.view(M, G, GW)
    tmps = [W.empty_dense_gemm_mnl_view(M, R, G, device=C.DEV, dtype=torch.bfloat16) for _ in range(LAYERS)]
    outs = [torch.empty((M, H, 1), device=C.DEV, dtype=torch.bfloat16) for _ in range(LAYERS)]
    alpha = W._cached_alpha_one(C.DEV)

    def k1(store=True):
        B.inv_rope_quant128(o3, pos, cs, xq, groups=G, group_width=GW, nope_dim=dims["nope"],
                            rope_dim=dims["rope"], rope_out=rope if store else None)

    # eager reference per layer (off)
    k1()
    refs_tmp, refs_out = [], []
    for i in range(LAYERS):
        k2(src, ws[i], tmps[i], M, ARMS["off"])
        k3(tmps[i], ws[i], outs[i], M, ARMS["off"])
        refs_tmp.append(tmps[i].clone())
        refs_out.append(outs[i].clone())
    o3_ref, rope_ref = o3.clone(), rope.clone()
    torch.cuda.synchronize()
    rep = {}
    tim = {}
    for an, arm in ARMS.items():
        def chain_fns(arm=arm):
            fns = []
            arm2 = ARMS[K2_FIXED] if K2_FIXED else arm
            for i in range(LAYERS):
                def f(i=i):
                    k1()
                    k2(src, ws[i], tmps[i], M, arm2)
                    k3(tmps[i], ws[i], outs[i], M, arm)
                fns.append(f)
            return fns
        g = capture(chain_fns())
        # untimed repeated sequence, same warmup/replay counts as timing
        bad_iters, max_bad = 0, 0
        for it in range(WARM + REPLAYS):
            for t in tmps:
                t.fill_(float("nan"))
            for o in outs:
                o.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            nb = sum(bitdiff(outs[i], refs_out[i]) + bitdiff(tmps[i], refs_tmp[i]) for i in range(LAYERS))
            if nb:
                bad_iters += 1
                max_bad = max(max_bad, nb)
        inputs_ok = bool(torch.equal(o3, o3_ref)) and bool(torch.equal(rope, rope_ref))
        rep[an] = {"iters": WARM + REPLAYS, "bad_iters": bad_iters, "max_bad_elems": max_bad,
                   "inputs_unmodified": inputs_ok}
        if bad_iters or not inputs_ok:
            all_pass = False
        log(f"M={M} repeat {an}", json.dumps(rep[an]))
        if WAGG:
            def check(reset, refs_o=refs_out, refs_t=refs_tmp):
                if reset:
                    for t in tmps:
                        t.fill_(float("nan"))
                    for o in outs:
                        o.fill_(float("nan"))
                    return 0
                return (sum(bitdiff(tmps[i], refs_t[i]) for i in range(LAYERS)),
                        sum(bitdiff(outs[i], refs_o[i]) for i in range(LAYERS)))
            wr = wagg_run(g, WAGG * (3 if M == 1 else 1), check)
            results["wagg"].setdefault(M, {})[an] = wr
            if wr["bad_replays"]:
                all_pass = False
            log(f"M={M} WAGG {an}", json.dumps(wr))
        if DO_TIMING:
            tim[f"chain_{an}"] = time_graph(g, LAYERS)
            del g
            g2 = capture([lambda i=i, arm=arm: k2(src, ws[i], tmps[i], M, arm) for i in range(LAYERS)])
            tim[f"K2_{an}"] = time_graph(g2, LAYERS)
            del g2
            g3 = capture([lambda i=i, arm=arm: k3(tmps[i], ws[i], outs[i], M, arm) for i in range(LAYERS)])
            tim[f"K3_{an}"] = time_graph(g3, LAYERS)
            del g3
            log(f"M={M} timing {an}", tim[f"chain_{an}"], tim[f"K2_{an}"], tim[f"K3_{an}"])
        else:
            del g
    if WAGG:
        # positive control: main's racy TMA-A K2 (V0) chain on the same slot
        def v0_f(i):
            k1(store=False)
            W.wo_a_dense_gemm_mxfp8(xq, ws[i].wa_b12x, out=tmps[i], alpha=alpha, expected_m=M,
                                    sfb_k_replicated=True)
            k3(tmps[i], ws[i], outs[i], M, ARMS["off"])
        for i in range(LAYERS):
            v0_f(i)
        v0_tmp = [t.clone() for t in tmps]
        v0_out = [o.clone() for o in outs]
        torch.cuda.synchronize()
        g = capture([lambda i=i: v0_f(i) for i in range(LAYERS)])

        def check_v0(reset):
            if reset:
                for t in tmps:
                    t.fill_(float("nan"))
                for o in outs:
                    o.fill_(float("nan"))
                return 0
            return (sum(bitdiff(tmps[i], v0_tmp[i]) for i in range(LAYERS)),
                    sum(bitdiff(outs[i], v0_out[i]) for i in range(LAYERS)))
        wr = wagg_run(g, WAGG * (3 if M == 1 else 1), check_v0)
        del g
        results["wagg"].setdefault(M, {})["V0_tma_a_control"] = wr
        log(f"M={M} WAGG V0 control", json.dumps(wr))
    if DO_TIMING:
        # old racy TMA-A chain (K1 E4M3+SF -> wo_a_dense_gemm_mxfp8 -> K3 off), timing only
        def old_f(i):
            k1(store=False)
            W.wo_a_dense_gemm_mxfp8(xq, ws[i].wa_b12x, out=tmps[i], alpha=alpha, expected_m=M,
                                    sfb_k_replicated=True)
            k3(tmps[i], ws[i], outs[i], M, ARMS["off"])
        g = capture([lambda i=i: old_f(i) for i in range(LAYERS)])
        tim["chain_old_tma_a"] = time_graph(g, LAYERS)
        del g

        # main's chain with only K3 switched (K3-only landing): K1 -> TMA-A K2 -> K3 arm
        for an, arm in ARMS.items():
            def mk3(i, arm=arm):
                k1(store=False)
                W.wo_a_dense_gemm_mxfp8(xq, ws[i].wa_b12x, out=tmps[i], alpha=alpha, expected_m=M,
                                        sfb_k_replicated=True)
                k3(tmps[i], ws[i], outs[i], M, arm)
            g = capture([lambda i=i: mk3(i) for i in range(LAYERS)])
            tim[f"chain_mainK2_K3{an}"] = time_graph(g, LAYERS)
            del g
        g = capture([lambda: k1() for _ in range(LAYERS)])
        tim["K1_store_bf16"] = time_graph(g, LAYERS)
        del g
        # second pass of off and best arms to check drift
        for an in ("off",):
            def f2(i, arm=ARMS[an]):
                k1()
                k2(src, ws[i], tmps[i], M, arm)
                k3(tmps[i], ws[i], outs[i], M, arm)
            g = capture([lambda i=i: f2(i) for i in range(LAYERS)])
            tim[f"chain_{an}_repeat"] = time_graph(g, LAYERS)
            del g
        log(f"M={M} timing", json.dumps(tim))
    results["repeat"][M] = rep
    results["timing"][M] = tim

summary = {"passed": all_pass}
if WAGG:
    summary["wagg_bad_replays"] = {M: {a: v["bad_replays"] for a, v in r.items()} for M, r in results["wagg"].items()}
if DO_TIMING:
    summary["chain_us_per_layer"] = {
        M: {k[len("chain_"):]: v["us_per_layer"] for k, v in results["timing"][M].items() if k.startswith("chain_")}
        for M in MS}
    summary["K2_us"] = {M: {k[3:]: v["us_per_layer"] for k, v in results["timing"][M].items() if k.startswith("K2_")}
                        for M in MS}
    summary["K3_us"] = {M: {k[3:]: v["us_per_layer"] for k, v in results["timing"][M].items() if k.startswith("K3_")}
                        for M in MS}
out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
with open(os.path.join(out_dir, "fq_pipe_screen.json"), "w") as f:
    json.dump(results, f, indent=1, default=str)
print(json.dumps(summary, default=str))
