# Idea 164: GDN prefill state I/O BLOCK_V=32 (fla/fused_state_io.py), exact.
# Brace-free on purpose: run_gpu_pair passes this file verbatim via python -u -c,
# because the base tree lacks experiments/. In the base tree (no
# _STATE_IO_BLOCK_V) only the default-config timing runs. In the candidate tree
# it also runs: Triton cache count, bitwise checks (cand vs base=128 vs torch
# reference, 10+50 repeats from pristine), and an in-process seeded-shuffle
# A/A/B timing (base_a, base_b at 128 = main's code path; cand at 32).
import json, os, random, subprocess, sys, time
import torch

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:  # noqa: BLE001
        return "unknown (%s)" % e


TREE = sh("git log -1 --format=%T")
DIRTY = sh("git status --porcelain | wc -l")
print("TREE_COMMIT tree=%s dirty=%s" % (TREE, DIRTY), flush=True)
for k_, v_ in sorted(os.environ.items()):
    if any(t in k_.upper() for t in ("RECEIPT", "COMMIT", "ASCENT_SOURCE", "SOURCE_REV")):
        print("ENV", k_, "=", v_, flush=True)

import tensorrt_llm._torch.modules.fla.fused_state_io as M  # noqa: E402

HAS = hasattr(M, "_STATE_IO_BLOCK_V")
DEFAULT_BV = getattr(M, "_STATE_IO_BLOCK_V", None)
log("module", M.__file__, "HAS_BLOCK_V_GLOBAL", HAS, "default", DEFAULT_BV)
gather = M.gather_cast_vk_to_fp32_vk
scatter = M.cast_scatter_fp32_vk_to_vk

L, NB, H, V, K, CAP = 48, 16, 48, 128, 128, 8
SSM_E = H * V * K  # 786,432 bf16 elements per slot
STRIDE_E = 817152  # production slot stride incl. conv gap
SLOTS = [3, 7, 1, 10]
SIZES = [1, 2, 3, 4]
WARM, ITERS = 10, 50
SLEEP_CYCLES = 8000000
dev = torch.device("cuda")
torch.manual_seed(0)

buf = torch.empty(L, NB, STRIDE_E, dtype=torch.bfloat16, device=dev)
for l in range(L):
    buf[l].copy_(torch.randn(NB, STRIDE_E, device=dev).to(torch.bfloat16))
pools = [buf[l, :, :SSM_E].view(NB, H, V, K) for l in range(L)]
assert pools[0].stride(0) == STRIDE_E and pools[0].stride(3) == 1
pristine = buf.clone()
ws0 = torch.zeros(CAP, H, V, K, dtype=torch.float32, device=dev)
ws1 = torch.randn(CAP, H, V, K, dtype=torch.float32, device=dev)
idxs = dict((n, torch.tensor(SLOTS[:n], dtype=torch.int32, device=dev)) for n in SIZES)
flush_src = torch.randn(64 * 1024 * 1024, dtype=torch.float32, device=dev)  # 256 MB READ flush
torch.cuda.synchronize()
log("pool", tuple(buf.shape), "ready; per-layer pool stride(0)", pools[0].stride(0))


def set_bv(bv):
    if HAS:
        M._STATE_IO_BLOCK_V = bv


def window(n):
    idx = idxs[n]
    w0 = ws0[:n]
    w1 = ws1[:n]
    for l in range(L):
        p = pools[l]
        gather(p, idx, out_dtype=torch.float32, output=w0)
        scatter(w1, p, idx)


SINK = []


def time_one(n):
    for s in SLOTS[:n]:
        buf[:, s].copy_(pristine[:, s])
    SINK.append(flush_src.sum())
    if len(SINK) > 8:
        SINK.pop(0)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(SLEEP_CYCLES)  # hide host enqueue; GPU-side window only
    e0.record()
    h0 = time.perf_counter()
    window(n)
    h1 = time.perf_counter()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1), (h1 - h0) * 1000.0


def q3(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)).tolist()
    return dict(p10=q[0], median=q[1], p90=q[2])


# ---------------------------------------------------------------- cache count
def describe(kfn, ck):
    parts = []
    try:
        cs = ck.src.constants
        names = kfn.arg_names
        for k2, v2 in cs.items():
            if isinstance(k2, tuple) and len(k2) == 1:
                nm = names[k2[0]]
            elif isinstance(k2, int):
                nm = names[k2]
            else:
                nm = str(k2)
            if nm in ("BLOCK_V", "BLOCK_K", "HAS_INDICES"):
                parts.append("%s=%s" % (nm, v2))
    except Exception as e:  # noqa: BLE001
        parts.append("constants?(%s)" % type(e).__name__)
    try:
        parts.append("num_warps=%s" % ck.metadata.num_warps)
    except Exception:  # noqa: BLE001
        pass
    return ",".join(sorted(parts))


def cache_info(kfn):
    dc = getattr(kfn, "device_caches", None)
    if dc is None:
        return -1, ["no device_caches"]
    n = 0
    desc = []
    for d in list(dc.keys()):
        ent = dc[d]
        kc = ent[0] if isinstance(ent, tuple) else ent
        try:
            for _key, ck in kc.items():
                n += 1
                desc.append(describe(kfn, ck))
        except Exception as e:  # noqa: BLE001
            desc.append("introspect error %s" % e)
    return n, sorted(desc)


def exercise_all(bv):
    set_bv(bv)
    fresh = torch.empty(CAP, H, V, K, dtype=torch.bfloat16, device=dev)
    for n in SIZES:
        p = pools[0]
        gather(p, idxs[n], out_dtype=torch.float32, output=ws0[:n])
        scatter(ws1[:n], p, idxs[n])
        gather(p[:n], None, out_dtype=torch.float32, output=ws0[:n])
        scatter(ws1[:n], p[:n], None)
        scatter(ws1[:n], fresh[:n], None)
    torch.cuda.synchronize()
    buf.copy_(pristine)


def run_cache_count():
    kg = M._gather_cast_vk_to_fp32_vk_kernel
    ks = M._cast_scatter_fp32_vk_to_vk_kernel
    exercise_all(128)
    gb, gbd = cache_info(kg)
    sb, sbd = cache_info(ks)
    exercise_all(32)
    gc, gcd = cache_info(kg)
    scn, scd = cache_info(ks)
    res = dict(gather_after_base=gb, gather_after_cand=gc, gather_added=gc - gb,
               scatter_after_base=sb, scatter_after_cand=scn, scatter_added=scn - sb,
               gather_entries=gcd, scatter_entries=scd)
    res["passed"] = bool(gb > 0 and sb > 0 and gc - gb == gb and scn - sb == sb)
    log("triton cache", res)
    return res


# ---------------------------------------------------------------- bitwise check
def iv(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def eqb(a, b):
    return bool(torch.equal(iv(a), iv(b)))


WS0A = WS1A = FRESH = None
FRESH_FILL = -3.0


def check_buffers():
    global WS0A, WS1A, FRESH
    WS0A = torch.zeros(L, CAP, H, V, K, dtype=torch.float32, device=dev)
    WS1A = torch.empty(L, CAP, H, V, K, dtype=torch.float32, device=dev)
    for l in range(L):
        WS1A[l].normal_()
    FRESH = torch.empty(L, CAP, H, V, K, dtype=torch.bfloat16, device=dev)


def reset_all():
    buf.copy_(pristine)
    WS0A.zero_()
    FRESH.fill_(FRESH_FILL)


def run_case(n, mode):
    for l in range(L):
        p = pools[l]
        if mode == "idx":
            gather(p, idxs[n], out_dtype=torch.float32, output=WS0A[l, :n])
            scatter(WS1A[l, :n], p, idxs[n])
        else:
            gather(p[:n], None, out_dtype=torch.float32, output=WS0A[l, :n])
            scatter(WS1A[l, :n], p[:n], None)
        scatter(WS1A[l, :n], FRESH[l, :n], None)


def snap():
    torch.cuda.synchronize()
    return (buf.clone(), WS0A.clone(), FRESH.clone())


def same_as(s):
    return eqb(buf, s[0]) and eqb(WS0A, s[1]) and eqb(FRESH, s[2])


def ref_ok(s, n, mode):
    b_buf, b_ws0, b_fresh = s
    rows = SLOTS[:n] if mode == "idx" else list(range(n))
    rt = torch.tensor(rows, dtype=torch.long, device=dev)
    ok = True
    for l in range(L):
        pp = pristine[l, :, :SSM_E].view(NB, H, V, K)
        ok &= eqb(b_ws0[l, :n], pp.index_select(0, rt).float())
        ok &= bool((b_ws0[l, n:] == 0).all().item())
        e = pristine[l].clone()
        ev = e[:, :SSM_E].view(NB, H, V, K)
        ev.index_copy_(0, rt, WS1A[l, :n].to(torch.bfloat16))
        ok &= eqb(b_buf[l], e)
        ok &= eqb(b_fresh[l, :n], WS1A[l, :n].to(torch.bfloat16))
        ok &= bool((b_fresh[l, n:] == FRESH_FILL).all().item())
    return bool(ok)


def run_checks():
    check_buffers()
    cases = []
    for n in SIZES:
        for mode in ("idx", "none"):
            reset_all()
            set_bv(128)
            run_case(n, mode)
            base = snap()
            base_ref = ref_ok(base, n, mode)
            set_bv(DEFAULT_BV)
            first = None
            drift = 0
            mism = 0
            for it in range(WARM + ITERS):
                reset_all()
                SINK.append(flush_src.sum())
                SINK.pop(0) if len(SINK) > 8 else None
                run_case(n, mode)
                torch.cuda.synchronize()
                if first is None:
                    first = snap()
                elif not same_as(first):
                    drift += 1
                if not same_as(base):
                    mism += 1
            del first, base
            c = dict(n=n, mode=mode, iters=WARM + ITERS, base_eq_torch_ref=base_ref,
                     cand_drift_vs_iter1=drift, cand_mismatch_vs_base=mism,
                     passed=bool(base_ref and drift == 0 and mism == 0))
            cases.append(c)
            log("check", c)
    torch.cuda.empty_cache()
    return dict(cases=cases, passed=all(c["passed"] for c in cases))


# ---------------------------------------------------------------- timing
def run_ab():
    out = dict()
    for n in SIZES:
        rng = random.Random(1640 + n)
        times = dict(base_a=[], base_b=[], cand=[])
        host = []
        for r in range(WARM + ITERS):
            order = ["base_a", "base_b", "cand"]
            rng.shuffle(order)
            for a in order:
                set_bv(DEFAULT_BV if a == "cand" else 128)
                t, h = time_one(n)
                if r >= WARM:
                    times[a].append(t)
                    host.append(h)
        sa, sb, sc = q3(times["base_a"]), q3(times["base_b"]), q3(times["cand"])
        aa = abs(sb["median"] / sa["median"] - 1.0)
        base_med = 0.5 * (sa["median"] + sb["median"])
        gain = 1.0 - sc["median"] / sa["median"]
        sep = sc["p90"] < min(sa["p10"], sb["p10"])
        out[str(n)] = dict(base_a_ms=sa, base_b_ms=sb, cand_ms=sc, aa_gap=aa,
                           gain_vs_base_a=gain, gain_vs_base_mean=1.0 - sc["median"] / base_med,
                           cand_p90_lt_base_p10=sep, host_enqueue_ms_median=q3(host)["median"])
        log("AB n=%d" % n, out[str(n)])
    set_bv(DEFAULT_BV)
    g1 = out["1"]
    pass1 = g1["gain_vs_base_a"] > 0.04 and g1["gain_vs_base_a"] > g1["aa_gap"] and g1["cand_p90_lt_base_p10"]
    not_slower = all(out[k]["cand_ms"]["median"] <= max(out[k]["base_a_ms"]["median"], out[k]["base_b_ms"]["median"])
                     for k in ("2", "3", "4"))
    return dict(sizes=out, pass_n1=bool(pass1), not_slower_2_3_4=bool(not_slower), passed=bool(pass1 and not_slower))


def run_default_timing():
    batch = dict()
    extra = dict()
    for n in SIZES:
        ts = []
        for it in range(WARM + ITERS):
            t, _h = time_one(n)
            if it >= WARM:
                ts.append(t)
        s = q3(ts)
        rec = dict(kernel_ms=s["median"], noise_floor=(s["p90"] - s["p10"]) / s["median"],
                   p10_ms=s["p10"], p90_ms=s["p90"])
        log("default-config n=%d block_v=%s" % (n, DEFAULT_BV if HAS else "128 (stock)"), rec)
        if n in (1, 2, 4):
            batch[str(n)] = rec
        else:
            extra[str(n)] = rec
    return batch, extra


result = dict(tree=TREE, dirty=DIRTY, has_block_v_global=HAS, default_block_v=DEFAULT_BV)
passed = True
if HAS:
    result["cache"] = run_cache_count()
    result["checks"] = run_checks()
    result["ab"] = run_ab()
    passed = result["cache"]["passed"] and result["checks"]["passed"]
    log("BITWISE", "PASS" if result["checks"]["passed"] else "FAIL",
        "| AB", "PASS" if result["ab"]["passed"] else "FAIL",
        "| CACHE", "PASS" if result["cache"]["passed"] else "FAIL")
batch, extra = run_default_timing()
result["batch"] = batch
result["extra_sizes"] = extra
result["passed"] = bool(passed)
result["output"] = dict([("class", "exact")])
print(json.dumps(result), flush=True)
