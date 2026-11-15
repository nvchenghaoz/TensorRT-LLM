"""Gemma4 fused QKV prep with a direct FP8 paged-KV store (idea 23, stage 2).

Modes:
  bench : per layer shape (sliding d256, global d512 K_EQ_V) and token mix,
          times the tree's per-layer "prep + KV write" on a sentinel-filled
          FP8 paged pool with shuffled page tables:
            main (no paged_kv_cache kwarg): Route A prep (BF16 Q + FP8 K/V
              temporaries) -> .to(fp8) no-op -> flashinfer append kernel
              (flashinfer.py _append_paged_kv_cache);
            branch: the prep storing K/V straight into the pool.
          The branch arm also times Route A in the same job (diag_routeA).
  check : (branch only) byte-compares the whole pool and Q of the direct store
          against Route A + append on the same inputs, for NHD, HND and a
          strided-slot pool view, contiguous and nonzero-start (cached
          prefix) packs; checks unwritten bytes stay 0x3C, the untimed
          repeated sequence (WARMUP + ITERS calls, no reset), an A,B,A shape
          permutation, and the FP32 reference error of the written K/V.

Shapes come from <checkpoint>/config.json. Synthetic tensors only.
"""

import argparse
import inspect
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gemma4_qkv_prep_bench as gb  # noqa: E402

WARMUP = 10
ITERS = 50
PAGE_SIZE = 32  # KvCacheConfig.tokens_per_block default, as main runs it
SENTINEL = 0x3C

# (label, new tokens per request, cached prefix per request)
CASES = [
    ("300", [300], None),
    ("1000", [1000], None),
    ("2000", [2000], None),
    ("4000", [4000], None),
    ("8000", [8000], None),
    ("4x300", [300] * 4, None),
    ("16x300", [300] * 16, None),
    ("27x300", [300] * 27, None),
    ("4x1000", [1000] * 4, None),
    ("8x1000", [1000] * 8, None),
    ("4x2000", [2000] * 4, None),
    ("2x4000", [4000] * 2, None),
]
CHECK_EXTRA = [
    ("prefix_pack", [300, 517, 1, 1000], [100, 37, 64, 2000]),
    ("1", [1], None),
    ("3x333_7", [333, 333, 333, 7], [5, 0, 31, 32]),
]


def log(*a):
    print(*a, flush=True)


def has_paged():
    from tensorrt_llm._torch.modules.gemma4.fused_qkv import gemma4_fused_qkv_norm_rope_quant
    return "paged_kv_cache" in inspect.signature(gemma4_fused_qkv_norm_rope_quant).parameters


def build_tables(seqs, prefixes, page_size, seed, dev, spare_pages=37):
    import flashinfer
    prefixes = prefixes or [0] * len(seqs)
    g = torch.Generator().manual_seed(seed)
    kv_lens = [p + s for p, s in zip(prefixes, seqs)]
    npages = [math.ceil(n / page_size) for n in kv_lens]
    total = sum(npages)
    pool_pages = total + spare_pages
    indices = torch.randperm(pool_pages, generator=g)[:total].to(torch.int32)
    indptr = torch.tensor([0] + list(torch.tensor(npages).cumsum(0).tolist()), dtype=torch.int32)
    last = torch.tensor([n - (p - 1) * page_size for n, p in zip(kv_lens, npages)], dtype=torch.int32)
    qo = torch.tensor([0] + list(torch.tensor(seqs).cumsum(0).tolist()), dtype=torch.int32)
    nnz = sum(seqs)
    indices, indptr, last, qo = (t.to(dev) for t in (indices, indptr, last, qo))
    seq_lens = flashinfer.get_seq_lens(indptr, last, page_size)
    bidx, pos = flashinfer.get_batch_indices_positions(qo, seq_lens, nnz)
    return dict(indices=indices.contiguous(), indptr=indptr.contiguous(), last=last.contiguous(),
                bidx=bidx.to(torch.int32).contiguous(), pos=pos.to(torch.int32).contiguous(),
                pool_pages=pool_pages, nnz=nnz)


def make_pool(pool_pages, nkv, hd, page_size, layout, dev, strided=False):
    kv = 3 if strided else 2
    if layout == "NHD":
        shape = [pool_pages, kv, page_size, nkv, hd]
    else:
        shape = [pool_pages, kv, nkv, page_size, hd]
    base = torch.empty(shape, dtype=torch.uint8, device=dev)
    base.fill_(SENTINEL)
    view = base.view(torch.float8_e4m3fn)
    if strided:
        view = view[:, :2]
    return base, view


def route_a(layer, qkv, pos, tb, pool, layout):
    from tensorrt_llm._torch.attention.backends.flashinfer import _append_paged_kv_cache
    q, k, v = layer.fused_chain(qkv, pos)
    _append_paged_kv_cache(append_key=k, append_value=v, batch_indices=tb["bidx"],
                           positions=tb["pos"], paged_kv_cache=pool, kv_indices=tb["indices"],
                           kv_indptr=tb["indptr"], kv_last_page_len=tb["last"], kv_layout=layout)
    return q


def direct(layer, qkv, pos, tb, pool, layout):
    from tensorrt_llm._torch.modules.gemma4.fused_qkv import gemma4_fused_qkv_norm_rope_quant
    q, k, v = gemma4_fused_qkv_norm_rope_quant(
        qkv, pos, layer.cos_sin, layer.q_norm.weight, layer.k_norm.weight, layer.eps,
        layer.nq, layer.nkv, layer.hd, out_fp8=False, packed_output=False, kv_out_fp8=True,
        cute_norm_order=True, k_eq_v=layer.k_eq_v, paged_kv_cache=pool,
        kv_batch_indices=tb["bidx"], kv_positions=tb["pos"], kv_indptr=tb["indptr"],
        kv_indices=tb["indices"], kv_layout=layout)
    assert k is None and v is None
    return q


def time_fn(fn, base):
    # Stateful op: sentinel-fill the pool once before the timed sequence; every
    # iteration rewrites the same slots (page tables rebuilt by the caller).
    base.fill_(SENTINEL)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    med = ts[len(ts) // 2]
    p10, p90 = ts[int(0.1 * len(ts))], ts[int(0.9 * len(ts))]
    return dict(kernel_ms=med, p10_ms=p10, p90_ms=p90, noise_floor=(p90 - p10) / med)


def byte_hashes(fn, base):
    """sha256 of the whole pool and of Q after one call on a fresh sentinel pool;
    equal across the pair's arms means main's Route A + append and the branch
    wrote the same bytes."""
    import hashlib
    base.fill_(SENTINEL)
    q = fn()
    torch.cuda.synchronize()
    hp = hashlib.sha256(base.cpu().numpy().tobytes()).hexdigest()[:16]
    hq = hashlib.sha256(q.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
    return {"pool_sha": hp, "q_sha": hq}


def bench(ckpt):
    dev = torch.device("cuda")
    log("device:", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    shapes, eps = gb.load_shapes(ckpt)
    paged = has_paged()
    log("tree:", "branch (direct paged store)" if paged else "main (Route A + append)",
        "page_size", PAGE_SIZE, "layout NHD")
    out = {}
    diag = {}
    for li, (lname, sh) in enumerate(shapes.items()):
        layer = gb.Layer(sh, eps, dev, seed=100 + li)
        for cname, seqs, pref in CASES:
            qkv, _ = layer.make_inputs(seqs, dev, seed=1)
            tb = build_tables(seqs, pref, PAGE_SIZE, seed=7 + li, dev=dev)
            pos = tb["pos"]
            base, pool = make_pool(tb["pool_pages"], layer.nkv, layer.hd, PAGE_SIZE, "NHD", dev)
            key = f"{lname}_{cname}"
            prim = (lambda: direct(layer, qkv, pos, tb, pool, "NHD")) if paged else (
                lambda: route_a(layer, qkv, pos, tb, pool, "NHD"))
            if paged:
                r = time_fn(prim, base)
                r.update(byte_hashes(prim, base))
                ra = time_fn(lambda: route_a(layer, qkv, pos, tb, pool, "NHD"), base)
                diag[key] = {"routeA_ms": ra["kernel_ms"], "direct_ms": r["kernel_ms"],
                             "saved_us": (ra["kernel_ms"] - r["kernel_ms"]) * 1e3}
                log(f"{key}: n={tb['nnz']} direct={r['kernel_ms']*1e3:.1f}us "
                    f"(p10 {r['p10_ms']*1e3:.1f} p90 {r['p90_ms']*1e3:.1f}) "
                    f"routeA+append={ra['kernel_ms']*1e3:.1f}us")
            else:
                r = time_fn(prim, base)
                r.update(byte_hashes(prim, base))
                log(f"{key}: n={tb['nnz']} routeA+append={r['kernel_ms']*1e3:.1f}us "
                    f"(p10 {r['p10_ms']*1e3:.1f} p90 {r['p90_ms']*1e3:.1f})")
            out[key] = r
            del base, pool
    res = {"batch": out, "output": {"class": "exact"}}
    if diag:
        res["diag_same_job"] = diag
    print(json.dumps(res))


def written_mask(tb, page_size, pool_pages, dev):
    flat = tb["indptr"].long()[tb["bidx"].long()] * page_size + tb["pos"].long()
    page = tb["indices"].long()[flat // page_size]
    entry = flat % page_size
    m = torch.zeros(pool_pages, page_size, dtype=torch.bool, device=dev)
    m[page, entry] = True
    return m, page, entry


def gather_kv(pool, page, entry, layout):
    # -> [N, 2, nkv, hd] fp8
    if layout == "NHD":
        return pool[page, :, entry]
    return pool.permute(0, 1, 3, 2, 4)[page, :, entry]


def unwritten_ok(base, m, layout, strided):
    b = base[:, :2] if strided else base
    if layout == "HND":
        b = b.permute(0, 1, 3, 2, 4)
    sel = b.permute(0, 2, 1, 3, 4)[~m]  # [rows, 2, nkv, hd]
    ok = bool((sel == SENTINEL).all().item()) if sel.numel() else True
    extra = bool((base[:, 2] == SENTINEL).all().item()) if strided else True
    return ok and extra


def check(ckpt):
    dev = torch.device("cuda")
    shapes, eps = gb.load_shapes(ckpt)
    assert has_paged(), "check needs the branch's paged_kv_cache mode"
    passed = True
    report = {}
    variants = [("NHD", False, PAGE_SIZE), ("HND", False, PAGE_SIZE), ("NHD", True, PAGE_SIZE),
                ("NHD", False, 64)]
    for li, (lname, sh) in enumerate(shapes.items()):
        layer = gb.Layer(sh, eps, dev, seed=100 + li)
        for cname, seqs, pref in CASES + CHECK_EXTRA:
            for layout, strided, ps in variants:
                if (layout, strided, ps) != ("NHD", False, PAGE_SIZE) and cname not in (
                        "8000", "16x300", "4x2000", "2x4000", "prefix_pack", "3x333_7"):
                    continue
                key = f"{lname}_{cname}_{layout}{'_strided' if strided else ''}_ps{ps}"
                qkv, _ = layer.make_inputs(seqs, dev, seed=1)
                tb = build_tables(seqs, pref, ps, seed=7 + li, dev=dev)
                pos = tb["pos"]
                snap_in = [qkv.clone()] + [tb[k].clone() for k in ("bidx", "pos", "indptr", "indices")]
                # Reference: Route A + FlashInfer append on a sentinel pool.
                base_r, pool_r = make_pool(tb["pool_pages"], layer.nkv, layer.hd, ps, layout, dev, strided)
                q_ref = route_a(layer, qkv, pos, tb, pool_r, layout).clone()
                # Candidate: untimed sequence matching timing (WARMUP + ITERS, no reset).
                base_c, pool_c = make_pool(tb["pool_pages"], layer.nkv, layer.hd, ps, layout, dev, strided)
                rep_ok = True
                for i in range(WARMUP + ITERS):
                    q = direct(layer, qkv, pos, tb, pool_c, layout)
                    if not torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)):
                        rep_ok = False
                    if i == 0:
                        first_pool_ok = torch.equal(base_c, base_r)
                torch.cuda.synchronize()
                pool_eq = torch.equal(base_c, base_r)
                n_diff = int((base_c != base_r).sum().item())
                inp_ok = all(torch.equal(a, b) for a, b in zip(
                    snap_in, [qkv] + [tb[k] for k in ("bidx", "pos", "indptr", "indices")]))
                m, page, entry = written_mask(tb, ps, tb["pool_pages"], dev)
                unw_ok = unwritten_ok(base_c, m, layout, strided) and unwritten_ok(base_r, m, layout, strided)
                # FP32 reference (b) on the written slots.
                _, rk, rv = layer.fp32_ref(qkv, pos)
                got = gather_kv(pool_c, page, entry, layout).float()
                mref = gather_kv(pool_r, page, entry, layout).float()
                ck = gb.stats(got[:, 0], rk)
                cv = gb.stats(got[:, 1], rv)
                mk = gb.stats(mref[:, 0], rk)
                mv = gb.stats(mref[:, 1], rv)
                finite = bool(torch.isfinite(got).all().item())
                # Main's unfused chain (rmsnorm -> flashinfer rope -> .to(fp8)), which
                # Route A matches bitwise (fact 631): written K/V bytes and Q bytes.
                uq, uk, uv = layer.main_chain(qkv, pos)
                gb8 = gather_kv(pool_c, page, entry, layout).contiguous().view(torch.uint8)
                unf_ok = (torch.equal(gb8[:, 0], uk.contiguous().view(torch.uint8))
                          and torch.equal(gb8[:, 1], uv.contiguous().view(torch.uint8))
                          and torch.equal(q.view(torch.uint8), uq.contiguous().view(torch.uint8)))
                ok = pool_eq and first_pool_ok and rep_ok and inp_ok and unw_ok and finite and unf_ok
                passed &= ok
                report[key] = dict(ok=ok, pool_bytes_equal=pool_eq, pool_diff_bytes=n_diff,
                                   first_call_pool_equal=first_pool_ok, q_repeat_bitwise_vs_main=rep_ok,
                                   inputs_unchanged=inp_ok, unwritten_0x3C=unw_ok, finite=finite,
                                   bytes_equal_main_unfused_chain=unf_ok,
                                   written_rows=int(m.sum().item()),
                                   k_vs_fp32=ck, v_vs_fp32=cv, main_k_vs_fp32=mk, main_v_vs_fp32=mv)
                log(key, json.dumps(report[key]))
                del base_r, pool_r, base_c, pool_c
        # A, B, A permutation across two shapes (fresh sentinel pools each call).
        aba_ok = True
        seq_a, seq_b = ([8000], None), ([300] * 16, None)
        results = {}
        for tag, (seqs, pref) in (("A", seq_a), ("B", seq_b), ("A2", seq_a)):
            qkv, _ = layer.make_inputs(seqs, dev, seed=1)
            tb = build_tables(seqs, pref, PAGE_SIZE, seed=7 + li, dev=dev)
            base_r, pool_r = make_pool(tb["pool_pages"], layer.nkv, layer.hd, PAGE_SIZE, "NHD", dev)
            route_a(layer, qkv, tb["pos"], tb, pool_r, "NHD")
            base_c, pool_c = make_pool(tb["pool_pages"], layer.nkv, layer.hd, PAGE_SIZE, "NHD", dev)
            q = direct(layer, qkv, tb["pos"], tb, pool_c, "NHD")
            results[tag] = (base_c.clone(), q.clone())
            aba_ok &= torch.equal(base_c, base_r)
        aba_ok &= torch.equal(results["A"][0], results["A2"][0]) and torch.equal(
            results["A"][1].view(torch.uint8), results["A2"][1].view(torch.uint8))
        report[f"{lname}_ABA"] = dict(ok=aba_ok)
        log(f"{lname}_ABA", aba_ok)
        passed &= aba_ok
    print(json.dumps({
        "passed": bool(passed),
        "compared": ("direct FP8 paged-KV store by the fused prep vs Route A prep + FlashInfer "
                     "append_paged_kv_cache (main), whole-pool bytes and Q bytes; sentinel 0x3C on "
                     "unwritten slots; FP32 torch reference on written K/V; "
                     f"{WARMUP + ITERS} repeated calls without reset; A,B,A permutation"),
        "failed": [k for k, r in report.items() if not r["ok"]],
        "cases": len(report),
        "m8000": {k: r for k, r in report.items() if "_8000_" in k},
    }))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "check"])
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    with torch.inference_mode():
        if a.mode == "bench":
            bench(a.ckpt)
        else:
            check(a.ckpt)
