"""Untimed correctness for the CSA2FlashInfer eager plan LRU (idea 175).

1. >=8 alternating eager steps per band (6144/2048 w640, 384/128 w128), fresh
   inputs every step, through the production CSA2FlashInfer (one float
   workspace per device shared by all its plans); every call bitwise equal to
   a fresh-plan reference (a new CSA2FlashInfer per call = base's re-plan).
2. Every plan of an instance uses that instance's single float workspace.
3. A protected plan of the same (heads, width, device, scale) survives and
   stays bitwise; unprotected per-key size <= _EAGER_PLANS_PER_KEY after
   every step; a 3rd count evicts the least recently used one.
4. Steady-state cache-hit steps run under set_sync_debug_mode('error');
   the base policy (bound 1) raises in plan().
5. Ragged (_forward_pools) path: same alternation, bitwise and bounded.
6. Split-KV ragged decode plans (counts 1-16): >= 8 alternating repeated
   calls per count, NaN-poisoned outputs, bitwise vs the default int buffer.
7. Right-sized int workspaces (idea 183): every plan's private int buffer and
   pinned mirror < 8 MiB, float workspace still shared, no shared int buffer;
   40-layer bound-2 int bytes (device and pinned) <= base's 40 x 8 MiB.
8. Fallback: a too-small bound replans with the default 8 MiB buffer, bitwise.
All references are fresh plans with FlashInfer's default 8 MiB int buffer.
"""

import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as C  # noqa: E402

STEPS = 10


def fresh_shared(inp, count, width, variant):
    # Fresh plan with FlashInfer's default 8 MiB int buffer = base's re-plan.
    ref = C.default_buffer_instance()
    out = torch.empty(count, inp.heads, C.DIM, dtype=torch.bfloat16, device="cuda")
    out.fill_(float("nan"))
    C.layer_call(ref, inp, count, width, out=out, variant=variant)
    return out


def same_tail_unprotected(cache, key):
    return [k for k, p in cache.items() if k[1:] == key[1:] and not p.protected]


def main():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer

    heads, _ = C.heads_per_rank()
    inp = C.Inputs(heads)
    bound = getattr(CSA2FlashInfer, "_EAGER_PLANS_PER_KEY", 1)
    C.log(f"heads/rank={heads} bound={bound}")
    res = {"heads_per_rank": heads, "bound": bound, "failures": []}
    fail = res["failures"]
    dev = torch.device("cuda", torch.cuda.current_device())

    def key_of(count, width):
        return (count, heads, width, dev, inp.scale)

    # ---- 1-3: shared (paged) path, one instance per band ----
    calls = 0
    max_err = 0.0
    for band, (c3, c1, w) in enumerate(C.BANDS):
        inst = CSA2FlashInfer()
        # Protected (graph-warmup) plan with the same tail key.
        prot_out = torch.empty(4, heads, C.DIM, dtype=torch.bfloat16, device="cuda")
        C.layer_call(inst, inp, 4, w, out=prot_out, protect=True, meta=C.ctx_meta(0))
        if not torch.equal(prot_out, fresh_shared(inp, 4, w, 0)):
            fail.append(f"band{band} protected first call mismatch")
        for s in range(STEPS):
            count = c3 if s % 2 == 0 else c1
            out = torch.empty(count, heads, C.DIM, dtype=torch.bfloat16, device="cuda")
            out.fill_(float("nan"))
            C.layer_call(inst, inp, count, w, out=out, variant=s)
            ref = fresh_shared(inp, count, w, s)
            calls += 1
            if not torch.equal(out, ref):
                err = (out.float() - ref.float()).abs().nan_to_num(1e9).max().item()
                max_err = max(max_err, err)
                fail.append(f"band{band} step{s} count{count} mismatch max_abs {err}")
            if not torch.isfinite(out).all():
                fail.append(f"band{band} step{s} non-finite")
            # repeat the same call (cache hit) -> deterministic
            out2 = torch.empty_like(out)
            C.layer_call(inst, inp, count, w, out=out2, variant=s)
            if not torch.equal(out, out2):
                fail.append(f"band{band} step{s} repeat not deterministic")
            n = len(same_tail_unprotected(inst._paged_plans, key_of(count, w)))
            if n > bound:
                fail.append(f"band{band} step{s} {n} unprotected plans > bound {bound}")
            if key_of(4, w) not in inst._paged_plans:
                fail.append(f"band{band} step{s} protected plan evicted")
        ws = inst._workspaces[dev]
        for k, p in inst._paged_plans.items():
            if p.wrapper._float_workspace_buffer.data_ptr() != ws.data_ptr():
                fail.append(f"band{band} plan {k[:3]} has its own float workspace")
        res[f"band{band}_plans_after_alternation"] = sorted(k[0] for k in inst._paged_plans)
        res[f"band{band}_float_workspaces"] = len(inst._workspaces)
        # protected plan still bitwise
        C.layer_call(inst, inp, 4, w, out=prot_out, variant=0, meta=C.ctx_meta(0))
        if not torch.equal(prot_out, fresh_shared(inp, 4, w, 0)):
            fail.append(f"band{band} protected plan mismatch after alternation")
        # 3rd count evicts the LRU one. Last step was c1 (STEPS even): LRU is c3.
        third = (c3 + c1) // 2
        lru = c3 if (STEPS - 1) % 2 == 1 else c1
        mru = c1 if lru == c3 else c3
        out3 = torch.empty(third, heads, C.DIM, dtype=torch.bfloat16, device="cuda")
        C.layer_call(inst, inp, third, w, out=out3, variant=99)
        if not torch.equal(out3, fresh_shared(inp, third, w, 99)):
            fail.append(f"band{band} third count mismatch")
        keys = set(inst._paged_plans)
        if bound >= 2:
            if key_of(lru, w) in keys or key_of(mru, w) not in keys or key_of(third, w) not in keys:
                fail.append(f"band{band} third count LRU eviction wrong: {sorted(k[0] for k in keys)}")
        res[f"band{band}_plans_after_third"] = sorted(k[0] for k in keys)
        # MRU plan still bitwise after the third count
        outm = torch.empty(mru, heads, C.DIM, dtype=torch.bfloat16, device="cuda")
        C.layer_call(inst, inp, mru, w, out=outm, variant=7)
        if not torch.equal(outm, fresh_shared(inp, mru, w, 7)):
            fail.append(f"band{band} mru after third mismatch")
        del inst
        torch.cuda.empty_cache()
    res["shared_calls_compared"] = calls
    res["shared_max_abs_err"] = max_err

    # ---- 4: sync-free steady state (cache hits) vs base policy ----
    def alternate(inst_list, steps, mode):
        for s in range(steps):
            three = s % 2 == 0
            for li, inst in enumerate(inst_list):
                c3, c1, w = C.BANDS[li]
                count = c3 if three else c1
                # variant 0: inputs already on device, so only the op can sync.
                C.layer_call(inst, inp, count, w, variant=0)
            if mode is not None and s == 1:
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode(mode)

    layers = [CSA2FlashInfer(), CSA2FlashInfer()]
    try:
        alternate(layers, 6, "error")  # warm 2 steps, then 4 steps under 'error'
        res["candidate_sync_error_raised"] = False
    except RuntimeError as e:
        res["candidate_sync_error_raised"] = True
        res["candidate_sync_error"] = str(e)[:300]
        fail.append("steady-state cache-hit step synchronized")
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    base_like = [CSA2FlashInfer(), CSA2FlashInfer()]
    for inst in base_like:
        inst._EAGER_PLANS_PER_KEY = 1  # the base policy: evict every other count
    try:
        alternate(base_like, 6, "error")
        res["base_policy_sync_error_raised"] = False
    except RuntimeError as e:
        res["base_policy_sync_error_raised"] = True
        res["base_policy_sync_error"] = str(e)[:300]
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    del layers, base_like
    torch.cuda.empty_cache()

    # ---- 5: ragged path (_forward_pools) ----
    def pools_call(inst, count, wswa, wext, variant, meta=None):
        g = torch.Generator(device="cpu").manual_seed(1000 + 7 * count + variant)
        q = (torch.randn(count, heads, C.DIM, generator=g) * 0.5).to(torch.bfloat16).cuda()
        swa = (torch.randn(count, wswa, C.DIM, generator=g) * 0.5).to(torch.bfloat16).cuda()
        ext = None
        if wext:
            ext = (torch.randn(count, wext, C.DIM, generator=g) * 0.5).to(torch.bfloat16).cuda()
        lengths = torch.randint(0, wswa + wext + 1, (count,), generator=g, dtype=torch.int32).cuda()
        key = (count, heads, wswa + wext, dev, inp.scale)
        inst._evict_eager_plans(meta or C.ctx_meta(), key)
        out = torch.full_like(q, float("nan"))
        with torch.cuda.device(dev):
            return inst._forward_pools(q, swa, ext, lengths, inp.sink, inp.scale, out=out), key

    rag = CSA2FlashInfer()
    rag_calls = 0
    for wswa, wext, counts in ((128, 512, (512, 128)), (128, 0, (384, 64))):
        for s in range(STEPS):
            count = counts[s % 2]
            out, key = pools_call(rag, count, wswa, wext, s)
            ref, _ = pools_call(C.default_buffer_instance(), count, wswa, wext, s)
            rag_calls += 1
            if not torch.equal(out, ref):
                fail.append(f"ragged w{wswa + wext} step{s} mismatch")
            n = len(same_tail_unprotected(rag._plans, key))
            if n > bound:
                fail.append(f"ragged step{s} {n} plans > bound")

    # ---- 6: split-KV ragged decode plans (count <= 16, chunks of 64 rows) ----
    # The int workspace bound includes the chunk count; these are the plans
    # graph warmup creates (protected). >= 8 alternating repeated calls per count.
    split = CSA2FlashInfer()
    split_ref = C.default_buffer_instance()
    split_calls = 0
    for wswa, wext, counts in ((128, 512, (1, 2, 4, 8)), (128, 128, (1, 4, 16))):
        for s in range(8 * len(counts)):
            count = counts[s % len(counts)]
            out, key = pools_call(split, count, wswa, wext, s, meta=C.ctx_meta(0))
            ref, _ = pools_call(split_ref, count, wswa, wext, s, meta=C.ctx_meta(0))
            split_calls += 1
            if not torch.equal(out, ref) or not torch.isfinite(out).all():
                fail.append(f"split w{wswa + wext} count{count} step{s} mismatch")
    res["split_plans"] = sorted(
        (k[0], k[2], p.split_rows, p.wrapper._int_workspace_buffer.numel())
        for k, p in split._plans.items()
    )
    if not any(p.split_rows for p in split._plans.values()):
        fail.append("split case built no split-KV plan")
    res["split_calls_compared"] = split_calls

    # ---- 7: int workspace right-size: sizes, sharing, memory vs base ----
    ws_rows = []
    for inst in (rag, split):
        for cache in (inst._plans, inst._paged_plans):
            for k, p in cache.items():
                ib = p.wrapper._int_workspace_buffer
                pb = p.wrapper._pin_memory_int_workspace_buffer
                ws_rows.append((k[0], k[2], ib.numel()))
                if ib.numel() >= CSA2FlashInfer._WORKSPACE_BYTES:
                    fail.append(f"plan {k[:3]} kept the 8 MiB int buffer")
                if pb.numel() != ib.numel():
                    fail.append(f"plan {k[:3]} pinned mirror {pb.numel()} != {ib.numel()}")
                if p.wrapper._float_workspace_buffer.data_ptr() != inst._workspaces[dev].data_ptr():
                    fail.append(f"plan {k[:3]} lost the shared float workspace")
    ptrs = [p.wrapper._int_workspace_buffer.data_ptr() for p in rag._plans.values()]
    if len(set(ptrs)) != len(ptrs):
        fail.append("two plans share an int workspace")
    res["int_workspace_rows"] = sorted(ws_rows)
    # Production layout memory: 40 layers, bound-2 alternation of both bands.
    lay = C.make_layers()
    for s in range(4):
        for li, inst in enumerate(lay):
            c, w = C.step_counts(li, s % 2 == 0)
            C.layer_call(inst, inp, c, w, variant=0)
    torch.cuda.synchronize()
    cand_int = C.int_ws_bytes(lay)
    cand_pin = C.pinned_bytes(lay)
    base_plans_per_layer = 1  # base keeps one eager plan per key
    base_int = C.LAYERS * base_plans_per_layer * CSA2FlashInfer._WORKSPACE_BYTES
    res["layers40_candidate_plans"] = C.plan_counts(lay)
    res["layers40_candidate_int_device_bytes"] = cand_int
    res["layers40_candidate_int_pinned_bytes"] = cand_pin
    res["layers40_base_int_device_bytes"] = base_int
    res["layers40_base_int_pinned_bytes"] = base_int
    if cand_int > base_int or cand_pin > base_int:
        fail.append("candidate int workspaces exceed base's")
    del lay
    torch.cuda.empty_cache()

    # ---- 7b: FlashInfer's own workspace_size() vs our bound (paged plans) ----
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    wsz_rows = []
    probe = BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(CSA2FlashInfer._WORKSPACE_BYTES, dtype=torch.uint8, device=dev),
        kv_layout="NHD",
        backend="fa2",
    )
    try:
        for count, width in ((1, 128), (4, 640), (128, 128), (384, 128), (2048, 640), (6144, 640), (8192, 640)):
            qo = torch.arange(count + 1, dtype=torch.int32)
            _, need = probe.workspace_size(
                qo, qo * width, torch.zeros(count * width, dtype=torch.int32),
                torch.ones(count, dtype=torch.int32), heads, 1, C.DIM, 1, head_dim_vo=C.DIM,
                custom_mask=torch.ones(count * width, dtype=torch.bool, device=dev),
                causal=False, sm_scale=inp.scale, q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16, disable_split_kv=True,
            )
            bound_b = CSA2FlashInfer.int_workspace_bytes(count, heads)
            wsz_rows.append((count, width, int(need), bound_b))
            if int(need) > bound_b:
                fail.append(f"workspace_size {need} > bound {bound_b} at count {count}")
    except Exception as e:  # noqa: BLE001
        res["workspace_size_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["workspace_size_vs_bound"] = wsz_rows

    # ---- 8: fallback: a too-small bound replans with the default buffer ----
    fb = CSA2FlashInfer()
    fb.int_workspace_bytes = lambda count, heads, chunks=1: 512
    fb_ok = True
    try:
        for s in range(8):
            count = (384, 64)[s % 2]
            out, _ = pools_call(fb, count, 128, 0, s)
            ref, _ = pools_call(C.default_buffer_instance(), count, 128, 0, s)
            fb_ok &= bool(torch.equal(out, ref))
    except Exception as e:  # noqa: BLE001
        fb_ok = False
        res["fallback_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    fb_sizes = sorted(p.wrapper._int_workspace_buffer.numel() for p in fb._plans.values())
    res["fallback_int_sizes"] = fb_sizes
    res["fallback_logged"] = CSA2FlashInfer._int_workspace_fallback_logged
    if not fb_ok or any(n != CSA2FlashInfer._WORKSPACE_BYTES for n in fb_sizes):
        fail.append("fallback to the default int buffer failed")
    del fb
    torch.cuda.empty_cache()
    for k, p in rag._plans.items():
        if p.wrapper._float_workspace_buffer.data_ptr() != rag._workspaces[dev].data_ptr():
            fail.append("ragged plan has its own float workspace")
    res["ragged_calls_compared"] = rag_calls
    res["ragged_plans"] = sorted((k[0], k[2]) for k in rag._plans)
    res["passed"] = not fail
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
