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
"""

import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as C  # noqa: E402

STEPS = 10


def fresh_shared(inp, count, width, variant):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer

    ref = CSA2FlashInfer()
    out = torch.empty(count, inp.heads, C.DIM, dtype=torch.bfloat16, device="cuda")
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
    def pools_call(inst, count, wswa, wext, variant):
        g = torch.Generator(device="cpu").manual_seed(1000 + 7 * count + variant)
        q = (torch.randn(count, heads, C.DIM, generator=g) * 0.5).to(torch.bfloat16).cuda()
        swa = (torch.randn(count, wswa, C.DIM, generator=g) * 0.5).to(torch.bfloat16).cuda()
        ext = None
        if wext:
            ext = (torch.randn(count, wext, C.DIM, generator=g) * 0.5).to(torch.bfloat16).cuda()
        lengths = torch.randint(0, wswa + wext + 1, (count,), generator=g, dtype=torch.int32).cuda()
        key = (count, heads, wswa + wext, dev, inp.scale)
        inst._evict_eager_plans(C.ctx_meta(), key)
        with torch.cuda.device(dev):
            return inst._forward_pools(q, swa, ext, lengths, inp.sink, inp.scale), key

    rag = CSA2FlashInfer()
    rag_calls = 0
    for wswa, wext, counts in ((128, 512, (512, 128)), (128, 0, (384, 64))):
        for s in range(STEPS):
            count = counts[s % 2]
            out, key = pools_call(rag, count, wswa, wext, s)
            ref, _ = pools_call(CSA2FlashInfer(), count, wswa, wext, s)
            rag_calls += 1
            if not torch.equal(out, ref):
                fail.append(f"ragged w{wswa + wext} step{s} mismatch")
            n = len(same_tail_unprotected(rag._plans, key))
            if n > bound:
                fail.append(f"ragged step{s} {n} plans > bound")
    for k, p in rag._plans.items():
        if p.wrapper._float_workspace_buffer.data_ptr() != rag._workspaces[dev].data_ptr():
            fail.append("ragged plan has its own float workspace")
    res["ragged_calls_compared"] = rag_calls
    res["ragged_plans"] = sorted((k[0], k[2]) for k in rag._plans)
    res["passed"] = not fail
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
