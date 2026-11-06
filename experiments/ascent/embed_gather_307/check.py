"""Idea 307 correctness: DeepSeek-V4 embed_tokens with use_gather_kernel=True is bitwise
main's masked lookup (get_masked_input_and_mask + F.embedding + masked_fill_) and an
independent torch reference, for ids in range and out of range (-1, V, V+7, INT32_MIN,
INT32_MAX, and for int64 also +-2^40, INT64_MIN/MAX), int32 and int64 ids,
M = 1/2/4/8/2048, PDL on and off, eager and CUDA-graph replay.

Every iteration is checked, not only the first. Eager: before each call the caching
allocator is seeded with a NaN block of the output's size (poison). Graph: before each
replay the captured output buffers (embedding and mHC seed) are filled with NaN and new
ids are copied into the static id buffer. The table is checked unchanged at the end.
Usage: check.py <checkpoint>
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ckpt_arg, log, make_embedding, model_dims, nvsmi, tree_flag  # noqa: E402

I32_MIN, I32_MAX = -2**31, 2**31 - 1
I64_MIN, I64_MAX = -2**63, 2**63 - 1
WARMUP, ITERS = 3, 12


def specials(V, dtype):
    s = [-1, V, V + 7, I32_MIN, I32_MAX, 0, V - 1]
    if dtype == torch.int64:
        s += [-2**40, 2**40, I64_MIN, I64_MAX]
    return s


def make_ids(M, V, dtype, it, gen):
    ids = torch.randint(0, V, (M, ), generator=gen, dtype=torch.int64)
    sp = specials(V, dtype)
    if M == 1:
        # cycle the specials one per iteration, every 3rd iteration in range
        if it % 3 != 2:
            ids[0] = sp[it % len(sp)]
    else:
        n = min(M // 2, len(sp))
        pos = torch.randperm(M, generator=gen)[:n]
        for k, p in enumerate(pos.tolist()):
            ids[p] = sp[(k + it) % len(sp)]
    return ids.to(dtype).cuda()


def reference(ids, w):
    V = w.shape[0]
    flat = ids.reshape(-1).to(torch.int64)
    ok = (flat >= 0) & (flat < V)
    rows = w.index_select(0, torch.where(ok, flat, torch.zeros_like(flat)))
    ref = torch.where(ok[:, None], rows, torch.zeros_like(rows))
    return ref.view(*ids.shape, w.shape[1])


def bits(t):
    return t.contiguous().view(torch.int16)


def compare(name, got, want, fails, errs):
    if got.shape != want.shape or got.dtype != want.dtype:
        fails.append(f"{name}: shape/dtype {tuple(got.shape)}/{got.dtype} vs "
                     f"{tuple(want.shape)}/{want.dtype}")
        return
    if not got.is_contiguous():
        fails.append(f"{name}: not contiguous, stride {got.stride()}")
    if not torch.isfinite(got).all():
        fails.append(f"{name}: non-finite values")
    n_bad = int((bits(got) != bits(want)).sum())
    err = float((got.float() - want.float()).abs().max()) if got.numel() else 0.0
    errs.append(err)
    if n_bad:
        fails.append(f"{name}: {n_bad} differing bf16 words, max abs err {err}")


def main():
    nvsmi()
    flag, path, call = tree_flag()
    V, H, hc, dt = model_dims(ckpt_arg())
    log(f"tree file {path}; ctor Embedding{call}; use_gather_kernel={flag}")
    log(f"V={V} H={H} hc_mult={hc} dtype={dt}")
    fails = []
    if not flag:
        fails.append("DeepseekV4Model.__init__ does not pass use_gather_kernel=True")
    torch.manual_seed(0)
    w = torch.randn(V, H, dtype=dt, device="cuda")
    w_sum = bits(w).to(torch.int64).sum().item()
    w_head = w[:4].clone()
    w_tail = w[-4:].clone()
    gath = make_embedding(V, H, dt, True, weight=w)  # this branch's path
    mask = make_embedding(V, H, dt, False, weight=w)  # main's masked path
    gen = torch.Generator().manual_seed(1234)
    stats = {}
    n_cmp = 0

    # ---- eager, every iteration poisoned and compared ----
    for pdl in ("1", "0"):
        os.environ["TRTLLM_ENABLE_PDL"] = pdl
        for M in (1, 2, 4, 8, 2048):
            for idt in (torch.int32, torch.int64):
                errs = []
                for it in range(WARMUP + ITERS):
                    ids = make_ids(M, V, idt, it, gen)
                    if it == WARMUP + ITERS - 1 and M == 8:
                        ids = ids.view(2, 4)  # [batch, seq] shaped input too
                    poison = torch.full((ids.numel(), H), float("nan"), dtype=dt, device="cuda")
                    del poison
                    got = gath(ids)
                    base = mask(ids)
                    ref = reference(ids, w)
                    torch.cuda.synchronize()
                    tag = f"eager pdl={pdl} M={M} {str(idt)[6:]} it={it}"
                    compare(tag + " gather~ref", got, ref, fails, errs)
                    compare(tag + " gather~masked", got, base, fails, errs)
                    compare(tag + " masked~ref", base, ref, fails, [])
                    n_cmp += 1
                key = f"eager_pdl{pdl}_M{M}_{str(idt)[6:]}"
                stats[key] = max(errs)
                log(f"{key}: {WARMUP + ITERS} iterations, max abs err per iteration "
                    f"{[round(e, 6) for e in errs[::2]]}")

    # ---- CUDA graph capture + replay (decode shapes), poisoned before every replay ----
    os.environ["TRTLLM_ENABLE_PDL"] = "1"
    for M in (1, 2, 4, 8):
        for idt in (torch.int32, torch.int64):
            static_ids = make_ids(M, V, idt, 0, gen)
            graphs = {}
            for name, mod in (("gather", gath), ("masked", mask)):
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        mod(static_ids).unsqueeze(1).repeat(1, hc, 1)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    e = mod(static_ids)
                    seed = e.unsqueeze(1).repeat(1, hc, 1)
                graphs[name] = (g, e, seed)
            errs = []
            for it in range(WARMUP + ITERS):
                static_ids.copy_(make_ids(M, V, idt, it, gen))
                for name, (g, e, seed) in graphs.items():
                    e.fill_(float("nan"))
                    seed.fill_(float("nan"))
                    g.replay()
                torch.cuda.synchronize()
                ref = reference(static_ids, w)
                ref_seed = ref.unsqueeze(1).repeat(1, hc, 1)
                tag = f"graph M={M} {str(idt)[6:]} it={it}"
                ge, gs = graphs["gather"][1], graphs["gather"][2]
                me, ms = graphs["masked"][1], graphs["masked"][2]
                compare(tag + " gather~ref", ge, ref, fails, errs)
                compare(tag + " gather~masked", ge, me, fails, errs)
                compare(tag + " seed gather~ref", gs, ref_seed, fails, errs)
                compare(tag + " seed gather~masked", gs, ms, fails, errs)
                n_cmp += 1
            key = f"graph_M{M}_{str(idt)[6:]}"
            stats[key] = max(errs)
            log(f"{key}: captured+replayed {WARMUP + ITERS}x, max abs err "
                f"{max(errs)}")
            del graphs
            torch.cuda.synchronize()

    # ---- the table must be untouched ----
    if (bits(w).to(torch.int64).sum().item() != w_sum or not torch.equal(w[:4], w_head)
            or not torch.equal(w[-4:], w_tail)):
        fails.append("embedding table changed")

    for f in fails[:20]:
        log("FAIL", f)
    res = {"passed": not fails, "use_gather_kernel": flag, "comparisons": n_cmp,
           "failures": len(fails), "max_abs_err": stats,
           "compared": "Embedding(use_gather_kernel=True) vs main masked path and an "
                       "independent torch index_select+where reference, bitwise bf16"}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
