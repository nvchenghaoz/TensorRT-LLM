"""Untimed correctness for the FA2 hd512 K/V-alias skip (idea 34).

For every case, the alias call (K and V the same tensor: the overlay's new path) is compared
bitwise (torch.equal, out and LSE) with the non-alias call (V cloned: the unmodified path of the
same binary) on every iteration of a sequence matching bench.py (3 warmup + 30 iterations),
and with iteration 0. Inputs are checked unchanged against pristine copies. A one-time FP32
reference (tolerance 1.6e-2) guards against both paths being wrong. Digests of out/LSE are
printed; when pinned_hashes.json (written from the BASE tree, pinned FlashInfer) is present
they must match, which compares against the pinned FlashInfer output.
Cases: ragged decode count 1/2/4/8 width 640; partial-tile widths 600 / 700 and short 64 / 128;
paged run_shared prefill 2048 x 640 and a partial 256 x 600; hd128 ragged (non-shared-smem
instance); CUDA graph capture/replay of the ragged run and of the full 40-layer-style
_forward_pools call.
"""

import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as fa  # noqa: E402
from common import log  # noqa: E402

WARMUP = 3
ITERS = int(os.environ.get("FA2_ALIAS_ITERS", "30"))
PINNED = os.path.join(HERE, "pinned_hashes.json")

failures = []
hashes = {}
maxerr = {}


def fail(msg):
    log("FAIL:", msg)
    failures.append(msg)


def ref_attention(q, k, v, valid, scale):
    """q [n,h,d], k/v [n,w,d], valid [n,w] -> out [n,h,d] fp32, lse base-2 [n,h]."""
    s = torch.einsum("nhd,nwd->nhw", q.float(), k.float()) * scale
    s = s.masked_fill(~valid[:, None, :], -float("inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.softmax(s, dim=-1)
    out = torch.einsum("nhw,nwd->nhd", p, v.float())
    return out, lse / math.log(2.0)


def compare_seq(name, call_alias, call_clone, inputs):
    pristine = [t.clone() for t in inputs]
    o0, l0 = None, None
    total = WARMUP + ITERS
    for it in range(total):
        oa, la = call_alias()
        oc, lc = call_clone()
        torch.cuda.synchronize()
        if not (torch.isfinite(oa).all() and torch.isfinite(la).all()):
            fail(f"{name} iter {it}: non-finite alias output")
        if not (torch.equal(oa, oc) and torch.equal(la, lc)):
            d = (oa.float() - oc.float()).abs().max().item()
            fail(f"{name} iter {it}: alias != clone-V (max abs {d})")
        if o0 is None:
            o0, l0 = oa.clone(), la.clone()
        elif not (torch.equal(oa, o0) and torch.equal(la, l0)):
            fail(f"{name} iter {it}: alias output changed vs iter 0")
        for a, b in zip(inputs, pristine):
            if not torch.equal(a, b):
                fail(f"{name} iter {it}: input mutated")
    hashes[name] = fa.digest(o0, l0)
    return o0, l0


def check_ref(name, out, lse, ref_out, ref_lse):
    e_o = (out.float() - ref_out).abs().max().item()
    e_l = (lse.float() - ref_lse).abs().max().item()
    maxerr[name] = {"out": e_o, "lse": e_l}
    tol = 1.6e-2 + 1.6e-2 * ref_out.abs().max().item()
    if e_o > tol or e_l > 1e-2 * max(1.0, ref_lse.abs().max().item()):
        fail(f"{name}: vs fp32 reference out {e_o} lse {e_l}")


def decode_case(adapter, count, width, dim=fa.DIM, heads=fa.HEADS, graph=False):
    name = f"ragged_d{dim}_c{count}_w{width}"
    q, rows, valid, _ = fa.make_decode(count, width, 1000 + 17 * count + width + dim, dim, heads)
    plan = fa.decode_plan(adapter, count, width, valid, dim, heads)
    rows_c = rows.clone()
    out, lse = compare_seq(
        name,
        lambda: plan.wrapper.run(q, rows, rows, return_lse=True),
        lambda: plan.wrapper.run(q, rows, rows_c, return_lse=True),
        [q, rows],
    )
    kv = rows.view(count, width, dim)
    ro, rl = ref_attention(q, kv, kv, valid, fa.SCALE)  # decode_plan always plans fa.SCALE
    check_ref(name, out, lse, ro, rl)
    if graph:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            plan.wrapper.run(q, rows, rows, return_lse=True)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            go, gl = plan.wrapper.run(q, rows, rows, return_lse=True)
        for it in range(10):
            go.zero_()
            g.replay()
            torch.cuda.synchronize()
            if not (torch.equal(go, out) and torch.equal(gl, lse)):
                fail(f"{name}: graph replay {it} != eager")
        log(f"  {name}: graph capture/replay ok x10")
    log(f"{name}: ok so far ({len(failures)} failures), hash {hashes[name]}")


def pools_graph_case(count):
    """The tree's full decode adapter (_forward_pools) eager vs CUDA-graph replay."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "glue_common", os.path.join(os.path.dirname(HERE), "csa2_fi_glue", "common.py")
    )
    glue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(glue)
    name = f"forward_pools_c{count}"
    adapters = [fa.new_adapter() for _ in range(4)]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(99 + count)
    layers = [glue.make_layer(count, gen, glue.EXTRA) for _ in adapters]
    outs = [torch.empty(count, 16 * 512, dtype=torch.bfloat16, device="cuda") for _ in layers]

    def step():
        for inst, layer, out in zip(adapters, layers, outs):
            glue.fused_call(inst, layer, out)

    step()
    torch.cuda.synchronize()
    eager = [o.clone() for o in outs]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for it in range(WARMUP + ITERS):
        for o in outs:
            o.zero_()
        g.replay()
        torch.cuda.synchronize()
        for o, e in zip(outs, eager):
            if not torch.equal(o, e):
                fail(f"{name}: replay {it} != eager")
    hashes[name] = fa.digest(*eager)
    log(f"{name}: graph replay x{WARMUP + ITERS} ok, hash {hashes[name]}")


def prefill_case(adapter, count, width):
    name = f"paged_q{count}_w{width}"
    q, bank, indices, sink = fa.make_prefill(5000 + count + width, count, width)
    plan, pages = fa.prefill_plan(adapter, q, bank, indices, sink)
    pages_c = pages.clone()
    qc = q.contiguous()
    out, lse = compare_seq(
        name,
        lambda: plan.wrapper.run(qc, (pages, pages), return_lse=True),
        lambda: plan.wrapper.run(qc, (pages, pages_c), return_lse=True),
        [qc, bank, indices],
    )
    # FP32 reference on a subset of queries (clamped indices, -1 masked as run_shared does).
    sel = torch.arange(0, count, max(1, count // 64), device="cuda")
    idx = indices[sel]
    valid = idx >= 0
    kv = bank[idx.clamp_min(0).long()]
    ro, rl = ref_attention(q[sel], kv, kv, valid, fa.SCALE)
    check_ref(name, out[sel], lse[sel], ro, rl)
    # Full run_shared (prefill path incl. its sink epilogue) is deterministic.
    r0 = adapter.run_shared(q, bank, indices, sink, fa.SCALE)
    r1 = adapter.run_shared(q, bank, indices, sink, fa.SCALE)
    if not torch.equal(r0, r1):
        fail(f"{name}: run_shared not deterministic")
    hashes[name + "_run_shared"] = fa.digest(r0)
    log(f"{name}: ok so far ({len(failures)} failures), hash {hashes[name]}")


def main():
    fa.start_heartbeat()
    torch.cuda.set_device(0)
    log("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    fa.log_flashinfer()
    adapter = fa.new_adapter()
    for count in (1, 2, 4, 8):
        decode_case(adapter, count, fa.WIDTH, graph=(count in (1, 4)))
    decode_case(adapter, 1, 600, graph=True)
    decode_case(adapter, 4, 600)
    decode_case(adapter, 2, 700)
    decode_case(adapter, 1, 64)
    decode_case(adapter, 2, 128)
    fa.log_jit_modules("after ragged")
    for count in (1, 4):
        pools_graph_case(count)
    prefill_case(adapter, fa.PREFILL_Q, fa.WIDTH)
    prefill_case(adapter, 256, 600)
    fa.log_jit_modules("after paged")
    if os.environ.get("FA2_ALIAS_HD128", "1") == "1":
        # Fresh adapter: plan keys do not include head_dim.
        adapter128 = fa.new_adapter()
        decode_case(adapter128, 4, fa.WIDTH, dim=128)
        decode_case(adapter128, 2, 600, dim=128)
    pinned_compared = 0
    if os.path.exists(PINNED) and os.environ.get("FA2_ALIAS_WRITE_PINNED") != "1":
        pinned = json.load(open(PINNED))
        for k, v in pinned.items():
            if k in hashes:
                pinned_compared += 1
                if hashes[k] != v:
                    fail(f"{k}: digest {hashes[k]} != pinned FlashInfer {v}")
            else:
                fail(f"{k}: pinned case not run")
    log("hashes", json.dumps(hashes))
    log("max_err_vs_fp32", json.dumps(maxerr))
    passed = not failures and len(hashes) > 0
    print(json.dumps({
        "passed": passed,
        "compared": "alias (K is V) vs clone-V non-alias path of the same binary, bitwise out+LSE, "
                    f"{WARMUP}+{ITERS} iterations each; vs pinned FlashInfer digests; fp32 ref tol",
        "cases": len(hashes),
        "pinned_compared": pinned_compared,
        "failures": failures[:20],
        "hashes": hashes,
        "max_err_vs_fp32": maxerr,
    }))


if __name__ == "__main__":
    main()
