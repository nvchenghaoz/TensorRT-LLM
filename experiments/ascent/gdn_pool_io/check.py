# Idea 5 (gdn-fusion): SM120 GDN chunk prefill with in-place indexed bf16 pool I/O.
# Bitwise check of the TRT-LLM adapter (fla/flashinfer_chunk.py) new path
# (pool + slot ids straight into the FlashInfer SM120 kernel) against the legacy
# path main runs (Triton gather->fp32, packed fp32 kernel, Triton cast+scatter),
# selected in-process by patching sm120_pool_io_eligible -> False.
# Real shapes: 16 qk heads, 48 v heads, D=128; recipe seed 777, alpha shift 6.
# Pool: per-layer slot stride 817152 elements (incl. a 30720-element gap placed
# BEFORE the state, so the pool view has a nonzero storage offset), 40 slots,
# scattered unsorted slot ids. Outputs AND the whole buffer (touched slots,
# untouched slots, gaps) must be bitwise equal; 10 + 50 repeated in-place calls
# from a pristine copy, each equal to the reference and to call 1.
import hashlib
import json
import sys
import time

import torch

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


import tensorrt_llm._torch.modules.fla.flashinfer_chunk as fc  # noqa: E402

H_QK, H_V, D = 16, 48, 128
SSM_E = H_V * D * D
STRIDE_E = 817152
GAP = STRIDE_E - SSM_E
NB = 40
F = torch.nn.functional
dev = torch.device("cuda")


def make(seq_lens, seed=777, shift=6.0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    T = sum(seq_lens)
    q = F.normalize(torch.randn(1, T, H_QK, D, generator=g, device=dev), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, T, H_QK, D, generator=g, device=dev), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, T, H_V, D, generator=g, device=dev).to(torch.bfloat16)
    alpha = torch.exp(-F.softplus(torch.randn(1, T, H_V, generator=g, device=dev) - shift))
    g_log = torch.log(alpha).contiguous()
    beta = torch.sigmoid(torch.randn(1, T, H_V, generator=g, device=dev)).to(torch.bfloat16)
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    cu = torch.tensor(cu, dtype=torch.int64, device=dev)
    return q, k, v, g_log, beta, cu


def make_pool(dtype, seed=4242, nb=None):
    nb = NB if nb is None else nb
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if nb == NB:
        buf = (torch.randn(nb, STRIDE_E, generator=g, device=dev) * 0.05).to(dtype)
    else:
        buf = torch.randn(nb, STRIDE_E, generator=g, device=dev, dtype=dtype)
        buf.mul_(0.05)
    pool = buf[:, GAP:].view(nb, H_V, D, D)
    assert pool.stride(0) == STRIDE_E and pool.storage_offset() == GAP
    return buf, pool


def slot_ids(n, seed=99, nb=None):
    nb = NB if nb is None else nb
    g = torch.Generator()
    g.manual_seed(seed + n)
    perm = torch.randperm(nb, generator=g)[:n]
    if nb != NB:
        # make sure the top slots (element offsets > 2^31) are exercised
        perm[0] = nb - 1
        perm[1] = nb - 7
    return perm.to(torch.int32).to(dev)


def call(inp, pool, idx, n_long, legacy, ws):
    q, k, v, g_log, beta, cu = inp
    orig = fc.sm120_pool_io_eligible
    if legacy:
        fc.sm120_pool_io_eligible = lambda s: False
    try:
        out, fin = fc.chunk_gated_delta_rule(
            q, k, v, g_log, beta,
            scale=D ** -0.5,
            initial_state=pool,
            initial_state_indices=idx,
            inplace_indexed_state_update=True,
            output_final_state=False,
            cu_seqlens=cu,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
            state_workspace=ws if legacy else None,
            num_long_seqs=n_long,
        )
    finally:
        fc.sm120_pool_io_eligible = orig
    assert fin is None
    return out


def iv(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def sha(t):
    return hashlib.sha256(iv(t.contiguous()).cpu().numpy().tobytes()).hexdigest()[:12]


def case(name, seq_lens, dtype=torch.bfloat16, zero_first=False, warm=10, iters=50, nb=None):
    nb = NB if nb is None else nb
    inp = make(seq_lens)
    n = len(seq_lens)
    n_long = sum(1 for s in seq_lens if s > 1)
    idx = slot_ids(n, nb=nb)
    buf, pool = make_pool(dtype, nb=nb)
    if zero_first:
        # has_initial_state=False row: production zeroes the slot first
        # (reset_recurrent_state_rows in gdn_mixer.forward_core).
        pool[int(idx[0].item())].zero_()
    pristine = buf.clone()
    ws = (torch.empty(n, H_V, D, D, dtype=torch.float32, device=dev),
          torch.empty(n, H_V, D, D, dtype=torch.float32, device=dev))
    # reference: main's legacy path
    ref_out = call(inp, pool, idx, n_long, True, ws).clone()
    torch.cuda.synchronize()
    ref_buf = buf.clone()
    touched = sorted(int(x) for x in idx.tolist())
    untouched = [s for s in range(nb) if s not in touched]
    rows = torch.tensor(touched, dtype=torch.long, device=dev)
    changed_slots = [s for s in touched if not torch.equal(iv(ref_buf[s]), iv(pristine[s]))]
    first = None
    bad_ref = 0
    bad_first = 0
    for it in range(warm + iters):
        buf.copy_(pristine)
        out = call(inp, pool, idx, n_long, False, None)
        torch.cuda.synchronize()
        ok = torch.equal(iv(out), iv(ref_out)) and torch.equal(iv(buf), iv(ref_buf))
        if not ok:
            bad_ref += 1
            if bad_ref == 1:
                d = (out.float() - ref_out.float()).abs()
                db = (buf[rows].float() - ref_buf[rows].float()).abs()
                log(name, "MISMATCH it", it, "out maxdiff", d.max().item(), "n", int((d > 0).sum()),
                    "buf maxdiff", db.max().item(), "n", int((db > 0).sum()))
        if first is None:
            first = (out.clone(), buf.clone())
        elif not (torch.equal(iv(out), iv(first[0])) and torch.equal(iv(buf), iv(first[1]))):
            bad_first += 1
    keep = torch.ones(nb, dtype=torch.bool, device=dev)
    keep[rows] = False
    untouched_ok = torch.equal(iv(buf[keep]), iv(pristine[keep]))
    gaps_ok = torch.equal(iv(buf[:, :GAP]), iv(pristine[:, :GAP]))
    big = nb != NB
    r = dict(
        case=name, seq_lens="%d seqs, %d long, T=%d" % (n, n_long, sum(seq_lens)),
        dtype=str(dtype), pool_slots=nb,
        max_slot_element_offset=max(touched) * STRIDE_E + GAP,
        slot_ids=idx.tolist() if n <= 32 else "%d ids" % n, calls=warm + iters,
        mismatch_vs_main_path=bad_ref, drift_vs_call1=bad_first,
        untouched_slots_and_gaps_unchanged=bool(untouched_ok and gaps_ok),
        touched_slots_updated=len(changed_slots), n_touched=len(touched),
        ref_sha_out=sha(ref_out), cand_sha_out=sha(out),
        ref_sha_pool=sha(ref_buf[rows] if big else ref_buf),
        cand_sha_pool=sha(buf[rows] if big else buf),
    )
    r["passed"] = bool(bad_ref == 0 and bad_first == 0 and untouched_ok and gaps_ok
                       and len(changed_slots) == len(touched))
    log(json.dumps(r))
    return r


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    log("eligible(bf16 pool):", fc.sm120_pool_io_eligible(make_pool(torch.bfloat16)[1]),
        "supported:", fc.sm120_indexed_state_io_supported())
    cases = [
        ("1x2048", [2048]),
        ("2048+7", [2048] + [1] * 7),
        ("2048+15", [2048] + [1] * 15),
        ("2048+31", [2048] + [1] * 31),
        ("2x2048+30", [2048, 2048] + [1] * 30),
    ]
    res = []
    big = 2700  # 2699 * 817152 > 2^31 elements: a serving-sized pool (~3900 slots)
    if quick:
        res.append(case("2048+7", [2048] + [1] * 7, warm=1, iters=2))
        res.append(case("2048+31 big pool", [2048] + [1] * 31, warm=1, iters=2, nb=big))
    else:
        res.append(case("2048+31 big pool", [2048] + [1] * 31, warm=2, iters=8, nb=big))
        res.append(case("2x2048+30 big pool", [2048, 2048] + [1] * 30, warm=2, iters=8, nb=big))
        for name, sl in cases:
            res.append(case(name, sl))
        res.append(case("1x2048 has_initial_state=False", [2048], zero_first=True))
        res.append(case("2048+7 fp16 pool", [2048] + [1] * 7, dtype=torch.float16, warm=2, iters=8))
        res.append(case("1x8192 (CP-routed)", [8192], warm=2, iters=8))
        res.append(case("3x700 ragged", [700, 1, 1300, 1, 64], warm=2, iters=8))
    passed = all(r["passed"] for r in res)
    print(json.dumps(dict(
        passed=passed,
        compared=("TRT-LLM flashinfer_chunk adapter: SM120 in-place indexed pool path vs main's "
                  "gather->fp32 / packed fp32 kernel / RNE scatter path (same process); outputs + "
                  "entire pool buffer (touched, untouched slots, slot gaps) bitwise, 10+50 calls "
                  "from a pristine pool, each == main path and == call 1; plus a 2700-slot pool "
                  "whose top slots sit beyond 2^31 elements (serving pools hold ~3900 slots)"),
        cases=res,
    )), flush=True)


main()
