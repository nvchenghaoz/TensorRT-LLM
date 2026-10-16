# Correctness check for the T == 1 GDN decode kernel (gdn_decode_t1.py).
# Compares the candidate (public fused_sigmoid_gating_delta_rule_update, which
# dispatches to the T == 1 kernel for pure decode) against
#   (a) the original generic Triton kernel on identical inputs (same tree, T == 1
#       dispatch disabled), and
#   (b) an independent FP32 PyTorch reference of the same recurrence with the
#       state rounded to bf16 after every step (the pool's storage precision).
# Sections:
#   A. stateful 120-step window per N in {1,2,4,8}, no reset: output and full
#      indexed state compared to the reference evolution after every step;
#      candidate max/mean abs error <= 1.5x original's; non-indexed pool slots
#      bitwise unchanged; read-only inputs bitwise unchanged.
#   B. repeated execution with exactly the timed loop's structure (30-layer
#      CUDA graph, 20 warmup + 100 iterations, restore + L2 evict before every
#      iteration): every invocation's outputs kept and compared bitwise to the
#      first invocation and to the reference; indexed state compared every
#      iteration; non-indexed slots and read-only inputs checked before reset.
#   C. CUDA graph capture, then replay with changed inputs and state indices.
#   D. PAD (-1) state index and non-decode (T != N) fallback dispatch.
# Bitwise equality with the original kernel is reported in every section
# (informational; pass/fail uses the tolerances above).
import json

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.fla import \
    fused_sigmoid_gating_recurrent as fsgr
from tensorrt_llm._torch.modules.fla import gdn_decode_t1 as t1mod

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def log(*args):
    print(*args, flush=True)


H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VAL_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VAL_DIM
QKVZ_DIM = CONV_DIM + VAL_DIM
SLOTS, WARMUP, ITERS, LAYERS = 16, 20, 100, 30
SCALE = K**-0.5
RATIO = 1.25
dev = torch.device("cuda")
flush_buf = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device=dev)

# Count T == 1 dispatches through the public entry point.
_t1_calls = [0]
_t1_fn = fsgr.fused_sigmoid_gating_delta_rule_update_t1


def _counting_t1(*args, **kwargs):
    _t1_calls[0] += 1
    return _t1_fn(*args, **kwargs)


fsgr.fused_sigmoid_gating_delta_rule_update_t1 = _counting_t1


def bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(
        torch.int32)


def beq(x, y):
    return torch.equal(bits(x), bits(y))


class Inputs:
    """Serving-layout buffers: q/k/v views of one in_proj_qkvz output row
    block, a/b column slices of one in_proj_ba output."""

    def __init__(self, n, gen):
        self.n = n
        self.gen = gen
        self.qkvz = torch.empty(n, QKVZ_DIM, dtype=torch.bfloat16, device=dev)
        self.ba = torch.empty(n, 2 * HV, dtype=torch.bfloat16, device=dev)
        mixed = self.qkvz[:, :CONV_DIM]
        self.q = mixed[:, :KEY_DIM].view(1, n, H, K)
        self.k = mixed[:, KEY_DIM:2 * KEY_DIM].view(1, n, H, K)
        self.v = mixed[:, 2 * KEY_DIM:].view(1, n, HV, V)
        self.b = self.ba[:, :HV]
        self.a = self.ba[:, HV:]
        # Log-uniform decay rates: mix of long- and short-memory heads.
        A = torch.exp(
            torch.rand(HV, device=dev, generator=gen) * 6.0 - 5.3)
        self.A_log = torch.log(A).float()
        self.dt_bias = (torch.randn(HV, device=dev, generator=gen) * 0.5 -
                        1.0).float()
        self.cu = torch.arange(n + 1, dtype=torch.long, device=dev)
        self.refill()

    def refill(self):
        self.qkvz.copy_(
            torch.randn(self.qkvz.shape, device=dev, generator=self.gen) *
            0.7)
        self.ba.copy_(
            torch.randn(self.ba.shape, device=dev, generator=self.gen) * 1.5)

    def snapshot(self):
        return [t.clone() for t in (self.qkvz, self.ba, self.A_log,
                                    self.dt_bias, self.cu)]

    def unchanged(self, snap):
        return all(
            beq(x, y) for x, y in zip(
                snap, (self.qkvz, self.ba, self.A_log, self.dt_bias, self.cu)))


def run(inp, pool, idx, out, original):
    t1mod.GDN_T1_DECODE_ENABLED = not original
    try:
        before = _t1_calls[0]
        fsgr.fused_sigmoid_gating_delta_rule_update(
            A_log=inp.A_log,
            dt_bias=inp.dt_bias,
            q=inp.q,
            k=inp.k,
            v=inp.v,
            a=inp.a,
            b=inp.b,
            initial_state_source=pool,
            initial_state_indices=idx,
            cu_seqlens=inp.cu,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            output=out,
        )
        used_t1 = _t1_calls[0] > before
    finally:
        t1mod.GDN_T1_DECODE_ENABLED = True
    assert used_t1 == (not original), "unexpected dispatch"


def reference(inp, state_bf16):
    """Independent FP32 recurrence. state_bf16: [N, HV, V, K]; returns
    (o fp32 [N, HV, V], new state rounded to bf16)."""
    n = inp.n
    q = inp.q.reshape(n, H, K).float()
    k = inp.k.reshape(n, H, K).float()
    v = inp.v.reshape(n, HV, V).float()
    a = inp.a.float()
    b = inp.b.float()
    q = q / (torch.sqrt((q * q).sum(-1, keepdim=True)) + 1e-6)
    k = k / (torch.sqrt((k * k).sum(-1, keepdim=True)) + 1e-6)
    q = q * SCALE
    rep = HV // H
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)
    g = -torch.exp(inp.A_log.float()) * F.softplus(
        a + inp.dt_bias.float(), beta=1.0, threshold=20.0)
    beta = torch.sigmoid(b)
    h = state_bf16.float() * torch.exp(g)[:, :, None, None]
    kv = (h * k[:, :, None, :]).sum(-1)
    vn = (v - kv) * beta[:, :, None]
    h = h + vn[:, :, :, None] * k[:, :, None, :]
    o = (h * q[:, :, None, :]).sum(-1)
    return o, h.to(torch.bfloat16)


def err(x, ref):
    d = (x.float() - ref.float()).abs()
    return d.max().item(), d.mean().item()


def nonidx_mask(idx):
    m = torch.ones(SLOTS, dtype=torch.bool, device=dev)
    valid = idx[idx >= 0].long()
    m[valid] = False
    return m


failures = []
summary = {}
bitwise_vs_original = {"compared": 0, "identical": 0}


def note_bitwise(same):
    bitwise_vs_original["compared"] += 1
    bitwise_vs_original["identical"] += int(same)


def fail(msg):
    failures.append(msg)
    log("FAIL:", msg)


# ---------------------------------------------------------------- Section A
def section_a(n):
    gen = torch.Generator(device=dev)
    gen.manual_seed(7000 + n)
    inp = Inputs(n, gen)
    pool0 = (torch.randn(SLOTS, HV, V, K, device=dev, generator=gen) *
             0.2).to(torch.bfloat16)
    idx = torch.randperm(SLOTS, device=dev, generator=gen)[:n].to(torch.int32)
    pool_o, pool_c = pool0.clone(), pool0.clone()
    ref_state = pool0[idx.long()].clone()
    out_o = torch.zeros(1, n, HV, V, dtype=torch.bfloat16, device=dev)
    out_c = torch.zeros_like(out_o)
    nmask = nonidx_mask(idx)
    agg = {"o_out_max": 0.0, "c_out_max": 0.0, "o_st_max": 0.0,
           "c_st_max": 0.0, "o_out_mean": 0.0, "c_out_mean": 0.0,
           "o_st_mean": 0.0, "c_st_mean": 0.0}
    bit_out = bit_state = 0
    steps = WARMUP + ITERS
    for step in range(steps):
        inp.refill()
        snap = inp.snapshot()
        idx_snap = idx.clone()
        run(inp, pool_o, idx, out_o, original=True)
        run(inp, pool_c, idx, out_c, original=False)
        ref_o, ref_state = reference(inp, ref_state)
        torch.cuda.synchronize()
        if not inp.unchanged(snap) or not beq(idx, idx_snap):
            fail(f"A N={n} step {step}: read-only input modified")
        if not beq(pool_c[nmask], pool0[nmask]):
            fail(f"A N={n} step {step}: non-indexed slot written (cand)")
        if not torch.isfinite(out_c.float()).all():
            fail(f"A N={n} step {step}: non-finite output")
        eo = err(out_o[0], ref_o)
        ec = err(out_c[0], ref_o)
        so = err(pool_o[idx.long()], ref_state)
        sc = err(pool_c[idx.long()], ref_state)
        agg["o_out_max"] = max(agg["o_out_max"], eo[0])
        agg["c_out_max"] = max(agg["c_out_max"], ec[0])
        agg["o_st_max"] = max(agg["o_st_max"], so[0])
        agg["c_st_max"] = max(agg["c_st_max"], sc[0])
        agg["o_out_mean"] += eo[1] / steps
        agg["c_out_mean"] += ec[1] / steps
        agg["o_st_mean"] += so[1] / steps
        agg["c_st_mean"] += sc[1] / steps
        same_out, same_pool = beq(out_o, out_c), beq(pool_o, pool_c)
        bit_out += int(same_out)
        bit_state += int(same_pool)
        note_bitwise(same_out and same_pool)
        if step in (0, 19, 59, 119):
            log(f"A N={n} step {step}: out err orig {eo[0]:.3e}/{eo[1]:.3e} "
                f"cand {ec[0]:.3e}/{ec[1]:.3e}; state err orig "
                f"{so[0]:.3e}/{so[1]:.3e} cand {sc[0]:.3e}/{sc[1]:.3e}; "
                f"ref |o| max {ref_o.abs().max().item():.3e} |h| max "
                f"{ref_state.float().abs().max().item():.3e}")
    for kind in ("out_max", "out_mean", "st_max", "st_mean"):
        o_e, c_e = agg["o_" + kind], agg["c_" + kind]
        if not (c_e <= RATIO * o_e) or c_e != c_e:
            fail(f"A N={n}: cand {kind} {c_e:.3e} > {RATIO} x orig {o_e:.3e}")
    agg["bitwise_equal_out_steps"] = bit_out
    agg["bitwise_equal_pool_steps"] = bit_state
    agg["steps"] = steps
    log(f"A N={n}: {json.dumps(agg)}")
    summary[f"A_N{n}"] = agg


# ---------------------------------------------------------------- Section B
def section_b(n):
    gen = torch.Generator(device=dev)
    gen.manual_seed(9000 + n)
    layers = []
    for _ in range(LAYERS):
        inp = Inputs(n, gen)
        pool0 = (torch.randn(SLOTS, HV, V, K, device=dev, generator=gen) *
                 0.2).to(torch.bfloat16)
        idx = torch.randperm(SLOTS, device=dev,
                             generator=gen)[:n].to(torch.int32)
        lay = dict(inp=inp, pool0=pool0, pool=pool0.clone(), idx=idx,
                   out=torch.zeros(1, n, HV, V, dtype=torch.bfloat16,
                                   device=dev),
                   nmask=nonidx_mask(idx))
        # Reference + original-kernel error from the pristine state.
        ref_o, ref_s = reference(inp, pool0[idx.long()])
        p_o = pool0.clone()
        o_o = torch.zeros_like(lay["out"])
        run(inp, p_o, idx, o_o, original=True)
        lay.update(ref_o=ref_o, ref_s=ref_s, orig_out=o_o,
                   orig_pool=p_o,
                   orig_out_err=err(o_o[0], ref_o),
                   orig_st_err=err(p_o[idx.long()], ref_s))
        lay["snap"] = inp.snapshot()
        layers.append(lay)

    def reset():
        for lay in layers:
            lay["pool"].copy_(lay["pool0"])
            lay["out"].zero_()
        flush_buf.sum()

    # Same warmup / capture as the timed harness.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for lay in layers:
            run(lay["inp"], lay["pool"], lay["idx"], lay["out"], False)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for lay in layers:
            run(lay["inp"], lay["pool"], lay["idx"], lay["out"], False)
    torch.cuda.synchronize()

    outs = []
    first_states = None
    worst = {"out_max": 0.0, "st_max": 0.0, "out_ratio": 0.0,
             "st_ratio": 0.0}
    drift = 0
    bit_orig = 0
    for it in range(WARMUP + ITERS):
        reset()
        g.replay()
        torch.cuda.synchronize()
        outs.append(torch.stack([lay["out"].clone() for lay in layers]))
        states = [lay["pool"][lay["idx"].long()].clone() for lay in layers]
        if first_states is None:
            first_states = states
            for lay in layers:
                same = beq(lay["out"], lay["orig_out"]) and beq(
                    lay["pool"], lay["orig_pool"])
                bit_orig += int(same)
                note_bitwise(same)
        for li, lay in enumerate(layers):
            if not beq(states[li], first_states[li]):
                drift += 1
                fail(f"B N={n} iter {it} layer {li}: state differs from "
                     f"first invocation")
            if not beq(lay["pool"][lay["nmask"]],
                       lay["pool0"][lay["nmask"]]):
                fail(f"B N={n} iter {it} layer {li}: non-indexed slot "
                     f"written")
            if not lay["inp"].unchanged(lay["snap"]):
                fail(f"B N={n} iter {it} layer {li}: read-only input "
                     f"modified")
        if not beq(outs[-1], outs[0]):
            drift += 1
            fail(f"B N={n} iter {it}: output differs from first invocation")
        if it == 0 or it == WARMUP + ITERS - 1:
            for li, lay in enumerate(layers):
                eo = err(lay["out"][0], lay["ref_o"])
                es = err(states[li], lay["ref_s"])
                worst["out_max"] = max(worst["out_max"], eo[0])
                worst["st_max"] = max(worst["st_max"], es[0])
                for kind, c_e, o_e in (
                    ("out_max", eo[0], lay["orig_out_err"][0]),
                    ("out_mean", eo[1], lay["orig_out_err"][1]),
                    ("st_max", es[0], lay["orig_st_err"][0]),
                    ("st_mean", es[1], lay["orig_st_err"][1]),
                ):
                    if not (c_e <= RATIO * o_e):
                        fail(f"B N={n} iter {it} layer {li}: cand {kind} "
                             f"{c_e:.3e} > {RATIO} x orig {o_e:.3e}")
                worst["out_ratio"] = max(
                    worst["out_ratio"], eo[0] / max(lay["orig_out_err"][0],
                                                    1e-30))
                worst["st_ratio"] = max(
                    worst["st_ratio"], es[0] / max(lay["orig_st_err"][0],
                                                   1e-30))
    # Every invocation's outputs, compared once more as a block.
    all_same = all(beq(o, outs[0]) for o in outs)
    if not all_same:
        fail(f"B N={n}: invocations not bitwise identical")
    res = dict(invocations=len(outs), layers=LAYERS, drift_events=drift,
               all_bitwise_identical=all_same,
               layers_bitwise_equal_to_original=bit_orig, **worst)
    log(f"B N={n}: {json.dumps(res)}")
    summary[f"B_N{n}"] = res
    del g


# ---------------------------------------------------------------- Section C
def section_c():
    n = 4
    gen = torch.Generator(device=dev)
    gen.manual_seed(4242)
    inp = Inputs(n, gen)
    pool0 = (torch.randn(SLOTS, HV, V, K, device=dev, generator=gen) *
             0.2).to(torch.bfloat16)
    idx = torch.tensor([3, 11, 0, 7], dtype=torch.int32, device=dev)
    pool = pool0.clone()
    out = torch.zeros(1, n, HV, V, dtype=torch.bfloat16, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(inp, pool, idx, out, False)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(inp, pool, idx, out, False)
    torch.cuda.synchronize()
    res = []
    for trial, new_idx in enumerate(([5, 2, 14, 9], [15, 1, 6, 12],
                                     [8, 3, 10, 4])):
        inp.refill()
        idx.copy_(torch.tensor(new_idx, dtype=torch.int32, device=dev))
        pool.copy_(pool0)
        g.replay()
        torch.cuda.synchronize()
        # Eager candidate and original on the same new inputs.
        p_e, o_e = pool0.clone(), torch.zeros_like(out)
        run(inp, p_e, idx, o_e, False)
        p_o, o_o = pool0.clone(), torch.zeros_like(out)
        run(inp, p_o, idx, o_o, True)
        ref_o, ref_s = reference(inp, pool0[idx.long()])
        same_eager = beq(out, o_e) and beq(pool, p_e)
        if not same_eager:
            fail(f"C trial {trial}: graph replay != eager candidate")
        same_orig = beq(out, o_o) and beq(pool, p_o)
        note_bitwise(same_orig)
        ec, eo = err(out[0], ref_o), err(o_o[0], ref_o)
        sc, so = err(pool[idx.long()], ref_s), err(p_o[idx.long()], ref_s)
        for kind, c_e, o_e2 in (("out_max", ec[0], eo[0]),
                                ("out_mean", ec[1], eo[1]),
                                ("st_max", sc[0], so[0]),
                                ("st_mean", sc[1], so[1])):
            if not (c_e <= RATIO * o_e2):
                fail(f"C trial {trial}: cand {kind} {c_e:.3e} > {RATIO} x "
                     f"orig {o_e2:.3e}")
        nmask = nonidx_mask(idx)
        if not beq(pool[nmask], pool0[nmask]):
            fail(f"C trial {trial}: non-indexed slot written")
        res.append(dict(trial=trial, graph_eq_eager=same_eager,
                        bitwise_eq_original=same_orig,
                        cand_out_err=ec, orig_out_err=eo, cand_st_err=sc,
                        orig_st_err=so))
    log(f"C: {json.dumps(res)}")
    summary["C"] = res
    del g


# ---------------------------------------------------------------- Section D
def section_d():
    n = 4
    gen = torch.Generator(device=dev)
    gen.manual_seed(5151)
    inp = Inputs(n, gen)
    pool0 = (torch.randn(SLOTS, HV, V, K, device=dev, generator=gen) *
             0.2).to(torch.bfloat16)
    idx = torch.tensor([6, -1, 13, 2], dtype=torch.int32, device=dev)
    p_c, o_c = pool0.clone(), torch.zeros(1, n, HV, V, dtype=torch.bfloat16,
                                          device=dev)
    p_o, o_o = pool0.clone(), torch.zeros_like(o_c)
    run(inp, p_c, idx, o_c, False)
    run(inp, p_o, idx, o_o, True)
    st = pool0[idx.clamp(min=0).long()].clone()
    st[1] = 0
    ref_o, ref_s = reference(inp, st)
    ec, eo = err(o_c[0], ref_o), err(o_o[0], ref_o)
    nmask = nonidx_mask(idx)
    ok = beq(p_c[nmask], pool0[nmask]) and ec[0] <= RATIO * eo[0] and \
        ec[1] <= RATIO * eo[1]
    if not ok:
        fail(f"D pad index: cand {ec} orig {eo}")
    same_orig = beq(o_c, o_o) and beq(p_c, p_o)
    note_bitwise(same_orig)
    # Non-decode varlen batch (T != N) must keep the generic kernel.
    before = _t1_calls[0]
    T = 5
    qkvz = torch.randn(T, QKVZ_DIM, device=dev,
                       generator=gen).to(torch.bfloat16)
    ba = torch.randn(T, 2 * HV, device=dev, generator=gen).to(torch.bfloat16)
    mixed = qkvz[:, :CONV_DIM]
    fsgr.fused_sigmoid_gating_delta_rule_update(
        A_log=inp.A_log, dt_bias=inp.dt_bias,
        q=mixed[:, :KEY_DIM].view(1, T, H, K),
        k=mixed[:, KEY_DIM:2 * KEY_DIM].view(1, T, H, K),
        v=mixed[:, 2 * KEY_DIM:].view(1, T, HV, V),
        a=ba[:, HV:], b=ba[:, :HV],
        initial_state_source=pool0.clone(),
        initial_state_indices=torch.tensor([1, 4], dtype=torch.int32,
                                           device=dev),
        cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.long, device=dev),
        use_qk_l2norm_in_kernel=True, softplus_beta=1.0,
        softplus_threshold=20.0)
    torch.cuda.synchronize()
    if _t1_calls[0] != before:
        fail("D: T != N batch dispatched to the T == 1 kernel")
    summary["D"] = dict(pad_cand_err=ec, pad_orig_err=eo,
                        pad_bitwise_eq_original=same_orig,
                        varlen_fallback=_t1_calls[0] == before)
    log(f"D: {json.dumps(summary['D'])}")


def main():
    log("module", fsgr.__file__, "gpu", torch.cuda.get_device_name())
    for n in (1, 2, 4, 8):
        section_a(n)
    for n in (1, 2, 4, 8):
        section_b(n)
    section_c()
    section_d()
    passed = not failures
    print(json.dumps({
        "passed": passed,
        "compared": "T==1 GDN decode kernel vs original generic Triton "
                    "kernel and independent FP32 reference (bf16 state per "
                    "step): 120-step stateful window, 30-layer graph "
                    "20+100 repeated execution with timed-loop resets, graph "
                    "replay with changed inputs/indices, PAD index, varlen "
                    "fallback",
        "tolerance": "candidate max/mean abs error <= 1.25x original's; "
                     "repeated invocations bitwise identical",
        "bitwise_vs_original": bitwise_vs_original,
        "num_failures": len(failures),
        "failures": failures[:20],
        "summary": summary,
    }), flush=True)


main()
