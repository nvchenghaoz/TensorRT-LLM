"""Untimed correctness for the fused CSA2FlashInfer glue (candidate tree only).

Compares, bitwise, the fused path against the current adapter (reference glue,
also in this tree via fused_glue=False / CSA2FlashInfer._forward) on:
  * packed custom-mask bytes (vs CSA2FlashInfer.pack_query_masks) and the plan buffer,
  * ragged KV rows (vs torch.where(valid, cat(swa, extra), 0)),
  * BF16 attention output through CSA2FlashInfer._forward_selected (real
    CSA2TrtllmAttention.prepare_selected / prepare_selected_pools on stub staging),
at decode counts 1/2/4/8 (+ an SWA-only 128-wide layer and a 64-query tile), with
mixed prepared_lens (1..640) and stale non-zero staging data beyond the lengths.
Then: an FP64 independent reference (rtol/atol 1.6e-2), a repeated-call sequence
matching bench.py (3 eager warmups, graph capture, 10 + 125 replays, every replay
compared), and graph replays at 1/2/4/8 with prepared_lens/pools/q updated in
place before each replay (8 steps), compared with the eager reference each time.
"""

import json
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DIM, EXTRA, HEADS, SCALE, SWA, log, make_layer  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2 import flashinfer_glue  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (  # noqa: E402
    CSA2FlashInfer,
    CSA2TrtllmAttention,
)

failures = []


def bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def expect_equal(name, a, b):
    ok = a.shape == b.shape and torch.equal(bits(a), bits(b))
    if not ok:
        diff = (a.float() - b.float()).abs()
        failures.append(name)
        log(f"FAIL {name}: shapes {tuple(a.shape)} {tuple(b.shape)} max|d| {diff.max().item():.3e} "
            f"mismatches {(bits(a) != bits(b)).sum().item()}")
    return ok


class Harness:
    """Stub staging around the real CSA2TrtllmAttention selected-row helpers."""

    def __init__(self, layer):
        self.layer = layer
        count = layer["q"].shape[0]
        extra = layer["extra"]
        width = SWA + (0 if extra is None else extra.shape[1])
        self.metadata = types.SimpleNamespace(
            shared_plan=None,
            num_contexts=0,
            _csa2_graph_reserved=True,
            swa_pool=layer["swa"],
            extra_pool=extra if extra is not None else torch.zeros(
                count, 1, DIM, dtype=torch.bfloat16, device="cuda"),
            num_sparse_topk=width,
            prepared_lens=layer["lens"],
            num_tokens=count,
        )
        attention = types.SimpleNamespace(q_scaling=1.0)
        attention._stage_selected_query = lambda q, m, f: q.view(m.num_tokens, HEADS, DIM)
        attention.prepare_selected = types.MethodType(CSA2TrtllmAttention.prepare_selected, attention)
        attention.prepare_selected_pools = types.MethodType(
            CSA2TrtllmAttention.prepare_selected_pools, attention)
        self.fused = CSA2FlashInfer(attention)
        self.fused.fused_glue = True
        self.ref = CSA2FlashInfer(attention)
        self.ref.fused_glue = False
        self.out_fused = torch.full((count, HEADS * DIM), float("nan"), dtype=torch.bfloat16,
                                    device="cuda")
        self.out_ref = torch.full_like(self.out_fused, float("nan"))

    def run(self, inst, out):
        args = types.SimpleNamespace(output=out, attention_sinks=self.layer["sink"])
        return inst._forward_selected(self.layer["q"].flatten(1), self.metadata, args)

    def plan_mask(self, inst):
        (plan,) = inst._plans.values()
        return plan.packed_mask.clone()


def reference_rows_and_mask(layer):
    swa, extra, lens = layer["swa"], layer["extra"], layer["lens"]
    kv = swa if extra is None else torch.cat((swa, extra), dim=1)
    valid = torch.arange(kv.shape[1], device="cuda")[None, :] < lens[:, None]
    rows = torch.where(valid[..., None], kv, 0).reshape(-1, 1, DIM).contiguous()
    return rows, CSA2FlashInfer.pack_query_masks(valid), valid


def fp64_reference(layer):
    q, swa, extra, lens, sink = (layer[k] for k in ("q", "swa", "extra", "lens", "sink"))
    kv = swa if extra is None else torch.cat((swa, extra), dim=1)
    out = torch.zeros(q.shape, dtype=torch.float64, device="cuda")
    for i in range(q.shape[0]):
        rows = kv[i, : int(lens[i])].double()
        logits = q[i].double() @ rows.T * SCALE
        m = logits.max(-1, keepdim=True).values
        p = torch.exp(logits - m)
        denom = p.sum(-1, keepdim=True) + torch.exp(sink.double()[:, None] - m)
        out[i] = (p @ rows) / denom
    return out


def check_static(count, extra_width, tag, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    layer = make_layer(count, gen, extra_width)
    h = Harness(layer)
    rows_ref, mask_ref, _ = reference_rows_and_mask(layer)
    rows = torch.full_like(rows_ref, float("nan"))
    mask = torch.full_like(mask_ref, 0xA5)
    flashinfer_glue.stage_rows_and_mask(layer["swa"], layer["extra"], layer["lens"], rows, mask)
    expect_equal(f"{tag} mask bytes", mask, mask_ref)
    expect_equal(f"{tag} ragged rows", rows, rows_ref)
    r_ref = h.run(h.ref, h.out_ref)
    m_ref = h.plan_mask(h.ref)
    r_fused = h.run(h.fused, h.out_fused)
    m_fused = h.plan_mask(h.fused)
    torch.cuda.synchronize()
    expect_equal(f"{tag} plan mask buffer", m_fused, m_ref)
    expect_equal(f"{tag} plan mask vs pack_query_masks", m_fused, mask_ref)
    expect_equal(f"{tag} output", r_fused, r_ref)
    if r_fused.data_ptr() != h.out_fused.data_ptr():
        failures.append(f"{tag} output not written in place")
    ref64 = fp64_reference(layer).reshape(count, -1)
    err = (r_fused.double() - ref64).abs()
    tol_ok = bool((err <= 1.6e-2 + 1.6e-2 * ref64.abs()).all()) and bool(torch.isfinite(r_fused).all())
    if not tol_ok:
        failures.append(f"{tag} fp64 tolerance")
    log(f"{tag}: lens {layer['lens'].tolist()[:8]} max|fused-fp64| {err.max().item():.3e} "
        f"tol_ok {tol_ok}")
    return {"max_abs_vs_fp64": err.max().item(), "tolerance_ok": tol_ok}


def check_repeated(count):
    """bench.py's sequence: 3 eager warmups, capture, 10 + 125 replays, all compared."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234 + count)
    hs = [Harness(make_layer(count, gen, EXTRA)) for _ in range(4)]
    for h in hs:
        h.run(h.ref, h.out_ref)
    refs = [h.out_ref.clone() for h in hs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    worst = 0
    with torch.cuda.stream(stream):
        for _ in range(3):
            for h, ref in zip(hs, refs):
                h.run(h.fused, h.out_fused)
                worst += int(not torch.equal(bits(h.out_fused), bits(ref)))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for h in hs:
            h.run(h.fused, h.out_fused)
    mismatched = 0
    for it in range(10 + 125):
        for h in hs:
            h.out_fused.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for h, ref in zip(hs, refs):
            if not torch.equal(bits(h.out_fused), bits(ref)):
                mismatched += 1
        # State the next call expects: the inputs are untouched by the adapter.
    for h in hs:
        rows_ref, mask_ref, _ = reference_rows_and_mask(h.layer)
        expect_equal(f"repeated c{count} final plan mask", h.plan_mask(h.fused), mask_ref)
    if worst or mismatched:
        failures.append(f"repeated c{count}: eager {worst} replay {mismatched} mismatches")
    log(f"repeated c{count}: eager mismatches {worst}, replay mismatches {mismatched}/{135 * len(hs)}")
    return {"eager_mismatches": worst, "replay_mismatches": mismatched, "replays": 135}


def check_graph_updates(count):
    """Capture once, then update prepared_lens/pools/q in place before each replay."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(99 + count)
    h = Harness(make_layer(count, gen, EXTRA))
    h.run(h.fused, h.out_fused)  # warm plan + JIT
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        h.run(h.fused, h.out_fused)
    mismatches = 0
    for step in range(8):
        fresh = make_layer(count, gen, EXTRA, step=step + 1)
        for key in ("q", "swa", "extra", "sink", "lens"):
            h.layer[key].copy_(fresh[key])
        h.out_fused.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        replay_out = h.out_fused.clone()
        replay_mask = h.plan_mask(h.fused)
        h.run(h.ref, h.out_ref)
        _, mask_ref, _ = reference_rows_and_mask(h.layer)
        ok = expect_equal(f"graph c{count} step {step} output", replay_out, h.out_ref)
        ok &= expect_equal(f"graph c{count} step {step} mask", replay_mask, mask_ref)
        mismatches += int(not ok)
    log(f"graph c{count}: 8 replays with updated lens, mismatching steps {mismatches}")
    return {"steps": 8, "mismatching_steps": mismatches}


def main():
    torch.cuda.set_device(0)
    log("device", torch.cuda.get_device_name(0))
    report = {"static": {}, "repeated": {}, "graph_updates": {}}
    for count in (1, 2, 4, 8):
        report["static"][f"c{count}"] = check_static(count, EXTRA, f"c{count} w640", 7 + count)
    report["static"]["c4 w128"] = check_static(4, 0, "c4 w128 (SWA)", 3)
    report["static"]["c64 w640"] = check_static(64, EXTRA, "c64 w640", 5)
    for count in (1, 2, 4, 8):
        report["repeated"][f"c{count}"] = check_repeated(count)
    for count in (1, 2, 4, 8):
        report["graph_updates"][f"c{count}"] = check_graph_updates(count)
    passed = not failures
    print(json.dumps({
        "passed": passed,
        "compared": "fused CSA2FlashInfer glue vs current adapter (prepare_selected cat/valid + "
                    "_forward + output.copy_), bitwise: packed mask, ragged rows, BF16 output; "
                    "plus FP64 reference tolerance 1.6e-2",
        "failures": failures[:20],
        **report,
    }))


if __name__ == "__main__":
    main()
