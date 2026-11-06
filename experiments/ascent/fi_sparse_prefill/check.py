"""Acceptance-2 check for idea 239 (FI FullBF16 shared-context prefill), no model load.

Drives CSA2FlashInfer._forward_selected (the real backend dispatch) with a minimal
attention/metadata stand-in around CSA2's real staging kernels and pools:
  * FI route: every iteration of warmup+iters within main's error vs FP64, repeated
    outputs bitwise equal, output bitwise equal to the direct FISharedPrefill call;
  * steady state: 0 host syncs (torch sync debug mode 'error'), 0 new FlashInfer plans,
    0 new index-scratch allocations;
  * unsupported inputs keep the staged fused_glue FA2 path, bitwise equal to main:
    premapped (no mapping) main slots, and a 64-row tile (decode-form count).
Prints {"passed": ...} last.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage0 import DEV, HEADS, SCALE, Scenario, errors, gpu_memory, log  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import stage_shared_rows  # noqa: E402
from tensorrt_llm._torch.attention.backends.interface import (  # noqa: E402
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (  # noqa: E402
    CSA2TrtllmAttention,
)


class FakeAttention:
    """The slice of CSA2TrtllmAttention the helper uses, over one Scenario."""

    _validate_helper_output = staticmethod(CSA2TrtllmAttention._validate_helper_output)

    def __init__(self, s, premapped=False):
        self.s = s
        self.num_heads = HEADS
        self.q_scaling = 1.0
        self.sparse_params = SimpleNamespace(use_packed_sparse_attention=False)
        self.premapped = premapped
        self.prepare_calls = 0

    def _inputs(self):
        s = self.s
        topk = logical = mapping = None
        if s.has_main:
            logical = s.main_logical
            if self.premapped:
                x = logical.long()
                mc = s.main_count
                ok = (x >= 0) & (x < s.visible[:, None]) & (x < mc)
                lx = x.clamp(0, mc - 1)
                phys = s.main_table[s.requests[:, None], lx // s.main_psz].long() * s.main_psz + lx % s.main_psz
                topk = torch.where(ok, phys, -1)
            else:
                topk = logical
                mapping = s.mapping
        return SimpleNamespace(
            swa_pool=s.swa_pool,
            swa_indices=s.swa_slots,
            main_pool=s.main_pool,
            topk_indices=topk,
            main_logical_indices=logical,
            main_mapping=mapping,
        )

    def prepare_sparse_inputs(self, q, metadata, forward_args):
        self.prepare_calls += 1
        return self._inputs()

    def prepare_selected(self, q, metadata, forward_args):
        inputs = self.prepare_sparse_inputs(q, metadata, forward_args)
        metadata.stage_selected(inputs, main_logical=inputs.main_logical_indices)
        return (
            q.view(metadata.num_tokens, HEADS, 512),
            metadata.shared_pool,
            metadata.prepared_indices,
            forward_args.attention_sinks,
            SCALE,
        )


class FakeMetadata(SimpleNamespace):
    def stage_selected(self, inputs, main_logical=None):
        stage_shared_rows(
            inputs.swa_pool,
            inputs.swa_indices,
            inputs.main_pool,
            None if inputs.main_mapping is not None else inputs.topk_indices,
            main_logical,
            self,
            self.shared_plan,
            main_mapping=inputs.main_mapping,
            softmax_scale=SCALE,
        )


def make_metadata(s):
    m = s.metadata
    return FakeMetadata(
        num_tokens=m.num_tokens,
        num_sparse_topk=m.num_sparse_topk,
        shared_pool=m.shared_pool,
        _csa2_shared_selected=m._csa2_shared_selected,
        prepared_indices=m.prepared_indices,
        prepared_lens=m.prepared_lens,
        prepared_counter=m.prepared_counter,
        shared_plan=s.plan,
        _csa2_graph_reserved=False,
    )


def forward_args(s, out):
    return SimpleNamespace(
        attention_input_type=AttentionInputType.context_only,
        attention_mask=PredefinedAttentionMask.FULL,
        attention_mask_data=None,
        out_scale=None,
        out_scale_sf=None,
        output_sf=None,
        enable_dsv4_epilogue_fusion=False,
        output=out,
        attention_sinks=s.sink,
    )


def staged_reference(s, premapped):
    """main's staged shared path on the same inputs (stage_shared_rows + run_shared)."""
    att = FakeAttention(s, premapped)
    inputs = att._inputs()
    md = make_metadata(s)
    md.stage_selected(inputs, main_logical=inputs.main_logical_indices)
    out = torch.empty_like(s.q)
    CSA2FlashInfer(attention=None).run_shared(s.q, md.shared_pool, md.prepared_indices, s.sink, SCALE, out=out)
    torch.cuda.synchronize()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    log("gpu memory.used at start:", gpu_memory())
    report = {"cases": {}}
    ok = True
    shapes = [(1, 2048, True), (1, 2048, False), (1, 128, True), (1, 128, False), (3, 2048, True), (3, 128, False)]
    for i, (r, t, hm) in enumerate(shapes):
        s = Scenario(r, t, hm, seed=2000 + i)
        ref = s.reference()
        main_out = staged_reference(s, premapped=False)
        main_err = errors(main_out, ref)
        helper = CSA2FlashInfer(attention=FakeAttention(s))
        md = make_metadata(s)
        out = torch.empty(s.count, HEADS * 512, dtype=torch.bfloat16, device=DEV)
        fa = forward_args(s, out)
        q2d = s.q.view(s.count, HEADS * 512)
        assert helper._use_fi_prefill(q2d, md, fa), "FI route not selected"
        first = None
        worst = 0.0
        repeat_bitwise = True
        plans = scratch = None
        for it in range(args.warmup + args.iters):
            out.fill_(float("nan"))
            if it == args.warmup:
                # Steady state from here: no host syncs, no new plans or scratch.
                fi = helper._fi_prefill
                plans = {k: len(w._prepared_calls) for k, w in fi._wrappers.items()}
                scratch = {k: v["swa"].data_ptr() for k, v in fi._buffers.items()}
            if it >= args.warmup:
                torch.cuda.set_sync_debug_mode("error")
            res = helper._forward_selected(q2d, md, fa)
            if it >= args.warmup:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
            assert res is out
            e = errors(out.view_as(s.q), ref)
            worst = max(worst, e["max_abs"])
            if not e["finite"] or e["max_abs"] > main_err["max_abs"] * 1.25 + 1e-6:
                ok = False
            if first is None:
                first = out.clone()
            elif not torch.equal(first, out):
                repeat_bitwise = False
        fi = helper._fi_prefill
        plans_after = {k: len(w._prepared_calls) for k, w in fi._wrappers.items()}
        scratch_after = {k: v["swa"].data_ptr() for k, v in fi._buffers.items()}
        direct = torch.empty_like(s.q)
        s.out_fi = direct
        s.run_fi(fi)
        torch.cuda.synchronize()
        route_equals_direct = torch.equal(direct.view_as(out), out)
        # Fallback 1: premapped main slots (no mapping) -> staged path, bitwise main.
        fb = {}
        if hm:
            helper_pm = CSA2FlashInfer(attention=FakeAttention(s, premapped=True))
            out_pm = torch.empty_like(out)
            res = helper_pm._forward_selected(q2d, make_metadata(s), forward_args(s, out_pm))
            torch.cuda.synchronize()
            exp = staged_reference(s, premapped=True)
            fb["premapped_staged_bitwise"] = bool(torch.equal(out_pm.view_as(exp), exp))
            fb["premapped_used_fi"] = helper_pm._fi_prefill is not None
            ok &= fb["premapped_staged_bitwise"] and not fb["premapped_used_fi"]
        case_ok = (
            repeat_bitwise
            and route_equals_direct
            and plans == plans_after
            and scratch == scratch_after
            and worst <= main_err["max_abs"] * 1.25 + 1e-6
        )
        ok &= case_ok
        report["cases"][s.name] = {
            "main_vs_fp64": main_err,
            "fi_route_worst_max_abs_over_iters": worst,
            "fi_route_vs_fp64_first": errors(first.view_as(s.q), ref),
            "repeat_bitwise": repeat_bitwise,
            "route_equals_direct_call": route_equals_direct,
            "steady_plans_before_after": [str(plans), str(plans_after)],
            "steady_new_scratch": scratch != scratch_after,
            "host_syncs_in_steady_state": 0,  # sync debug mode 'error' would have raised
            "iterations": args.warmup + args.iters,
            **fb,
            "passed": case_ok,
        }
        log(s.name, json.dumps(report["cases"][s.name]))
        del ref
        torch.cuda.empty_cache()
    # Fallback 2: a 64-row tile is decode-form for FlashInfer -> staged FA2 path, bitwise main.
    s = Scenario(1, 64, True, seed=2999)
    helper = CSA2FlashInfer(attention=FakeAttention(s))
    md = make_metadata(s)
    out = torch.empty(s.count, HEADS * 512, dtype=torch.bfloat16, device=DEV)
    fa = forward_args(s, out)
    q2d = s.q.view(s.count, HEADS * 512)
    used = helper._use_fi_prefill(q2d, md, fa)
    helper._forward_selected(q2d, md, fa)
    torch.cuda.synchronize()
    exp = staged_reference(s, premapped=False)
    small = {
        "fi_selected": used,
        "staged_bitwise": bool(torch.equal(out.view_as(exp), exp)),
        "fi_instantiated": helper._fi_prefill is not None,
    }
    ok &= (not used) and small["staged_bitwise"] and not small["fi_instantiated"]
    report["cases"]["64x640_fallback"] = small
    log("64x640_fallback", json.dumps(small))
    report["passed"] = bool(ok)
    print(json.dumps(report))


if __name__ == "__main__":
    sys.exit(main())
