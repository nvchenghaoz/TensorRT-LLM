"""Unit test for the MiniMax-H3 AdaLN table prefetch slot (idea 105), 1 GPU, synthetic.

(tests/ is protected in this campaign, so the test lives here; it runs under pytest or as a script.)
Covers: a prefetched table equals the in-place table bit for bit; a miss gives the same bits; a
controller generation bump (set_step / reset) invalidates the slot; peek -> peek -> op still hits
and peek changes no count or key; peek returns None after the pop, for another layer, at a skipped
boundary (disabled prefetch), after a generation bump; a different temb or weight misses; reset logs
and clears; tp_size 1 blocks get no slot (main's table path).
Last stdout line (script mode): {"passed": ..., "tests": {...}}
"""

import json
import os
import sys
import traceback

import torch
import torch.nn.functional as F

os.environ.setdefault("TLLM_DISABLE_MPI", "1")

from tensorrt_llm._torch.visual_gen.models.minimax_h3 import adaln_prefetch as ap  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.minimax_h3.step_precision import (  # noqa: E402
    DenoiseStepController,
)

N, K = 96768, 2688


class _Holder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.step_precision_controller = DenoiseStepController()


class _Lin(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        self.weight = torch.nn.Parameter(
            (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            (torch.randn(N, device="cuda", generator=g) * 0.02).to(torch.bfloat16), requires_grad=False
        )


def _setup(m=2):
    holder = _Holder()
    slot = ap.H3AdalnTableSlot(holder)
    ctrl = holder.step_precision_controller
    ctrl.set_step(3, 49)
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    temb = torch.randn(m, K, device="cuda", generator=g) * 5
    x_act = F.silu(temb).to(torch.bfloat16)
    return holder, slot, ctrl, temb, x_act


def _op(slot, temb, x_act, lin):
    return ap.h3_adaln_table(temb, x_act, lin.weight, lin.bias, slot.slot_id)


LIN1 = LIN2 = None


def _lins():
    global LIN1, LIN2
    if LIN1 is None:
        LIN1, LIN2 = _Lin(1), _Lin(2)
    return LIN1, LIN2


def test_hit_bitwise_and_peek_is_side_effect_free():
    lin1, _ = _lins()
    _, slot, _, temb, x_act = _setup()
    ref = F.linear(x_act, lin1.weight, lin1.bias)
    hook = ap.make_prefetch_hook(slot, 1, lin1)
    hook(temb, x_act)
    counts0, key0 = slot.counts(), slot._key
    p1 = slot.peek(1)
    p2 = slot.peek(1)
    assert p1 is not None and p1 is p2
    assert slot.peek(2) is None  # another layer
    assert slot.counts() == counts0 and slot._key == key0  # peek changes nothing
    out = _op(slot, temb, x_act, lin1)
    assert slot.hits == 1 and slot.misses == 0, slot.counts()
    assert torch.equal(out, p1) and torch.equal(out, ref)
    assert out.data_ptr() == p1.data_ptr()  # the op hands over the peeked tensor
    assert slot.peek(1) is None  # after the pop
    assert slot._key is None and slot._table is None


def test_miss_same_bits():
    lin1, _ = _lins()
    _, slot, _, temb, x_act = _setup()
    ref = F.linear(x_act, lin1.weight, lin1.bias)
    out = _op(slot, temb, x_act, lin1)  # empty slot (block 0)
    assert slot.misses == 1 and slot.hits == 0
    assert torch.equal(out, ref)


def test_generation_bump_invalidates():
    lin1, _ = _lins()
    _, slot, ctrl, temb, x_act = _setup()
    ref = F.linear(x_act, lin1.weight, lin1.bias)
    slot.fill(1, lin1, temb, x_act)
    ctrl.set_step(4, 49)  # next forward
    assert slot.peek(1) is None
    out = _op(slot, temb, x_act, lin1)
    assert slot.hits == 0 and slot.misses == 1 and slot.dropped == 1, slot.counts()
    assert torch.equal(out, ref)
    slot.fill(1, lin1, temb, x_act)
    gen = ctrl.generation
    ctrl.reset()  # end of denoising: logs, clears
    assert ctrl.generation == gen + 1
    assert slot.peek(1) is None and slot._table is None
    assert slot.hits == 0 and slot.misses == 0 and slot.total["misses"] >= 1


def test_wrong_layer_or_temb_misses():
    lin1, lin2 = _lins()
    _, slot, _, temb, x_act = _setup()
    slot.fill(1, lin1, temb, x_act)  # prefetched for block 1, consumed by a block with lin2
    out = _op(slot, temb, x_act, lin2)
    assert slot.hits == 0 and slot.misses == 1
    assert torch.equal(out, F.linear(x_act, lin2.weight, lin2.bias))
    slot.fill(1, lin1, temb, x_act)
    temb2 = temb.clone()  # same values, different storage: still a miss (conservative)
    out = _op(slot, temb2, x_act, lin1)
    assert slot.hits == 0 and slot.misses == 2
    assert torch.equal(out, F.linear(x_act, lin1.weight, lin1.bias))


def test_disabled_skips():
    lin1, _ = _lins()
    _, slot, _, temb, x_act = _setup()
    slot.enabled = False
    slot.fill(1, lin1, temb, x_act)
    assert slot.peek(1) is None and slot.prefetches == 0
    out = _op(slot, temb, x_act, lin1)
    assert slot.misses == 1 and torch.equal(out, F.linear(x_act, lin1.weight, lin1.bias))


def test_inference_mode_and_m1():
    lin1, _ = _lins()
    with torch.inference_mode():
        _, slot, _, temb, x_act = _setup(m=1)
        ref = F.linear(x_act, lin1.weight, lin1.bias)
        slot.fill(1, lin1, temb, x_act)
        p = slot.peek(1)
        out = _op(slot, temb, x_act, lin1)
        assert slot.hits == 1 and out.shape == (1, N) and torch.equal(out, ref) and out is not None
        assert p.data_ptr() == out.data_ptr()


def test_compiled_op_no_graph_break():
    lin1, _ = _lins()
    _, slot, _, temb, _ = _setup()
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()

    def f(temb):
        x_act = F.silu(temb).to(torch.bfloat16)
        mod = ap.h3_adaln_table(temb, x_act, lin1.weight, lin1.bias, slot.slot_id)
        return mod.view(-1, N // 3) * 2, x_act

    cf = torch.compile(f, mode="default", dynamic=None)
    a, x_act = cf(temb)
    slot.fill(1, lin1, temb, x_act)
    b, _ = cf(temb)
    assert slot.misses == 1 and slot.hits == 1, slot.counts()
    assert torch.equal(a, b)
    assert len(counters["graph_break"]) == 0, dict(counters["graph_break"])


def test_tp1_block_has_no_slot():
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3

    vgm = VisualGenMapping(1, 0, tp_size=1)
    mc = DiffusionModelConfig(
        pretrained_config=None,
        mapping=vgm.to_llm_mapping(),
        visual_gen_mapping=vgm,
        skip_create_weights_in_init=True,
    )
    blocks = []
    for i in range(2):
        b = h3.MiniMaxH3TransformerBlock(
            hidden_size=5376, num_attention_heads=42, attention_head_dim=128, ffn_dim=14336,
            time_embed_dim=K, norm_eps=1e-6, qk_norm_eps=1e-6, model_config=mc, layer_idx=i,
            fuse_bf16_gate_up_swiglu=True,
        )
        b.setup_tp_overlap(mc)
        blocks.append(b)
    holder = torch.nn.Module()
    holder.transformer_blocks = torch.nn.ModuleList(blocks)
    sd_before = sorted(holder.state_dict().keys())
    h3.MiniMaxH3Transformer3DModel.setup_adaln_prefetch(holder)
    assert holder._adaln_table_slot is None
    assert all(b._adaln_slot_id is None and not b._drain_hooks for b in blocks)
    assert sorted(holder.state_dict().keys()) == sd_before


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    res = {}
    for t in TESTS:
        try:
            t()
            res[t.__name__] = "ok"
        except Exception:
            res[t.__name__] = traceback.format_exc()[-1500:]
        print(f"{t.__name__}: {res[t.__name__]}", flush=True)
    print(json.dumps({"passed": all(v == "ok" for v in res.values()), "tests": res}), flush=True)
    sys.exit(0)
