"""Idea 208 check through the REAL DeepseekV41DecoderLayer.forward (40 layers, real mHC split boundaries).

Reuses experiments/ascent/mhc_split/chain.py (check mode: graphs REF = split off, SPLIT, SLEEP = split + aux sleep
before every K2, NOPDL = TRTLLM_ENABLE_PDL=0; per replay new inputs and NaN-poisoned outputs; every boundary output
and the final output compared bitwise with REF), with two stand-ins made closer to the model:
  * the MoE stand-in ends with the stand-in MoE all-reduce (router_prewait/pdl_ext.py producer, 4*M x 160, 5 us spin)
    whose trigger point comes from the tree's modeling_deepseekv4._moe_ar_trigger_at_end(M) (end trigger in a tree
    without it), so the next layer's hc_attn K1 (trigger-first in this branch, via the real _mhc_boundary(which=0))
    is a live PDL dependent of it;
  * the attention stand-in first runs the real attention-input chain (DeepseekV41Attention._project_inputs:
    wq_a||wkv low-M bit 1 -> fused q_norm + wq_b) on the layer input and adds 1e-3 * q into its output.
argv: --ckpt <checkpoint> [--batches ...] [--replays N]
"""

import argparse
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "router_prewait"))
import csa2_shared_proj_bench as PB  # noqa: E402
import pdl_ext  # noqa: E402

_spec = importlib.util.spec_from_file_location("mhc_split_chain", os.path.join(HERE, "..", "mhc_split", "chain.py"))
MC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MC)

from tensorrt_llm._torch.models import modeling_deepseekv4 as V4  # noqa: E402

TRIGGER_FN = getattr(V4, "_moe_ar_trigger_at_end", None)
STATS = {"early": 0, "end": 0}


def log(*a):
    print(*a, flush=True)


class AttnStandIn(torch.nn.Module):
    def __init__(self, base, probe):
        super().__init__()
        self.base = base
        self.probe = probe
        self.rec = None

    def forward(self, position_ids=None, hidden_states=None, attn_metadata=None, all_reduce_params=None, **kw):
        if self.rec is not None:
            self.rec.append(hidden_states)
        qr, q, swa, idx = PB.branch(self.probe, hidden_states)
        y = torch.matmul(torch.matmul(hidden_states, self.base.wa), self.base.wb) * 0.1
        return y + q[:, : y.shape[1]] * 1e-3


def moe_with_ar(moe):
    ext = pdl_ext.ext()

    def fwd(hidden_states, **kw):
        y = moe.run(hidden_states)
        m = y.shape[0]
        at_end = True if TRIGGER_FN is None else TRIGGER_FN(m)
        STATS["end" if at_end else "early"] += 1
        out = torch.empty_like(y)
        ext.producer(y, out, 5000, 4 * m, 160, pdl_ext.empty(), not at_end)
        return out

    return fwd


_orig_build = MC.build_chain


def build_chain(cfg, n_layers, dev, gen):
    layers, model, hidden = _orig_build(cfg, n_layers, dev, gen)
    g = torch.Generator(device="cuda").manual_seed(PB.SEED)
    for i, layer in enumerate(layers):
        probe = PB.Probe(i % 2 == 0, g)
        assert PB.attach_branch(probe)
        layer.self_attn = AttnStandIn(layer.self_attn, probe)
        layer.forward_MoE = moe_with_ar(layer._moe_standin)
    log(f"patched {len(layers)} layers: real _project_inputs attention stand-in, MoE stand-in + stand-in AR "
        f"(trigger fn in tree: {TRIGGER_FN is not None})")
    return layers, model, hidden


MC.build_chain = build_chain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--replays", type=int, default=6)
    p.add_argument("--sleep-cycles", type=int, default=800000)
    p.add_argument("--mode", default="check")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--inner", type=int, default=5)
    args = p.parse_args()
    torch.cuda.set_device(0)
    pdl_ext.ext()
    cfg = MC.read_cfg(args.ckpt)
    if args.mode == "bench":
        # Real-topology timing (hc_ffn join + split fork between the AR and K1): CUDA-graph replays of the
        # 40-layer chain per M; prints mhc_split/chain.py's evidence JSON (kernel_ms = one step).
        MC.bench(args, cfg)
        return
    k1_tf_calls = {"tf": 0, "plain": 0}
    split = MC._split
    orig_k1 = split.mhc_lagged_layer_input

    def k1(*a, **k):
        k1_tf_calls["tf" if k.get("trigger_first") else "plain"] += 1
        return orig_k1(*a, **k)

    split.mhc_lagged_layer_input = k1
    # mhc_split/chain.py check() prints its JSON last; capture it and add this script's counters.
    import contextlib
    import io

    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

        def flush(self):
            sys.__stdout__.flush()

    with contextlib.redirect_stdout(Tee()):
        MC.check(args, cfg)
    last = buf.getvalue().strip().splitlines()[-1]
    res = json.loads(last)
    res["moe_ar_trigger_calls"] = STATS
    res["k1_calls"] = k1_tf_calls
    # In the candidate tree, decode M<=8 must take the early AR trigger and hc_attn K1 trigger-first.
    if TRIGGER_FN is not None:
        res["passed"] = bool(res["passed"] and STATS["early"] > 0 and k1_tf_calls["tf"] > 0)
    res["compared"] = ("real DeepseekV41DecoderLayer chain (stand-in MoE AR early trigger, real A/B): "
                       + res.get("compared", ""))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
