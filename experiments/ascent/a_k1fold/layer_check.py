"""Idea 386 check through the REAL DeepseekV41DecoderLayer.forward (40 layers, real mHC split boundaries).

Built on experiments/ascent/mhc_split/chain.py (real decoder layers / mHC modules, stand-in sublayers) with:
  * the MoE stand-in ending in the stand-in MoE all-reduce (router_prewait/pdl_ext.py producer, 4*M x 160, 5 us spin,
    trigger point from modeling_deepseekv4._moe_ar_trigger_at_end), so the next hc_attn consumer is its live PDL
    dependent;
  * an attention stand-in that follows CSA2 forward's K1-fold protocol with the REAL methods
    (DeepseekV41Attention.stash_k1_fold / _take_k1_fold / _k1_fold_applies / _project_inputs): the aux stream runs
    the boundary's K1 copy (K1FoldPending.run_copy) and an auxiliary reader on it (stand-in for _prepare_auxiliary:
    a [5120, 64] GEMM), the main stream runs the real attention-input chain A (folded) -> B; its output adds 1e-3*q and
    1e-2*aux into the stand-in GEMM output, and qr / q / swa / index_q / aux are recorded.
Graphs per M: REF (mHC split off: fused boundary, no K1, no fold), FOLD (split + fold, the tree default), NOFOLD
(split, TRTLLM_CSA2_A_K1_FOLD gate off: K1 -> A), SLEEP (FOLD + aux sleep before every K2 and every aux K1 copy),
NOPDL (FOLD, TRTLLM_ENABLE_PDL=0). Every replay: new inputs, NaN-poisoned outputs, every recorded tensor (each
boundary's residual/pre/post/comb/layer_input, K2's discarded layer_input, the stand-in inputs and records, the final
output) compared bitwise with REF. Fold launches are counted per M (must be > 0 at M = 1, 2 and 0 at M = 4, and 0 in
NOFOLD); no Triton K1 JIT inside capture.
argv: --ckpt <checkpoint> [--batches 1 2 4] [--replays N]
"""

import argparse
import importlib.util
import json
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "router_prewait"))
import csa2_shared_proj_bench as PB  # noqa: E402
import pdl_ext  # noqa: E402

_spec = importlib.util.spec_from_file_location("mhc_split_chain", os.path.join(HERE, "..", "mhc_split", "chain.py"))
MC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MC)

from tensorrt_llm._torch.attention.backends.sparse.csa2 import low_m_gemm as LMG  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention as CSA2Attn  # noqa: E402
from tensorrt_llm._torch.models import modeling_deepseekv4 as V4  # noqa: E402

TRIGGER_FN = getattr(V4, "_moe_ar_trigger_at_end", None)
S = MC._split
CALLS = {"fold": 0, "k1": 0}


def log(*a):
    print(*a, flush=True)


class FoldAttnStandIn(torch.nn.Module):
    def __init__(self, base, probe, aux, gen):
        super().__init__()
        self.base, self.probe, self.aux = base, probe, aux
        self.waux = (torch.randn(5120, 64, generator=gen, device="cuda") / 70.0).to(torch.bfloat16)
        self.ev0, self.ev1 = torch.cuda.Event(), torch.cuda.Event()
        self.rec = None

    def stash_k1_fold(self, pending):
        self.probe.stash_k1_fold(pending)

    def forward(self, position_ids=None, hidden_states=None, attn_metadata=None, all_reduce_params=None, **kw):
        k1 = self.probe._take_k1_fold(hidden_states, attn_metadata, None)
        self.ev0.record()
        with torch.cuda.stream(self.aux):
            self.ev0.wait()
            hidden_states.record_stream(self.aux)
            aux_h = k1.run_copy() if k1 is not None else hidden_states
            aux_out = torch.matmul(aux_h, self.waux)
            self.ev1.record()
        qr, q, swa, idx = self.probe._project_inputs(hidden_states, k1_fold=k1)
        if self.probe.with_index and idx is None:
            idx = self.probe.index_wq_b(qr)
        self.ev1.wait()
        aux_out.record_stream(torch.cuda.current_stream())
        if self.rec is not None:
            self.rec.extend([hidden_states, qr, q, swa, aux_out] + ([idx] if idx is not None else []))
        y = self.base.run(hidden_states)
        y = y + q[:, : y.shape[1]] * 1e-3
        return y + torch.nn.functional.pad(aux_out * 1e-2, (0, y.shape[1] - aux_out.shape[1]))


def moe_with_ar(moe):
    ext = pdl_ext.ext()

    def fwd(hidden_states, **kw):
        y = moe.run(hidden_states)
        m = y.shape[0]
        at_end = True if TRIGGER_FN is None else TRIGGER_FN(m)
        out = torch.empty_like(y)
        ext.producer(y, out, 5000, 4 * m, 160, pdl_ext.empty(), not at_end)
        return out

    return fwd


def build(cfg, n_layers):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(7)
    layers, model, hidden = MC.build_chain(cfg, n_layers, dev, gen)
    g = torch.Generator(device="cuda").manual_seed(PB.SEED)
    aux = torch.cuda.Stream()
    for i, layer in enumerate(layers):
        probe = PB.Probe(i % 2 == 0, g)
        assert PB.attach_branch(probe)
        for name in ("stash_k1_fold", "_take_k1_fold", "_k1_fold_applies"):
            setattr(probe, name, types.MethodType(getattr(CSA2Attn, name), probe))
        layer.self_attn = FoldAttnStandIn(layer.self_attn, probe, aux, g)
        layer.forward_MoE = moe_with_ar(layer._moe_standin)
        layer.mapping = types.SimpleNamespace(enable_attention_dp=False)
        layer.mlp = types.SimpleNamespace()
    LMG.k1fold_warmup(*layers[0].self_attn.probe._wq_a_wkv)
    return layers, model, hidden


def k1_jit_size():
    n = 0
    for kern in (S._mhc_lagged_layer_input_kernel, getattr(S, "_mhc_lagged_layer_input_fast_kernel", None)):
        for c in getattr(kern, "device_caches", {}).values():
            try:
                n += len(c[0])
            except Exception:  # noqa: BLE001
                pass
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--replays", type=int, default=6)
    p.add_argument("--sleep-cycles", type=int, default=800000)
    args = p.parse_args()
    torch.cuda.set_device(0)
    pdl_ext.ext()
    cfg = MC.read_cfg(args.ckpt)
    os.environ["TRTLLM_MHC_SPLIT_TEST_KEEP_K2"] = "1"
    S.TEST_KEEP_K2_LAYER_INPUT = True
    layers, model, hidden = build(cfg, args.layers)

    orig_fold = LMG.low_m_mm_k1fold
    orig_k1 = S.mhc_lagged_layer_input

    def counting_fold(*a, **k):
        CALLS["fold"] += 1
        return orig_fold(*a, **k)

    def counting_k1(*a, **k):
        CALLS["k1"] += 1
        return orig_k1(*a, **k)

    LMG.low_m_mm_k1fold = counting_fold
    S.mhc_lagged_layer_input = counting_k1

    recs = {}
    for layer in layers:
        recs[layer.layer_idx] = {"bnd": [], "sub": []}
        orig = layer._mhc_boundary

        def wrapped(*a, _orig=orig, _r=recs[layer.layer_idx]["bnd"], **k):
            outs, join = _orig(*a, **k)
            _r.append((outs, join.k2_layer_input if join is not None else None))
            return outs, join

        layer._mhc_boundary = wrapped
        layer.self_attn.rec = recs[layer.layer_idx]["sub"]
        layer._moe_standin.rec = recs[layer.layer_idx]["sub"]

    failures, summary = [], {}
    for M in args.batches:
        residual0 = torch.empty(M, 4, hidden, device="cuda", dtype=torch.bfloat16)
        pre0 = torch.empty(M, 4, 1, device="cuda", dtype=torch.float32)
        MC.fill_inputs(residual0, pre0, 1)
        MC.tune(layers, model, residual0, pre0)

        def fn():
            bufs = []
            for r in recs.values():
                r["bnd"].clear()
                r["sub"].clear()

            def after(layer):
                r = recs[layer.layer_idx]
                for outs, k2li in r["bnd"]:
                    for t in outs:
                        bufs.append(t.clone())
                    bufs.append((k2li if k2li is not None else outs[4]).clone())
                for t in r["sub"]:
                    bufs.append(t.clone())
                r["bnd"].clear()
                r["sub"].clear()

            bufs.append(MC.run_chain(layers, model, residual0, pre0, after_layer=after))
            return bufs

        cfgs = [("REF", False, True, 0, "1"), ("FOLD", True, True, 0, "1"), ("NOFOLD", True, False, 0, "1"),
                ("SLEEP", True, True, args.sleep_cycles, "1"), ("NOPDL", True, True, 0, "0")]
        graphs, counts = {}, {}
        for name, split_on, fold_on, sleep, pdl in cfgs:
            S._SPLIT_ENV_ON = split_on
            LMG.A_K1_FOLD_ENV_ON = fold_on
            S.TEST_AUX_SLEEP_CYCLES = sleep
            os.environ["TRTLLM_ENABLE_PDL"] = pdl
            CALLS["fold"] = CALLS["k1"] = 0
            j0 = k1_jit_size()
            # MC.capture: 2 eager warmups (multi-stream on) on a side stream, then the capture.
            g, bufs = MC.capture(fn)
            counts[name] = {"fold_calls_3_forwards": CALLS["fold"], "k1_calls_3_forwards": CALLS["k1"],
                            "k1_jit_after_warmup_and_capture": k1_jit_size() - j0}
            graphs[name] = (g, bufs)
            log(f"M={M} captured {name}: {counts[name]}, outputs {len(bufs)}")
        S._SPLIT_ENV_ON, LMG.A_K1_FOLD_ENV_ON, S.TEST_AUX_SLEEP_CYCLES = True, True, 0
        os.environ["TRTLLM_ENABLE_PDL"] = "1"
        # The fold runs exactly where its gate admits it: hc_attn boundaries at M <= A_K1_FOLD_MAX_M.
        expect_fold = 1 <= M <= LMG.A_K1_FOLD_MAX_M
        for name in ("FOLD", "SLEEP", "NOPDL"):
            if (counts[name]["fold_calls_3_forwards"] > 0) != expect_fold:
                failures.append(f"M={M} {name}: fold calls {counts[name]['fold_calls_3_forwards']}")
        if counts["NOFOLD"]["fold_calls_3_forwards"] or counts["REF"]["fold_calls_3_forwards"]:
            failures.append(f"M={M}: fold ran with the gate off / split off")
        mism = {}
        for rep in range(args.replays):
            seed = 1000 + rep
            ref = None
            for name, *_ in cfgs:
                g, bufs = graphs[name]
                MC.fill_inputs(residual0, pre0, seed)
                for b in bufs:
                    if b.dtype.is_floating_point:
                        b.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                vals = [b.clone() for b in bufs]
                if name == "REF":
                    ref = vals
                    nf = sum(int((~torch.isfinite(v.float())).sum().item()) for v in vals)
                    if nf:
                        failures.append(f"M={M} REF non-finite {nf}")
                    continue
                if len(vals) != len(ref):
                    failures.append(f"M={M} {name}: {len(vals)} outputs vs REF {len(ref)}")
                    continue
                bad = sum(int((v.contiguous().view(torch.uint8) != r.contiguous().view(torch.uint8)).sum().item())
                          for v, r in zip(vals, ref))
                mism.setdefault(name, []).append(bad)
                if bad:
                    failures.append(f"M={M} {name} replay {rep}: {bad} bytes differ from REF")
        summary[str(M)] = {"mismatch_bytes_per_replay": mism, "counts": counts, "tensors": len(graphs["REF"][1])}
        log(f"M={M}: {json.dumps(summary[str(M)])}")
        del graphs
        torch.cuda.empty_cache()
    print(json.dumps({"passed": not failures, "failures": failures[:20],
                      "compared": "real DeepseekV41DecoderLayer chain (40 layers, PDL-live stand-in MoE AR, real A/B "
                                  "with the hc_attn K1 fold, aux-stream K1 copy + aux reader): FOLD / NOFOLD / SLEEP / "
                                  "NOPDL graphs vs REF (split off) bitwise, every boundary output, K2 layer_input, "
                                  "attention records (qr, q, swa, index_q, aux) and final output, NaN-poisoned",
                      "summary": summary}))


if __name__ == "__main__":
    main()
