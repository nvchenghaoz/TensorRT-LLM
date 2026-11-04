"""40-layer DeepSeek-V4.1 decode chain through the REAL DeepseekV41DecoderLayer.forward.

The layers are real decoder-layer objects (unchanged forward, real mHC modules with distinct
random weights so every W_T is cold, real RMSNorm weights); only the attention and MoE
sublayers are all-SM stand-ins (two bf16 GEMMs over a 126 MB weight pool, ~80 us at M=1).
Layer 0 and the engram layers of config.json enter non-deferred (pre_mapping_lagged), as in
the model; the last layer resolves its post_mapping, then the model's _finalize_hc_state.

  --mode bench : CUDA-graph replay timing per M (both trees; prints the evidence JSON).
  --mode check : candidate only. Per M, graphs REF (split off), SPLIT, SLEEP (split + aux
                 sleep before every K2), NOPDL (split, TRTLLM_ENABLE_PDL=0); per replay new
                 inputs, NaN-poisoned outputs; every boundary's residual/pre/post/comb and
                 layer_input compared bitwise with REF, K1 vs K2 layer_input bitwise, plus
                 a no-recording final-output graph pair (lifetime race probe).
"""

import argparse
import json
import os
import statistics
import sys
import types

import torch
from torch import nn

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.models.modeling_deepseekv41 import (
    DeepseekV41DecoderLayer,
    DeepseekV41Model,
)
from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState, mHC
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

try:
    from tensorrt_llm._torch.modules.mhc import mhc_lagged_split as _split
except Exception:  # baseline tree
    _split = None

HAS_SPLIT = _split is not None


def log(*a):
    print(*a, flush=True)


def read_cfg(ckpt):
    try:
        with open(os.path.join(ckpt, "config.json")) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        log("config read failed:", e)
        return {}


class StandIn(nn.Module):
    """All-SM stand-in sublayer: y = 0.1 * (x @ Wa) @ Wb (bf16)."""

    def __init__(self, wa, wb, rec=None):
        super().__init__()
        self.wa, self.wb, self.rec = wa, wb, rec

    def run(self, x):
        if self.rec is not None:
            self.rec.append(x)
        return torch.matmul(torch.matmul(x, self.wa), self.wb) * 0.1

    def forward(self, position_ids=None, hidden_states=None, attn_metadata=None,
                all_reduce_params=None, **kw):
        return self.run(hidden_states)


def build_chain(cfg, n_layers, dev, gen):
    hidden = int(cfg.get("hidden_size", 5120))
    iters = int(cfg.get("hc_sinkhorn_iters", 20))
    hc_eps = float(cfg.get("hc_eps", 1e-6))
    rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
    engram_ids = set(cfg.get("engram_layer_ids", []) or [])
    inter = 6144
    pool = []
    for _ in range(4):
        wa = (torch.randn(hidden, inter, generator=gen, device=dev) / hidden ** 0.5).to(torch.bfloat16)
        wb = (torch.randn(inter, hidden, generator=gen, device=dev) / inter ** 0.5).to(torch.bfloat16)
        pool.append((wa, wb))
    aux = torch.cuda.Stream() if HAS_SPLIT else None
    layers = []
    for i in range(n_layers):
        layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_idx = i
        layer.engram = None
        layer.disable_attn_allreduce = False
        for name in ("hc_attn", "hc_ffn"):
            m = mHC(4, hidden, iters, dtype=torch.float32, eps=hc_eps, norm_eps=rms_eps,
                    sinkhorn_eps=hc_eps, post_mult_value=2.0).to(dev)
            with torch.no_grad():
                m.fn.copy_(torch.randn(24, 4 * hidden, generator=gen, device=dev) * 1e-2)
                m.base.copy_(torch.randn(24, generator=gen, device=dev) * 0.1)
                m.scale.copy_(torch.tensor([0.5, 0.7, 0.9], device=dev))
            setattr(layer, name, m)
        for name in ("input_layernorm", "post_attention_layernorm"):
            nrm = RMSNorm(hidden_size=hidden, eps=rms_eps, dtype=torch.bfloat16, device=dev)
            with torch.no_grad():
                nrm.weight.copy_((1.0 + 0.1 * torch.randn(hidden, generator=gen, device=dev)))
            setattr(layer, name, nrm)
        wa, wb = pool[(2 * i) % 4]
        layer.self_attn = StandIn(wa, wb)
        wa2, wb2 = pool[(2 * i + 1) % 4]
        moe = StandIn(wa2, wb2)
        layer._moe_standin = moe
        layer.forward_MoE = (lambda m: (lambda hidden_states, **kw: m.run(hidden_states)))(moe)
        nxt = i + 1
        layer._v41_defer_post_mapping = nxt < n_layers and nxt not in engram_ids
        layer._mhc_split_stream = aux
        layer._mhc_split_events = ([(torch.cuda.Event(), torch.cuda.Event()) for _ in range(2)]
                                   if aux is not None else None)
        layers.append(layer)
    model = types.SimpleNamespace(layers=layers)
    model._resolve_hc_state = types.MethodType(DeepseekV41Model._resolve_hc_state, model)
    model._finalize_hc_state = types.MethodType(DeepseekV41Model._finalize_hc_state, model)
    n_entries_nondeferred = 1 + sum(1 for i in range(1, n_layers) if not layers[i - 1]._v41_defer_post_mapping)
    log(f"chain: {n_layers} layers, engram ids {sorted(engram_ids)}, non-deferred entries "
        f"{n_entries_nondeferred}, lagged fused boundaries {2 * n_layers - n_entries_nondeferred}")
    return layers, model, hidden


def run_chain(layers, model, residual0, pre0, after_layer=None):
    hc = HCState.resolved(residual0, pre_mix=pre0)
    pos = None
    for layer in layers:
        hc = layer.forward(pos, hc, None)
        if after_layer is not None:
            after_layer(layer)
    return model._finalize_hc_state(hc)


def fill_inputs(residual0, pre0, seed):
    g = torch.Generator(device=residual0.device)
    g.manual_seed(seed)
    residual0.copy_(torch.randn(residual0.shape, generator=g, device=residual0.device))
    pre0.copy_(torch.rand(pre0.shape, generator=g, device=pre0.device) + 0.05)


def tune(layers, model, residual0, pre0):
    with autotune(skip_dynamic_tuning_buckets=True):
        run_chain(layers, model, residual0, pre0)
    torch.cuda.synchronize()


def capture(fn):
    # Warm up (Triton JIT, workspace) on a side stream as the CUDA-graph runner does.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), with_multi_stream(True):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with with_multi_stream(True):
        with torch.cuda.graph(g):
            out = fn()
    torch.cuda.synchronize()
    return g, out


def bench(args, cfg):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(7)
    layers, model, hidden = build_chain(cfg, args.layers, dev, gen)
    res = {}
    for M in args.batches:
        residual0 = torch.empty(M, 4, hidden, device=dev, dtype=torch.bfloat16)
        pre0 = torch.empty(M, 4, 1, device=dev, dtype=torch.float32)
        fill_inputs(residual0, pre0, 1)
        tune(layers, model, residual0, pre0)
        g, out = capture(lambda: run_chain(layers, model, residual0, pre0))
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.reps):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(args.inner):
                g.replay()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) / args.inner)
        times.sort()
        med = statistics.median(times)
        p10 = times[int(0.1 * (len(times) - 1))]
        p90 = times[int(0.9 * (len(times) - 1))]
        res[str(M)] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med,
                       "per_layer_us": med * 1000.0 / args.layers,
                       "finite": bool(torch.isfinite(out.float()).all().item())}
        log(f"M={M}: chain {med:.4f} ms, per layer {med * 1000.0 / args.layers:.2f} us, "
            f"noise {(p90 - p10) / med:.4f}")
        del g
    print(json.dumps({"batch": res, "output": {"class": "exact"}, "split_tree": HAS_SPLIT}))


def check(args, cfg):
    assert HAS_SPLIT, "check mode needs the candidate tree"
    os.environ["TRTLLM_MHC_SPLIT_TEST_KEEP_K2"] = "1"
    _split.TEST_KEEP_K2_LAYER_INPUT = True
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(7)
    layers, model, hidden = build_chain(cfg, args.layers, dev, gen)

    calls = {"k1": 0}
    orig_k1 = _split.mhc_lagged_layer_input

    def counting_k1(*a, **k):
        calls["k1"] += 1
        return orig_k1(*a, **k)

    _split.mhc_lagged_layer_input = counting_k1

    # Recording: boundary outputs (+K2's discarded layer_input) and stand-in inputs per layer.
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

    all_ok = True
    summary = {}
    for M in args.batches:
        residual0 = torch.empty(M, 4, hidden, device=dev, dtype=torch.bfloat16)
        pre0 = torch.empty(M, 4, 1, device=dev, dtype=torch.float32)
        fill_inputs(residual0, pre0, 1)
        tune(layers, model, residual0, pre0)

        def make_fn(record):
            def fn():
                bufs = []
                for r in recs.values():
                    r["bnd"].clear()
                    r["sub"].clear()

                def after(layer):
                    if not record:
                        return
                    r = recs[layer.layer_idx]
                    for outs, k2li in r["bnd"]:
                        for t in outs:
                            bufs.append(t.clone())
                        bufs.append((k2li if k2li is not None else outs[4]).clone())
                    for t in r["sub"]:
                        bufs.append(t.clone())
                    r["bnd"].clear()
                    r["sub"].clear()

                out = run_chain(layers, model, residual0, pre0, after_layer=after)
                bufs.append(out)
                return bufs
            return fn

        graphs = {}
        cfgs = [("REF", False, 0, "1"), ("SPLIT", True, 0, "1"), ("SLEEP", True, args.sleep_cycles, "1"),
                ("NOPDL", True, 0, "0")]
        for record in (True, False):
            for name, on, sleep, pdl in cfgs:
                if not record and name == "NOPDL":
                    continue
                _split._SPLIT_ENV_ON = on
                _split.TEST_AUX_SLEEP_CYCLES = sleep
                os.environ["TRTLLM_ENABLE_PDL"] = pdl
                calls["k1"] = 0
                g, bufs = capture(make_fn(record))
                k1_in_capture = calls["k1"]
                graphs[(name, record)] = (g, bufs)
                log(f"M={M} captured {name} record={record}: K1 launches per forward (warmup+capture) "
                    f"{k1_in_capture}, outputs {len(bufs)}")
        _split._SPLIT_ENV_ON = True
        _split.TEST_AUX_SLEEP_CYCLES = 0
        os.environ["TRTLLM_ENABLE_PDL"] = "1"

        mism = {}
        for rep in range(args.replays):
            seed = 1000 + rep
            ref_vals = {}
            for record in (True, False):
                for name, *_ in cfgs:
                    key = (name, record)
                    if key not in graphs:
                        continue
                    g, bufs = graphs[key]
                    fill_inputs(residual0, pre0, seed)
                    for b in bufs:
                        if b.dtype.is_floating_point:
                            b.fill_(float("nan"))
                    g.replay()
                    torch.cuda.synchronize()
                    vals = [b.clone() for b in bufs]
                    if name == "REF":
                        ref_vals[record] = vals
                        nonfinite = sum(int((~torch.isfinite(v.float())).sum().item()) for v in vals)
                        if nonfinite:
                            log(f"WARNING REF non-finite {nonfinite}")
                        continue
                    ref = ref_vals[record]
                    bad = 0
                    for v, r in zip(vals, ref):
                        bad += int((v.view(torch.uint8) != r.view(torch.uint8)).sum().item())
                    mism.setdefault(f"{name}/rec{int(record)}", []).append(bad)
                    if bad:
                        all_ok = False
            # K1 vs K2 layer_input inside the SPLIT recording graph: boundary tuples are
            # (residual, pre, post, comb, layer_input, k2_layer_input).
        g, bufs = graphs[("SPLIT", True)]
        fill_inputs(residual0, pre0, 4242)
        g.replay()
        torch.cuda.synchronize()
        # Walk the recorded layout: per layer, n_bnd * 6 tensors then n_sub tensors.
        k1k2_bad = 0
        idx = 0
        n_bnd_layers = []
        for layer in layers:
            deferred_entry = layer.layer_idx > 0 and layers[layer.layer_idx - 1]._v41_defer_post_mapping
            nb = 2 if deferred_entry else 1
            n_bnd_layers.append(nb)
            for _ in range(nb):
                li, k2li = bufs[idx + 4], bufs[idx + 5]
                k1k2_bad += int((li.view(torch.int16) != k2li.view(torch.int16)).sum().item())
                idx += 6
            idx += 2
        if k1k2_bad:
            all_ok = False
        summary[str(M)] = {"mismatch_bytes_per_replay": mism, "k1_vs_k2_layer_input_bad": k1k2_bad,
                           "boundaries_split": sum(n_bnd_layers)}
        log(f"M={M}: {json.dumps(summary[str(M)])}")
        del graphs

    # M > 8 takes main's path (no K1 launched), and eager (no multi-stream) too.
    calls["k1"] = 0
    M = 16
    residual0 = torch.empty(M, 4, hidden, device=dev, dtype=torch.bfloat16)
    pre0 = torch.empty(M, 4, 1, device=dev, dtype=torch.float32)
    fill_inputs(residual0, pre0, 3)
    with with_multi_stream(True):
        run_chain(layers, model, residual0, pre0)
    k1_m16 = calls["k1"]
    calls["k1"] = 0
    residual0 = torch.empty(2, 4, hidden, device=dev, dtype=torch.bfloat16)
    pre0 = torch.empty(2, 4, 1, device=dev, dtype=torch.float32)
    fill_inputs(residual0, pre0, 3)
    run_chain(layers, model, residual0, pre0)  # eager: multi-stream off
    k1_eager = calls["k1"]
    torch.cuda.synchronize()
    log(f"K1 launches: M=16 multi-stream {k1_m16}, eager M=2 {k1_eager}")
    passed = all_ok and k1_m16 == 0 and k1_eager == 0
    print(json.dumps({"passed": bool(passed), "compared": "split/sleep/nopdl graphs vs main path (REF) bitwise, "
                      "every boundary residual/pre/post/comb/layer_input + stand-in inputs + final output; "
                      "K1 vs K2 layer_input", "summary": summary, "k1_launches_m16": k1_m16,
                      "k1_launches_eager": k1_eager, "sleep_cycles": args.sleep_cycles}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="bench")
    p.add_argument("--ckpt", default="")
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--replays", type=int, default=8)
    p.add_argument("--sleep-cycles", type=int, default=800000)
    args = p.parse_args()
    cfg = read_cfg(args.ckpt)
    torch.cuda.set_device(0)
    if args.mode == "bench":
        bench(args, cfg)
    else:
        check(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
