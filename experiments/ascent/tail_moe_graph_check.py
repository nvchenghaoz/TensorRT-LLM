"""1-GPU checks for the DeepSeek-V4.1 128-row tail MoE-window CUDA graph (idea 156, stages 1-2).

A. Import receipts: TRTLLM_DSV4_TAIL_MOE_GRAPH unset / 0 / 1 / 1+VERIFY=8, each in a fresh
   subprocess: the module bools ENABLED/VERIFY and modeling_deepseekv41._TAIL_MOE_GRAPH.
B. Code receipt: the Python source of every function in the captured segment is scanned for
   host syncs (.item(), .tolist(), nonzero, .cpu(), synchronize, torch.tensor(...)); the
   M>=SHARED_ROW_SPLIT_MIN_TOKENS / capture-only branches are listed.
C. Runtime: the module's real run()/moe_window() on a fake layer whose window is the real mHC
   fused lagged boundary (hc_ffn) + an MXFP8 shared-expert chain through
   trtllm::mxfp8_quantize and trtllm::flashinfer_mm_mxfp8 (the memo path at M=128) +
   mHC post_mapping, at the per-rank shapes (128 rows, hidden 5120, hc_mult 4). Two layers
   (deferred and resolved) share the pool. 10 steps with fresh inputs: step 0 warms (eager),
   step 1 captures, steps 1-9 replay; each step's graph outputs are compared with torch.equal
   against the eager window on the same inputs (9 graphed steps per layer). Eligibility is
   probed at 127/128/129 rows, with a generation row, under is_cuda_graph, and without
   replay rows. A negative control (a window with .item()) must fail capture under
   thread_local mode and fall back to eager.
Last stdout line: {"passed": ...}.
"""

import inspect
import json
import os
import re
import subprocess
import sys
import time

os.environ["TRTLLM_DSV4_TAIL_MOE_GRAPH"] = "1"
os.environ.pop("TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY", None)

RESULT = {"passed": False}


def log(msg):
    print(msg, flush=True)


def import_receipts():
    code = (
        "import tensorrt_llm._torch.models.modeling_deepseekv41 as m41;"
        "import tensorrt_llm._torch.models._dsv41_tail_moe_graph as g;"
        "import json;print('RECEIPT '+json.dumps({'ENABLED': g.ENABLED, 'VERIFY': g.VERIFY,"
        "'m41': m41._TAIL_MOE_GRAPH}))"
    )
    cases = {"unset": None, "0": "0", "1": "1", "1_verify8": "1"}
    out = {}
    for name, val in cases.items():
        env = dict(os.environ)
        env.pop("TRTLLM_DSV4_TAIL_MOE_GRAPH", None)
        env.pop("TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY", None)
        if val is not None:
            env["TRTLLM_DSV4_TAIL_MOE_GRAPH"] = val
        if name == "1_verify8":
            env["TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY"] = "8"
        t0 = time.time()
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("RECEIPT ")]
        rec = json.loads(line[-1][8:]) if line else {"error": p.stderr[-2000:]}
        rec["rc"] = p.returncode
        out[name] = rec
        log(f"[A] env={name}: {rec} ({time.time() - t0:.0f}s)")
    ok = (
        out["unset"].get("ENABLED") is False
        and out["unset"].get("m41") is False
        and out["0"].get("ENABLED") is False
        and out["0"].get("m41") is False
        and out["1"].get("ENABLED") is True
        and out["1"].get("m41") is True
        and out["1"].get("VERIFY") == 0
        and out["1_verify8"].get("VERIFY") == 8
        and all(v.get("rc") == 0 for v in out.values())
    )
    return ok, out


SYNC_RE = re.compile(r"\.item\(|\.tolist\(|nonzero|\.cpu\(|synchronize\(|torch\.tensor\(")


def code_receipt():
    from tensorrt_llm._torch.distributed import ops as dist_ops
    from tensorrt_llm._torch.models import _dsv41_tail_moe_graph as g
    from tensorrt_llm._torch.models import modeling_deepseekv4 as m4
    from tensorrt_llm._torch.models import modeling_deepseekv41 as m41
    from tensorrt_llm._torch.modules import multi_stream_utils as msu
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
    from tensorrt_llm._torch.moe.fused_moe.cutlass import base as cbase
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8 as w4

    funcs = {
        "_dsv41_tail_moe_graph.moe_window": g.moe_window,
        "DeepseekV41DecoderLayer._mhc_boundary": m41.DeepseekV41DecoderLayer._mhc_boundary,
        "mHC.fused_hc_lagged": mHC.fused_hc_lagged,
        "mHC.post_mapping": mHC.post_mapping,
        "DeepseekV4DecoderLayer.forward_MoE": m4.DeepseekV4DecoderLayer.forward_MoE,
        "DeepseekV4MoE.forward": m4.DeepseekV4MoE.forward,
        "DeepseekV4MoE.compute_routed_output": m4.DeepseekV4MoE.compute_routed_output,
        "DeepseekV4MoE._use_shared_row_split": m4.DeepseekV4MoE._use_shared_row_split,
        "DeepseekV4Gate.forward": m4.DeepseekV4Gate.forward,
        "GatedMLP.forward": GatedMLP.forward,
        "maybe_execute_in_parallel": msu.maybe_execute_in_parallel,
        "maybe_execute_in_parallel_deferred_join": msu.maybe_execute_in_parallel_deferred_join,
        "AllReduce.forward": dist_ops.AllReduce.forward,
        "TrtllmCutlassW4a8Mxfp4Mxfp8Impl.quantize_input": w4.TrtllmCutlassW4a8Mxfp4Mxfp8Impl.quantize_input,
        "TrtllmCutlassW4a8Mxfp4Mxfp8Impl.run_moe": w4.TrtllmCutlassW4a8Mxfp4Mxfp8Impl.run_moe,
        "TrtllmCutlassW4a8Mxfp4Mxfp8Impl._folds_quant_into_cutlass": w4.TrtllmCutlassW4a8Mxfp4Mxfp8Impl._folds_quant_into_cutlass,
        "TrtllmCutlassW4a8Mxfp4Mxfp8Impl._takes_fi_decode": w4.TrtllmCutlassW4a8Mxfp4Mxfp8Impl._takes_fi_decode,
        "cutlass.base.quantize_input_mxfp8": cbase.quantize_input_mxfp8,
        "CutlassFusedMoEBase.run_moe": cbase.CutlassFusedMoEBase.run_moe,
    }
    hits = {}
    for name, fn in funcs.items():
        try:
            src = inspect.getsource(fn)
        except Exception as e:  # pragma: no cover
            hits[name] = [f"<no source: {e!r}>"]
            continue
        lines = []
        for i, ln in enumerate(src.splitlines()):
            code = ln.split("#", 1)[0]
            if SYNC_RE.search(code):
                lines.append(f"+{i}: {ln.strip()}")
        hits[name] = lines
        log(f"[B] {name}: {len(src.splitlines())} lines, sync-pattern hits {lines}")
    return hits


def runtime_check():
    import types

    import torch

    import tensorrt_llm  # noqa: F401  (loads the trtllm ops)
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.models import _dsv41_tail_moe_graph as g
    from tensorrt_llm._torch.models import modeling_deepseekv41 as m41
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

    assert g.ENABLED and m41._TAIL_MOE_GRAPH
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    cfg = {}
    ckpt = os.environ.get("CKPT")
    if ckpt and os.path.exists(os.path.join(ckpt, "config.json")):
        with open(os.path.join(ckpt, "config.json")) as f:
            cfg = json.load(f)
    hidden = int(cfg.get("hidden_size", 5120))
    mult = int(cfg.get("hc_mult", 4))
    iters = int(cfg.get("hc_sinkhorn_iters", 20))
    inter = int(cfg.get("moe_intermediate_size", 2048)) * int(cfg.get("n_shared_experts", 1)) // 4
    eps_hc = float(cfg.get("hc_eps", 1e-6))
    eps_rms = float(cfg.get("rms_norm_eps", 1e-6))
    M = g.TAIL_ROWS
    log(f"[C] hidden {hidden} hc_mult {mult} sinkhorn {iters} shared inter/rank {inter} M {M}")

    def make_mhc():
        m = mHC(mult, hidden, iters, dtype=torch.float32, eps=eps_hc, norm_eps=eps_rms,
                sinkhorn_eps=eps_hc, post_mult_value=2.0).to(dev)
        with torch.no_grad():
            m.fn.copy_(torch.randn_like(m.fn) * 1e-3)
            m.base.copy_(torch.randn_like(m.base) * 0.1)
            m.scale.copy_(torch.rand_like(m.scale) + 0.5)
        return m

    def quant_w(n, k):
        w = (torch.randn(n, k, device=dev) * 0.05).to(torch.bfloat16)
        wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True, alignment=32)
        return wq, wsf

    class FakeLayer:
        _mhc_boundary = m41.DeepseekV41DecoderLayer._mhc_boundary

        def __init__(self, idx, deferred, bad=False):
            self.layer_idx = idx
            self._v41_defer_post_mapping = deferred
            self._mhc_split_stream = None
            self.hc_ffn = make_mhc()
            self.post_attention_layernorm = types.SimpleNamespace(
                weight=(1 + 0.1 * torch.randn(hidden, device=dev)).to(torch.bfloat16),
                variance_epsilon=eps_rms)
            self.mapping = types.SimpleNamespace(rank=0, enable_attention_dp=False)
            self.mlp = types.SimpleNamespace(gate=types.SimpleNamespace(is_hashed=False))
            self.w1 = quant_w(2 * inter, hidden)
            self.w2 = quant_w(hidden, inter)
            self.bad = bad

        def forward_MoE(self, hidden_states, attn_metadata, spec_metadata=None, input_ids=None,
                        hidden_states_normalized=False):
            assert hidden_states_normalized and spec_metadata is None
            a, asf = torch.ops.trtllm.mxfp8_quantize(hidden_states, True, alignment=32)
            h = torch.ops.trtllm.flashinfer_mm_mxfp8(a, asf, self.w1[0], self.w1[1], torch.bfloat16)
            gte, up = h.chunk(2, dim=-1)
            h = (torch.nn.functional.silu(gte.float()) * up.float()).to(torch.bfloat16)
            if self.bad:
                _ = h.sum().item()  # host sync: capture must fail
            b, bsf = torch.ops.trtllm.mxfp8_quantize(h.contiguous(), True, alignment=32)
            out = torch.ops.trtllm.flashinfer_mm_mxfp8(b, bsf, self.w2[0], self.w2[1], torch.bfloat16)
            if input_ids is not None:
                out = out + input_ids.to(out.dtype).unsqueeze(-1) * 0
            return out

    def inputs(m):
        x_attn = torch.randn(m, hidden, device=dev).to(torch.bfloat16)
        residual = torch.randn(m, mult, hidden, device=dev).to(torch.bfloat16)
        post = torch.rand(m, mult, 1, device=dev) * 2
        comb = torch.softmax(torch.randn(m, mult, mult, device=dev), -1)
        pre = torch.rand(m, mult, 1, device=dev)
        ids = torch.randint(0, 1000, (m,), device=dev, dtype=torch.int32)
        return x_attn, residual, post, comb, pre, ids

    meta = types.SimpleNamespace(csa2_replay_query_rows=torch.arange(M, device=dev),
                                 is_cuda_graph=False, num_generations=0)
    layers = [FakeLayer(20, True), FakeLayer(21, False)]

    # Eligibility probes.
    probe = {}
    L = layers[0]
    for rows in (127, 128, 129):
        probe[f"rows{rows}"] = g.eligible(L, torch.empty(rows, hidden, device=dev), meta, None, None)
    probe["gen1"] = g.eligible(L, torch.empty(M, hidden, device=dev),
                               types.SimpleNamespace(csa2_replay_query_rows=1, is_cuda_graph=False,
                                                     num_generations=1), None, None)
    probe["is_cuda_graph"] = g.eligible(L, torch.empty(M, hidden, device=dev),
                                        types.SimpleNamespace(csa2_replay_query_rows=1,
                                                              is_cuda_graph=True, num_generations=0),
                                        None, None)
    probe["no_replay_rows"] = g.eligible(L, torch.empty(M, hidden, device=dev),
                                         types.SimpleNamespace(csa2_replay_query_rows=None,
                                                               is_cuda_graph=False, num_generations=0),
                                         None, None)
    probe["spec_metadata"] = g.eligible(L, torch.empty(M, hidden, device=dev), meta, object(), None)
    log(f"[C] eligibility {probe}")
    probe_ok = probe == {"rows127": False, "rows128": True, "rows129": False, "gen1": False,
                         "is_cuda_graph": False, "no_replay_rows": False, "spec_metadata": False}

    steps = []
    graphed = {l.layer_idx: 0 for l in layers}
    mismatches = 0
    host = {"eager_us": [], "graph_us": []}
    for step in range(10):
        rec = {"step": step}
        for l in layers:
            inp = inputs(M)
            ref_in = tuple(t.clone() for t in inp)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ref = g.moe_window(l, *ref_in[:5], meta, ref_in[5])
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            assert g.eligible(l, inp[0], meta, None, None)
            t2 = time.perf_counter()
            out = g.run(l, *inp[:5], meta, inp[5])
            t3 = time.perf_counter()
            torch.cuda.synchronize()
            if out is None:
                rec[l.layer_idx] = "eager(warm)"
                continue
            graphed[l.layer_idx] += 1
            if step >= 2:
                host["eager_us"].append((t1 - t0) * 1e6)
                host["graph_us"].append((t3 - t2) * 1e6)
            eq = [bool(torch.equal(a, b)) and a.dtype == b.dtype and a.shape == b.shape
                  for a, b in zip(out, ref)]
            fin = all(bool(torch.isfinite(a.float()).all()) for a in out)
            if not all(eq) or not fin or len(out) != len(ref):
                mismatches += 1
            rec[l.layer_idx] = {"equal": eq, "finite": fin, "n_out": len(out)}
        steps.append(rec)
        log(f"[C] step {step}: {rec}")

    states = {l.layer_idx: (l._tail_moe_graph_state.graph is not None,
                            l._tail_moe_graph_state.replays) for l in layers}
    log(f"[C] states (captured, replays) {states}; graphed steps {graphed}; mismatches {mismatches}")
    med = lambda v: sorted(v)[len(v) // 2] if v else None  # noqa: E731
    host_med = {"eager_window_us_med": med(host["eager_us"]), "graph_run_us_med": med(host["graph_us"])}
    log(f"[C] host per window (synthetic chain, includes input copies) {host_med}")

    # Negative control: a host sync inside the window must fail the capture and fall back.
    bad = FakeLayer(22, False, bad=True)
    inp = inputs(M)
    first = g.run(bad, *inp[:5], meta, inp[5])  # warm (eager, returns None)
    second = g.run(bad, *inp[:5], meta, inp[5])  # capture -> fails -> None
    torch.cuda.synchronize()
    neg = {"warm_returned_none": first is None, "capture_failed_fallback": second is None,
           "disabled": g._disabled}
    log(f"[C] negative control {neg}")
    # Stream still usable after the failed capture.
    x = torch.randn(4, 4, device=dev)
    ok_after = bool(torch.isfinite((x @ x).sum()).item())
    ok = (probe_ok and mismatches == 0 and all(v >= 9 for v in graphed.values())
          and all(s[0] for s in states.values()) and all(neg.values()) and ok_after)
    return ok, {"probe": probe, "graphed_steps": graphed, "mismatches": mismatches,
                "states": {str(k): v for k, v in states.items()}, "host": host_med,
                "negative_control": neg, "stream_ok_after_failed_capture": ok_after}


def main():
    a_ok, a = import_receipts()
    hits = code_receipt()
    c_ok, c = runtime_check()
    # Code receipt: sync patterns only allowed in lines that are not on the M=128 eager path:
    # none are expected in the listed functions.
    b_ok = all(not v for v in hits.values())
    RESULT.update({
        "passed": bool(a_ok and b_ok and c_ok),
        "import_receipts_ok": a_ok, "import_receipts": a,
        "code_receipt_ok": b_ok, "code_receipt_hits": hits,
        "runtime_ok": c_ok, "runtime": c,
        "compared_against": "eager moe_window on the same inputs (torch.equal), protected op = "
                            "main's inline MoE window code path",
    })
    print(json.dumps(RESULT), flush=True)


if __name__ == "__main__":
    main()
