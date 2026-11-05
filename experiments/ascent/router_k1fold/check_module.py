"""Idea 313 module-level gate: the real plumbing (mHC.fused_hc_lagged_split(defer_k1) -> DeepseekV4MoE.forward ->
DeepseekV4Gate.fused_forward(k1_fold) and the shared expert's own aux-stream K1), with stub experts.

Per arm (a subprocess with its own env; the parent runs them all and aggregates):
  for M in 1, 2, 4, 8 under with_multi_stream(True) in ONE CUDA graph per M:
    stand-in attention AR (router_prewait/pdl_ext.py producer, 4 CTAs, EARLY trigger, spins 6 us, then writes x_attn)
    -> hc_ffn boundary split (real mHC, tuned tactic) with defer_k1 as the decoder layer decides it
    -> DeepseekV4MoE.forward (real forward / compute_routed_output / DeepseekV4Gate, stub experts = x * 1 on main,
       stub shared expert = x * 1 on the MoE aux stream) -> join.
  Triton / CuTe JIT compiles during capture are counted (must be 0). Then the repeated sequence: 5 warmup + 20
  replays, fresh inputs every replay, x_attn and every output NaN-poisoned before it; every replay compared bitwise
  with the eager reference (mHC.fused_hc_lagged's layer_input = main's K2, and dsv4_fused_router on it): layer_input,
  the routed experts' input, the shared expert's input, logits / ids / weights, MoE output; router counter 0.
  Eager M = 128 (tail-MoE rows) and M = 2048 (prefill): no split, no fold, output == 2 x the reference layer_input.
Arms: default (fold at M <= 2), FOLD=0 (byte-identical path off), MAX_M=4 (fold at M <= 4), aux SLEEP, PDL=0, and the
negative control SKIP_WAIT (router skips griddepcontrol.wait: must be caught at the fold M).
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "router_prewait"))

ARMS = {
    "default": {},
    "fold_off": {"TRTLLM_DSV4_ROUTER_K1_FOLD": "0"},
    "max_m4": {"TRTLLM_DSV4_ROUTER_K1_FOLD_MAX_M": "4"},
    "aux_sleep": {"TRTLLM_MHC_SPLIT_TEST_SLEEP_CYCLES": "200000"},
    "pdl0": {"TRTLLM_ENABLE_PDL": "0"},
    "skip_wait": {"TRTLLM_DSV4_FUSED_ROUTER_TEST_SKIP_WAIT": "1"},
}
HIDDEN = 5120
WARMUP, ITERS = 5, 20


def log(*a):
    print(*a, flush=True)


def run_arm(arm):
    import torch
    import torch.nn as nn

    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl
    from tensorrt_llm._torch.modules.mhc import mhc_lagged_split as split_mod
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._torch.utils import EventType

    import pdl_ext

    dev = torch.device("cuda")
    torch.manual_seed(0)
    gen = torch.Generator(device=dev).manual_seed(31302)
    ext = pdl_ext.ext()
    mhc = mHC(4, HIDDEN, 20, dtype=torch.float32, eps=1e-6, norm_eps=1e-6, sinkhorn_eps=1e-6,
              post_mult_value=2.0).to(dev)
    with torch.no_grad():
        mhc.fn.normal_(0, 1e-2)
        mhc.base.normal_(0, 0.1)
        mhc.scale.copy_(torch.tensor([0.5, 0.7, 0.9]))
    nw = (1 + 0.1 * torch.randn(HIDDEN, device=dev, generator=gen)).to(torch.bfloat16)
    gate = mdl.DeepseekV4Gate(HIDDEN, 384, 6, 1, 1, 1.5, False, dtype=torch.bfloat16).to(dev)
    with torch.no_grad():
        gate.weight.copy_((torch.randn(384, HIDDEN, device=dev, generator=gen) * HIDDEN ** -0.5).to(torch.bfloat16))
        gate.e_score_correction_bias.copy_(torch.randn(384, device=dev, generator=gen) * 0.05)

    class ExpertsStub(nn.Module):
        def forward(self, x, router_logits, **kw):
            self.last = x * 1
            return self.last

    class SharedStub(nn.Module):
        def forward(self, x):
            self.last = x * 1
            return self.last

    moe = mdl.DeepseekV4MoE.__new__(mdl.DeepseekV4MoE)
    nn.Module.__init__(moe)
    moe.gate = gate
    moe.experts = ExpertsStub()
    moe.shared_experts = SharedStub()
    moe.shared_output_scale = None
    moe._shared_row_split_parts = 1
    moe._shared_decode_split = False
    moe._per_layer_dynamic_eplb_enabled = False
    moe.use_dp = False
    moe.top_k = 6
    moe.mapping = type("Mp", (), {"tp_size": 1})()
    moe.aux_stream = torch.cuda.Stream()
    moe.event_dict = {EventType.Main: torch.cuda.Event(), EventType.MoeShared: torch.cuda.Event()}
    aux_mhc = torch.cuda.Stream()
    ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()
    empty = pdl_ext.empty()
    counter_ref = torch.zeros(1, dtype=torch.int32, device=dev)

    import triton
    import triton.compiler.compiler as tcc
    jit = {"n": 0}
    orig_compile = tcc.compile

    def counting_compile(*a, **k):
        if torch.cuda.is_current_stream_capturing():
            jit["n"] += 1
        return orig_compile(*a, **k)
    tcc.compile = counting_compile
    triton.compile = counting_compile

    def fold_wanted(m):
        # DeepseekV41DecoderLayer._k1_fold_eligible, minus the layer attributes
        return (split_mod.K1_FOLD_ENV_ON and 1 <= m <= split_mod.K1_FOLD_MAX_TOKENS
                and moe.k1_fold_supported(m))

    def fresh(m, t):
        g2 = torch.Generator(device=dev).manual_seed(1000 * m + t)
        src = torch.randn(m, HIDDEN, device=dev, generator=g2).to(torch.bfloat16)
        res = torch.randn(m, 4, HIDDEN, device=dev, generator=g2).to(torch.bfloat16)
        post = (2 * torch.rand(m, 4, 1, device=dev, generator=g2)).float()
        comb = torch.rand(m, 4, 4, device=dev, generator=g2)
        comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
        pre = (torch.rand(m, 4, 1, device=dev, generator=g2) + 1e-3).float()
        return src, res, post, comb, pre

    def reference(inp):
        src, res, post, comb, pre = inp
        li = mhc.fused_hc_lagged(src, res, post, comb, pre, norm_weight=nw, norm_eps=1e-6)[4]
        li = li.reshape(src.shape[0], HIDDEN).contiguous()
        ids, wts, lg = torch.ops.trtllm.dsv4_fused_router(li, gate.weight, gate.e_score_correction_bias,
                                                          counter_ref, 1.5)
        return li, lg, ids, wts

    def bits(t):
        t = t.contiguous()
        return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)

    def same(a, b):
        return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))

    report = {"arm": arm, "cases": {}, "jit_in_capture": 0}
    h_all = hashlib.sha256()
    for m in (1, 2, 4, 8):
        bufs = [t.clone() for t in fresh(m, 0)]
        src, res, post, comb, pre = bufs
        xa = torch.empty_like(src)
        with autotune(skip_dynamic_tuning_buckets=True):
            mhc.fused_hc_lagged(src, res, post, comb, pre, norm_weight=nw, norm_eps=1e-6)
        state = {}

        def body():
            ext.producer(src, xa, 6000, 4, 256, empty, True)
            sp = mhc.fused_hc_lagged_split(xa, res, post, comb, pre, nw, 1e-6, aux_mhc, ev0, ev1,
                                           defer_k1=fold_wanted(m))
            assert sp is not None, "split did not engage"
            outs, join = sp
            li = outs[4]
            fold = join.k1_fold
            if fold is not None:
                moe.stash_k1_fold(fold)
            y = moe.forward(li)
            if fold is not None:
                assert fold.done, "fold not consumed"
            join.join()
            lg, ids, wts = gate._routing_method.fused_routing
            state.update(li=li, y=y, routed_in=moe.experts.last, shared_in=moe.shared_experts.last, lg=lg,
                         ids=ids, wts=wts, folded=fold is not None and fold.folded,
                         deferred=fold is not None)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with with_multi_stream(True), torch.cuda.stream(s):
            body()
            body()
            torch.cuda.synchronize()
            jit0 = jit["n"]
            with torch.cuda.graph(g, stream=s):
                body()
            jit_cap = jit["n"] - jit0
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        report["jit_in_capture"] += jit_cap
        bad_iters, per_iter = 0, []
        h = hashlib.sha256()
        for it in range(WARMUP + ITERS):
            inp = fresh(m, it + 1)
            for b, v in zip(bufs, inp):
                b.copy_(v)
            ref_li, ref_lg, ref_ids, ref_wts = reference(inp)
            torch.cuda.synchronize()
            xa.fill_(float("nan"))
            for k in ("li", "y", "routed_in", "shared_in", "lg", "wts"):
                state[k].fill_(float("nan"))
            state["ids"].fill_(-7)
            g.replay()
            torch.cuda.synchronize()
            mis = {
                "li": int(not same(state["li"].reshape(m, HIDDEN), ref_li)),
                "routed_in": int(not same(state["routed_in"], ref_li)),
                "shared_in": int(not same(state["shared_in"], ref_li)),
                "logits": int(not same(state["lg"], ref_lg)),
                "ids": int(not torch.equal(state["ids"], ref_ids)),
                "weights": int(not same(state["wts"], ref_wts)),
                "moe_out": int(not same(state["y"], ref_li + ref_li)),
                "counter": int(int(gate._fused_router_counter.item()) != 0),
            }
            err = (state["li"].reshape(m, HIDDEN).float() - ref_li.float()).abs().nan_to_num(1e30).max().item()
            nb = sum(mis.values())
            bad_iters += int(nb > 0)
            per_iter.append({"iter": it, "warmup": it < WARMUP, "mismatch": mis, "max_abs_li": err})
            for k in ("li", "lg", "ids", "wts", "y"):
                h.update(state[k].contiguous().view(torch.uint8).cpu().numpy().tobytes())
        h_all.update(h.digest())
        report["cases"][str(m)] = {"folded": state["folded"], "jit_in_capture": jit_cap, "bad_iterations": bad_iters,
                                   "iterations": WARMUP + ITERS, "hash": h.hexdigest()[:16],
                                   "first_bad": next((p for p in per_iter if sum(p["mismatch"].values())), None),
                                   "max_abs_li": max(p["max_abs_li"] for p in per_iter)}
        log(f"[{arm}] M={m} folded={state['folded']} jit_in_capture={jit_cap} bad {bad_iters}/{WARMUP + ITERS} "
            f"max|d li| {report['cases'][str(m)]['max_abs_li']:.3g} hash {h.hexdigest()[:16]}")
        del g
    # eager: tail-MoE 128 rows and prefill 2048 rows (no split, no fold)
    for m in (128, 2048):
        inp = fresh(m, 99)
        src, res, post, comb, pre = inp
        sp = mhc.fused_hc_lagged_split(src, res, post, comb, pre, nw, 1e-6, aux_mhc, ev0, ev1, defer_k1=True)
        li = mhc.fused_hc_lagged(src, res, post, comb, pre, norm_weight=nw, norm_eps=1e-6)[4].reshape(m, HIDDEN)
        y = moe.forward(li)
        torch.cuda.synchronize()
        ok = sp is None and getattr(moe, "_k1_fold", None) is None and same(y, li + li)
        report["cases"][str(m)] = {"split": sp is not None, "ok": ok}
        log(f"[{arm}] eager M={m}: split={sp is not None} moe_out == 2*layer_input: {ok}")
    report["hash"] = h_all.hexdigest()[:16]
    print("ARM_RESULT " + json.dumps(report), flush=True)


def main():
    results = {}
    for arm, env in ARMS.items():
        e = dict(os.environ)
        e.update(env)
        log(f"=== arm {arm} env {env}")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "arm", arm], env=e, capture_output=True,
                           text=True, timeout=1500)
        out = p.stdout
        for line in out.splitlines():
            if not line.startswith("ARM_RESULT "):
                log("   ", line)
        tail = [l for l in out.splitlines() if l.startswith("ARM_RESULT ")]
        if p.returncode != 0 or not tail:
            log(p.stderr[-4000:])
            results[arm] = {"error": p.returncode}
            continue
        results[arm] = json.loads(tail[-1][len("ARM_RESULT "):])
    failures = []
    for arm, r in results.items():
        if "error" in r:
            failures.append(f"{arm}: subprocess failed")
            continue
        if r["jit_in_capture"]:
            failures.append(f"{arm}: {r['jit_in_capture']} JIT compiles during capture")
        for m in ("128", "2048"):
            if not r["cases"][m]["ok"]:
                failures.append(f"{arm}: eager M={m} changed")
        for m in ("1", "2", "4", "8"):
            c = r["cases"][m]
            if arm == "skip_wait":
                continue
            if c["bad_iterations"]:
                failures.append(f"{arm}: M={m} {c['bad_iterations']} bad iterations, first {c['first_bad']}")
    exp_fold = {"default": {"1": True, "2": True, "4": False, "8": False},
                "fold_off": {"1": False, "2": False, "4": False, "8": False},
                "max_m4": {"1": True, "2": True, "4": True, "8": False}}
    for arm, exp in exp_fold.items():
        if arm in results and "cases" in results[arm]:
            got = {m: results[arm]["cases"][m]["folded"] for m in exp}
            if got != exp:
                failures.append(f"{arm}: folded {got} != {exp}")
    sw = results.get("skip_wait", {})
    caught = None
    if "cases" in sw:
        caught = {m: sw["cases"][m]["bad_iterations"] for m in ("1", "2")}
        if not all(v > 0 for v in caught.values()):
            failures.append(f"skip_wait negative control not caught at the fold M: {caught}")
    hashes = {a: r.get("hash") for a, r in results.items()}
    same_hash = len({hashes[a] for a in ("default", "fold_off", "max_m4", "aux_sleep", "pdl0") if a in hashes}) == 1
    if not same_hash:
        failures.append(f"output hashes differ across arms: {hashes}")
    print(json.dumps({"passed": not failures, "failures": failures, "skip_wait_bad_iterations": caught,
                      "hashes": hashes,
                      "compared": "fold graph outputs vs eager mHC.fused_hc_lagged layer_input + dsv4_fused_router",
                      "arms": {a: {m: {k: v for k, v in c.items() if k != "first_bad"}
                                   for m, c in r.get("cases", {}).items()} for a, r in results.items()}}))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    import torch
    with torch.no_grad():
        if len(sys.argv) > 2 and sys.argv[1] == "arm":
            run_arm(sys.argv[2])
        else:
            main()
