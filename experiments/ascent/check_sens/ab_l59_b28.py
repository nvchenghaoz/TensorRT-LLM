"""In-process A/X/A' for branch 28 (idea 45): layer-59 attention on read rows only.

Template: experiments/ascent/mf2_llm_ab.py (top-lead ruling 1083) on the
knowledge-618 mix: own natural-text prompts (make_text, seed 20261114),
ISL 300/1k/2k/4k/8k x c1 (sent singly) / c4 (groups 4,4,2) / c10 (one group
of 10), NPER prompts per class, 1 greedy token, top-20 logprobs,
TLLM_WORKER_USE_SINGLE_PROCESS=1, batch_wait_timeout_ms 1500.

Switch: flashinfer._LAST_LAYER_ATTN_ROWS (the TRTLLM_GEMMA4_LAST_LAYER_ATTN_ROWS
kill switch, read at call time by Gemma4TextModel.forward and the backend).
Passes: A (off) / X (on) / A2 (off), then a re-send with X on in the same
process (classes 8k, 1k, 8k: 2 c1 prompts per ISL plus the first 4x8k and
4x1k groups), judged against A.
Per pass: read-rows dispatches and fallbacks, model forwards with the
final-norm gather (apply_final_norm=False), and SHA1 of layer 59's KV slots
for every token of every step (compared step by step with A).
Stage 2 (item 2, zero-row invariance on real activations): one 1x8000 step
and one 16x300 (4800-row) step under A and under X, capturing layer 59's
o_proj input/output, the junction h, gate_up's FP4 payload + swizzled SF and
output, the gelu FP4 payload + SF and down_proj output, and the layer output;
read rows are compared byte for byte, and every AutoTuner pick inside layer
59 is recorded in both arms.

usage: ab_l59_b28.py <checkpoint> [nper=10]
"""
import hashlib
import json
import os
import random
import sys
import threading
import time


def _heartbeat():
    t0 = time.time()
    while True:
        time.sleep(60)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


WORDS = ("the of and to in is was for on that with as by at from his her it an were are which this be "
         "or has had not but have one their they its new first after who been also two time more year "
         "city river music school system state world history data model energy market water light "
         "village company science paper season team game church station family garden island bridge "
         "language theory number field road power north south east west small large early late").split()

TOL = {300: 4.8174, 1000: 0.5713, 2000: 5.0, 4000: 0.125, 8000: 0.125}


def make_text(rng, n_words):
    out = []
    while len(out) < n_words:
        k = rng.randint(6, 18)
        s = [rng.choice(WORDS) for _ in range(k)]
        s[0] = s[0].capitalize()
        if rng.random() < 0.3:
            s.append(str(rng.randint(1, 2030)))
        out.append(" ".join(s) + ".")
    return " ".join(out)


def main():
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
    threading.Thread(target=_heartbeat, daemon=True).start()
    ck = sys.argv[1]
    nper = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch import autotuner as at
    from tensorrt_llm._torch.attention.backends import flashinfer as fi
    from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import sf_swizzled_offsets
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.utils import Fp4QuantizedTensor
    from tensorrt_llm.llmapi import KvCacheConfig

    cfg = json.load(open(os.path.join(ck, "config.json")))
    tc = cfg.get("text_config", cfg)
    LAST = tc["num_hidden_layers"] - 1
    print("config:", json.dumps({k: tc.get(k) for k in (
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "num_global_key_value_heads", "head_dim", "global_head_dim", "intermediate_size",
        "sliding_window")}), "last layer type:", tc["layer_types"][-1], flush=True)

    tok = AutoTokenizer.from_pretrained(ck)
    rng = random.Random(20261114)
    # classes: (isl, conc, [groups of prompt ids])
    classes = []
    for isl in (300, 1000, 2000, 4000, 8000):
        for conc in (1, 4, 10):
            ps = []
            for _ in range(nper):
                ids = tok(make_text(rng, isl * 2), add_special_tokens=False)["input_ids"][: isl - 1]
                ids = [tok.bos_token_id or 2] + ids
                assert len(ids) == isl
                ps.append(ids)
            groups = [ps[i:i + conc] for i in range(0, len(ps), conc)]
            classes.append((isl, conc, groups))
    n_prompts = sum(len(g) for _, _, gs in classes for g in gs)
    print(f"built {n_prompts} prompts", flush=True)

    # ---- instrumentation ----
    counts = {"model_fwd": 0, "final_norm_gather": 0}
    _fwd = mg.Gemma4TextModel.forward

    def fwd(self, *a, **k):
        counts["model_fwd"] += 1
        if k.get("apply_final_norm", True) is False:
            counts["final_norm_gather"] += 1
        return _fwd(self, *a, **k)

    mg.Gemma4TextModel.forward = fwd

    KV = {"on": True, "hashes": []}
    _fimpl = fi.FlashInferAttention.forward_impl

    def fimpl(self, q, k, v, metadata, *a, **kw):
        r = _fimpl(self, q, k, v, metadata, *a, **kw)
        if KV["on"] and self.layer_idx == LAST and metadata.num_contexts > 0 and k is not None:
            kvc = metadata.kv_cache_manager.get_buffers(self.layer_idx, kv_layout=metadata.kv_layout)
            nt = metadata.num_tokens
            ps = metadata.page_size
            bi = metadata.batch_indices[:nt].long()
            pos = metadata.positions[:nt].long()
            indptr = metadata.paged_kv_indptr.long()
            idx = metadata._paged_kv_indices.long()
            page = idx[indptr[bi] + pos // ps]
            ent = pos % ps
            if metadata.kv_layout == "HND":
                slots = kvc[page, :, :, ent, :]
            else:
                slots = kvc[page, :, ent, :, :]
            KV["hashes"].append((nt, hashlib.sha1(slots.contiguous().view(torch.uint8).cpu().numpy()
                                                  .tobytes()).hexdigest()))
        return r

    fi.FlashInferAttention.forward_impl = fimpl

    CAP = {"on": False, "in": False, "layer": None, "steps": [], "cur": None}

    def rows_bytes(t, rows):
        return t[rows].contiguous().view(torch.uint8).clone()

    _lfwd = mg.Gemma4DecoderLayer.forward

    def lfwd(self, *a, **k):
        if not (CAP["on"] and self.layer_idx == LAST):
            return _lfwd(self, *a, **k)
        md = k.get("attn_metadata")
        rows = (torch.cumsum(md.seq_lens_cuda, 0, dtype=torch.long) - 1)
        cur = {"M": int(md.num_tokens), "rows": rows, "rec": {}, "tactics": []}
        CAP["layer"], CAP["cur"], CAP["in"] = self, cur, True
        try:
            out = _lfwd(self, *a, **k)
        finally:
            CAP["in"] = False
        o = out[0] if isinstance(out, tuple) else out
        cur["rec"]["layer_out"] = rows_bytes(o, rows)
        CAP["steps"].append(cur)
        return out

    mg.Gemma4DecoderLayer.forward = lfwd

    _linfwd = Linear.forward

    def linfwd(self, input, *a, **k):  # noqa: A002 (Linear.forward's parameter name)
        x = input
        out = _linfwd(self, input, *a, **k)
        if CAP["in"]:
            L = CAP["layer"]
            cur = CAP["cur"]
            rows = cur["rows"]
            name = {id(L.self_attn.o_proj): "o_proj", id(L.mlp.gate_up_proj): "gate_up",
                    id(L.mlp.down_proj): "down"}.get(id(self))
            if name is not None:
                rec = cur["rec"]
                if isinstance(x, Fp4QuantizedTensor):
                    fp4 = x.fp4_tensor.view(torch.uint8)
                    m, kb = fp4.shape[0], fp4.shape[1]
                    nvec = kb * 2 // 16
                    off = sf_swizzled_offsets(m, nvec, fp4.device).view(m, nvec)
                    sf = x.scaling_factor.reshape(-1).view(torch.uint8)
                    rec[name + "_in_fp4"] = rows_bytes(fp4, rows)
                    rec[name + "_in_sf"] = sf[off[rows].reshape(-1)].clone()
                else:
                    rec[name + "_in"] = rows_bytes(x, rows)
                    if name == "o_proj":
                        mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
                        mask[rows] = False
                        rec["o_proj_in_nonread_zero"] = bool((x[mask].view(torch.int16) == 0).all().item())
                        rec["o_proj_route"] = repr(getattr(self, "custom_cublas_mm_tokens", None)) + \
                            f" use_custom_cublas_mm={getattr(self, 'use_custom_cublas_mm', None)}"
                rec[name + "_out"] = rows_bytes(out, rows)
        return out

    Linear.forward = linfwd

    for fname in ("rmsnorm_residual_add_fp4_quant", "rmsnorm_residual_add"):
        orig = getattr(mg, fname)

        def make(orig, fname):
            def w(*a, **k):
                r = orig(*a, **k)
                if CAP["in"]:
                    h = r[0] if isinstance(r, tuple) else r
                    CAP["cur"]["rec"]["h(" + fname + ")"] = rows_bytes(h, CAP["cur"]["rows"])
                return r
            return w
        setattr(mg, fname, make(orig, fname))

    _choose = at.AutoTuner.choose_one

    def choose(self, custom_op, runners, tuning_config, inputs, **kw):
        r = _choose(self, custom_op, runners, tuning_config, inputs, **kw)
        if CAP["in"]:
            shapes = [tuple(t.shape) if hasattr(t, "shape") else None for t in inputs]
            CAP["cur"]["tactics"].append([custom_op, str(shapes), type(r[0]).__name__, repr(r[1])])
        return r

    at.AutoTuner.choose_one = choose

    llm = LLM(
        model=ck,
        max_num_tokens=8192,
        max_seq_len=8300,
        max_batch_size=64,
        batch_wait_timeout_ms=1500,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.85, enable_block_reuse=False),
    )
    print("loaded", flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def gen(groups):
        res = []
        for g in groups:
            outs = llm.generate([{"prompt_token_ids": p} for p in g], sp, use_tqdm=False)
            for o in outs:
                c = o.outputs[0]
                lp = c.logprobs[0] if c.logprobs else {}
                res.append({"tok": int(c.token_ids[0]),
                            "top": {int(t): float(v.logprob) for t, v in lp.items()}})
        return res

    def stats_snapshot():
        return dict(counts), dict(fi.READ_ROWS_STATS)

    def run_pass(tag, on):
        fi._LAST_LAYER_ATTN_ROWS = on
        c0, s0 = stats_snapshot()
        KV["hashes"] = []
        t0 = time.time()
        out = {}
        for isl, conc, groups in classes:
            out[(isl, conc)] = gen(groups)
        c1, s1 = stats_snapshot()
        calls = {k: c1[k] - c0[k] for k in c1}
        calls.update({"read_rows_" + k: s1[k] - s0[k] for k in s1})
        print(f"[{tag}] on={on} {time.time() - t0:.1f}s calls={calls} kv_steps={len(KV['hashes'])}",
              flush=True)
        return out, calls, list(KV["hashes"])

    # warmup both paths
    for on in (False, True):
        fi._LAST_LAYER_ATTN_ROWS = on
        gen([[classes[0][2][0][0]], [classes[-1][2][0][0]]])

    passes, calls, kvh = {}, {}, {}
    for tag, on in (("A", False), ("X", True), ("A2", False)):
        passes[tag], calls[tag], kvh[tag] = run_pass(tag, on)

    def compare(a, b):
        tot = eq = flips = strict = fails = 0
        maxd = 0.0
        per = {}
        for key in a:
            pa, pb = a[key], b[key]
            isl, conc = key
            e = per.setdefault(f"isl{isl}@c{conc}", {"n": 0, "bitwise": 0, "max_d": 0.0})
            for x, y in zip(pa, pb):
                tot += 1
                same = x["tok"] == y["tok"] and x["top"] == y["top"]
                eq += same
                flips += x["tok"] != y["tok"]
                ref = x["tok"]
                if ref not in y["top"]:
                    d = float("inf")
                else:
                    d = abs(x["top"].get(ref, float("nan")) - y["top"][ref])
                strict += not same
                w = 2 if conc == 10 else 1
                fails += w * (d > TOL[isl])
                maxd = max(maxd, d)
                e["n"] += 1
                e["bitwise"] += same
                e["max_d"] = max(e["max_d"], d)
        return {"total": tot, "bitwise_equal": eq, "token_flips": flips, "not_bitwise": strict,
                "rule_iii_fails_scaled": fails, "max_abs_dlogprob_ref_top1": maxd, "per_class": per}

    res = {"A_vs_A2": compare(passes["A"], passes["A2"]), "A_vs_X": compare(passes["A"], passes["X"]),
           "X_vs_A2": compare(passes["X"], passes["A2"])}
    for k, v in res.items():
        print(k, json.dumps(v), flush=True)
    kv_cmp = {t: {"steps": len(kvh[t]), "equal_to_A": kvh[t] == kvh["A"]} for t in ("X", "A2")}
    print("layer-59 KV slot hashes:", json.dumps(kv_cmp), flush=True)

    # ---- re-send (X on, same process, no reset), classes 8k, 1k, 8k ----
    fi._LAST_LAYER_ATTN_ROWS = True
    byc = {(isl, conc): groups for isl, conc, groups in classes}
    resend_plan = [(8000, 1, [[p] for p in byc[(8000, 1)][0][:1] + byc[(8000, 1)][1][:1]]),
                   (8000, 4, [byc[(8000, 4)][0]]),
                   (1000, 1, [[p] for p in byc[(1000, 1)][0][:1] + byc[(1000, 1)][1][:1]]),
                   (1000, 4, [byc[(1000, 4)][0]])]
    for isl in (300, 2000, 4000):
        resend_plan.append((isl, 1, [[p] for p in byc[(isl, 1)][0][:1] + byc[(isl, 1)][1][:1]]))
    resend_plan.append((8000, 4, [byc[(8000, 4)][0]]))
    s0 = dict(fi.READ_ROWS_STATS)
    resend = {}
    for isl, conc, groups in resend_plan:
        got = gen(groups)
        if conc == 1:
            ref = passes["A"][(isl, 1)][:2]
        else:
            ref = passes["A"][(isl, conc)][:len(got)]
        diffs = sum(1 for x, y in zip(ref, got) if not (x["tok"] == y["tok"] and x["top"] == y["top"]))
        key = f"isl{isl}@c{conc}"
        r = resend.setdefault(key, {"n": 0, "diff_vs_A": 0})
        r["n"] += len(got)
        r["diff_vs_A"] += diffs
    s1 = dict(fi.READ_ROWS_STATS)
    resend_ok = all(v["diff_vs_A"] == 0 for v in resend.values())
    print("re-send vs A:", json.dumps(resend), "read_rows ran", s1["ran"] - s0["ran"], flush=True)

    # ---- stage 2: zero-row invariance on real layer-59 activations ----
    KV["on"] = False
    cap_groups = [("M8000", [byc[(8000, 1)][0][0]]),
                  ("M4800", [p for g in byc[(300, 10)] for p in g][:10] + [p for g in byc[(300, 4)] for p in g][:6])]
    caps = {}
    for tag, on in (("A", False), ("X", True)):
        fi._LAST_LAYER_ATTN_ROWS = on
        for name, ps in cap_groups:
            CAP["steps"] = []
            CAP["on"] = True
            gen([ps])
            CAP["on"] = False
            caps[(tag, name)] = list(CAP["steps"])
    L = CAP["layer"]
    scales = {}
    for nm, mod in (("gate_up_proj", L.mlp.gate_up_proj), ("down_proj", L.mlp.down_proj)):
        s = getattr(mod, "input_scale", None)
        scales[nm] = {"input_scale": None if s is None else [float(x) for x in s.float().reshape(-1)[:4].cpu()],
                      "dtype": str(getattr(s, "dtype", None)), "shape": list(getattr(s, "shape", [])),
                      "force_dynamic_quantization": getattr(mod, "force_dynamic_quantization", None),
                      "pre_quant_scale": None if getattr(mod, "pre_quant_scale", None) is None else "SET"}
    print("layer-59 static scales:", json.dumps(scales), flush=True)
    inv = {}
    inv_ok = True
    for name, _ in cap_groups:
        sa, sx = caps[("A", name)], caps[("X", name)]
        r = {"steps_A": [s["M"] for s in sa], "steps_X": [s["M"] for s in sx]}
        ok = len(sa) == len(sx) and len(sa) > 0
        for i, (a, x) in enumerate(zip(sa, sx)):
            st = {"M": a["M"], "read_rows": int(a["rows"].numel())}
            for k in a["rec"]:
                if k in ("o_proj_route",):
                    st[k] = a["rec"][k]
                    continue
                if k == "o_proj_in_nonread_zero":
                    st["A_o_proj_in_nonread_zero"] = a["rec"][k]
                    st["X_o_proj_in_nonread_zero"] = x["rec"].get(k)
                    ok &= bool(x["rec"].get(k))
                    continue
                eqb = k in x["rec"] and torch.equal(a["rec"][k], x["rec"][k])
                st[k] = eqb
                ok &= eqb
            st["tactics_equal"] = a["tactics"] == x["tactics"]
            st["tactics_A"] = a["tactics"]
            ok &= st["tactics_equal"]
            r[f"step{i}"] = st
        r["passed"] = bool(ok)
        inv_ok &= ok
        inv[name] = r
        print("zero-row invariance", name, json.dumps(r), flush=True)
    llm.shutdown()

    aa, ax, xa = res["A_vs_A2"], res["A_vs_X"], res["X_vs_A2"]
    ok = (aa["bitwise_equal"] == aa["total"] and ax["bitwise_equal"] == ax["total"]
          and xa["bitwise_equal"] == xa["total"] and calls["X"]["read_rows_ran"] > 0
          and calls["A"]["read_rows_ran"] == 0 and calls["A2"]["read_rows_ran"] == 0
          and calls["X"]["read_rows_ran"] + calls["X"].get("read_rows_below_min_tokens", 0)
          == calls["X"]["final_norm_gather"] and calls["X"]["read_rows_fallback"] == 0
          and all(v["equal_to_A"] for v in kv_cmp.values()) and resend_ok and inv_ok)
    print(json.dumps({"passed": bool(ok), "compared": "in-process LLM, layer-59 read-rows attention off/on/off "
                      "(kill switch), first-token id + top-20 logprobs bitwise on the knowledge-618 mix; layer-59 "
                      "KV slot hashes per step; re-send; zero-row invariance of layer 59's post-attention chain "
                      "on real activations at M=8000 and 4800",
                      "prompts_per_pass": n_prompts, "A_vs_A2": aa, "A_vs_X": ax, "X_vs_A2": xa,
                      "calls": calls, "kv_slots": kv_cmp, "resend": resend, "static_scales": scales,
                      "zero_row_invariance": inv}))


if __name__ == "__main__":
    main()
