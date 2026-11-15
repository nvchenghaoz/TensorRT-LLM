#!/usr/bin/env python3
"""Idea 23 stage 2: full-model in-process checks of the direct FP8 paged-KV store.

One LLM per job, with this tree's engine.json overlay (BCG on, as main serves),
TLLM_WORKER_USE_SINGLE_PROCESS=1. The direct store is flipped between passes with
TRTLLM_GEMMA4_PREP_PAGED_KV (read at call time); "=0" is main's path (Route A
temporaries + FlashInfer append, the prep kernel bytes equal to main's).

Hooks (observation only, results returned unchanged):
  * modeling_gemma4.gemma4_fused_qkv_norm_rope_quant: counts direct-store calls vs
    temporaries calls, and any direct-store call while capturing (must be 0);
  * flashinfer._append_paged_kv_cache: counts append calls;
  * FlashInferAttention.forward_impl: after each layer (eager steps and the eager
    attention break of replayed steps, never while capturing), a GPU checksum of the
    pool bytes at every slot this forward wrote (the append kernel's addresses) and of
    the attention output; per-pass lists are compared bitwise.

Modes:
  ab   : item 2 (BCG replay): 1x300, 1x1000, 1x2000, 2x1000 groups in passes
         R1 (replay, direct on), R0 (replay, off), E0 (replay disabled via has_graph,
         off = main eager), E1 (replay disabled, on); then item 3: 200 own prompts
         (20 per ISL 300/1k/2k/4k/8k at c=1, 5 groups of 4 per ISL at c=4) plus a 4x8k
         and a 4x1k group sent together, passes off0, on0, off1, on1.
         Compares token ids + top-20 logprobs and per-layer digests bitwise; logs
         replay/eager step counts and hook counters per pass.
  nsys : isl8k@1 requests inside NVTX ranges "prep_direct_on" and "prep_direct_off".
Last stdout line: JSON.
"""
import json
import os
import sys
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
os.environ.setdefault("TRTLLM_FI_HOST_PREFILL_PLAN", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

CKPT = os.environ.get("CKPT") or (sys.argv[2] if len(sys.argv) > 2 else None)
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_JSON = os.path.join(HERE, "..", "..", "ascent-config", "engine.json")
FLAG = "TRTLLM_GEMMA4_PREP_PAGED_KV"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def print_dims():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "global_head_dim", "num_global_key_value_heads", "sliding_window", "attention_k_eq_v"]
    log("config: " + json.dumps({k: tc.get(k) for k in keys}))
    flags = {k: os.environ.get(k, "<default>") for k in (
        "TRTLLM_FI_HOST_PREFILL_PLAN", "TRTLLM_FI_PLAN_PARAMS_MEMO", "TLLM_KVCM_V2_VIEW_CACHE",
        "TRTLLM_GEMMA4_LAST_LAYER_ATTN_ROWS", "FLASHINFER_FA2_D512_TILE64", "TRTLLM_GEMMA4_FUSED_QKV_PREP",
        "TLLM_AUTOTUNER_CACHE_PATH")}
    log("flags: " + json.dumps(flags))


def overlay():
    if os.path.exists(ENGINE_JSON):
        with open(ENGINE_JSON) as f:
            return json.load(f)
    return {}


class Prompts:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def make(self, isl):
        return [int(x) for x in self.rng.integers(1000, 200000, size=isl)]


def build_llm(**extra):
    ov = dict(overlay())
    log(f"engine overlay: {json.dumps(ov)}")
    for k, v in (ov.pop("env", None) or {}).items():
        os.environ[k] = str(v)
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    t0 = time.time()
    llm = LLM(model=CKPT,
              kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False, free_gpu_memory_fraction=0.8),
              max_num_tokens=8192, max_batch_size=64, max_seq_len=9216, **ov, **extra)
    log(f"startup {time.time() - t0:.1f}s")
    return llm


def sp(logprobs=20):
    from tensorrt_llm import SamplingParams
    kw = dict(max_tokens=1, temperature=0.0, ignore_eos=True)
    if logprobs:
        kw["logprobs"] = logprobs
    return SamplingParams(**kw)


def run_group(llm, prompts, params):
    """Enqueue every prompt of a group before the executor thread can fetch any."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(0.5)
    try:
        futs = [llm.generate_async(p, sampling_params=params) for p in prompts]
    finally:
        sys.setswitchinterval(old)
    return [f.result() for f in futs]


def summarize(o):
    c = o.outputs[0]
    lps = []
    if c.logprobs:
        for d in c.logprobs:
            if isinstance(d, dict):
                lps.append(sorted((int(k), float(getattr(v, "logprob", v))) for k, v in d.items()))
            else:
                lps.append(float(d))
    return {"ids": list(c.token_ids), "lp": lps}


# ----------------------------------------------------------------------------- hooks
CNT = {"direct": 0, "temporaries": 0, "direct_while_capturing": 0, "append": 0, "digest_layers": 0}
DIGESTS = []


def _checksum(t):
    b = t.contiguous().view(torch.uint8).view(-1)
    pad = (-b.numel()) % 4
    if pad:
        b = torch.cat([b, b.new_zeros(pad)])
    w = b.view(torch.int32).to(torch.int64)
    idx = torch.arange(w.numel(), device=w.device, dtype=torch.int64) % 1000003 + 1
    return (int(w.sum().item()), int((w * idx).sum().item()))


def install_hooks():
    import tensorrt_llm._torch.attention.backends.flashinfer as fi
    import tensorrt_llm._torch.models.modeling_gemma4 as mg

    orig_prep = mg.gemma4_fused_qkv_norm_rope_quant

    def prep(*a, **kw):
        if kw.get("paged_kv_cache") is not None:
            CNT["direct"] += 1
            if torch.cuda.is_current_stream_capturing():
                CNT["direct_while_capturing"] += 1
        else:
            CNT["temporaries"] += 1
        return orig_prep(*a, **kw)

    mg.gemma4_fused_qkv_norm_rope_quant = prep

    orig_append = fi._append_paged_kv_cache

    def append(*a, **kw):
        CNT["append"] += 1
        return orig_append(*a, **kw)

    fi._append_paged_kv_cache = append

    orig_fwd = fi.FlashInferAttention.forward_impl

    def fwd(self, *a, **kw):
        r = orig_fwd(self, *a, **kw)
        if torch.cuda.is_current_stream_capturing():
            return r
        md = kw.get("metadata")
        out = kw.get("output")
        if md is None or md.kv_cache_manager is None or out is None:
            return r
        layer = self.layer_idx
        kv = md.kv_cache_manager.get_buffers(layer, kv_layout=md.kv_layout)
        n = md.num_tokens
        ps = kv.shape[2] if md.kv_layout == "NHD" else kv.shape[3]
        bidx = md.batch_indices[:n].long()
        pos = md.positions[:n].long()
        flat = md.paged_kv_indptr.long()[bidx] * ps + pos
        idx = md.get_paged_kv_indices_for_layer(layer).long()
        page = idx[flat // ps]
        entry = flat % ps
        view = kv if md.kv_layout == "NHD" else kv.permute(0, 1, 3, 2, 4)
        slots = view[page, :, entry]
        DIGESTS.append((int(layer), int(n), _checksum(slots), _checksum(out[:n])))
        CNT["digest_layers"] += 1
        return r

    fi.FlashInferAttention.forward_impl = fwd
    return fi


def step_counts():
    try:
        from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
        return dict(getattr(_g, "_BCG_STEP_COUNTS", {}) or {})
    except Exception:  # noqa: BLE001
        return None


def run_pass(llm, tag, groups, params, flag_on, fi):
    os.environ[FLAG] = "1" if flag_on else "0"
    c0, s0 = dict(CNT), step_counts()
    rr0 = dict(fi.READ_ROWS_STATS)
    DIGESTS.clear()
    t0 = time.time()
    res = []
    for _, ps in groups:
        outs = run_group(llm, ps, params)
        res.append([summarize(o) for o in outs])
    torch.cuda.synchronize()
    s1 = step_counts()
    info = {"flag": os.environ[FLAG], "secs": round(time.time() - t0, 1),
            "counters": {k: CNT[k] - c0[k] for k in CNT},
            "bcg_steps": ({k: s1[k] - s0[k] for k in s1} if s1 else None),
            "read_rows": {k: fi.READ_ROWS_STATS[k] - rr0[k] for k in rr0}}
    log(f"pass {tag}: {json.dumps(info)}")
    return res, list(DIGESTS), info


def cmp(a, b):
    fa = [x for g in a for x in g]
    fb = [x for g in b for x in g]
    diff = [i for i, (x, y) in enumerate(zip(fa, fb)) if x != y]
    return {"n": len(fa), "n_diff": len(diff) + abs(len(fa) - len(fb)), "first": diff[:5]}


def cmp_dig(a, b):
    """Per-layer-call digests (layer, rows, kv_slots_checksum, attn_out_checksum)."""
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    kv_diff = [i for i, (x, y) in enumerate(zip(a, b)) if x[:3] != y[:3]]
    out_diff_layers = sorted({a[i][0] for i in diff if a[i][:3] == b[i][:3]})
    return {"n_layer_calls": len(a), "n_diff": len(diff) + abs(len(a) - len(b)), "first": diff[:5],
            "n_kv_diff": len(kv_diff) + abs(len(a) - len(b)), "out_only_diff_layers": out_diff_layers}


def mode_ab():
    print_dims()
    llm = build_llm(batch_wait_timeout_ms=100.0)
    fi = install_hooks()
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
    replay_on = [True]
    orig_has_graph = BreakableCUDAGraphRunner.has_graph
    BreakableCUDAGraphRunner.has_graph = lambda self, n: replay_on[0] and orig_has_graph(self, n)
    params = sp(20)
    warm = Prompts(5)
    for isl in (300, 1000, 2000, 8000):
        run_group(llm, [warm.make(isl)], params)
    run_group(llm, [warm.make(1000) for _ in range(2)], params)
    log(f"capture-time counters after startup+warmup: {json.dumps(CNT)}")
    cap_direct = CNT["direct_while_capturing"]

    # ---- item 2: BCG replayed buckets
    p2 = Prompts(9090)
    g2 = []
    for _ in range(3):
        g2.append(("1x300", [p2.make(300)]))
        g2.append(("1x1000", [p2.make(1000)]))
        g2.append(("1x2000", [p2.make(2000)]))
        g2.append(("2x1000", [p2.make(1000) for _ in range(2)]))
    passes2 = {}
    for tag, rep, on in (("R1", True, True), ("R0", True, False), ("E0", False, False), ("E1", False, True),
                         ("R1b", True, True)):
        replay_on[0] = rep
        passes2[tag] = run_pass(llm, tag, g2, params, on, fi)
    replay_on[0] = True
    item2 = {}
    for a, b in (("R0", "R1"), ("E0", "R1"), ("E0", "R0"), ("E0", "E1"), ("R1", "R1b")):
        item2[f"{b}_vs_{a}"] = {"outputs": cmp(passes2[a][0], passes2[b][0]),
                                "layer_digests": cmp_dig(passes2[a][1], passes2[b][1])}
        log(f"item2 {b} vs {a}: {json.dumps(item2[f'{b}_vs_{a}'])}")
    # Same graph mode (replay vs replay, eager vs eager): everything bitwise. Replay vs
    # eager (main's own BCG property): outputs and every KV slot bitwise; the attention
    # output may differ only on the last layer, whose eager steps run main's read-rows
    # FA2 call (unread rows zero) and whose replayed steps run the full-row call.
    last_layer = max(d[0] for d in passes2["E0"][1])
    item2_ok = True
    for k, v in item2.items():
        same_mode = k in ("R1_vs_R0", "E1_vs_E0", "R1b_vs_R1")
        if same_mode:
            item2_ok &= v["outputs"]["n_diff"] == 0 and v["layer_digests"]["n_diff"] == 0
        else:
            item2_ok &= (v["outputs"]["n_diff"] == 0 and v["layer_digests"]["n_kv_diff"] == 0
                         and set(v["layer_digests"]["out_only_diff_layers"]) <= {last_layer})
    r_info = [passes2[t][2] for t in ("R1", "R0", "R1b")]
    replay_ok = all(i["bcg_steps"] and i["bcg_steps"].get("eager", 0) == 0 and i["counters"]["direct"] == 0
                    for i in r_info)
    e1 = passes2["E1"][2]
    e1_ok = e1["counters"]["append"] == 0 and e1["counters"]["direct"] > 0
    log(f"item2 ok={item2_ok} replay_steps_ok={replay_ok} eager_direct_ok={e1_ok} "
        f"direct_while_capturing={cap_direct}")

    # ---- item 3: 200 own prompts at c=1 and c=4 + 4x8k and 4x1k groups
    p3 = Prompts(20261115)
    g3 = []
    for isl in (300, 1000, 2000, 4000, 8000):
        g3 += [(f"isl{isl}@1", [p3.make(isl)]) for _ in range(20)]
    for isl in (300, 1000, 2000, 4000, 8000):
        g3 += [(f"isl{isl}@4", [p3.make(isl) for _ in range(4)]) for _ in range(5)]
    g3.append(("4x8k", [p3.make(8000) for _ in range(4)]))
    g3.append(("4x1k", [p3.make(1000) for _ in range(4)]))
    log(f"item3 set: {len(g3)} groups, {sum(len(g) for _, g in g3)} prompts")
    passes3 = {}
    for tag, on in (("off0", False), ("on0", True), ("off1", False), ("on1", True)):
        passes3[tag] = run_pass(llm, tag, g3, params, on, fi)
    os.environ[FLAG] = "1"
    item3 = {}
    for a, b in (("off0", "off1"), ("off0", "on0"), ("off0", "on1"), ("on0", "on1"), ("off1", "on1")):
        item3[f"{b}_vs_{a}"] = {"outputs": cmp(passes3[a][0], passes3[b][0]),
                                "layer_digests": cmp_dig(passes3[a][1], passes3[b][1])}
        log(f"item3 {b} vs {a}: {json.dumps(item3[f'{b}_vs_{a}'])}")
    per_label = {}
    for (lab, _), ga, gb_ in zip(g3, passes3["off0"][0], passes3["on0"][0]):
        e = per_label.setdefault(lab, {"n": 0, "bitwise_equal": 0})
        for x, y in zip(ga, gb_):
            e["n"] += 1
            e["bitwise_equal"] += int(x == y)
    item3_ok = all(v["outputs"]["n_diff"] == 0 and v["layer_digests"]["n_diff"] == 0 for v in item3.values())
    on_info = [passes3[t][2] for t in ("on0", "on1")]
    off_info = [passes3[t][2] for t in ("off0", "off1")]
    switch_ok = (all(i["counters"]["direct"] > 0 for i in on_info)
                 and all(i["counters"]["direct"] == 0 for i in off_info)
                 and all(i["counters"]["append"] > 0 for i in off_info))
    has_lp = bool(passes3["off0"][0][0][0]["lp"])
    passed = bool(item2_ok and replay_ok and e1_ok and cap_direct == 0 and item3_ok and switch_ok and has_lp)
    print(json.dumps({
        "passed": passed,
        "compared": ("token ids + top-20 logprobs and per-layer GPU checksums of every KV slot written + "
                     "attention output, bitwise, TRTLLM_GEMMA4_PREP_PAGED_KV=1 (direct store) vs =0 "
                     "(main path: Route A temporaries + FlashInfer append), one BCG-on process"),
        "item2_bcg": {"ok": item2_ok, "replay_steps_ok": replay_ok, "eager_direct_ok": e1_ok,
                      "direct_while_capturing": cap_direct, "comparisons": item2,
                      "passes": {t: passes2[t][2] for t in passes2}},
        "item3_ab": {"ok": item3_ok, "switch_ok": switch_ok, "comparisons": item3,
                     "passes": {t: passes3[t][2] for t in passes3}, "per_label_on0_vs_off0": per_label},
        "logprobs_present": has_lp,
    }), flush=True)
    BreakableCUDAGraphRunner.has_graph = orig_has_graph


def mode_nsys():
    print_dims()
    llm = build_llm(batch_wait_timeout_ms=100.0)
    params = sp(0)
    pf = Prompts(31337)
    for on in (True, False):
        os.environ[FLAG] = "1" if on else "0"
        run_group(llm, [pf.make(8000)], params)  # untimed warmup per arm
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for tag, on in (("prep_direct_on", True), ("prep_direct_off", False)):
        os.environ[FLAG] = "1" if on else "0"
        torch.cuda.nvtx.range_push(tag)
        for _ in range(2):
            run_group(llm, [pf.make(8000)], params)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        log(f"{tag} done")
    torch.cuda.cudart().cudaProfilerStop()
    os.environ[FLAG] = "1"
    print(json.dumps({"done": True, "ranges": ["prep_direct_on", "prep_direct_off"],
                      "requests_per_range": 2, "isl": 8000}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "ab":
        mode_ab()
    elif sys.argv[1] == "nsys":
        mode_nsys()
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]}")
