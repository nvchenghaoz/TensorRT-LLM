"""Idea 9 / branch 30: in-process LLM A/X/A' + re-send + real-activation capture.

A copy of experiments/ascent/mf2_llm_ab.py (top-lead ruling: never edit the
original), changed for the dense NVFP4 tuning switch
(tensorrt_llm._torch.custom_ops.torch_custom_ops.NVFP4_DENSE_TUNING).

One process (TLLM_WORKER_USE_SINGLE_PROCESS=1), one LLM, batch_wait_timeout_ms
1500, TRTLLM_FI_HOST_PREFILL_PLAN=1 for every pass. The engine starts with the
switch ON, so the autotuner warmup tunes main's entries AND the dense entries;
toggling the module flag then routes every call to main's pick (OFF) or the
dense pick (ON).

Phases:
 C  (switch OFF) capture pass: own isl300/1k/2k/8k singles, one 4x8k group,
    one 16x300 group. gate_up/down act_fp4/act_sf/alpha (+ main's output at
    M<4096) at layers 0/30/59 for every step, and those layers' weights, go to
    <shared>/<commit>.tmp-<tag>/ with manifest.json written last, then renamed.
 A / X / A'  (OFF / ON / OFF) over 100 own prompts: c=1 isl300 x20, isl1k x20,
    isl2k x10, isl8k x10, 5 groups of 4x1k, 5 groups of 4x8k (groups sent
    together). 1 greedy token, top-20 logprobs. A/A' must be 100/100 bitwise.
 R  re-send, no reset: switch ON, re-send 2 prompts per class in order
    8k, 1k, 8k (c1 and groups) and compare with X's first send; then switch
    OFF and the same against A.
Counts dense/main dispatches of trtllm::nvfp4_gemm per pass.

usage: llm_ab.py <checkpoint> [capture_dir_root|-] [commit_tag]
"""
import hashlib
import json
import os
import random
import shutil
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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


CAPTURE_LAYERS = (0, 30, 59)


def main():
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ["TRTLLM_FI_HOST_PREFILL_PLAN"] = "1"
    os.environ["TRTLLM_NVFP4_DENSE_TUNING"] = "1"
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
    threading.Thread(target=_heartbeat, daemon=True).start()
    ck = sys.argv[1]
    cap_root = sys.argv[2] if len(sys.argv) > 2 else "-"
    commit = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    if cap_root != "-":
        try:
            os.makedirs(cap_root, exist_ok=True)
            probe = os.path.join(cap_root, f".write_test_{os.getpid()}")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            print(f"[capture] write test OK: {cap_root}", flush=True)
        except OSError as e:
            cap_root = os.path.abspath(os.path.join(art, "captures"))
            os.makedirs(cap_root, exist_ok=True)
            print(f"[capture] shared dir not writable ({e}); using {cap_root}", flush=True)

    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm._torch.modules import linear as lin
    from tensorrt_llm.llmapi import KvCacheConfig

    with open(os.path.join(ck, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    hidden, inter, nl = tc["hidden_size"], tc["intermediate_size"], tc["num_hidden_layers"]
    print(f"[dims] hidden={hidden} intermediate={inter} layers={nl}", flush=True)
    GU = (2 * inter, hidden)
    DN = (hidden, inter)

    tok = AutoTokenizer.from_pretrained(ck)
    rng = random.Random(20261115)

    def mk(isl):
        ids = tok(make_text(rng, isl * 2), add_special_tokens=False)["input_ids"][: isl - 1]
        ids = [tok.bos_token_id or 2] + ids
        assert len(ids) == isl, (len(ids), isl)
        return ids

    # groups: (label, [ids...]) sent together in one generate()
    groups = []
    for label, isl, conc, ngroups in (("isl300@1", 300, 1, 20), ("isl1k@1", 1000, 1, 20),
                                      ("isl2k@1", 2000, 1, 10), ("isl8k@1", 8000, 1, 10),
                                      ("isl1k@4", 1000, 4, 5), ("isl8k@4", 8000, 4, 5)):
        for _ in range(ngroups):
            groups.append((label, [mk(isl) for _ in range(conc)]))
    n_prompts = sum(len(g[1]) for g in groups)
    cap_groups = [("cap_isl300@1", [mk(300)]), ("cap_isl1k@1", [mk(1000)]), ("cap_isl2k@1", [mk(2000)]),
                  ("cap_isl8k@1", [mk(8000)]), ("cap_isl8k@4", [mk(8000) for _ in range(4)]),
                  ("cap_isl300@16", [mk(300) for _ in range(16)])]
    print(f"built {n_prompts} A/B prompts in {len(groups)} groups; capture groups {len(cap_groups)}", flush=True)

    # ---- dispatch counting
    counts = {}
    _fwd = tco.NVFP4GemmUnifiedRunner.forward

    def fwd(self, inputs, tactic=-1, bias=None, **kw):
        key = ("dense" if self.dense_min_m is not None else "main") + f"_N{inputs[1].shape[0]}"
        counts[key] = counts.get(key, 0) + 1
        return _fwd(self, inputs, tactic=tactic, bias=bias, **kw)

    tco.NVFP4GemmUnifiedRunner.forward = fwd

    # ---- capture hooks (active only in the capture pass)
    cap = {"on": False, "label": None, "step": {}, "entries": [], "weights": {}, "dir": None,
           "gu_calls": 0, "dn_calls": 0, "skipped_capturing": 0, "bytes": 0}
    _prep = lin.NVFP4LinearMethod._input_prepare
    _apply = lin.NVFP4LinearMethod.apply
    stash = threading.local()

    def prep(self, module, input):
        r = _prep(self, module, input)
        stash.last = r
        return r

    def save(obj, name):
        path = os.path.join(cap["dir"], name)
        torch.save(obj, path)
        cap["bytes"] += os.path.getsize(path)
        return name

    def apply(self, module, input, bias):
        out = _apply(self, module, input, bias)
        if cap["on"]:
            act_fp4, act_sf, alpha = stash.last
            record(module.weight, module.weight_scale, act_fp4, act_sf, alpha, out, "apply")
        return out

    _chunked = mg.gate_up_gelu_fp4_chunked

    def chunked(act_fp4, act_sf, weight, weight_scale, alpha, *a, **k):
        res = _chunked(act_fp4, act_sf, weight, weight_scale, alpha, *a, **k)
        cap["chunked_calls"] += res is not None
        if cap["on"] and res is not None:
            record(weight, weight_scale, act_fp4, act_sf, alpha, None, "chunked")
        return res

    mg.gate_up_gelu_fp4_chunked = chunked
    cap["chunked_calls"] = 0

    def record(weight, weight_scale, act_fp4, act_sf, alpha, out, path):
        n, kp = weight.shape[0], weight.shape[1] * 2
        if (n, kp) == GU:
            shape, idx = "gate_up", cap["gu_calls"]
            cap["gu_calls"] += 1
        elif (n, kp) == DN:
            shape, idx = "down", cap["dn_calls"]
            cap["dn_calls"] += 1
        else:
            return
        layer = idx % nl
        step = idx // nl
        if layer not in CAPTURE_LAYERS:
            return
        if torch.cuda.is_current_stream_capturing():
            cap["skipped_capturing"] += 1
            return
        m = int(act_fp4.shape[0])
        base = f"{cap['label']}_s{step}_L{layer}_{shape}_M{m}"
        ent = {"prompt": cap["label"], "step": step, "layer": layer, "shape": shape, "M": m,
               "N": n, "K": kp, "path": path,
               "act_shape": list(act_fp4.shape), "act_dtype": str(act_fp4.dtype),
               "sf_shape": list(act_sf.shape), "sf_dtype": str(act_sf.dtype),
               "alpha": float(alpha.float().reshape(-1)[0].item()) if torch.is_tensor(alpha) else float(alpha)}
        tensors = {"act_fp4": act_fp4.detach().cpu().clone(), "act_sf": act_sf.detach().cpu().clone(),
                   "alpha": alpha.detach().cpu().clone() if torch.is_tensor(alpha) else torch.tensor(alpha)}
        if m < 4096 and out is not None:
            o = out.reshape(-1, out.shape[-1])
            tensors["out"] = o.detach().cpu().clone()
            ent["out_shape"] = list(o.shape)
        ent["file"] = save(tensors, base + ".pt")
        cap["entries"].append(ent)
        wkey = f"L{layer}_{shape}"
        if wkey not in cap["weights"]:
            wt = {"weight": weight.detach().cpu().clone(),
                  "weight_scale": weight_scale.detach().cpu().clone()}
            cap["weights"][wkey] = {"file": save(wt, f"weight_{wkey}.pt"), "layer": layer, "shape": shape,
                                    "w_shape": list(weight.shape), "w_dtype": str(weight.dtype),
                                    "ws_shape": list(weight_scale.shape),
                                    "ws_dtype": str(weight_scale.dtype)}
        print(f"[capture] {base} path={path} act={tuple(act_fp4.shape)} sf={tuple(act_sf.shape)}", flush=True)

    lin.NVFP4LinearMethod._input_prepare = prep
    lin.NVFP4LinearMethod.apply = apply

    t_load = time.time()
    llm = LLM(
        model=ck,
        max_num_tokens=8192,
        max_seq_len=8300,
        max_batch_size=64,
        batch_wait_timeout_ms=1500,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.85, enable_block_reuse=False),
    )
    print(f"loaded in {time.time() - t_load:.1f}s", flush=True)
    from tensorrt_llm._torch.autotuner import AutoTuner
    tuner_cache = AutoTuner.get().profiling_cache.cache
    picks = {}
    for key, val in tuner_cache.items():
        if key[0] in ("trtllm::nvfp4_gemm::gemm", "trtllm::nvfp4_gemm::gemm_dense"):
            shapes = key[3]
            picks[f"{key[0].split('::')[-1]}|N{shapes[1][0]}|M{shapes[0][0]}"] = str(val[1])
    print(f"[picks] {len(picks)} nvfp4 entries", flush=True)
    for k in sorted(picks, key=lambda s: (s.split('|')[0], s.split('|')[1], int(s.split('|')[2][1:]))):
        print(f"[pick] {k} -> {picks[k]}", flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def send(gs, tag):
        res = []
        t0 = time.time()
        for label, ids_list in gs:
            outs = llm.generate([{"prompt_token_ids": ids} for ids in ids_list], sp, use_tqdm=False)
            for o in outs:
                c = o.outputs[0]
                lp = c.logprobs[0] if c.logprobs else {}
                top = {int(t): float(v.logprob) for t, v in lp.items()}
                res.append({"label": label, "tok": int(c.token_ids[0]), "top": top})
        print(f"[{tag}] {len(res)} prompts in {time.time() - t0:.1f}s counts={counts}", flush=True)
        return res

    def compare(a, b):
        eq = flips = 0
        maxd = 0.0
        per = {}
        first_diff = []
        for i, (x, y) in enumerate(zip(a, b)):
            same = x["tok"] == y["tok"] and x["top"] == y["top"]
            eq += same
            flips += x["tok"] != y["tok"]
            common = set(x["top"]) & set(y["top"])
            d = max((abs(x["top"][t] - y["top"][t]) for t in common), default=float("inf"))
            maxd = max(maxd, d)
            e = per.setdefault(x["label"], [0, 0, 0])
            e[0] += 1
            e[1] += same
            e[2] += x["tok"] != y["tok"]
            if not same and len(first_diff) < 5:
                first_diff.append({"idx": i, "label": x["label"], "tok_a": x["tok"], "tok_b": y["tok"], "dmax": d})
        return {"bitwise_equal": eq, "total": len(a), "token_flips": flips, "max_abs_dlogprob": maxd,
                "per_case_equal_total_flips": per, "first_diffs": first_diff}

    # warmup on both paths
    for flag in (False, True):
        tco.NVFP4_DENSE_TUNING = flag
        send([groups[0], groups[-1]], f"warmup_{flag}")

    out = {"settings": {"TRTLLM_FI_HOST_PREFILL_PLAN": "1", "batch_wait_timeout_ms": 1500,
                        "max_num_tokens": 8192, "single_process": True}, "picks": picks}

    # ---- capture pass (switch OFF = main)
    if cap_root != "-":
        final = os.path.join(cap_root, commit)
        tmp = f"{final}.tmp-{os.getpid()}-{int(time.time())}"
        os.makedirs(tmp, exist_ok=True)
        cap["dir"] = tmp
        tco.NVFP4_DENSE_TUNING = False
        c0 = dict(counts)
        for label, ids_list in cap_groups:
            cap["label"] = label
            cap["gu_calls"] = cap["dn_calls"] = 0
            cap["on"] = True
            llm.generate([{"prompt_token_ids": ids} for ids in ids_list], sp, use_tqdm=False)
            cap["on"] = False
            torch.cuda.synchronize()
            print(f"[capture] {label}: gu_calls={cap['gu_calls']} dn_calls={cap['dn_calls']} "
                  f"bytes={cap['bytes'] / 1e9:.2f}GB", flush=True)
        files = {}
        for fn in sorted(os.listdir(tmp)):
            files[fn] = sha256(os.path.join(tmp, fn))
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        producers = {}
        for rel in ("tensorrt_llm/_torch/models/modeling_gemma4.py",
                    "tensorrt_llm/_torch/kernels/fused_ops/gelu_tanh_mul_fp4_quant.py",
                    "tensorrt_llm/_torch/modules/gated_mlp.py",
                    "tensorrt_llm/_torch/kernels/fused_ops/gate_up_gelu_fp4_chunked.py",
                    "tensorrt_llm/_torch/modules/linear.py"):
            p = os.path.join(repo, rel)
            if os.path.exists(p):
                producers[rel] = sha256(p)
        man = {"commit": commit, "created": time.time(), "capture_switch": "TRTLLM_NVFP4_DENSE_TUNING OFF (main path)",
               "layers": list(CAPTURE_LAYERS), "entries": cap["entries"], "weights": cap["weights"],
               "files_sha256": files, "producer_sources_sha256": producers,
               "skipped_in_graph_capture": cap["skipped_capturing"], "total_bytes": cap["bytes"],
               "prompt_seed": 20261115, "dispatch_counts": {k: counts[k] - c0.get(k, 0) for k in counts}}
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
        if os.path.exists(final):
            print(f"[capture] {final} already exists; leaving new set at {tmp}", flush=True)
            final = tmp
        else:
            os.rename(tmp, final)
        shutil.copy(os.path.join(final, "manifest.json"), os.path.join(art, "capture_manifest.json"))
        out["capture_dir"] = final
        out["capture_entries"] = [(e["prompt"], e["step"], e["layer"], e["shape"], e["M"]) for e in cap["entries"]]
        print(f"[capture] wrote {len(cap['entries'])} entries, {cap['bytes'] / 1e9:.2f} GB to {final}", flush=True)

    # ---- A / X / A'
    passes, calls = {}, {}
    for tag, flag in (("A", False), ("X", True), ("A2", False)):
        tco.NVFP4_DENSE_TUNING = flag
        c0 = dict(counts)
        passes[tag] = send(groups, tag)
        calls[tag] = {k: counts[k] - c0.get(k, 0) for k in counts}
    aa = compare(passes["A"], passes["A2"])
    ax = compare(passes["A"], passes["X"])
    print("A vs A2:", json.dumps(aa), flush=True)
    print("A vs X :", json.dumps(ax), flush=True)

    # ---- re-send, no reset: classes 8k, 1k, 8k
    def pick_resend(labels):
        sel = []
        for lab in labels:
            idx = [i for i, g in enumerate(groups) if g[0] == lab]
            sel.extend(idx[:2])
        return sel

    order = pick_resend(["isl8k@1", "isl8k@4"]) + pick_resend(["isl1k@1", "isl1k@4"]) + \
        pick_resend(["isl8k@1", "isl8k@4"])[::-1]
    offsets = []
    o = 0
    for g in groups:
        offsets.append(o)
        o += len(g[1])
    resend = {}
    for tag, flag, ref in (("R_on", True, "X"), ("R_off", False, "A")):
        tco.NVFP4_DENSE_TUNING = flag
        c0 = dict(counts)
        got = send([groups[i] for i in order], tag)
        refs = []
        for i in order:
            refs.extend(passes[ref][offsets[i]: offsets[i] + len(groups[i][1])])
        resend[tag] = compare(refs, got)
        resend[tag]["calls"] = {k: counts[k] - c0.get(k, 0) for k in counts}
        print(f"{tag} vs {ref}:", json.dumps(resend[tag]), flush=True)
    llm.shutdown()

    ok = (aa["bitwise_equal"] == aa["total"] and ax["bitwise_equal"] == ax["total"]
          and all(r["bitwise_equal"] == r["total"] for r in resend.values())
          and sum(v for k, v in calls["X"].items() if k.startswith("dense")) > 0
          and sum(v for k, v in calls["A"].items() if k.startswith("dense")) == 0)
    out.update({"passed": bool(ok), "compared": "in-process LLM, dense NVFP4 tuning OFF/ON/OFF (A/X/A'), "
                "first-token id + top-20 logprobs bitwise; re-send 8k,1k,8k no reset vs first send, ON and OFF",
                "n_prompts": n_prompts, "A_vs_A2": aa, "A_vs_X": ax, "resend": resend, "dispatch_calls": calls,
                "chunked_gate_up_calls_total": cap["chunked_calls"]})
    with open(os.path.join(art, "llm_ab.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("passed", "compared", "n_prompts", "A_vs_A2", "A_vs_X", "resend",
                                          "dispatch_calls", "chunked_gate_up_calls_total")}
                     | {"capture_dir": out.get("capture_dir")}))


if __name__ == "__main__":
    main()
