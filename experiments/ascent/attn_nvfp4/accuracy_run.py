# Accuracy proxy for the gated-attention decode NVFP4 W4A16 copies (qkv_proj, o_proj),
# reusing the items of experiments/ascent/gdn_nvfp4/accuracy.py (2 x 32 needle retrieval
# on a 33.1K haystack with a 32768 snapshot, retrieval max_tokens 160; 16 short items;
# 4 x 256-token generations). Arms (GDN copies stay as on main):
#   A: TRTLLM_ATTN_QKV_NVFP4_DECODE=0 TRTLLM_ATTN_O_NVFP4_DECODE=0  (= main)
#   B: both attention copies on (branch default)
#   Q: qkv copy only
# Decode width matches serving: max_batch_size=4, CudaGraphConfig(max_batch_size=4).
# Usage:
#   accuracy_run.py all <ckpt> <outdir> A B   # arms in subprocesses + compare, heartbeat
#   accuracy_run.py run <arm> <ckpt> <out.json>
import collections
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()


def log(*a):
    print(*a, flush=True)


def heartbeat(tag, stop, period=30.0):
    while not stop.wait(period):
        log(f"[hb {tag} t={time.time() - T0:.0f}s]")


ARM_ENV = {"A": ("0", "0"), "B": ("1", "1"), "Q": ("1", "0")}


def run(arm, ckpt, out_path):
    os.environ["CKPT"] = ckpt
    qkv, o = ARM_ENV[arm]
    os.environ["TRTLLM_GDN_INPROJ_NVFP4_DECODE"] = "1"
    os.environ["TRTLLM_ATTN_QKV_NVFP4_DECODE"] = qkv
    os.environ["TRTLLM_ATTN_O_NVFP4_DECODE"] = o
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"  # in-process worker -> counters visible
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(f"arm {arm}", stop), daemon=True).start()
    sys.path.insert(0, os.path.join(HERE, "..", "gdn_nvfp4"))
    import accuracy as acc  # noqa: E402
    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.modules import linear as lin_mod
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    counts = collections.Counter()
    replays = collections.Counter()
    orig_apply = lin_mod.FP8QDQLinearMethod.apply

    def role(m):
        nk = (m.out_features, m.in_features)
        if nk == (14336, 5120):
            return "attn_qkv"
        if nk == (5120, 6144):
            return "attn_o" if getattr(m, "lora", None) is not None else "gdn_out"
        if nk == (16384, 5120):
            return "gdn_qkvz"
        return f"other{nk}"

    def apply(self, module, input, bias):
        rows = input.reshape(-1, input.shape[-1]).shape[0]
        marlin = (getattr(module, "_nvfp4_decode_weight", None) is not None
                  and input.dtype == torch.bfloat16 and rows <= 16)
        where = "capture" if torch.cuda.is_current_stream_capturing() else "eager"
        counts[f"{role(module)}/{'marlin' if marlin else 'fp8'}/{where}/"
               f"{'le16' if rows <= 16 else 'gt16'}/{str(input.dtype).split('.')[-1]}"] += 1
        return orig_apply(self, module, input, bias)

    lin_mod.FP8QDQLinearMethod.apply = apply
    orig_replay = cgr.CUDAGraphRunner.replay

    def replay(self, key, current_inputs, **kw):
        replays[str(current_inputs["input_ids"].shape[0])] += 1
        return orig_replay(self, key, current_inputs, **kw)

    cgr.CUDAGraphRunner.replay = replay

    tok = AutoTokenizer.from_pretrained(ckpt)
    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=4, max_num_tokens=8192,
              enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=4),
              kv_cache_config=KvCacheConfig(enable_block_reuse=True, mamba_state_config={
                  "additional_snapshot_offsets_from_start": [32768]}))
    copies = collections.Counter()
    try:
        model = llm._executor.engine.model_engine.model
        for mod in model.modules():
            if getattr(mod, "_nvfp4_decode_weight", None) is not None:
                copies[role(mod)] += 1
    except Exception as e:  # noqa: BLE001
        log(f"[{arm}] could not walk model for copy count: {e!r}")
    log(f"[{arm}] LLM ready in {time.time() - t0:.1f}s; decode copies {dict(copies)}; "
        f"mem alloc {torch.cuda.memory_allocated() / 1e9:.2f} GB; free/total "
        f"{[x / 1e9 for x in torch.cuda.mem_get_info()]}")
    log(f"[{arm}] counts after warmup/capture: {dict(counts)} replays {dict(replays)}")
    results = {"mode": arm, "items": {}}

    def lp(c):
        return [list(d.values())[0].logprob if isinstance(d, dict) else d for d in (c.logprobs or [])]

    for seed in (4401, 5501):
        text, names, codes = acc.build_haystack(tok, seed)
        prompts, keys, expects = [], [], []
        for nm, cd in zip(names, codes):
            prompts.append(text + f"\n\nBased on the document above, answer the question.\n"
                           f"Question: What is the access code for the {nm} vault?\n"
                           f"Answer: The access code for the {nm} vault is")
            keys.append(f"ret{seed}:{nm}")
            expects.append([cd])
        sp = SamplingParams(max_tokens=160, temperature=0.0, logprobs=1)
        t1 = time.time()
        outs = llm.generate(prompts[:1], sp)
        outs += llm.generate(prompts[1:], sp)
        log(f"[{arm}] retrieval seed {seed}: {time.time() - t1:.1f}s")
        for k, e, o_ in zip(keys, expects, outs):
            c = o_.outputs[0]
            results["items"][k] = {"text": c.text, "ids": list(c.token_ids), "expect": e,
                                   "ok": all(x in c.text for x in e), "lp": lp(c)}
    sp = SamplingParams(max_tokens=48, temperature=0.0, logprobs=1)
    outs = llm.generate([p for p, _ in acc.SHORT], sp)
    for i, ((p, e), o_) in enumerate(zip(acc.SHORT, outs)):
        c = o_.outputs[0]
        results["items"][f"short{i}"] = {"text": c.text, "ids": list(c.token_ids), "expect": e,
                                         "ok": all(x in c.text for x in e), "lp": lp(c)}
    sp = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True, logprobs=1)
    outs = llm.generate(acc.LONG_GEN, sp)
    for i, o_ in enumerate(outs):
        c = o_.outputs[0]
        results["items"][f"gen{i}"] = {"text": c.text, "ids": list(c.token_ids), "expect": [],
                                       "ok": True, "lp": lp(c)}
    n_ok = sum(v["ok"] for k, v in results["items"].items() if not k.startswith("gen"))
    n = sum(1 for k in results["items"] if not k.startswith("gen"))
    small = sum(v for k, v in replays.items() if int(k) <= 16)
    results["counts"] = {"apply_python": dict(counts), "graph_replays_by_tokens": dict(replays),
                         "decode_copies": dict(copies), "graph_decode_replays_le16": small,
                         "attn_qkv_marlin_replay_hits":
                             small * copies.get("attn_qkv", 0)
                             if any(k.startswith("attn_qkv/marlin/capture") for k in counts) else 0,
                         "attn_o_marlin_replay_hits":
                             small * copies.get("attn_o", 0)
                             if any(k.startswith("attn_o/marlin/capture") for k in counts) else 0}
    log(f"[{arm}] correct {n_ok}/{n}; counts {json.dumps(results['counts'])}")
    for k, v in results["items"].items():
        if not v["ok"]:
            log(f"[{arm}] WRONG {k}: expect {v['expect']} got {v['text'][-200:]!r}")
    with open(out_path, "w") as f:
        json.dump(results, f)
    stop.set()
    llm.shutdown()


def compare(a_path, b_path, b_arm):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    ia_all, ib_all = a["items"], b["items"]
    ret = [k for k in ia_all if k.startswith("ret")]
    short = [k for k in ia_all if k.startswith("short")]
    scored = ret + short
    lost = [k for k in scored if ia_all[k]["ok"] and not ib_all[k]["ok"]]
    gained = [k for k in scored if not ia_all[k]["ok"] and ib_all[k]["ok"]]
    firstdiv, flips, lpd = [], [], []
    for k in ia_all:
        ia, ib = ia_all[k]["ids"], ib_all[k]["ids"]
        d = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
        if d is None and len(ia) != len(ib):
            d = min(len(ia), len(ib))
        if d is not None:
            firstdiv.append([k, d])
            if k in scored and ia_all[k]["ok"] == ib_all[k]["ok"]:
                flips.append(k)
        la, lb = ia_all[k]["lp"], ib_all[k]["lp"]
        upto = d if d is not None else min(len(ia), len(ib))
        for i in range(min(upto, len(la), len(lb))):
            if isinstance(la[i], (int, float)) and isinstance(lb[i], (int, float)):
                lpd.append(abs(la[i] - lb[i]))
    ret_a = sum(ia_all[k]["ok"] for k in ret)
    ret_b = sum(ib_all[k]["ok"] for k in ret)
    short_a = sum(ia_all[k]["ok"] for k in short)
    short_b = sum(ib_all[k]["ok"] for k in short)
    a_valid = ret_a >= 62 and short_a >= 15
    lost_ret = [k for k in lost if k.startswith("ret")]
    go = a_valid and not lost_ret and len(lost) <= 1
    res = {"arm_a": "A (attention copies off = main)", "arm_b": b_arm,
           "retrieval_A": f"{ret_a}/{len(ret)}", "retrieval_B": f"{ret_b}/{len(ret)}",
           "short_A": f"{short_a}/{len(short)}", "short_B": f"{short_b}/{len(short)}",
           "A_valid": a_valid, "lost": lost, "lost_retrieval": lost_ret, "gained": gained,
           "flips_same_correctness": flips,
           "identical_token_sequences": len(ia_all) - len(firstdiv), "n_sequences": len(ia_all),
           "first_divergence": firstdiv,
           "logprob_absdiff_before_divergence": {
               "n": len(lpd), "mean": (sum(lpd) / len(lpd)) if lpd else None,
               "max": max(lpd) if lpd else None},
           "counts_A": a.get("counts"), "counts_B": b.get("counts"),
           "GO": go}
    for k, d in firstdiv:
        log(f"diverge {k} at token {d}: A {ia_all[k]['text'][:100]!r} | {b_arm} {ib_all[k]['text'][:100]!r}")
    return res


FILT = re.compile(r"^\[(A|B|Q)\]|^\[hb|haystack|Traceback|Error|error:|nvfp4_decode_copy|"
                  r"CUDA graph|cuda graph|KV cache|kv cache")


def run_all(ckpt, outdir, arms):
    os.makedirs(outdir, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=("driver", stop), daemon=True).start()
    paths = {}
    for arm in arms:
        out = os.path.join(outdir, f"{arm}.json")
        log(f"=== arm {arm} start t={time.time() - T0:.0f}s ===")
        p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "run", arm, ckpt, out],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            if FILT.search(line):
                log(line.rstrip()[:600])
        rc = p.wait()
        log(f"=== arm {arm} rc={rc} t={time.time() - T0:.0f}s ===")
        if rc != 0 or not os.path.exists(out):
            stop.set()
            print(json.dumps({"passed": False, "error": f"arm {arm} failed rc={rc}"}))
            return
        paths[arm] = out
    reports = {}
    for b_arm in arms[1:]:
        reports[b_arm] = compare(paths[arms[0]], paths[b_arm], b_arm)
    stop.set()
    go = all(r["GO"] for r in reports.values())
    print(json.dumps({"passed": go, "GO": go, "elapsed_s": round(time.time() - T0),
                      "reports": reports}))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "all":
        run_all(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        print(json.dumps(compare(sys.argv[2], sys.argv[3], "B")))
