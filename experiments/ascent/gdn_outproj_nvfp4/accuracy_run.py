# Accuracy proxy arms for the GDN out_proj decode NVFP4 copy, reusing the items of
# experiments/ascent/gdn_nvfp4/accuracy.py (2 x 32 needle retrieval on a 33.1K
# haystack with a 32768 snapshot, 16 short items, 4 x 256-token generations).
#   A: TRTLLM_GDN_OUTPROJ_NVFP4_DECODE=0 (= main: in_proj_qkvz copy on, out_proj FP8)
#   B: out_proj copy on (default)
# Decode width matches serving: max_batch_size=4, CudaGraphConfig(max_batch_size=4).
# Retrieval max_tokens 160. Counts per arm: _postprocess_gdn_output path decisions
# (python-level: eager / CUDA-graph capture) and CUDA-graph replays by token count,
# so out_proj Marlin hits = sum(replays at <= 16 tokens) x 48 when the captured
# graphs contain the Marlin path.
# Usage: accuracy_run.py run A|B out.json ; compare via ../gdn_nvfp4/accuracy.py compare
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gdn_nvfp4"))
import accuracy as acc  # noqa: E402


def run(arm, out_path):
    os.environ["TRTLLM_GDN_INPROJ_NVFP4_DECODE"] = "1"
    os.environ["TRTLLM_GDN_OUTPROJ_NVFP4_DECODE"] = "0" if arm == "A" else "1"
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"  # in-process worker -> counters visible
    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.modules.mamba import gdn_mixer as gm
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    counts = collections.Counter()
    replays = collections.Counter()
    with_copy, without_copy = set(), set()
    orig_post = gm.Qwen3NextGatedDeltaNet._postprocess_gdn_output

    def post(self, attn_out, z, all_reduce_params=None):
        rows = z.shape[0]
        has_copy = getattr(self.out_proj, "_nvfp4_decode_weight", None) is not None
        (with_copy if has_copy else without_copy).add(id(self))
        marlin = rows <= 16 and has_copy
        where = "capture" if torch.cuda.is_current_stream_capturing() else "eager"
        counts[("marlin" if marlin else "fp8") + "/" + where + "/" + ("le16" if rows <= 16 else "gt16")] += 1
        return orig_post(self, attn_out, z, all_reduce_params)

    gm.Qwen3NextGatedDeltaNet._postprocess_gdn_output = post
    orig_replay = cgr.CUDAGraphRunner.replay

    def replay(self, key, current_inputs, **kw):
        replays[str(current_inputs["input_ids"].shape[0])] += 1
        return orig_replay(self, key, current_inputs, **kw)

    cgr.CUDAGraphRunner.replay = replay

    tok = AutoTokenizer.from_pretrained(acc.CKPT)
    t0 = time.time()
    llm = LLM(model=acc.CKPT, max_seq_len=36864, max_batch_size=4, max_num_tokens=8192,
              enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=4),
              kv_cache_config=KvCacheConfig(enable_block_reuse=True, mamba_state_config={
                  "additional_snapshot_offsets_from_start": [32768]}))
    n_copy = 0
    try:
        model = llm._executor.engine.model_engine.model
        for mod in model.modules():
            if isinstance(mod, gm.Qwen3NextGatedDeltaNet):
                n_copy += getattr(mod.out_proj, "_nvfp4_decode_weight", None) is not None
    except Exception as e:  # noqa: BLE001
        acc.log(f"[{arm}] could not walk model for copy count: {e!r}")
        n_copy = -1
    acc.log(f"[{arm}] LLM ready in {time.time()-t0:.1f}s; gdn out_proj copies {n_copy}; "
            f"mem alloc {torch.cuda.memory_allocated()/1e9:.2f} GB; free/total "
            f"{[x/1e9 for x in torch.cuda.mem_get_info()]}")
    acc.log(f"[{arm}] counts after warmup/capture: {dict(counts)} replays {dict(replays)}")
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
        acc.log(f"[{arm}] retrieval seed {seed}: {time.time()-t1:.1f}s")
        for k, e, o in zip(keys, expects, outs):
            c = o.outputs[0]
            results["items"][k] = {"text": c.text, "ids": list(c.token_ids), "expect": e,
                                   "ok": all(x in c.text for x in e), "lp": lp(c)}
    sp = SamplingParams(max_tokens=48, temperature=0.0, logprobs=1)
    outs = llm.generate([p for p, _ in acc.SHORT], sp)
    for i, ((p, e), o) in enumerate(zip(acc.SHORT, outs)):
        c = o.outputs[0]
        results["items"][f"short{i}"] = {"text": c.text, "ids": list(c.token_ids), "expect": e,
                                         "ok": all(x in c.text for x in e), "lp": lp(c)}
    sp = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True, logprobs=1)
    outs = llm.generate(acc.LONG_GEN, sp)
    for i, o in enumerate(outs):
        c = o.outputs[0]
        results["items"][f"gen{i}"] = {"text": c.text, "ids": list(c.token_ids), "expect": [],
                                       "ok": True, "lp": lp(c)}
    n_ok = sum(v["ok"] for k, v in results["items"].items() if not k.startswith("gen"))
    n = sum(1 for k in results["items"] if not k.startswith("gen"))
    small = sum(v for k, v in replays.items() if int(k) <= 16)
    in_graph = counts.get("marlin/capture/le16", 0) > 0
    results["counts"] = {"postprocess_python": dict(counts), "graph_replays_by_tokens": dict(replays),
                         "gdn_out_proj_copies_walk": n_copy,
                         "gdn_layers_with_copy": len(with_copy),
                         "gdn_layers_without_copy": len(without_copy),
                         "graph_decode_replays_le16": small,
                         "out_proj_marlin_replay_hits": small * len(with_copy) if in_graph else 0,
                         "out_proj_marlin_eager_hits": counts.get("marlin/eager/le16", 0),
                         "out_proj_fp8_replay_hits": small * len(without_copy)}
    acc.log(f"[{arm}] correct {n_ok}/{n}; counts {json.dumps(results['counts'])}")
    for k, v in results["items"].items():
        if not v["ok"]:
            acc.log(f"[{arm}] WRONG {k}: expect {v['expect']} got {v['text'][-200:]!r}")
    with open(out_path, "w") as f:
        json.dump(results, f)
    llm.shutdown()


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
