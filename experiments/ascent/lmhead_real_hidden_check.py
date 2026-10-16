# Real-hidden-state accuracy + determinism + CUDA-graph capture check for the
# Qwen3.8-27B-NVFP4 lm_head on SM120.
# 1) Loads the model with the LLM API in-process (TLLM_WORKER_USE_SINGLE_PROCESS=1),
#    CUDA graphs for batch 1/2/4/8/16, and hooks LMHead.forward to record the real
#    final hidden states fed to lm_head (eager steps: prefill and batch-20 decode)
#    and every lm_head call made during CUDA-graph capture.
# 2) Runs decode at batch 1/2/4/8/16 (graph replay) to show the graph path works.
# 3) Builds main's lm_head (load-time BF16 dequant, cuBLAS) and this tree's
#    lm_head (Marlin W4A16 on the branch) from the checkpoint tensors and reports
#    max/mean abs logit error vs an FP64 exact-dequant reference at M=1..16 on the
#    real hidden states, plus 100 repeated eager calls and 100 graph replays
#    (bitwise vs the first call).
# usage: python lmhead_real_hidden_check.py <checkpoint>
import json
import os
import sys

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import lmhead_bench as lb  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the n-th Fibonacci number.\n\ndef fib(n):",
    "Explain in two sentences why the sky is blue.",
    "Q: What is 17 * 23?\nA:",
    "Translate to German: 'The weather is nice today, let's go for a walk.'",
    "Once upon a time, in a small village by the sea, there lived",
    "List three prime numbers greater than 100:",
    "The theory of general relativity states that",
    "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING",
    "In machine learning, overfitting happens when",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "The chemical formula for table salt is",
    "Summarize the plot of Romeo and Juliet in one sentence:",
    "To be, or not to be, that is the question:",
    "The derivative of sin(x) * x^2 with respect to x is",
    "Here is a haiku about autumn leaves:\n",
    "The three laws of thermodynamics are",
    "A good recipe for pancakes needs",
    "import numpy as np\n\n# compute the mean of each row of a matrix\n",
    "The largest planet in the solar system is",
]
GRAPH_BS = [1, 2, 4, 8, 16]  # 8/16 also cover MTP verify/draft shapes


def main():
    ckpt = sys.argv[1]
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.modules import embedding as emb
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    rec = {"rows": [], "eager_calls": [], "capture_calls": [], "method": None}
    orig = emb.LMHead.forward

    def fwd(self, input, *a, **kw):
        rec["method"] = type(self.quant_method).__name__
        if torch.cuda.is_current_stream_capturing():
            rec["capture_calls"].append(int(input.reshape(-1, input.shape[-1]).shape[0]))
        elif rec.get("on"):
            x = input.detach().reshape(-1, input.shape[-1])
            rec["eager_calls"].append(int(x.shape[0]))
            if sum(r.shape[0] for r in rec["rows"]) < 256:
                rec["rows"].append(x.to("cpu", copy=True))
        elif rec.get("on2") is not None:
            rec["on2"].append(int(input.reshape(-1, input.shape[-1]).shape[0]))
        return orig(self, input, *a, **kw)

    emb.LMHead.forward = fwd
    lb.log("loading LLM ...")
    llm = LLM(model=ckpt, max_batch_size=24, max_seq_len=4096, max_num_tokens=8192,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.25,
                                            enable_block_reuse=False),
              cuda_graph_config=CudaGraphConfig(batch_sizes=GRAPH_BS))
    lb.log("LLM ready; lm_head method=%s graph-capture lm_head calls (M)=%s" %
           (rec["method"], rec["capture_calls"]))
    sp = SamplingParams(max_tokens=24, temperature=0.0)
    # Batch of len(PROMPTS)=20 has no graph (padding off), so decode runs eager
    # and the hook sees the real final hidden states.
    rec["on"] = True
    outs_e = llm.generate(PROMPTS, sp)
    rec["on"] = False
    tok_e = [list(o.outputs[0].token_ids) for o in outs_e]
    for p, o in zip(PROMPTS[:4], outs_e[:4]):
        lb.log("  [eager b%d] %r -> %r" % (len(PROMPTS), p[:40], o.outputs[0].text[:80]))
    n_eager_before = len(rec["eager_calls"])
    # Graph path: decode at batch 1/2/4/8/16 replays the captured graphs; the hook
    # counts eager lm_head calls (expected: prefill only, no decode steps).
    graph_runs = {}
    for b in GRAPH_BS:
        rec["on2"] = []
        outs = llm.generate(PROMPTS[:b], sp)
        eager_calls = rec["on2"]
        rec["on2"] = None
        toks = [list(o.outputs[0].token_ids) for o in outs]
        graph_runs[str(b)] = {
            "n_tokens": [len(t) for t in toks],
            "eager_lm_head_calls_M": eager_calls,
            "token_match_vs_eager_batch": [
                sum(int(a == c) for a, c in zip(t, tok_e[i])) / max(1, len(t))
                for i, t in enumerate(toks)],
            "text0": outs[0].outputs[0].text[:80]}
        lb.log("  [graph b%d] %s" % (b, graph_runs[str(b)]))
    capture_calls = list(rec["capture_calls"])
    method = rec["method"]
    rows = torch.cat(rec["rows"])[::4][:64].cuda()
    lb.log("captured %d real hidden rows (eager lm_head calls before graph runs: %d, M=%s)" %
           (rows.shape[0], n_eager_before, sorted(set(rec["eager_calls"]))))
    emb.LMHead.forward = orig
    llm.shutdown()
    del llm
    torch.cuda.empty_cache()

    t = lb.load_lm_head(ckpt)
    main_head, main_arm = lb.build_head(t, force_bf16=True)
    tree_head, tree_arm = lb.build_head(t)
    ref = lb.ref_fp64(t, rows)
    lb.log("reference built: %s" % (tuple(ref.shape),))
    floor = float((ref.to(torch.bfloat16).double() - ref).abs().max())
    report = {"model_lm_head_method": method, "graph_capture_lm_head_calls_M": capture_calls,
              "graph_runs": graph_runs, "n_real_rows": int(rows.shape[0]),
              "bf16_output_rounding_floor_max_abs": floor, "per_M": {}}
    # Context and mixed context+decode steps run eager by design (requests of
    # one generate() call arrive staggered, so a batch of b has a few of them);
    # every pure-decode step replays the graph. Require the lm_head to be in the
    # captured graph for batch b and most of the 23 decode steps to bypass eager.
    for b in GRAPH_BS:
        r = graph_runs[str(b)]
        r["steps_without_eager_lm_head"] = sp.max_tokens - len(r["eager_lm_head_calls_M"])
    graph_ok = {str(b): (b in capture_calls
                         and min(graph_runs[str(b)]["n_tokens"]) == sp.max_tokens
                         and len(graph_runs[str(b)]["eager_lm_head_calls_M"])
                         <= sp.max_tokens // 4)
                for b in GRAPH_BS}
    report["graph_ok_per_batch"] = graph_ok
    ok = method == tree_arm and all(graph_ok.values())
    for m in (1, 2, 4, 8, 16):
        outs = {"main": [], "tree": []}
        with torch.inference_mode():
            for r0 in range(0, rows.shape[0], m):
                x = rows[r0:r0 + m].contiguous()
                outs["main"].append(main_head(x))
                outs["tree"].append(tree_head(x))
        o_main = torch.cat(outs["main"])
        o_tree = torch.cat(outs["tree"])
        d = {"main_cublas_bf16": lb.err_stats(o_main, ref),
             "tree_" + tree_arm: lb.err_stats(o_tree, ref)}
        d["tree_vs_main"] = {
            "max_abs": float((o_tree.double() - o_main.double()).abs().max()),
            "mean_abs": float((o_tree.double() - o_main.double()).abs().mean()),
            "argmax_match": float((o_tree.float().argmax(-1) ==
                                   o_main.float().argmax(-1)).float().mean())}
        # 100 repeated eager calls + 100 graph replays on a real batch, bitwise.
        x = rows[:m].contiguous()
        with torch.inference_mode():
            first = tree_head(x).clone()
            mis_eager = sum(int(not torch.equal(tree_head(x), first)) for _ in range(100))
        g, gout = lb.capture(tree_head, x)
        mis_graph = 0
        for _ in range(100):
            g.replay()
            torch.cuda.synchronize()
            mis_graph += int(not torch.equal(gout, first))
        d["repeat"] = {"eager_calls": 100, "eager_mismatch_vs_first": mis_eager,
                       "graph_replays": 100, "graph_mismatch_vs_eager": mis_graph,
                       "finite": bool(torch.isfinite(o_tree).all())}
        ok = ok and mis_eager == 0 and mis_graph == 0 and d["repeat"]["finite"]
        report["per_M"][str(m)] = d
        lb.log("  M=%d %s" % (m, json.dumps(d)))
        del g, gout
    print(json.dumps(dict(report, passed=bool(ok), tree_arm=tree_arm,
                          compared="logits on real final hidden states vs FP64 exact-dequant "
                                   "reference (main cuBLAS BF16 and this tree's lm_head); "
                                   "100 eager calls and 100 graph replays bitwise")))


if __name__ == "__main__":
    main()
