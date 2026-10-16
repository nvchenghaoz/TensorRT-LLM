# STEP 2 end-to-end exactness for the Marlin persistent workspace + PDL change.
# One process per arm (LLM API, serving engine settings incl. FP8 KV cache, CUDA-graph decode
# for batch 1-8): arm A = TRTLLM_MARLIN_PERSISTENT_WORKSPACE=0 TRTLLM_MARLIN_PDL=0 (= main's
# Marlin path), arm B = both on (default). Prompts: experiments/ascent/gdn_nvfp4/accuracy.py
# SHORT (16 x 48 tokens) and LONG_GEN (4 x 256, ignore_eos), greedy, logprobs=1.
# Token ids and logprobs must be identical. Heartbeat every 30 s.
# Usage: e2e.py <checkpoint> <workdir>
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


def heartbeat(tag):
    t0 = time.time()

    def loop():
        while True:
            time.sleep(30)
            log(f"[heartbeat {tag}] {time.time() - t0:.0f}s")

    threading.Thread(target=loop, daemon=True).start()


def run_arm(arm, ckpt, out_path):
    heartbeat(arm)
    if arm == "A":
        os.environ["TRTLLM_MARLIN_PERSISTENT_WORKSPACE"] = "0"
        os.environ["TRTLLM_MARLIN_PDL"] = "0"
    else:
        os.environ.pop("TRTLLM_MARLIN_PERSISTENT_WORKSPACE", None)
        os.environ.pop("TRTLLM_MARLIN_PDL", None)
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ["CKPT"] = ckpt
    sys.path.insert(0, os.path.join(HERE, "..", "gdn_nvfp4"))
    import accuracy as acc  # noqa: E402
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    captured = set()
    runner_cls = getattr(cgr, "CUDAGraphRunner", None)
    if runner_cls is not None and hasattr(runner_cls, "capture"):
        orig_capture = runner_cls.capture

        def capture(self, key, *a, **kw):
            captured.add(str(key))
            return orig_capture(self, key, *a, **kw)

        runner_cls.capture = capture

    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=8, max_num_tokens=8192,
              enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=8),
              kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=True, mamba_state_config={
                  "additional_snapshot_offsets_from_start": [32768]}))
    log(f"[{arm}] LLM ready in {time.time() - t0:.1f}s; graphs captured {sorted(captured)}")

    def lp(c):
        return [list(d.values())[0].logprob if isinstance(d, dict) else d for d in (c.logprobs or [])]

    items = {}
    sp = SamplingParams(max_tokens=48, temperature=0.0, logprobs=1)
    outs = llm.generate([p for p, _ in acc.SHORT], sp)
    for i, o in enumerate(outs):
        c = o.outputs[0]
        items[f"short{i}"] = {"ids": list(c.token_ids), "lp": lp(c), "text": c.text}
    sp = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True, logprobs=1)
    outs = llm.generate(acc.LONG_GEN, sp)
    for i, o in enumerate(outs):
        c = o.outputs[0]
        items[f"gen{i}"] = {"ids": list(c.token_ids), "lp": lp(c), "text": c.text}
    torch.cuda.synchronize()
    snap = None
    if hasattr(torch.ops.trtllm, "marlin_nvfp4_persistent_workspace_snapshot"):
        s = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device="cuda"))
        snap = {"ints": int(s.numel()), "all_zero": bool((s == 0).all().item()) if s.numel() else None}
    res = {"arm": arm, "items": items, "graphs_captured": sorted(captured), "workspace": snap}
    log(f"[{arm}] done; workspace {snap}")
    with open(out_path, "w") as f:
        json.dump(res, f)
    llm.shutdown()


def main(ckpt, workdir):
    os.makedirs(workdir, exist_ok=True)
    paths = {}
    for arm in ("A", "B"):
        p = os.path.join(workdir, f"e2e_{arm}.json")
        paths[arm] = p
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "arm", arm, ckpt, p])
        log(f"arm {arm} exit {r.returncode} in {time.time() - t0:.0f}s")
        if r.returncode != 0:
            print(json.dumps({"passed": False, "error": f"arm {arm} failed rc={r.returncode}"}))
            return
    a = json.load(open(paths["A"]))
    b = json.load(open(paths["B"]))
    same_ids, same_lp, diffs = 0, 0, []
    for k in a["items"]:
        ia, ib = a["items"][k], b["items"][k]
        si = ia["ids"] == ib["ids"]
        sl = ia["lp"] == ib["lp"]
        same_ids += si
        same_lp += sl
        if not (si and sl):
            d = next((i for i, (x, y) in enumerate(zip(ia["ids"], ib["ids"])) if x != y), None)
            diffs.append({"item": k, "ids_equal": si, "lp_equal": sl, "first_token_div": d})
    n = len(a["items"])
    res = {"passed": same_ids == n and same_lp == n, "n_sequences": n,
           "identical_token_ids": same_ids, "identical_logprobs": same_lp, "diffs": diffs,
           "graphs_captured_A": a["graphs_captured"], "graphs_captured_B": b["graphs_captured"],
           "workspace_A": a["workspace"], "workspace_B": b["workspace"],
           "compared": "greedy token ids + logprobs, switches off (A) vs on (B), 16 SHORT x 48 + 4 LONG_GEN x 256"}
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "arm":
        run_arm(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2])
