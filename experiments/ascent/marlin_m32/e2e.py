"""Idea 55 (branch 36) end-to-end: real model, CUDA-graph decode at 17-32 rows.

Arms, one process each (LLM API, serving engine settings: FP8 KV cache, block reuse,
chunked prefill, max_batch_size 32, CUDA graphs up to 32), one shared AutoTuner cache:
  main  TRTLLM_FP8_NVFP4_DECODE_WIDE=0  (every cap 16: W4A4 at 17-32)
  cand  default (cap 32 for qkv, (5120,6144) and qkvz; kM32Table launches)
  err   default + TRTLLM_GDN_GATED_NORM_NVFP4=0 + TRTLLM_ATTN_GATE_NVFP4=0 so every FP8-origin
        linear sees BF16 prefill input; the first 32 rows of up to 6 eager prefill inputs per
        (N,K) (real activations, real weights) are run through Marlin kM32Table, Marlin generic
        (TRTLLM_MARLIN_M32_TABLE=0, branch 18) and W4A4 at M=17/24/32 and compared with the
        FP32 reference x @ (W_fp8 * s)^T.
Each of main/cand generates 32 prompts concurrently (decode batch 32), then 20 (graph batch
24 padded), 64 greedy tokens, ignore_eos, logprobs=1.
Instrumentation (wrapping FP8QDQLinearMethod.apply / CUDAGraphRunner.replay):
  - route of every FP8-origin linear call during graph capture, by (N,K) and rows: 'w4a4'
    (prefill copy called, BF16 or Fp4QuantizedTensor input) or 'decode' (Marlin / GEMV), and
    the input kind ('bf16' / 'fp4');
  - eager (non-capturing) calls with 17..32 rows after engine init: must be 0;
  - graph replays per batch size.
Pass: cand captures route all three shapes to 'decode' on BF16 input at rows <= 32 and to
'w4a4' above (with Fp4QuantizedTensor input for (5120,6144)); main routes everything above 16
to 'w4a4'; no eager 17..32-row calls; replays at batch 24/32 > 0; Marlin lock workspace all
zero; err: table rel-RMS <= 1.02 x generic's (+1e-6) for every sample, and qkvz table <= W4A4.
Usage: e2e.py <checkpoint> <workdir>
"""
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = [
    "List the first ten prime numbers: 2, 3, 5,",
    "Count from 20 to 30: 20, 21,",
    "The days of the week in order are Monday, Tuesday,",
    "The months of the year in order are January, February, March,",
    "The planets of the solar system, in order from the Sun, are Mercury,",
    "Q: What is 12 * 12? Show the result only.\nA: 12 * 12 =",
    "Q: What is 365 - 120?\nA: 365 - 120 =",
    "Q: What is 1000 / 8?\nA: 1000 / 8 =",
    "Q: What are the capital cities of France, Japan and Italy?\nA: The capitals are",
    "Q: Name the author of 'Pride and Prejudice' and the author of 'Hamlet'.\nA:",
    "The powers of two are 1, 2, 4, 8,",
    "The square numbers are 1, 4, 9,",
    "Write a detailed explanation of how a refrigerator works.\n\n",
    "Explain the causes and consequences of the French Revolution.\n\n",
    "Describe step by step how to implement binary search in Python.\n\n",
    "Write a short story about a lighthouse keeper.\n\n",
]
PROMPTS = BASE + [f"Answer briefly. {p}" for p in BASE]
LONG = ("Summarize the following text in one sentence.\n\n" +
        " ".join(f"Paragraph {i}: The quick brown fox jumps over the lazy dog while the "
                 f"committee reviews budget item number {i * 7 % 13}." for i in range(40)))
WIDE = ("14336,5120", "5120,6144", "16384,5120")


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
    if arm == "main":
        os.environ["TRTLLM_FP8_NVFP4_DECODE_WIDE"] = "0"
    else:
        os.environ.pop("TRTLLM_FP8_NVFP4_DECODE_WIDE", None)
    if arm == "err":
        os.environ["TRTLLM_GDN_GATED_NORM_NVFP4"] = "0"
        os.environ["TRTLLM_ATTN_GATE_NVFP4"] = "0"
    os.environ.pop("TRTLLM_MARLIN_M32_TABLE", None)
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn
    from tensorrt_llm._torch.modules import linear as L
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    state = {"init": True, "busy": False}
    capture_routes = {}  # "N,K" -> {rows: route/kind}
    eager_mid = {}
    replays = {}
    errs = {}  # "N,K" -> list of {M: {...}}
    orig_apply = L.FP8QDQLinearMethod.apply

    def marlin(module, x):
        return torch.ops.trtllm.marlin_nvfp4_gemm(
            x, module._nvfp4_decode_weight, scale_a=None, scale_b=module._nvfp4_decode_scale,
            alpha=None, weight_global_scale=module._nvfp4_decode_global_scale, bias=None,
            out_dtype=torch.bfloat16, size_n=module.out_features, size_k=module.in_features,
            workspace_mode=2)

    def measure(module, key, x2):
        xs = x2[:32].contiguous()
        Wd = module.weight.float() * module.weight_scale.float()
        rec = {}
        for M in (17, 24, 32):
            x = xs[:M]
            ref = x.float() @ Wd.t()
            outs = {}
            outs["marlin_table"] = marlin(module, x)
            os.environ["TRTLLM_MARLIN_M32_TABLE"] = "0"
            outs["marlin_generic_b18"] = marlin(module, x)
            os.environ.pop("TRTLLM_MARLIN_M32_TABLE")
            copy = fpn.get_prefill_nvfp4_copy(module)
            if copy is not None:
                outs["w4a4"] = fpn.prefill_nvfp4_apply(copy, module, x, x.shape, None)
            e = {}
            for nm, y in outs.items():
                d = y.float() - ref
                e[nm] = {"rel_rms": float(d.norm() / ref.norm()), "max_abs": float(d.abs().max())}
            rec[str(M)] = e
        del Wd
        errs.setdefault(key, []).append(rec)

    def apply(self, module, input, bias):
        if isinstance(input, torch.Tensor):
            rows = input.reshape(-1, input.shape[-1]).shape[0]
            kind = "bf16" if input.dtype == torch.bfloat16 else str(input.dtype)
        else:  # Fp4QuantizedTensor from an NVFP4-emitting producer
            rows = int(input.shape[0])
            kind = "fp4"
        c0 = fpn._stats["calls"]
        out = orig_apply(self, module, input, bias)
        if getattr(module, "_nvfp4_decode_weight", None) is None:
            return out
        key = f"{module.out_features},{module.in_features}"
        route = "w4a4" if fpn._stats["calls"] > c0 else "decode"
        if torch.cuda.is_current_stream_capturing():
            capture_routes.setdefault(key, {})[rows] = f"{route}/{kind}"
        elif not state["init"]:
            if 17 <= rows <= 32:
                k2 = f"{key},{rows},{route}"
                eager_mid[k2] = eager_mid.get(k2, 0) + 1
            if (arm == "err" and rows >= 32 and kind == "bf16" and not state["busy"]
                    and len(errs.get(key, [])) < 6):
                state["busy"] = True
                try:
                    measure(module, key, input.reshape(-1, input.shape[-1]))
                finally:
                    state["busy"] = False
        return out

    L.FP8QDQLinearMethod.apply = apply
    runner_cls = cgr.CUDAGraphRunner
    orig_replay = runner_cls.replay

    def replay(self, key, *a, **kw):
        bs = str(getattr(key, "batch_size", key))
        replays[bs] = replays.get(bs, 0) + 1
        return orig_replay(self, key, *a, **kw)

    runner_cls.replay = replay

    t0 = time.time()
    llm = LLM(model=ckpt, max_seq_len=36864, max_batch_size=32,
              max_num_tokens=8192, enable_chunked_prefill=True,
              cuda_graph_config=CudaGraphConfig(max_batch_size=32),
              kv_cache_config=KvCacheConfig(
                  dtype="fp8", enable_block_reuse=True,
                  mamba_state_config={"additional_snapshot_offsets_from_start": [32768]}))
    state["init"] = False
    log(f"[{arm}] LLM ready in {time.time() - t0:.1f}s; capture routes "
        f"{json.dumps(capture_routes, sort_keys=True)}")

    def lp(c):
        return [list(d.values())[0].logprob if isinstance(d, dict) else d
                for d in (c.logprobs or [])]

    items = {}
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True, logprobs=1)
    sets = (("long", [LONG, LONG + " Be concise."]),) if arm == "err" else (
        ("b32", PROMPTS), ("b20", PROMPTS[:20]))
    for tag, prompts in sets:
        outs = llm.generate(prompts, sp)
        for i, o in enumerate(outs):
            c = o.outputs[0]
            items[f"{tag}_{i}"] = {"ids": list(c.token_ids), "lp": lp(c), "text": c.text}
        log(f"[{arm}] {tag} done; replays {replays}; eager 17-32 {eager_mid}")
    torch.cuda.synchronize()
    snap = None
    if hasattr(torch.ops.trtllm, "marlin_nvfp4_persistent_workspace_snapshot"):
        s = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device="cuda"))
        snap = {"ints": int(s.numel()),
                "all_zero": bool((s == 0).all().item()) if s.numel() else None}
    res = {"arm": arm, "items": items, "capture_routes": capture_routes,
           "eager_mid_rows": eager_mid, "replays": replays, "workspace": snap, "errs": errs}
    with open(out_path, "w") as f:
        json.dump(res, f)
    log(f"[{arm}] done")
    llm.shutdown()


def compare(a, b):
    n_same, max_dlp, n_tok, n_match, firsts = 0, 0.0, 0, 0, []
    for k in a["items"]:
        ia, ib = a["items"][k], b["items"][k]
        n_same += ia["ids"] == ib["ids"]
        d = next((i for i, (x, y) in enumerate(zip(ia["ids"], ib["ids"])) if x != y), None)
        upto = len(ia["ids"]) if d is None else d + 1
        for x, y in zip(ia["lp"][:upto], ib["lp"][:upto]):
            if x is not None and y is not None:
                max_dlp = max(max_dlp, abs(x - y))
        n_tok += len(ia["ids"])
        n_match += upto if d is None else d
        if d is not None:
            firsts.append((k, d))
    return {"n_sequences": len(a["items"]), "identical_sequences": n_same,
            "tokens_before_first_divergence_frac": n_match / max(n_tok, 1),
            "max_abs_dlogprob_up_to_first_divergence": max_dlp, "first_divergence": firsts}


def check_routes(r, wide):
    bad = []
    for key, by_rows in r["capture_routes"].items():
        for rows, rk in by_rows.items():
            rows = int(rows)
            route, kind = rk.split("/")
            if rows <= 16:
                want = ("decode", "bf16")
            elif rows <= 32 and wide and key in WIDE:
                want = ("decode", "bf16")
            else:
                want = ("w4a4", "fp4" if key == "5120,6144" else "bf16")
            if (route, kind) != want:
                bad.append((key, rows, rk, "/".join(want)))
    return bad


def check_errs(e):
    bad = []
    for key, recs in e["errs"].items():
        for i, rec in enumerate(recs):
            for M, r in rec.items():
                t, g = r["marlin_table"]["rel_rms"], r["marlin_generic_b18"]["rel_rms"]
                if t > 1.02 * g + 1e-6:
                    bad.append((key, i, M, "table>generic", t, g))
                if key == "16384,5120" and "w4a4" in r and t > r["w4a4"]["rel_rms"]:
                    bad.append((key, i, M, "table>w4a4", t, r["w4a4"]["rel_rms"]))
    return bad


def main(ckpt, workdir):
    os.makedirs(workdir, exist_ok=True)
    cache = os.path.join(workdir, "autotuner_cache.json")
    if os.path.exists(cache):
        os.remove(cache)
    os.environ["TLLM_AUTOTUNER_CACHE_PATH"] = cache
    paths = {}
    for arm in ("main", "cand", "err"):
        p = os.path.join(workdir, f"e2e_{arm}.json")
        paths[arm] = p
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "arm", arm, ckpt, p])
        log(f"arm {arm} exit {r.returncode} in {time.time() - t0:.0f}s")
        if r.returncode != 0:
            print(json.dumps({"passed": False, "error": f"arm {arm} rc={r.returncode}"}))
            return
    a = json.load(open(paths["main"]))
    b = json.load(open(paths["cand"]))
    e = json.load(open(paths["err"]))
    bad_main = check_routes(a, False)
    bad_cand = check_routes(b, True)
    bad_err = check_errs(e)
    captured_mid = sorted({int(r) for v in b["capture_routes"].values() for r in v
                           if 17 <= int(r) <= 32})
    err_keys = sorted(e["errs"])
    ok = (not bad_main and not bad_cand and not a["eager_mid_rows"] and not b["eager_mid_rows"]
          and b["replays"].get("32", 0) > 0 and b["replays"].get("20", 0) + b["replays"].get("24", 0) > 0
          and captured_mid == list(range(17, 33))
          and (b["workspace"] is None or b["workspace"]["all_zero"] is not False)
          and not bad_err and err_keys == sorted(WIDE))
    res = {"passed": ok,
           "compared": "real model, CUDA-graph decode at batch 32 and 20 (padded): FP8-origin "
                       "linear routes + input kind per captured graph, eager 17-32-row calls, "
                       "graph replays, lock workspace; cand vs main (WIDE=0) greedy tokens/"
                       "logprobs; real-activation error vs FP32 of the FP8 op (err arm)",
           "route_violations_main": bad_main, "route_violations_cand": bad_cand,
           "err_violations": bad_err, "err_keys": err_keys,
           "captured_rows_17_32": captured_mid,
           "capture_routes_cand": b["capture_routes"], "capture_routes_main": a["capture_routes"],
           "eager_17_32_main": a["eager_mid_rows"], "eager_17_32_cand": b["eager_mid_rows"],
           "replays_main": a["replays"], "replays_cand": b["replays"],
           "workspace_cand": b["workspace"], "cand_vs_main": compare(a, b),
           "real_activation_errs": e["errs"],
           "sample_text": {k: [a["items"][k]["text"][:120], b["items"][k]["text"][:120]]
                           for k in ("b32_5", "b32_8", "b20_2")}}
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "arm":
        run_arm(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2])
