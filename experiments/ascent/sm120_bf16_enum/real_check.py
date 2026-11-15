#!/usr/bin/env python3
"""Model-level gate for the branch-43 LUT pins: pitfall 629 (c) real activations, and BCG capture/replay bitwise.

Two sequential processes on one GPU, both with the tree's engine overlay (experiments/ascent-config/engine.json:
breakable prefill CUDA graphs, buckets 300..2000), in-process LLM (bcg_prefill/bcg_ab.py settings), sharing ONE
autotuner cache: arm OFF fills a fresh TLLM_AUTOTUNER_CACHE_PATH file, arm ON loads a copy of it (sha256 logged).
  OFF: every Gemma4 projection forced to F.linear (linear._custom_cublas_mm_covers -> False before the engine is
       built, so the BCG graphs hold main's bits), then
       (1) the bucket items x 3 sends (replayed graphs), (2) with graph replay switched off (eager), one isl300 and
       one isl8k prompt while capturing the real qkv_proj / o_proj inputs at layers {0, first global, 30, last};
       on those activations the candidate dispatch (trtllm::cublas_mm with this tree's LUT where its routes cover
       M, else F.linear) must equal F.linear and main's dispatch (check_enum.MAIN_LUT) bit for bit at every BCG
       bucket M <= rows, and at the full row count.
  ON : this tree's routes (the pins), the bucket items x 3 sends (replayed graphs).
Bucket items (packed, one context step each): 1x300, 2x300, 3x300, 1x1000, 4x300, 5x300, 6x300, 1x2000.
Passed: ON == OFF bitwise (token ids + top-20 logprobs) on every item and send; every repeat == first send; all
items ran as replayed BCG steps; real-activation compares all bitwise.

usage: python real_check.py <checkpoint>
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKET_ITEMS = [(300, 1), (300, 2), (300, 3), (1000, 1), (300, 4), (300, 5), (300, 6), (2000, 1)]
SENDS = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p):
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def arm(name, ckpt, cache, out):
    os.environ["TLLM_AUTOTUNER_CACHE_PATH"] = cache
    os.environ["CKPT"] = ckpt
    os.environ["BCG_PACK"] = "1"
    sys.path.insert(0, os.path.join(HERE, "..", "bcg_prefill"))
    sys.path.insert(0, HERE)
    import bcg_ab  # noqa: E402  (TLLM_WORKER_USE_SINGLE_PROCESS=1)
    log(f"arm {name}: cache {cache} sha256 at start {sha(cache)}")
    import torch
    import torch.nn.functional as F

    import tensorrt_llm._torch.modules.linear as lin
    orig_covers = lin._custom_cublas_mm_covers
    if name == "off":
        lin._custom_cublas_mm_covers = lambda module, input: False
    llm = bcg_ab.build_llm(use_overlay=True, batch_wait_timeout_ms=1500.0)
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
    counts = getattr(_g, "_BCG_STEP_COUNTS", {})
    params = bcg_ab.sp(logprobs=20)
    warm = bcg_ab.Prompts(5)
    for isl in (300, 1000, 8000):
        bcg_ab.run_batch(llm, [warm.make(isl) for _ in range(2)], params)
    pf = bcg_ab.Prompts(4242)
    items = [[pf.make(isl) for _ in range(n)] for isl, n in BUCKET_ITEMS]
    res = {"sends": [], "replay_counts": []}
    for s in range(SENDS):
        before = dict(counts)
        dig = []
        for prompts in items:
            _, outs = bcg_ab.run_batch(llm, prompts, params)
            dig.append([bcg_ab.summarize(o) for o in outs])
        after = dict(counts)
        res["sends"].append(dig)
        res["replay_counts"].append({k: after.get(k, 0) - before.get(k, 0) for k in ("replayed", "eager")})
        log(f"arm {name} send {s}: BCG steps {res['replay_counts'][-1]}")
    if name == "off":
        # real activations, eager (graph replay switched off), main's path
        from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
        orig_has = BreakableCUDAGraphRunner.has_graph
        BreakableCUDAGraphRunner.has_graph = lambda self, n: False
        cfg = json.load(open(os.path.join(ckpt, "config.json")))
        types = cfg.get("text_config", cfg)["layer_types"]
        first_gl = types.index("full_attention") if "full_attention" in types else next(
            i for i, t in enumerate(types) if t != "sliding_attention")
        want = sorted({0, first_gl, 30, len(types) - 1})
        cls = lin.UnquantizedLinearMethod
        orig_apply = cls.apply
        st = {"on": False, "calls": 0, "caps": []}

        def apply(self, module, input, bias, weight=None):
            if st["on"] and getattr(module, "custom_cublas_mm_tokens", None) is not None and input.dim() == 2:
                k = st["calls"]
                st["calls"] += 1
                if k // 2 in want:
                    st["caps"].append((st["isl"], k // 2, "qkv" if k % 2 == 0 else "o_proj",
                                       input.detach().clone(), module.weight))
            return orig_apply(self, module, input, bias, weight)

        cls.apply = apply
        try:
            for isl in (300, 8000):
                st.update(on=True, calls=0, isl=isl)
                bcg_ab.run_batch(llm, [pf.make(isl)], params)
                st["on"] = False
                log(f"real activations isl{isl}: {st['calls']} routed-projection calls")
        finally:
            cls.apply = orig_apply
            BreakableCUDAGraphRunner.has_graph = orig_has
        import check_enum
        from ltx import LtX, main_dispatch
        op, routed, mg = main_dispatch()
        ltx = LtX()

        def main_op(x, W):
            N, K = W.shape
            M = x.shape[0]
            if routed(N, K, M) and M not in check_enum.ADDED_ROUTES.get((K, N), set()):
                c = check_enum.lut_lookup(check_enum.MAIN_LUT[(K, N)], M) if M <= 2099 else None
                if c is not None:
                    o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                    ltx.matmul(x, W, o, ltx.make_algo(c))
                    return o
                return torch.ops.trtllm.cublas_mm(x, W.t(), None, None)  # unchanged region (m > 2099)
            return F.linear(x, W)

        buckets = sorted({isl * n for isl, n in BUCKET_ITEMS})
        real = {"captures": len(st["caps"]), "cases": 0, "bitwise": 0, "failures": [], "rows": []}
        for isl, layer, which, x, W in st["caps"]:
            ms = sorted({m for m in buckets if m <= x.shape[0]} | {x.shape[0]})
            n_ok = 0
            for m in ms:
                a = F.linear(x[:m], W)
                b = op(x[:m], W)
                c = main_op(x[:m], W)
                good = torch.equal(a.view(torch.int16), b.view(torch.int16)) and torch.equal(
                    c.view(torch.int16), b.view(torch.int16))
                real["cases"] += 1
                n_ok += int(good)
                if not good:
                    real["failures"].append([isl, layer, which, m, int((a.view(torch.int16) != b.view(
                        torch.int16)).sum())])
            real["bitwise"] += n_ok
            real["rows"].append([isl, layer, which, list(W.shape), int(x.shape[0]), n_ok, len(ms),
                                 bool(routed(W.shape[0], W.shape[1], 300))])
        real["layers"] = want
        log(f"real-activation check: {json.dumps(real)}")
        res["real"] = real
    lin._custom_cublas_mm_covers = orig_covers
    with open(out, "w") as f:
        json.dump(res, f)
    llm.shutdown()
    log(f"arm {name}: cache sha256 at end {sha(cache)}")


def main():
    ckpt = sys.argv[1]
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    fa = os.path.join(art, "autotuner_FA.json")
    if os.path.exists(fa):
        os.remove(fa)
    res, shas = {}, {}
    for name in ("off", "on"):
        cache = fa if name == "off" else os.path.join(art, "autotuner_FA_copy_on.json")
        if name == "on":
            shutil.copyfile(fa, cache)
        shas[name] = {"start": sha(cache)}
        out = os.path.join(art, f"real_check_{name}.json")
        log(f"starting arm {name}")
        p = subprocess.run([sys.executable, __file__, "arm", name, ckpt, cache, out])
        shas[name]["end"] = sha(cache)
        if p.returncode != 0 or not os.path.exists(out):
            print(json.dumps({"passed": False, "error": f"arm {name} rc={p.returncode}"}))
            return
        res[name] = json.load(open(out))
    on, off = res["on"], res["off"]
    item_names = [f"{n}x{isl}" for isl, n in BUCKET_ITEMS]
    on_vs_off = {nm: all(on["sends"][s][i] == off["sends"][s][i] for s in range(SENDS))
                 for i, nm in enumerate(item_names)}
    repeats = {a: all(r["sends"][s] == r["sends"][0] for s in range(SENDS)) for a, r in res.items()}
    replayed = {a: r["replay_counts"] for a, r in res.items()}
    all_replayed = all(c["replayed"] >= len(BUCKET_ITEMS) for r in res.values() for c in r["replay_counts"])
    real = off["real"]
    real_ok = real["captures"] == 16 and real["cases"] > 0 and not real["failures"]
    has_lp = bool(off["sends"][0][0][0]["lp"])
    passed = all(on_vs_off.values()) and all(repeats.values()) and real_ok and has_lp and all_replayed
    print(json.dumps({"passed": bool(passed),
                      "compared": "BCG engine, routes ON (branch pins) vs OFF (all F.linear = main bits), shared "
                                  "autotuner cache; token ids + top-20 logprobs bitwise per bucket item and send; "
                                  "real isl300/isl8k qkv/o_proj inputs: cand dispatch vs F.linear and main dispatch",
                      "on_vs_off_bitwise": on_vs_off, "repeats_bitwise": repeats, "bcg_step_counts": replayed,
                      "all_items_replayed": all_replayed, "real_activations": real, "autotuner_cache_sha256": shas,
                      "logprobs_present": has_lp}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "arm":
        arm(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
