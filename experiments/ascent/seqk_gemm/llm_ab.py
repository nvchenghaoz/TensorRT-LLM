"""Idea 73 in-process end-to-end check (derived from experiments/ascent/mf2_llm_ab.py and
experiments/ascent/bcg_prefill/bcg_ab.py mode_inproc). ONE process (TLLM_WORKER_USE_SINGLE_PROCESS=1),
ONE LLM with this tree's engine overlay (BCG breakable prefill graphs), batch_wait_timeout_ms=1500.

Passes over the same 100 own prompts (make_text; isl300@1 x40, isl300@4 x5, isl1k@1 x10, 4x1k x3,
4x8k x2, c=1 odd lengths 289/301/310/317/320 x2), 1 greedy token, top-20 logprobs:
  A  : TRTLLM_GEMMA4_SEQK_GEMM off (module switch), graph replay off (eager main path)
  X  : switch on, replay off (eager, new kernel at routed M)
  A2 : switch off, replay off
  R  : switch on, replay on (BCG graphs captured at startup with the switch on)
Then, switch on, no reset: re-send 10 isl300 prompts after an 8k, 1k, 8k sequence (eager and replay).
Bitwise: token ids + top-20 logprobs. Also gate (c): qkv_proj inputs captured from the model (layer 0
sliding qkv and the first global [q|k]) for an isl300 and an isl8k prompt, int16-compared kernel vs
main's dispatch at the captured M and on row slices 288..320 (every M) and 336..1280 (16-row steps).
Kernel-call census: routed / fallback counts per (N, M), split eager vs inside CUDA-graph capture.

usage: llm_ab.py <checkpoint>"""
import collections
import json
import os
import random
import sys
import threading
import time

os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_JSON = os.path.join(HERE, "..", "..", "ascent-config", "engine.json")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _heartbeat():
    t0 = time.time()
    while True:
        time.sleep(60)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


def main():
    threading.Thread(target=_heartbeat, daemon=True).start()
    ck = sys.argv[1]
    from mf2_llm_ab import make_text

    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm.llmapi import KvCacheConfig

    ov = json.load(open(ENGINE_JSON)) if os.path.exists(ENGINE_JSON) else {}
    for k, v in (ov.pop("env", None) or {}).items():
        os.environ[k] = str(v)
    log(f"engine overlay {ov}; routes {mg._SM120_SEQK_GEMM_ROUTES}; switch {mg._SEQK_GEMM}")

    # census + real-activation capture, wrapped around the routed apply_linear
    census = {"eager": collections.Counter(), "capture": collections.Counter()}
    captures = {}
    want_capture = [False]
    orig_apply = mg._Sm120SeqKGemmMixin.apply_linear

    def counted(self, input, bias, lora_params=None, layer_idx=None):
        M = input.shape[0] if input.dim() == 2 else -1
        routed = mg._SEQK_GEMM and input.dim() == 2 and M in self.seqk_gemm_tokens
        cap = torch.cuda.is_current_stream_capturing()
        census["capture" if cap else "eager"][f"{self.out_features}@{M}:{'seqk' if routed else 'main'}"] += 1
        if want_capture[0] and not cap and M in (300, 8000) and (self.out_features, M) not in captures:
            captures[(self.out_features, M)] = (input.detach().clone(), self.weight)
        return orig_apply(self, input, bias, lora_params, layer_idx)

    mg._Sm120SeqKGemmMixin.apply_linear = counted

    tok = AutoTokenizer.from_pretrained(ck)
    rng = random.Random(20261115)

    def prompt(isl):
        ids = tok(make_text(rng, isl * 2), add_special_tokens=False)["input_ids"][: isl - 1]
        ids = [tok.bos_token_id or 2] + ids
        assert len(ids) == isl
        return {"prompt_token_ids": ids}

    groups = []
    groups += [("isl300@1", [prompt(300)]) for _ in range(40)]
    groups += [("isl300@4", [prompt(300) for _ in range(4)]) for _ in range(5)]
    groups += [("isl1k@1", [prompt(1000)]) for _ in range(10)]
    groups += [("4x1k", [prompt(1000) for _ in range(4)]) for _ in range(3)]
    groups += [("4x8k", [prompt(8000) for _ in range(4)]) for _ in range(2)]
    for L in (289, 301, 310, 317, 320):
        groups += [(f"odd{L}", [prompt(L)]) for _ in range(2)]
    n_prompts = sum(len(g) for _, g in groups)
    log(f"built {len(groups)} groups, {n_prompts} prompts")
    cap300, cap8k = prompt(300), prompt(8000)
    resend = [prompt(300) for _ in range(10)]
    seq_8k_1k_8k = [prompt(8000), prompt(1000), prompt(8000)]

    t0 = time.time()
    llm = LLM(model=ck, kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False,
                                                      free_gpu_memory_fraction=0.8),
              max_num_tokens=8192, max_batch_size=64, max_seq_len=9216, batch_wait_timeout_ms=1500.0, **ov)
    log(f"startup {time.time() - t0:.1f}s; census after startup {dict(census['capture'])}")
    startup_capture = dict(census["capture"])

    replay_on = [True]
    has_bcg = False
    try:
        from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
        orig_has_graph = BreakableCUDAGraphRunner.has_graph
        BreakableCUDAGraphRunner.has_graph = lambda self, n: replay_on[0] and orig_has_graph(self, n)
        has_bcg = True
    except Exception as ex:  # noqa: BLE001
        log(f"no BreakableCUDAGraphRunner: {ex!r}")
    try:
        from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as _g
        steps = getattr(_g, "_BCG_STEP_COUNTS", None)
    except Exception:  # noqa: BLE001
        steps = None

    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def summarize(o):
        c = o.outputs[0]
        lp = c.logprobs[0] if c.logprobs else {}
        return {"tok": int(c.token_ids[0]), "top": sorted((int(t), float(v.logprob)) for t, v in lp.items())}

    def run_groups(gs):
        res = []
        for _, g in gs:
            outs = llm.generate(g, sp, use_tqdm=False)
            res.append([summarize(o) for o in outs])
        return res

    # warmup both switch states, eager and replay
    for sw in (False, True):
        mg._SEQK_GEMM = sw
        for rp in (False, True):
            replay_on[0] = rp
            for isl in (300, 1000, 8000, 317):
                llm.generate([prompt(isl)], sp, use_tqdm=False)

    # real-activation captures (eager, switch on)
    mg._SEQK_GEMM, replay_on[0] = True, False
    want_capture[0] = True
    llm.generate([cap300], sp, use_tqdm=False)
    llm.generate([cap8k], sp, use_tqdm=False)
    want_capture[0] = False
    log(f"captured {[(k, tuple(v[0].shape)) for k, v in captures.items()]}")

    passes, per_census, per_steps = {}, {}, {}
    for tag, sw, rp in (("A", False, False), ("X", True, False), ("A2", False, False), ("R", True, True)):
        mg._SEQK_GEMM, replay_on[0] = sw, rp
        c0 = {k: collections.Counter(v) for k, v in census.items()}
        s0 = dict(steps) if steps is not None else None
        t1 = time.time()
        passes[tag] = run_groups(groups)
        per_census[tag] = {k: dict(census[k] - c0[k]) for k in census}
        if steps is not None:
            per_steps[tag] = {k: steps[k] - s0[k] for k in steps}
        log(f"pass {tag} switch={sw} replay={rp} {time.time() - t1:.1f}s steps={per_steps.get(tag)} "
            f"census={per_census[tag]}")

    # re-send after 8k, 1k, 8k, no reset (eager then replay), switch on
    resend_res = {}
    mg._SEQK_GEMM = True
    for rp in (False, True):
        replay_on[0] = rp
        first = [summarize(o) for o in llm.generate(resend, sp, use_tqdm=False)]
        for p in seq_8k_1k_8k:
            llm.generate([p], sp, use_tqdm=False)
        again = [summarize(o) for o in llm.generate(resend, sp, use_tqdm=False)]
        resend_res["replay" if rp else "eager"] = (first, again)
    replay_on[0] = True

    def cmp(a, b):
        per = {}
        eq = tot = flips = 0
        for (label, _), x, y in zip(groups, a, b):
            for u, v in zip(x, y):
                tot += 1
                same = u == v
                eq += same
                flips += u["tok"] != v["tok"]
                d = per.setdefault(label, [0, 0])
                d[0] += same
                d[1] += 1
        return {"bitwise_equal": eq, "total": tot, "token_flips": flips,
                "per_case": {k: f"{v[0]}/{v[1]}" for k, v in per.items()}}

    comps = {"A_vs_A2": cmp(passes["A"], passes["A2"]), "X_vs_A": cmp(passes["X"], passes["A"]),
             "R_vs_X": cmp(passes["R"], passes["X"]), "R_vs_A": cmp(passes["R"], passes["A"])}
    for k, v in comps.items():
        log(f"{k}: {v['bitwise_equal']}/{v['total']} flips {v['token_flips']} {v['per_case']}")
    rs = {}
    for k, (first, again) in resend_res.items():
        rs[k] = {"resend_equal": sum(int(u == v) for u, v in zip(first, again)), "of": len(first)}
    rs["eager_vs_replay"] = sum(int(u == v) for u, v in zip(resend_res["eager"][0], resend_res["replay"][0]))

    # gate (c): real activations
    from common import load_main
    main_op, _ = load_main()
    gate_c = {}
    gate_ok = True
    for (N, M), (x, w) in sorted(captures.items()):
        K = x.shape[1]
        route = mg._SM120_SEQK_GEMM_ROUTES.get((K, N))
        if route is None:
            continue
        cfg = route[1]
        ms = sorted({M} | {m for m in range(288, 321) if m <= M} | {m for m in range(336, 1281, 16) if m <= M})
        fails = []
        for m in ms:
            xs = x[:m]
            y = torch.ops.trtllm.sm120_bf16_seqk_gemm(xs, w, cfg)
            ref = main_op(xs, w)
            if not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                d = (y.view(torch.int16) != ref.view(torch.int16)).nonzero()
                fails.append({"m": m, "count": int(d.shape[0]), "first": d[0].tolist()})
        gate_c[f"N{N}_capturedM{M}"] = {"slices": len(ms), "fails": fails[:5], "n_fail": len(fails),
                                        "x_absmax": float(x.abs().max())}
        gate_ok &= not fails
    log(f"gate c: {gate_c}")
    llm.shutdown()

    ok = (comps["A_vs_A2"]["bitwise_equal"] == n_prompts and comps["X_vs_A"]["bitwise_equal"] == n_prompts
          and comps["R_vs_X"]["bitwise_equal"] == n_prompts
          and all(v["resend_equal"] == v["of"] for k, v in rs.items() if isinstance(v, dict))
          and rs["eager_vs_replay"] == len(resend) and gate_ok and len(gate_c) >= 4
          and sum(v for k, v in per_census["X"]["eager"].items() if k.endswith(":seqk")) > 0
          and sum(v for k, v in per_census["A"]["eager"].items() if k.endswith(":seqk")) == 0)
    print(json.dumps({"passed": bool(ok),
                      "compared": "in-process LLM (BCG overlay), token id + top-20 logprobs bitwise: "
                                  "A/A2 = TRTLLM_GEMMA4_SEQK_GEMM off eager, X = on eager, R = on BCG replay; "
                                  "re-send after 8k,1k,8k; gate c real qkv_proj activations int16 vs main",
                      "n_prompts": n_prompts, "comparisons": comps, "resend": rs, "gate_c": gate_c,
                      "census_startup_capture": startup_capture, "census_per_pass": per_census,
                      "bcg_steps_per_pass": per_steps, "has_bcg": has_bcg, "overlay": ov}))


if __name__ == "__main__":
    main()
