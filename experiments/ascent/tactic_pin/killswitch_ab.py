"""Idea 116 (branch 50): kill switches vs main across trees, one shared autotuner cache.

usage (run_gpu_pair, same command in both arms): killswitch_ab.py <checkpoint> <shared_dir>

Ruling (cross-area A/B): separate-process arms must run identical tuner picks, so both arms use
ONE TLLM_AUTOTUNER_CACHE_PATH under <shared_dir>. The baseline arm (main's tree, no pins) clears
<shared_dir>, tunes in its engine warmup and saves the cache; the candidate arm (this branch)
loads that file, so the warmup does not re-profile. sha256 of the cache is logged before/after.

Each arm: one in-process eager LLM (TLLM_WORKER_USE_SINGLE_PROCESS=1; no BCG overlay, so the pin
module flags reach every NVFP4 GEMM call), FP8 KV, block reuse off, max_num_tokens 8192,
batch_wait_timeout_ms 300 (each group lands in one step). Own text prompts (seed 116): c=1 isl
300 x20, 1000 x20, 2000 x20, 4000 x10; 4x300 x5 (M=1200), 5x300 x2 (M=1500), 6x300 x2 (M=1800).
1 greedy token, top-20 logprobs. Plus the per-M kernel log: trtllm::nvfp4_gemm on synthetic
operands at every served M (same tuner cache), torch.profiler kernel name + grid.

Candidate passes (module flags): K0 both pins off, G0 gate_up pin off (down on), D0 down pin off
(gate_up on), ON both on, K0b both off again. Checked against main's pass:
  K0 and K0b bitwise (tokens + top-20 logprobs) and the K0 kernel log == main's at every M;
  D0 bitwise (the gate_up pins are exact);
  G0 / ON: token agreement and max |dlogprob| reported (the down 1664-2047 pin is reordered).
"""
import glob
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time

os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
CKPT, SHARED = sys.argv[1], sys.argv[2]
CACHE = os.path.join(SHARED, "autotuner_cache.json")
MAIN_OUT = os.path.join(SHARED, "main_outputs.json")
SERVED_M = [300, 1000, 1200, 1500, 1800, 2000, 2300, 4000, 4800]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _hb():
    t0 = time.time()
    while True:
        time.sleep(60)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


def sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def main():
    threading.Thread(target=_hb, daemon=True).start()
    # Which tree: main has no pins in torch_custom_ops (checked by source text, before import).
    import importlib.util
    spec = importlib.util.find_spec("tensorrt_llm")
    src = os.path.join(os.path.dirname(spec.origin), "_torch", "custom_ops", "torch_custom_ops.py")
    arm = "candidate" if "def nvfp4_pinned_tactic" in open(src).read() else "main"
    log(f"[arm] {arm} (torch_custom_ops at {src})")
    os.makedirs(SHARED, exist_ok=True)
    if arm == "main":
        for p in glob.glob(CACHE + "*") + [MAIN_OUT]:
            if os.path.exists(p):
                os.remove(p)
    else:
        files = glob.glob(CACHE + "*")
        if not files or not os.path.exists(MAIN_OUT):
            print(json.dumps({"passed": False, "error": f"main arm cache/outputs missing in {SHARED}"}), flush=True)
            os._exit(0)
        log(f"[cache] before: {files} sha256={sha(files)}")
    os.environ["TLLM_AUTOTUNER_CACHE_PATH"] = CACHE
    os.environ.pop("TRTLLM_CACHE_DIR", None)

    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    h, inter = tc["hidden_size"], tc["intermediate_size"]
    D = {"gate_up": (2 * inter, h), "down": (h, inter)}
    log(f"[dims] hidden={h} intermediate={inter} layers={tc['num_hidden_layers']}")
    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
    from tensorrt_llm.llmapi import KvCacheConfig

    llm = LLM(model=CKPT, kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False,
                                                        free_gpu_memory_fraction=0.8),
              max_num_tokens=8192, max_batch_size=64, max_seq_len=9216, batch_wait_timeout_ms=300.0)
    files = glob.glob(CACHE + "*")
    log(f"[cache] after startup: {files} sha256={sha(files) if files else None}")

    tok = AutoTokenizer.from_pretrained(CKPT)
    rng = random.Random(116)
    words = ("the of and to in is was for on that with as by at from his her it an were are which this be "
             "or has had not but have one their they its new first after who been also two time more year "
             "city river music school system state world history data model energy market water light "
             "village company science paper season team game church station family garden island bridge "
             "language theory number field road power north south east west small large early late").split()

    def mk(isl):
        out = []
        while len(out) < isl * 2:
            s = [rng.choice(words) for _ in range(rng.randint(6, 18))]
            s[0] = s[0].capitalize()
            out.extend(s)
            out[-1] += "."
        ids = tok(" ".join(out), add_special_tokens=False)["input_ids"][: isl - 1]
        return [tok.bos_token_id or 2] + ids

    groups = []
    for isl, n in ((300, 20), (1000, 20), (2000, 20), (4000, 10)):
        groups += [(f"isl{isl}@1", [mk(isl)]) for _ in range(n)]
    for c, n in ((4, 5), (5, 2), (6, 2)):
        groups += [(f"{c}x300", [mk(300) for _ in range(c)]) for _ in range(n)]
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, logprobs=20)

    def send(tag):
        res = []
        for label, ps in groups:
            for o in llm.generate(ps, sampling_params=sp, use_tqdm=False):
                c = o.outputs[0]
                lp = c.logprobs[0] if c.logprobs else {}
                res.append({"label": label, "tok": int(c.token_ids[0]),
                            "top": {str(int(t)): float(getattr(v, "logprob", v)) for t, v in lp.items()}})
        log(f"[{tag}] {len(res)} outputs")
        return res

    # per-M kernel log on synthetic operands (same tuner cache as the engine)
    def quantize(x):
        gs = (448.0 * 6.0 / x.abs().amax().float()).reshape(1).to(torch.float32)
        q, sf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)
        return q, sf, gs

    def kernel_log():
        from torch.profiler import ProfilerActivity, profile
        out = {}
        for name, (n, k) in D.items():
            g = torch.Generator(device="cuda").manual_seed(5 + n)
            wq, wsf, wgs = quantize((torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16))
            for m in SERVED_M:
                g = torch.Generator(device="cuda").manual_seed(m)
                xq, xsf, xgs = quantize(torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16))
                alpha = (1.0 / (xgs * wgs)).float()
                f = lambda: torch.ops.trtllm.nvfp4_gemm(xq, wq, xsf, wsf, alpha, torch.bfloat16,  # noqa: E731
                                                         allowed_backends="cutlass,cublaslt,cutedsl,cuda_core")
                f()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    f()
                    torch.cuda.synchronize()
                p = tempfile.mktemp(suffix=".json")
                prof.export_chrome_trace(p)
                tr = json.load(open(p))
                os.remove(p)
                out[f"{name}@{m}"] = [(e["name"][:120], e.get("args", {}).get("grid"))
                                      for e in tr["traceEvents"] if e.get("cat") == "kernel"]
            del wq, wsf
        return out

    def flags(gu, dn):
        if arm == "candidate":
            tco.NVFP4_GATEUP_PIN = gu
            tco.NVFP4_DOWN_PIN = dn

    # warmup (untimed, outside the compared passes)
    for label, ps in groups[::10]:
        llm.generate(ps, sampling_params=sp, use_tqdm=False)

    if arm == "main":
        res = {"outputs": send("MAIN"), "kernels": kernel_log()}
        with open(MAIN_OUT + ".tmp", "w") as f:
            json.dump(res, f)
        os.replace(MAIN_OUT + ".tmp", MAIN_OUT)
        files = glob.glob(CACHE + "*")
        log(f"[cache] end: {files} sha256={sha(files) if files else None}")
        for k, v in res["kernels"].items():
            log(f"[kernels] main {k}: {v}")
        print(json.dumps({"passed": True, "arm": "main", "n_outputs": len(res["outputs"]),
                          "cache_files": files, "cache_sha256": sha(files) if files else None}), flush=True)
        llm.shutdown()
        os._exit(0)

    main_res = json.load(open(MAIN_OUT))
    passes = {}
    kern = {}
    for tag, gu, dn in (("K0", False, False), ("G0", False, True), ("D0", True, False), ("ON", True, True),
                        ("K0b", False, False), ("ON2", True, True)):
        flags(gu, dn)
        passes[tag] = send(tag)
        if tag in ("K0", "ON"):
            kern[tag] = kernel_log()
    flags(True, True)

    def compare(a, b):
        eq = flips = 0
        maxd = 0.0
        per = {}
        first = []
        for i, (x, y) in enumerate(zip(a, b)):
            same = x["tok"] == y["tok"] and x["top"] == y["top"]
            eq += same
            flips += x["tok"] != y["tok"]
            common = set(x["top"]) & set(y["top"])
            d = max((abs(x["top"][t] - y["top"][t]) for t in common), default=0.0)
            maxd = max(maxd, d)
            e = per.setdefault(x["label"], [0, 0, 0, 0.0])
            e[0] += 1
            e[1] += same
            e[2] += x["tok"] != y["tok"]
            e[3] = max(e[3], d)
            if not same and len(first) < 5:
                first.append({"idx": i, "label": x["label"], "tok": [x["tok"], y["tok"]], "dmax": d})
        return {"bitwise_equal": eq, "total": len(a), "token_agreement": len(a) - flips, "max_abs_dlogprob": maxd,
                "per_case_n_equal_flips_maxd": {k: [v[0], v[1], v[2], round(v[3], 6)] for k, v in per.items()},
                "first_diffs": first}

    # main's kernel log came through JSON (lists); normalize ours the same way before comparing
    kern = json.loads(json.dumps(kern))
    cmp = {tag: compare(main_res["outputs"], passes[tag]) for tag in passes}
    # in-process A/X/A' (A = K0, X = D0 gate_up pins only, A' = K0b) and the pinned repeat
    inproc = {"X_D0_vs_A_K0": compare(passes["K0"], passes["D0"]),
              "Aprime_K0b_vs_A_K0": compare(passes["K0"], passes["K0b"]),
              "ON2_vs_ON": compare(passes["ON"], passes["ON2"])}
    for tag, c in inproc.items():
        log(f"[inproc] {tag}: {json.dumps(c)}")
    k_eq = {k: kern["K0"][k] == main_res["kernels"].get(k) for k in kern["K0"]}
    k_on_diff = {k: {"main": main_res["kernels"].get(k), "pins_on": kern["ON"][k]} for k in kern["ON"]
                 if kern["ON"][k] != main_res["kernels"].get(k)}
    for tag, c in cmp.items():
        log(f"[cmp] {tag} vs main: {json.dumps(c)}")
    log(f"[kernels] K0 == main per M: {k_eq}")
    for k, v in kern["K0"].items():
        log(f"[kernels] K0 {k}: {v}")
    for k, v in k_on_diff.items():
        log(f"[kernels] pins on differ from main at {k}: {v}")
    files = glob.glob(CACHE + "*")
    log(f"[cache] end: {files} sha256={sha(files)}")
    ok = (cmp["K0"]["bitwise_equal"] == cmp["K0"]["total"] and cmp["K0b"]["bitwise_equal"] == cmp["K0b"]["total"]
          and cmp["D0"]["bitwise_equal"] == cmp["D0"]["total"] and all(k_eq.values())
          and all(c["bitwise_equal"] == c["total"] for c in inproc.values()))
    print(json.dumps({"passed": bool(ok), "arm": "candidate", "n_outputs": len(passes["K0"]),
                      "inprocess_AXA": inproc,
                      "both_off_K0_vs_main": cmp["K0"], "both_off_again_K0b_vs_main": cmp["K0b"],
                      "down_pin_off_D0_vs_main": cmp["D0"], "gate_up_pin_off_G0_vs_main": cmp["G0"],
                      "both_on_vs_main": cmp["ON"], "kernel_log_K0_equal_main": k_eq,
                      "kernel_log_pins_on_vs_main_diff": k_on_diff, "cache_sha256_end": sha(files),
                      "compared": "separate processes, main tree vs branch tree, one shared autotuner cache; first "
                                  "token + top-20 logprobs bitwise per prompt (K0/K0b/D0 must be 100%), per-M "
                                  "nvfp4_gemm kernel name+grid"}), flush=True)
    llm.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
