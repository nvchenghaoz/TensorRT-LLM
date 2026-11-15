"""In-process full-model checks for idea 64 (down_proj wave split).

Copy of experiments/ascent/mf2_llm_ab.py (never edited), adapted per the
top-lead rulings 1055/1083 and nvfp4_mlp plan rev 34:
- one process (TLLM_WORKER_USE_SINGLE_PROCESS=1), batch_wait_timeout_ms 1500,
  TRTLLM_FI_HOST_PREFILL_PLAN=1 in every pass, own make_text prompts
  (seed 20261114), 1 greedy token + top-20 logprobs;
- cases isl300@1, isl1k@1, isl8k@1 (c=1), isl1k@4 and isl8k@4 groups sent
  together, isl300@16 groups (M=4800 steps);
- passes A (switch OFF = main), X (ON), A2 (OFF); X judged only if A==A2;
- RE-SEND in the same process, no reset: switch ON, each class's first 2
  prompts (first group for c>1) re-sent in order 8k,1k,8k(,300), compared
  bitwise with pass X; then the same with the switch OFF vs pass A;
- CAPTURE pass (switch ON): at layers 0/30/59 the real down_proj inputs are
  run through main's nvfp4_gemm and the split, compared int16 in-process;
  gate_up and down act_fp4/act_sf/alpha are written to the shared captures
  dir with a manifest (plan rev 34 SHARED CAPTURES).
usage: llm_ab.py <checkpoint>
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


SHARED = "/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/shared/nvfp4_mlp_captures"
CAP_LAYERS = (0, 30, 59)


def main():
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ["TRTLLM_FI_HOST_PREFILL_PLAN"] = "1"
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
    threading.Thread(target=_heartbeat, daemon=True).start()
    ck = sys.argv[1]
    npc = int(os.environ.get("NPC", "20"))
    main_commit = os.environ.get("MAIN_COMMIT", "e58d0587")

    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm._torch.modules import linear as lm
    from tensorrt_llm._torch.modules.gemma4 import down_wave_split as dws
    from tensorrt_llm.llmapi import KvCacheConfig

    tok = AutoTokenizer.from_pretrained(ck)
    rng = random.Random(20261114)

    def mk(isl):
        ids = tok(make_text(rng, isl * 2), add_special_tokens=False)["input_ids"][: isl - 1]
        ids = [tok.bos_token_id or 2] + ids
        assert len(ids) == isl, (len(ids), isl)
        return ids

    cases = [("isl300@1", 300, 1, npc), ("isl1k@1", 1000, 1, npc), ("isl8k@1", 8000, 1, npc),
             ("isl1k@4", 1000, 4, npc), ("isl8k@4", 8000, 4, npc), ("isl300@16", 300, 16, 32)]
    prompts = []  # (label, conc, ids)
    for label, isl, conc, cnt in cases:
        for _ in range(cnt):
            prompts.append((label, conc, mk(isl)))
    cap_prompts = [("cap_isl300@1", 1, [mk(300)]), ("cap_isl1k@1", 1, [mk(1000)]),
                   ("cap_isl2k@1", 1, [mk(2000)]), ("cap_isl8k@1", 1, [mk(8000)]),
                   ("cap_4x8k", 4, [mk(8000) for _ in range(4)]),
                   ("cap_16x300", 16, [mk(300) for _ in range(16)])]
    print(f"built {len(prompts)} prompts in {len(cases)} classes", flush=True)

    # ---- instrumentation ----
    counts = {"split": 0, "down_calls_ge4096": 0}
    _orig_split = dws.down_wave_split_gemm

    def split_counted(*a, **k):
        counts["split"] += 1
        return _orig_split(*a, **k)

    dws.down_wave_split_gemm = split_counted

    cap = {"on": False, "layer": None, "label": None, "step": 0, "dir": None, "files": [], "opcmp": []}
    _orig_mlp_fwd = mg._Gemma4GeluQuantMLP.forward

    def save(name, t):
        p = os.path.join(cap["dir"], name)
        torch.save(t.detach().cpu(), p)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        cap["files"].append(dict(file=name, sha256=h, shape=list(t.shape), dtype=str(t.dtype)))

    def mlp_fwd(self, x, *a, **k):
        cap["layer"] = self.layer_idx
        if cap["on"] and cap["dir"] and self.layer_idx in CAP_LAYERS and hasattr(x, "fp4_tensor"):
            gu = self.gate_up_proj
            af, asf, al = gu.quant_method._input_prepare(gu, x)
            base = f"{cap['label']}_s{cap['step']}_L{self.layer_idx}_M{af.shape[0]}_gate_up"
            save(base + "_act_fp4.pt", af)
            save(base + "_act_sf.pt", asf)
            save(base + "_alpha.pt", al)
        return _orig_mlp_fwd(self, x, *a, **k)

    mg._Gemma4GeluQuantMLP.forward = mlp_fwd
    _orig_apply = lm.NVFP4LinearMethod.apply

    def apply(self, module, input, bias):
        if getattr(module, "_gemma4_down_wave_split", False):
            m = input.fp4_tensor.shape[0] if hasattr(input, "fp4_tensor") else -1
            if m >= 4096:
                counts["down_calls_ge4096"] += 1
            if cap["on"] and cap["layer"] in CAP_LAYERS and hasattr(input, "fp4_tensor"):
                af, asf, al = self._input_prepare(module, input)
                ref = torch.ops.trtllm.nvfp4_gemm(af, module.weight, asf, module.weight_scale, al,
                                                  module.dtype)
                applies = dws.prepare() and dws.applies(af, module.weight, asf, al, module.dtype)
                if applies:
                    y = _orig_split(af, module.weight, asf, module.weight_scale, al)
                    nd = int((y.view(torch.int16) != ref.view(torch.int16)).sum())
                    first = None
                    if nd:
                        idx = int((y.view(torch.int16) != ref.view(torch.int16)).flatten().nonzero()[0])
                        first = divmod(idx, y.shape[1])
                    pieces = dws.plan(af.shape[0], module.weight.shape[0],
                                      torch.cuda.get_device_properties(af.device).multi_processor_count)
                else:
                    nd, first, pieces = None, None, None
                cap["opcmp"].append(dict(label=cap["label"], step=cap["step"], layer=cap["layer"],
                                         M=int(af.shape[0]), applies=bool(applies), int16_diff=nd,
                                         first=first, pieces=pieces))
                if cap["dir"]:
                    base = f"{cap['label']}_s{cap['step']}_L{cap['layer']}_M{af.shape[0]}_down"
                    save(base + "_act_fp4.pt", af)
                    save(base + "_act_sf.pt", asf)
                    save(base + "_alpha.pt", al)
                    if af.shape[0] < 4096:
                        save(base + "_main_out.pt", ref)
        return _orig_apply(self, module, input, bias)

    lm.NVFP4LinearMethod.apply = apply

    llm = LLM(
        model=ck,
        max_num_tokens=8192,
        max_seq_len=8300,
        max_batch_size=64,
        batch_wait_timeout_ms=1500,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.85, enable_block_reuse=False),
    )
    print(f"loaded; FI module loaded={dws._state['mod'] is not None} "
          f"TRTLLM_FI_HOST_PREFILL_PLAN={os.environ['TRTLLM_FI_HOST_PREFILL_PLAN']}", flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def send(group):
        outs = llm.generate([{"prompt_token_ids": g[2]} for g in group], sp, use_tqdm=False)
        res = []
        for g, o in zip(group, outs):
            c = o.outputs[0]
            lp = c.logprobs[0] if c.logprobs else {}
            res.append({"label": g[0], "tok": int(c.token_ids[0]),
                        "top": {int(t): float(v.logprob) for t, v in lp.items()}})
        return res

    def run_pass(tag):
        res = []
        i = 0
        t0 = time.time()
        c0 = dict(counts)
        while i < len(prompts):
            conc = prompts[i][1]
            res += send(prompts[i: i + conc])
            i += conc
        d = {k: counts[k] - c0[k] for k in counts}
        print(f"[{tag}] {len(res)} prompts in {time.time() - t0:.1f}s dispatch={d}", flush=True)
        return res, d

    # warmup on both paths (FI module JIT/load happens at engine warmup already)
    for flag in (False, True):
        dws.ENABLED = flag
        send([prompts[0]])
        send([p for p in prompts if p[0] == "isl8k@1"][:1])

    passes, calls = {}, {}
    for tag, flag in (("A", False), ("X", True), ("A2", False)):
        dws.ENABLED = flag
        passes[tag], calls[tag] = run_pass(tag)

    def compare(a, b):
        eq = flips = 0
        maxd = 0.0
        per = {}
        first_diff = []
        for x, y in zip(a, b):
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
                first_diff.append(dict(label=x["label"], a_tok=x["tok"], b_tok=y["tok"]))
        return {"bitwise_equal": eq, "total": len(a), "token_flips": flips, "max_abs_dlogprob": maxd,
                "per_case_equal_total_flips": per, "first_diffs": first_diff}

    aa = compare(passes["A"], passes["A2"])
    ax = compare(passes["A"], passes["X"])
    print("A vs A2:", json.dumps(aa), flush=True)
    print("A vs X :", json.dumps(ax), flush=True)

    # ---- re-send, same process, no reset ----
    def class_items(res_list, label):
        idx = [i for i, p in enumerate(prompts) if p[0] == label]
        conc = prompts[idx[0]][1]
        take = idx[:2] if conc == 1 else idx[:conc]
        return take

    order = ["isl8k@1", "isl8k@4", "isl1k@1", "isl1k@4", "isl8k@1", "isl8k@4", "isl300@1", "isl300@16"]
    resend = {}
    for tag, flag, refp in (("resend_ON", True, "X"), ("resend_OFF", False, "A")):
        dws.ENABLED = flag
        c0 = dict(counts)
        diffs = {}
        for label in order:
            take = class_items(passes[refp], label)
            conc = prompts[take[0]][1]
            if conc == 1:
                outs = []
                for i in take:
                    outs += send([prompts[i]])
            else:
                outs = send([prompts[i] for i in take])
            ref = [passes[refp][i] for i in take]
            bad = [j for j, (o, r) in enumerate(zip(outs, ref)) if o["tok"] != r["tok"] or o["top"] != r["top"]]
            e = diffs.setdefault(label, [0, 0])
            e[0] += len(outs)
            e[1] += len(bad)
        resend[tag] = dict(per_class_sent_mismatch=diffs, dispatch={k: counts[k] - c0[k] for k in counts})
        print(f"[{tag}] {json.dumps(resend[tag])}", flush=True)

    # ---- capture pass (switch ON) ----
    dws.ENABLED = True
    cap_dir = os.path.join(SHARED, main_commit)
    write_caps = not os.path.exists(os.path.join(cap_dir, "manifest.json"))
    tmp = None
    if write_caps:
        try:
            os.makedirs(SHARED, exist_ok=True)
            tmp = f"{cap_dir}.tmp-{os.getpid()}"
            os.makedirs(tmp, exist_ok=True)
            open(os.path.join(tmp, ".write_test"), "w").write("ok")
            print(f"capture dir write test OK: {tmp}", flush=True)
        except Exception as e:  # noqa
            tmp = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "captures")
            os.makedirs(tmp, exist_ok=True)
            print(f"shared dir not writable ({e}); using {tmp}", flush=True)
    else:
        print(f"captures already present at {cap_dir}; op-compare only", flush=True)
    cap["dir"] = tmp
    cap["on"] = True
    for label, conc, groups in cap_prompts:
        cap["label"] = label
        cap["step"] = 0
        group = [(label, conc, ids) for ids in groups]
        if label == "cap_4x8k":
            # 4 x 8000 tokens: one 8000-row step per request under max_num_tokens 8192
            send(group)
        else:
            send(group)
        cap["step"] += 1
    cap["on"] = False
    manifest_path = None
    if tmp is not None:
        manifest = dict(capture_commit=main_commit, branch="ascent/nvfp4_mlp/31", layers=list(CAP_LAYERS),
                        prompts={lab: [len(i) for i in g] for lab, _, g in cap_prompts},
                        note="act_fp4 uint8 [M,K/2], act_sf uint8 swizzled 128x4 (1-D), alpha fp32[1]; "
                             "down main_out only at M<4096 (recompute at >=4096). Weights: by layer index "
                             "from the checkpoint. Files named <label>_s<step>_L<layer>_M<M>_<op>_*.pt; "
                             "step = request-group index (rows of a packed step = concatenated prompts).",
                        files=cap["files"])
        json.dump(manifest, open(os.path.join(tmp, "manifest.json"), "w"), indent=1)
        art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
        shutil.copy(os.path.join(tmp, "manifest.json"), os.path.join(art, "captures_manifest.json"))
        if tmp.startswith(cap_dir + ".tmp-"):
            try:
                os.rename(tmp, cap_dir)
                manifest_path = os.path.join(cap_dir, "manifest.json")
            except OSError as e:
                print(f"rename failed ({e}); captures left at {tmp}", flush=True)
                manifest_path = os.path.join(tmp, "manifest.json")
        else:
            manifest_path = os.path.join(tmp, "manifest.json")
        print(f"captures written: {len(cap['files'])} files -> {manifest_path}", flush=True)
    llm.shutdown()

    op = cap["opcmp"]
    op_split = [o for o in op if o["applies"]]
    op_ok = all(o["int16_diff"] == 0 for o in op_split) and len(op_split) > 0
    for o in op:
        print(f"opcmp {o['label']} L{o['layer']} M={o['M']} applies={o['applies']} diff={o['int16_diff']} "
              f"pieces={o['pieces']}", flush=True)
    ok = (aa["bitwise_equal"] == aa["total"] and ax["bitwise_equal"] == ax["total"]
          and calls["X"]["split"] > 0 and calls["A"]["split"] == 0 and calls["A2"]["split"] == 0
          and all(v[1] == 0 for r in resend.values() for v in r["per_class_sent_mismatch"].values())
          and resend["resend_OFF"]["dispatch"]["split"] == 0 and resend["resend_ON"]["dispatch"]["split"] > 0
          and op_ok)
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    json.dump(dict(A_vs_A2=aa, A_vs_X=ax, dispatch=calls, resend=resend, opcmp=op, manifest=manifest_path),
              open(os.path.join(art, "llm_ab.json"), "w"), indent=1, default=str)
    print(json.dumps({"passed": bool(ok),
                      "compared": "in-process LLM, own prompts, wave split OFF/ON/OFF (A/X/A2) first-token id + "
                                  "top-20 logprobs bitwise; same-process re-send ON vs X and OFF vs A; op-level "
                                  "int16 split vs main nvfp4_gemm on real down_proj inputs at layers 0/30/59",
                      "TRTLLM_FI_HOST_PREFILL_PLAN": os.environ["TRTLLM_FI_HOST_PREFILL_PLAN"],
                      "A_vs_A2": {k: aa[k] for k in ("bitwise_equal", "total", "token_flips")},
                      "A_vs_X": {k: ax[k] for k in ("bitwise_equal", "total", "token_flips",
                                                    "per_case_equal_total_flips")},
                      "dispatch_per_pass": calls, "resend": resend,
                      "opcmp_split_cases": len(op_split), "opcmp_all_int16_equal": op_ok,
                      "captures_manifest": manifest_path}, default=str))


if __name__ == "__main__":
    main()
