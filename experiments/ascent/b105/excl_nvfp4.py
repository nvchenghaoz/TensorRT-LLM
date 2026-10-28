"""Branch 105 (idea 166): load-time NVFP4 W4A4 for the BF16 checkpoint-excluded FLUX.3 blocks.

Parent:  python excl_nvfp4.py <checkpoint> <modes> [--judge] [--reps N]
  modes: comma list of TRTLLM_FLUX3_EXCLUDED_NVFP4 values (none, all, single, or
  '+'-joined block lists such as single_blocks.24+single_blocks.25); 'main' runs with the
  variable unset (the baseline arm of a pair, where main's tree ignores it).
Each mode runs in a FRESH child process: VisualGen with the benchmark's server settings
(BASE_SERVER of the frozen common.py overlaid by experiments/ascent-config/engine.json:
torch.compile + autotune, warmup [[1024,1024]] t2i), TORCH_LOGS=recompiles. After warmup it
prints a TIMED-START marker and generates, at 1024x1024 / 16 steps / the benchmark's fixed seeds:
the 12 check prompts, the 24 accuracy prompts, then the 12 check prompts again (same-seed
repeat, bitwise compared). PNGs are saved and hashed.
The parent reports per mode: route log lines, recompiles after the marker, per-request
denoise time (pipeline metrics), repeat determinism, per-prompt LPIPS/PSNR vs the frozen
reference images (reference.json store) and vs the first mode's images, and (--judge) the
Qwen3-VL judge counts on the accuracy prompts (same rule as the frozen judge.py).
"""

import copy
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b105"), "b105")

STORE = (
    "/home/scratch.chengzhang/trtllm-dev-flux3-klein-20261027-125409/"
    "ascent-benchmark-data/flux3-klein-nvfp4/v1/ref"
)
REF_SHA = {
    "check00": "6281cc18a39f950ac89b3513513b4957941f8b3760a6abf6c6010d4614e65910",
    "check01": "3e8c204c77fd5ff41e9636fc3471f64a6264be31cc1af24d84619ea741efd95d",
    "check02": "337723400385cfb0dd51111bf108cb9db4ce196f6e069048667aa4ee322f275a",
    "check03": "e0f8f049f3d7f2366268f74900ca926dd7df3dfd8032af1be780c47a6bca18fa",
    "check04": "bb2987bccc6847beb6ed218b9162ee61ce1adee8d84defab4c41cd8706ef0a4b",
    "check05": "3e9da040e7bdc551108abe2e51c3f227556f2be791877f0c72fcd99c85636c16",
    "check06": "829a1e0775f116aa8b1e2aacf88d2e9996dd070ceb6686492ae688c5d119ca96",
    "check07": "f7049fd916722d2554e3fc26e801deeb94d7bf61af2cde00d4aef28041c43943",
    "check08": "75c27e96ba0b60f1ed2d723bb58f72f200b33c3c0343da51c242a55b9221845b",
    "check09": "3b52010c7b0037be7a14465bcb4201e73c7f83486dca04aa616dfbc70bb8bc0f",
    "check10": "5c908d8736812d476f53a3292db4a2945223f930a2a075b6dc738aa20010123d",
    "check11": "29f21a2f3aa6f63c5df84335b536c79fc1df1be1293c4f4504cbf0e3a933e054",
}
TOL = {"lpips_max": 0.265977, "psnr_min_db": 17.658}

CHECK = [
    ("check00", 100, "A red fox standing in fresh snow at sunrise"),
    ("check01", 101, "A lighthouse on a cliff during a thunderstorm, oil painting"),
    ("check02", 102, "A bowl of ramen with a soft boiled egg, studio photo"),
    ("check03", 103, "A futuristic city skyline at night with flying cars"),
    ("check04", 104, "Portrait of an elderly fisherman with a white beard, dramatic lighting"),
    ("check05", 105, "A cozy reading nook with a window, plants and warm lamp light"),
    ("check06", 106, "A watercolor painting of a hot air balloon over green hills"),
    ("check07", 107, "A golden retriever puppy playing with a tennis ball on grass"),
    ("check08", 108, "A vintage travel poster that says VISIT MARS in bold letters"),
    ("check09", 109, "Macro photo of a dew drop on a green leaf"),
    ("check10", 110, "A medieval castle surrounded by autumn forest, aerial view"),
    ("check11", 111, "A robot chef cooking pancakes in a modern kitchen"),
]
ACC = [
    ("acc00", 200, "A red apple and a yellow banana on a white kitchen table",
     [("Is there a red apple in the image?", "yes"), ("Is there a banana in the image?", "yes")]),
    ("acc01", 201, "Three blue coffee mugs in a row on a wooden shelf",
     [("Are there exactly three mugs in the image?", "yes"), ("Are the mugs blue?", "yes")]),
    ("acc02", 202, "A wooden sign that says OPEN hanging on a shop door",
     [("Does the sign say OPEN?", "yes"), ("Does the sign say CLOSED?", "no")]),
    ("acc03", 203, "A black cat sleeping on a green sofa",
     [("Is there a cat in the image?", "yes"), ("Is the sofa green?", "yes")]),
    ("acc04", 204, "A yellow school bus parked next to a red brick building",
     [("Is there a bus in the image?", "yes"), ("Is the bus yellow?", "yes")]),
    ("acc05", 205, "Two white swans swimming on a calm lake at sunset",
     [("Are there exactly two swans in the image?", "yes"), ("Are the birds white?", "yes")]),
    ("acc06", 206, "A neon sign with the word PIZZA glowing on a dark street",
     [("Does the sign show the word PIZZA?", "yes")]),
    ("acc07", 207, "A green bicycle leaning against a white wall",
     [("Is there a bicycle in the image?", "yes"), ("Is the bicycle red?", "no")]),
    ("acc08", 208, "A white plate with four chocolate chip cookies",
     [("Are there exactly four cookies on the plate?", "yes")]),
    ("acc09", 209, "An orange pumpkin next to a purple umbrella in the rain",
     [("Is there a pumpkin in the image?", "yes"), ("Is the umbrella purple?", "yes")]),
    ("acc10", 210, "A small dog wearing a red scarf sitting in the snow",
     [("Is there a dog in the image?", "yes"), ("Is the dog wearing a red scarf?", "yes")]),
    ("acc11", 211, "A blackboard with 2 + 2 = 4 written in white chalk",
     [("Does the blackboard show 2 + 2 = 4?", "yes")]),
    ("acc12", 212, "A white sailboat on a blue ocean under a clear sky",
     [("Is there a sailboat in the image?", "yes"), ("Is there snow in the image?", "no")]),
    ("acc13", 213, "A pink flamingo standing on one leg in shallow water",
     [("Is there a flamingo in the image?", "yes"), ("Is the bird pink?", "yes")]),
    ("acc14", 214, "A red stop sign at a quiet street corner",
     [("Is there a stop sign in the image?", "yes"), ("Is the sign green?", "no")]),
    ("acc15", 215, "Five colorful balloons tied to a park bench",
     [("Are there balloons in the image?", "yes"), ("Is there a bench in the image?", "yes")]),
    ("acc16", 216, "A glass of orange juice next to a croissant on a breakfast table",
     [("Is there a glass of orange juice in the image?", "yes"),
      ("Is there a croissant in the image?", "yes")]),
    ("acc17", 217, "A white coffee cup with the word HELLO printed on it",
     [("Does the cup say HELLO?", "yes")]),
    ("acc18", 218, "A brown horse running across a green meadow",
     [("Is there a horse in the image?", "yes"), ("Is the horse brown?", "yes")]),
    ("acc19", 219, "A blue vintage car parked in front of a yellow house",
     [("Is the car blue?", "yes"), ("Is the house yellow?", "yes")]),
    ("acc20", 220, "A lighthouse with red and white stripes on a rocky coast",
     [("Is there a lighthouse in the image?", "yes"),
      ("Does the lighthouse have red and white stripes?", "yes")]),
    ("acc21", 221, "A teddy bear sitting on a wooden chair",
     [("Is there a teddy bear in the image?", "yes"), ("Is there a cat in the image?", "no")]),
    ("acc22", 222, "Two red roses in a clear glass vase",
     [("Are there roses in the image?", "yes"), ("Are the flowers red?", "yes")]),
    ("acc23", 223, "A book cover with the title THE OCEAN in large white letters",
     [("Does the book cover show the title THE OCEAN?", "yes")]),
]

BASE_VGA = {
    "pipeline_config": {"variant": "nvfp4"},
    "parallel_config": {"tp_size": 1},
    "attention_config": {"backend": "VANILLA"},
    "cuda_graph_config": {"enable": False},
    "torch_compile_config": {"enable": True, "enable_fullgraph": False, "enable_autotune": True},
    "compilation_config": {"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
}

ROUTE_PATTERNS = {
    "converted": "load-time NVFP4 W4A4",
    "fp8r_blocks": "FP8R for BF16 checkpoint-excluded",
    "nvfp4_coverage": "NVFP4 PrimsTS coverage verified",
    "fused_adaln_nvfp4": "fused LN+adaLN+NVFP4 quantize",
    "tail_pair": "two-stream tail pair",
    "fp8r_adaln_head": "FP8R fused LN+adaLN+E4M3",
}


def log(msg):
    print(f"[b105 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ----------------------------------------------------------------------------- child


def child(mode_dir, ckpt, reps):
    from PIL import Image

    from tensorrt_llm import VisualGen
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine = json.load(open(os.path.join(WORKTREE, "experiments", "ascent-config", "engine.json")))
    vga = deep_merge(BASE_VGA, {k: v for k, v in engine.items() if k not in ("env",)})
    vga["pipeline_config"]["variant"] = "nvfp4"
    print("VGA", json.dumps(vga), flush=True)
    t0 = time.time()
    model = VisualGen(model=ckpt, args=VisualGenArgs(**vga))
    print(f"loaded+warmup {time.time() - t0:.0f}s", flush=True)
    print("=== TIMED-START ===", flush=True)
    sys.stderr.write("=== TIMED-START ===\n")
    sys.stderr.flush()

    reqs = []
    seq = [(k, s, p) for k, s, p in CHECK]
    if os.environ.get("B105_CHECK_ONLY") != "1":
        seq += [(k, s, p) for k, s, p, _ in ACC]
    for rep in range(1, reps):
        seq += [(f"{k}#r{rep}", s, p) for k, s, p in CHECK]
    for key, seed, prompt in seq:
        params = model.default_params
        params.height, params.width, params.seed = 1024, 1024, seed
        params.num_inference_steps = 16
        t1 = time.time()
        out = model.generate(inputs=prompt, params=params)
        wall = time.time() - t1
        if getattr(out, "error", None):
            raise RuntimeError(out.error)
        img = out.image
        if img.dim() == 4:
            img = img[0]
        arr = img.detach().cpu().numpy()
        path = os.path.join(mode_dir, f"{key}.png")
        Image.fromarray(arr).save(path)
        m = out.metrics
        rec = {"key": key, "wall_s": round(wall, 4), "denoise": getattr(m, "denoise", None),
               "sha256": hashlib.sha256(arr.tobytes()).hexdigest(), "png_sha256": sha256(path)}
        reqs.append(rec)
        print("REQ", json.dumps(rec, default=str), flush=True)
    print("=== TIMED-END ===", flush=True)
    json.dump(reqs, open(os.path.join(mode_dir, "requests.json"), "w"), default=str)
    try:
        model.shutdown()
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------- metrics


_LP = {}


def _lpips():
    if "m" in _LP:
        return _LP["m"]
    import lpips

    try:
        m = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    except TypeError:
        import lpips.pretrained_networks as pn
        import torchvision.models as tvm

        original = tvm.alexnet

        def alexnet(pretrained=False, **kwargs):
            if pretrained:
                kwargs.setdefault("weights", tvm.AlexNet_Weights.IMAGENET1K_V1)
            return original(**kwargs)

        pn.tv.alexnet = alexnet
        m = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    _LP["m"] = m.to("cuda").eval()
    return _LP["m"]


def compare(path_a, path_b):
    import numpy as np
    import torch
    from PIL import Image

    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.uint8)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.uint8)
    diff = a.astype(np.int16) - b.astype(np.int16)
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = 100.0 if mse == 0 else min(100.0, 10 * math.log10(255.0**2 / mse))

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32) / 127.5 - 1.0).permute(
            2, 0, 1)[None].cuda()

    with torch.no_grad():
        lp = float(_lpips()(t(a), t(b)).item())
    return {"lpips": round(lp, 4), "psnr": round(psnr, 3), "bitwise": mse == 0.0,
            "pass_v2": lp <= TOL["lpips_max"] and psnr >= TOL["psnr_min_db"],
            "black": bool(b.max() < 8), "mean": round(float(b.mean()), 2)}


# ----------------------------------------------------------------------------- judge


def judge_modes(ckpt, modes):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_dir = os.path.join(ckpt, "text_encoder")
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    proc = AutoProcessor.from_pretrained(model_dir)
    tok = proc.tokenizer
    yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0]
                      for w in ("yes", "Yes", "YES", " yes", " Yes")})
    no_ids = sorted({tok.encode(w, add_special_tokens=False)[0]
                     for w in ("no", "No", "NO", " no", " No")})

    @torch.inference_mode()
    def ask(image, question):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{question} Answer with a single word: yes or no."},
        ]}]
        inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt")
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        logits = model(**inputs).logits[0, -1].float()
        lse_yes = torch.logsumexp(logits[yes_ids], 0).item()
        lse_no = torch.logsumexp(logits[no_ids], 0).item()
        out = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        reply = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        cleaned = re.sub(r"<think>.*?</think>", " ", reply, flags=re.S).strip().lower()
        match = re.search(r"\b(yes|no)\b", cleaned)
        if match:
            return match.group(1), round(lse_yes - lse_no, 3)
        return ("yes" if lse_yes > lse_no else "no"), round(lse_yes - lse_no, 3)

    res = {}
    for mode in modes:
        d = os.path.join(OUT, mode)
        correct, total, wrong = 0, 0, []
        for key, _, _, qs in ACC:
            path = os.path.join(d, f"{key}.png")
            img = Image.open(path).convert("RGB") if os.path.exists(path) else None
            for qi, (q, expect) in enumerate(qs):
                total += 1
                if img is None:
                    wrong.append(f"{key}.q{qi}:noimg")
                    continue
                ans, margin = ask(img, q)
                if ans == expect:
                    correct += 1
                else:
                    wrong.append(f"{key}.q{qi}:{ans}({margin})")
        res[mode] = {"correct": correct, "total": total, "wrong": wrong}
        log(f"judge {mode}: {correct}/{total} wrong={wrong}")
    return res


# ----------------------------------------------------------------------------- parent


def parent(ckpt, modes, do_judge, reps):
    import tensorrt_llm

    os.makedirs(OUT, exist_ok=True)
    log(f"tensorrt_llm.__file__={tensorrt_llm.__file__}")

    def git(*a):
        return subprocess.run(["git", "-C", WORKTREE, *a], capture_output=True, text=True)

    head = git("rev-parse", "HEAD").stdout.strip()
    log("HEAD " + head)
    log("log\n" + git("log", "--oneline", "-5").stdout)
    anc = git("merge-base", "--is-ancestor", "ac3317ba", "HEAD").returncode
    log(f"MAIN_AT_START ac3317ba ancestor rc={anc}")
    summary = {"head": head, "ancestry_rc": anc, "modes": {}}
    for mode in modes:
        d = os.path.join(OUT, mode)
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ)
        env["TORCH_LOGS"] = "recompiles"
        env["TLLM_LOG_LEVEL"] = "INFO"
        env.pop("TRTLLM_FLUX3_EXCLUDED_NVFP4", None)
        knob = mode.split("@", 1)[0]  # "<value>@<tag>" repeats a value in another fresh process
        if knob not in ("main", "default"):
            env["TRTLLM_FLUX3_EXCLUDED_NVFP4"] = knob.replace("+", ",")
        log(f"=== mode {mode} env={env.get('TRTLLM_FLUX3_EXCLUDED_NVFP4')}")
        log_path = os.path.join(d, "log.txt")
        t0 = time.time()
        with open(log_path, "w") as lf:
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "child", d, ckpt, str(reps)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
            for line in p.stdout:
                lf.write(line)
                if line.startswith(("REQ", "===", "loaded")) or any(
                        pat in line for pat in ROUTE_PATTERNS.values()) or "Error" in line:
                    print(mode, line.rstrip()[:600], flush=True)
            rc = p.wait()
        text = open(log_path, errors="replace").read()
        warm, _, timed = text.partition("=== TIMED-START ===")
        rec_re = re.compile(r"Recompiling function|recompil", re.I)
        cap_re = re.compile(r"[Cc]apturing CUDA graph|graph capture", re.I)
        routes = {}
        for k, pat in ROUTE_PATTERNS.items():
            lines = [ln for ln in text.splitlines() if pat in ln]
            routes[k] = lines[-1].split(pat, 1)[-1][-500:] if lines else None
        reqs = json.load(open(os.path.join(d, "requests.json"))) if rc == 0 else []
        by_key = {r["key"]: r for r in reqs}
        repeat_ok = None
        if rc == 0 and reps > 1:
            repeat_ok = all(by_key[f"{k}#r{r}"]["sha256"] == by_key[k]["sha256"]
                            for k, _, _ in CHECK for r in range(1, reps))
        den = []
        for r in reqs:
            try:
                den.append(float(r["denoise"]))
            except (TypeError, ValueError):
                pass
        den_steady = den[1:] if len(den) > 1 else den
        s = {
            "rc": rc,
            "elapsed_s": round(time.time() - t0),
            "routes": routes,
            "recompile_lines_timed": [ln[:240] for ln in timed.splitlines() if rec_re.search(ln)][:8],
            "recompiles_timed": sum(1 for ln in timed.splitlines() if rec_re.search(ln)),
            "graph_captures_timed": sum(1 for ln in timed.splitlines() if cap_re.search(ln)),
            "graph_capture_lines_timed": [ln[:240] for ln in timed.splitlines() if cap_re.search(ln)][:4],
            "repeat_bitwise": repeat_ok,
            "denoise_median": statistics.median(den_steady) if den_steady else None,
            "denoise_mean": statistics.fmean(den_steady) if den_steady else None,
            "wall_median_s": statistics.median([r["wall_s"] for r in reqs[1:]]) if len(reqs) > 1 else None,
            "sha256": {r["key"]: r["sha256"][:16] for r in reqs if "#" not in r["key"]},
        }
        if rc != 0:
            s["tail"] = text[-3000:]
        summary["modes"][mode] = s
        log(f"mode {mode} summary " + json.dumps({k: v for k, v in s.items() if k != 'sha256'}))
        log(f"mode {mode} sha256 " + json.dumps(s["sha256"]))

    ok_modes = [m for m in modes if summary["modes"][m]["rc"] == 0]
    first = ok_modes[0] if ok_modes else None
    for mode in ok_modes:
        d = os.path.join(OUT, mode)
        vs_ref, vs_first = {}, {}
        for key, _, _ in CHECK:
            ref = os.path.join(STORE, f"{REF_SHA[key]}.png")
            if os.path.exists(ref):
                vs_ref[key] = compare(ref, os.path.join(d, f"{key}.png"))
            if mode != first:
                vs_first[key] = compare(os.path.join(OUT, first, f"{key}.png"),
                                        os.path.join(d, f"{key}.png"))
        if mode != first:
            for key, _, _, _ in ACC:
                if not os.path.exists(os.path.join(d, f"{key}.png")):
                    continue
                vs_first[key] = compare(os.path.join(OUT, first, f"{key}.png"),
                                        os.path.join(d, f"{key}.png"))
        s = summary["modes"][mode]
        s["vs_reference"] = vs_ref
        s["vs_reference_pass_v2"] = sum(v["pass_v2"] for v in vs_ref.values())
        s["vs_reference_max_lpips"] = max((v["lpips"] for v in vs_ref.values()), default=None)
        s["vs_reference_min_psnr"] = min((v["psnr"] for v in vs_ref.values()), default=None)
        if vs_first:
            s["vs_first_mode"] = vs_first
            s["vs_first_mode_name"] = first
            s["vs_first_max_lpips"] = max(v["lpips"] for v in vs_first.values())
            s["vs_first_mean_lpips"] = round(statistics.fmean(v["lpips"] for v in vs_first.values()), 4)
            s["vs_first_min_psnr"] = min(v["psnr"] for v in vs_first.values())
            s["vs_first_bitwise"] = sum(v["bitwise"] for v in vs_first.values())
            s["vs_first_black"] = sum(v["black"] for v in vs_first.values())
        log(f"mode {mode}: vs_ref pass {s['vs_reference_pass_v2']}/{len(vs_ref)} "
            f"maxLPIPS {s['vs_reference_max_lpips']} minPSNR {s['vs_reference_min_psnr']}; "
            + (f"vs {first}: maxLPIPS {s['vs_first_max_lpips']} mean {s['vs_first_mean_lpips']} "
               f"minPSNR {s['vs_first_min_psnr']} bitwise {s['vs_first_bitwise']}" if vs_first else ""))
        for key, v in vs_ref.items():
            log(f"  {mode} {key} ref lpips={v['lpips']} psnr={v['psnr']} pass={v['pass_v2']}"
                + (f" | vs {first} lpips={vs_first[key]['lpips']} psnr={vs_first[key]['psnr']}"
                   if key in vs_first else ""))
    if do_judge and ok_modes:
        _LP.clear()
        import torch

        torch.cuda.empty_cache()
        try:
            j = judge_modes(ckpt, ok_modes)
            for m, v in j.items():
                summary["modes"][m]["judge"] = v
        except Exception as exc:  # noqa: BLE001
            log(f"judge failed: {exc!r}"[:600])
            summary["judge_error"] = repr(exc)[:600]
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    passed = bool(ok_modes) and len(ok_modes) == len(modes) and all(
        summary["modes"][m]["repeat_bitwise"] in (True, None) for m in ok_modes)
    compact = {m: {k: v for k, v in s.items()
                   if k not in ("vs_reference", "vs_first_mode", "sha256", "tail")}
               for m, s in summary["modes"].items()}
    print(json.dumps({"passed": passed, "ancestry_rc": anc,
                      "compared": "per mode: frozen reference.json check images (gate v2 info), "
                                  "first mode's images, same-seed repeat bitwise, judge",
                      "modes": compact}, default=str))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        args = sys.argv[1:]
        do_judge = "--judge" in args
        reps = 2
        if "--reps" in args:
            i = args.index("--reps")
            reps = int(args[i + 1])
            args = args[:i] + args[i + 2:]
        pos = [a for a in args if not a.startswith("--")]
        parent(pos[0], pos[1].split(","), do_judge, reps)
