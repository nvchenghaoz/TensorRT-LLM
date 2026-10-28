"""Branch 103 (idea 162): gate-v2 row for the one-launch LPT NVFP4 tail pair, same-job baseline.

Parent:  python gate_v2.py <checkpoint>
Two arms, each in a FRESH child process, both pointed at the SAME fresh TORCHINDUCTOR_CACHE_DIR /
TRITON_CACHE_DIR (pitfall 1289), so A-vs-B differences come only from the pair kernel:
  A = TRTLLM_FLUX3_SM120_PAIR_KERNEL=0  (main's two-stream tactic-18 path)
  B = variable unset                    (default: LPT dynamic one-launch pair)
Each child: VisualGen with the benchmark's BASE_SERVER settings overlaid by
experiments/ascent-config/engine.json (torch.compile + autotune, warmup [[1024,1024]] t2i), then the
12 check prompts at their fixed seeds (1024x1024, 16 steps), then the 12 again (same-seed repeat,
bitwise compared). Call counters wrap flux3_sm120_lpt_pair.run (B) and
flux3_primsts._sm120_tail_pair (A) to prove the route and the pairs/request.
Report: per prompt LPIPS/PSNR vs the frozen reference (gate v2: LPIPS <= 0.265977 and
PSNR >= 17.658 dB, same metric code as experiments/ascent-benchmark/metrics.py), B-A margins,
failures/12 per arm, image B vs A, repeat bitwise per arm. Pass rule (plan 1309/1332): B failures
<= A failures and no prompt that passes in A fails in B; repeats bitwise in both arms.
"""

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b103"), "b103_gate")
MAIN = "0cb5fddfa0322c26e0c082294207881fa7a6b9ad"

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

BASE_VGA = {
    "pipeline_config": {"variant": "nvfp4"},
    "parallel_config": {"tp_size": 1},
    "attention_config": {"backend": "VANILLA"},
    "cuda_graph_config": {"enable": False},
    "torch_compile_config": {"enable": True, "enable_fullgraph": False, "enable_autotune": True},
    "compilation_config": {"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
}

ARMS = [("A_env0", "0"), ("B_lpt", None)]


def log(msg):
    print(f"[b103 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ----------------------------------------------------------------------------- child


def child(arm_dir, ckpt):
    from PIL import Image

    from tensorrt_llm import VisualGen
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    counts = {"lpt_run": 0, "two_stream_tail_pair": 0}
    orig_run, orig_pair = lpt.run, fp._sm120_tail_pair

    def run(*a, **k):
        counts["lpt_run"] += 1
        # Per (down M, attn M) shape. Counted at Python call time: eager steps and graph captures
        # (CUDA-graph replays run no Python, so served requests may show 0 here).
        key = f"lpt_M{a[0].shape[0]}_{a[6].shape[0]}"
        counts[key] = counts.get(key, 0) + 1
        return orig_run(*a, **k)

    def pair(*a, **k):
        counts["two_stream_tail_pair"] += 1
        return orig_pair(*a, **k)

    lpt.run = run
    fp._sm120_tail_pair = pair
    print("lpt.enabled()", lpt.enabled(), "sched", lpt.sched_mode(), flush=True)

    engine = json.load(open(os.path.join(WORKTREE, "experiments", "ascent-config", "engine.json")))
    structured = {k: engine[k] for k in ("visual_gen_args",) if k in engine}
    bare = {k: v for k, v in engine.items() if k not in ("visual_gen_args", "env")}
    vga = deep_merge(BASE_VGA, structured.get("visual_gen_args", {}))
    vga = deep_merge(vga, bare)
    vga["pipeline_config"]["variant"] = "nvfp4"
    for k, v in (engine.get("env") or {}).items():
        os.environ.setdefault(str(k), str(v))
    print("VGA", json.dumps(vga), "ENGINE_ENV", json.dumps(engine.get("env") or {}), flush=True)
    t0 = time.time()
    model = VisualGen(model=ckpt, args=VisualGenArgs(**vga))
    print(f"loaded+warmup {time.time() - t0:.0f}s counts_after_warmup {json.dumps(counts)}", flush=True)
    json.dump(dict(counts), open(os.path.join(arm_dir, "warmup_counts.json"), "w"))

    reqs = []
    seq = [(k, s, p) for k, s, p in CHECK] + [(f"{k}#r1", s, p) for k, s, p in CHECK]
    for key, seed, prompt in seq:
        params = model.default_params
        params.height, params.width, params.seed = 1024, 1024, seed
        params.num_inference_steps = 16
        before = dict(counts)
        t1 = time.time()
        out = model.generate(inputs=prompt, params=params)
        wall = time.time() - t1
        if getattr(out, "error", None):
            raise RuntimeError(out.error)
        img = out.image
        if img.dim() == 4:
            img = img[0]
        arr = img.detach().cpu().numpy()
        path = os.path.join(arm_dir, f"{key}.png")
        Image.fromarray(arr).save(path)
        m = out.metrics
        rec = {"key": key, "wall_s": round(wall, 4), "denoise": getattr(m, "denoise", None),
               "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
               "calls": {k: counts[k] - before.get(k, 0) for k in counts}}
        reqs.append(rec)
        print("REQ", json.dumps(rec, default=str), flush=True)
    json.dump(reqs, open(os.path.join(arm_dir, "requests.json"), "w"), default=str)
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
    return {"lpips": round(lp, 6), "psnr": round(psnr, 4), "bitwise": mse == 0.0,
            "max_abs": int(abs(diff).max()),
            "pass_v2": lp <= TOL["lpips_max"] and psnr >= TOL["psnr_min_db"]}


# ----------------------------------------------------------------------------- parent


def parent(ckpt):
    import tensorrt_llm

    os.makedirs(OUT, exist_ok=True)

    def git(*a):
        return subprocess.run(["git", "-C", WORKTREE, *a], capture_output=True, text=True)

    head = git("rev-parse", "HEAD").stdout.strip()
    anc = git("merge-base", "--is-ancestor", MAIN, "HEAD").returncode
    log(f"tensorrt_llm.__file__={tensorrt_llm.__file__}")
    log(f"HEAD {head}")
    log("log\n" + git("log", "--oneline", "-4").stdout)
    log(f"ancestry: git merge-base --is-ancestor {MAIN} HEAD -> rc={anc}")
    if anc != 0:
        print(json.dumps({"passed": False, "error": "ancestry probe rc != 0", "ancestry_rc": anc,
                          "head": head}))
        return
    cache = tempfile.mkdtemp(prefix="b103_gate_cache_")
    log(f"shared fresh caches under {cache}")
    summary = {"head": head, "main": MAIN, "ancestry_rc": anc,
               "tensorrt_llm_file": tensorrt_llm.__file__, "arms": {}}
    for arm, val in ARMS:
        d = os.path.join(OUT, arm)
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ)
        env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache, "inductor")
        env["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
        env.pop("TRTLLM_FLUX3_SM120_PAIR_KERNEL", None)
        env.pop("TRTLLM_FLUX3_SM120_PAIR_SCHED", None)
        if val is not None:
            env["TRTLLM_FLUX3_SM120_PAIR_KERNEL"] = val
        log(f"=== arm {arm} TRTLLM_FLUX3_SM120_PAIR_KERNEL={env.get('TRTLLM_FLUX3_SM120_PAIR_KERNEL')}")
        t0 = time.time()
        with open(os.path.join(d, "log.txt"), "w") as lf:
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "child", d, ckpt],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
            for line in p.stdout:
                lf.write(line)
                if line.startswith(("REQ", "loaded", "lpt.enabled", "VGA")) or "Error" in line:
                    print(arm, line.rstrip()[:700], flush=True)
            rc = p.wait()
        reqs = json.load(open(os.path.join(d, "requests.json"))) if rc == 0 else []
        by = {r["key"]: r for r in reqs}
        s = {"rc": rc, "elapsed_s": round(time.time() - t0)}
        if rc == 0:
            s["repeat_bitwise"] = all(by[f"{k}#r1"]["sha256"] == by[k]["sha256"] for k, _, _ in CHECK)
            s["calls_per_request"] = sorted({json.dumps(r["calls"], sort_keys=True) for r in reqs})
            s["calls_during_load_warmup_capture"] = json.load(open(os.path.join(d, "warmup_counts.json")))
            s["denoise_s"] = [r["denoise"] for r in reqs]
            s["sha256"] = {k: by[k]["sha256"][:16] for k, _, _ in CHECK}
        else:
            s["tail"] = open(os.path.join(d, "log.txt"), errors="replace").read()[-3000:]
        summary["arms"][arm] = s
        log(f"arm {arm}: " + json.dumps({k: v for k, v in s.items() if k not in ("sha256", "denoise_s")}))

    shutil.rmtree(cache, ignore_errors=True)
    a_name, b_name = ARMS[0][0], ARMS[1][0]
    ok = all(summary["arms"][a]["rc"] == 0 for a, _ in ARMS)
    rows = []
    if ok:
        for key, _, _ in CHECK:
            ref = os.path.join(STORE, f"{REF_SHA[key]}.png")
            ra = compare(ref, os.path.join(OUT, a_name, f"{key}.png"))
            rb = compare(ref, os.path.join(OUT, b_name, f"{key}.png"))
            ab = compare(os.path.join(OUT, a_name, f"{key}.png"), os.path.join(OUT, b_name, f"{key}.png"))
            row = {"key": key,
                   "A_lpips": ra["lpips"], "A_psnr": ra["psnr"], "A_pass": ra["pass_v2"],
                   "B_lpips": rb["lpips"], "B_psnr": rb["psnr"], "B_pass": rb["pass_v2"],
                   "B_minus_A_psnr": round(rb["psnr"] - ra["psnr"], 4),
                   "B_minus_A_lpips": round(rb["lpips"] - ra["lpips"], 6),
                   "B_vs_A_psnr": ab["psnr"], "B_vs_A_lpips": ab["lpips"],
                   "B_vs_A_bitwise": ab["bitwise"], "B_vs_A_max_abs_px": ab["max_abs"]}
            rows.append(row)
            log("ROW " + json.dumps(row))
    fa = [r["key"] for r in rows if not r["A_pass"]]
    fb = [r["key"] for r in rows if not r["B_pass"]]
    new_fail = [k for k in fb if k not in fa]
    rep_ok = ok and all(summary["arms"][a]["repeat_bitwise"] for a, _ in ARMS)
    gate_ok = ok and len(fb) <= len(fa) and not new_fail
    summary.update({"rows": rows, "A_failures": fa, "B_failures": fb, "B_new_failures": new_fail})
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    c11 = next((r for r in rows if r["key"] == "check11"), None)
    print(json.dumps({
        "passed": bool(gate_ok and rep_ok),
        "compared": "per check prompt: LPIPS(alex v0.1)/PSNR vs frozen reference.json images, gate v2 "
                    "(LPIPS<=0.265977, PSNR>=17.658); arm B (LPT default) vs arm A (env=0 main path) "
                    "in the same job with shared fresh inductor/triton caches",
        "head": head, "ancestry_rc_main_0cb5fdd": anc, "tensorrt_llm_file": tensorrt_llm.__file__,
        "gate_v2_failures": {"A_env0": f"{len(fa)}/12 {fa}", "B_lpt": f"{len(fb)}/12 {fb}"},
        "B_new_failures": new_fail,
        "repeated_execution_passed": rep_ok,
        "repeat_bitwise": {a: summary["arms"][a].get("repeat_bitwise") for a, _ in ARMS},
        "calls_per_request": {a: summary["arms"][a].get("calls_per_request") for a, _ in ARMS},
        "calls_during_load_warmup_capture": {
            a: summary["arms"][a].get("calls_during_load_warmup_capture") for a, _ in ARMS},
        "check11": c11,
        "B_vs_A_min_psnr": min((r["B_vs_A_psnr"] for r in rows), default=None),
        "B_vs_A_max_lpips": max((r["B_vs_A_lpips"] for r in rows), default=None),
        "rows": [{k: r[k] for k in ("key", "A_psnr", "B_psnr", "A_lpips", "B_lpips", "B_pass")}
                 for r in rows],
    }, default=str))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    else:
        parent(sys.argv[1])
