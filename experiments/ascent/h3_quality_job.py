"""Quality job for the MiniMax-H3 Cache-DiT branch: frozen check adapter + judge, per config.

For each named config (CONFIGS below, chosen on the command line):
1. Writes the config into this job's experiments/ascent-config/engine.json as
   {"visual_gen_args": {"cache_config": ...}} and runs the frozen check adapter
   (experiments/ascent-benchmark/run_check.py) with a harness built from the protocol's server
   settings (tp_size 2, ulysses 1, text-encoder TP 2, parallel VAE 2, VANILLA attention,
   torch.compile + autotune, no CUDA graphs, warmup at 768x1344x124). Per-prompt LPIPS/PSNR/audio
   SNR vs the frozen BF16 references, the variation floor, the same-server park_portrait repeat,
   cache_config quoted from the server's visual_gen_args.yaml, and both ranks' Cache-DiT decision
   lines. One retry if the server dies at init (rendezvous port race with other jobs).
2. After all servers stop, the package's Qwen3-VL judge answers the 21 accuracy items on each
   config's first-pass outputs and gives a same-subject verdict per prompt at frames 25, 62 and 98
   on (frozen reference, candidate) pairs. Stills are saved under <artifacts>/<tag>/stills.
The committed engine.json is restored at the end.

Usage: python h3_quality_job.py <checkpoint> <config> [<config> ...]
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "ascent-benchmark"
ENGINE = HERE.parent / "ascent-config" / "engine.json"
FRAMES = (25, 62, 98)
SAME_Q = ("The first image is a frame from a reference video and the second image is the same "
          "frame index from another generation of the same prompt. Do the two images show the same "
          "subject (the same person or object, with the same clothing, colors and details) in the "
          "same framing and composition?")

_BASE = {"cache_backend": "cache_dit", "Fn_compute_blocks": 8, "Bn_compute_blocks": 0,
         "residual_diff_threshold": 0.08, "scm_steps_policy": "static"}
CONFIGS = {
    # 14 dense head steps, alternate compute/cache over steps 14..43, 5 dense tail: 15 of 49 cached.
    "w14_alt": dict(_BASE, max_warmup_steps=14, max_continuous_cached_steps=1,
                    scm_compute_bins=[14] + [1] * 14 + [6], scm_cache_bins=[1] * 15),
    # 14 dense head steps, cache 2 / compute 1 over steps 14..43, 5 dense tail: 20 of 49 cached.
    "w14_c2": dict(_BASE, max_warmup_steps=14, max_continuous_cached_steps=2,
                   scm_compute_bins=[14] + [1] * 9 + [6], scm_cache_bins=[2] * 10),
    # Earlier config (jobs 55/74/153-154): 10 dense head, alternate, 6 dense tail: 17 cached.
    "w10_alt": dict(_BASE, max_warmup_steps=10, max_continuous_cached_steps=1,
                    scm_compute_bins=[10] + [1] * 16 + [6], scm_cache_bins=[1] * 17),
}


def harness(ckpt: str, store: str) -> dict:
    return {
        "server": {
            "visual_gen_args": {
                "parallel_config": {"tp_size": 2, "ulysses_size": 1, "text_encoder_tp_size": 2,
                                    "parallel_vae_size": 2},
                "attention_config": {"backend": "VANILLA"},
                "torch_compile_config": {"enable": True, "enable_autotune": True},
                "cuda_graph_config": {"enable": False},
                "compilation_config": {"resolutions": [[768, 1344]], "num_frames": [124]},
            },
            "env": {},
        },
        "workload_identity": {
            "workload": {"width": 1344, "height": 768, "num_frames": 124, "num_inference_steps": 50,
                         "kind": "visual_gen", "runtime": "trtllm-serve", "task": "text-to-video",
                         "gpus_per_instance": 2},
            "resolved_model": {"id": "MiniMaxAI/MiniMax-H3"},
        },
        "checkpoint": ckpt,
        "model_revision": "local",
        "reference_store": store,
    }


def run_check(h: dict, art: Path) -> dict:
    env = dict(os.environ, AUTO_DEV_ARTIFACTS=str(art))
    proc = subprocess.Popen([sys.executable, str(BENCH / "run_check.py"), json.dumps(h)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    last = None
    for line in proc.stdout:
        if line.startswith("{") and '"adapter"' in line:
            last = line
            sys.stdout.write(line[:2000] + "\n")
        else:
            sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    return json.loads(last) if last else {"valid": False, "errors": ["no check result"]}


def decision_lines(path: Path) -> list:
    if not path.is_file():
        return []
    import re

    out = []
    for ln in path.read_text(errors="replace").splitlines():
        # The context name embeds a per-process object id; compare only the decisions.
        m = re.search(r"Cache-DiT decisions rank=\d+: context=\S+ (steps=\d+ cached=\d+ "
                      r"cached_steps=\[[^\]]*\])", ln)
        if m:
            out.append(m.group(1))
    return out


def summarize(tag: str, check: dict, art: Path) -> dict:
    out = {"tag": tag, "check_valid": check.get("valid"), "check_errors": check.get("errors"),
           "output_class": check.get("output", {}).get("class"),
           "repeated_execution_passed": check.get("repeated_execution_passed"),
           "tolerance": check.get("tolerance")}
    yaml_path = art / "logs" / "server-check.visual_gen_args.yaml"
    if yaml_path.is_file():
        import yaml

        out["server_cache_config"] = yaml.safe_load(yaml_path.read_text()).get("cache_config")
    r0 = decision_lines(art / "logs" / "server-check.log")
    r1 = decision_lines(art / "logs" / "server-check-rank1.log")
    out["decisions_rank0"] = r0
    out["decisions_rank1"] = r1
    out["ranks_identical"] = bool(r0) and r0 == r1
    floor = {}
    for p in check.get("variation_pairs", []):
        if p.get("lpips") is not None and p.get("key"):
            floor[p["key"]] = max(floor.get(p["key"], 0.0), p["lpips"])
    out["floor_lpips"] = floor
    per_prompt = check.get("comparisons", {}).get("1", {}).get("per_prompt", [])
    rows = []
    for p in per_prompt:
        row = {k: p.get(k) for k in ("key", "lpips", "psnr", "audio_snr_db", "bitwise_equal", "passed")}
        f = floor.get(p.get("key"))
        if f is not None and p.get("lpips") is not None:
            row["floor"] = f
            row["margin_to_floor_plus_0.03"] = round(f + 0.03 - p["lpips"], 4)
        rows.append(row)
    out["per_prompt"] = rows
    out["repeat"] = [{"key": r.get("key"), "passed": r.get("passed"),
                      "vs_first_pass": {k: r.get("vs_first_pass", {}).get(k)
                                        for k in ("bitwise_equal", "lpips", "psnr", "audio_snr_db")}}
                     for r in check.get("repeated_execution", [])]
    out["requests"] = [{"key": r.get("key"), "client_e2e_s": r.get("client_e2e_s"),
                        "server_denoise_s": r.get("server_denoise_s"), "path": r.get("path")}
                       for r in check.get("requests", [])]
    return out


def main() -> None:
    ckpt, tags = sys.argv[1], sys.argv[2:]
    manifest = json.loads((BENCH / "reference_manifest.json").read_text())
    store = Path(manifest["store"])
    root = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    original = ENGINE.read_text() if ENGINE.is_file() else None
    h = harness(ckpt, str(store))
    results = {}
    try:
        for tag in tags:
            if tag != "engine":  # "engine": run with the committed engine.json as is
                # Replace cache_config only; keep main's other keys (attention, step_precision).
                eng = json.loads(original) if original else {}
                eng.setdefault("visual_gen_args", {})["cache_config"] = CONFIGS[tag]
                ENGINE.write_text(json.dumps(eng))
            art = root / tag
            print(f"[quality] === {tag}: {ENGINE.read_text()}", flush=True)
            check = run_check(h, art)
            errs = " ".join(map(str, check.get("errors") or []))
            if not check.get("valid") and ("EADDRINUSE" in errs or "died during initialization" in errs):
                print(f"[quality] {tag}: server init failed (port race); retrying once", flush=True)
                check = run_check(h, root / f"{tag}_retry")
                art = root / f"{tag}_retry"
            results[tag] = summarize(tag, check, art)
            print(f"[quality] {tag} check summary " + json.dumps(results[tag], default=str), flush=True)
    finally:
        if original is not None:
            ENGINE.write_text(original)

    sys.path.insert(0, str(BENCH))
    from ab_common import decode_video, ensure_ffmpeg
    from ab_judge import Judge, sample_frames
    from PIL import Image

    ensure_ffmpeg(str(store))  # the parent process has no ffmpeg on PATH (job 210)
    prompts = json.loads((BENCH / "prompts.json").read_text())
    judge = Judge(ckpt)
    refs = {k: decode_video(store / manifest["reference"][k])[0] for k in manifest["check_keys"]}
    for tag, res in results.items():
        first = {}
        for r in res["requests"]:
            first.setdefault(r["key"], r["path"])
        correct = total = 0
        wrong = []
        for key, questions in prompts["accuracy"].items():
            path = first.get(key)
            frames = sample_frames(Path(path))[0] if path else None
            for question, expected in questions:
                total += 1
                if frames is None:
                    wrong.append({"key": key, "q": question, "error": "no output"})
                    continue
                ans, margin = judge.ask(frames, question)
                if ans == expected:
                    correct += 1
                else:
                    wrong.append({"key": key, "q": question, "expected": expected, "answer": ans,
                                  "margin": round(margin, 2)})
        res["accuracy"] = {"correct": correct, "total": total, "wrong": wrong}
        stills = root / tag / "stills"
        stills.mkdir(parents=True, exist_ok=True)
        same = []
        for key in manifest["check_keys"]:
            path = first.get(key)
            if not path:
                same.append({"key": key, "error": "no output"})
                continue
            cand = decode_video(Path(path))[0]
            for f in FRAMES:
                a, b = Image.fromarray(refs[key][f]), Image.fromarray(cand[f])
                pair = Image.new("RGB", (a.width * 2, a.height))
                pair.paste(a, (0, 0))
                pair.paste(b, (a.width, 0))
                pair.reduce(2).save(stills / f"{key}_f{f}_ref_left_cand_right.jpg", quality=88)
                ans, margin = judge.ask([a, b], SAME_Q)
                same.append({"key": key, "frame": f, "same_subject": ans, "margin": round(margin, 2)})
        res["same_subject"] = same
        res["fidelity_pass"] = bool(res["per_prompt"]) and all(
            r.get("margin_to_floor_plus_0.03", -1) >= 0 for r in res["per_prompt"]) and all(
            s.get("same_subject") == "yes" for s in same)
        for r in res["requests"]:
            r.pop("path", None)
        print(f"[quality] {tag} judged " + json.dumps(res, default=str), flush=True)
    judge.close()
    passed = all(r.get("check_valid") and r.get("repeated_execution_passed") and r.get("ranks_identical")
                 and r.get("fidelity_pass") for r in results.values())
    print(json.dumps({"passed": bool(results) and passed, "configs": results}, default=str), flush=True)


if __name__ == "__main__":
    main()
