"""Quality job for the MiniMax-H3 Cache-DiT branch: frozen check adapter + judge.

1. Runs the frozen check adapter (experiments/ascent-benchmark/run_check.py) with a harness
   built from the protocol's server settings (tp_size 2, ulysses 1, text-encoder TP 2,
   parallel VAE 2, VANILLA attention, torch.compile + autotune, no CUDA graphs, warmup at
   768x1344x124). The adapter overlays this tree's experiments/ascent-config/engine.json, so
   the branch's cache_config reaches the server (quoted from its visual_gen_args.yaml).
   Gives per-prompt LPIPS/PSNR/audio SNR vs the frozen BF16 references, the variation floor,
   and the same-server park_portrait repeat.
2. After the server stops, loads the package's Qwen3-VL judge and answers
   (a) the 21 accuracy items (prompts.json "accuracy") on the check's first-pass outputs;
   (b) a same-subject verdict per prompt at frames 25, 62 and 98 on side-by-side pairs
       (frozen reference, candidate). The stills are saved as artifacts.

Usage: python h3_quality_job.py <checkpoint>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "ascent-benchmark"
FRAMES = (25, 62, 98)
SAME_Q = ("The first image is a frame from a reference video and the second image is the same "
          "frame index from another generation of the same prompt. Do the two images show the same "
          "subject (the same person or object, with the same clothing, colors and details) in the "
          "same framing and composition?")


def main() -> None:
    ckpt = sys.argv[1]
    manifest = json.loads((BENCH / "reference_manifest.json").read_text())
    harness = {
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
        "reference_store": manifest["store"],
    }
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    print(f"[quality] running check adapter, engine.json = "
          f"{(HERE.parent / 'ascent-config' / 'engine.json').read_text()}", flush=True)
    proc = subprocess.Popen([sys.executable, str(BENCH / "run_check.py"), json.dumps(harness)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in proc.stdout:
        if line.startswith("{") and '"adapter"' in line:
            last = line
            sys.stdout.write(line[:3000] + "\n")
        else:
            sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    check = json.loads(last) if last else {"valid": False, "errors": ["no check result"]}
    out = {"check_valid": check.get("valid"), "check_errors": check.get("errors"),
           "engine_overlay": check.get("engine_overlay", {}).get("content"),
           "output_class": check.get("output", {}).get("class"),
           "repeated_execution_passed": check.get("repeated_execution_passed"),
           "tolerance": check.get("tolerance")}
    yaml_path = art / "logs" / "server-check.visual_gen_args.yaml"
    if yaml_path.is_file():
        import yaml

        out["server_cache_config"] = yaml.safe_load(yaml_path.read_text()).get("cache_config")
    slog = art / "logs" / "server-check.log"
    if slog.is_file():
        out["cache_dit_decision_lines"] = [ln.strip()[-260:] for ln in slog.read_text(errors="replace")
                                           .splitlines() if "Cache-DiT decisions" in ln][:12]
    out["variation_floor"] = [{"key": p.get("key"), "kind": p.get("kind"), "lpips": p.get("lpips"),
                               "psnr": p.get("psnr"), "audio_snr_db": p.get("audio_snr_db")}
                              for p in check.get("variation_pairs", []) if p.get("kind") != "restart"]
    per_prompt = check.get("comparisons", {}).get("1", {}).get("per_prompt", [])
    out["per_prompt"] = [{k: p.get(k) for k in ("key", "lpips", "psnr", "audio_snr_db", "bitwise_equal",
                                                 "passed")} for p in per_prompt]
    out["repeat"] = [{"key": r.get("key"), "passed": r.get("passed"),
                      "vs_first_pass": {k: r.get("vs_first_pass", {}).get(k)
                                        for k in ("bitwise_equal", "lpips", "psnr", "audio_snr_db")}}
                     for r in check.get("repeated_execution", [])]
    reqs = check.get("requests", [])
    seen = {}
    for r in reqs:
        seen.setdefault(r.get("key"), r.get("path"))
    out["request_s"] = [(r.get("key"), r.get("client_e2e_s"), r.get("server_denoise_s")) for r in reqs]
    print("[quality] check summary " + json.dumps(out, default=str), flush=True)

    # Judge: accuracy items and same-subject verdicts.
    sys.path.insert(0, str(BENCH))
    from ab_common import decode_video
    from ab_judge import Judge, sample_frames
    from PIL import Image

    prompts = json.loads((BENCH / "prompts.json").read_text())
    items = prompts["accuracy"]
    judge = Judge(ckpt)
    answers, correct, total = [], 0, 0
    for key, questions in items.items():
        path = seen.get(key)
        frames = sample_frames(Path(path))[0] if path else None
        for question, expected in questions:
            total += 1
            if frames is None:
                answers.append({"key": key, "q": question, "ok": False, "error": "no output"})
                continue
            ans, margin = judge.ask(frames, question)
            correct += int(ans == expected)
            answers.append({"key": key, "q": question, "expected": expected, "answer": ans,
                            "margin": round(margin, 2), "ok": ans == expected})
    out["accuracy"] = {"correct": correct, "total": total,
                       "wrong": [a for a in answers if not a.get("ok")]}
    stills = art / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    store = Path(manifest["store"])
    same = []
    for key in manifest["check_keys"]:
        path = seen.get(key)
        if not path:
            same.append({"key": key, "error": "no output"})
            continue
        ref, _ = decode_video(store / manifest["reference"][key])
        cand, _ = decode_video(Path(path))
        for f in FRAMES:
            a, b = Image.fromarray(ref[f]), Image.fromarray(cand[f])
            pair = Image.new("RGB", (a.width * 2, a.height))
            pair.paste(a, (0, 0))
            pair.paste(b, (a.width, 0))
            pair.reduce(2).save(stills / f"{key}_f{f}_ref_left_cand_right.jpg", quality=88)
            ans, margin = judge.ask([a, b], SAME_Q)
            same.append({"key": key, "frame": f, "same_subject": ans, "margin": round(margin, 2)})
    judge.close()
    out["same_subject"] = same
    out["passed"] = bool(check.get("valid") and check.get("repeated_execution_passed"))
    print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
