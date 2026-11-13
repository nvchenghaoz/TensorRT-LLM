"""Fidelity of candidate MiniMax-H3 videos against the frozen BF16 references (one GPU, no server).

For each check prompt given as key=path (path absolute, or relative to the reference store):
  - LPIPS (alex, mean over every frame) and PSNR (all pixels) vs the frozen BF16 reference, plus
    audio SNR, with the frozen ab_metrics.VideoMetrics (same numbers the check adapter reports);
  - the stored BF16 reorder floor: the same metrics for the manifest's torch.compile-off variation;
  - stills at frames 25, 62, 98: reference (left) | candidate (right), saved under artifacts/stills/;
  - a same-subject verdict per still from the Qwen3-VL judge (the checkpoint's text_encoder, loaded
    as in ab_judge.Judge) on the side-by-side image, with control pairs (reference vs itself -> Yes,
    reference vs another prompt's reference -> No) to show the question discriminates.
Prints one JSON object as the last line.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()


def _bench_dir() -> Path:
    # The frozen benchmark rides a job when its files are named in the argv (--bench <file>).
    for i, a in enumerate(sys.argv):
        if a == "--bench" and i + 1 < len(sys.argv):
            p = Path(sys.argv[i + 1])
            p = p if p.is_absolute() else Path.cwd() / p
            return p.parent if p.suffix else p
    return HERE.parents[1] / "ascent-benchmark"


BENCH = _bench_dir()
sys.path.insert(0, str(BENCH))

FRAMES = (25, 62, 98)
QUESTION = (
    "The left half and the right half of this image are the same frame index from two videos "
    "generated from the same text prompt. Do both halves show the same subject: the same person or "
    "object, with the same appearance, clothing or design, in the same setting and framing? Small "
    "differences in fine detail, texture or exact pose do not matter."
)


def log(*a):
    print(*a, flush=True)


def side_by_side(a, b):
    from PIL import Image

    la, lb = Image.fromarray(a), Image.fromarray(b)
    w, h = la.size
    canvas = Image.new("RGB", (2 * w, h))
    canvas.paste(la, (0, 0))
    canvas.paste(lb, (w, 0))
    return canvas.reduce(2)


def step_times(server_log: Path, first_gated: int) -> list:
    """Per request: 'Step k/N | t s' times around the first gated (FP8) step, from a server log."""
    import re
    import statistics

    den = re.compile(r"Denoising(?: \[[^\]]*\])?: (\d+) steps")
    step = re.compile(r"Step (\d+)/(\d+) \| ([\d.]+)s")
    reqs = []
    for line in server_log.read_text(errors="replace").splitlines():
        if den.search(line):
            reqs.append({})
            continue
        m = step.search(line)
        if m and reqs:
            reqs[-1][int(m.group(1))] = float(m.group(3))
    out = []
    for i, r in enumerate(reqs):
        if not r:
            out.append({"request": i, "steps": 0})
            continue
        late = [t for k, t in r.items() if k > first_gated + 3]
        early = [t for k, t in r.items() if 2 <= k < first_gated]
        out.append({"request": i, "steps": len(r),
                    "around_gate": {k: r.get(k) for k in range(first_gated - 1, first_gated + 4)},
                    "median_bf16_steps_2plus": statistics.median(early) if early else None,
                    "median_late_fp8": statistics.median(late) if late else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", default=None, help="server log to extract Step k/N times from")
    ap.add_argument("--first-gated-step", type=int, default=15,
                    help="1-based Step number of the first FP8 forward (K+1)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--bench", default=None, help="a frozen benchmark file, e.g. "
                    "experiments/ascent-benchmark/ab_metrics.py")
    ap.add_argument("outputs", nargs="+", help="key=path")
    args = ap.parse_args()

    log(f"benchmark dir {BENCH} exists={BENCH.is_dir()} files="
        f"{sorted(p.name for p in BENCH.glob('*.py')) if BENCH.is_dir() else None}")
    from ab_common import artifact_dir, decode_video, ensure_ffmpeg, load_json
    from ab_metrics import VideoMetrics

    steps = None
    if args.server_log:
        try:
            steps = step_times(Path(args.server_log), args.first_gated_step)
            log(f"step times: {steps}")
        except Exception as exc:  # noqa: BLE001  (diagnostic only)
            steps = {"error": repr(exc)}
    manifest = load_json("reference_manifest.json")
    store = Path(manifest["store"])
    ensure_ffmpeg(str(store))
    cands = {}
    for item in args.outputs:
        key, path = item.split("=", 1)
        p = Path(path)
        cands[key] = p if p.is_absolute() else store / p
    floors = {v["key"]: store / v["path"] for v in manifest["variations"]
              if v["kind"] == "reordered_concurrency2"}

    vm = VideoMetrics()
    metrics = {}
    for key, cand in cands.items():
        ref = store / manifest["reference"][key]
        m = vm.compare_files(ref, cand)
        f = vm.compare_files(ref, floors[key]) if key in floors else {}
        metrics[key] = {
            "candidate": {k: m.get(k) for k in ("lpips", "psnr", "audio_snr_db", "bitwise_equal",
                                                 "ref_frames", "cand_frames")},
            "bf16_reorder_floor": {k: f.get(k) for k in ("lpips", "psnr", "audio_snr_db")},
            "candidate_path": str(cand),
        }
        if f.get("lpips") is not None and m.get("lpips") is not None:
            metrics[key]["lpips_minus_floor"] = m["lpips"] - f["lpips"]
        log(f"{key}: {metrics[key]}")
    del vm
    import torch

    torch.cuda.empty_cache()

    stills_dir = artifact_dir("stills")
    frames = {}
    refs = {}
    for key, cand in cands.items():
        a, _ = decode_video(store / manifest["reference"][key])
        b, _ = decode_video(cand)
        refs[key] = a
        frames[key] = (a, b)

    from ab_judge import Judge

    judge = Judge(args.checkpoint)
    verdicts = {}
    for key, (a, b) in frames.items():
        verdicts[key] = []
        for fi in FRAMES:
            img = side_by_side(a[fi], b[fi])
            name = f"{key}_f{fi:03d}_ref_vs_{args.label}.jpg"
            img.save(stills_dir / name, quality=90)
            ans, margin = judge.ask([img], QUESTION)
            verdicts[key].append({"frame": fi, "same_subject": ans, "margin": round(margin, 3),
                                  "still": f"stills/{name}"})
            log(f"judge {key} frame {fi}: {ans} ({margin:+.2f})")
    controls = []
    keys = list(refs)
    for i, key in enumerate(keys):
        a = refs[key]
        ans, margin = judge.ask([side_by_side(a[62], a[62])], QUESTION)
        controls.append({"pair": f"{key} ref vs itself", "expected": "yes", "answer": ans,
                         "margin": round(margin, 3)})
        other = keys[(i + 1) % len(keys)]
        if other != key:
            ans, margin = judge.ask([side_by_side(a[62], refs[other][62])], QUESTION)
            controls.append({"pair": f"{key} ref vs {other} ref", "expected": "no", "answer": ans,
                             "margin": round(margin, 3)})
    judge.close()
    for c in controls:
        log(f"control {c}")
    same = {k: all(v["same_subject"] == "yes" for v in vs) for k, vs in verdicts.items()}
    controls_ok = all(c["answer"] == c["expected"] for c in controls)
    out = {"passed": controls_ok, "label": args.label, "metrics": metrics, "same_subject": same,
           "server_step_times": steps,
           "verdicts": verdicts, "judge_controls": controls, "judge_controls_ok": controls_ok,
           "frames": list(FRAMES), "question": QUESTION,
           "compared": "candidate vs frozen BF16 reference (LPIPS alex all frames, PSNR, audio SNR); "
                       "stored compile-off BF16 variation vs reference as the floor; Qwen3-VL "
                       "same-subject verdict on side-by-side stills"}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
