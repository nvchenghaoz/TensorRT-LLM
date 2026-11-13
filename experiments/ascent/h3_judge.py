"""Judge already-generated check outputs (1 GPU): 21 accuracy items + same-subject at frames 25/62/98.

Usage: python h3_judge.py <checkpoint> <tag>=<pass1 media dir> [...]
The media dir holds the check adapter's pass-1 outputs 0000_0.mp4 (park_portrait), 0001_0.mp4
(perfume_bottle), 0002_0.mp4 (golden_robot), in reference_manifest.json check_keys order.
Side-by-side stills (frozen reference left, candidate right) go to <artifacts>/<tag>/stills.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "ascent-benchmark"
sys.path.insert(0, str(BENCH))
FRAMES = (25, 62, 98)
SAME_Q = ("The first image is a frame from a reference video and the second image is the same "
          "frame index from another generation of the same prompt. Do the two images show the same "
          "subject (the same person or object, with the same clothing, colors and details) in the "
          "same framing and composition?")


def main() -> None:
    from ab_common import decode_video, ensure_ffmpeg
    from ab_judge import Judge, sample_frames
    from PIL import Image

    ckpt = sys.argv[1]
    runs = dict(a.split("=", 1) for a in sys.argv[2:])
    manifest = json.loads((BENCH / "reference_manifest.json").read_text())
    prompts = json.loads((BENCH / "prompts.json").read_text())
    store = Path(manifest["store"])
    ensure_ffmpeg(str(store))
    root = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    keys = manifest["check_keys"]
    refs = {k: decode_video(store / manifest["reference"][k])[0] for k in keys}
    judge = Judge(ckpt)
    out = {}
    for tag, media in runs.items():
        paths = {k: Path(media) / f"{i:04d}_0.mp4" for i, k in enumerate(keys)}
        res = {"media": media}
        correct = total = 0
        answers = []
        for key, questions in prompts["accuracy"].items():
            p = paths.get(key)
            frames = sample_frames(p)[0] if p and p.is_file() else None
            for question, expected in questions:
                total += 1
                if frames is None:
                    answers.append({"key": key, "q": question, "error": "no output", "ok": False})
                    continue
                ans, margin = judge.ask(frames, question)
                correct += int(ans == expected)
                answers.append({"key": key, "q": question, "expected": expected, "answer": ans,
                                "margin": round(margin, 2), "ok": ans == expected})
        res["accuracy"] = {"correct": correct, "total": total,
                           "wrong": [a for a in answers if not a["ok"]]}
        res["answers"] = answers
        stills = root / tag / "stills"
        stills.mkdir(parents=True, exist_ok=True)
        same = []
        for key in keys:
            cand = decode_video(paths[key])[0]
            for f in FRAMES:
                a, b = Image.fromarray(refs[key][f]), Image.fromarray(cand[f])
                pair = Image.new("RGB", (a.width * 2, a.height))
                pair.paste(a, (0, 0))
                pair.paste(b, (a.width, 0))
                pair.reduce(2).save(stills / f"{key}_f{f}_ref_left_cand_right.jpg", quality=88)
                ans, margin = judge.ask([a, b], SAME_Q)
                same.append({"key": key, "frame": f, "same_subject": ans, "margin": round(margin, 2)})
        # Control: the judge on the reference against itself (should be yes).
        a = Image.fromarray(refs[keys[0]][62])
        res["same_subject_control_ref_vs_ref"] = judge.ask([a, a], SAME_Q)
        res["same_subject"] = same
        res["all_same_subject"] = all(s["same_subject"] == "yes" for s in same)
        out[tag] = res
        print(f"[judge] {tag}: accuracy {correct}/{total}, same_subject "
              f"{sum(s['same_subject'] == 'yes' for s in same)}/{len(same)}", flush=True)
    judge.close()
    print(json.dumps({"passed": all(r["all_same_subject"] for r in out.values()), "runs": out}), flush=True)


if __name__ == "__main__":
    main()
