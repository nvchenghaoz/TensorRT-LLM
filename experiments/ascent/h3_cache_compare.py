"""Compare outputs saved by h3_cache_check.py and parse per-rank Cache-DiT decisions.

  python h3_cache_compare.py --dir /tmp/h3 --logdir $AUTO_DEV_ARTIFACTS --tags base c1 c2

For every tag: outputs <tag>_<i>_seed<s>.pt. Checks
  - repeat: the last output of a repeated seed equals the first output of that seed
    (bitwise required -> passed),
  - vs base: first output of each tag vs base_0 (same seed) -> PSNR / audio SNR,
  - per-rank decisions identical in <tag>_arm.log (required -> passed),
  - cached vs full step transformer time from the rank-0 step log.
Last stdout line: {"passed": ..., ...}
"""

import argparse
import ast
import glob
import json
import math
import os
import re
from collections import defaultdict

import torch


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 * 255.0 / mse)


def snr(ref: torch.Tensor, x: torch.Tensor) -> float:
    ref = ref.float()
    noise = (ref - x.float()).pow(2).sum().item()
    return float("inf") if noise == 0 else 10 * math.log10(ref.pow(2).sum().item() / noise)


def compare(a: dict, b: dict) -> dict:
    va, vb = a["video"], b["video"]
    per_frame = [psnr(va[:, i], vb[:, i]) for i in range(va.shape[1])]
    r = dict(
        video_bitwise=bool(torch.equal(va, vb)),
        video_psnr=round(psnr(va, vb), 3),
        video_psnr_min_frame=round(min(per_frame), 3),
        video_max_abs=int((va.int() - vb.int()).abs().max().item()),
    )
    if a["audio"] is not None and b["audio"] is not None:
        r["audio_bitwise"] = bool(torch.equal(a["audio"], b["audio"]))
        r["audio_snr_db"] = round(snr(b["audio"], a["audio"]), 3)
    r["seconds"] = [round(a["seconds"], 2), round(b["seconds"], 2)]
    return r


def parse_log(path: str):
    per_rank = defaultdict(list)
    step_times, cur = [], []
    dec_re = re.compile(
        r"Cache-DiT decisions rank=(\d+): context=(\S+) steps=(\d+) cached=(\d+) "
        r"cached_steps=(\[[^\]]*\])"
    )
    step_re = re.compile(r"Step (\d+)/(\d+) \| ([0-9.]+)s \(trans=([0-9.]+)s")
    with open(path, errors="replace") as f:
        for line in f:
            m = dec_re.search(line)
            if m:
                per_rank[int(m.group(1))].append(
                    dict(steps=int(m.group(3)), cached=ast.literal_eval(m.group(5)))
                )
            m = step_re.search(line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                cur.append(float(m.group(4)))
                if i == n:
                    step_times.append(cur)
                    cur = []
    return per_rank, step_times


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--logdir", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument(
        "--cross",
        nargs="*",
        default=[],
        help="a:b -> output name a (e.g. alt_2_seed42) vs output name b; must be bitwise",
    )
    args = p.parse_args()

    med = lambda xs: round(sorted(xs)[len(xs) // 2], 3) if xs else None
    passed = True
    report = {}
    outputs = {}
    for tag in args.tags:
        files = sorted(
            glob.glob(os.path.join(args.dir, f"{tag}_*_seed*.pt")),
            key=lambda s: int(os.path.basename(s).split("_")[-2]),
        )
        outputs[tag] = [(os.path.basename(f)[:-3], torch.load(f)) for f in files]
    base = outputs.get("base", [])
    for tag in args.tags:
        r = {"requests": [(n, round(o["seconds"], 2)) for n, o in outputs[tag]]}
        first_by_seed = {}
        for name, o in outputs[tag]:
            seed = name.split("seed")[-1]
            if seed in first_by_seed:
                c = compare(first_by_seed[seed][1], o)
                r[f"repeat {first_by_seed[seed][0]} vs {name}"] = c
                if not c["video_bitwise"] or not c.get("audio_bitwise", True):
                    passed = False
            else:
                first_by_seed[seed] = (name, o)
        if tag != "base":
            for bname, bo in base:
                bseed = bname.split("seed")[-1]
                if bseed in first_by_seed:
                    r[f"vs {bname}"] = compare(first_by_seed[bseed][1], bo)
        log = os.path.join(args.logdir, f"{tag}_arm.log")
        if os.path.exists(log):
            per_rank, step_times = parse_log(log)
            ranks = sorted(per_rank)
            same = len(ranks) < 2 or all(per_rank[x] == per_rank[ranks[0]] for x in ranks[1:])
            if not same:
                passed = False
            r["ranks"] = ranks
            r["ranks_identical"] = same
            r["decisions_rank0"] = [
                (d["steps"], len(d["cached"]), d["cached"]) for d in per_rank.get(ranks[0], [])
            ] if ranks else []
            full_reqs = [d for d in per_rank.get(0, []) if d["steps"] >= 8]
            timings = []
            for k, times in enumerate([t for t in step_times if len(t) >= 8]):
                cached = set(full_reqs[k]["cached"]) if k < len(full_reqs) else set()
                c = [t for j, t in enumerate(times) if j in cached]
                full = [t for j, t in enumerate(times) if j not in cached and j > 0]
                timings.append(
                    dict(
                        n_cached=len(c),
                        cached_step_median_s=med(c),
                        full_step_median_s=med(full),
                        total_trans_s=round(sum(times), 2),
                    )
                )
            r["step_timings"] = timings
        report[tag] = r
        print(tag, json.dumps(r), flush=True)
    named = {n: o for tag in args.tags for n, o in outputs[tag]}
    for pair in args.cross:
        a, b = pair.split(":")
        c = compare(named[a], named[b])
        report[f"cross {a} vs {b}"] = c
        print("cross", a, b, json.dumps(c), flush=True)
        if not c["video_bitwise"] or not c.get("audio_bitwise", True):
            passed = False
    print(json.dumps({"passed": passed, "report": report}))


if __name__ == "__main__":
    main()
