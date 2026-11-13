"""U2 (one encode of a fixed decoded video) for idea 133 / branch 68.

Inputs: the 3 frozen reference MP4s decoded to uint8 [T, H, W, 3] (copied to the GPU,
contiguous, i.e. main's _video_to_uint8 layout) plus their audio as fp32 (N, 2).

Modes
  pair   : time this tree's tensorrt_llm.media.encoding.save_video (1 warm + --reps timed per
           prompt). Last line: evidence JSON (step_ms = mean of per-prompt medians).
  full   : J1 suite in one process: inline copy of main b2f9c52d's _FfmpegCliEncoder.encode_video
           ("main") vs this tree's save_video ("branch"), interleaved main/branch/main2 (A/A);
           byte-identity + SEI; 5 back-to-back reuse; two-thread park+robot x5; avi; CPU tensor;
           fresh-process first save (main / branch / branch+reservation). Last line: {"passed": ...}.
  first  : (internal) fresh-process first save of park, used by full.
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve()

# From the frozen experiments/ascent-benchmark/reference_manifest.json (protocol v2).
STORE = "/home/scratch.chengzhang/trtllm-dev-minimax-h3-2g-20261112-143634/ascent-benchmark-data/v1"
MANIFEST = {"reference": {
    "park_portrait": "archive/6ad9624f725b93e6d002b78f736db19630ed63f24f12fdd34b89ebb259ced714.mp4",
    "perfume_bottle": "archive/d77cbdfb9fe60654533772f322870d434e99056890d2611ffe4abb7c209f0fd7.mp4",
    "golden_robot": "archive/0448c5bdd801f124025614e3aadab4c6b3bbda1be819d76287291b466deeb1fb.mp4",
}}
FFMPEG_SHA = "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
KEYS = ["park_portrait", "perfume_bottle", "golden_robot"]
FPS = 24.0
WORK = Path(os.environ.get("TMPDIR") or "/tmp") / "pd68"
WORK.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[u2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _AB:
    """The two ab_common helpers this script needs (same ffmpeg binary the adapters use)."""

    @staticmethod
    def ensure_ffmpeg(store):
        import shutil

        if not shutil.which("ffmpeg"):
            bindir = WORK / "bin"
            bindir.mkdir(exist_ok=True)
            target = bindir / "ffmpeg"
            if not target.is_file():
                shutil.copyfile(Path(store) / "tools" / "ffmpeg", target)
                target.chmod(0o755)
            os.environ["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
        path = shutil.which("ffmpeg")
        digest = sha(path)
        log(f"ffmpeg {path} sha256 {digest[:16]} (frozen {FFMPEG_SHA[:16]}, match={digest == FFMPEG_SHA})")
        return path

    @staticmethod
    def decode_video(path):
        import numpy as np

        err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
        m = re.search(r"Video: \w+.*?, (\d{2,5})x(\d{2,5})", err)
        w, h = int(m.group(1)), int(m.group(2))
        a = re.search(r"Audio: \w+.*?, (\d+) Hz", err)
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-an", "-f",
                               "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, h, w, 3)
        return frames, {"audio_rate": int(a.group(1)) if a else None}


ab_common = _AB


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sei(path):
    data = Path(path).read_bytes()
    m = re.search(rb"x264 - core[^\x00]*", data)
    if not m:
        return None
    s = m.group(0).decode("latin-1")
    t = re.search(r"threads=(\d+)", s)
    return {"threads": t.group(1) if t else None, "len": len(s), "sha": hashlib.sha256(s.encode()).hexdigest()[:16]}


def spread(xs):
    xs = sorted(xs)
    n = len(xs)
    med = statistics.median(xs)
    p10 = xs[max(0, int(0.1 * (n - 1)))]
    p90 = xs[min(n - 1, int(round(0.9 * (n - 1))))]
    return {"median_ms": round(med, 1), "p10_ms": round(p10, 1), "p90_ms": round(p90, 1),
            "min_ms": round(xs[0], 1), "max_ms": round(xs[-1], 1),
            "noise_floor": round((p90 - p10) / med, 4), "n": n}


def gpu_info():
    import torch

    info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "affinity": sorted(os.sched_getaffinity(0))}
    aff = info["affinity"]
    info["affinity_summary"] = f"{len(aff)} cpus {aff[0]}-{aff[-1]}" if aff else "none"
    try:
        p = torch.cuda.get_device_properties(0)
        bus = f"{p.pci_bus_id:02x}:{p.pci_device_id:02x}".lower()
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,pcie.link.width.current",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            idx, pci, width = [x.strip() for x in line.split(",")]
            if pci.lower().endswith(bus + ".0"):
                info["physical_gpu"] = int(idx)
                info["pcie_width"] = width
    except Exception as exc:  # noqa: BLE001
        info["gpu_err"] = repr(exc)
    return info


def load_inputs(keys=KEYS):
    """Decode each reference once to WORK/<key>.npy (+ audio); return {key: (video_np, audio_np, sr)}."""
    import numpy as np

    out = {}
    for key in keys:
        vnpy, anpy, meta_p = WORK / f"{key}.v.npy", WORK / f"{key}.a.npy", WORK / f"{key}.json"
        if not (vnpy.is_file() and anpy.is_file() and meta_p.is_file()):
            src = Path(STORE) / MANIFEST["reference"][key]
            frames, meta = ab_common.decode_video(src)
            sr = int(meta.get("audio_rate") or 24000)
            proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn",
                                   "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"], capture_output=True)
            audio = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2)
            np.save(vnpy, np.ascontiguousarray(frames))
            np.save(anpy, np.ascontiguousarray(audio))
            meta_p.write_text(json.dumps({"sr": sr}))
        out[key] = (np.load(vnpy), np.load(anpy), json.loads(meta_p.read_text())["sr"])
        log(f"input {key}: video {out[key][0].shape} audio {out[key][1].shape} sr {out[key][2]}")
    return out


def to_gpu(inputs):
    import torch

    g = {}
    for key, (v, a, sr) in inputs.items():
        vt = torch.from_numpy(v).cuda().contiguous()
        at = torch.from_numpy(a).cuda().float()
        g[key] = (vt, at, sr)
    torch.cuda.synchronize()
    return g


# ----------------------------------------------------------------- main b2f9c52d's encode, verbatim

def main_encode_video(video, output_path, frame_rate, audio=None, audio_sample_rate=24000):
    """Copy of main b2f9c52d _FfmpegCliEncoder.encode_video (pageable .cpu(), communicate())."""
    import torch
    from tensorrt_llm.media import encoding as E

    enc = E._FfmpegCliEncoder
    enc._validate_video_tensor(video)
    video_np = video.cpu().numpy()
    num_frames, height, width, channels = video_np.shape
    if channels != 3:
        raise ValueError("Expected 3-channel RGB video")
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_input_args, audio_output_args = [], []
        if audio is not None:
            audio_tensor = enc._validate_audio_tensor(audio)
            required_samples = math.ceil(num_frames * audio_sample_rate / frame_rate)
            missing_samples = required_samples - audio_tensor.shape[0]
            if missing_samples > 0:
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 0, 0, missing_samples))
            audio_np = audio_tensor.cpu().numpy()
            tmp_audio_path = os.path.join(tmp_dir, "audio.wav")
            enc._write_wav(None, tmp_audio_path, audio_np, sample_rate=audio_sample_rate)
            audio_input_args = ["-i", tmp_audio_path]
            audio_output_args = ["-c:a", "aac", "-shortest"]
        cmd = [E._get_ffmpeg_path(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(frame_rate), "-i", "-", *audio_input_args, "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", E._x264_preset(), "-crf", "23", *audio_output_args, output_path]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_frames = memoryview(video_np.reshape(-1))
        _, stderr = process.communicate(input=raw_frames)
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode()}")
    return output_path


def branch_save(video, path, audio, sr):
    from tensorrt_llm.media import encoding as E

    return E.save_video(video, path, audio=audio, frame_rate=FPS, audio_sample_rate=sr)


def main_save(video, path, audio, sr):
    return main_encode_video(video, path, FPS, audio, sr)


def timed(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t0) * 1e3


# ----------------------------------------------------------------- modes

def mode_pair(args):
    import torch

    info = gpu_info()
    log(f"gpu {info}")
    ab_common.ensure_ffmpeg(STORE)
    g = to_gpu(load_inputs())
    per = {}
    for key in KEYS:
        v, a, sr = g[key]
        path = str(WORK / f"pair_{key}.mp4")
        branch_save(v, path, a, sr)  # warm
        ts = []
        for i in range(args.reps):
            torch.cuda.synchronize()
            ts.append(timed(branch_save, v, path, a, sr))
        per[key] = dict(spread(ts), sha256=sha(path), sei=sei(path))
        log(f"{key}: {per[key]}")
    step = statistics.fmean(per[k]["median_ms"] for k in KEYS)
    noise = max(per[k]["noise_floor"] for k in KEYS)
    print(json.dumps({"batch": {"1": {"step_ms": round(step, 2), "noise_floor": noise}},
                      "output": {"class": "exact"}, "per_prompt": per, "gpu": info}), flush=True)


def mode_first(args):
    """Fresh process: optional startup reservation, then the first full-size save of park."""
    import torch

    ab_common.ensure_ffmpeg(STORE)
    from tensorrt_llm.media import encoding as E

    v, a, sr = to_gpu(load_inputs(["park_portrait"]))["park_portrait"]
    res_ms = None
    if args.reserve:
        res_ms = E.reserve_pinned_video_bytes(v.numel() * v.element_size())
    path = str(WORK / f"first_{args.arm}_{int(bool(args.reserve))}.mp4")
    fn = main_save if args.arm == "main" else branch_save
    torch.cuda.synchronize()
    first = timed(fn, v, path, a, sr)
    second = timed(fn, v, path, a, sr)
    print(json.dumps({"arm": args.arm, "reserve": bool(args.reserve), "reserve_ms": res_ms,
                      "first_ms": round(first, 1), "second_ms": round(second, 1), "sha256": sha(path)}),
          flush=True)


def run_first(arm, reserve):
    cmd = [sys.executable, str(HERE), "--mode", "first", "--arm", arm] + (["--reserve"] if reserve else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        return {"arm": arm, "reserve": reserve, "error": (p.stderr or p.stdout)[-1500:]}
    return json.loads(lines[-1])


def mode_full(args):
    import torch
    from tensorrt_llm.media import encoding as E

    info = gpu_info()
    log(f"gpu {info}")
    ab_common.ensure_ffmpeg(STORE)
    inputs = load_inputs()
    g = to_gpu(inputs)
    errors = []
    res = {"gpu": info}

    # --- exactness: main vs branch per prompt, SEI threads=
    ref = {}
    exact = {}
    for key in KEYS:
        v, a, sr = g[key]
        mp, bp = str(WORK / f"main_{key}.mp4"), str(WORK / f"branch_{key}.mp4")
        main_save(v, mp, a, sr)
        branch_save(v, bp, a, sr)
        ref[key] = sha(mp)
        exact[key] = {"main_sha": ref[key], "branch_sha": sha(bp), "identical": sha(bp) == ref[key],
                      "sei_main": sei(mp), "sei_branch": sei(bp)}
        if not exact[key]["identical"]:
            errors.append(f"{key}: branch MP4 differs from main")
        if exact[key]["sei_main"] != exact[key]["sei_branch"]:
            errors.append(f"{key}: SEI differs")
        log(f"exact {key}: {exact[key]}")
    res["exact"] = exact

    # --- interleaved timing main / branch / main2 (A/A)
    times = {k: {"main": [], "branch": [], "main2": []} for k in KEYS}
    for r in range(args.reps):
        for key in KEYS:
            v, a, sr = g[key]
            for arm, fn in (("main", main_save), ("branch", branch_save), ("main2", main_save)):
                path = str(WORK / f"t_{arm}_{key}.mp4")
                torch.cuda.synchronize()
                times[key][arm].append(timed(fn, v, path, a, sr))
                if sha(path) != ref[key]:
                    errors.append(f"timed {arm} {key} rep {r}: bytes differ from main")
        log(f"round {r + 1}/{args.reps}: " + ", ".join(
            f"{k[:5]} m={times[k]['main'][-1]:.0f} b={times[k]['branch'][-1]:.0f} m2={times[k]['main2'][-1]:.0f}"
            for k in KEYS))
    table = {}
    for key in KEYS:
        row = {arm: spread(ts) for arm, ts in times[key].items()}
        row["branch_minus_main_ms"] = round(row["branch"]["median_ms"] - row["main"]["median_ms"], 1)
        row["aa_main2_minus_main_ms"] = round(row["main2"]["median_ms"] - row["main"]["median_ms"], 1)
        table[key] = row
    res["u2_table"] = table

    # --- 5 back-to-back saves reuse the buffer
    v, a, sr = g["park_portrait"]
    ptrs, shas = [], []
    for i in range(5):
        p = str(WORK / f"b2b_{i}.mp4")
        branch_save(v, p, a, sr)
        ptrs.append(E._PINNED_BUFFER.data_ptr() if E._PINNED_BUFFER is not None else None)
        shas.append(sha(p))
    b2b = {"same_ptr": len(set(ptrs)) == 1 and ptrs[0] is not None, "all_identical": set(shas) == {ref["park_portrait"]},
           "buffer_bytes": E._PINNED_BUFFER.numel() if E._PINNED_BUFFER is not None else None}
    res["back_to_back"] = b2b
    if not (b2b["same_ptr"] and b2b["all_identical"]):
        errors.append(f"back-to-back: {b2b}")

    # --- two threads: park + robot concurrently, 5 times
    thr = []
    for it in range(5):
        outs, errs = {}, []

        def work(key, it=it):
            try:
                vv, aa, ss = g[key]
                p = str(WORK / f"thr_{key}_{it}.mp4")
                branch_save(vv, p, aa, ss)
                outs[key] = sha(p)
            except Exception as exc:  # noqa: BLE001
                errs.append(repr(exc))

        ts = [threading.Thread(target=work, args=(k,)) for k in ("park_portrait", "golden_robot")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        ok = not errs and all(outs.get(k) == ref[k] for k in ("park_portrait", "golden_robot"))
        thr.append(ok)
        if not ok:
            errors.append(f"two-thread iter {it}: {errs} {outs}")
    res["two_thread_identical"] = thr

    # --- avi output (ffmpeg path) main vs branch, and CPU-tensor input
    v, a, sr = g["perfume_bottle"]
    ma, ba = str(WORK / "main_perfume.avi"), str(WORK / "branch_perfume.avi")
    main_save(v, ma, None, sr)
    branch_save(v, ba, None, sr)
    avi = {"identical": sha(ma) == sha(ba), "size": os.path.getsize(ba)}
    cp = str(WORK / "branch_cpu_perfume.mp4")
    branch_save(v.cpu(), cp, a.cpu(), sr)
    cpu_ok = sha(cp) == ref["perfume_bottle"]
    # pure-python fallback encoder on a CUDA tensor still works (smoke)
    pp = str(WORK / "pure.avi")
    E._PurePythonEncoder().encode_video(v[:4], pp, FPS)
    res["avi"] = avi
    res["cpu_tensor_identical"] = cpu_ok
    res["pure_python_avi_bytes"] = os.path.getsize(pp)
    if not avi["identical"]:
        errors.append("avi differs")
    if not cpu_ok:
        errors.append("cpu tensor input differs")

    # --- fresh-process first save
    first = []
    for rep in range(args.first_reps):
        for arm, reserve in (("main", False), ("branch", False), ("branch", True)):
            r = run_first(arm, reserve)
            r["rep"] = rep
            first.append(r)
            log(f"first: {r}")
            if "error" in r:
                errors.append(f"first {arm} reserve={reserve}: {r['error'][-300:]}")
            elif r["sha256"] != ref["park_portrait"]:
                errors.append(f"first {arm} reserve={reserve}: bytes differ")
    res["first_save"] = first

    res["errors"] = errors
    res["passed"] = not errors
    out = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or WORK) / "u2_full.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"passed": not errors, "compared": "branch save_video MP4/AVI bytes vs inline copy of "
                      "main b2f9c52d encode_video on the 3 decoded frozen references",
                      "errors": errors, "gpu": {k: info.get(k) for k in ("physical_gpu", "affinity_summary", "pcie_width")},
                      "u2_table": {k: {arm: table[k][arm]["median_ms"] for arm in ("main", "branch", "main2")}
                                   for k in KEYS},
                      "first_save": [{k: r.get(k) for k in ("arm", "reserve", "reserve_ms", "first_ms", "second_ms")}
                                     for r in first]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pair", "full", "first"], default="pair")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--first-reps", type=int, default=2)
    ap.add_argument("--arm", default="branch")
    ap.add_argument("--reserve", action="store_true")
    args = ap.parse_args()
    {"pair": mode_pair, "full": mode_full, "first": mode_first}[args.mode](args)
