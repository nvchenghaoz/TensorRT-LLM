"""U2 for idea 54: server-side MP4 encode of the MiniMax-H3 video (1344x768x124, uint8).

Runs in either tree (run_gpu_pair: baseline = main, candidate = branch). Inputs are the three
frozen reference MP4s, decoded and rebuilt on the GPU through main's own post
((v.permute(0,2,3,4,1)*255).round().to(uint8) on a channel-planar [1,3,T,H,W] tensor, then
to_tensor's clone() and save_video's [0]), so the video has main's planar strides.

Variants, interleaved per repeat with a rotating order, each with and without a
DiffusionClient-like busy thread:
  main_a, main_b : main's encode_video, copied verbatim below (pageable .cpu(), communicate())
  tree           : this tree's _FfmpegCliEncoder.encode_video (main in the baseline arm)
  pinned_only    : main's code with only the D2H replaced by this tree's _host_numpy (branch only)
  pipe_only      : main's code with only the feed replaced by this tree's _feed_ffmpeg (branch only)
Rows (information): pageable vs pinned D2H, .numpy(), reshape(-1) copy, and the communicate()
check (384 MB into `ffmpeg ... -f null -`: 3 pipes vs 1 pipe vs one direct write).
Exactness: sha256 of every MP4 of every variant and repeat vs main_a's, x264 SEI string, 5 back-to-back
tree encodes on the pinned buffer (pointer unchanged), CPU-tensor input, lock-busy pageable fallback,
AVI fallback.
Last stdout line: {"batch": {"1": {"step_ms", "noise_floor"}}, "output": {"class": "exact"}, "passed", ...}
"""

import contextlib
import hashlib
import json
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

import tensorrt_llm.media.encoding as enc

STORE = "/home/scratch.chengzhang/trtllm-dev-minimax-h3-2g-20261112-143634/ascent-benchmark-data/v1"
REFS = {
    "park_portrait": "archive/6ad9624f725b93e6d002b78f736db19630ed63f24f12fdd34b89ebb259ced714.mp4",
    "perfume_bottle": "archive/d77cbdfb9fe60654533772f322870d434e99056890d2611ffe4abb7c209f0fd7.mp4",
    "golden_robot": "archive/0448c5bdd801f124025614e3aadab4c6b3bbda1be819d76287291b466deeb1fb.mp4",
}
REPEATS = int(os.environ.get("U2_REPEATS", "10"))
FPS = 24.0
T0 = time.time()
IS_BRANCH = hasattr(enc, "_host_numpy") and hasattr(enc, "_feed_ffmpeg")


def log(msg):
    print(f"[u2 {time.time() - T0:7.1f}s] {msg}", flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"err {e}"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def x264_sei(path):
    data = open(path, "rb").read()
    m = re.search(rb"x264 - core[^\x00]*", data)
    return m.group(0).decode("latin1") if m else None


def ensure_ffmpeg(work):
    found = shutil.which("ffmpeg")
    if not found:
        bindir = os.path.join(work, "bin")
        os.makedirs(bindir, exist_ok=True)
        target = os.path.join(bindir, "ffmpeg")
        shutil.copyfile(os.path.join(STORE, "tools", "ffmpeg"), target)
        os.chmod(target, 0o755)
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        found = target
    return found


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    med = statistics.median(xs)
    p10 = xs[max(0, int(0.1 * (n - 1)))]
    p90 = xs[min(n - 1, int(round(0.9 * (n - 1))))]
    return {"median_ms": round(med, 2), "p10_ms": round(p10, 2), "p90_ms": round(p90, 2),
            "min_ms": round(xs[0], 2), "max_ms": round(xs[-1], 2), "n": n,
            "spread": round((p90 - p10) / med, 4) if med else None}


# --------------------------------------------------------------------------- busy thread


class Busy:
    """DiffusionClient._serve_forever-like: queue.get(10 ms) + 1 ms sleep + a little Python."""

    def __init__(self):
        self.stop = threading.Event()
        self.q = queue.Queue()
        self.th = threading.Thread(target=self.run, daemon=True)
        self.iters = 0

    def run(self):
        while not self.stop.is_set():
            try:
                self.q.get(timeout=0.01)
            except queue.Empty:
                pass
            time.sleep(0.001)
            sum(range(200))
            self.iters += 1

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join()


# --------------------------------------------------------------------------- main's encode, verbatim


def main_encode(video, output_path, frame_rate, audio=None, audio_sample_rate=24000,
                d2h="pageable", feed="communicate"):
    """main 18c4ba99 _FfmpegCliEncoder.encode_video, with d2h / feed swappable per item."""
    import math
    import subprocess as sp

    self = enc._FfmpegCliEncoder()
    self._validate_video_tensor(video)
    ctx = enc._host_numpy(video) if d2h == "pinned" else contextlib.nullcontext(video.cpu().numpy())
    with ctx as video_np:
        num_frames, height, width, channels = video_np.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB video, got {channels} channels")
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_input_args = []
            audio_output_args = []
            if audio is not None:
                audio_tensor = self._validate_audio_tensor(audio)
                required_samples = math.ceil(num_frames * audio_sample_rate / frame_rate)
                missing_samples = required_samples - audio_tensor.shape[0]
                if missing_samples > 0:
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 0, 0, missing_samples))
                audio_np = audio_tensor.cpu().numpy()
                tmp_audio_path = os.path.join(tmp_dir, "audio.wav")
                self._write_wav(tmp_audio_path, audio_np, sample_rate=audio_sample_rate)
                audio_input_args = ["-i", tmp_audio_path]
                audio_output_args = ["-c:a", "aac", "-shortest"]
            ffmpeg_path = enc._get_ffmpeg_path()
            cmd = [ffmpeg_path, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                   "-r", str(frame_rate), "-i", "-", *audio_input_args, "-c:v", "libx264", "-pix_fmt",
                   "yuv420p", "-preset", enc._x264_preset(), "-crf", "23", *audio_output_args, output_path]
            if feed == "communicate":
                process = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
                raw_frames = memoryview(video_np.reshape(-1))
                _, stderr = process.communicate(input=raw_frames)
                if process.returncode != 0:
                    raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode()}")
            else:
                enc._feed_ffmpeg(cmd, video_np, tmp_dir)
    return output_path


def tree_encode(video, output_path, frame_rate, audio=None, audio_sample_rate=24000):
    return enc._FfmpegCliEncoder().encode_video(video, output_path, frame_rate, audio, audio_sample_rate)


# --------------------------------------------------------------------------- inputs


def decode_ref(ffmpeg, path):
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    w, h = map(int, re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", probe).groups())
    rate = int(re.search(r"Audio: .*?, (\d+) Hz", probe).group(1))
    raw = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-an", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    araw = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-f", "f32le",
                           "-ac", "2", "-ar", str(rate), "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(araw, dtype=np.float32).reshape(-1, 2)
    return frames, audio, rate


def build_gpu_video(frames):
    """Main's post on a channel-planar GPU tensor, then to_tensor's clone() and save_video's [0]."""
    t, h, w, _ = frames.shape
    planar = torch.from_numpy(frames).cuda().permute(3, 0, 1, 2).contiguous()[None].float() / 255.0
    video = (planar.permute(0, 2, 3, 4, 1) * 255).round().to(torch.uint8)  # pipeline_minimax_h3.py:1244
    del planar
    video = video.clone()[0]
    want = (h * w, w, 1, t * h * w)
    if tuple(video.stride()) != want or video.is_contiguous():
        STRIDE_PROBLEMS.append(f"video strides {tuple(video.stride())} != planar {want}")
    return video


STRIDE_PROBLEMS = []


# --------------------------------------------------------------------------- rows


def time_rows(video, ffmpeg, work):
    out = {}
    torch.cuda.synchronize()
    pg, pn, npy, rs = [], [], [], []
    for _ in range(7):
        t0 = time.perf_counter()
        host = video.cpu()
        pg.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        arr = host.numpy()
        npy.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        flat = arr.reshape(-1)
        rs.append((time.perf_counter() - t0) * 1e3)
        out["reshape_is_view"] = bool(np.shares_memory(flat, arr))
        del host, arr, flat
        if IS_BRANCH:
            t0 = time.perf_counter()
            with enc._host_numpy(video) as a:
                pn.append((time.perf_counter() - t0) * 1e3)
                if a.strides != tuple(video.stride()):
                    STRIDE_PROBLEMS.append(f"pinned host view strides {a.strides} != {tuple(video.stride())}")
    out["d2h_pageable"] = stats(pg[2:])
    out["numpy"] = stats(npy[2:])
    out["reshape_copy"] = stats(rs[2:])
    if pn:
        out["d2h_pinned_incl_numpy"] = stats(pn[2:])

    # communicate() check: 384 MB of rgb24 into ffmpeg -f null.
    t, h, w, _ = video.shape
    raw = memoryview(np.ascontiguousarray(video.cpu().numpy()).reshape(-1))
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24.0", "-i", "-",
           "-f", "null", "-"]

    def comm3():
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate(input=raw)
        assert p.returncode == 0

    def comm1():
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.communicate(input=raw)
        assert p.returncode == 0

    def threaded():
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        errs = []
        th = threading.Thread(target=lambda: errs.append(p.stderr.read()))
        th2 = threading.Thread(target=lambda: p.stdout.read())
        th.start()
        th2.start()
        p.stdin.write(raw)
        p.stdin.close()
        th.join()
        th2.join()
        p.wait()
        assert p.returncode == 0

    def direct():
        with tempfile.TemporaryDirectory(dir=work) as d:
            enc._feed_ffmpeg(cmd, np.frombuffer(raw, dtype=np.uint8), d)

    feeds = {"communicate_3pipes": comm3, "communicate_1pipe": comm1, "threaded_write_3pipes": threaded}
    if IS_BRANCH:
        feeds["branch_feed"] = direct
    res = {}
    for busy in (False, True):
        times = {k: [] for k in feeds}
        for r in range(6):
            for k, fn in feeds.items():
                with (Busy() if busy else contextlib.nullcontext()):
                    t0 = time.perf_counter()
                    fn()
                    times[k].append((time.perf_counter() - t0) * 1e3)
        for k, v in times.items():
            res[f"{k}{'_busy' if busy else ''}"] = stats(v[1:])
    out["pipe_check_null_sink"] = res
    log(f"rows: {json.dumps(out)}")
    return out


# --------------------------------------------------------------------------- main


def main():
    work = tempfile.mkdtemp(prefix="u2_")
    ffmpeg = ensure_ffmpeg(work)
    env = {
        "tree": "branch" if IS_BRANCH else "main",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": sh(["nvidia-smi", "--query-gpu=index,pci.bus_id,name,pcie.link.width.current",
                   "--format=csv,noheader"]),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "nproc": os.cpu_count(),
        "ffmpeg": ffmpeg,
        "ffmpeg_sha256": sha(ffmpeg),
        "x264_preset": enc._x264_preset(),
        "python": sys.version.split()[0],
    }
    try:
        props = torch.cuda.get_device_properties(0)
        bus = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        env["this_gpu_pci"] = bus
        env["this_gpu_line"] = [ln for ln in env["gpu"].splitlines() if bus[5:].lower() in ln.lower()]
        p = f"/sys/bus/pci/devices/{bus}/numa_node"
        if os.path.exists(p):
            env["gpu_numa_node"] = open(p).read().strip()
        p = f"/sys/bus/pci/devices/{bus}/current_link_width"
        if os.path.exists(p):
            env["gpu_link_width"] = open(p).read().strip()
        env["numactl_show"] = sh(["bash", "-c", "numactl --show 2>/dev/null | head -5"])
    except Exception as e:  # noqa: BLE001
        env["numa_err"] = str(e)
    log(f"env: {json.dumps(env)}")

    inputs = {}
    for key, rel in REFS.items():
        frames, audio, rate = decode_ref(ffmpeg, os.path.join(STORE, rel))
        video = build_gpu_video(frames)
        audio_t = torch.from_numpy(audio.T.copy()).cuda()  # (channels, samples), like audios[i]
        inputs[key] = (video, audio_t, rate)
        log(f"{key}: video {tuple(video.shape)} stride {tuple(video.stride())} audio {tuple(audio_t.shape)} "
            f"@{rate} Hz")
    torch.cuda.synchronize()

    variants = {"main_a": lambda v, a, r, p: main_encode(v, p, FPS, a, r),
                "main_b": lambda v, a, r, p: main_encode(v, p, FPS, a, r),
                "tree": lambda v, a, r, p: tree_encode(v, p, FPS, a, r)}
    if IS_BRANCH:
        variants["pinned_only"] = lambda v, a, r, p: main_encode(v, p, FPS, a, r, d2h="pinned")
        variants["pipe_only"] = lambda v, a, r, p: main_encode(v, p, FPS, a, r, feed="direct")

    result = {"env": env, "prompts": {}}
    passed = True
    problems = []
    rows = time_rows(inputs["park_portrait"][0], ffmpeg, work)
    result["rows_park"] = rows

    for key, (video, audio_t, rate) in inputs.items():
        out_dir = os.path.join(work, key)
        os.makedirs(out_dir, exist_ok=True)
        names = list(variants)
        # warmup (also the first pinned allocation)
        ref_path = os.path.join(out_dir, "ref.mp4")
        main_encode(video, ref_path, FPS, audio_t, rate)
        ref_sha, ref_sei = sha(ref_path), x264_sei(ref_path)
        for n in names:
            p = os.path.join(out_dir, f"warm_{n}.mp4")
            variants[n](video, audio_t, rate, p)
        times = {f"{n}{s}": [] for n in names for s in ("", "_busy")}
        shas = {}
        for r in range(REPEATS):
            order = names[r % len(names):] + names[:r % len(names)]
            for busy in ((False, True) if r % 2 == 0 else (True, False)):
                for n in order:
                    p = os.path.join(out_dir, f"{n}_{r}_{int(busy)}.mp4")
                    with (Busy() if busy else contextlib.nullcontext()):
                        t0 = time.perf_counter()
                        variants[n](video, audio_t, rate, p)
                        dt = (time.perf_counter() - t0) * 1e3
                    times[f"{n}{'_busy' if busy else ''}"].append(dt)
                    s = sha(p)
                    shas.setdefault(n, set()).add(s)
                    if s != ref_sha:
                        problems.append(f"{key} {n} rep {r} busy {busy}: sha differs from main")
                    os.unlink(p)
            log(f"{key} rep {r}: " + ", ".join(f"{k}={v[-1]:.0f}" for k, v in times.items()))
        table = {k: stats(v) for k, v in times.items()}
        pinfo = {"main_sha256": ref_sha, "x264_sei": ref_sei, "mp4_bytes": os.path.getsize(ref_path),
                 "table": table,
                 "aa_delta_ms": round(table["main_b"]["median_ms"] - table["main_a"]["median_ms"], 2),
                 "tree_minus_main_a_ms": round(table["tree"]["median_ms"] - table["main_a"]["median_ms"], 2),
                 "variant_shas_equal_main": {n: (s == {ref_sha}) for n, s in shas.items()}}
        if IS_BRANCH:
            pinfo["tree_busy_minus_main_a_busy_ms"] = round(
                table["tree_busy"]["median_ms"] - table["main_a_busy"]["median_ms"], 2)
            # 5 back-to-back tree encodes on the pinned buffer
            ptr0 = enc._PINNED_BUFFER.data_ptr() if enc._PINNED_BUFFER is not None else None
            b2b = []
            for i in range(5):
                p = os.path.join(out_dir, f"b2b_{i}.mp4")
                tree_encode(video, p, FPS, audio_t, rate)
                b2b.append({"sha_equal": sha(p) == ref_sha, "sei_equal": x264_sei(p) == ref_sei})
                os.unlink(p)
            ptr1 = enc._PINNED_BUFFER.data_ptr()
            pinfo["back_to_back_5"] = b2b
            pinfo["pinned_buffer_reused"] = ptr0 == ptr1
            pinfo["pinned_buffer_bytes"] = enc._PINNED_BUFFER.numel()
            if not all(x["sha_equal"] and x["sei_equal"] for x in b2b) or ptr0 != ptr1:
                problems.append(f"{key}: back-to-back pinned encodes differ or buffer reallocated")
            # CPU-tensor input (strided CPU tensor -> pageable path)
            p = os.path.join(out_dir, "cpu_in.mp4")
            tree_encode(video.cpu(), p, FPS, audio_t.cpu(), rate)
            pinfo["cpu_tensor_sha_equal"] = sha(p) == ref_sha
            # lock busy -> pageable fallback
            p2 = os.path.join(out_dir, "lock_busy.mp4")
            with enc._PINNED_LOCK:
                tree_encode(video, p2, FPS, audio_t, rate)
            pinfo["lock_busy_fallback_sha_equal"] = sha(p2) == ref_sha
            if not (pinfo["cpu_tensor_sha_equal"] and pinfo["lock_busy_fallback_sha_equal"]):
                problems.append(f"{key}: CPU-tensor or lock-busy fallback differs")
        if key == "park_portrait":
            # AVI fallback (pure-Python MJPEG): GPU and CPU inputs, compare with each other
            pa = os.path.join(out_dir, "fb_gpu.avi")
            pb = os.path.join(out_dir, "fb_cpu.avi")
            enc._PurePythonEncoder().encode_video(video[:8], pa, FPS)
            enc._PurePythonEncoder().encode_video(video[:8].cpu(), pb, FPS)
            pinfo["avi_fallback_ok"] = os.path.getsize(pa) > 0 and sha(pa) == sha(pb)
            if not pinfo["avi_fallback_ok"]:
                problems.append("avi fallback failed")
        if not all(pinfo["variant_shas_equal_main"].values()):
            passed = False
        result["prompts"][key] = pinfo
        log(f"{key}: {json.dumps(pinfo)}")

    problems.extend(sorted(set(STRIDE_PROBLEMS)))
    passed = passed and not problems
    park = result["prompts"]["park_portrait"]["table"]
    result.update({
        "batch": {"1": {"step_ms": park["tree"]["median_ms"], "noise_floor": park["tree"]["spread"]}},
        "output": {"class": "exact"},
        "passed": passed,
        "problems": problems,
        "measure_note": "step_ms = park_portrait median wall of this tree's encode_video on main's planar "
                        "GPU video (save path: D2H + numpy + WAV + reshape copy + ffmpeg medium), no busy thread",
    })
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
