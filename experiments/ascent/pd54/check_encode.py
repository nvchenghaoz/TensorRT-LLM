"""Untimed exactness check for idea 54 (branch encode vs main's encode, verbatim copy in u2_encode.py).

For each of the three proxy videos (frames from the frozen reference MP4s) and for BOTH layouts:
  planar     : main pre-landing-25 post on a channel-planar GPU tensor (strides (HW, W, 1, THW))
  contiguous : the post-landing-25 [T, H, W, C]-contiguous GPU tensor (reshape(-1) is a view)
it runs main's encode once (reference), then 5 back-to-back branch encodes reusing the pinned buffer,
and compares sha256 and the x264 SEI of every output with the reference; checks the pinned buffer is
not reallocated; and checks the CPU-tensor input, the lock-busy pageable fallback and the AVI fallback.
Last stdout line: {"passed": bool, ...}.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2_encode as u2  # noqa: E402

enc = u2.enc


def main():
    work = tempfile.mkdtemp(prefix="chk54_")
    ffmpeg = u2.ensure_ffmpeg(work)
    out = {"tree_is_branch": u2.IS_BRANCH, "cases": {}}
    problems = []
    for key, rel in u2.REFS.items():
        frames, audio, rate = u2.decode_ref(ffmpeg, os.path.join(u2.STORE, rel))
        audio_t = torch.from_numpy(audio.T.copy()).cuda()
        layouts = {"planar": u2.build_gpu_video(frames),
                   "contiguous": torch.from_numpy(frames.copy()).cuda()}
        for lay, video in layouts.items():
            d = os.path.join(work, f"{key}_{lay}")
            os.makedirs(d, exist_ok=True)
            ref = os.path.join(d, "main.mp4")
            u2.main_encode(video, ref, u2.FPS, audio_t, rate)
            ref_sha, ref_sei = u2.sha(ref), u2.x264_sei(ref)
            case = {"stride": tuple(video.stride()), "main_sha256": ref_sha, "sei_threads":
                    (ref_sei or "").split("threads=")[1].split()[0] if ref_sei and "threads=" in ref_sei else None}
            with enc._host_numpy(video) as a:
                case["host_view_strides"] = a.strides
                case["reshape_is_view"] = bool(u2.np.shares_memory(a.reshape(-1), a))
            ptr0 = enc._PINNED_BUFFER.data_ptr()
            reps = []
            for i in range(5):
                p = os.path.join(d, f"b{i}.mp4")
                u2.tree_encode(video, p, u2.FPS, audio_t, rate)
                reps.append(u2.sha(p) == ref_sha and u2.x264_sei(p) == ref_sei)
                os.unlink(p)
            case["back_to_back_5_equal"] = reps
            case["pinned_buffer_reused"] = enc._PINNED_BUFFER.data_ptr() == ptr0
            p = os.path.join(d, "cpu.mp4")
            u2.tree_encode(video.cpu(), p, u2.FPS, audio_t.cpu(), rate)
            case["cpu_tensor_equal"] = u2.sha(p) == ref_sha
            p = os.path.join(d, "busy.mp4")
            with enc._PINNED_LOCK:
                u2.tree_encode(video, p, u2.FPS, audio_t, rate)
            case["lock_busy_fallback_equal"] = u2.sha(p) == ref_sha
            ok = all(reps) and case["pinned_buffer_reused"] and case["cpu_tensor_equal"] \
                and case["lock_busy_fallback_equal"]
            if lay == "contiguous" and not case["reshape_is_view"]:
                ok = False
            if not ok:
                problems.append(f"{key}/{lay}: {case}")
            case["passed"] = ok
            out["cases"][f"{key}/{lay}"] = case
            u2.log(f"{key}/{lay}: {json.dumps(case)}")
        if key == "park_portrait":
            v = layouts["planar"]
            pa, pb = os.path.join(work, "a.avi"), os.path.join(work, "b.avi")
            enc._PurePythonEncoder().encode_video(v[:8], pa, u2.FPS)
            enc._PurePythonEncoder().encode_video(v[:8].cpu(), pb, u2.FPS)
            out["avi_fallback_ok"] = os.path.getsize(pa) > 0 and u2.sha(pa) == u2.sha(pb)
            if not out["avi_fallback_ok"]:
                problems.append("avi fallback")
        del layouts
        torch.cuda.empty_cache()
    problems.extend(sorted(set(u2.STRIDE_PROBLEMS)))
    out.update(passed=not problems and u2.IS_BRANCH, problems=problems, output={"class": "exact"},
               compared="sha256 + x264 SEI of branch encode_video MP4s vs main 18c4ba99's encode_video "
                        "(verbatim copy) on the same GPU input, planar and contiguous layouts")
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
