"""Idea 152 check: frozen check adapter on this branch + per-MP4 sha256/probe + server-log grep.

Runs experiments/ascent/pd68/j2_check.py (-> run_check_wrapper.py -> the frozen
experiments/ascent-benchmark/run_check.py; TRTLLM_VISUAL_GEN_RESPONSE_TIMING=1), then for every
delivered MP4 (pass1 + park repeat) reports sha256, bytes and a probe (frames counted by a stream
copy, WxH, fps, pix_fmt, audio codec/rate/channels/duration), and greps every server log for
'ffmpeg encoding failed', pinned-allocation errors/warnings and failed request statuses.
Optional argv[3]: JSON {key: sha256} of main's MP4s to compare against.

usage: python experiments/ascent/pd82/check152.py <checkpoint> experiments/ascent-benchmark/run_check.py [main_shas.json]
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def find_ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    cands = sorted(glob.glob(os.path.join(os.environ.get("TMPDIR") or "/tmp", "ab-work", "ffmpeg-bin", "ffmpeg")))
    return cands[0] if cands else "ffmpeg"


def probe(ff, path):
    out = {"bytes": os.path.getsize(path), "sha256": sha(path)}
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        r = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_entries",
                            "stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,nb_read_frames,"
                            "sample_rate,channels,duration", "-of", "json", path], capture_output=True, text=True)
        try:
            out["ffprobe"] = json.loads(r.stdout).get("streams")
        except Exception:
            out["ffprobe_err"] = r.stderr[-300:]
    info = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Video: (\w+)[^,]*, (\w+)[^,]*(?:\([^)]*\))?, (\d+)x(\d+)", info)
    if m:
        out.update(vcodec=m.group(1), pix_fmt=m.group(2), width=int(m.group(3)), height=int(m.group(4)))
    m = re.search(r"([\d.]+) fps", info)
    if m:
        out["fps"] = float(m.group(1))
    m = re.search(r"Audio: (\w+)[^,]*, (\d+) Hz, ([\w.()]+)", info)
    if m:
        out.update(acodec=m.group(1), audio_rate=int(m.group(2)), audio_layout=m.group(3))
    m = re.search(r"Duration: ([\d:.]+)", info)
    if m:
        out["container_duration"] = m.group(1)
    # Frames: decode every frame to rgb24 and count bytes (as ab_common.decode_video does).
    v = subprocess.run([ff, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", path, "-map", "0:v:0",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    if out.get("width") and out.get("height"):
        out["frames"] = len(v.stdout) // (out["width"] * out["height"] * 3)
    a = subprocess.run([ff, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", path, "-map", "0:a:0",
                        "-f", "s16le", "-ac", "2", "-ar", "32000", "-"], capture_output=True)
    out["audio_samples_32k"] = len(a.stdout) // 4
    out["audio_duration_s"] = round(len(a.stdout) / 4 / 32000, 4)
    return out


def main() -> int:
    checkpoint, adapter = sys.argv[1], sys.argv[2]
    main_shas = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}
    proc = subprocess.run([sys.executable, str(ROOT / "experiments/ascent/pd68/j2_check.py"), checkpoint, adapter],
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-20000:])
    sys.stderr.write(proc.stderr[-5000:])
    j2 = {}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                j2 = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    art = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or (Path.cwd() / "ab-artifacts"))
    res_path = art / "adapter_result.json"
    res = json.loads(res_path.read_text()) if res_path.is_file() else {}
    ff = find_ffmpeg()
    mp4s = []
    for i, r in enumerate(res.get("requests", [])):
        p = r.get("path")
        rec = {"order": i, "key": r.get("key"), "path": p}
        if p and os.path.isfile(p):
            rec.update(probe(ff, p))
        mp4s.append(rec)
        print("[c152] mp4", json.dumps(rec), flush=True)
    bad_re = re.compile(r"ffmpeg encoding failed|Pinned video staging buffer|pinned video buffer reservation failed|"
                        r"cudaHostAlloc|status[\"'=: ]+failed|\"failed\"|Error processing response", re.I)
    bad_lines = []
    logs = sorted(glob.glob(str(art / "logs" / "server-*.log")))
    for f in logs:
        for line in open(f, errors="replace"):
            if bad_re.search(line):
                bad_lines.append(f"{os.path.basename(f)}: {line.strip()[:300]}")
    failed_reqs = [r for r in res.get("requests", []) if r.get("success") is False]
    geometry_ok = bool(mp4s) and all(m.get("frames") == 124 and m.get("width") == 1344 and m.get("height") == 768
                                     and m.get("fps") == 24.0 and m.get("audio_rate") == 32000 for m in mp4s)
    park = [m for m in mp4s if m["key"] == "park_portrait"]
    park_repeat_bitwise = len(park) == 2 and park[0].get("sha256") == park[1].get("sha256")
    sha_vs_main = {m["key"]: (m.get("sha256") == main_shas[m["key"]]) for m in mp4s if m["key"] in main_shas}
    passed = bool(j2.get("passed")) and bool(res.get("valid")) and bool(res.get("repeated_execution_passed")) \
        and geometry_ok and not bad_lines and not failed_reqs and all(sha_vs_main.values())
    print("[c152] bad log lines:", bad_lines[:20], flush=True)
    print(json.dumps({
        "passed": passed,
        "compared": "frozen check adapter vs original-source refs; same-server park repeat; MP4 probe; "
                    "server-log grep; MP4 sha256 vs main 58518211 when given",
        "valid": res.get("valid"), "repeated_execution_passed": res.get("repeated_execution_passed"),
        "errors": res.get("errors"), "per_prompt": j2.get("per_prompt"), "repeat_bitwise": j2.get("repeat_bitwise"),
        "tolerance": {k: v for k, v in (res.get("tolerance") or {}).items() if k in ("lpips_tol", "psnr_tol")},
        "mp4": mp4s, "geometry_ok": geometry_ok, "park_repeat_sha_equal": park_repeat_bitwise,
        "sha_vs_main": sha_vs_main, "bad_log_lines": bad_lines, "failed_requests": len(failed_reqs),
        "e2e_minus_gen": j2.get("e2e_minus_gen"), "park_first_minus_repeat_s": j2.get("park_first_minus_repeat_s"),
        "reservation_log": j2.get("reservation_log"), "startup_s": res.get("startup_s"),
    }, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
