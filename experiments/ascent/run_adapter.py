"""Run a frozen benchmark adapter (experiments/ascent-benchmark/<name>.py) from a GPU job.

usage: run_adapter.py <check|profile_nsys|proxy> <checkpoint> [--route-log]

Builds the adapter's harness JSON for the frozen workload (FLUX.3 klein nvfp4 t2i 1024^2,
16 steps, c1; revision read from the checkpoint's HF metadata) and runs the adapter unchanged.
--route-log (diagnostic, this job only): puts a sitecustomize.py on PYTHONPATH that wraps
flashinfer.run_fmha_v2_fp8_sm12x in every server process and logs, per call, the sequence
length, causal/LSE flags, the contiguity of the E4M3 Q/K/V it receives (landing 29's K2
output) and which module the branch's guard selects. Nothing in the source tree changes.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "ascent-benchmark"

SITE = r'''
import importlib.abc, importlib.util, json, os, sys, threading
_TARGET = "tensorrt_llm._torch.visual_gen.attention.backends.flashinfer"
_LOG = os.environ.get("FLUX3_ROUTE_LOG")
_LOCK = threading.Lock()
_STATE = {"n": 0, "counts": {}}

def _write(rec):
    try:
        with _LOCK, open(_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass

def _patch(mod):
    orig = mod.run_fmha_v2_fp8_sm12x
    def run_fmha_v2_fp8_sm12x(q_fp8, k_fp8, v_fp8, output, num_heads, head_dim, seq_len, *,
                              causal, return_lse=False, **kw):
        flag = getattr(mod, "_USE_FMHA_V2_FLUX3_B2", None)
        contig = [bool(t.is_contiguous()) for t in (q_fp8, k_fp8, v_fp8)]
        routed = bool(flag) and head_dim == 128 and not causal and not return_lse and all(contig)
        key = f"S={seq_len} d={head_dim} causal={causal} lse={return_lse} contig={contig} b2={routed}"
        _STATE["n"] += 1
        _STATE["counts"][key] = _STATE["counts"].get(key, 0) + 1
        if _STATE["n"] <= 40:
            _write({"pid": os.getpid(), "call": _STATE["n"], "S": seq_len, "head_dim": head_dim,
                    "causal": causal, "lse": return_lse, "contiguous_qkv": contig,
                    "strides": [list(t.stride()) for t in (q_fp8, k_fp8, v_fp8)],
                    "dtype": str(q_fp8.dtype), "flag_b2": flag, "guard_routes_b2": routed})
        if _STATE["n"] % 608 == 0:
            _write({"pid": os.getpid(), "calls": _STATE["n"], "counts": dict(_STATE["counts"])})
        return orig(q_fp8, k_fp8, v_fp8, output, num_heads, head_dim, seq_len, causal=causal,
                    return_lse=return_lse, **kw)
    mod.run_fmha_v2_fp8_sm12x = run_fmha_v2_fp8_sm12x
    _write({"pid": os.getpid(), "patched": _TARGET, "argv0": sys.argv[0][-80:]})

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            pass
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module
        def exec_module(module, orig_exec=orig_exec):
            orig_exec(module)
            _patch(module)
        spec.loader.exec_module = exec_module
        return spec

if _LOG:
    sys.meta_path.insert(0, _Finder())
'''


def log(msg):
    print(f"[run_adapter {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    adapter, ckpt = sys.argv[1], sys.argv[2]
    route_log = "--route-log" in sys.argv[3:]
    meta = Path(ckpt) / ".cache" / "huggingface" / "download" / "config.json.metadata"
    revision = meta.read_text().splitlines()[0].strip() if meta.is_file() else None
    harness = {"workload_identity": {
        "workload": {"kind": "visual-gen", "task": "text-to-image", "variant": "nvfp4",
                     "model": "black-forest-labs/flux-3-image-klein", "width": 1024, "height": 1024,
                     "num_inference_steps": 16, "concurrency": [1]},
        "resolved_model": {"checkpoint": ckpt, "revision": revision}}}
    log(f"harness {json.dumps(harness)}")
    env = dict(os.environ)
    rl = None
    if route_log:
        art = Path(os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp())
        art.mkdir(parents=True, exist_ok=True)
        site_dir = Path(tempfile.mkdtemp(prefix="flux3_route_site_"))
        (site_dir / "sitecustomize.py").write_text(SITE)
        rl = art / "flux3_route_log.jsonl"
        env["FLUX3_ROUTE_LOG"] = str(rl)
        env["PYTHONPATH"] = str(site_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        log(f"route log -> {rl}; sitecustomize in {site_dir}")
    cmd = [sys.executable, str(BENCH / f"{adapter}.py"), json.dumps(harness)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                            cwd=str(BENCH))
    last = ""
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            last = line
        print(line, flush=True)
    rc = proc.wait()
    log(f"adapter rc={rc}")
    if rl is not None:
        recs = []
        if rl.is_file():
            recs = [json.loads(x) for x in rl.read_text().splitlines() if x.strip()]
        log(f"route log: {len(recs)} records")
        for r in recs[:12]:
            log(json.dumps(r))
        counts = [r for r in recs if "counts" in r]
        if counts:
            log("final counts: " + json.dumps(counts[-1]))
        try:
            res = json.loads(last)
        except Exception:  # noqa: BLE001
            res = {"adapter_last_line": last[-2000:]}
        res["route_log"] = {"first": [r for r in recs if "call" in r][:6],
                            "patched": [r for r in recs if "patched" in r],
                            "final_counts": counts[-1] if counts else None}
        print(json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
