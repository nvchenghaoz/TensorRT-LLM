"""Idea 237 acceptance item 4: 4-GPU candidate timeline of the served config (one job, two server lifetimes).

Arms, back to back on the job's 4 GPUs, both with the default TRTLLM_SM12X_W4A8_FI_PREFILL=graph (unset):
  overlap_default : TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS unset (default 1024, landing 60's eager shared overlap)
  serial_100000   : TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS=100000 (serial reference)
Each arm runs the FROZEN profile adapter (experiments/ascent-benchmark/profile_run.py) in-process with the protocol
server settings overlaid by this tree's experiments/ascent-config/engine.json (main a9ae1a75's: EP2xTP2,
LOWPRECISION AR, tokenization cache): start trtllm-serve, prime the 32768-token prefix, warm up c1/c2/c4, then the
prefill window (3 workload requests, max_tokens=1, CPU+GPU Kineto on every rank) and the two decode windows.
Extra env per arm (attribution only, nothing timed):
  * TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY=1: each tail layer's first replay is compared with eager, and layers 20/39
    log '[TAIL_MOE_GRAPH KERNELS]' with the graph's own kernel sequence (receipt independent of Kineto).
  * PYTHONPATH += jitwatch/: sitecustomize logs every triton.compile / cutlass.cute.compile call per process
    with a wall-clock stamp (jitwatch/sitecustomize.py); readiness = the adapter's Server.start returning.
Checks per arm:
  (a) the 'FlashInfer PREFILL chain warmed up' line (token counts + tile table) is in the server log before readiness;
  (b) prefill-window trace, every rank: kernels launched by cudaGraphLaunch (correlation ids) in the 128-row tail
      MoE window graphs are the FlashInfer route + fc1_act_q1 + fc2_finalize, no CUTLASS fused_moe kernel;
  (c) zero Triton / CuTe-DSL compile events after readiness, in every server process.
Usage: timeline.py <checkpoint> [arms]. Last stdout line: JSON {passed, arms}. Not timing evidence.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/fi237_timeline"))
T0 = time.time()

SERVER_SETTINGS = {
    "context_parallel_size": 1, "cuda_graph_config": {"batch_sizes": [1, 2, 4, 8], "enable_padding": True},
    "enable_attention_dp": False, "enable_chunked_prefill": True,
    "kv_cache_config": {"enable_block_reuse": True, "free_gpu_memory_fraction": 0.5, "tokens_per_block": 128},
    "max_batch_size": 8, "max_num_tokens": 8192, "max_seq_len": 36864, "moe_expert_parallel_size": 4,
    "pipeline_parallel_size": 1, "tensor_parallel_size": 4, "trust_remote_code": True}
WORKLOAD = {"cached_prefix_tokens": 32768, "unique_input_tokens": 2048, "output_tokens": 256,
            "concurrency": [1, 2, 4], "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "model_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170"}
ARMS = {
    "overlap_default": {},
    "serial_100000": {"TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS": "100000"},
}

FI_RE = re.compile(r"flashinfer|route_assign|route_count|route_prefix|_route_|mxfp8_q0", re.I)
CUTLASS_MOE_RE = re.compile(r"MoeFCGemm|moe_gemm|moegemm|expandInputRows|finalizeMoeRouting|fused_moe|cutlass_kernels|"
                            r"computeStrides|buildExpertMaps|ExpertPrefixSum|doActivation|KernelPtrArray|"
                            r"GemmUniversal|MoeGemm|moe_permute|unpermute", re.I)
WARM_RE = re.compile(r"FlashInfer PREFILL chain warmed up")
LOG_RE = re.compile(r"PREFILL chain warmed|decode chain warmed|FlashInfer decode chain on|TAIL_MOE_GRAPH|"
                    r"shared-expert eager overlap|\[fc12_fused\]|capture failed|Traceback", re.I)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.1f}s]", *a, flush=True)


def lib_info():
    """Pitfall 3271: path + mtime of the libtensorrt_llm.so the job loads (this change is Python-only)."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("tensorrt_llm")
        root = Path(spec.origin).parent
        libs = sorted(glob.glob(str(root / "libs" / "libtensorrt_llm*.so")))
        return [{"path": p, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))}
                for p in libs] or [{"root": str(root), "libs": "none found"}]
    except Exception as exc:  # noqa: BLE001
        return [{"error": repr(exc)}]


def warm_list():
    code = ("from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as d;"
            "import json;print(json.dumps({'mode': d.FI_PREFILL_MODE, 'max': d.FI_PREFILL_MAX_TOKENS,"
            "'tiles': [list(map(str, t)) for t in d._PREFILL_TILE_TABLE], 'ms': d.prefill_warm_tokens(),"
            "'tile_of_128': list(map(str, d.prefill_tiles(128)))}))")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)[:300]}


def run_arm(arm, ckpt):
    """Run the frozen profile adapter once with this arm's env; returns readiness time and log offsets."""
    bench = None
    for c in (Path.cwd() / "experiments" / "ascent-benchmark", HERE.parents[2] / "ascent-benchmark"):
        if (c / "profile_run.py").exists():
            bench = c
            break
    if bench is None:
        raise SystemExit("frozen profile_run.py not found")
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    import common  # noqa: E402  (frozen harness)
    import profile_run  # noqa: E402

    out_root = ART / arm
    out_root.mkdir(parents=True, exist_ok=True)
    jw = out_root / "jitwatch"
    jw.mkdir(exist_ok=True)
    saved = dict(os.environ)
    os.environ.pop("TRTLLM_SM12X_W4A8_FI_PREFILL", None)
    os.environ.pop("TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS", None)
    os.environ.update(ARMS[arm])
    os.environ["TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY"] = "1"
    os.environ["JITWATCH_DIR"] = str(jw)
    os.environ["PYTHONPATH"] = str(HERE / "jitwatch") + (
        os.pathsep + saved["PYTHONPATH"] if saved.get("PYTHONPATH") else "")
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
    log(f"[{arm}] env: " + json.dumps({k: os.environ.get(k) for k in (
        "TRTLLM_SM12X_W4A8_FI_PREFILL", "TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS", "TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY",
        "JITWATCH_DIR", "PYTHONPATH", "TLLM_LOG_LEVEL")}))

    harness = {"server": dict(SERVER_SETTINGS), "server_start_timeout_s": 2700}
    orig_wl = common.workload

    def workload(h):
        try:
            wl = orig_wl(h)
        except Exception as exc:  # noqa: BLE001
            log(f"common.workload failed ({exc!r}); using the protocol workload")
            wl = {}
        wl = dict(wl or {})
        for k, v in WORKLOAD.items():
            wl.setdefault(k, v)
        wl.setdefault("checkpoint", ckpt)
        return wl

    def artifact_dir(name):
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    info = {"ready_at": None}
    orig_start = common.Server.start

    def start(self, *a, **k):
        r = orig_start(self, *a, **k)
        info["ready_at"] = time.time()
        sizes = {}
        for f in glob.glob(str(out_root / "**" / "*.log"), recursive=True):
            try:
                sizes[f] = os.path.getsize(f)
            except OSError:
                pass
        info["log_sizes_at_ready"] = sizes
        log(f"[{arm}] server ready; log sizes {sizes}")
        return r

    patches = [(common, "workload", workload), (common, "load_config", lambda argv: (harness, Path.cwd().resolve())),
               (common, "artifact_dir", artifact_dir), (common.Server, "start", start),
               (profile_run, "analyze_files",
                lambda paths, hb=None: {p: (p, {"kernels": 0, "parser": "skipped"}) for p in paths})]
    olds = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, v in patches:
        setattr(o, n, v)
    try:
        profile_run.main()
    except SystemExit as e:
        log(f"[{arm}] profile adapter exit {e.code}")
    except Exception as exc:  # noqa: BLE001
        log(f"[{arm}] profile adapter raised {exc!r}\n{traceback.format_exc()[-2000:]}")
    finally:
        for o, n, v in olds:
            setattr(o, n, v)
        os.environ.clear()
        os.environ.update(saved)
    info["done_at"] = time.time()
    return info


def load_trace(path):
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    return [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"]


def short(n, k=150):
    n = n.strip()
    if n.startswith("void "):
        n = n[5:]
    return n[:k]


def reduce_prefill(files):
    """Per rank: graph-launched kernels (correlation of cudaGraphLaunch*) by launch, classified."""
    out = {}
    for f in files:
        m = re.search(r"rank-(\d+)", os.path.basename(f))
        r = int(m.group(1)) if m else len(out)
        try:
            evs = load_trace(f)
        except Exception as exc:  # noqa: BLE001
            out[r] = {"error": repr(exc)[:300], "file": f}
            continue
        glaunch = {}
        for e in evs:
            if e.get("cat") in ("cuda_runtime", "cuda_driver") and "GraphLaunch" in e.get("name", ""):
                c = (e.get("args") or {}).get("correlation")
                if c is not None:
                    glaunch[c] = e["ts"]
        kern = sorted((e for e in evs if e.get("cat") == "kernel" and e.get("dur") is not None), key=lambda e: e["ts"])
        arg_keys = sorted({k for e in kern[:2000] for k in (e.get("args") or {})})
        by_launch, eager = {}, {"fi": {}, "cutlass_moe": {}}
        for e in kern:
            a = e.get("args") or {}
            c = a.get("correlation")
            nm = e.get("name", "")
            if c in glaunch:
                by_launch.setdefault(c, []).append(nm)
            else:
                key = "fi" if FI_RE.search(nm) else ("cutlass_moe" if CUTLASS_MOE_RE.search(nm) else None)
                if key:
                    d = eager[key]
                    d[short(nm, 110)] = d.get(short(nm, 110), 0) + 1
        launches = []
        for c, names in sorted(by_launch.items(), key=lambda kv: glaunch[kv[0]]):
            fi = [n for n in names if FI_RE.search(n)]
            fc12 = [n for n in fi if re.search(r"fc12", n, re.I)]
            fc1 = [n for n in fi if re.search(r"fc1", n, re.I) and n not in fc12]
            fc2 = [n for n in fi if re.search(r"fc2", n, re.I)]
            route = [n for n in fi if n not in fc1 and n not in fc2 and n not in fc12]
            cut = [n for n in names if CUTLASS_MOE_RE.search(n) and not FI_RE.search(n)]
            launches.append({"kernels": len(names), "fi_fc1": len(fc1), "fi_fc2": len(fc2), "fi_fc12": len(fc12),
                             "fi_route": len(route), "cutlass_moe": len(cut)})
        moe_launches = [x for x in launches if x["fi_fc1"] or x["fi_fc2"] or x["cutlass_moe"] or x["fi_fc12"]]
        names_union = {}
        for names in by_launch.values():
            for n in names:
                names_union[short(n)] = names_union.get(short(n), 0) + 1
        out[r] = {
            "kernels": len(kern), "graph_launches": len(glaunch), "graph_launches_with_kernels": len(by_launch),
            "moe_graph_launches": len(moe_launches),
            "moe_graph_launches_fi_fc1_and_fc2": sum(1 for x in moe_launches if x["fi_fc1"] and x["fi_fc2"]),
            "moe_graph_launches_with_cutlass_moe": sum(1 for x in moe_launches if x["cutlass_moe"]),
            "moe_graph_launches_with_fi_fc12_fused": sum(1 for x in moe_launches if x["fi_fc12"]),
            "kernels_per_graph_launch": sorted({x["kernels"] for x in launches})[:10],
            "graph_kernel_names": dict(sorted(names_union.items(), key=lambda kv: -kv[1])[:40]),
            "eager_moe_kernels": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:12]) for k, v in eager.items()},
            "kernel_arg_keys": arg_keys,
        }
        log(f"rank {r}: " + json.dumps({k: v for k, v in out[r].items() if k not in (
            "graph_kernel_names", "eager_moe_kernels", "kernel_arg_keys")}))
        del evs, kern
    return out


def scan_logs(out_root, info):
    ready = info.get("ready_at") or float("inf")
    sizes = info.get("log_sizes_at_ready") or {}
    logs = sorted(glob.glob(str(out_root / "**" / "*.log"), recursive=True))
    logs = [f for f in logs if "/jitwatch/" not in f]
    warm, hits = [], []
    for f in logs:
        try:
            data = Path(f).read_text(errors="replace")
        except OSError:
            continue
        cut = sizes.get(f, 0)
        off = 0
        for line in data.splitlines(keepends=True):
            if WARM_RE.search(line):
                warm.append({"file": os.path.basename(f), "before_ready": off < cut, "line": line.strip()[-600:]})
            if LOG_RE.search(line) and len(hits) < 80:
                hits.append(f"{os.path.basename(f)}@{'pre' if off < cut else 'post'}: {line.strip()[-300:]}")
            off += len(line.encode(errors="replace"))
    events, procs = [], {}
    for f in sorted(glob.glob(str(out_root / "jitwatch" / "*.log"))):
        pid = Path(f).stem
        p = procs.setdefault(pid, {"argv": "", "patched": [], "pre": 0, "post": 0})
        for line in Path(f).read_text(errors="replace").splitlines():
            parts = line.split(" ", 2)
            if len(parts) < 2:
                continue
            try:
                ts = float(parts[0])
            except ValueError:
                continue
            kind, detail = parts[1], parts[2] if len(parts) > 2 else ""
            if kind == "hook":
                p["argv"] = detail[:120]
            elif kind == "patched":
                p["patched"].append(detail)
            elif kind.endswith("compile"):
                post = ts > ready
                p["post" if post else "pre"] += 1
                if post:
                    events.append({"pid": pid, "t_after_ready_s": round(ts - ready, 1), "kind": kind,
                                   "detail": detail[:200]})
    return {"warm_lines": warm, "log_hits": hits, "jit_post_ready": events, "jit_processes": procs}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    arms = (sys.argv[2] if len(sys.argv) > 2 else "overlap_default,serial_100000").split(",")
    ART.mkdir(parents=True, exist_ok=True)
    head = {"arms": arms, "lib": lib_info(), "engine_json": (Path.cwd() / "experiments" / "ascent-config" /
                                                              "engine.json").read_text().strip(),
            "warm_list": warm_list()}
    log("setup " + json.dumps(head))
    summary, ok_all = {}, True
    for arm in arms:
        log(f"===== arm {arm} =====")
        info = run_arm(arm, ckpt)
        out_root = ART / arm
        files = sorted(glob.glob(str(out_root / "profile" / "traces" / "prefill" / "*.json*")))
        red = reduce_prefill(files)
        logs = scan_logs(out_root, info)
        a_ok = bool(logs["warm_lines"]) and all(w["before_ready"] for w in logs["warm_lines"])
        b_ok = bool(red) and all(
            isinstance(v, dict) and not v.get("error") and v["moe_graph_launches"] > 0
            and v["moe_graph_launches_fi_fc1_and_fc2"] == v["moe_graph_launches"]
            and v["moe_graph_launches_with_cutlass_moe"] == 0 for v in red.values())
        workers = [p for p in logs["jit_processes"].values() if "cutlass.cute" in p["patched"]]
        c_ok = not logs["jit_post_ready"] and len(workers) >= 4
        res = {"ready_at": info.get("ready_at"), "prefill_trace_files": len(files),
               "a_warm_line_before_ready": a_ok, "b_tail_graph_moe_is_fi": b_ok, "c_no_jit_after_ready": c_ok,
               "ranks": red, "warm_lines": logs["warm_lines"][:8], "log_hits": logs["log_hits"],
               "jit_post_ready_events": logs["jit_post_ready"][:40], "jit_post_ready_count": len(logs["jit_post_ready"]),
               "jit_processes": logs["jit_processes"], "worker_processes_with_cute_patched": len(workers)}
        summary[arm] = res
        ok_all &= a_ok and b_ok and c_ok
        (ART / "timeline_summary.json").write_text(json.dumps({"head": head, "arms": summary}, indent=1, default=str))
        log(f"[{arm}] a={a_ok} b={b_ok} c={c_ok} post-ready JIT={len(logs['jit_post_ready'])} "
            f"workers patched={len(workers)}")
    print(json.dumps({"passed": bool(ok_all), "head": {k: head[k] for k in ("lib", "engine_json", "warm_list")},
                      "arms": {a: {k: v for k, v in r.items() if k in (
                          "a_warm_line_before_ready", "b_tail_graph_moe_is_fi", "c_no_jit_after_ready",
                          "jit_post_ready_count", "worker_processes_with_cute_patched", "prefill_trace_files")}
                          | {"ranks": {rk: {kk: vv for kk, vv in rv.items() if kk.startswith("moe_graph")
                                            or kk == "graph_launches"} for rk, rv in r["ranks"].items()
                                       if isinstance(rv, dict)}}
                          for a, r in summary.items()}}, default=str))


if __name__ == "__main__":
    main()
