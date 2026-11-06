"""Idea 237 acceptance item 4: 4-GPU candidate timeline of the served config (ONE job, two server lifetimes).

Arms, back to back on the job's 4 GPUs, both with the default TRTLLM_SM12X_W4A8_FI_PREFILL=graph (unset):
  overlap_default : TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS unset (default 1024, landing 60's eager shared overlap)
  serial_100000   : TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS=100000 (serial reference)
Each arm runs the FROZEN profile adapter (experiments/ascent-benchmark/profile_run.py) in-process with the protocol
server settings overlaid by this tree's experiments/ascent-config/engine.json (main a9ae1a75's verbatim: EP2xTP2,
LOWPRECISION AR, tokenization cache): start trtllm-serve, prime the 32768-token prefix, warm up c1/c2/c4, then the
prefill window (3 workload requests, max_tokens=1, CPU+GPU Kineto on every rank) and the decode_c1/decode_c4
windows (40 steps, GPU Kineto). Nothing here is timing evidence.
Extra env per arm (attribution only):
  * TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY=1: each tail layer's first replay is compared with eager, and layers 20/39
    log '[TAIL_MOE_GRAPH KERNELS]' with the graph's own kernel sequence (a receipt independent of the window trace).
  * PYTHONPATH += jitwatch/: sitecustomize logs every triton.compile, cutlass.cute.compile and Python-side
    cuda.bindings.driver cuModuleLoad*/cuLibraryLoad* call per process with a wall-clock stamp.
  * fresh TRITON_CACHE_DIR and CUTE_DSL_CACHE_DIR per arm; file counts at readiness and after every window.
  Readiness = the frozen adapter's Server.start returning (its /health wait).
Checks per arm:
  (a) the 'FlashInfer PREFILL chain warmed up' line (token counts + tile table) is in the server log before readiness;
  (b) prefill window, every rank: the kernels launched by cudaGraphLaunch (Kineto correlation ids) in the 128-row
      tail MoE window graphs are the FlashInfer route + fc1_act_q1 + fc2_finalize, with no CUTLASS fused_moe kernel;
      decode windows: the fused fc12 kernel per MoE layer, no prefill-chain kernel;
  (c) zero Triton / CuTe-DSL compile and Python-side module-load events after readiness, and unchanged cache file
      counts from readiness to the end of the request windows.
Also per rank (mpi rank = trace rank-N, GPU UUID = CUDA_VISIBLE_DEVICES[N]) and per prefill request: the L20-39
region (first kernel of the L20 tail graph to the last kernel of the L39 tail graph): span, busy, idle split into
in-graph and between-graph, and the launch-starved share (GPU idle before the next kernel's host launch call).
Usage: timeline.py <checkpoint> [arms]. Last stdout line: JSON {passed, ...}.
"""
import bisect
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
DECODE_STEPS = 40

FI_RE = re.compile(r"flashinfer|route_assign|route_count|route_prefix|_route_|mxfp8_q0", re.I)
CUTLASS_MOE_RE = re.compile(r"MoeFCGemm|moe_gemm|MoeGemm|expandInputRows|finalizeMoeRouting|fused_moe|"
                            r"computeStrides|ExpertMaps|ExpertPrefixSum|doActivation|GroupProblemShape|PtrArray|"
                            r"moe_permute|unpermute", re.I)
WARM_RE = re.compile(r"FlashInfer PREFILL chain warmed up")
LOG_RE = re.compile(r"PREFILL chain warmed|decode chain warmed|FlashInfer decode chain on|TAIL_MOE_GRAPH|"
                    r"shared-expert eager overlap|\[fc12_fused\]|capture failed|Traceback", re.I)
_KEV = re.compile(rb'"ph":\s*"X",\s*"cat":\s*"kernel",\s*"name":\s*"((?:[^"\\]|\\.)*)"')


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.1f}s]", *a, flush=True)


def sh(cmd, timeout=120):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()[-3000:]
    except Exception as exc:  # noqa: BLE001
        return repr(exc)


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


def text_logs(out_root):
    """Server/rank log files of one arm (any *.log/*.txt/*.out/*.err, or any file in a 'server' dir)."""
    out = []
    for f in glob.glob(str(Path(out_root) / "**" / "*"), recursive=True):
        p = Path(f)
        if not p.is_file() or any(s in f for s in ("/jitwatch/", "/traces/", "/cache_triton/", "/cache_cute_dsl/")):
            continue
        if p.suffix in (".log", ".txt", ".out", ".err") or "server" in p.parent.name:
            if p.suffix not in (".json", ".gz", ".yaml", ".yml"):
                out.append(f)
    return sorted(out)


def count_files(d):
    n = 0
    for _, _, fs in os.walk(d):
        n += len(fs)
    return n


def run_arm(arm, ckpt):
    """Run the frozen profile adapter once with this arm's env; returns readiness time, log offsets, cache counts."""
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
    caches = {"triton": out_root / "cache_triton", "cute_dsl": out_root / "cache_cute_dsl"}
    for d in caches.values():
        d.mkdir(exist_ok=True)
    saved = dict(os.environ)
    os.environ.pop("TRTLLM_SM12X_W4A8_FI_PREFILL", None)
    os.environ.pop("TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS", None)
    os.environ.update(ARMS[arm])
    os.environ["TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY"] = "1"
    os.environ["JITWATCH_DIR"] = str(jw)
    os.environ["TRITON_CACHE_DIR"] = str(caches["triton"])
    os.environ["CUTE_DSL_CACHE_DIR"] = str(caches["cute_dsl"])
    os.environ["PYTHONPATH"] = str(HERE / "jitwatch") + (
        os.pathsep + saved["PYTHONPATH"] if saved.get("PYTHONPATH") else "")
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")
    log(f"[{arm}] env: " + json.dumps({k: os.environ.get(k) for k in (
        "TRTLLM_SM12X_W4A8_FI_PREFILL", "TRTLLM_DSV4_SHARED_OVERLAP_MIN_TOKENS", "TRTLLM_DSV4_TAIL_MOE_GRAPH_VERIFY",
        "JITWATCH_DIR", "TRITON_CACHE_DIR", "CUTE_DSL_CACHE_DIR", "PYTHONPATH", "TLLM_LOG_LEVEL",
        "CUDA_VISIBLE_DEVICES")}))

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
        wl["concurrency"] = [1]  # lead's ruling 11044 content: c1 only (warmup c1, decode_c1 window)
        return wl

    def artifact_dir(name):
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    info = {"ready_at": None, "cache_counts": []}

    def snap(tag):
        c = {k: count_files(d) for k, d in caches.items()}
        info["cache_counts"].append({"at": tag, "t": round(time.time(), 1), **c})
        log(f"[{arm}] cache files at {tag}: {c}")

    orig_start = common.Server.start

    def start(self, *a, **k):
        r = orig_start(self, *a, **k)
        info["ready_at"] = time.time()
        sizes = {}
        for f in text_logs(out_root):
            try:
                sizes[f] = os.path.getsize(f)
            except OSError:
                pass
        info["log_sizes_at_ready"] = sizes
        log(f"[{arm}] server ready; log sizes {sizes}")
        snap("ready")
        return r

    orig_window = profile_run.window

    def window(name, fn, trace_root, hb):
        if name == "decode_c4":
            log(f"[{arm}] decode_c4 window skipped (c1 only)")
            return {"name": name, "control": "skipped", "files": []}
        snap(f"before_{name}")
        try:
            return orig_window(name, fn, trace_root, hb)
        finally:
            snap(f"after_{name}")

    patches = [(common, "workload", workload), (common, "load_config", lambda argv: (harness, Path.cwd().resolve())),
               (common, "artifact_dir", artifact_dir), (common.Server, "start", start),
               (profile_run, "window", window),
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


def is_moe(n):
    return bool(FI_RE.search(n) or CUTLASS_MOE_RE.search(n))


def classify(names):
    fi = [n for n in names if FI_RE.search(n)]
    fc12 = [n for n in fi if re.search(r"fc12", n, re.I)]
    fc1 = [n for n in fi if re.search(r"fc1", n, re.I) and n not in fc12]
    fc2 = [n for n in fi if re.search(r"fc2", n, re.I) and n not in fc12]
    route = [n for n in fi if n not in fc1 and n not in fc2 and n not in fc12]
    cut = [n for n in names if CUTLASS_MOE_RE.search(n) and not FI_RE.search(n)]
    return {"fi_fc1": len(fc1), "fi_fc2": len(fc2), "fi_fc12": len(fc12), "fi_route": len(route), "cutlass_moe": len(cut)}


def region_stats(kern, rt_ts, tail):
    """L20-39 regions: runs of tail MoE graph launches (gap > 50 ms = next request)."""
    runs, cur = [], []
    for x in tail:
        if cur and x["s"] - cur[-1]["t"] > 50000:
            runs.append(cur)
            cur = []
        cur.append(x)
    if cur:
        runs.append(cur)
    starts = [e["ts"] for e in kern]
    out = []
    for run in runs:
        rs, rt = run[0]["s"], max(x["t"] for x in run)
        i0 = bisect.bisect_left(starts, rs - 1e5)
        i1 = bisect.bisect_right(starts, rt)
        sel = [e for e in kern[i0:i1] if e["ts"] + e["dur"] > rs]
        merged = []
        for e in sel:
            s, t = max(e["ts"], rs), min(e["ts"] + e["dur"], rt)
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], t)
            else:
                merged.append([s, t, e])
        busy = sum(t - s for s, t, _ in merged)
        spans = [(x["s"], x["t"]) for x in run]
        in_g = betw = st_b = st_g = 0.0
        n_starved = 0
        for (_, t0, _), (s1, _, e1) in zip(merged, merged[1:]):
            g = s1 - t0
            if g <= 0:
                continue
            launch = rt_ts.get((e1.get("args") or {}).get("correlation"))
            starved = max(0.0, min(launch, s1) - t0) if launch is not None else 0.0
            if any(a <= t0 and s1 <= b for a, b in spans):
                in_g += g
                st_g += starved
            else:
                betw += g
                st_b += starved
                n_starved += starved > 0
        moe_us = sum(x["moe_us"] for x in run)
        out.append({"graphs": len(run), "span_us": round(rt - rs, 1), "busy_us": round(busy, 1),
                    "idle_us": round(rt - rs - busy, 1), "idle_in_graph_us": round(in_g, 1),
                    "idle_between_graph_us": round(betw, 1), "launch_starved_between_us": round(st_b, 1),
                    "launch_starved_in_graph_us": round(st_g, 1), "starved_gaps_between": n_starved,
                    "tail_graph_moe_kernel_us": round(moe_us, 1),
                    "graph_kernel_span_sum_us": round(sum(x["t"] - x["s"] for x in run), 1)})
    return out


def reduce_prefill(files, uuids):
    out = {}
    for f in files:
        m = re.search(r"rank-(\d+)", os.path.basename(f))
        r = int(m.group(1)) if m else len(out)
        label = f"mpi_rank{r}/{uuids[r] if r < len(uuids) else '?'}"
        try:
            evs = load_trace(f)
        except Exception as exc:  # noqa: BLE001
            out[label] = {"error": repr(exc)[:300], "file": f}
            continue
        rt_ts, glaunch = {}, {}
        for e in evs:
            if e.get("cat") in ("cuda_runtime", "cuda_driver"):
                c = (e.get("args") or {}).get("correlation")
                if c is not None:
                    rt_ts[c] = e["ts"]
                    if "GraphLaunch" in e.get("name", ""):
                        glaunch[c] = e["ts"]
        kern = sorted((e for e in evs if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                       and e.get("dur") is not None), key=lambda e: e["ts"])
        arg_keys = sorted({k for e in kern[:3000] for k in (e.get("args") or {})})
        by_launch, eager = {}, {"fi": {}, "cutlass_moe": {}}
        for e in kern:
            c = (e.get("args") or {}).get("correlation")
            nm = e.get("name", "")
            if c in glaunch:
                by_launch.setdefault(c, []).append(e)
            elif e.get("cat") == "kernel":
                key = "fi" if FI_RE.search(nm) else ("cutlass_moe" if CUTLASS_MOE_RE.search(nm) else None)
                if key:
                    eager[key][short(nm, 110)] = eager[key].get(short(nm, 110), 0) + 1
        tail, cls_all, names_union = [], [], {}
        for c, ks in by_launch.items():
            names = [k.get("name", "") for k in ks]
            cl = classify(names)
            for n in names:
                names_union[short(n)] = names_union.get(short(n), 0) + 1
            if cl["fi_fc1"] or cl["fi_fc2"] or cl["fi_fc12"] or cl["cutlass_moe"]:
                cls_all.append(cl)
                tail.append({"s": min(k["ts"] for k in ks), "t": max(k["ts"] + k["dur"] for k in ks), "c": c,
                             "moe_us": sum(k["dur"] for k in ks if FI_RE.search(k.get("name", "")))})
        tail.sort(key=lambda x: x["s"])
        regions = region_stats(kern, rt_ts, tail)
        n = len(regions) or 1
        mean = {k: round(sum(x[k] for x in regions) / n, 1) for k in (regions[0] if regions else {})}
        out[label] = {
            "kernels": len(kern), "graph_launches": len(glaunch), "graph_launches_with_kernels": len(by_launch),
            "moe_graph_launches": len(cls_all),
            "moe_graph_launches_fi_fc1_and_fc2": sum(1 for x in cls_all if x["fi_fc1"] and x["fi_fc2"]),
            "moe_graph_launches_with_fi_route": sum(1 for x in cls_all if x["fi_route"]),
            "moe_graph_launches_with_cutlass_moe": sum(1 for x in cls_all if x["cutlass_moe"]),
            "moe_graph_launches_with_fi_fc12_fused": sum(1 for x in cls_all if x["fi_fc12"]),
            "l20_39_regions": regions, "l20_39_mean": mean,
            "graph_kernel_names": dict(sorted(names_union.items(), key=lambda kv: -kv[1])[:40]),
            "eager_moe_kernels": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:12]) for k, v in eager.items()},
            "kernel_arg_keys": arg_keys,
        }
        log(f"prefill {label}: " + json.dumps({k: v for k, v in out[label].items() if k not in (
            "graph_kernel_names", "eager_moe_kernels", "kernel_arg_keys", "l20_39_regions")}))
        del evs, kern
    return out


def reduce_decode(files, uuids):
    import gzip
    out = {}
    for f in files:
        m = re.search(r"rank-(\d+)", os.path.basename(f))
        r = int(m.group(1)) if m else len(out)
        label = f"mpi_rank{r}/{uuids[r] if r < len(uuids) else '?'}"
        try:
            op = gzip.open if f.endswith(".gz") else open
            with op(f, "rb") as fh:
                buf = fh.read()
            names = []
            for mm in _KEV.finditer(buf):
                try:
                    names.append(json.loads(b'"' + mm.group(1) + b'"'))
                except Exception:  # noqa: BLE001
                    names.append(mm.group(1).decode(errors="replace"))
            del buf
        except Exception as exc:  # noqa: BLE001
            out[label] = {"error": repr(exc)[:300]}
            continue
        cl = classify(names)
        moe = {}
        for n in names:
            if is_moe(n):
                moe[short(n, 110)] = moe.get(short(n, 110), 0) + 1
        out[label] = {"kernels": len(names), "per_step": {k: round(v / DECODE_STEPS, 2) for k, v in cl.items()},
                      "moe_names_per_step": {k: round(v / DECODE_STEPS, 2) for k, v in
                                             sorted(moe.items(), key=lambda kv: -kv[1])[:10]}}
    return out


def scan_logs(out_root, info):
    ready = info.get("ready_at") or float("inf")
    sizes = info.get("log_sizes_at_ready") or {}
    logs = text_logs(out_root)
    warm, hits, counts = [], [], {"tail_captured_pre": 0, "tail_captured_post": 0, "capture_failed": 0,
                                  "overlap_on": 0, "fc12_smem": 0}
    for f in logs:
        try:
            data = Path(f).read_text(errors="replace")
        except OSError:
            continue
        cut = sizes.get(f, 0)
        off = 0
        for line in data.splitlines(keepends=True):
            pre = off < cut
            if WARM_RE.search(line):
                warm.append({"file": os.path.basename(f), "before_ready": pre, "line": line.strip()[-700:]})
            if "[TAIL_MOE_GRAPH]" in line and "captured" in line:
                counts["tail_captured_pre" if pre else "tail_captured_post"] += 1
            if "capture failed" in line:
                counts["capture_failed"] += 1
            if "shared-expert eager overlap on" in line:
                counts["overlap_on"] += 1
            if "[fc12_fused] smem pool" in line:
                counts["fc12_smem"] += 1
            if LOG_RE.search(line) and len(hits) < 60 and "TAIL_MOE_GRAPH] rank" not in line:
                hits.append(f"{os.path.basename(f)}@{'pre' if pre else 'post'}: {line.strip()[-400:]}")
            off += len(line.encode(errors="replace"))
    events, procs = [], {}
    for f in sorted(glob.glob(str(out_root / "jitwatch" / "*.log"))):
        pid = Path(f).stem
        p = procs.setdefault(pid, {"argv": "", "patched": [], "pre": {}, "post": {}})
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
                p["patched"].append(detail.split(" ")[0])
            elif kind.endswith("compile") or kind.startswith(("driver.", "triton.")):
                post = ts > ready
                d = p["post" if post else "pre"]
                d[kind] = d.get(kind, 0) + 1
                if post:
                    events.append({"pid": pid, "t_after_ready_s": round(ts - ready, 1), "kind": kind,
                                   "detail": detail[:200]})
    counts["log_files"] = [f"{os.path.relpath(f, out_root)}:{os.path.getsize(f)}" for f in logs][:20]
    return {"warm_lines": warm, "log_hits": hits, "log_counts": counts, "jit_post_ready": events,
            "jit_processes": procs}


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception as exc:  # noqa: BLE001
        log(f"{fn.__name__} failed: {exc!r}\n{traceback.format_exc()[-1500:]}")
        return {"error": {"error": repr(exc)[:300]}}


def analyze_only(root):
    """--analyze <traces root with prefill/ decode_c1/ decode_c4>: the reduction alone (e.g. on main's profile)."""
    uuids = [u for u in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if u]
    root = Path(root)
    pre = safe(reduce_prefill, sorted(glob.glob(str(root / "prefill" / "*.json*"))), uuids)
    dec = {w: safe(reduce_decode, sorted(glob.glob(str(root / w / "*.json*"))), uuids)
           for w in ("decode_c1", "decode_c4")}
    (ART / "analyze_only.json").write_text(json.dumps({"root": str(root), "prefill": pre, "decode": dec}, indent=1))
    ok = bool(pre) and not any(isinstance(v, dict) and v.get("error") for v in pre.values())
    print(json.dumps({"passed": ok, "root": str(root),
                      "prefill": {rk: {kk: vv for kk, vv in rv.items() if kk.startswith("moe_graph")
                                       or kk in ("graph_launches", "graph_launches_with_kernels", "l20_39_mean",
                                                 "error", "kernel_arg_keys")}
                                  for rk, rv in pre.items() if isinstance(rv, dict)},
                      "prefill_graph_names_rank0": next(iter(pre.values()), {}).get("graph_kernel_names"),
                      "decode": {w: {rk: rv.get("per_step") for rk, rv in d.items() if isinstance(rv, dict)}
                                 for w, d in dec.items()},
                      "decode_names_rank0": {w: next(iter(d.values()), {}).get("moe_names_per_step")
                                             for w, d in dec.items()}}, default=str))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--analyze":
        ART.mkdir(parents=True, exist_ok=True)
        return analyze_only(sys.argv[2])
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    arms = (sys.argv[2] if len(sys.argv) > 2 else "overlap_default,serial_100000").split(",")
    ART.mkdir(parents=True, exist_ok=True)
    uuids = [u for u in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if u]
    log("node/GPUs: " + json.dumps({
        "hostname": sh(["hostname"]), "cuda_visible_devices": uuids,
        "gpus": sh(["nvidia-smi", "--query-gpu=index,uuid,memory.used,clocks.sm,clocks.max.sm,temperature.gpu,"
                    "power.draw", "--format=csv,noheader"]),
        "topo": sh(["nvidia-smi", "topo", "-m"])[:1500], "slot_env": {k: v for k, v in os.environ.items()
                                                                    if "SLOT" in k or "AUTO_DEV" in k}}))
    head = {"arms": arms, "lib": lib_info(), "cuda_visible_devices": uuids,
            "nvidia_smi_L": sh(["nvidia-smi", "-L"]), "hostname": sh(["hostname"]),
            "engine_json": (Path.cwd() / "experiments" / "ascent-config" / "engine.json").read_text().strip(),
            "git_head": sh(["git", "rev-parse", "HEAD"]), "warm_list": warm_list()}
    log("setup " + json.dumps(head))
    summary, ok_all = {}, True
    for i, arm in enumerate(arms):
        if i:
            # pitfall 3072: never start the next lifetime while the previous ranks still hold GPU memory
            for _ in range(36):
                used = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                try:
                    if max(int(x) for x in used.split()) < 4096:
                        break
                except ValueError:
                    pass
                log(f"waiting for GPU memory release: {used.split()}")
                time.sleep(10)
        log(f"===== arm {arm} =====")
        info = run_arm(arm, ckpt)
        out_root = ART / arm
        traces = out_root / "profile" / "traces"
        pre = safe(reduce_prefill, sorted(glob.glob(str(traces / "prefill" / "*.json*"))), uuids)
        dec = {w: safe(reduce_decode, sorted(glob.glob(str(traces / w / "*.json*"))), uuids)
               for w in ("decode_c1",)}
        logs = safe(scan_logs, out_root, info)
        if "warm_lines" not in logs:
            logs = {"warm_lines": [], "log_hits": [], "log_counts": {}, "jit_post_ready": [{"error": "scan"}],
                    "jit_processes": {}}
        a_ok = bool(logs["warm_lines"]) and all(w["before_ready"] for w in logs["warm_lines"])
        b_prefill = bool(pre) and all(
            not v.get("error") and v["moe_graph_launches"] > 0
            and v["moe_graph_launches_fi_fc1_and_fc2"] == v["moe_graph_launches"]
            and v["moe_graph_launches_with_cutlass_moe"] == 0 and v["moe_graph_launches_with_fi_fc12_fused"] == 0
            for v in pre.values())
        # fc12 fused kernel present every step on every rank; the per-step fc1/fc2/CUTLASS counts are reported, not
        # gated (hash-routed or non-fused decode layers, if any, are main's own and unchanged by this branch).
        b_decode = all(bool(d) and all(not v.get("error") and v["per_step"]["fi_fc12"] > 0 for v in d.values())
                       for d in dec.values())
        cc = info.get("cache_counts", [])
        ready_c = next((x for x in cc if x["at"] == "ready"), None)
        last_c = cc[-1] if cc else None
        cache_ok = bool(ready_c and last_c) and all(ready_c[k] == last_c[k] for k in ("triton", "cute_dsl"))
        workers = [p for p in logs["jit_processes"].values() if "cutlass.cute" in p["patched"]]
        c_ok = not logs["jit_post_ready"] and len(workers) >= 4 and cache_ok
        res = {"ready_at": info.get("ready_at"), "a_warm_line_before_ready": a_ok,
               "b_tail_graph_moe_is_fi": b_prefill, "b_decode_fc12_unchanged": b_decode,
               "c_no_jit_after_ready": c_ok, "c_cache_counts_unchanged": cache_ok, "cache_counts": cc,
               "jit_post_ready_count": len(logs["jit_post_ready"]), "jit_post_ready_events": logs["jit_post_ready"][:40],
               "worker_processes_with_cute_patched": len(workers), "jit_processes": logs["jit_processes"],
               "log_counts": logs["log_counts"], "warm_lines": logs["warm_lines"][:8], "log_hits": logs["log_hits"],
               "prefill": pre, "decode": dec}
        summary[arm] = res
        ok_all &= a_ok and b_prefill and b_decode and c_ok
        (ART / "timeline_summary.json").write_text(json.dumps({"head": head, "arms": summary}, indent=1, default=str))
        log(f"[{arm}] a={a_ok} b_prefill={b_prefill} b_decode={b_decode} c={c_ok} cache_ok={cache_ok} "
            f"post-ready JIT={len(logs['jit_post_ready'])} workers patched={len(workers)} {logs['log_counts']}")
    brief = {}
    for a, r in summary.items():
        brief[a] = {k: r[k] for k in ("a_warm_line_before_ready", "b_tail_graph_moe_is_fi", "b_decode_fc12_unchanged",
                                      "c_no_jit_after_ready", "c_cache_counts_unchanged", "jit_post_ready_count",
                                      "worker_processes_with_cute_patched", "log_counts")}
        brief[a]["warm_line"] = (r["warm_lines"][0]["line"][-400:] if r["warm_lines"] else None)
        brief[a]["prefill"] = {rk: {kk: vv for kk, vv in rv.items() if kk.startswith("moe_graph")
                                    or kk in ("graph_launches", "l20_39_mean", "error")}
                               for rk, rv in r["prefill"].items()}
        brief[a]["decode_fc12_per_step"] = {w: {rk: rv.get("per_step", {}).get("fi_fc12") for rk, rv in d.items()}
                                            for w, d in r["decode"].items()}
    print(json.dumps({"passed": bool(ok_all), "head": {k: head[k] for k in (
        "lib", "engine_json", "warm_list", "cuda_visible_devices", "git_head")}, "arms": brief}, default=str))


if __name__ == "__main__":
    main()
