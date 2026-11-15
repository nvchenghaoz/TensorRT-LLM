"""TEMPORARY analysis instrumentation (branch 15, idea 21). Not for landing.

Enabled only when ASCENT_TRACE_DIR is set. Every process appends JSON lines to
$ASCENT_TRACE_DIR/trace_<pid>.jsonl from a daemon thread (no I/O on the hot
path). Host stamps are time.perf_counter() (CLOCK_MONOTONIC on Linux, shared by
all processes on the host). GPU stamps: CUDA timing events recorded in program
order; the flusher thread resolves each one (query only, never a sync) to an
accumulated GPU time "g" (ms, float64 sum of consecutive elapsed_time) so the
analysis can map GPU time to host time through anchors (host stamp taken just
before an anchor event is recorded on an idle stream; min offset per window).
ASCENT_NVTX=1 additionally pushes NVTX ranges for the stages.
"""
import collections
import json
import os
import threading
import time

_DIR = os.environ.get("ASCENT_TRACE_DIR") or None
ENABLED = _DIR is not None
NVTX = ENABLED and os.environ.get("ASCENT_NVTX", "0") == "1"
now = time.perf_counter

_buf = collections.deque()
_gpu_pending = collections.deque()
_started_pid = [None]
_lock = threading.Lock()
_gpu_state = {"prev": None, "g": 0.0, "seq": 0}


def _ensure_thread():
    pid = os.getpid()
    if _started_pid[0] == pid:
        return
    with _lock:
        if _started_pid[0] == pid:
            return
        _started_pid[0] = pid
        threading.Thread(target=_flusher, name="ascent_trace_flusher", daemon=True).start()


def ev(name, **kw):
    if not ENABLED:
        return
    kw["n"] = name
    if "t" not in kw:
        kw["t"] = now()
    _buf.append(kw)
    _ensure_thread()


def gpu_ev(name, stream=None, **kw):
    """Record a timing event on `stream` (default current); resolved async."""
    if not ENABLED:
        return None
    import torch
    e = torch.cuda.Event(enable_timing=True)
    t = now()
    if stream is None:
        e.record()
    else:
        e.record(stream)
    kw["n"] = name
    kw["t"] = t
    _gpu_pending.append((e, kw))
    _ensure_thread()
    return e


def clk_mark():
    """NVTX mark carrying perf_counter, to align nsys time with our stamps."""
    if NVTX:
        import torch
        torch.cuda.nvtx.mark("AT:clk %.9f" % now())


def rpush(name):
    if NVTX:
        import torch
        torch.cuda.nvtx.range_push(name)


def rpop():
    if NVTX:
        import torch
        torch.cuda.nvtx.range_pop()


def _resolve_gpu(out):
    st = _gpu_state
    while _gpu_pending:
        e, kw = _gpu_pending[0]
        try:
            if not e.query():
                break
        except Exception:  # noqa: BLE001
            break
        _gpu_pending.popleft()
        prev = st["prev"]
        if prev is not None:
            try:
                st["g"] += float(prev.elapsed_time(e))
            except Exception as ex:  # noqa: BLE001
                kw["err"] = str(ex)[:80]
        st["prev"] = e
        kw["g"] = st["g"]
        kw["gs"] = st["seq"]
        st["seq"] += 1
        out.append(kw)


def _flusher():
    path = os.path.join(_DIR, f"trace_{os.getpid()}.jsonl")
    with open(path, "a") as f:
        while True:
            time.sleep(0.25)
            out = []
            while _buf:
                out.append(_buf.popleft())
            if _gpu_pending:
                _resolve_gpu(out)
            if out:
                f.write("".join(json.dumps(r, default=str) + "\n" for r in out))
                f.flush()
