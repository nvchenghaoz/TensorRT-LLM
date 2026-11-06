"""Branch 185 (idea 199): host time of CSA2TrtllmMetadata.prepare at the step start (a) and the
decoder bounded-replay entry/exit at the L20 boundary (b), DeepSeek-V4.1-Flash production layout.

One GPU, no model weights: a real CSA2CacheManager (KVCacheManagerV2) built from the checkpoint
config.json layout at the served settings (max_seq_len 36864, tokens_per_block 128,
max_num_tokens 8192, max_batch_size 8, SWA scratch reuse off), with synthetic LlmRequests of
34816 prompt tokens.  The CSA2 metadata is per-rank-independent (one KV head, 512 dims), so the
TP4 rank sees exactly these shapes.

Shapes (eager EXTEND steps, metadata object reused across steps, as the engine does):
  c1: 1 context row  (32768 cached + 2048 new)
  c2: 1 context row + 1 generation row
  c4: 3 context rows + 1 generation row

Modes (argv[2]):
  time    : per shape, WARM warmups then ITERS timed steps; host ms of (a) = prepare_context_replay
            + prepare, (b) entry = plan_decoder_replay + enter_decoder_replay, (b) exit; then a
            cProfile of (a) and of (b) entry (top-15 by tottime and by cumtime).
  digest  : >= 8 alternating c1/c4 steps (metadata reuse); after (a) (+ forward-time SWA slot
            resolution) and after (b) entry, a SHA-256 of every csa2_* tensor / host tuple.  Prints
            the per-step digests; run it in both trees (or with TRTLLM_CSA2_FAST_PREPARE=0/1) and
            compare.

Last stdout line: one JSON object.
"""

import cProfile
import hashlib
import io
import json
import os
import pstats
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CKPT", "")
MODE = sys.argv[2] if len(sys.argv) > 2 else "time"
PREFIX = int(os.environ.get("B185_PREFIX", "32768"))
SUFFIX = int(os.environ.get("B185_SUFFIX", "2048"))
PROMPT = PREFIX + SUFFIX
MAX_SEQ_LEN = int(os.environ.get("B185_MAX_SEQ_LEN", "36864"))
TPB = 128
WARM = int(os.environ.get("B185_WARM", "5"))
ITERS = int(os.environ.get("B185_ITERS", "40"))
PROF_ITERS = int(os.environ.get("B185_PROF_ITERS", "20"))
DIGEST_STEPS = int(os.environ.get("B185_DIGEST_STEPS", "10"))
REUSE = os.environ.get("B185_BLOCK_REUSE", "1") == "1"
SHAPES = {"c1": (1, 0), "c2": (1, 1), "c4": (3, 1)}


def log(*a):
    print(*a, flush=True)


def load_cfg():
    with open(os.path.join(CKPT, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


class Rig:
    def __init__(self):
        from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
        from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
        from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
        from tensorrt_llm.bindings import DataType, SamplingConfig
        from tensorrt_llm.bindings.internal.batch_manager import CacheType
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig
        from tensorrt_llm.mapping import Mapping

        cfg = load_cfg()
        self.layout = CSA2Layout.from_hf_config(cfg)
        num_layers = len(self.layout.compress_ratios)
        self.split = max(self.layout.kv_source_layer_ids)
        self.window = self.layout.window_size
        log(f"layout: layers {num_layers} ratios {self.layout.compress_ratios} kv_sources "
            f"{self.layout.kv_source_layer_ids} window {self.window} split {self.split} reuse {REUSE}")
        self.manager = CSA2CacheManager(
            KvCacheConfig(
                enable_block_reuse=REUSE,
                max_gpu_total_bytes=int(os.environ.get("B185_KV_GB", "24")) << 30,
                enable_swa_scratch_reuse=False,
                dtype="fp8",
            ),
            CacheType.SELFKONLY,
            layout=self.layout,
            num_layers=num_layers,
            tokens_per_block=TPB,
            mapping=Mapping(),
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=8,
            max_input_len=MAX_SEQ_LEN,
            max_num_tokens=8192,
            dtype=DataType.BF16,
            vocab_size=int(cfg.get("vocab_size", 129280)),
            pretrained_config=SimpleNamespace(engram_layer_ids=cfg.get("engram_layer_ids")),
        )
        log(f"manager built: decoder window {self.manager._decoder_replay_window} "
            f"private suffix {self.manager.has_private_swa_suffix(self.split)}")
        self.requests = []
        for i in range(5):
            req = LlmRequest(
                request_id=1000 + i,
                max_new_tokens=256,
                input_tokens=[(7 * j + 13 * i) % 30000 + 5 for j in range(PROMPT)],
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            assert self.manager.prepare_context(req)
            assert self.manager.resize_context(req, req.context_chunk_size)
            self.requests.append(req)
        # Generation rows reserve a few positions past the prompt.
        for req in self.requests[3:]:
            assert self.manager.kv_cache_map[req.py_request_id].resize(PROMPT + 16)
        self.manager._stream.synchronize()
        torch.cuda.synchronize()
        self.metadata = CSA2TrtllmMetadata(
            max_num_requests=8, max_num_tokens=8192, kv_cache_manager=self.manager
        )
        self.gen_step = 0

    def batch(self, shape):
        num_ctx, num_gen = SHAPES[shape]
        ctx = self.requests[:num_ctx]
        gen = self.requests[3 : 3 + num_gen]
        self.gen_step = (self.gen_step + 1) % 8
        starts = [PREFIX] * num_ctx + [PROMPT + self.gen_step] * num_gen
        lengths = [SUFFIX] * num_ctx + [1] * num_gen
        return ctx, gen, starts, lengths

    def step_a(self, shape):
        """Step-start prepare, as DecoderInputs._prepare_attn_metadata calls it."""
        from tensorrt_llm._torch.metadata import KVCacheParams

        ctx, gen, starts, lengths = self.batch(shape)
        md = self.metadata
        t0 = time.perf_counter()
        md.request_ids = [r.py_request_id for r in ctx + gen]
        md.prompt_lens = [PROMPT] * (len(ctx) + len(gen))
        md.seq_lens = torch.tensor(lengths, dtype=torch.int32)
        md.num_contexts = len(ctx)
        md.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(starts))
        md.prepare_context_replay(ctx, allow_final_window=True)
        md.prepare()
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3, ctx + gen

    def step_b(self, requests):
        """L20 boundary: bounded-replay plan + entry (private decoder)."""
        from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
            enter_decoder_replay,
            plan_decoder_replay,
        )

        md = self.metadata
        md.csa2_precomputed_kv_layers = {self.split}
        t0 = time.perf_counter()
        plan = plan_decoder_replay(md, self.window, requests, private_decoder=True, allow_replay=True)
        assert plan is not None and plan.updates_local_metadata
        enter_decoder_replay(md, plan, self.split)
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3, plan

    def exit_b(self, plan):
        from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
            exit_decoder_replay,
        )

        t0 = time.perf_counter()
        exit_decoder_replay(self.metadata, plan)
        self.metadata.csa2_precomputed_kv_layers.clear()
        return (time.perf_counter() - t0) * 1e3


def stats(xs):
    xs = sorted(xs)
    med = statistics.median(xs)
    p10 = xs[int(0.1 * (len(xs) - 1))]
    p90 = xs[int(0.9 * (len(xs) - 1))]
    return {"median_ms": round(med, 4), "p10_ms": round(p10, 4), "p90_ms": round(p90, 4),
            "noise_floor": round((p90 - p10) / med, 4) if med else 0.0, "n": len(xs)}


def top(prof, key, n=15):
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats(key).print_stats(n)
    return s.getvalue()


def run_time(rig):
    result = {}
    for shape in ("c1", "c4", "c2"):
        for _ in range(WARM):
            _, reqs = rig.step_a(shape)
            _, plan = rig.step_b(reqs)
            rig.exit_b(plan)
            torch.cuda.synchronize()
        a, b, e = [], [], []
        for i in range(ITERS):
            ta, reqs = rig.step_a(shape)
            tb, plan = rig.step_b(reqs)
            te = rig.exit_b(plan)
            torch.cuda.synchronize()
            a.append(ta)
            b.append(tb)
            e.append(te)
        result[shape] = {"a_prepare": stats(a), "b_enter": stats(b), "b_exit": stats(e)}
        log(f"SHAPE {shape}: (a) {result[shape]['a_prepare']}  (b) enter {result[shape]['b_enter']}"
            f"  exit {result[shape]['b_exit']}")
    if PROF_ITERS:
        for shape in ("c1", "c4"):
            pa, pb = cProfile.Profile(), cProfile.Profile()
            for _ in range(PROF_ITERS):
                pa.enable()
                _, reqs = rig.step_a(shape)
                pa.disable()
                pb.enable()
                _, plan = rig.step_b(reqs)
                pb.disable()
                rig.exit_b(plan)
                torch.cuda.synchronize()
            log(f"===== cProfile (a) {shape}: {PROF_ITERS} steps, tottime =====")
            log(top(pa, "tottime", 25))
            log(f"===== cProfile (a) {shape}: cumulative =====")
            log(top(pa, "cumulative", 40))
            log(f"===== cProfile (b) enter {shape}: tottime =====")
            log(top(pb, "tottime", 25))
            log(f"===== cProfile (b) enter {shape}: cumulative =====")
            log(top(pb, "cumulative", 40))
    return result


# ----------------------------------------------------------------------------- digest
_SKIP = {
    "_csa2_buffers", "_csa2_eager_latest", "_csa2_host_staging", "_csa2_upload_slab",
    "_csa2_swa_table_cache", "_csa2_swa_descriptors", "_csa2_owner_descriptors",
    "_csa2_prepared_manager", "_csa2_swa_ratio_cache", "_csa2_query_tiles",
    "_csa2_indexer_workspaces", "_csa2_manager_states",
}
_BIG = 64 << 20


def _feed(h, value, depth=0):
    if isinstance(value, torch.Tensor):
        h.update(f"T{tuple(value.shape)}{value.dtype}".encode())
        if value.numel() * value.element_size() > _BIG:
            return  # manager pools (compressor state buffers): identity, not content
        h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
                 if value.numel() else b"")
    elif isinstance(value, np.ndarray):
        h.update(f"N{value.shape}{value.dtype}".encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        h.update(b"{")
        for k in sorted(value, key=repr):
            h.update(repr(k).encode())
            _feed(h, value[k], depth + 1)
        h.update(b"}")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        h.update(f"{type(value).__name__}[".encode())
        for v in items:
            _feed(h, v, depth + 1)
        h.update(b"]")
    elif value is None or isinstance(value, (bool, int, float, str)):
        h.update(repr(value).encode())
    elif hasattr(value, "__dataclass_fields__") and depth < 4:
        h.update(type(value).__name__.encode())
        for name in value.__dataclass_fields__:
            h.update(name.encode())
            _feed(h, getattr(value, name), depth + 1)
    else:
        h.update(type(value).__name__.encode())


def digest(md):
    names = sorted(
        n for n in vars(md)
        if (n.startswith("csa2_") or n.startswith("_csa2_")) and n not in _SKIP
        and not n.startswith("_csa2_fast")
    )
    per = {}
    total = hashlib.sha256()
    for n in names:
        h = hashlib.sha256()
        _feed(h, getattr(md, n))
        per[n] = h.hexdigest()[:16]
        total.update(n.encode())
        total.update(per[n].encode())
    # Generic step tensors the CSA2 consumers read.
    for n in ("kv_lens_cuda", "seq_lens", "prompt_lens"):
        h = hashlib.sha256()
        v = getattr(md, n, None)
        _feed(h, v[: md.num_seqs] if isinstance(v, torch.Tensor) else v)
        per[n] = h.hexdigest()[:16]
        total.update(per[n].encode())
    return total.hexdigest()[:24], per


def run_digest(rig):
    steps = []
    for i in range(DIGEST_STEPS):
        shape = "c1" if i % 2 == 0 else "c4"
        _, reqs = rig.step_a(shape)
        rig.metadata._ensure_swa_slots()
        torch.cuda.synchronize()
        da, pa = digest(rig.metadata)
        _, plan = rig.step_b(reqs)
        rig.metadata._ensure_swa_slots()
        torch.cuda.synchronize()
        db, pb = digest(rig.metadata)
        rig.exit_b(plan)
        torch.cuda.synchronize()
        steps.append({"step": i, "shape": shape, "after_prepare": da, "after_replay_entry": db,
                      "fields_a": len(pa), "fields_b": len(pb)})
        log(f"DIGEST step {i} {shape}: after prepare {da} ({len(pa)} fields), "
            f"after replay entry {db} ({len(pb)} fields)")
        if i < 2:
            log("  fields(a): " + json.dumps(pa, sort_keys=True))
            log("  fields(b): " + json.dumps(pb, sort_keys=True))
    total = hashlib.sha256(json.dumps(steps, sort_keys=True).encode()).hexdigest()[:24]
    return steps, total


def run_diag(rig):
    """Where does (a)'s coalesced-upload copy_ spend its time?"""
    import tensorrt_llm._torch.host_staging as hs
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import metadata as mdmod

    records = []
    orig = hs.copy_host_to_device

    def timed(store, key, destination, value):
        stream = torch.cuda.current_stream()
        idle = stream.query()
        t0 = time.perf_counter()
        slot = (key, value.dtype)
        ring = store.get(slot)
        event = None if ring is None else ring["events"][ring["next"]]
        if event is not None:
            event.synchronize()
        t1 = time.perf_counter()
        orig(store, key, destination, value)
        t2 = time.perf_counter()
        records.append((key, value.numel() * value.element_size(), value.is_pinned(),
                        idle, (t1 - t0) * 1e3, (t2 - t1) * 1e3))

    hs.copy_host_to_device = timed
    orig_packed = getattr(hs, "copy_packed_host_to_device", None)
    if orig_packed is not None:
        def timed_packed(store, key, destination, nbytes, fill):
            idle = torch.cuda.current_stream().query()
            ring = store.get((key, torch.uint8))
            event = None if ring is None else ring["events"][ring["next"]]
            t0 = time.perf_counter()
            if event is not None:
                event.synchronize()
            t1 = time.perf_counter()
            orig_packed(store, key, destination, nbytes, fill)
            t2 = time.perf_counter()
            records.append((f"{key}(packed)", nbytes, True, idle, (t1 - t0) * 1e3, (t2 - t1) * 1e3))

        hs.copy_packed_host_to_device = timed_packed
    orig_flush = mdmod._CoalescedUploads.flush

    def flush(self):
        stream = torch.cuda.current_stream()
        idle = stream.query()
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        orig_flush(self)
        t2 = time.perf_counter()
        records.append(("FLUSH(sync-first)", self._bytes, None, idle, (t1 - t0) * 1e3, (t2 - t1) * 1e3))

    out = {}
    for shape in ("c1", "c4"):
        for i in range(6):
            records.clear()
            if i == 3:
                mdmod._CoalescedUploads.flush = flush
            ta, reqs = rig.step_a(shape)
            ra = list(records)
            records.clear()
            tb, plan = rig.step_b(reqs)
            rb = list(records)
            rig.exit_b(plan)
            torch.cuda.synchronize()
            mdmod._CoalescedUploads.flush = orig_flush
            log(f"DIAG {shape} iter {i}: (a) {ta:.3f} ms  (b) {tb:.3f} ms")
            for name, recs in (("a", ra), ("b", rb)):
                for key, nbytes, pinned, idle, wait_ms, copy_ms in recs:
                    log(f"   {name} {key!s:28} bytes {nbytes:8} pinned {pinned} gpu_idle {idle} "
                        f"wait {wait_ms:.3f} copy {copy_ms:.3f}")
            out[f"{shape}_{i}"] = {"a_ms": ta, "b_ms": tb}
    hs.copy_host_to_device = orig
    if orig_packed is not None:
        hs.copy_packed_host_to_device = orig_packed
    return out


def main():
    torch.manual_seed(0)
    if os.environ.get("B185_THREADS"):
        torch.set_num_threads(int(os.environ["B185_THREADS"]))
    log(f"threads {torch.get_num_threads()} affinity {len(os.sched_getaffinity(0))}")
    log(f"host {os.uname().nodename} gpu {torch.cuda.get_device_name()} mode {MODE} "
        f"fast_prepare env {os.environ.get('TRTLLM_CSA2_FAST_PREPARE', '<unset>')}")
    with torch.inference_mode():
        rig = Rig()
        try:
            if MODE == "time":
                res = run_time(rig)
                out = {"batch": {s: {"step_ms": r["a_prepare"]["median_ms"],
                                     "noise_floor": r["a_prepare"]["noise_floor"]}
                                 for s, r in res.items()},
                       "detail": res, "output": {"class": "exact"}}
            elif MODE == "pair":
                # Digest on the fresh rig first (identical in both trees), then timing.
                steps, total = run_digest(rig)
                log(f"PAIR DIGEST {total}")
                res = run_time(rig)
                out = {"batch": {s: {"step_ms": r["a_prepare"]["median_ms"],
                                     "noise_floor": r["a_prepare"]["noise_floor"]}
                                 for s, r in res.items()},
                       "detail": res, "digest": total,
                       "digest_steps": [(s["after_prepare"], s["after_replay_entry"]) for s in steps],
                       "output": {"class": "exact"}}
            elif MODE == "diag":
                out = {"passed": True, "diag": run_diag(rig)}
            else:
                steps, total = run_digest(rig)
                out = {"passed": True, "digest": total, "steps": steps,
                       "compared": "sha256 of every csa2_* metadata tensor/host tuple after "
                                   "prepare and after the decoder replay entry"}
        finally:
            for req in rig.requests:
                rig.manager.free_resources(req)
            rig.manager.shutdown()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
