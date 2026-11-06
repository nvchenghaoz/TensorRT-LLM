"""Stable-greedy new-token D2H: main stream vs the sampler's private side stream.

Copies py_executor._forward_step's stream ordering with a REAL TorchSampler:
  current(main) stream: event0.record -> execution stream waits on event0
  execution stream: stand-in prep (position add_ + index_select of the sampler's
    new_tokens store) -> CUDA-graph replay of a ~5 ms ELEMENTWISE dummy that also
    writes this step's logits [B, 129280] bf16 -> event1.record
  main stream waits on event1 -> TorchSampler.sample_async (stable-greedy path:
    fused argmax+scatter, then the new-token D2H)
  host (overlap loop): after issuing step k+1, synchronize step k's SamplerEvent
    and read its host tokens.
Metric: CUDA events on the execution stream, end of replay k -> start of replay
k+1 (the between-replay gap), median / p90 over the timed steps.

Usage:
  harness.py bench   -> evidence_format JSON (step_ms = median gap in ms)
  harness.py check   -> untimed correctness: bitwise host tokens vs torch.argmax
                        reference and env-off (main path), poisoned host buffers,
                        batch changes, async-worker and greedy_tokens paths.
"""

import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

V = 129280
NB = 8  # logits bank depth (step k reads bank[k % NB])
BATCHES = [1, 2, 4, 8]
BMAX = 8
MAX_SEQS = 16
WARMUP = 20
ITERS = 220
SENTINEL = 0x7F7F7F7F


def log(*a):
    print(*a, flush=True)


from tensorrt_llm._torch.pyexecutor.sampler import sampler_features as sf  # noqa: E402
from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler  # noqa: E402
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests  # noqa: E402


class _PoisonTorch:
    """Proxy for sampler_features.torch: every pinned host destination it allocates
    (both _copy_to_host and _SideStreamCopier.stage_copy_to_host) starts as 0x7f."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def empty_like(src, *a, **kw):
        t = torch.empty_like(src, *a, **kw)
        if t.device.type == "cpu":
            t.view(torch.uint8).fill_(0x7F)
        return t


def make_sampler(enable_async_worker=False):
    args = TorchSampler.Args(
        max_seq_len=40000,
        max_draft_len=0,
        max_num_sequences=MAX_SEQS,
        max_beam_width=1,
        max_total_draft_tokens=0,
        disable_overlap_scheduler=False,
        enable_async_worker=enable_async_worker,
    )
    s = TorchSampler(args)
    if enable_async_worker:
        s.async_worker_start()
    return s


def make_sched(ids, slots):
    sched = ScheduledRequests()
    for rid, slot in zip(ids, slots):
        sched.generation_requests.append(
            SimpleNamespace(
                py_request_id=rid,
                py_seq_slot=slot,
                is_dummy=False,
                is_attention_dp_dummy=False,
                is_finished=False,
                py_draft_tokens=None,
                draft_tokens=[],
                py_embedding_bias=None,
                py_bad_words=None,
                py_no_repeat_ngram_size=None,
                sampling_config=SimpleNamespace(
                    repetition_penalty=None, presence_penalty=None, frequency_penalty=None
                ),
                py_min_length=None,
                py_return_log_probs=False,
                py_num_logprobs=0,
                py_stop_words_list=None,
            )
        )
    return sched


class Rig:
    def __init__(self, dummy_ms=5.0):
        self.main = torch.cuda.Stream()
        self.exe = torch.cuda.Stream()
        torch.cuda.set_stream(self.main)
        g = torch.Generator(device="cuda").manual_seed(1234)
        self.bank = torch.randn(NB, BMAX, V, device="cuda", generator=g).to(torch.bfloat16)
        self.bank_ref = torch.argmax(self.bank, dim=-1).to(torch.int32).cpu()  # [NB, BMAX]
        self.logits = torch.empty(BMAX, V, device="cuda", dtype=torch.bfloat16)
        self.pos = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.pos_host = 0
        self.x = torch.ones(1 << 24, device="cuda")
        # calibrate the elementwise dummy to ~dummy_ms
        for _ in range(3):
            self.x.mul_(1.0)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20):
            self.x.mul_(1.0)
        b.record()
        b.synchronize()
        per = a.elapsed_time(b) / 20
        self.n_dummy = max(1, int(round(dummy_ms / per)))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.logits.copy_(self.bank.index_select(0, torch.remainder(self.pos, NB))[0])
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(self.n_dummy):
                self.x.mul_(1.0)
            self.logits.copy_(self.bank.index_select(0, torch.remainder(self.pos, NB))[0])
        torch.cuda.synchronize()
        self.ev0 = torch.cuda.Event()
        self.ev1 = torch.cuda.Event()
        log(f"rig: dummy {self.n_dummy} x mul_ ({per * self.n_dummy:.2f} ms)")

    def reset(self):
        """Reset point: rewind the logits-bank position so every sequence sees the
        same logits from step 0 (device counter and host mirror)."""
        torch.cuda.synchronize()
        self.pos.zero_()
        self.pos_host = 0
        torch.cuda.synchronize()

    def forward(self, sampler, B, slots_dev, start_ev, end_ev):
        """_forward_step ordering: event0 on current -> exe waits -> prep + replay -> event1."""
        new_tokens_flat = sampler.store.new_tokens[0, :, 0]
        self.ev0.record(torch.cuda.current_stream())
        self.exe.wait_event(self.ev0)
        with torch.cuda.stream(self.exe):
            self.pos.add_(1)  # stand-in position add_
            _inp = torch.index_select(new_tokens_flat, 0, slots_dev)  # next input ids
            start_ev.record(self.exe)
            self.graph.replay()
            end_ev.record(self.exe)
            self.ev1.record(self.exe)
        torch.cuda.current_stream().wait_event(self.ev1)
        self.pos_host += 1
        return _inp


def run_sequence(rig, sampler, phases, check, greedy_tokens_mode=False):
    """phases: list of (ids, slots, n_steps). Returns (gaps_ms, stats)."""
    total = sum(p[2] for p in phases)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(total)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(total)]
    stats = dict(steps=0, mismatches=0, side_events=0, poison_seen_before_sync=0,
                 poison_after_sync=0, tokens=[])
    prev = None
    k = 0

    def finish(p):
        state, ref, B = p
        # update_requests' contract: synchronize before any host read
        state.sampler_event.synchronize()
        if state.sampler_event.side_stream_event is not None:
            assert state.sampler_event.side_stream_event.query()
        host = state.host.new_tokens
        got = host.tolist()
        stats["tokens"].append(got)
        if any(v == SENTINEL for v in got):
            stats["poison_after_sync"] += 1
        if check and got != ref:
            stats["mismatches"] += 1
            if stats["mismatches"] <= 5:
                log(f"  MISMATCH step got {got} ref {ref}")

    for ids, slots, n in phases:
        B = len(ids)
        sched = make_sched(ids, slots)
        slots_dev = torch.tensor(slots, dtype=torch.int64, device="cuda")
        for _ in range(n):
            rig.forward(sampler, B, slots_dev, starts[k], ends[k])
            ref = rig.bank_ref[rig.pos_host % NB, :B].tolist()
            if greedy_tokens_mode:
                outputs = {"greedy_tokens": rig.bank_ref[rig.pos_host % NB, :B].cuda()}
            else:
                outputs = {"logits": rig.logits[:B]}
            state = sampler.sample_async(sched, outputs, [0])
            if state.sampler_event.side_stream_event is not None:
                stats["side_events"] += 1
            if check:
                # the replay of this step is still running (~5 ms): the poisoned
                # host buffer must not have been written yet
                if all(v == SENTINEL for v in state.host.new_tokens.tolist()):
                    stats["poison_seen_before_sync"] += 1
            if prev is not None:
                finish(prev)
            prev = (state, ref, B)
            stats["steps"] += 1
            k += 1
    finish(prev)
    torch.cuda.synchronize()
    gaps = []
    k = 0
    for ids, slots, n in phases:
        for i in range(n):
            if i + 1 < n:
                gaps.append(ends[k].elapsed_time(starts[k + 1]))
            k += 1
    return gaps, stats


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def bench():
    log(f"nvidia: {torch.cuda.get_device_name()}  env TRTLLM_SAMPLER_SIDE_STREAM_D2H="
        f"{os.environ.get('TRTLLM_SAMPLER_SIDE_STREAM_D2H')}")
    rig = Rig()
    sampler = make_sampler()
    out = {}
    detail = {}
    for B in BATCHES:
        ids = list(range(1000 * B, 1000 * B + B))
        slots = list(range(B))
        # warmup (stable cache miss on first step, hits after)
        run_sequence(rig, sampler, [(ids, slots, WARMUP)], check=False)
        reps = []
        for r in range(3):
            gaps, st = run_sequence(rig, sampler, [(ids, slots, ITERS)], check=False)
            reps.append(gaps)
        allg = [g * 1000.0 for gs in reps for g in gs]  # us
        med = statistics.median(allg)
        p10, p90 = pct(allg, 0.1), pct(allg, 0.9)
        rep_meds = [statistics.median(g) * 1000.0 for g in reps]
        disp = (max(rep_meds) - min(rep_meds)) / med
        out[str(B)] = {"step_ms": med / 1000.0, "noise_floor": max(disp, 0.0)}
        detail[str(B)] = {"gap_median_us": med, "gap_p10_us": p10, "gap_p90_us": p90,
                          "rep_medians_us": rep_meds, "n": len(allg),
                          "side_events_last_rep": st["side_events"]}
        log(f"B={B}: gap median {med:.2f} us p10 {p10:.2f} p90 {p90:.2f} "
            f"rep medians {[round(x, 2) for x in rep_meds]} side_events {st['side_events']}")
    print(json.dumps({"batch": out, "output": {"class": "exact"}, "detail": detail}))


def check():
    sf.torch = _PoisonTorch()
    rig = Rig(dummy_ms=5.0)
    results = {}
    ok = True

    def seq_for(B):
        ids = list(range(1000 * B, 1000 * B + B))
        return [(ids, list(range(B)), WARMUP + ITERS)]

    # batch-change schedule: B=4 -> subset (miss) -> back (miss, then hits)
    change = [
        ([1, 2, 3, 4], [0, 1, 2, 3], 10),
        ([2, 4], [1, 3], 8),
        ([1, 2, 3, 4], [0, 1, 2, 3], 10),
        ([7, 8, 9, 10, 11, 12, 13, 14], [8, 9, 10, 11, 12, 13, 14, 15], 9),
    ]

    toks = {}
    for env in ("1", "0"):
        os.environ["TRTLLM_SAMPLER_SIDE_STREAM_D2H"] = env
        sampler = make_sampler()
        log(f"env={env}: side-stream flag on sampler = "
            f"{getattr(sampler, '_side_stream_new_tokens_d2h', None)}")
        for name, phases in [(f"B{B}", seq_for(B)) for B in BATCHES] + [("batch_change", change)]:
            rig.reset()
            _, st = run_sequence(rig, sampler, phases, check=True)
            toks[(env, name)] = st.pop("tokens")
            want_side = st["steps"] if env == "1" else 0
            passed = (st["mismatches"] == 0 and st["poison_after_sync"] == 0
                      and st["side_events"] == want_side)
            ok &= passed
            results[f"env{env}_{name}"] = dict(st, passed=passed)
            log(f"env={env} {name}: {st} passed={passed}")
    for name in [f"B{B}" for B in BATCHES] + ["batch_change"]:
        same = toks[("1", name)] == toks[("0", name)]
        results[f"on_vs_off_bitwise_{name}"] = same
        ok &= same
        log(f"on vs off (main path) bitwise {name}: {same}")

    # Paths that must keep main's copy (no side-stream event).
    os.environ["TRTLLM_SAMPLER_SIDE_STREAM_D2H"] = "1"
    sampler = make_sampler(enable_async_worker=True)
    _, st = run_sequence(rig, sampler, [([1, 2, 3, 4], [0, 1, 2, 3], 12)], check=True)
    st.pop("tokens")
    passed = st["mismatches"] == 0 and st["side_events"] == 0 and st["poison_after_sync"] == 0
    ok &= passed
    results["async_worker_keeps_main_copy"] = dict(st, passed=passed)
    log(f"async worker: {st} passed={passed}")
    sampler.async_worker_stop()

    sampler = make_sampler()
    _, st = run_sequence(rig, sampler, [([1, 2, 3, 4], [0, 1, 2, 3], 12)], check=True,
                         greedy_tokens_mode=True)
    st.pop("tokens")
    passed = st["mismatches"] == 0 and st["side_events"] == 0 and st["poison_after_sync"] == 0
    ok &= passed
    results["greedy_tokens_keeps_main_copy"] = dict(st, passed=passed)
    log(f"greedy_tokens path: {st} passed={passed}")

    print(json.dumps({"passed": bool(ok),
                      "compared": "host new tokens (bitwise) vs torch.argmax of the same bf16 "
                                  "logits, and env-on vs env-off (main's _copy_to_host path)",
                      "results": results}))


if __name__ == "__main__":
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    with torch.inference_mode():
        bench() if mode == "bench" else check()
    # stderr, so the evidence JSON stays the last stdout line
    print(f"done in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
