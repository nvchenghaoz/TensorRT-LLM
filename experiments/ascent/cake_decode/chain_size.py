"""Branch 167 (idea 291, analysis): size the CSA2 Cake decode chain on one TP4 rank.

Per routed layer the chain is: _cake_index_prep_kernel (Triton) -> Cake decode_body -> merge
(CakeMergeF32Kernel for main's pinned fp32 `default`, the stock merge body for BF16 partials).
Shapes (one TP4 rank): q [count, 16, 512] bf16, SWA width 128 (2 chunks), main width 512
(8 chunks) for the 38 dual layers; SWA-only width 128 for the 2 single layers.
Pools / selections / FP64 reference come from common.py (CSA2's own QuantizeScatter writer,
mapped logical main positions through a page table, -1 holes, out-of-pool slots).

mode time (default):
  for count in 1/2/4/8 and every config
    - main's pinned fp32 `default` (CakeDecode's route), fp32 default with num_splits 1/2/5,
    - every exported 16-head dual variant (BF16 partials; the binding refuses fp32 mid_out
      for any body but h16 default),
    - every exported 8-head dual variant run as two head blocks (two calls on q[:, :8] and
      q[:, 8:], their own sinks; the binding refuses an h8 body for a 16-head call),
    - h16 single default fp32 / bf16 and h8 single default x2 for the SWA-only layers;
  reports
    - per-launch device durations (torch.profiler / CUPTI, eager, merge launched WITHOUT PDL so
      its duration is its own; 256 MB L2 flush before every iteration): median, p10, p90 of
      prep, body, merge (and the h8 path's 2 bodies + 2 merges);
    - the chain span (prep + body + merge, production PDL on) as a CUDA graph replay with a
      cold L2, 100 replays, median / p10 / p90;
    - max abs / rel error vs the fp32 default output and vs the FP64 reference, bitwise
      equality with default, repeat determinism;
    - SM clock (pynvml sampler) per config.
mode ncu: count 1 and 4 inside a cudaProfilerStart/Stop region: prep + dual fp32 default body
  + CakeMergeF32Kernel, then prep + single fp32 default body + merge (12 launches, no others).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

COUNTS = (1, 2, 4, 8)
DUAL_LAYERS, SINGLE_LAYERS = 38, 2
PROF_WARMUP, PROF_ITERS = 3, 50
GRAPH_WARMUP, GRAPH_ITERS = 10, 100
FLUSH = None


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- clocks
class ClockSampler:
    def __init__(self):
        self.handle = None
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            want = str(torch.cuda.get_device_properties(0).uuid).lower().replace("gpu-", "")
            n = pynvml.nvmlDeviceGetCount()
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                u = pynvml.nvmlDeviceGetUUID(h)
                u = (u.decode() if isinstance(u, bytes) else u).lower().replace("gpu-", "")
                if u == want or want in u or u in want:
                    self.handle = h
                    break
            if self.handle is None:
                vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(vis) if vis.isdigit() else 0)
        except Exception as e:  # noqa: BLE001
            log(f"[clock] pynvml unavailable: {e}")

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(
                    self.nvml.nvmlDeviceGetClockInfo(self.handle, self.nvml.NVML_CLOCK_SM)
                )
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.02)

    def start(self):
        if self.handle is None:
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join()
        self._thread = None
        s = sorted(self.samples)
        if not s:
            return None
        info = dict(sm_mhz_median=s[len(s) // 2], sm_mhz_min=s[0], n=len(s))
        try:
            info["temp_c"] = self.nvml.nvmlDeviceGetTemperature(
                self.handle, self.nvml.NVML_TEMPERATURE_GPU
            )
            info["throttle"] = int(
                self.nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
            )
        except Exception:  # noqa: BLE001
            pass
        return info


# --------------------------------------------------------------------------- stats
def stats(xs):
    if not xs:
        return None
    t = sorted(xs)
    n = len(t)
    med = t[n // 2]
    p10, p90 = t[n // 10], t[min(n - 1, (9 * n) // 10)]
    return dict(median_us=round(med, 3), p10_us=round(p10, 3), p90_us=round(p90, 3),
                noise=round((p90 - p10) / med, 4) if med > 0 else None, n=n)


def classify(name):
    if "cake_index_prep" in name:
        return "prep"
    if "CakeMergeF32" in name:
        return "merge"
    low = name.lower()
    if "merge" in low:
        return "merge"
    if "fill" in low or "memset" in low:
        return "flush"
    if "copy" in low or "elementwise" in low or "cat" in low:
        return "copy"
    return "body"


NAMES_SEEN = {}


def kernel_durations(step, iters=PROF_ITERS, warmup=PROF_WARMUP):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        FLUSH.zero_()
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            FLUSH.zero_()
            step()
        torch.cuda.synchronize()
    per = {}
    evs = None
    try:
        from torch.autograd import DeviceType

        evs = []
        for e in prof.profiler.kineto_results.events():
            if e.device_type() == DeviceType.CUDA:
                evs.append((e.name(), e.duration_ns() / 1000.0))
    except Exception as ex:  # noqa: BLE001
        log(f"[prof] kineto_results path failed ({ex}); falling back to FunctionEvents")
        evs = None
    if evs is None:
        from torch.autograd import DeviceType

        evs = [
            (e.name, e.time_range.elapsed_us())
            for e in prof.events()
            if e.device_type == DeviceType.CUDA
        ]
    for name, dur in evs:
        cat = classify(name)
        NAMES_SEEN.setdefault(name[:160], cat)
        per.setdefault(cat, []).append(dur)
    out = {}
    for cat, xs in per.items():
        if cat == "flush":
            continue
        out[cat] = stats(xs)
        out[cat]["launches_per_iter"] = round(len(xs) / iters, 3)
        out[cat]["sum_per_iter_us"] = round(sum(xs) / iters, 3)
    return out


def time_graph(fn):
    st, _ = C.time_graph(fn, warmup=GRAPH_WARMUP, iters=GRAPH_ITERS, flush=True)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}


def errs(o, ref):
    d = (o.double() - ref.double()).abs()
    amax = ref.double().abs().max().item()
    return dict(
        max_abs=d.max().item(),
        max_rel_norm=d.max().item() / max(amax, 1e-30),
        max_rel_elem=(d / (ref.double().abs() + 1e-2)).max().item(),
    )


# --------------------------------------------------------------------------- arms
class Arm:
    """One Cake decode configuration (pool views and int32 indices shared with the count)."""

    def __init__(self, cdm, q, sink, swa_pool, swa_idx, main_pool, main_idx, *, variant,
                 f32, num_splits=None):
        self.cdm = cdm
        self.q, self.sink = q, sink
        self.swa_pool, self.swa_idx = swa_pool, swa_idx
        self.main_pool, self.main_idx = main_pool, main_idx
        self.variant, self.num_splits = variant, num_splits
        T, h, _ = q.shape
        dual = main_pool is not None
        chunks = cdm.cake_sparse_mla_sm120_dsv41_mixed_num_chunks(128, 512 if dual else 0)
        dev = q.device
        self.out = torch.empty_like(q)
        self.lse = torch.empty((T, h), dtype=torch.float32, device=dev)
        self.mid = torch.empty((T, h, chunks, 512),
                               dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
        self.mid_lse = torch.empty((T, h, chunks), dtype=torch.float32, device=dev)
        self.info = {}

    def __call__(self, pdl):
        r = self.cdm.cake_sparse_mla_sm120_dsv41_mixed_decode(
            self.q, self.swa_pool, self.swa_idx, self.out, self.lse, C.SCALE,
            attn_sink=self.sink,
            extra_kv_cache=self.main_pool,
            extra_indices=self.main_idx,
            mid_out=self.mid, mid_lse=self.mid_lse,
            variant=self.variant, num_splits=self.num_splits, enable_pdl=pdl,
        )
        self.info = r
        return self.out


class H8x2:
    """A 16-head call served by two 8-head calls (q / sink / output split by head block)."""

    def __init__(self, cdm, q, sink, swa_pool, swa_idx, main_pool, main_idx, *, variant):
        self.q = q
        self.q_lo = q[:, :8].contiguous()
        self.q_hi = q[:, 8:].contiguous()
        s_lo, s_hi = sink[:8].contiguous(), sink[8:].contiguous()
        self.lo = Arm(cdm, self.q_lo, s_lo, swa_pool, swa_idx, main_pool, main_idx,
                      variant=variant, f32=False)
        self.hi = Arm(cdm, self.q_hi, s_hi, swa_pool, swa_idx, main_pool, main_idx,
                      variant=variant, f32=False)
        self.out = torch.empty_like(q)
        self.info = {}

    def kernels(self, pdl):
        self.lo(pdl)
        self.hi(pdl)
        self.info = dict(self.lo.info)

    def __call__(self, pdl, copies=True):
        if copies:
            self.q_lo.copy_(self.q[:, :8])
            self.q_hi.copy_(self.q[:, 8:])
        self.kernels(pdl)
        self.out[:, :8].copy_(self.lo.out)
        self.out[:, 8:].copy_(self.hi.out)
        return self.out


# --------------------------------------------------------------------------- main
def topo():
    for cmd in (["nvidia-smi", "topo", "-m"], ["nvidia-smi", "topo", "-p2p", "r"]):
        try:
            subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr, timeout=30)
        except Exception as e:  # noqa: BLE001
            print(f"[topo] {cmd}: {e}", file=sys.stderr)


def setup():
    from flashinfer.mla._sparse_mla_sm120 import cake_dsv41_mixed as cdm

    t0 = time.time()
    cdm.get_cake_sparse_mla_sm120_dsv41_mixed_module()
    log(f"[setup] cake module build/load {time.time() - t0:.1f}s")
    swa_pool, main_pool = C.build_pools(1)
    log(f"[setup] pools swa {tuple(swa_pool.shape)} main {tuple(main_pool.shape)}")
    return cdm, swa_pool, main_pool


def count_inputs(count, swa_pool, main_pool):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cake_decode import (
        prepare_cake_indices,
    )

    sel = C.build_selection(count, seed=1000 + count)
    q, sink = C.make_q(count, seed=2000 + count)
    dev = q.device
    swa_i = torch.full((count, C.SWA_W), -1, dtype=torch.int32, device=dev)
    main_i = torch.full((count, C.MAIN_W), -1, dtype=torch.int32, device=dev)
    swa_s = torch.full((count, C.SWA_W), -1, dtype=torch.int32, device=dev)

    def prep_dual():
        prepare_cake_indices(sel["swa"], C.SWA_ROWS, swa_i, sel["logical"], C.MAIN_ROWS,
                             main_i, sel["mapping"])

    def prep_single():
        prepare_cake_indices(sel["swa"], C.SWA_ROWS, swa_s)

    prep_dual()
    prep_single()
    torch.cuda.synchronize()
    rs, rm = C.map_reference(sel)
    prep_ok = bool(torch.equal(swa_i.long(), rs) and torch.equal(main_i.long(), rm)
                   and torch.equal(swa_s.long(), rs))
    return sel, q, sink, swa_i, main_i, swa_s, prep_dual, prep_single, prep_ok


def run_ncu():
    cdm, swa_pool, main_pool = setup()
    steps = []
    for count in (1, 4):
        sel, q, sink, swa_i, main_i, swa_s, prep_dual, prep_single, prep_ok = count_inputs(
            count, swa_pool, main_pool)
        dual = Arm(cdm, q, sink, swa_pool, swa_i, main_pool, main_i, variant="default", f32=True)
        single = Arm(cdm, q, sink, swa_pool, swa_s, None, None, variant="default", f32=True)
        for _ in range(3):
            prep_dual()
            dual(False)
            prep_single()
            single(False)
        torch.cuda.synchronize()
        log(f"[ncu] count {count} dual plan {dual.info} single plan {single.info} prep_ok {prep_ok}")
        steps.append((count, prep_dual, dual, prep_single, single))
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for count, prep_dual, dual, prep_single, single in steps:
        prep_dual()
        dual(False)
        prep_single()
        single(False)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    log("[ncu] region order per count (1 then 4): prep_dual, body_dual(h16 default f32), "
        "CakeMergeF32Kernel, prep_single, body_single(h16 single default f32), CakeMergeF32Kernel")
    print(json.dumps(dict(passed=True, mode="ncu", launches=12)))


def run_time(only_counts):
    global FLUSH
    topo()
    cdm, swa_pool, main_pool = setup()
    FLUSH = torch.empty(C.FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    d = cdm.kernel_dispatch()
    h16 = [n for n, v in d.variants.items()
           if (16, True, v.decode) in d.decode_bodies and (16, v.merge) in d.merge_bodies]
    h8 = [n for n, v in d.variants.items()
          if (8, True, v.decode) in d.decode_bodies and (8, v.merge) in d.merge_bodies]
    h16s = [n for n, v in d.variants.items()
            if (16, False, v.decode) in d.decode_bodies and (16, v.merge) in d.merge_bodies]
    h8s = [n for n, v in d.variants.items()
           if (8, False, v.decode) in d.decode_bodies and (8, v.merge) in d.merge_bodies]
    log(f"[setup] h16 dual variants {h16}")
    log(f"[setup] h8 dual variants {h8}")
    log(f"[setup] h16 single {h16s} h8 single {h8s}")
    # Q2 probe: does the public API / binding accept an h8 body for a 16-head call?
    q2 = {}
    clock = ClockSampler()
    results = {}
    summary = {}
    for count in only_counts:
        sel, q, sink, swa_i, main_i, swa_s, prep_dual, prep_single, prep_ok = count_inputs(
            count, swa_pool, main_pool)
        log(f"[count {count}] prep matches torch map_reference: {prep_ok}")
        ref_d = C.reference(q, sink, swa_pool, main_pool, sel)
        ref_s = C.reference(q, sink, swa_pool, main_pool, dict(sel, width=C.SWA_W))
        if count == only_counts[0]:
            try:
                probe = Arm(cdm, q, sink, swa_pool, swa_i, main_pool, main_i,
                            variant="sa+tl+ip+ew+ra", f32=False)
                probe(False)
                q2["h16_call_with_h8_only_variant"] = f"accepted {probe.info}"
            except Exception as e:  # noqa: BLE001
                q2["h16_call_with_h8_only_variant"] = f"refused: {type(e).__name__}: {str(e)[:300]}"
            try:
                probe = Arm(cdm, q, sink, swa_pool, swa_i, main_pool, main_i,
                            variant="ip", f32=True)
                probe(False)
                q2["fp32_mid_out_non_default"] = f"accepted {probe.info}"
            except Exception as e:  # noqa: BLE001
                q2["fp32_mid_out_non_default"] = f"refused: {type(e).__name__}: {str(e)[:300]}"
            log(f"[Q2] {q2}")
        configs = []  # (label, kind, arm, prep, ref)
        configs.append(("dual:default_f32", "dual", Arm(cdm, q, sink, swa_pool, swa_i, main_pool,
                        main_i, variant="default", f32=True), prep_dual, ref_d))
        for s in (1, 2, 5):
            configs.append((f"dual:default_f32_s{s}", "dual", Arm(cdm, q, sink, swa_pool, swa_i,
                            main_pool, main_i, variant="default", f32=True, num_splits=s),
                            prep_dual, ref_d))
        for n in h16:
            configs.append((f"dual:{n}_bf16", "dual", Arm(cdm, q, sink, swa_pool, swa_i,
                            main_pool, main_i, variant=n, f32=False), prep_dual, ref_d))
        for n in h8:
            configs.append((f"dual:h8x2:{n}", "dual_h8", H8x2(cdm, q, sink, swa_pool, swa_i,
                            main_pool, main_i, variant=n), prep_dual, ref_d))
        configs.append(("single:default_f32", "single", Arm(cdm, q, sink, swa_pool, swa_s, None,
                        None, variant="default", f32=True), prep_single, ref_s))
        for n in h16s:
            configs.append((f"single:{n}_bf16", "single", Arm(cdm, q, sink, swa_pool, swa_s, None,
                            None, variant=n, f32=False), prep_single, ref_s))
        for n in h8s:
            configs.append((f"single:h8x2:{n}", "single_h8", H8x2(cdm, q, sink, swa_pool, swa_s,
                            None, None, variant=n), prep_single, ref_s))
        base_out = {}
        rows = {}
        for label, kind, arm, prep, ref in configs:
            row = dict(kind=kind)
            try:
                # correctness (untimed): two runs, bitwise repeat, error vs default and FP64 ref
                prep()
                o1 = arm(False).clone()
                prep()
                o2 = arm(False).clone()
                torch.cuda.synchronize()
                row["plan"] = dict(arm.info)
                row["deterministic"] = bool(torch.equal(o1, o2))
                row["finite"] = bool(torch.isfinite(o1).all())
                row["err_vs_ref"] = errs(o1, ref)
                key = kind.split("_")[0]
                if label in ("dual:default_f32", "single:default_f32"):
                    base_out[key] = o1
                b = base_out[key]
                row["err_vs_default_f32"] = errs(o1, b)
                row["bitwise_vs_default_f32"] = bool(torch.equal(o1, b))
                # per-launch durations (eager, merge without PDL)
                clock.start()
                if kind.endswith("h8"):
                    k = kernel_durations(lambda: (prep(), arm.kernels(False)))
                else:
                    k = kernel_durations(lambda: (prep(), arm(False)))
                row["kernels"] = k
                # chain span (graph, PDL as production)
                row["chain_graph_pdl"] = time_graph(lambda: (prep(), arm(True)))
                if label in ("dual:default_f32", "single:default_f32"):
                    row["chain_graph_nopdl"] = time_graph(lambda: (prep(), arm(False)))
                    row["prep_only_graph"] = time_graph(prep)
                    # drop-one: the same chain with prep removed (indices already prepared and
                    # unchanged), so exposed prep = chain_graph_pdl - body_merge_graph_pdl.
                    prep()
                    torch.cuda.synchronize()
                    row["body_merge_graph_pdl"] = time_graph(lambda: arm(True))
                    row["exposed_prep_us"] = round(
                        row["chain_graph_pdl"]["median_us"]
                        - row["body_merge_graph_pdl"]["median_us"], 3)
                    o3 = arm(False).clone()
                    torch.cuda.synchronize()
                    row["repeat_after_graphs_bitwise"] = bool(torch.equal(o3, o1))
                row["clock"] = clock.stop()
                body = k.get("body", {}) or {}
                merge = k.get("merge", {}) or {}
                log(f"[count {count}] {label:48s} plan={row['plan'].get('variant')}/"
                    f"s{row['plan'].get('num_splits')} "
                    f"prep={k.get('prep', {}).get('median_us')} "
                    f"body={body.get('median_us')}[{body.get('p10_us')},{body.get('p90_us')}]"
                    f"x{body.get('launches_per_iter')} "
                    f"merge={merge.get('median_us')}x{merge.get('launches_per_iter')} "
                    f"chain={row['chain_graph_pdl']['median_us']}"
                    f"[{row['chain_graph_pdl']['p10_us']},{row['chain_graph_pdl']['p90_us']}] "
                    f"dAbs_def={row['err_vs_default_f32']['max_abs']:.3g} "
                    f"dAbs_ref={row['err_vs_ref']['max_abs']:.3g} det={row['deterministic']} "
                    f"clk={(row['clock'] or {}).get('sm_mhz_median')}")
            except Exception as e:  # noqa: BLE001
                clock.stop()
                row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                log(f"[count {count}] {label}: ERROR {row['error']}")
            rows[label] = row
        results[str(count)] = dict(prep_ok=prep_ok, rows=rows)

        def kern_sum(r):
            k = r.get("kernels") or {}
            return sum((k.get(c) or {}).get("sum_per_iter_us", 0.0)
                       for c in ("prep", "body", "merge"))

        s = {}
        for kind, base_label, layers in (("dual", "dual:default_f32", DUAL_LAYERS),
                                         ("single", "single:default_f32", SINGLE_LAYERS)):
            base = rows.get(base_label, {})
            if "chain_graph_pdl" not in base:
                continue
            bch = base["chain_graph_pdl"]
            cands = [(lab, r) for lab, r in rows.items()
                     if lab.startswith(kind + ":") and "chain_graph_pdl" in r and lab != base_label]
            table = []
            for lab, r in cands:
                ch = r["chain_graph_pdl"]
                table.append(dict(
                    label=lab,
                    chain_median_us=ch["median_us"],
                    chain_delta_us=round(bch["median_us"] - ch["median_us"], 3),
                    beyond_noise=bool(ch["p90_us"] < bch["p10_us"]),
                    kernel_sum_us=round(kern_sum(r), 3),
                    kernel_delta_us=round(kern_sum(base) - kern_sum(r), 3),
                    max_abs_vs_default=r["err_vs_default_f32"]["max_abs"],
                    max_abs_vs_ref=r["err_vs_ref"]["max_abs"],
                    bitwise=r["bitwise_vs_default_f32"],
                ))
            table.sort(key=lambda t: t["chain_median_us"])
            best = table[0] if table else None
            s[kind] = dict(
                base_chain=bch,
                base_kernels=base.get("kernels"),
                base_kernel_sum_us=round(kern_sum(base), 3),
                base_body_merge_graph=base.get("body_merge_graph_pdl"),
                base_exposed_prep_us=base.get("exposed_prep_us"),
                base_repeat_after_graphs_bitwise=base.get("repeat_after_graphs_bitwise"),
                base_err_vs_ref=base["err_vs_ref"]["max_abs"],
                best=best,
                recoverable_ms_per_step=(
                    round(max(best["chain_delta_us"], 0.0) * layers / 1000.0, 4) if best else None
                ),
                ranked=table[:6],
            )
        summary[str(count)] = s
        log(f"[count {count}] SUMMARY {json.dumps(s)[:4000]}")
    log(f"[names] {json.dumps(NAMES_SEEN)}")
    log("[full] " + json.dumps(results))
    compact = {c: {k: dict(base_chain_median_us=v["base_chain"]["median_us"],
                           base_kernel_sum_us=v["base_kernel_sum_us"],
                           base_exposed_prep_us=v["base_exposed_prep_us"],
                           base_kernels={c: (v["base_kernels"] or {}).get(c)
                                         for c in ("prep", "body", "merge")},
                           best=v["best"], recoverable_ms_per_step=v["recoverable_ms_per_step"])
                   for k, v in s.items()} for c, s in summary.items()}
    print(json.dumps(dict(passed=True, mode="time", q2=q2, summary=compact)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["time", "ncu"], default="time")
    ap.add_argument("--counts", default="1,2,4,8")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.mode == "ncu":
        run_ncu()
    else:
        run_time(tuple(int(x) for x in a.counts.split(",")))


if __name__ == "__main__":
    main()
