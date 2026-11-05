"""Branch 129: the 40-layer CSA2 decode attention chain through the production entry point.

Every layer calls CSA2FlashInfer.forward_generation(q, metadata, args), exactly the call
csa2/module.py makes for a generation-only tile, with a stub attention backend whose
prepare_sparse_inputs returns the layer's packed-pool inputs (the real
prepare_selected_pools / _stage_selected_query / stage_selected / _stage_scratch code
runs in the base tree; the candidate routes to the Cake mixed-cache decode).

Layer layout = job 1063's decode census (job 1090): per step 2 SWA-only layers (width 128,
one shared staging group) + 38 width-640 layers in 8 staging groups (StageCompact 9 per
step). Each layer has its own SWA pool [8192, 528]; each group its own main pool
[65536, 288] and main selection (logical, mapped through a page table) -- CSA2's writers
fill them (QuantizeScatter). q [count, 16, 512] bf16 per layer, sink [16] fp32.

Modes
  time  : either tree. CUDA graph of the whole 40-layer chain, 256 MB L2 flush before every
          replay, 10 warmup + 100 timed replays; counts 1/2/4/8. Prints the evidence JSON.
  check : candidate tree. (1) the harness-matched sequence (same capture, 10 + 100 replays,
          outputs and Cake scratch poisoned before every replay) compared every iteration
          with the FP64 reference and with main's staged chain error; (2) 8 advancing decode
          steps (new rows written with CSA2's writer, selections / visible lengths
          advanced in place, graph replayed) vs FP64 and main, final pools vs a mirror;
          (3) capture with no allocation / sync; (4) no StageDequant / StageCompact /
          _split_sink_* launch on routed layers (torch.profiler).
"""

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from tensorrt_llm._torch.attention.backends.interface import (  # noqa: E402
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (  # noqa: E402
    CSA2FlashInfer,
    CSA2TrtllmAttention,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import (  # noqa: E402
    CSA2TrtllmMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (  # noqa: E402
    CSA2BackendForwardArgs,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (  # noqa: E402
    store_rows,
)

COUNTS = (1, 2, 4, 8)
WARMUP, ITERS = 10, 100
# (width, group) per layer: 2 SWA-only layers in one group, 38 width-640 in 8 groups.
GROUP_SIZES = (5, 5, 5, 5, 5, 5, 4, 4)
LAYERS = [(128, 0), (128, 0)] + [
    (640, 1 + g) for g, n in enumerate(GROUP_SIZES) for _ in range(n)
]
NEW = hasattr(CSA2FlashInfer, "_forward_cake_decode")


class StubAttention:
    prepare_selected_pools = CSA2TrtllmAttention.prepare_selected_pools
    prepare_selected = CSA2TrtllmAttention.prepare_selected
    _stage_selected_query = CSA2TrtllmAttention._stage_selected_query
    _validate_helper_output = staticmethod(CSA2TrtllmAttention._validate_helper_output)

    def __init__(self, inputs):
        self.inputs = inputs
        self.num_heads = C.HEADS
        self.q_scaling = 1.0
        self.sparse_params = SimpleNamespace(use_packed_sparse_attention=False)

    def prepare_sparse_inputs(self, q, metadata, forward_args):
        return self.inputs


class StubMeta:
    stage_selected = CSA2TrtllmMetadata.stage_selected
    _stage_scratch = CSA2TrtllmMetadata._stage_scratch

    def __init__(self, count, width, device="cuda"):
        extra = width - C.SWA_W
        self.num_tokens = count
        self.num_sparse_topk = width
        self.shared_plan = None
        self.num_contexts = 0
        self._csa2_graph_reserved = True
        self.is_cuda_graph = True
        self.kv_cache_manager = None
        self.swa_pool = torch.zeros((count, C.SWA_W, C.DIM), dtype=torch.bfloat16, device=device)
        self.extra_pool = torch.zeros((count, max(extra, 1), C.DIM), dtype=torch.bfloat16,
                                      device=device)
        self.prepared_counter = torch.zeros(1, dtype=torch.uint32, device=device)
        self.prepared_lens = torch.empty(count, dtype=torch.int32, device=device)
        self.prepared_indices = torch.full((count, width), -1, dtype=torch.int32, device=device)
        self.mla_bmm1_scale = None
        self.mla_bmm2_scale = None


def gpu_values(rows, seed, realistic=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    v = torch.randn((rows, C.DIM), generator=g, device="cuda")
    if not realistic:
        expo = torch.randint(-6, 7, (rows, C.DIM // 16), generator=g, device="cuda").float()
        v = v * torch.exp2(expo).repeat_interleave(16, dim=1)
    return v.to(torch.bfloat16)


class Chain:
    def __init__(self, count, seed=0, realistic=True):
        self.count = count
        self.swa_pools = []
        self.main_pools = {}
        self.sels = []
        self.metas = {}
        self.layers = []
        groups_sel = {}
        for li, (width, group) in enumerate(LAYERS):
            pool = torch.zeros((C.SWA_ROWS, 528), dtype=torch.uint8, device="cuda")
            store_rows(pool, torch.arange(C.SWA_ROWS, device="cuda"),
                       gpu_values(C.SWA_ROWS, seed * 1000 + li, realistic), "swa")
            self.swa_pools.append(pool)
            if width > C.SWA_W and group not in self.main_pools:
                mp = torch.zeros((C.MAIN_ROWS, 288), dtype=torch.uint8, device="cuda")
                store_rows(mp, torch.arange(C.MAIN_ROWS, device="cuda"),
                           gpu_values(C.MAIN_ROWS, seed * 1000 + 500 + group, realistic), "main")
                self.main_pools[group] = mp
            if group not in groups_sel:
                groups_sel[group] = C.build_selection(count, seed * 100 + group, width=width)
            gsel = groups_sel[group]
            # Private SWA slots per layer; main selection shared by the group.
            own = C.build_selection(count, seed * 100 + 50 + li, width=width)
            sel = dict(gsel)
            sel["swa"] = own["swa"]
            self.sels.append(sel)
            if group not in self.metas:
                self.metas[group] = StubMeta(count, width)
            q, sink = C.make_q(count, seed * 100 + 70 + li)
            if realistic:
                q = (q.float() * 0.5).to(torch.bfloat16)
            inputs = CSA2BackendForwardArgs(
                swa_pool=pool,
                swa_indices=sel["swa"],
                main_pool=self.main_pools.get(group) if width > C.SWA_W else None,
                topk_indices=sel["logical"] if width > C.SWA_W else None,
                main_logical_indices=sel["logical"] if width > C.SWA_W else None,
                main_mapping=sel["mapping"] if width > C.SWA_W else None,
                stage_group=(group, group, 0),
            )
            out = torch.empty_like(q)
            fi = CSA2FlashInfer(StubAttention(inputs))
            args = AttentionForwardArgs(
                attention_input_type=AttentionInputType.generation_only,
                attention_sinks=sink,
                output=out.flatten(1),
                sparse_backend_args=inputs,
            )
            self.layers.append(dict(width=width, group=group, q=q, sink=sink, out=out, fi=fi,
                                    args=args, inputs=inputs))
        torch.cuda.synchronize()

    def __call__(self):
        for L in self.layers:
            L["fi"].forward_generation(L["q"].flatten(1), self.metas[L["group"]], L["args"])

    def references(self):
        refs = []
        for li, L in enumerate(self.layers):
            mp = self.main_pools.get(L["group"])
            sel = self.sels[li]
            refs.append(C.reference(L["q"], L["sink"], self.swa_pools[li],
                                    mp if mp is not None else self.swa_pools[li], sel))
        return refs

    def main_outputs(self):
        """main's staged chain (StageCompact + StageDequant + split sink), eager."""
        outs = []
        for li, L in enumerate(self.layers):
            arm = C.MainArm(self.count, L["width"])
            o = torch.empty_like(L["q"])
            mp = self.main_pools.get(L["group"])
            arm(L["q"], L["sink"], self.swa_pools[li], mp, self.sels[li], o)
            outs.append(o)
        torch.cuda.synchronize()
        return outs


def stats_ms(t):
    t = sorted(t)
    med = t[len(t) // 2]
    p10, p90 = t[len(t) // 10], t[(9 * len(t)) // 10]
    return med, (p90 - p10) / med, p10, p90


def control_alloc_delta():
    """Allocation count of capturing a graph with one in-place op (capture's own overhead)."""
    x = torch.zeros(16, device="cuda")
    x.add_(1)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.add_(1)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    del graph
    return after - before


def capture(chain):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            chain()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    return graph, after - before


def mode_time(args):
    flush = torch.empty(C.FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    batch = {}
    control = control_alloc_delta()
    for count in COUNTS:
        chain = Chain(count, seed=count)
        graph, alloc_delta = capture(chain)
        C.log(f"count {count}: capture alloc delta {alloc_delta} (control {control})")
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for _ in range(WARMUP):
            flush.zero_()
            graph.replay()
        for i in range(ITERS):
            # Reset point: every mutable buffer (outputs, staging pools, scratch) is fully
            # rewritten by each replay; inputs (pools, selections, q) are never written.
            flush.zero_()
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
        med, noise, p10, p90 = stats_ms([a.elapsed_time(b) for a, b in zip(starts, ends)])
        clock = C.sm_clock()
        batch[str(count)] = {"kernel_ms": med, "noise_floor": noise, "p10_ms": p10,
                             "p90_ms": p90, "per_layer_us": med * 1e3 / len(LAYERS),
                             "clock": clock}
        C.log(f"count {count}: chain {med*1e3:.1f} us ({med*1e3/len(LAYERS):.2f} us/layer) "
              f"noise {noise:.3f} p10 {p10*1e3:.1f} p90 {p90*1e3:.1f} clock {clock}")
        del graph, chain
        torch.cuda.empty_cache()
    print(json.dumps({
        "batch": batch,
        "path": "Cake mixed-cache decode (int32 prep + decode + PDL merge)" if NEW
        else "StageCompact/StageDequant + _split_sink_main + _split_sink_merge",
        "layers": len(LAYERS),
        "output": {"class": "reordered"},
    }))


def errs(x, ref):
    d = (x.double() - ref).abs()
    scale = ref.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    return d.max().item(), (d / scale).max().item()


def within(e_new, e_main):
    return e_new <= max(e_main * 1.25, e_main + 2 ** -8)


def poison(chain):
    for L in chain.layers:
        L["out"].fill_(float("nan"))
        cake = getattr(L["fi"], "_cake", None)
        if cake is not None:
            for buf in cake._buffers.values():
                buf["mid_out"].fill_(float("nan"))
                buf["mid_lse"].fill_(float("nan"))
                buf["lse"].fill_(float("nan"))
                buf["swa"].fill_(-7)
                buf["main"].fill_(-7)


def mode_check(args):
    assert NEW, "check runs in the candidate tree"
    ok = True
    report = {}
    control = control_alloc_delta()
    report["capture_alloc_delta_control"] = control
    C.log("control capture alloc delta", control)
    for realistic in (True, False):
        for count in COUNTS:
            tag = f"{'real' if realistic else 'wide'}/c{count}"
            chain = Chain(count, seed=10 + count, realistic=realistic)
            refs = chain.references()
            mains = chain.main_outputs()
            e_main = [errs(m, r) for m, r in zip(mains, refs)]
            graph, alloc_delta = capture(chain)
            worst_abs_ratio = worst_rel_ratio = 0.0
            iter_ok = True
            first = last = None
            for it in range(WARMUP + ITERS):
                poison(chain)
                graph.replay()
                torch.cuda.synchronize()
                es = [errs(L["out"], r) for L, r in zip(chain.layers, refs)]
                fin = all(bool(torch.isfinite(L["out"]).all()) for L in chain.layers)
                good = fin and all(within(e[0], m[0]) for e, m in zip(es, e_main))
                iter_ok &= good
                worst_abs_ratio = max(worst_abs_ratio, max(e[0] / max(m[0], 1e-30)
                                                           for e, m in zip(es, e_main)))
                worst_rel_ratio = max(worst_rel_ratio, max(e[1] / max(m[1], 1e-30)
                                                           for e, m in zip(es, e_main)))
                outs = [L["out"].clone() for L in chain.layers]
                if first is None:
                    first = outs
                last = outs
            det = all(torch.equal(a, b) for a, b in zip(first, last))
            rep = dict(
                iterations=WARMUP + ITERS, every_iter_within_main=iter_ok,
                deterministic_first_vs_last=det, capture_alloc_delta=alloc_delta,
                max_abs_err_cake=max(errs(o, r)[0] for o, r in zip(last, refs)),
                max_abs_err_main=max(e[0] for e in e_main),
                max_rel_err_cake=max(errs(o, r)[1] for o, r in zip(last, refs)),
                max_rel_err_main=max(e[1] for e in e_main),
                worst_abs_ratio_cake_over_main=worst_abs_ratio,
                worst_rel_ratio_cake_over_main=worst_rel_ratio,
                max_abs_diff_cake_vs_main=max((o.float() - m.float()).abs().max().item()
                                              for o, m in zip(last, mains)),
            )
            rep["strict_le_main_all_layers_last_iter"] = all(
                errs(o, r)[0] <= m[0] for o, r, m in zip(last, refs, e_main))
            good = iter_ok and det and alloc_delta <= control
            rep["passed"] = good
            ok &= good
            report[f"repeated/{tag}"] = rep
            C.log(f"repeated/{tag}", json.dumps(rep))
            if realistic:
                ok &= advancing(chain, graph, report, count)
            del graph, chain
            torch.cuda.empty_cache()
    report["launches"] = launch_census()
    ok &= report["launches"]["passed"]
    print(json.dumps({
        "passed": bool(ok),
        "compared": "Cake route via CSA2FlashInfer.forward_generation (40-layer chain, CUDA graph) "
        "vs FP64 sink attention over CSA2's exactly dequantized rows and vs main's staged "
        "StageCompact/StageDequant + split-sink chain; tolerance: err <= max(1.25 x main, "
        "main + 2^-8) per layer per iteration",
        "report": report,
    }))


def advancing(chain, graph, report, count, steps=8):
    """8 decode steps: write new rows (CSA2 writer), advance selections in place, replay."""
    mirror_swa = [p.clone() for p in chain.swa_pools]
    mirror_main = {g: p.clone() for g, p in chain.main_pools.items()}
    ok = True
    rows = []
    gen = torch.Generator().manual_seed(77 + count)
    for step in range(steps):
        new_swa_slot = torch.full((count,), 0, dtype=torch.int64)
        for li, L in enumerate(chain.layers):
            # distinct slots: duplicate slots in one scatter would race (pool vs mirror)
            slots = torch.randperm(C.SWA_ROWS, generator=gen)[:count].cuda()
            vals = gpu_values(count, 9000 + step * 100 + li)
            store_rows(chain.swa_pools[li], slots, vals, "swa")
            store_rows(mirror_swa[li], slots, vals, "swa")
            sel = chain.sels[li]
            # ring: newest slot replaces column step % 128
            sel["swa"][:, step % C.SWA_W] = slots
        for g, mp in chain.main_pools.items():
            slots = torch.randperm(C.MAIN_ROWS, generator=gen)[:count].cuda()
            vals = gpu_values(count, 9500 + step * 10 + g)
            store_rows(mp, slots, vals, "main")
            store_rows(mirror_main[g], slots, vals, "main")
        done = set()
        for li, L in enumerate(chain.layers):
            if L["width"] > C.SWA_W and L["group"] not in done:
                done.add(L["group"])
                sel = chain.sels[li]
                # in place: tensors shared by the group's layers and the captured graph
                sel["visible"].add_(1)
                col = (7 * step) % C.MAIN_W
                sel["logical"][:, col] = (sel["visible"] - 1).to(torch.int32)
        del new_swa_slot
        torch.cuda.synchronize()
        refs = chain.references()
        mains = chain.main_outputs()
        poison(chain)
        graph.replay()
        torch.cuda.synchronize()
        es = [errs(L["out"], r) for L, r in zip(chain.layers, refs)]
        em = [errs(m, r) for m, r in zip(mains, refs)]
        good = all(within(e[0], m[0]) for e, m in zip(es, em)) and all(
            bool(torch.isfinite(L["out"]).all()) for L in chain.layers)
        ok &= good
        rows.append(dict(step=step, max_abs_err_cake=max(e[0] for e in es),
                         max_abs_err_main=max(m[0] for m in em), passed=good))
    pools_ok = all(torch.equal(a, b) for a, b in zip(chain.swa_pools, mirror_swa)) and all(
        torch.equal(chain.main_pools[g], mirror_main[g]) for g in chain.main_pools)
    ok &= pools_ok
    report[f"advancing/c{count}"] = dict(steps=rows, final_pools_equal_mirror=pools_ok, passed=ok)
    C.log(f"advancing/c{count}", json.dumps(report[f"advancing/c{count}"]))
    return ok


def launch_census():
    from torch.profiler import ProfilerActivity, profile

    chain = Chain(2, seed=3)
    chain()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        chain()
        torch.cuda.synchronize()
    names = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            names[e.name[:90]] = names.get(e.name[:90], 0) + 1
    banned = [n for n in names if any(b in n for b in ("StageDequant", "StageCompact",
                                                        "_split_sink"))]
    C.log("launches", names)
    return dict(kernels=names, banned=banned, passed=not banned)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("time", "check"))
    args = p.parse_args()
    C.log(f"new={NEW} device={torch.cuda.get_device_name()} layers={len(LAYERS)}")
    {"time": mode_time, "check": mode_check}[args.mode](args)


if __name__ == "__main__":
    main()
