"""Branch 145 (idea 245 Q1): 40-layer CSA2 decode attention + output-projection chain on main's Cake route.

Per layer, exactly as csa2/module.py calls them for a generation-only step:
  CSA2FlashInfer.forward_generation (stub attention backend returning the layer's packed-pool inputs)
    -> _cake_index_prep_kernel -> Cake decode_body -> CakeMergeF32Kernel (fp32 split partials, PDL)
  DeepseekV41Attention._project_output on a module stand-in (experiments/ascent/b12x_wo/harness.py)
    -> LowMWOProjection: K2' csa2_lowm_wo_a -> K3' csa2_mxfp8_low_m_gemm (wo_b)
Layer layout as branch 129's chain (job 1090 census): 2 SWA-only layers (width 128) + 38 width-640 layers in
8 main-pool groups. TP4 rank shapes: q [M, 16, 512], wo_a [G=2, 1024, 4096] E4M3, wo_b [5120, 2048] MXFP8.
Every layer has its own weights and pools; L2 is flushed (256 MB) before every replay.

Arms (candidate tree only; the base tree has no flag and runs 'default' = main):
  default : production behaviour (candidate: Q1 flag from DeepseekV41Attention._lowm_wo_follows)
  off     : flag forced False -> CakeMergeF32Kernel<false> (= main's merge, end-of-grid hand-off)
  q1      : flag from _lowm_wo_follows -> CakeMergeF32Kernel<true> (trigger-first)
  q1k3    : q1 + K3' QoutArgs.flags bit 1 (K3' triggers after its pre-wait copies) -- report only

Modes
  time   : CUDA graph of the 40-layer chain per M in 1/2/4/8; arms interleaved in rounds (5 x (2 warmup + 20)).
           Last line = evidence JSON for the 'default' arm (kernel_ms = whole chain).
  spans  : time + Kineto per-kernel spans per arm (median, p10, p90 over layers x replays), merge->K2' overlap.
  check  : correctness (candidate): per M, eager 'off' reference; graph replays of q1 / q1k3 / off for
           10 warmup + 100 iterations, outputs (attention out + projection out, all 40 layers) poisoned with
           finite sentinels before every replay and compared bitwise every iteration; Cake scratch (partials,
           lse, int32 indices) SHARED across layers of equal shape. Live-overlap control: K2' with debug bit 0
           (skip griddepcontrol.wait) behind the trigger-first merge must FAIL; the same skip-wait behind
           the 'off' merge is reported.
  sass   : SASS of every CakeMergeF32Kernel instance in the JIT-built binding (run in both trees).
"""

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ASC = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("cake_common", os.path.join(ASC, "cake_decode", "common.py"))
CC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CC)
sys.path.insert(0, os.path.join(ASC, "b12x_wo"))
import common as WC  # noqa: E402  (b12x_wo common; sets TRTLLM_ENABLE_PDL=1)
import harness as Hn  # noqa: E402
import torch  # noqa: E402

from tensorrt_llm._torch.attention.backends.interface import (  # noqa: E402
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2 import b12x_wo as BW  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (  # noqa: E402
    CSA2FlashInfer,
    CSA2TrtllmAttention,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import store_rows  # noqa: E402

CANDIDATE = hasattr(DeepseekV41Attention, "_lowm_wo_follows")
GROUP_SIZES = (5, 5, 5, 5, 5, 5, 4, 4)
LAYERS = [(128, 0), (128, 0)] + [(640, 1 + g) for g, n in enumerate(GROUP_SIZES) for _ in range(n)]
K_PREP, K_BODY, K_MERGE, K2, K3 = ("_cake_index_prep_kernel", "decode_body", "CakeMergeF32Kernel",
                                   "csa2_lowm_wo_a_kernel", "csa2_mxfp8_low_m_gemm_kernel")
ART = Path(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/cake_q1"))
ART.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class StubAttention:
    prepare_selected_pools = CSA2TrtllmAttention.prepare_selected_pools
    prepare_selected = CSA2TrtllmAttention.prepare_selected
    _stage_selected_query = CSA2TrtllmAttention._stage_selected_query
    _validate_helper_output = staticmethod(CSA2TrtllmAttention._validate_helper_output)

    def __init__(self, inputs):
        self.inputs = inputs
        self.num_heads = CC.HEADS
        self.q_scaling = 1.0
        self.sparse_params = SimpleNamespace(use_packed_sparse_attention=False)

    def prepare_sparse_inputs(self, q, metadata, forward_args):
        return self.inputs


class StubMeta:
    stage_selected = CSA2TrtllmMetadata.stage_selected
    _stage_scratch = CSA2TrtllmMetadata._stage_scratch

    def __init__(self, count, width, device="cuda"):
        extra = width - CC.SWA_W
        self.num_tokens = count
        self.num_sparse_topk = width
        self.shared_plan = None
        self.num_contexts = 0
        self._csa2_graph_reserved = True
        self.is_cuda_graph = True
        self.kv_cache_manager = None
        self.swa_pool = torch.zeros((count, CC.SWA_W, CC.DIM), dtype=torch.bfloat16, device=device)
        self.extra_pool = torch.zeros((count, max(extra, 1), CC.DIM), dtype=torch.bfloat16, device=device)
        self.prepared_counter = torch.zeros(1, dtype=torch.uint32, device=device)
        self.prepared_lens = torch.empty(count, dtype=torch.int32, device=device)
        self.prepared_indices = torch.full((count, width), -1, dtype=torch.int32, device=device)
        self.mla_bmm1_scale = None
        self.mla_bmm2_scale = None


def gpu_values(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, CC.DIM), generator=g, device="cuda").to(torch.bfloat16)


class World:
    """Per-layer pools and WO modules, built once (shared by every M)."""

    def __init__(self, ckpt):
        self.dims = WC.dims_from_ckpt(ckpt)
        assert self.dims["heads_tp"] == CC.HEADS and self.dims["head_dim"] == CC.DIM, self.dims
        self.swa_pools, self.main_pools = [], {}
        for li, (width, group) in enumerate(LAYERS):
            pool = torch.zeros((CC.SWA_ROWS, 528), dtype=torch.uint8, device="cuda")
            store_rows(pool, torch.arange(CC.SWA_ROWS, device="cuda"), gpu_values(CC.SWA_ROWS, li), "swa")
            self.swa_pools.append(pool)
            if width > CC.SWA_W and group not in self.main_pools:
                mp = torch.zeros((CC.MAIN_ROWS, 288), dtype=torch.uint8, device="cuda")
                store_rows(mp, torch.arange(CC.MAIN_ROWS, device="cuda"), gpu_values(CC.MAIN_ROWS, 500 + group),
                           "main")
                self.main_pools[group] = mp
        w0 = WC.make_weights(self.dims)
        cs = WC.make_cos_sin(self.dims)
        ws = [w0] + [WC.clone_weights(w0, self.dims) for _ in range(len(LAYERS) - 1)]
        self.mods = [Hn.make_module(self.dims, w, cs) for w in ws]
        torch.cuda.synchronize()
        log(f"world: {len(LAYERS)} layers dims={self.dims} candidate={CANDIDATE}")


class Chain:
    def __init__(self, world, count, seed=0):
        self.world, self.count = world, count
        self.arm = "default"
        self.layers = []
        metas, gsel = {}, {}
        _, self.pos = WC.make_inputs(world.dims, count, seed=300 + count)
        for li, (width, group) in enumerate(LAYERS):
            if group not in gsel:
                gsel[group] = CC.build_selection(count, seed * 100 + group, width=width)
            own = CC.build_selection(count, seed * 100 + 50 + li, width=width)
            sel = dict(gsel[group])
            sel["swa"] = own["swa"]
            if group not in metas:
                metas[group] = StubMeta(count, width)
            q, sink = CC.make_q(count, seed * 100 + 70 + li)
            q = (q.float() * 0.5).to(torch.bfloat16)
            state = SimpleNamespace(output_pdl_consumer=False)
            mp = world.main_pools.get(group) if width > CC.SWA_W else None
            inputs = CSA2BackendForwardArgs(
                swa_pool=world.swa_pools[li], swa_indices=sel["swa"], main_pool=mp,
                topk_indices=sel["logical"] if mp is not None else None,
                main_logical_indices=sel["logical"] if mp is not None else None,
                main_mapping=sel["mapping"] if mp is not None else None,
                stage_group=(group, group, 0), state=state)
            out = torch.empty_like(q)
            fi = CSA2FlashInfer(StubAttention(inputs))
            args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only,
                                        attention_sinks=sink, output=out.flatten(1), sparse_backend_args=inputs)
            self.layers.append(dict(width=width, q=q, out=out, fi=fi, args=args, state=state,
                                    meta=metas[group], mod=world.mods[li]))

    def flag(self, L):
        if not CANDIDATE or self.arm == "off":
            return False
        if self.arm in ("default", "q1"):
            return DeepseekV41Attention._lowm_wo_follows(L["mod"], L["out"].view(self.count, -1))
        raise ValueError(self.arm)

    def __call__(self):
        res = []
        for L in self.layers:
            L["state"].output_pdl_consumer = self.flag(L)
            L["fi"].forward_generation(L["q"].flatten(1), L["meta"], L["args"])
            res.append(Hn.project(L["mod"], L["out"].view(self.count, -1), self.pos))
        return res

    def share_scratch(self):
        """Every layer of one shape uses the first such layer's Cake scratch (stricter than production)."""
        first, n = {}, 0
        for L in self.layers:
            cake = L["fi"]._cake
            for key, buf in cake._buffers.items():
                if key not in first:
                    first[key] = buf
                    continue
                for name in ("swa", "main", "lse", "mid_out", "mid_lse"):
                    buf[name] = first[key][name]
                n += 1
        return n

    def poison(self, outs=()):
        for L in self.layers:
            L["out"].fill_(3.0)
            cake = getattr(L["fi"], "_cake", None)
            if cake is not None:
                for buf in cake._buffers.values():
                    buf["mid_out"].fill_(5.0)
                    buf["mid_lse"].fill_(-3.0)
                    buf["lse"].fill_(-3.0)
        for o in outs:
            o.fill_(7.0)


class Patch:
    """K3' flags bit 1 and/or K2' debug bit 0 (skip wait), applied while warming and capturing."""

    def __init__(self, k3_bit1=False, k2_skip_wait=False):
        self.k3_bit1, self.k2_skip_wait = k3_bit1, k2_skip_wait

    def __enter__(self):
        self.orig = (BW.LowMWOProjection._wo_b_mm, BW.LowMWOProjection.o_lora)
        if self.k3_bit1:
            def _wo_b_mm(proj, o_lora):
                warps, cluster_k, row_tiles = BW._LOWM_WO_B_CONFIGS[o_lora.shape[0]]
                return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(
                    o_lora, proj.wo_b[0], proj.wo_b[1], warps, cluster_k, row_tiles, 2, True)[0]
            BW.LowMWOProjection._wo_b_mm = _wo_b_mm
        if self.k2_skip_wait:
            def o_lora(proj, output, positions):
                m = output.shape[0]
                warps, cluster_k, row_tiles = BW._LOWM_WO_A_CONFIGS[m]
                return torch.ops.trtllm.csa2_lowm_wo_a(
                    output.view(m, -1), positions.reshape(-1), proj.cos_sin, proj.wa, proj.wsc, proj.groups,
                    proj.head_dim, proj.nope_dim, warps, cluster_k, row_tiles, True, None, None, 1)
            BW.LowMWOProjection.o_lora = o_lora
        return self

    def __exit__(self, *a):
        BW.LowMWOProjection._wo_b_mm, BW.LowMWOProjection.o_lora = self.orig


def capture(chain, arm, patch=None):
    chain.arm = arm
    with patch or Patch():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                chain()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = chain()
    torch.cuda.synchronize()
    return g, outs


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summ(v):
    med = st.median(v)
    return dict(median=med, p10=pct(v, 0.1), p90=pct(v, 0.9),
                noise=(pct(v, 0.9) - pct(v, 0.1)) / med if med else 0.0)


_FLUSH = []


def flush():
    if not _FLUSH:
        _FLUSH.append(torch.empty(CC.FLUSH_BYTES, dtype=torch.uint8, device="cuda"))
    _FLUSH[0].zero_()


def time_arms(graphs, rounds=5, warm=2, iters=20):
    times = {a: [] for a in graphs}
    for _ in range(rounds):
        for arm, g in graphs.items():
            for _ in range(warm):
                flush()
                g.replay()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(iters)]
            for a, b in ev:
                flush()
                a.record()
                g.replay()
                b.record()
            torch.cuda.synchronize()
            times[arm] += [a.elapsed_time(b) for a, b in ev]
    return {a: summ(t) for a, t in times.items()}


def kineto_spans(g, tag, reps=5):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            flush()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
    path = ART / f"kineto_{tag}.json"
    prof.export_chrome_trace(str(path))
    evs = json.load(open(path))
    evs = evs.get("traceEvents", evs) if isinstance(evs, dict) else evs
    kern = sorted((float(e["ts"]), float(e.get("dur", 0)), e.get("name", "")) for e in evs
                  if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel")
    kinds = {K_PREP: "prep", K_BODY: "decode_body", K_MERGE: "merge", K2: "K2", K3: "K3"}

    def kind(n):
        for k, v in kinds.items():
            if k in n:
                return v
        return None

    seq = [(ts, d, kind(n), n) for ts, d, n in kern if kind(n)]
    others = sorted({n[:80] for ts, d, n in kern if not kind(n)})
    dur = {v: [] for v in kinds.values()}
    for ts, d, k, n in seq:
        dur[k].append(d)
    gap, follow, early, n_merge = [], 0, 0, 0
    for i, (ts, d, k, n) in enumerate(seq):
        if k == "merge":
            n_merge += 1
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            if nxt is not None and nxt[2] == "K2":
                follow += 1
                gap.append(nxt[0] - (ts + d))  # < 0: K2' started before the merge ended
                early += int(nxt[0] < ts + d)
    preps = [i for i, s in enumerate(seq) if s[2] == "prep"]
    span, body_k2 = [], []
    for a, b in zip(preps, preps[1:] + [len(seq)]):
        blk = seq[a:b]
        k3 = [s for s in blk if s[2] == "K3"]
        if k3:
            span.append(k3[-1][0] + k3[-1][1] - blk[0][0])
        bod = [s for s in blk if s[2] == "decode_body"]
        k2 = [s for s in blk if s[2] == "K2"]
        if bod and k2:
            body_k2.append(k2[0][0] - (bod[0][0] + bod[0][1]))
    return dict(
        per_kernel_us={k: summ(v) | {"n": len(v)} for k, v in dur.items() if v},
        merges=n_merge, merge_then_k2=follow, k2_starts_before_merge_end=early,
        k2_start_minus_merge_end_us=summ(gap) if gap else None,
        k2_start_minus_body_end_us=summ(body_k2) if body_k2 else None,
        layer_span_us=summ(span) if span else None,
        other_kernels=others[:12])


def lease_info():
    return dict(clock=CC.sm_clock(), host=os.uname().nodename,
                lease=os.environ.get("AUTO_DEV_LEASE", os.environ.get("RSI_LEASE_ID")),
                cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))


def mode_time(args, spans=False):
    world = World(args.ckpt)
    arms = ["off", "q1", "q1k3"] if (CANDIDATE and spans) else ["default"]
    batch, table = {}, {}
    for count in [int(c) for c in args.counts.split(",")]:
        chain = Chain(world, count, seed=count)
        graphs = {}
        for arm in arms:
            graphs[arm], _ = capture(chain, "q1" if arm == "q1k3" else arm, Patch(k3_bit1=(arm == "q1k3")))
        t = time_arms(graphs)
        row = {a: {"whole_us": {k: (v * 1e3 if k != "noise" else v) for k, v in s.items()},
                   "per_layer_us": s["median"] * 1e3 / len(LAYERS)} for a, s in t.items()}
        if spans:
            for arm, g in graphs.items():
                row[arm]["spans"] = kineto_spans(g, f"M{count}_{arm}")
        table[str(count)] = row
        main_arm = "default" if "default" in t else "q1"
        batch[str(count)] = {"kernel_ms": t[main_arm]["median"], "noise_floor": t[main_arm]["noise"],
                             "p10_ms": t[main_arm]["p10"], "p90_ms": t[main_arm]["p90"],
                             "per_layer_us": t[main_arm]["median"] * 1e3 / len(LAYERS)}
        log(f"M={count} " + json.dumps(row)[:6000])
        del graphs, chain
        torch.cuda.empty_cache()
    (ART / "table.json").write_text(json.dumps(table, indent=1))
    print(json.dumps({"batch": batch, "candidate": CANDIDATE, "layers": len(LAYERS), "lease": lease_info(),
                      "output": {"class": "exact"}}))


def mode_check(args):
    assert CANDIDATE, "check runs in the candidate tree"
    world = World(args.ckpt)
    ok, report = True, {}
    for count in [int(c) for c in args.counts.split(",")]:
        chain = Chain(world, count, seed=20 + count)
        chain.arm = "off"
        chain()
        torch.cuda.synchronize()
        n_shared = chain.share_scratch()
        ref_res = [r.clone() for r in chain()]
        ref_att = [L["out"].clone() for L in chain.layers]
        torch.cuda.synchronize()
        chain.arm = "q1"
        q1_flags = [bool(chain.flag(L)) for L in chain.layers]
        rep = {"shared_scratch_layers": n_shared, "q1_flag_layers": sum(q1_flags)}
        if not all(q1_flags):
            ok = False
            rep["error"] = "Q1 flag not set on every layer"
        for arm, patch, expect in (("q1", Patch(), True), ("q1k3", Patch(k3_bit1=True), True),
                                   ("off", Patch(), True),
                                   ("q1_skipwait", Patch(k2_skip_wait=True), False),
                                   ("off_skipwait", Patch(k2_skip_wait=True), None)):
            g, outs = capture(chain, "off" if arm.startswith("off") else "q1", patch)
            iters = args.warmup + args.iters if expect else 20
            bad_iters, max_abs, finite, per_iter = 0, 0.0, True, []
            for it in range(iters):
                chain.poison(outs)  # reset point: every output and the Cake scratch hold finite sentinels
                flush()
                g.replay()
                torch.cuda.synchronize()
                diff_layers, worst = 0, 0.0
                for o, r, L, a in zip(outs, ref_res, chain.layers, ref_att):
                    if not (torch.equal(o, r) and torch.equal(L["out"], a)):
                        diff_layers += 1
                        worst = max(worst, (o.float() - r.float()).abs().max().item(),
                                    (L["out"].float() - a.float()).abs().max().item())
                    finite &= bool(torch.isfinite(o).all())
                bad_iters += int(diff_layers > 0)
                max_abs = max(max_abs, worst)
                if it < 2 or (diff_layers and len(per_iter) < 6):
                    per_iter.append((it, diff_layers, round(worst, 4)))
            res = {"iterations": iters, "bad_iterations": bad_iters, "max_abs_diff": max_abs, "finite": finite,
                   "samples(iter,diff_layers,max_abs)": per_iter}
            if expect is True:
                res["passed"] = bad_iters == 0 and finite
                ok &= res["passed"]
            elif expect is False:
                res["control_failed_as_expected"] = bad_iters > 0
                ok &= res["control_failed_as_expected"]
            rep[arm] = res
            log(f"M={count} {arm}: {json.dumps(res)}")
            del g, outs
        report[str(count)] = rep
        del chain
        torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(ok),
                      "compared": "40-layer Cake attention + LowMWO projection CUDA-graph chain: trigger-first "
                                  "merge (q1), q1 + K3' bit 1, and the ungated merge (off), each replay's 40 "
                                  "attention outputs and 40 projection outputs bitwise vs the eager ungated chain "
                                  "(main's merge code), 110 iterations, finite sentinels before every replay, Cake "
                                  "scratch shared across layers; skip-wait K2' control behind the trigger-first "
                                  "merge must differ",
                      "report": report}))


def mode_sass(args):
    """SASS of every CakeMergeF32Kernel instance in the JIT-built binding."""
    import hashlib
    import re
    import subprocess

    from flashinfer.jit.cake_sparse_mla_sm120_dsv41_mixed import gen_cake_sparse_mla_sm120_dsv41_mixed_module
    from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import get_cake_sparse_mla_sm120_dsv41_mixed_module

    get_cake_sparse_mla_sm120_dsv41_mixed_module()
    lib = gen_cake_sparse_mla_sm120_dsv41_mixed_module().get_library_path()
    txt = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", str(lib)], capture_output=True,
                         text=True).stdout
    funcs, cur, name = {}, [], None
    for line in txt.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if name:
                funcs[name] = cur
            name, cur = m.group(1), []
            continue
        if name:
            m2 = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
            if m2:
                cur.append(re.sub(r"\s+", " ", m2.group(1)))
    if name:
        funcs[name] = cur
    out = {}
    for n, ins in funcs.items():
        if "CakeMergeF32Kernel" in n:
            h = hashlib.sha256("\n".join(ins).encode()).hexdigest()
            out[n] = {"instructions": len(ins), "sha256": h, "first10": ins[:10]}
            (ART / f"sass_{h[:12]}.txt").write_text("\n".join(ins))
    print(json.dumps({"passed": bool(out), "library": str(lib), "n_functions": len(funcs), "kernels": out}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["time", "spans", "check", "sass"])
    ap.add_argument("ckpt", nargs="?", default="")
    ap.add_argument("--counts", default="1,2,4,8")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    if a.mode == "time":
        mode_time(a)
    elif a.mode == "spans":
        mode_time(a, spans=True)
    elif a.mode == "check":
        mode_check(a)
    else:
        mode_sass(a)
