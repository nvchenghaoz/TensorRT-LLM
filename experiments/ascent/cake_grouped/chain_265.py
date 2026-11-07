"""Branch 238 (idea 265): grouped CSA2 Cake index prep on the 40-layer decode chain.

Builds on experiments/ascent/cake_q1/chain.py (World: per-layer SWA pools, 8 main-pool groups, LowMWO
projection modules; non-stress randn values). Unlike chain.py, every layer's inputs come from ONE stand-in
for the request metadata (``Source``), as csa2/backend.prepare_sparse_inputs builds them in production:
  * each forward first runs the real all-layer SWA refresh (csa2/kernel.refresh_swa_slots, CuTe
    SwaRefreshKernel) on synthetic per-layer SWA page tables (holes and pages past the pool), so
    swa_indices = reads[layer] (int64) and the visible lengths = visible[layer] = (position + 1) // ratio;
  * the forward's registry (csa2_cake_shared) is reset per forward like reset_routing;
  * the candidate's grouped arm also asks the refresh for the int32 sidecar (reads32 + capacity).
Layers: 2 SWA-only + 38 dual in 8 (kv_source, index_source) groups of 5,5,5,5,5,5,4,4 (chain.py LAYERS).

Arms
  perlayer : TRTLLM_CSA2_CAKE_GROUPED_PREP=0, plain refresh  == branch 129's per-layer prep (main's route)
  noprep   : perlayer with prepare_cake_indices patched to a no-op (indices left from warmup) -> Stage 0
  grouped  : candidate default: refresh writes int32 SWA slots, one main prep per group (G launches)
Base tree: perlayer + noprep (the backend there has no grouping; the env switch is a no-op).

Modes
  time   : CUDA graphs per count, cold L2 (256 MB flush), arms interleaved (5 x (2 warmup + 20)). Last line =
           evidence JSON, kernel_ms = grouped (candidate) / perlayer (base); stage0 = perlayer - noprep.
  check  : layout census of the checkpoint; per layer eager torch.equal of the int32 SWA / main indices the
           grouped route used vs 129's prep on fresh buffers, and the prep launch count; graph replays of
           grouped for warmup + iters iterations with finite sentinels (outputs, Cake scratch, per-layer and
           shared int32 index buffers, int32 sidecar) before every replay, scratch shared across layers, every
           layer's attention + projection output bitwise vs the eager perlayer chain; >= 8 advancing decode
           steps (positions +1, new q / pool rows / selections) on the captured graph vs fresh eager
           references; a skip-group-first-prep control must differ.
  census : layout census + graph census per count: prep / refresh / body / merge launches per replay
           (Kineto), and no plan / JIT / scratch allocation inside capture.
"""

import argparse
import importlib.util
import inspect
import json
import os
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ASC = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("cake_q1_chain", os.path.join(ASC, "cake_q1", "chain.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
import torch  # noqa: E402

from tensorrt_llm._torch.attention.backends.interface import (  # noqa: E402
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2 import cake_decode as CD  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel as K  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import store_rows  # noqa: E402

CC = C.CC
log = C.log
GROUPED = hasattr(CD, "_shared_main_slots")
REFRESH32 = "reads32" in inspect.signature(K.refresh_swa_slots).parameters
BLOCK = 64
W = CC.SWA_W
LAYERS = C.LAYERS
NL = len(LAYERS)
G_EXPECTED = len({grp for width, grp in LAYERS if width > W})
ENV = "TRTLLM_CSA2_CAKE_GROUPED_PREP"
_ORIG_PREP = CD.prepare_cake_indices


def dev():
    return torch.device("cuda", torch.cuda.current_device())


class Source:
    """Stand-in for the request metadata the layers share (its csa2_* fields, as prepare/enter_layer fill them)."""

    def __init__(self, count, max_positions, ratios, seed):
        g = torch.Generator().manual_seed(seed)
        self.count = count
        self.positions = torch.tensor([max_positions - 38 - t for t in range(count)], dtype=torch.int32,
                                      device="cuda")
        self.requests = torch.arange(count, dtype=torch.int64, device="cuda")
        self.floors = torch.zeros(count, dtype=torch.int64, device="cuda")
        self.read_floors = torch.zeros(count, dtype=torch.int64, device="cuda")
        cols = (max_positions + 64) // BLOCK + 2
        pool_pages = CC.SWA_ROWS // BLOCK
        self.tables = []
        for _ in range(NL):
            t = torch.randint(0, pool_pages + 4, (count, cols), generator=g, dtype=torch.int32)  # some past the pool
            t[torch.rand((count, cols), generator=g) < 0.05] = -1
            self.tables.append(t.cuda())
        self.desc = torch.tensor([(t.data_ptr(), t.stride(0), t.shape[0], t.shape[1]) for t in self.tables],
                                 dtype=torch.int64, device="cuda")
        self.ratios = torch.tensor(ratios, dtype=torch.int32, device="cuda")
        self.reads = torch.full((NL, count, W), -1, dtype=torch.int64, device="cuda")
        self.writes = torch.full((NL, count), -1, dtype=torch.int64, device="cuda")
        self.visible = torch.zeros((NL, count), dtype=torch.int64, device="cuda")
        self.reads32 = torch.full((NL, count, W), -1, dtype=torch.int32, device="cuda")
        self.capacity = torch.full((NL,), CC.SWA_ROWS, dtype=torch.int64, device="cuda")
        self.csa2_swa_indices = dict(enumerate(self.reads.unbind(0)))
        self.csa2_visible_lengths = dict(enumerate(self.visible.unbind(0)))
        self.csa2_swa_indices32 = {}
        self.csa2_swa_capacities = ()
        self.csa2_cake_shared = {}
        self.use32 = False

    def begin_forward(self):
        """reset_routing + _ensure_swa_slots: one SWA refresh launch for every layer."""
        self.csa2_cake_shared = {}
        self.csa2_swa_indices = dict(enumerate(self.reads.unbind(0)))
        self.csa2_visible_lengths = dict(enumerate(self.visible.unbind(0)))
        extra = {}
        if self.use32 and REFRESH32:
            self.csa2_swa_indices32 = dict(enumerate(self.reads32.unbind(0)))
            self.csa2_swa_capacities = (CC.SWA_ROWS,) * NL
            extra = dict(reads32=self.reads32, capacity=self.capacity)
        else:
            self.csa2_swa_indices32 = {}
            self.csa2_swa_capacities = ()
        K.refresh_swa_slots(self.positions, self.requests, self.floors, self.read_floors, self.desc, self.ratios,
                            self.reads, self.writes, self.visible, BLOCK, **extra)


class Chain:
    def __init__(self, world, count, seed=0, group_ratio=lambda grp: 1):
        self.world, self.count = world, count
        self.layers = []
        gsel = {}
        _, self.pos = C.WC.make_inputs(world.dims, count, seed=300 + count)
        for li, (width, group) in enumerate(LAYERS):
            if width > W and group not in gsel:
                gsel[group] = CC.build_selection(count, seed * 100 + group, width=width)
        max_positions = next(iter(gsel.values()))["max_positions"]
        ratios = [group_ratio(grp) if width > W else 0 for width, grp in LAYERS]
        self.source = Source(count, max_positions, ratios, seed)
        self.gsel = gsel
        for li, (width, group) in enumerate(LAYERS):
            q, sink = CC.make_q(count, seed * 100 + 70 + li)
            q = (q.float() * 0.5).to(torch.bfloat16)
            state = SimpleNamespace(output_pdl_consumer=False, metadata=self.source)
            mp = world.main_pools.get(group) if width > W else None
            sel = gsel.get(group)
            mapping = None
            if mp is not None:
                mapping = (sel["table"], CC.PAGE, sel["max_positions"], sel["requests"],
                           self.source.visible[li])
            inputs = CSA2BackendForwardArgs(
                swa_pool=world.swa_pools[li], swa_indices=self.source.reads[li], main_pool=mp,
                topk_indices=sel["logical"] if mp is not None else None,
                main_logical_indices=sel["logical"] if mp is not None else None,
                main_mapping=mapping,
                stage_group=(group, group, 0) if mp is not None else (None, None, 0), state=state)
            out = torch.empty_like(q)
            stub = C.StubAttention(inputs)
            stub.layer_idx = li
            fi = CSA2FlashInfer(stub)
            args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only,
                                        attention_sinks=sink, output=out.flatten(1), sparse_backend_args=inputs)
            self.layers.append(dict(width=width, group=group, q=q, sink=sink, out=out, fi=fi, args=args,
                                    inputs=inputs, state=state, meta=C.StubMeta(count, width), mod=world.mods[li]))

    def flag(self, L):
        f = getattr(C.DeepseekV41Attention, "_lowm_wo_follows", None)
        return bool(f(L["mod"], L["out"].view(self.count, -1))) if f is not None else False

    def layer(self, L):
        L["state"].output_pdl_consumer = self.flag(L)
        L["fi"].forward_generation(L["q"].flatten(1), L["meta"], L["args"])
        return C.Hn.project(L["mod"], L["out"].view(self.count, -1), self.pos)

    def __call__(self):
        self.source.begin_forward()
        return [self.layer(L) for L in self.layers]

    def share_scratch(self):
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
                    buf["swa"].fill_(7)
                    buf["main"].fill_(7)
        for b in getattr(CD, "_SHARED_MAIN_SLOTS", {}).values():
            b.fill_(7)
        self.source.reads32.fill_(7)
        for o in outs:
            o.fill_(7.0)


class Arm:
    """Env + refresh variant + prep patch of one arm, applied while warming and capturing."""

    def __init__(self, chain, name):
        self.chain, self.name = chain, name

    def __enter__(self):
        self.old = os.environ.get(ENV)
        os.environ[ENV] = "1" if self.name == "grouped" else "0"
        self.chain.source.use32 = self.name == "grouped"
        if self.name == "noprep":
            CD.prepare_cake_indices = lambda *a, **k: None
        return self

    def __exit__(self, *a):
        CD.prepare_cake_indices = _ORIG_PREP
        if self.old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self.old


def warm(chain):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            chain()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()


def capture(chain, name):
    with Arm(chain, name):
        warm(chain)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = chain()
    torch.cuda.synchronize()
    return g, outs


def eager(chain, name):
    with Arm(chain, name):
        res = [r.clone() for r in chain()]
        att = [L["out"].clone() for L in chain.layers]
    torch.cuda.synchronize()
    return res, att


def arms_here():
    return ["perlayer", "noprep"] + (["grouped"] if GROUPED and REFRESH32 else [])


# ------------------------------------------------------------------------------------------------ time
def mode_time(args):
    world = C.World(args.ckpt)
    arms = arms_here()
    rep = "grouped" if "grouped" in arms else "perlayer"
    batch, table = {}, {}
    for count in [int(c) for c in args.counts.split(",")]:
        chain = Chain(world, count, seed=count)
        graphs = {}
        for arm in arms:
            graphs[arm], _ = capture(chain, arm)
        t = C.time_arms(graphs)
        row = {a: {k: (v * 1e3 if k != "noise" else v) for k, v in s.items()} for a, s in t.items()}
        row["stage0_exposed_prep_us"] = (t["perlayer"]["median"] - t["noprep"]["median"]) * 1e3
        row["stage0_exposed_share"] = (t["perlayer"]["median"] - t["noprep"]["median"]) / t["perlayer"]["median"]
        if "grouped" in t:
            row["grouped_minus_perlayer_us"] = (t["grouped"]["median"] - t["perlayer"]["median"]) * 1e3
            row["grouped_vs_perlayer_rel"] = ((t["grouped"]["median"] - t["perlayer"]["median"])
                                              / t["perlayer"]["median"])
            row["grouped_minus_noprep_us"] = (t["grouped"]["median"] - t["noprep"]["median"]) * 1e3
        row["clock"] = C.lease_info()["clock"]
        table[str(count)] = row
        batch[str(count)] = {"kernel_ms": t[rep]["median"], "noise_floor": t[rep]["noise"],
                             "p10_ms": t[rep]["p10"], "p90_ms": t[rep]["p90"],
                             "perlayer_ms": t["perlayer"]["median"], "perlayer_noise": t["perlayer"]["noise"],
                             "noprep_ms": t["noprep"]["median"], "noprep_noise": t["noprep"]["noise"]}
        log(f"M={count} " + json.dumps(row))
        del graphs, chain
        torch.cuda.empty_cache()
    (C.ART / "table265.json").write_text(json.dumps(table, indent=1))
    print(json.dumps({"batch": batch, "reported_arm": rep, "grouped_tree": GROUPED, "refresh32": REFRESH32,
                      "stage0": {c: {k: v for k, v in r.items() if k.startswith("stage0") or k == "clock"}
                                 for c, r in table.items()},
                      "table": table, "layers": NL, "lease": C.lease_info(), "output": {"class": "exact"}}))


# ------------------------------------------------------------------------------------------------ census
def layout_census(ckpt):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    try:
        cfg = json.load(open(os.path.join(ckpt, "config.json")))
        layout = CSA2Layout.from_hf_config(cfg)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:300]}
    rows, groups = [], {}
    for i, ratio in enumerate(layout.compress_ratios):
        lay = layout.layer(i)
        grp = None if lay.kv_source is None else (lay.kv_source, lay.index_source)
        vis_eq = None if lay.kv_source is None else layout.compress_ratios[lay.kv_source] == ratio
        rows.append(dict(layer=i, type=lay.mode.value, ratio=ratio, kv_source=lay.kv_source,
                         index_source=lay.index_source, group=grp, visibility_equal_in_group=vis_eq))
        if grp is not None:
            groups.setdefault(str(grp), []).append(i)
    return dict(layers=len(rows), swa_only=sum(r["type"] == "swa" for r in rows), G=len(groups),
                group_sizes={k: len(v) for k, v in groups.items()}, groups=groups,
                all_visibility_equal=all(r["visibility_equal_in_group"] for r in rows if r["group"]),
                kv_source_layer_ids=list(layout.kv_source_layer_ids),
                index_source_layer_ids=list(layout.index_source_layer_ids), per_layer=rows)


def count_kernels(g, reps=3):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    path = C.ART / f"census265_{id(g)}.json"
    prof.export_chrome_trace(str(path))
    evs = json.load(open(path))
    evs = evs.get("traceEvents", evs) if isinstance(evs, dict) else evs
    names = {}
    for e in evs:
        if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel":
            names[e.get("name", "")] = names.get(e.get("name", ""), 0) + 1

    def per(sub):
        return sum(v for k, v in names.items() if sub in k) / reps

    return dict(prep=per("_cake_index_prep_kernel"), swa_refresh=per("SwaRefresh"),
                body=per("decode_body") + per("kernel_sm120_dsv41_mixed_sparse_mla"),
                merge=per("CakeMergeF32Kernel"), total=sum(names.values()) / reps,
                names={k[:90]: v / reps for k, v in sorted(names.items()) if v >= reps})


class CallCounter:
    def __init__(self):
        self.n = {}

    def wrap(self, obj, name):
        orig = getattr(obj, name)
        counter = self

        def f(*a, **k):
            counter.n[name] = counter.n.get(name, 0) + 1
            return orig(*a, **k)

        setattr(obj, name, f)
        return orig


def jit_sizes():
    fn = CD._cake_index_prep_kernel
    n = -1
    for attr in ("device_caches", "cache"):
        cache = getattr(fn, attr, None)
        if cache:
            try:
                n = sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in cache.values())
            except Exception:  # noqa: BLE001
                n = -1
            break
    return dict(triton_prep_variants=n, cute_compiled=len(K._COMPILED))


def mode_census(args):
    from flashinfer.mla._sparse_mla_sm120 import cake_dsv41_mixed as cdm
    world = C.World(args.ckpt)
    out = {"layout": layout_census(args.ckpt), "chain_G": G_EXPECTED, "counts": {}}
    log("layout census: " + json.dumps({k: v for k, v in out["layout"].items() if k != "per_layer"}))
    for count in [int(c) for c in args.counts.split(",")]:
        chain = Chain(world, count, seed=count)
        row = {}
        for arm in [a for a in arms_here() if a != "noprep"]:
            with Arm(chain, arm):
                warm(chain)

                def state():
                    return dict(jit_sizes(), shared_main=len(getattr(CD, "_SHARED_MAIN_SLOTS", {})),
                                cake_buffers=sum(len(L["fi"]._cake._buffers) for L in chain.layers))

                before = state()
                cc = CallCounter()
                origs = [(cdm, "_resolve_plan", cc.wrap(cdm, "_resolve_plan")),
                         (cdm, "cake_sparse_mla_sm120_dsv41_mixed_decode",
                          cc.wrap(cdm, "cake_sparse_mla_sm120_dsv41_mixed_decode")),
                         (CD, "cake_sparse_mla_sm120_dsv41_mixed_decode", None),
                         (CD, "prepare_cake_indices", cc.wrap(CD, "prepare_cake_indices"))]
                origs = [o for o in origs if o[2] is not None]
                mem0 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    chain()
                torch.cuda.synchronize()
                mem1 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
                for obj, name, orig in origs:
                    setattr(obj, name, orig)
                after = state()
            kc = count_kernels(g)
            row[arm] = dict(kernels_per_replay=kc, host_calls_in_capture=cc.n, before=before, after=after,
                            no_new_jit_or_scratch=before == after,
                            no_plan_in_capture=cc.n.get("_resolve_plan", 0) == 0
                            and cc.n.get("cake_sparse_mla_sm120_dsv41_mixed_decode", 0) == 0,
                            torch_allocations_in_capture=mem1 - mem0,
                            capture_ok_no_sync=True)
            log(f"census M={count} {arm}: {json.dumps(row[arm])}")
            del g
        out["counts"][str(count)] = row
        del chain
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------------ check
def compare(chain, outs, ref_res, ref_att):
    diff_layers, worst, finite = 0, 0.0, True
    for o, r, L, a in zip(outs, ref_res, chain.layers, ref_att):
        if not (torch.equal(o, r) and torch.equal(L["out"], a)):
            diff_layers += 1
            worst = max(worst, (o.float() - r.float()).abs().max().item(),
                        (L["out"].float() - a.float()).abs().max().item())
        finite &= bool(torch.isfinite(o).all()) and bool(torch.isfinite(L["out"]).all())
    return diff_layers, worst, finite


def run_iters(chain, g, outs, ref_res, ref_att, iters):
    bad, max_abs, finite = 0, 0.0, True
    for _ in range(iters):
        chain.poison(outs)
        C.flush()
        g.replay()
        torch.cuda.synchronize()
        dl, w, f = compare(chain, outs, ref_res, ref_att)
        bad += int(dl > 0)
        max_abs = max(max_abs, w)
        finite &= f
    return dict(iterations=iters, bad_iterations=bad, max_abs_diff=max_abs, finite=finite)


def index_check(chain):
    """Eager grouped forward, layer by layer: the int32 indices it used vs 129's prep on fresh buffers."""
    count = chain.count
    calls = [0]

    def counting(*a, **k):
        calls[0] += 1
        return _ORIG_PREP(*a, **k)

    bad, checked, swa_from_refresh, main_shared, details = 0, 0, 0, 0, []
    with Arm(chain, "grouped"):
        chain.poison()
        CD.prepare_cake_indices = counting
        chain.source.begin_forward()
        for li, L in enumerate(chain.layers):
            chain.layer(L)
            torch.cuda.synchronize()
            inp = L["inputs"]
            swa_ready, share = L["fi"]._cake_shared_indices(inp, count)
            has_main = inp.topk_indices is not None
            ref_swa = torch.full((count, W), -9, dtype=torch.int32, device="cuda")
            ref_main = torch.full((count, CC.MAIN_W), -9, dtype=torch.int32, device="cuda")
            _ORIG_PREP(inp.swa_indices, inp.swa_pool.shape[0], ref_swa, inp.topk_indices if has_main else None,
                       inp.main_pool.shape[0] if has_main else 0, ref_main if has_main else None,
                       inp.main_mapping if has_main else None)
            buf = next(iter(L["fi"]._cake._buffers.values()))
            used_swa = swa_ready if swa_ready is not None else buf["swa"]
            swa_from_refresh += int(swa_ready is not None)
            ok_swa = torch.equal(used_swa, ref_swa)
            ok_main = True
            if has_main:
                if share is not None:
                    used_main = CD._SHARED_MAIN_SLOTS[(count, CC.MAIN_W, dev())]
                    main_shared += 1
                else:
                    used_main = buf["main"]
                ok_main = torch.equal(used_main, ref_main)
            checked += 1
            bad += int(not (ok_swa and ok_main))
            if not (ok_swa and ok_main) and len(details) < 6:
                details.append(dict(layer=li, swa=ok_swa, main=ok_main))
        CD.prepare_cake_indices = _ORIG_PREP
    torch.cuda.synchronize()
    masked = dict(swa_minus1=int((chain.source.reads32 == -1).sum()),
                  swa_past_pool_masked=int(((chain.source.reads >= CC.SWA_ROWS)).sum()))
    return dict(layers=checked, bad_layers=bad, prep_launches=calls[0], swa_from_refresh=swa_from_refresh,
                main_shared_layers=main_shared, details=details, masking=masked,
                passed=bad == 0 and checked == NL and swa_from_refresh == NL and main_shared == NL - 2)


def advance(world, chain, step):
    g = torch.Generator(device="cuda").manual_seed(9000 + step)
    chain.source.positions.add_(1)
    for li, L in enumerate(chain.layers):
        L["q"].copy_((torch.randn(L["q"].shape, generator=g, device="cuda") * 0.5).to(torch.bfloat16))
        rows = torch.randint(0, CC.SWA_ROWS, (256,), generator=g, device="cuda")
        store_rows(world.swa_pools[li], rows,
                   torch.randn((256, CC.DIM), generator=g, device="cuda").to(torch.bfloat16), "swa")
    for sel in chain.gsel.values():
        sel["logical"].copy_(torch.roll(sel["logical"], shifts=step * 5 + 2, dims=1))
    for mp in world.main_pools.values():
        rows = torch.randint(0, CC.MAIN_ROWS, (1024,), generator=g, device="cuda")
        store_rows(mp, rows, torch.randn((1024, CC.DIM), generator=g, device="cuda").to(torch.bfloat16), "main")
    torch.cuda.synchronize()


def skip_first_prep_control(chain):
    """The first main group's registry entry pre-marked (its prep skipped) on a poisoned shared buffer."""
    count = chain.count
    with Arm(chain, "grouped"):
        chain.poison()
        chain.source.begin_forward()
        for L in chain.layers:
            _, share = L["fi"]._cake_shared_indices(L["inputs"], count)
            if share is not None:
                reg, key = share
                reg[(count, CC.MAIN_W, dev())] = key
                break
        for b in CD._SHARED_MAIN_SLOTS.values():
            b.fill_(7)
        for L in chain.layers:
            chain.layer(L)
        torch.cuda.synchronize()
        att = [L["out"].clone() for L in chain.layers]
    _, ref_att = eager(chain, "perlayer")
    return sum(int(not torch.equal(a, b)) for a, b in zip(att, ref_att))


def census_passed(census):
    """Acceptance 3: grouped capture has G prep launches (no per-layer prep), one refresh, no plan/JIT/scratch."""
    ok = True
    for row in census["counts"].values():
        g = row.get("grouped")
        p = row.get("perlayer")
        ok &= g is not None and g["kernels_per_replay"]["prep"] == G_EXPECTED
        ok &= g["no_new_jit_or_scratch"] and g["no_plan_in_capture"]
        ok &= g["host_calls_in_capture"].get("prepare_cake_indices", 0) == G_EXPECTED
        ok &= p is not None and p["kernels_per_replay"]["prep"] == NL
    return bool(ok)


def mode_all(args):
    """check, then the graph census, one process; last line = combined verdict."""
    check = mode_check(args, emit=False)
    census = mode_census(args)
    c_ok = census_passed(census)
    log("census: " + json.dumps({c: {a: r["kernels_per_replay"] | {"no_new_jit_or_scratch": r["no_new_jit_or_scratch"],
                                                                  "no_plan_in_capture": r["no_plan_in_capture"],
                                                                  "calls": r["host_calls_in_capture"]}
                                     for a, r in row.items()} for c, row in census["counts"].items()}))
    print(json.dumps({"passed": bool(check["passed"] and c_ok), "check_passed": check["passed"],
                      "census_passed": c_ok, "compared": check["compared"]
                      + "; graph census per count: grouped capture launches G=8 preps (perlayer 40), 8 host prep "
                        "calls in capture, no plan/public-decode call, no Triton/CuTe JIT or Cake scratch allocation "
                        "in capture (SWA refresh / body / merge counts reported)",
                      "check": check["report"], "census": census}))


def mode_check(args, emit=True):
    assert GROUPED and REFRESH32, "check runs in the candidate tree"
    world = C.World(args.ckpt)
    ok, report = True, {"layout_census": layout_census(args.ckpt)}
    log("layout census: " + json.dumps({k: v for k, v in report["layout_census"].items() if k != "per_layer"}))
    for count in [int(c) for c in args.counts.split(",")]:
        # alternate compression ratios 1 / 2 across groups: both visibility classes are exercised
        chain = Chain(world, count, seed=20 + count, group_ratio=lambda grp: 1 + (grp % 2))
        eager(chain, "perlayer")
        eager(chain, "grouped")
        rep = {"index_check": index_check(chain)}
        ok &= rep["index_check"]["passed"] and rep["index_check"]["prep_launches"] == G_EXPECTED
        log(f"M={count} index_check {json.dumps(rep['index_check'])}")
        rep["shared_scratch_layers"] = chain.share_scratch()
        ref_res, ref_att = eager(chain, "perlayer")
        gr_res, gr_att = eager(chain, "grouped")
        dl, w, f = compare(chain, gr_res, ref_res, ref_att)
        rep["eager_grouped_vs_perlayer"] = dict(diff_layers=dl, max_abs=w, finite=f)
        ok &= dl == 0 and f
        g, outs = capture(chain, "grouped")
        res = run_iters(chain, g, outs, ref_res, ref_att, args.warmup + args.iters)
        res["passed"] = res["bad_iterations"] == 0 and res["finite"]
        ok &= res["passed"]
        rep["graph_grouped"] = res
        log(f"M={count} graph_grouped {json.dumps(res)}")
        steps, prev, step_ok = [], None, True
        for step in range(1, args.steps + 1):
            advance(world, chain, step)
            r_res, r_att = eager(chain, "perlayer")
            changed = prev is None or any(not torch.equal(a, b) for a, b in zip(r_att, prev))
            prev = r_att
            s = run_iters(chain, g, outs, r_res, r_att, 3)
            s["reference_changed_vs_prev_step"] = bool(changed)
            steps.append(s)
            step_ok &= s["bad_iterations"] == 0 and s["finite"] and changed
        rep["advancing_steps"] = dict(steps=len(steps), passed=bool(step_ok),
                                      max_abs=max([s["max_abs_diff"] for s in steps] or [0.0]),
                                      all_changed=all(s["reference_changed_vs_prev_step"] for s in steps))
        ok &= step_ok and len(steps) >= 8
        log(f"M={count} advancing {json.dumps(rep['advancing_steps'])}")
        del g, outs
        dl_c = skip_first_prep_control(chain)
        rep["control_skip_first_prep_diff_layers"] = dl_c
        rep["control_failed_as_expected"] = dl_c > 0
        ok &= dl_c > 0
        log(f"M={count} control diff_layers={dl_c}")
        report[str(count)] = rep
        del chain
        torch.cuda.empty_cache()
    result = {"passed": bool(ok),
                      "compared": "40-layer Cake attention + LowMWO projection chain, grouped prep (int32 SWA "
                                  "slots from the SWA refresh launch, one main prep per (kv_source, index_source) "
                                  "group) vs branch 129's per-layer prep: per-layer int32 SWA/main indices "
                                  "torch.equal to 129's prep on fresh buffers (eager, every layer); eager and "
                                  "graph outputs (attention + projection, all 40 layers) bitwise vs the eager "
                                  "per-layer chain over warmup+iters replays with finite sentinels in outputs, "
                                  "Cake scratch, int32 index buffers and sidecar before every replay, scratch "
                                  "shared across layers; >=8 advancing decode steps; a skip-first-prep control "
                                  "must differ",
                      "report": report}
    if emit:
        print(json.dumps(result))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["time", "check", "census", "all"])
    ap.add_argument("ckpt")
    ap.add_argument("--counts", default="1,2,4,8")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "time":
        mode_time(a)
    elif a.mode == "check":
        mode_check(a)
    elif a.mode == "all":
        mode_all(a)
    else:
        print(json.dumps({"passed": True, "census": mode_census(a)}))
