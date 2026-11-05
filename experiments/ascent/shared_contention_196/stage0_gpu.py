"""Idea 196 STAGE 0 (GPU part): contention of the aux-stream decode shared expert on the
routed FlashInfer SM12x decode-MoE chain (DeepSeek-V4.1-Flash, one TP4 rank, SM120).

No source edit.  One CUDA graph of NL=40 MoE layers with the production fork/join
(tensorrt_llm/_torch/modules/multi_stream_utils.maybe_execute_in_parallel, as
DeepseekV4MoE.forward: event0.record -> routed on MAIN -> aux waits event0 -> shared on aux
-> event1.record -> main waits event1 -> shared + routed add).

  routed = the production leaf Sm12xW4a8DecodeState.run (route -> fc1_act_q1 -> fc2_finalize),
           forced-D routing (exactly D distinct local experts per layer, every layer a fresh
           expert set: 5 weight sets x 96 (or 192) experts, layer l uses set l % 5 and experts
           perm[(l // 5) * D : (l // 5 + 1) * D], so no layer re-reads a previous layer's
           experts; plus a read-only 512 MB flush before every replay -> cold L2).
  shared = the real FP8 block-scale shared GatedMLP built as DeepseekV4MoE builds it
           (experiments/ascent/shared_row_split/common.py: per_token_quant + gate_up +
           SwiGLU(limit) + down), * shared_output_scale (0.5); 40 distinct copies.

Arms (interleaved per block, same process, same GPU):
  A  routed only
  B  routed + shared at 1152 cols per rank (main: shared_tp 2, each half computed twice)
  C  routed + shared at  640 cols per rank (the 4-way SHARD design's max rank: 5 of 18 blocks)
  D  shared (1152 cols) alone
  E  shared (640 cols) alone (context)

Per arm, per block: Kineto trace of REPLAYS graph replays; span = first node after the flush to
the last node's end; per-layer critical path = span / NL; fc1/fc2/shared node sums / NL.
Median over NB blocks, dispersion = (max - min) / median over blocks, and replay-level
(p90 - p10) / p50.  SM clock (NVML, by UUID) logged per block.

Env: BENCH_SHAPE=ep4 (96 x 2304, main's layout) | ep2tp2 (192 x 1152, context);
     S0_CASES (default 'M1:0,1,2,3,4,5,6;M2:0,1,2,3,4,6,8;M4:1,2,4,6,8,10,12'), S0_NB (6),
     S0_REPLAYS (10), CKPT (checkpoint dir for config.json / quant config).
Output: $AUTO_DEV_ARTIFACTS/stage0_<shape>.json and a last-line JSON summary.
Run:    {python} experiments/ascent/shared_contention_196/stage0_gpu.py {checkpoint}
"""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))
if len(sys.argv) > 1 and not os.environ.get("CKPT"):
    os.environ["CKPT"] = sys.argv[1]

import torch  # noqa: E402

import common as _common  # noqa: E402  (w4a8_fi_chain/common.py)

SHAPE = os.environ.get("BENCH_SHAPE", "ep4")
if SHAPE == "ep2tp2":
    _common.E, _common.I = 192, 1152

from common import E_GLOBAL, H, TOPK, log, make_weights, trtllm_layout  # noqa: E402

E = _common.E
NL = 40
NSETS = 5  # 5 x 96 experts: 8 layers per set x D <= 12 distinct experts each, no reuse
NB = int(os.environ.get("S0_NB", "6"))
REPLAYS = int(os.environ.get("S0_REPLAYS", "10"))
CASES = os.environ.get("S0_CASES", "M1:0,1,2,3,4,5,6;M2:0,1,2,3,4,6,8;M4:1,2,4,6,8,10,12")
TAG = f"STAGE0-{SHAPE}"
FLUSH = None


def parse_cases():
    out = []
    for part in CASES.split(";"):
        m, ds = part.split(":")
        m = int(m[1:])
        for d in ds.split(","):
            d = int(d)
            if d <= TOPK * m and (NL // NSETS) * d <= E:
                out.append((m, d))
    return out


def forced_ids(m, experts, seed):
    """Exactly len(experts) distinct local experts over m tokens; other picks off-rank."""
    g = torch.Generator(device="cpu")
    g.manual_seed(7000 + 31 * m + len(experts) + seed)
    d = len(experts)
    per_tok = [[] for _ in range(m)]
    for i, e in enumerate(experts):
        per_tok[i % m].append(e)
    if 0 < d < m:  # tokens without a local pick reuse the set (pairs = max(d, m))
        for t in range(d, m):
            per_tok[t].append(experts[t % d])
    ids = torch.empty(m, TOPK, dtype=torch.int32)
    for t in range(m):
        loc = per_tok[t]
        rem = (torch.randperm(E_GLOBAL - E, generator=g)[: TOPK - len(loc)] + E).tolist()
        row = torch.tensor(loc + rem, dtype=torch.int64)
        ids[t] = row[torch.randperm(TOPK, generator=g)].to(torch.int32)
    w = torch.rand(m, TOPK, generator=g) + 0.1
    w = (w / w.sum(-1, keepdim=True) * 1.5).float()
    return ids.cuda(), w.cuda()


def build_shared(inter_total, n_copies):
    """n_copies of one rank's shared GatedMLP with intermediate_size inter_total (pre-TP).

    shared_row_split/common.py is also named common.py: load it under another name and
    override its read_dims so the shared intermediate is inter_total (2304 -> 1152 cols per
    rank at shared_tp 2; 1280 -> 640 cols per rank).
    """
    spec = importlib.util.spec_from_file_location(
        "srs_common", os.path.join(HERE, "..", "shared_row_split", "common.py"))
    srs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srs)
    orig = srs.read_dims

    def patched():
        hidden, _inter, limit, cfg = orig()
        return hidden, inter_total, limit, cfg

    srs.read_dims = patched
    mlp, _shared_tp, scale, info = srs.build_shared_mlp(rank=0)
    mlps = [mlp] + [copy.deepcopy(mlp) for _ in range(n_copies - 1)]
    return mlps, scale, info


class Layers:
    def __init__(self, states, sh_b, sh_c, scale, m, d):
        from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
        self.mep = maybe_execute_in_parallel
        self.states, self.sh = states, {"B": sh_b, "C": sh_c, "D": sh_b, "E": sh_c}
        self.scale = scale
        self.aux = torch.cuda.Stream()
        self.ev0 = torch.cuda.Event()
        self.ev1 = torch.cuda.Event()
        g = torch.Generator(device="cpu").manual_seed(99)
        perms = [torch.randperm(E, generator=g).tolist() for _ in range(NSETS)]
        self.xs, self.ids, self.ws = [], [], []
        for layer in range(NL):
            j = layer // NSETS
            experts = perms[layer % NSETS][j * d:(j + 1) * d]
            ids, w = forced_ids(m, experts, layer)
            self.ids.append(ids)
            self.ws.append(w)
            self.xs.append(torch.randn(m, H, dtype=torch.bfloat16, device="cuda") * 0.5)

    def shared(self, arm, layer):
        out = self.sh[arm][layer](self.xs[layer])
        if self.scale is not None:
            out *= self.scale
        return out

    def run(self, arm):
        outs = []
        for layer in range(NL):
            st = self.states[layer % NSETS]
            if arm == "A":
                outs.append(st.run(self.xs[layer], self.ids[layer], self.ws[layer]))
            elif arm in ("D", "E"):
                outs.append(self.shared(arm, layer))
            else:
                r, s = self.mep(lambda: st.run(self.xs[layer], self.ids[layer], self.ws[layer]),
                                lambda: self.shared(arm, layer), self.ev0, self.ev1, self.aux)
                outs.append(s + r)
        return outs


def capture(fn):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with with_multi_stream(True):
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = fn()
    torch.cuda.synchronize()
    return g, outs


def max_diff(outs, ref):
    return max(float((o.float() - r.float()).abs().max().item()) for o, r in zip(outs, ref))


def node_split(reps, names):
    """Per in-layer position of the nodes whose name is in `names` (a serial stream, so
    position = index % k): name, median over replays of the mean-over-layers us, grid, block."""
    per_pos = None
    for r in reps:
        sel = sorted((e for e in r if e["name"] in names), key=lambda e: float(e["ts"]))
        if not sel or len(sel) % NL:
            continue
        k = len(sel) // NL
        if per_pos is None:
            per_pos = [{"name": sel[p]["name"], "grid": sel[p].get("args", {}).get("grid"),
                        "block": sel[p].get("args", {}).get("block"), "us": []}
                       for p in range(k)]
        if len(per_pos) != k:
            continue
        for p in range(k):
            per_pos[p]["us"].append(sum(float(sel[l * k + p]["dur"]) for l in range(NL)) / NL)
    out = []
    for p in per_pos or []:
        out.append({"name": p["name"][:110], "grid": p["grid"], "block": p["block"],
                    "us": round(pct(p["us"], 0.5), 3) if p["us"] else None})
    return out


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


WARM = int(os.environ.get("S0_WARM", "30"))


def warm(g, n=WARM):
    for _ in range(n):
        FLUSH.sum()
        g.replay()


def time_events(g, n, clk=None):
    """CUDA-event span per replay (us per layer), flush before each replay, after a warm-up
    so the SM clock is up; the NVML clock is sampled while the GPU is busy."""
    warm(g)
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(n)]
    for a, b in evs:
        FLUSH.sum()
        a.record()
        g.replay()
        b.record()
    if clk is not None:
        clk()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) * 1000.0 / NL for a, b in evs]


def kineto(g):
    from torch.profiler import ProfilerActivity, profile
    warm(g)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPLAYS):
            FLUSH.sum()
            g.replay()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.gettempdir(), f"s0_{os.getpid()}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    os.remove(path)
    evs = sorted((e for e in tr.get("traceEvents", [])
                  if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")),
                 key=lambda e: float(e["ts"]))
    reps, cur = [], None
    for e in evs:
        if "reduce_kernel" in e["name"] and float(e["dur"]) > 100.0:
            if cur:
                reps.append(cur)
            cur = []
        elif cur is not None:
            cur.append(e)
    if cur:
        reps.append(cur)
    return reps


def summarize(reps, shared_names, routed_names):
    spans, f1, f2, shs, nodes = [], [], [], [], []
    for r in reps:
        if not r:
            continue
        st = min(float(e["ts"]) for e in r)
        en = max(float(e["ts"]) + float(e["dur"]) for e in r)
        spans.append((en - st) / NL)
        f1.append(sum(float(e["dur"]) for e in r if "fc1" in e["name"].lower()) / NL)
        f2.append(sum(float(e["dur"]) for e in r if "fc2" in e["name"].lower()) / NL)
        shs.append(sum(float(e["dur"]) for e in r
                       if e["name"] in shared_names and e["name"] not in routed_names) / NL)
        nodes.append(len(r))
    return {"span": pct(spans, 0.5), "span_rnoise": (pct(spans, 0.9) - pct(spans, 0.1))
            / max(pct(spans, 0.5), 1e-9), "fc1": pct(f1, 0.5), "fc2": pct(f2, 0.5),
            "shared": pct(shs, 0.5), "nodes": nodes[0] if nodes else 0, "replays": len(spans)}


def nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        # NVML indexes physical GPUs (CUDA_VISIBLE_DEVICES does not remap it): go by UUID.
        u = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        u = u if u.startswith("GPU-") else "GPU-" + u
        try:
            h = pynvml.nvmlDeviceGetHandleByUUID(u)
        except Exception:  # noqa: BLE001
            h = pynvml.nvmlDeviceGetHandleByUUID(u.encode())
        uuid = pynvml.nvmlDeviceGetUUID(h)
        return h, (uuid.decode() if isinstance(uuid, bytes) else uuid), pynvml
    except Exception:  # noqa: BLE001
        return None, None, None


def main():
    global FLUSH
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf

    h, uuid, pynvml = nvml()
    log(f"{TAG}: GPU {torch.cuda.get_device_name()} uuid={uuid} E={E} I={_common.I} NL={NL} "
        f"NB={NB} REPLAYS={REPLAYS} leaf={leaf.__file__}")
    torch.manual_seed(0)
    states = []
    for s in range(NSETS):
        w1, s1, w2, s2 = make_weights(seed=s)
        w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
        del s1, s2
        st = leaf.Sm12xW4a8DecodeState(
            w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
            num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=True)
        states.append(st)
        log(f"{TAG}: weight set {s} built; mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    states[0].warmup(TOPK)
    sh_b, scale, info_b = build_shared(2304, NL)
    sh_c, scale_c, info_c = build_shared(1280, NL)
    assert scale == scale_c
    log(f"{TAG}: shared B {info_b}")
    log(f"{TAG}: shared C {info_c}")
    log(f"{TAG}: shared per-rank weights B gate_up {tuple(sh_b[0].gate_up_proj.weight.shape)} "
        f"down {tuple(sh_b[0].down_proj.weight.shape)} | C gate_up "
        f"{tuple(sh_c[0].gate_up_proj.weight.shape)} down {tuple(sh_c[0].down_proj.weight.shape)}"
        f"; mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

    table, clocks = [], []
    t0 = time.time()
    arms = ["A", "B", "C", "D", "E"]
    # pre-warm: keep the GPU busy until the SM clock settles (it steps 2295 -> 2422 MHz)
    prev = None
    for i in range(60):
        for _ in range(100):
            FLUSH.sum()
        c = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM) if h is not None else None
        torch.cuda.synchronize()
        if i >= 10 and c is not None and c == prev:
            break
        prev = c
    log(f"{TAG}: pre-warm done after {i + 1} rounds, SM {prev} MHz")

    def case_iter():
        """Re-run a case (up to 2 more times) when any arm's block dispersion is > 3%."""
        for c in parse_cases():
            for att in range(3):
                yield c
                r = table[-1]
                worst_d = max(r[a]["disp"] for a in arms)
                r["attempt"] = att + 1
                if worst_d <= 0.03 or att == 2:
                    break
                log(f"{TAG} M{c[0]}_D{c[1]}: block dispersion {worst_d:.4f} > 3%, re-running")
                table.pop()

    for (m, d) in case_iter():
        lay = Layers(states, sh_b, sh_c, scale, m, d)
        caps = {a: capture(lambda a=a: lay.run(a)) for a in arms}
        graphs = {a: caps[a][0] for a in arms}
        outs = {a: caps[a][1] for a in arms}
        # reference snapshot: first replay of each arm after capture
        refs = {}
        for a in arms:
            FLUSH.sum()
            graphs[a].replay()
            torch.cuda.synchronize()
            refs[a] = [o.clone() for o in outs[a]]
        # B = routed + shared(1152): compare with A's routed + D's shared from their graphs
        b_vs_ad = max(float((b.float() - (r.float() + s.float())).abs().max().item())
                      for b, r, s in zip(refs["B"], refs["A"], refs["D"]))
        c_vs_ae = max(float((c.float() - (r.float() + s.float())).abs().max().item())
                      for c, r, s in zip(refs["C"], refs["A"], refs["E"]))
        finite = all(bool(torch.isfinite(o.float()).all().item()) for a in arms for o in refs[a])
        # node classification from one trace per arm: shared nodes = names seen in the
        # shared-alone graphs and never in the routed-only graph
        names = {a: {e["name"] for r in kineto(graphs[a]) for e in r} for a in ("A", "D", "E")}
        routed_names = names["A"]
        shared_names = names["D"] | names["E"]
        sh_only = shared_names - routed_names
        blocks = {a: [] for a in arms}
        splits = {k: [] for k in ("D_sh", "B_sh", "E_sh", "C_sh", "A_rt", "B_rt", "C_rt")}
        rep_diff = {a: 0.0 for a in arms}
        for _b in range(NB):
            blk_clk = []

            def clk():
                if h is not None:
                    blk_clk.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))

            for a in arms[_b % len(arms):] + arms[:_b % len(arms)]:  # rotate arm order
                ev = time_events(graphs[a], REPLAYS, clk)
                reps = kineto(graphs[a])
                sm = summarize(reps, shared_names, routed_names)
                sm["kspan"] = sm["span"]
                sm["span"] = pct(ev, 0.5)
                sm["span_rnoise"] = (pct(ev, 0.9) - pct(ev, 0.1)) / max(pct(ev, 0.5), 1e-9)
                blocks[a].append(sm)
                torch.cuda.synchronize()
                rep_diff[a] = max(rep_diff[a], max_diff(outs[a], refs[a]))
                if a in ("B", "C", "D", "E"):
                    splits[f"{a}_sh"].append(node_split(reps, sh_only))
                if a in ("A", "B", "C"):
                    splits[f"{a}_rt"].append(node_split(reps, routed_names))
            if blk_clk:
                clocks.append(min(blk_clk))
        row = {"M": m, "nD": d,
               "check": {"B_vs_A_plus_D": b_vs_ad, "C_vs_A_plus_E": c_vs_ae, "finite": finite,
                         "replay_maxdiff_vs_first": rep_diff}}
        nodes_out = {}
        for k, lst in splits.items():
            lst = [x for x in lst if x]
            if not lst:
                nodes_out[k] = []
                continue
            npos = min(len(x) for x in lst)
            nodes_out[k] = [{**lst[0][p], "us": round(pct([x[p]["us"] for x in lst
                                                             if x[p]["us"] is not None], 0.5), 3)}
                            for p in range(npos)]
        row["nodes_split"] = nodes_out
        gu_n, dn_n = sh_b[0].gate_up_proj.weight.shape[0], sh_b[0].down_proj.weight.shape[0]
        row["tiles_main"] = {"gate_up_N": int(gu_n), "gate_up_tiles": -(-m // 32) * -(-int(gu_n) // 128),
                             "down_N": int(dn_n), "down_tiles": -(-m // 32) * -(-int(dn_n) // 128)}
        dsh, bsh = nodes_out.get("D_sh", []), nodes_out.get("B_sh", [])
        log(f"{TAG} M{m}_D{d} shared per-node D vs B: " + "; ".join(
            f"[{p}] {x['name'][:60]} grid {x['grid']} block {x['block']} D {x['us']} / B "
            f"{bsh[p]['us'] if p < len(bsh) else None} us" for p, x in enumerate(dsh))
            + f" | tiles {row['tiles_main']}")
        art, brt = nodes_out.get("A_rt", []), nodes_out.get("B_rt", [])
        log(f"{TAG} M{m}_D{d} routed per-node A vs B: " + "; ".join(
            f"[{p}] {x['name'][:50]} A {x['us']} / B {brt[p]['us'] if p < len(brt) else None} us"
            for p, x in enumerate(art)))
        log(f"{TAG} M{m}_D{d} check: B-(A+D) {b_vs_ad:.3g} C-(A+E) {c_vs_ae:.3g} finite {finite} "
            f"replay maxdiff {rep_diff}")
        for a in arms:
            sp = [x["span"] for x in blocks[a]]
            med = pct(sp, 0.5)
            row[a] = {
                "us": round(med, 3),
                "disp": round((max(sp) - min(sp)) / max(med, 1e-9), 4),
                "blk_std": round(float(torch.tensor(sp).std().item()) if len(sp) > 1 else 0.0, 3),
                "rnoise": round(pct([x["span_rnoise"] for x in blocks[a]], 0.5), 4),
                "kspan": round(pct([x["kspan"] for x in blocks[a]], 0.5), 3),
                "fc1": round(pct([x["fc1"] for x in blocks[a]], 0.5), 3),
                "fc2": round(pct([x["fc2"] for x in blocks[a]], 0.5), 3),
                "shared": round(pct([x["shared"] for x in blocks[a]], 0.5), 3),
                "nodes": blocks[a][0]["nodes"], "blocks": [round(v, 3) for v in sp]}
        row["B_minus_A"] = round(row["B"]["us"] - row["A"]["us"], 3)
        row["B_minus_C"] = round(row["B"]["us"] - row["C"]["us"], 3)
        row["C_minus_A"] = round(row["C"]["us"] - row["A"]["us"], 3)
        table.append(row)
        clk = clocks[-NB:] if clocks else []
        log(f"{TAG} M{m}_D{d}: " + " | ".join(
            f"{a} {row[a]['us']}us (kineto {row[a]['kspan']}, disp {row[a]['disp']}, rn "
            f"{row[a]['rnoise']}, fc1 "
            f"{row[a]['fc1']} fc2 {row[a]['fc2']} sh {row[a]['shared']} n{row[a]['nodes']})"
            for a in arms) + f" || B-A {row['B_minus_A']} B-C {row['B_minus_C']} "
            f"C-A {row['C_minus_A']} | SM MHz {min(clk) if clk else None}-"
            f"{max(clk) if clk else None} | t {time.time() - t0:.0f}s")
        del graphs, lay, caps, outs, refs
        torch.cuda.empty_cache()
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", tempfile.gettempdir())
    res = {"tag": TAG, "shape": SHAPE, "E": E, "I": _common.I, "NL": NL, "NB": NB,
           "replays": REPLAYS, "uuid": uuid, "sm_mhz_min": min(clocks) if clocks else None,
           "sm_mhz_max": max(clocks) if clocks else None, "shared_B": info_b,
           "shared_C": info_c, "table": table}
    with open(os.path.join(out_dir, f"stage0_{SHAPE}.json"), "w") as f:
        json.dump(res, f, indent=1)
    worst = max((row[a]["disp"] for row in table for a in ("A", "B", "C")), default=0.0)
    rep_ok = all(v == 0.0 for r in table for v in r["check"]["replay_maxdiff_vs_first"].values())
    fin_ok = all(r["check"]["finite"] for r in table)
    print(json.dumps({"tag": TAG, "worst_block_dispersion": worst,
                      "replays_bitwise_vs_first": rep_ok, "finite": fin_ok,
                      "max_B_vs_A_plus_D": max((r["check"]["B_vs_A_plus_D"] for r in table),
                                               default=None),
                      "sm_mhz_min": res["sm_mhz_min"],
                      "table": [[f"M{r['M']}_D{r['nD']}", r["A"]["us"], r["B"]["us"], r["C"]["us"],
                                 r["D"]["us"], r["B_minus_A"], r["B_minus_C"]] for r in table]}))


if __name__ == "__main__":
    main()
