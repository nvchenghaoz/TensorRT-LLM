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
            fn()
    torch.cuda.synchronize()
    return g


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def kineto(g):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        FLUSH.sum()
        g.replay()
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
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
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
    for (m, d) in parse_cases():
        lay = Layers(states, sh_b, sh_c, scale, m, d)
        graphs = {a: capture(lambda a=a: lay.run(a)) for a in arms}
        # node classification from one trace per arm: shared nodes = names seen in the
        # shared-alone graphs and never in the routed-only graph
        names = {a: {e["name"] for r in kineto(graphs[a]) for e in r} for a in ("A", "D", "E")}
        routed_names = names["A"]
        shared_names = names["D"] | names["E"]
        blocks = {a: [] for a in arms}
        for _b in range(NB):
            for a in arms:
                blocks[a].append(summarize(kineto(graphs[a]), shared_names, routed_names))
            if h is not None:
                clocks.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        row = {"M": m, "D": d}
        for a in arms:
            sp = [x["span"] for x in blocks[a]]
            med = pct(sp, 0.5)
            row[a] = {
                "us": round(med, 3),
                "disp": round((max(sp) - min(sp)) / max(med, 1e-9), 4),
                "blk_std": round(float(torch.tensor(sp).std().item()) if len(sp) > 1 else 0.0, 3),
                "rnoise": round(pct([x["span_rnoise"] for x in blocks[a]], 0.5), 4),
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
            f"{a} {row[a]['us']}us (disp {row[a]['disp']}, rn {row[a]['rnoise']}, fc1 "
            f"{row[a]['fc1']} fc2 {row[a]['fc2']} sh {row[a]['shared']} n{row[a]['nodes']})"
            for a in arms) + f" || B-A {row['B_minus_A']} B-C {row['B_minus_C']} "
            f"C-A {row['C_minus_A']} | SM MHz {min(clk) if clk else None}-"
            f"{max(clk) if clk else None} | t {time.time() - t0:.0f}s")
        del graphs, lay
        torch.cuda.empty_cache()
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", tempfile.gettempdir())
    res = {"tag": TAG, "shape": SHAPE, "E": E, "I": _common.I, "NL": NL, "NB": NB,
           "replays": REPLAYS, "uuid": uuid, "sm_mhz_min": min(clocks) if clocks else None,
           "sm_mhz_max": max(clocks) if clocks else None, "shared_B": info_b,
           "shared_C": info_c, "table": table}
    with open(os.path.join(out_dir, f"stage0_{SHAPE}.json"), "w") as f:
        json.dump(res, f, indent=1)
    worst = max((row[a]["disp"] for row in table for a in ("A", "B", "C")), default=0.0)
    print(json.dumps({"tag": TAG, "worst_block_dispersion": worst,
                      "sm_mhz_min": res["sm_mhz_min"],
                      "table": [[f"M{r['M']}_D{r['D']}", r["A"]["us"], r["B"]["us"], r["C"]["us"],
                                 r["D"]["us"], r["B_minus_A"], r["B_minus_C"]] for r in table]}))


if __name__ == "__main__":
    main()
