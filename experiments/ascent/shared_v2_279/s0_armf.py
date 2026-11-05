"""Idea 279 Stage 0: arm F of the decode shared expert beside main's post-45 routed chain.

Layout from experiments/ascent-config/engine.json (moe_tp 2 x moe_ep 2 -> 192 local experts x
inter 1152 per rank, slot_start 192*(r%2); EP4 -> 96 x 2304, slot_start 96r).  Rank 0 is built.

One 40-layer CUDA graph per arm with the production fork/join
(multi_stream_utils.maybe_execute_in_parallel under with_multi_stream):
  MAIN : DeepseekV4Gate.fused_forward(x) (router GEMV + gate, dsv4_fused_router) then the
         production Sm12xW4a8DecodeState.run (route + persistent fused fc1+fc2 when
         TRTLLM_SM12X_FI_FC12_FUSED=1, main's default) on forced-D ids (5 weight sets,
         fresh experts every layer), 512 MB read-only flush before every replay.
  AUX  : shared_decode_mlp over block range k on the MoeShared-like aux stream.
Arms: A (routed only); Bk = A + v1 (tree fp8_shared_decode) range k; Vk = A + v2
(shared_decode_254/fsd_v2.py tuned GU_SUB2 STAGES3 DN_BN64 DN_STAGES2) range k;
k in 0 (empty), 4 ([5,9)), 5 ([0,5)), 9 ([0,9)); S5v1 / S5v2 = range 5 alone.
CUDA events per replay / 40, median of REPLAYS per block, median over NB rotated blocks;
a case is re-run (<= 2x) when any arm's block dispersion > 3%.
Note: at M <= 2 production fuses the route into the router (Stage 2); here the router runs
Stage 1 and run() routes itself (the forced-D ids are not the router's), equal in every arm.
Output: $AUTO_DEV_ARTIFACTS/s0_armf_<layout>.json; last line JSON summary.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if len(sys.argv) > 1 and not os.environ.get("CKPT"):
    os.environ["CKPT"] = sys.argv[1]

with open(os.path.join(ROOT, "experiments", "ascent-config", "engine.json")) as _f:
    ENGINE = json.load(_f)
_MTP, _MEP = ENGINE.get("moe_tensor_parallel_size"), ENGINE.get("moe_expert_parallel_size")
LAYOUT = "ep2tp2" if (_MTP, _MEP) == (2, 2) else "ep4"
os.environ["BENCH_SHAPE"] = LAYOUT
for k, v in {"V2_GU_SUB": "2", "V2_GU_STAGES": "3", "V2_DN_BN": "64", "V2_DN_STAGES": "2"}.items():
    os.environ.setdefault(k, v)
DEFAULT_CASES = ("M1:1,2,3,4,5,6;M2:1,2,3,4,5,6,7,8,10,12;M4:1,2,3,4,5,6,7,8,10,12,14,16"
                 if LAYOUT == "ep2tp2" else "M1:1,2,3,4,5,6;M2:1,2,3,4,5,6,7,8;M4:1,2,3,4,5,6,7,8")
os.environ.setdefault("S0_CASES", os.environ.get("ARMF_CASES", DEFAULT_CASES))
os.environ.setdefault("S0_NB", "4")
RANK = int(os.environ.get("ARMF_RANK", "0"))

sys.path.insert(0, os.path.join(HERE, "..", "shared_contention_196"))
sys.path.insert(0, os.path.join(HERE, "..", "shared_decode_254"))
import stage0_gpu as s0  # noqa: E402
import torch  # noqa: E402

KS = ("0", "4", "5", "9")
ARMS = ["A"] + [f"B{k}" for k in KS] + [f"V{k}" for k in KS] + ["S5v1", "S5v2"]
TAG = f"ARMF-{LAYOUT}"


def impls():
    from tensorrt_llm._torch.modules import fp8_shared_decode as fsd
    import fsd_v2
    v1 = getattr(fsd, "shared_decode_mlp_v1", fsd.shared_decode_mlp)
    return {"v1": v1, "v2": fsd_v2.shared_decode_mlp}


class LayersF(s0.Layers):
    def __init__(self, states, gates, sh, tabs, rngs, m, d, fns):
        super().__init__(states, sh, sh, None, m, d)
        self.gates, self.tabs, self.rngs, self.sh2, self.fns = gates, tabs, rngs, sh, fns

    def new_shared(self, impl, key, layer):
        mlp = self.sh2[layer]
        s1, s2 = self.tabs[layer]
        return self.fns[impl](self.xs[layer], mlp.gate_up_proj.weight, s1,
                              mlp.down_proj.weight, s2, self.rngs[key], mlp.swiglu_limit)

    def run(self, arm):
        outs = []
        for layer in range(s0.NL):
            st = self.states[layer % s0.NSETS]
            gate = self.gates[layer]

            def routed(layer=layer, st=st, gate=gate):
                gate.fused_forward(self.xs[layer])
                return st.run(self.xs[layer], self.ids[layer], self.ws[layer])

            if arm == "A":
                outs.append(routed())
            elif arm.startswith("S5"):
                outs.append(self.new_shared(arm[2:], "5", layer))
            else:
                impl = "v1" if arm[0] == "B" else "v2"
                fn1 = (lambda layer=layer, impl=impl, k=arm[1:]: self.new_shared(impl, k, layer))
                r, s = self.mep(routed, fn1, self.ev0, self.ev1, self.aux)
                outs.append(s + r)
        return outs


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate
    from tensorrt_llm._torch.modules import fp8_shared_decode as fsd
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf

    fns = impls()
    h, uuid, pynvml = s0.nvml()
    e_loc = s0.E
    slot = e_loc * (RANK % 2 if LAYOUT == "ep2tp2" else RANK)
    s0.log(f"{TAG}: layout={LAYOUT} engine.json={ENGINE} GPU {torch.cuda.get_device_name()} "
           f"uuid={uuid} NL={s0.NL} NB={s0.NB} REPLAYS={s0.REPLAYS} FC12_FUSED={leaf.FC12_FUSED} "
           f"v2 env GU_SUB={os.environ['V2_GU_SUB']} GU_STAGES={os.environ['V2_GU_STAGES']} "
           f"DN_BN={os.environ['V2_DN_BN']} DN_STAGES={os.environ['V2_DN_STAGES']}")
    torch.manual_seed(0)
    states = []
    for s in range(s0.NSETS):
        w1, s1, w2, s2 = s0.make_weights(seed=s)
        w31, s31, w2v, s2s = s0.trtllm_layout(w1, s1, w2, s2)
        del w1, s1, w2, s2
        # rank-0 local ids are generated in [0, E): slot_start 0 keeps them local (rank r's
        # slot_start only shifts ids, read from the state below).
        states.append(leaf.Sm12xW4a8DecodeState(
            w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
            num_local_experts=e_loc, slot_start=0, swiglu_limit=10.0, enable_pdl=True))
    st0 = states[0]
    s0.log(f"{TAG}: decode state E={st0.num_experts} inter={st0.inter} hidden={st0.hidden} "
           f"slot_start(built)={st0.slot_start} slot_start(rank {RANK})={slot}")
    st0.warmup(s0.TOPK)
    s0.log(f"{TAG}: fused fc12 op={bool(st0._fc12_op())} bad={sorted(leaf._FC12_BAD)}")
    gen = torch.Generator(device="cpu").manual_seed(5)
    gates = []
    for _ in range(s0.NL):
        g = DeepseekV4Gate(hidden_size=s0.H, num_experts=384, top_k=6, n_group=1, topk_group=1,
                           routed_scaling_factor=1.5, is_hashed=False, vocab_size=1024,
                           dtype=torch.bfloat16).cuda()
        g.weight.data.copy_((torch.randn(384, s0.H, generator=gen) * 0.02).to(torch.bfloat16))
        g.e_score_correction_bias.data.copy_(torch.randn(384, generator=gen) * 0.05)
        gates.append(g)
    sh, _scale, info = s0.build_shared(2304, s0.NL)
    tabs = []
    for mlp in sh:
        gu, dn = mlp.gate_up_proj, mlp.down_proj
        tabs.append((fsd.block_scale_table(gu.weight_scale, *gu.weight.shape),
                     fsd.block_scale_table(dn.weight_scale, *dn.weight.shape)))
    nb = sh[0].gate_up_proj.weight.shape[0] // 2 // 128
    half = -(-nb // 2)
    rngs = {"0": torch.tensor([0, 0], dtype=torch.int32, device="cuda"),
            "5": torch.tensor([0, half], dtype=torch.int32, device="cuda"),
            "4": torch.tensor([half, nb], dtype=torch.int32, device="cuda"),
            "9": torch.tensor([0, nb], dtype=torch.int32, device="cuda")}
    s0.log(f"{TAG}: shared {info}; blocks {nb}, parts [0,{half}) [{half},{nb}); "
           f"kernels v1={fns['v1'].__module__}.{fns['v1'].__name__} "
           f"v2={fns['v2'].__module__}.{fns['v2'].__name__}")
    s0.FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    prev = None
    for i in range(60):
        for _ in range(100):
            s0.FLUSH.sum()
        c = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM) if h is not None else None
        torch.cuda.synchronize()
        if i >= 10 and c is not None and c == prev:
            break
        prev = c
    s0.log(f"{TAG}: pre-warm done, SM {prev} MHz")
    table, clocks = [], []
    t0 = time.time()
    for (m, d) in s0.parse_cases():
        for att in range(3):
            lay = LayersF(states, gates, sh, tabs, rngs, m, d, fns)
            caps = {a: s0.capture(lambda a=a: lay.run(a)) for a in ARMS}
            graphs = {a: caps[a][0] for a in ARMS}
            outs = {a: caps[a][1] for a in ARMS}
            refs = {}
            for a in ARMS:
                s0.FLUSH.sum()
                graphs[a].replay()
                torch.cuda.synchronize()
                refs[a] = [o.clone() for o in outs[a]]
            # shared parts: V_k - A vs B_k - A (v2 vs v1, both minus the same routed graph A)
            v_vs_b = {k: max(float((v.float() - b.float()).abs().max().item())
                             for v, b in zip(refs[f"V{k}"], refs[f"B{k}"])) for k in KS}
            s5 = max(float((x.float() - y.float()).abs().max().item())
                     for x, y in zip(refs["S5v2"], refs["S5v1"]))
            s5_amax = max(float(y.float().abs().max().item()) for y in refs["S5v1"])
            finite = all(bool(torch.isfinite(o.float()).all().item()) for a in ARMS
                         for o in refs[a])
            blocks = {a: [] for a in ARMS}
            blk_clocks = []
            for b in range(s0.NB):
                blk_clk = []

                def clk():
                    if h is not None:
                        blk_clk.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))

                for a in ARMS[b % len(ARMS):] + ARMS[:b % len(ARMS)]:
                    ev = s0.time_events(graphs[a], s0.REPLAYS, clk)
                    blocks[a].append({"span": s0.pct(ev, 0.5),
                                      "rn": (s0.pct(ev, 0.9) - s0.pct(ev, 0.1)) / s0.pct(ev, 0.5)})
                if blk_clk:
                    blk_clocks.append(min(blk_clk))
            row = {"M": m, "nD": d, "layout": LAYOUT,
                   "check": {"V_minus_B_abs": v_vs_b, "S5v2_minus_S5v1_abs": s5,
                             "S5v1_absmax": s5_amax, "finite": finite}}
            for a in ARMS:
                sp = [x["span"] for x in blocks[a]]
                med = s0.pct(sp, 0.5)
                row[a] = {"us": round(med, 3),
                          "disp": round((max(sp) - min(sp)) / max(med, 1e-9), 4),
                          "blk_std": round(float(torch.tensor(sp).std().item())
                                           if len(sp) > 1 else 0.0, 3),
                          "rnoise": round(s0.pct([x["rn"] for x in blocks[a]], 0.5), 4),
                          "blocks": [round(v, 3) for v in sp]}
            for a in ARMS[1:9]:
                row[f"{a}_minus_A"] = round(row[a]["us"] - row["A"]["us"], 3)
            worst = max(row[a]["disp"] for a in ARMS)
            row["attempt"] = att + 1
            row["sm_mhz"] = [min(blk_clocks), max(blk_clocks)] if blk_clocks else None
            del graphs, lay, caps, outs, refs
            torch.cuda.empty_cache()
            if worst <= 0.03 or att == 2:
                break
            s0.log(f"{TAG} M{m}_D{d}: worst block dispersion {worst:.4f} > 3%, re-running")
        clocks.extend(blk_clocks)
        table.append(row)
        s0.log(f"{TAG} layout={LAYOUT} M{m}_D{d} [A=gate.fused_forward+Sm12xW4a8DecodeState.run"
               f"(fused fc12); B=v1 _shared_gate_up_kernel+_shared_down_kernel; V=v2 _xq_kernel+"
               f"_gu_kernel+_dn_kernel]: "
               + " | ".join(f"{a} {row[a]['us']} (disp {row[a]['disp']})" for a in ARMS)
               + " || " + " ".join(f"{a}-A {row[f'{a}_minus_A']}" for a in ARMS[1:9])
               + f" | V-B {json.dumps({k: round(v, 4) for k, v in v_vs_b.items()})} "
               f"S5v2-S5v1 {s5:.3g} (amax {s5_amax:.3g}) finite {finite} | SM {row['sm_mhz']} "
               f"| att {att + 1} | t {time.time() - t0:.0f}s")
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    res = {"tag": TAG, "shape": LAYOUT, "layout": LAYOUT, "engine": ENGINE, "E": s0.E,
           "inter": st0.inter, "NL": s0.NL, "NB": s0.NB, "replays": s0.REPLAYS, "uuid": uuid,
           "sm_mhz_min": min(clocks) if clocks else None,
           "sm_mhz_max": max(clocks) if clocks else None, "shared": info, "arms": ARMS,
           "table": table}
    path = os.path.join(out_dir, f"s0_armf_{LAYOUT}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    s0.log(f"{TAG}: wrote {path}")
    if os.environ.get("ARMF_EXPMAX", "1") == "1":
        import subprocess
        subprocess.run([sys.executable, os.path.join(HERE, "s0_expmax.py"), path], check=False)
    print(json.dumps({"tag": TAG, "layout": LAYOUT, "sm_mhz_min": res["sm_mhz_min"],
                      "worst_disp": max(r[a]["disp"] for r in table for a in ARMS),
                      "arms": ARMS[1:9],
                      "minus_A": [[f"M{r['M']}_D{r['nD']}"] + [r[f"{a}_minus_A"]
                                                               for a in ARMS[1:9]]
                                  for r in table]}))


if __name__ == "__main__":
    main()
