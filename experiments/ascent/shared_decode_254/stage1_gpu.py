"""Idea 254 arm F: the decode block-split shared kernel beside the routed FI chain.

Same method as experiments/ascent/shared_contention_196/stage0_gpu.py (fact 2202): one
40-layer CUDA graph per arm with the production fork/join (maybe_execute_in_parallel under
with_multi_stream), the production Sm12xW4a8DecodeState.run routed chain with forced-D
routing on 5 weight sets, a 512 MB read-only flush per replay, CUDA-event spans per replay
/ 40 after a warm-up, median of REPLAYS per block, median over NB rotated-order blocks.

Arms:
  A   routed only
  B   routed + main's shared (GatedMLP 1152 cols x 0.5)            (MAIN)
  F5  routed + shared_decode_mlp blocks [0,5)                     (SHARD, part 0)
  F4  routed + shared_decode_mlp blocks [5,9)                     (SHARD, part 1)
  F9  routed + shared_decode_mlp blocks [0,9)                     (SKEW, lighter rank)
  D   main's shared alone;  S5 / S9  new kernel [0,5) / [0,9) alone
Checks: every arm's replays bitwise vs its first replay (routed arms: main's own FC2
scatter spread, fact 2202 (4), reported), F5 + F4 shared part vs D (main) within BF16 tol.
Env: S1_CASES (default 'M1:0,1,2,3,4,6;M2:0,1,2,4,6,8;M4:1,2,4,6,8,12'), S0_NB (4),
S0_REPLAYS (10), BENCH_SHAPE ep4 | ep2tp2.
Output: $AUTO_DEV_ARTIFACTS/stage1_<shape>.json (stage0 table format, arms above).
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
S0 = os.path.join(HERE, "..", "shared_contention_196")
sys.path.insert(0, S0)
os.environ.setdefault("S0_NB", "4")
os.environ.setdefault("S0_CASES", os.environ.get(
    "S1_CASES", "M1:0,1,2,3,4,6;M2:0,1,2,4,6,8;M4:1,2,4,6,8,12"))

import stage0_gpu as s0  # noqa: E402  (sets up w4a8_fi_chain/common.py)
import torch  # noqa: E402

ARMS = ["A", "B", "F5", "F4", "F9", "D", "S5", "S9"]
TAG = f"STAGE1-{s0.SHAPE}"


class LayersF(s0.Layers):
    def __init__(self, states, sh_b, scale, tabs, rngs, m, d):
        super().__init__(states, sh_b, sh_b, scale, m, d)
        self.tabs, self.rngs, self.sh_b = tabs, rngs, sh_b

    def new_shared(self, key, layer):
        if os.environ.get("FSD_IMPL", "tree") == "v2":
            sys.path.insert(0, HERE)
            from fsd_v2 import shared_decode_mlp
        else:
            from tensorrt_llm._torch.modules.fp8_shared_decode import shared_decode_mlp
        mlp = self.sh_b[layer]
        s1, s2 = self.tabs[layer]
        return shared_decode_mlp(self.xs[layer], mlp.gate_up_proj.weight, s1,
                                 mlp.down_proj.weight, s2, self.rngs[key], mlp.swiglu_limit)

    def run(self, arm):
        outs = []
        for layer in range(s0.NL):
            st = self.states[layer % s0.NSETS]

            def routed(layer=layer, st=st):
                return st.run(self.xs[layer], self.ids[layer], self.ws[layer])

            if arm == "A":
                outs.append(routed())
            elif arm == "D":
                outs.append(self.shared("D", layer))
            elif arm in ("S5", "S9"):
                outs.append(self.new_shared(arm[1], layer))
            else:
                if arm == "B":
                    fn1 = (lambda layer=layer: self.shared("B", layer))
                else:
                    fn1 = (lambda layer=layer, k=arm[1]: self.new_shared(k, layer))
                r, s = self.mep(routed, fn1, self.ev0, self.ev1, self.aux)
                outs.append(s + r)
        return outs


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.modules import fp8_shared_decode as fsd
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf

    h, uuid, pynvml = s0.nvml()
    s0.log(f"{TAG}: GPU {torch.cuda.get_device_name()} uuid={uuid} E={s0.E} NL={s0.NL} "
           f"NB={s0.NB} REPLAYS={s0.REPLAYS} cfg GU_BN={fsd._GU_BN} GU_BK={fsd._GU_BK} "
           f"DN_BN={fsd._DN_BN} PDL={fsd._USE_PDL}")
    torch.manual_seed(0)
    states = []
    for s in range(s0.NSETS):
        w1, s1, w2, s2 = s0.make_weights(seed=s)
        w31, s31, w2v, s2s = s0.trtllm_layout(w1, s1, w2, s2)
        del s1, s2
        states.append(leaf.Sm12xW4a8DecodeState(
            w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
            num_local_experts=s0.E, slot_start=0, swiglu_limit=10.0, enable_pdl=True))
    states[0].warmup(s0.TOPK)
    sh_b, scale, info_b = s0.build_shared(2304, s0.NL)
    tabs = []
    for mlp in sh_b:
        gu, dn = mlp.gate_up_proj, mlp.down_proj
        tabs.append((fsd.block_scale_table(gu.weight_scale, *gu.weight.shape),
                     fsd.block_scale_table(dn.weight_scale, *dn.weight.shape)))
    nb = sh_b[0].gate_up_proj.weight.shape[0] // 2 // 128
    half = -(-nb // 2)
    rngs = {"5": torch.tensor([0, half], dtype=torch.int32, device="cuda"),
            "4": torch.tensor([half, nb], dtype=torch.int32, device="cuda"),
            "9": torch.tensor([0, nb], dtype=torch.int32, device="cuda")}
    s0.log(f"{TAG}: shared {info_b}; blocks {nb}, parts [0,{half}) [{half},{nb})")
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
        lay = LayersF(states, sh_b, scale, tabs, rngs, m, d)
        caps = {a: s0.capture(lambda a=a: lay.run(a)) for a in ARMS}
        graphs = {a: caps[a][0] for a in ARMS}
        outs = {a: caps[a][1] for a in ARMS}
        refs = {}
        for a in ARMS:
            s0.FLUSH.sum()
            graphs[a].replay()
            torch.cuda.synchronize()
            refs[a] = [o.clone() for o in outs[a]]
        # shared parts: F5 - A + F4 - A ~ D (main's x0.5 half doubled: D is 0.5 * full)
        pair_vs_main = max(float(((f5.float() - a.float()) + (f4.float() - a.float())
                                  - 2.0 * dd.float()).abs().max().item())
                           for f5, f4, a, dd in zip(refs["F5"], refs["F4"], refs["A"], refs["D"]))
        s9_vs_main = max(float((s9.float() - 2.0 * dd.float()).abs().max().item())
                         for s9, dd in zip(refs["S9"], refs["D"]))
        finite = all(bool(torch.isfinite(o.float()).all().item()) for a in ARMS for o in refs[a])
        blocks = {a: [] for a in ARMS}
        rep_diff = {a: 0.0 for a in ARMS}
        for b in range(s0.NB):
            blk_clk = []

            def clk():
                if h is not None:
                    blk_clk.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))

            for a in ARMS[b % len(ARMS):] + ARMS[:b % len(ARMS)]:
                ev = s0.time_events(graphs[a], s0.REPLAYS, clk)
                blocks[a].append({"span": s0.pct(ev, 0.5),
                                  "rn": (s0.pct(ev, 0.9) - s0.pct(ev, 0.1)) / s0.pct(ev, 0.5)})
                torch.cuda.synchronize()
                rep_diff[a] = max(rep_diff[a], s0.max_diff(outs[a], refs[a]))
            if blk_clk:
                clocks.append(min(blk_clk))
        row = {"M": m, "nD": d, "check": {"pair_minus_main_abs": pair_vs_main,
                                          "S9_minus_main_abs": s9_vs_main, "finite": finite,
                                          "replay_maxdiff_vs_first": rep_diff}}
        for a in ARMS:
            sp = [x["span"] for x in blocks[a]]
            med = s0.pct(sp, 0.5)
            row[a] = {"us": round(med, 3), "disp": round((max(sp) - min(sp)) / max(med, 1e-9), 4),
                      "blk_std": round(float(torch.tensor(sp).std().item()) if len(sp) > 1 else 0.0, 3),
                      "rnoise": round(s0.pct([x["rn"] for x in blocks[a]], 0.5), 4),
                      "blocks": [round(v, 3) for v in sp]}
        for a in ("B", "F5", "F4", "F9"):
            row[f"{a}_minus_A"] = round(row[a]["us"] - row["A"]["us"], 3)
        table.append(row)
        clk = clocks[-s0.NB:] if clocks else []
        s0.log(f"{TAG} M{m}_D{d}: " + " | ".join(
            f"{a} {row[a]['us']} (disp {row[a]['disp']}, rn {row[a]['rnoise']})" for a in ARMS)
            + " || " + " ".join(f"{a}-A {row[f'{a}_minus_A']}" for a in ("B", "F5", "F4", "F9"))
            + f" | check pair-main {pair_vs_main:.3g} S9-main {s9_vs_main:.3g} finite {finite} "
            f"replay {rep_diff} | SM {min(clk) if clk else None}-{max(clk) if clk else None} "
            f"| t {time.time() - t0:.0f}s")
        del graphs, lay, caps, outs, refs
        torch.cuda.empty_cache()
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    res = {"tag": TAG, "shape": s0.SHAPE, "E": s0.E, "NL": s0.NL, "NB": s0.NB,
           "replays": s0.REPLAYS, "uuid": uuid, "sm_mhz_min": min(clocks) if clocks else None,
           "sm_mhz_max": max(clocks) if clocks else None, "shared": info_b, "arms": ARMS,
           "table": table}
    with open(os.path.join(out_dir, f"stage1_{s0.SHAPE}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"tag": TAG, "sm_mhz_min": res["sm_mhz_min"],
                      "worst_disp": max(r[a]["disp"] for r in table for a in ARMS),
                      "max_pair_minus_main": max(r["check"]["pair_minus_main_abs"] for r in table),
                      "table": [[f"M{r['M']}_D{r['nD']}"] + [r[a]["us"] for a in ARMS]
                                for r in table], "arms": ARMS}))


if __name__ == "__main__":
    main()
