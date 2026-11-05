"""Idea 279 paired microbenchmark (run_gpu_pair): the production decode shared expert
(tensorrt_llm/_torch/modules/fp8_shared_decode.shared_decode_mlp of whichever tree runs it;
base = v1 two-kernel chain, candidate = v2 default) on the aux stream beside main's routed
decode MoE chain, one EP2xTP2 rank (layout from experiments/ascent-config/engine.json).

40 layers in one CUDA graph with the production fork/join (maybe_execute_in_parallel under
with_multi_stream): MAIN = DeepseekV4Gate.fused_forward + Sm12xW4a8DecodeState.run (persistent
fused fc12) on forced-D routing (fresh experts per layer, 5 weight sets); AUX =
shared_decode_mlp over range [0,5) (rank 0/1: the 5-block part, the max rank of every layer).
512 MB read-only flush before every replay. CUDA events per replay; per M the median over NB
blocks of REPS replays. D per M = round(E[max_g D_g]) of the REAL routing at EP2xTP2 (job 1480:
M1 3.93 -> 4, M2 6.11 -> 6, M4 9.12 -> 9).
Last line: {"batch": {"<M>": {"kernel_ms": ms per 40-layer step, "noise_floor": ...}}, ...}
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if len(sys.argv) > 1 and not os.environ.get("CKPT"):
    os.environ["CKPT"] = sys.argv[1]
with open(os.path.join(ROOT, "experiments", "ascent-config", "engine.json")) as _f:
    ENGINE = json.load(_f)
LAYOUT = ("ep2tp2" if (ENGINE.get("moe_tensor_parallel_size"),
                       ENGINE.get("moe_expert_parallel_size")) == (2, 2) else "ep4")
os.environ["BENCH_SHAPE"] = LAYOUT
sys.path.insert(0, os.path.join(HERE, "..", "shared_contention_196"))
import stage0_gpu as s0  # noqa: E402
import torch  # noqa: E402

CASES = [(int(a), int(b)) for a, b in (c.split(":") for c in
                                       os.environ.get("PAIR_CASES", "1:4,2:6,4:9").split(","))]
NB = int(os.environ.get("PAIR_NB", "5"))
REPS = int(os.environ.get("PAIR_REPS", "20"))
TAG = os.environ.get("TAG", "ARM")


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate
    from tensorrt_llm._torch.modules import fp8_shared_decode as fsd
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf

    ver = "v2" if getattr(fsd, "_V2", False) else "v1"
    h, uuid, pynvml = s0.nvml()
    s0.log(f"{TAG}: layout={LAYOUT} engine={ENGINE} uuid={uuid} shared_decode_mlp impl={ver} "
           f"FC12_FUSED={leaf.FC12_FUSED} cases={CASES} NB={NB} REPS={REPS}")
    torch.manual_seed(0)
    states = []
    for s in range(s0.NSETS):
        w1, s1, w2, s2 = s0.make_weights(seed=s)
        w31, s31, w2v, s2s = s0.trtllm_layout(w1, s1, w2, s2)
        del w1, s1, w2, s2
        states.append(leaf.Sm12xW4a8DecodeState(
            w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
            num_local_experts=s0.E, slot_start=0, swiglu_limit=10.0, enable_pdl=True))
    states[0].warmup(s0.TOPK)
    s0.log(f"{TAG}: routed E={states[0].num_experts} inter={states[0].inter} "
           f"fc12 op={bool(states[0]._fc12_op())}")
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
    rng = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    limit = sh[0].swiglu_limit
    for m in (1, 2, 8):  # compile as _prepare_shared_decode_split does, outside capture
        fsd.shared_decode_mlp(torch.zeros(m, s0.H, dtype=torch.bfloat16, device="cuda"),
                              sh[0].gate_up_proj.weight, tabs[0][0], sh[0].down_proj.weight,
                              tabs[0][1], rng, limit)
    s0.log(f"{TAG}: shared {info} range [0,5)")
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
    s0.log(f"{TAG}: pre-warm SM {prev} MHz")

    res = {}
    for (m, d) in CASES:
        lay = s0.Layers(states, sh, sh, None, m, d)

        def step(lay=lay):
            outs = []
            for layer in range(s0.NL):
                st = states[layer % s0.NSETS]
                gate, mlp = gates[layer], sh[layer]
                x = lay.xs[layer]

                def routed(st=st, gate=gate, x=x, layer=layer):
                    gate.fused_forward(x)
                    return st.run(x, lay.ids[layer], lay.ws[layer])

                def shared(mlp=mlp, x=x, layer=layer):
                    return fsd.shared_decode_mlp(x, mlp.gate_up_proj.weight, tabs[layer][0],
                                                 mlp.down_proj.weight, tabs[layer][1], rng,
                                                 limit)

                r, s = lay.mep(routed, shared, lay.ev0, lay.ev1, lay.aux)
                outs.append(s + r)
            return outs

        gr, outs = s0.capture(step)
        s0.FLUSH.sum()
        gr.replay()
        torch.cuda.synchronize()
        first = [o.clone() for o in outs]
        blocks, allv, clk = [], [], []
        for _b in range(NB):
            ev = s0.time_events(gr, REPS, (lambda: clk.append(
                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))) if h else None)
            ms = [v * s0.NL / 1000.0 for v in ev]  # us/layer -> ms/step
            allv += ms
            blocks.append(s0.pct(ms, 0.5))
        torch.cuda.synchronize()
        last_diff = max(float((o.float() - f.float()).abs().max().item())
                        for o, f in zip(outs, first))
        finite = all(bool(torch.isfinite(o.float()).all().item()) for o in outs)
        med = s0.pct(blocks, 0.5)
        noise = (s0.pct(allv, 0.9) - s0.pct(allv, 0.1)) / med
        disp = (max(blocks) - min(blocks)) / med
        res[str(m)] = {"kernel_ms": round(med, 5), "noise_floor": round(noise, 4),
                       "block_dispersion": round(disp, 4), "D": d,
                       "us_per_layer": round(med * 1000 / s0.NL, 3),
                       "last_vs_first_replay_maxabs": last_diff, "finite": finite,
                       "sm_mhz": [min(clk), max(clk)] if clk else None}
        s0.log(f"{TAG} impl={ver} layout={LAYOUT} M={m} D={d} [routed fused_forward+fc12 || "
               f"aux shared_decode_mlp {ver} [0,5)]: {med:.4f} ms/step "
               f"({med * 1000 / s0.NL:.3f} us/layer) noise {noise:.4f} disp {disp:.4f} "
               f"blocks {[round(b, 4) for b in blocks]} SM {res[str(m)]['sm_mhz']} "
               f"last-vs-first {last_diff:.3g} finite {finite}")
        del gr, outs, first, lay
        torch.cuda.empty_cache()
    print(json.dumps({"batch": res, "output": {"class": "reordered"}, "tag": TAG,
                      "impl": ver, "layout": LAYOUT}))


if __name__ == "__main__":
    main()
