"""Stage 1 for idea 73: main's CSA2 WO chain vs b12x WO paths, per-rank TP4 shapes.

Cold-L2 CUDA graphs with weights rotated over > 128 MB. Numerics per M:
bitwise vs main (exact variant), and max abs/rel vs main and vs an FP64 reference
of the original math. Usage: stage1.py <ckpt> [MS]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import torch  # noqa: E402

ckpt = sys.argv[1]
MS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8,128").split(",")]
REPLAYS = int(os.environ.get("REPLAYS", "20"))
ROUNDS = int(os.environ.get("ROUNDS", "2"))
ROT_BYTES = int(os.environ.get("ROT_MB", "320")) * 1024 * 1024

dims = C.dims_from_ckpt(ckpt)
print("dims", dims, flush=True)
w0 = C.make_weights(dims)
cs = C.make_cos_sin(dims)
wbytes = w0.wa_main.numel() + w0.wb_q.numel() + w0.sa_main.numel() * 4 + w0.wb_sf.numel()
ncopy = max(2, -(-ROT_BYTES // wbytes))
print(f"weight bytes/call {wbytes/1e6:.1f} MB, ncopy {ncopy}", flush=True)
ws = [w0] + [C.clone_weights(w0, dims) for _ in range(ncopy - 1)]

# ---------------- numerics ----------------
numerics = {}
for M in MS:
    o3, pos = C.make_inputs(dims, M, seed=100 + M)
    ref64, lora64 = C.fp64_reference(o3, pos, cs, w0, dims)
    # main, with intermediates
    from tensorrt_llm._torch.attention.mla import fp8_block_scaling_bmm_out
    from tensorrt_llm.quantization.utils import fp8_utils
    om = o3.clone()
    torch.ops.trtllm.mla_rope_inplace(om, pos, cs, dims["heads_tp"], dims["nope"], dims["rope"], True,
                                      False)
    rope_main = om.clone()
    m_fp8, m_sc = fp8_utils.per_token_quant_and_transform(om.view(M, dims["groups_local"], -1),
                                                          need_permute102=True)
    out_main = C.main_chain(o3.clone(), pos, cs, w0, dims)
    out_main2 = C.main_chain(o3.clone(), pos, cs, w0, dims)
    lora_main = torch.empty([M, dims["groups_local"], dims["rank"]], device=C.DEV, dtype=torch.bfloat16)
    fp8_block_scaling_bmm_out(rope_main.clone().view(M, dims["groups_local"], -1), w0.wa_main,
                              w0.sa_main, lora_main.transpose(0, 1), None, False)
    row = {"main_det": bool(torch.equal(out_main, out_main2)),
           "main_vs_fp64": C.err(out_main, ref64),
           "main_lora_vs_fp64": C.err(lora_main.flatten(1), lora64)}
    # exact variant: try every rope variant, report x_q equality vs main's quant
    for var in (0, 1, 2):
        sc = C.Scratch(dims, M)
        oin = o3.clone()
        out_e = C.exact_chain(oin, pos, cs, w0, dims, sc, rope_variant=var).clone()
        xq_vals = sc.x_q.values.permute(2, 0, 1)  # [G, M, GW]
        xq_eq = C.bit_cmp(xq_vals.contiguous(), m_fp8.contiguous())
        # main scale bytes: int32 [G, M, GW/512] packing 4 x 1x128 exponents
        msb = m_sc.contiguous().view(torch.uint8).view(dims["groups_local"], M, -1)  # [G,M,GW/128]
        ours = sc.x_q.scale_mma.permute(5, 2, 4, 0, 1, 3).contiguous()  # phys (g,mt,kt,32,4,4)
        # logical [g, row, chunk32]
        G = dims["groups_local"]
        phys = ours.view(G, -1, ours.shape[2], 32, 4, 4)
        mt = phys.shape[1]
        logical = phys.permute(0, 1, 4, 3, 2, 5).reshape(G, mt * 128, -1).view(torch.uint8)[:, :M]
        sc_eq = bool(torch.equal(logical[:, :, ::4].contiguous(), msb[:, :, :logical.shape[2] // 4]))
        rep_ok = bool(all(torch.equal(logical[:, :, j::4], logical[:, :, ::4]) for j in range(1, 4)))
        tmp_lora = sc.tmp.permute(0, 2, 1).reshape(M, -1)
        sc2 = C.Scratch(dims, M)
        out_e1 = C.exact_chain(o3.clone(), pos, cs, w0, dims, sc2, rope_variant=var, wo_b="main").clone()
        out_e_rep = C.exact_chain(o3.clone(), pos, cs, w0, dims, sc, rope_variant=var).clone()
        row[f"E_v{var}"] = {
            "x_q_vs_main_quant": xq_eq, "x_scale_vs_main": sc_eq, "x_scale_replicated": rep_ok,
            "o3_unmodified": bool(torch.equal(oin, o3)),
            "tmp_vs_main_lora": C.bit_cmp(tmp_lora.contiguous(), lora_main.flatten(1)),
            "tmp_vs_main_lora_err": C.err(tmp_lora, lora_main.flatten(1)),
            "out_vs_main": C.bit_cmp(out_e, out_main), "out_vs_main_err": C.err(out_e, out_main),
            "out_vs_fp64": C.err(out_e, ref64), "det": bool(torch.equal(out_e, out_e_rep)),
            "E1_out_vs_main": C.bit_cmp(out_e1, out_main), "E1_vs_E": C.bit_cmp(out_e1, out_e),
        }
    if M <= 8 or True:
        out_s = C.stock_chain(o3.clone(), pos, cs, w0, dims).clone()
        out_s2 = C.stock_chain(o3.clone(), pos, cs, w0, dims).clone()
        row["S"] = {"out_vs_main": C.bit_cmp(out_s, out_main), "out_vs_main_err": C.err(out_s, out_main),
                    "out_vs_fp64": C.err(out_s, ref64), "det": bool(torch.equal(out_s, out_s2))}
    numerics[M] = row
    print(f"[numerics M={M}] " + json.dumps(row), flush=True)

# pick the rope variant that matched main (default 0)
best_var = 0
for var in (0, 1, 2):
    if all(numerics[M][f"E_v{var}"]["x_q_vs_main_quant"]["equal"] for M in MS):
        best_var = var
        break
print("rope variant", best_var, flush=True)


# ---------------- timing ----------------
def time_graph(calls, warm=2):
    """calls: list of zero-arg callables (one per rotation copy)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in calls[:warm]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPLAYS):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / len(calls))
    del g
    ts.sort()
    med = ts[len(ts) // 2]
    return {"us": round(med, 3), "noise": round((ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med, 4)}


from tensorrt_llm._torch.attention.mla import fp8_block_scaling_bmm_out  # noqa: E402
from tensorrt_llm.quantization.utils import fp8_utils  # noqa: E402
from b12x.gemm._shared import wo_mxfp8 as W  # noqa: E402

timing = {}
for M in MS:
    o3, pos = C.make_inputs(dims, M, seed=100 + M)
    G, R, GW, H = dims["groups_local"], dims["rank"], dims["group_width"], dims["hidden"]
    bufs = [o3.clone() for _ in range(ncopy)]
    scs = [C.Scratch(dims, M) for _ in range(ncopy)]
    # prepared intermediates for per-launch timing
    m_fp8, m_sc = fp8_utils.per_token_quant_and_transform(o3.view(M, G, -1), need_permute102=True)
    lora = torch.empty([M, G, R], device=C.DEV, dtype=torch.bfloat16)
    bmm_out = torch.empty([G, M, R], device=C.DEV, dtype=torch.bfloat16)
    act, act_sf = torch.ops.trtllm.mxfp8_quantize(lora.flatten(1), True)
    sc0 = C.Scratch(dims, M)
    C.inv_rope_quant128(o3, pos, cs, sc0.x_q, dims, best_var)
    chains = {
        "main": lambda i: C.main_chain(bufs[i], pos, cs, ws[i], dims),
        "E": lambda i: C.exact_chain(bufs[i], pos, cs, ws[i], dims, scs[i], best_var),
        "E1": lambda i: C.exact_chain(bufs[i], pos, cs, ws[i], dims, scs[i], best_var, wo_b="main"),
        "S": lambda i: C.stock_chain(bufs[i], pos, cs, ws[i], dims),
        # per-launch pieces
        "main.rope": lambda i: torch.ops.trtllm.mla_rope_inplace(bufs[i], pos, cs, dims["heads_tp"],
                                                                 dims["nope"], dims["rope"], True, False),
        "main.ptq": lambda i: fp8_utils.per_token_quant_and_transform(bufs[i].view(M, G, -1),
                                                                      need_permute102=True),
        "main.bmm": lambda i: torch.ops.trtllm.fp8_block_scaling_bmm_out(m_fp8, ws[i].wa_main, m_sc,
                                                                         ws[i].sa_main, bmm_out),
        "main.copy": lambda i: lora.transpose(0, 1).copy_(bmm_out),
        "main.mxq": lambda i: torch.ops.trtllm.mxfp8_quantize(lora.flatten(1), True),
        "main.wo_b": lambda i: torch.ops.trtllm.flashinfer_mm_mxfp8(act, act_sf, ws[i].wb_q, ws[i].wb_sf,
                                                                    torch.bfloat16),
        "E.quant": lambda i: C.inv_rope_quant128(bufs[i], pos, cs, scs[i].x_q, dims, best_var),
        "E.wo_a": lambda i: W.wo_a_dense_gemm_mxfp8(sc0.x_q, ws[i].wa_b12x, out=scs[i].tmp,
                                                    alpha=sc0.alpha, expected_m=M, sfb_k_replicated=True),
        "E.wo_b_fused": lambda i: W.wo_b_dense_gemm_fused_quant_mxfp8(scs[i].tmp, ws[i].wb_b12x,
                                                                      out=scs[i].out, expected_m=M,
                                                                      sfb_k_replicated=False),
        "E.wo_a_unrep": lambda i: W.wo_a_dense_gemm_mxfp8(sc0.x_q, ws[i].wa_b12x, out=scs[i].tmp,
                                                          alpha=sc0.alpha, expected_m=M,
                                                          sfb_k_replicated=False),
    }
    res = {}
    for rnd in range(ROUNDS):
        for name, mk in chains.items():
            try:
                t = time_graph([(lambda i=i, mk=mk: mk(i)) for i in range(ncopy)])
            except Exception as e:  # noqa
                t = {"error": repr(e)[:300]}
            res.setdefault(name, []).append(t)
    agg = {}
    for name, lst in res.items():
        ok = [x for x in lst if "us" in x]
        if not ok:
            agg[name] = lst[0]
            continue
        agg[name] = {"us": round(sorted(x["us"] for x in ok)[len(ok) // 2], 3),
                     "rounds": [x["us"] for x in ok], "noise": max(x["noise"] for x in ok)}
    timing[M] = agg
    print(f"[timing M={M}] " + " ".join(f"{k}={v.get('us')}({v.get('noise')})" for k, v in agg.items()),
          flush=True)
    del bufs, scs
    torch.cuda.empty_cache()

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/stage1.json", "w") as f:
    json.dump({"dims": dims, "numerics": numerics, "timing": timing, "rope_variant": best_var}, f, indent=1)
summary = {str(M): {k: timing[M][k].get("us") for k in ("main", "E", "E1", "S")} for M in MS}
print(json.dumps({"passed": True, "rope_variant": best_var, "summary": summary,
                  "E_exact": {str(M): numerics[M][f"E_v{best_var}"]["out_vs_main"]["equal"] for M in MS}}))
