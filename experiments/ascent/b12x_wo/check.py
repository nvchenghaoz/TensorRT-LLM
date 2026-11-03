"""Untimed repeated-execution check for the b12x CSA2 decode output projection.

Runs pair.py's sequence exactly (same module copies, same graph, same warmup +
iters replays, inputs restored before every replay) and compares EVERY replay's
output of every module copy bitwise with main's eager chain on the pristine
input (deepseek_v4 project_sparse_attn_output, which the candidate tree still
carries). The b12x scratch (x_q values and scales, tmp) allocated inside the
graph is recorded; before each replay it is poisoned with NaN bytes and after
each replay it must hold exactly main's quantized activation / o_lora.
Also: a large-M bitwise check of the BF16 inverse RoPE against mla_rope_inplace,
the stock 1x32 b12x path error (diagnostic), and FP64 reference errors.
Usage: check.py <ckpt> [--batches 1,2,4,8] [--warmup 5] [--iters 30]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import harness as Hn  # noqa: E402
import torch  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import \
    project_sparse_attn_output  # noqa: E402
from tensorrt_llm._torch.attention.mla import fp8_block_scaling_bmm_out  # noqa: E402
from tensorrt_llm.quantization.utils import fp8_utils  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--batches", default="1,2,4,8")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--rot-mb", type=int, default=320)
args = ap.parse_args()
BATCHES = [int(b) for b in args.batches.split(",")]
assert Hn.HAS_B12X_WO, "check.py runs on the candidate tree"
B = Hn.B
from b12x.gemm._shared import wo_mxfp8 as W  # noqa: E402

dims = C.dims_from_ckpt(args.ckpt)
G, R, GW, H = dims["groups_local"], dims["rank"], dims["group_width"], dims["hidden"]
w0 = C.make_weights(dims)
cs = C.make_cos_sin(dims)
wbytes = w0.wa_main.numel() + w0.wb_q.numel() + w0.sa_main.numel() * 4 + w0.wb_sf.numel()
ncopy = max(2, -(-args.rot_mb * 1024 * 1024 // wbytes))
ws = [w0] + [C.clone_weights(w0, dims) for _ in range(ncopy - 1)]
mods = [Hn.make_module(dims, w, cs) for w in ws]
ok = True
report = {"ncopy": ncopy, "warmup": args.warmup, "iters": args.iters}

# ---- 1. inverse RoPE bitwise vs mla_rope_inplace at large M ----
rope = {}
for var in (0, 1):
    big_m = 16384
    g = torch.Generator(device="cuda").manual_seed(5)
    o = (torch.randn(big_m, dims["heads_tp"], dims["head_dim"], device="cuda", generator=g) *
         torch.exp2(torch.randint(-6, 6, (big_m, dims["heads_tp"], 1), device="cuda",
                                  generator=g).float())).to(torch.bfloat16)
    p = torch.randint(0, cs.shape[0], (big_m,), device="cuda", generator=g, dtype=torch.int32)
    ref = o.clone()
    torch.ops.trtllm.mla_rope_inplace(ref, p, cs, dims["heads_tp"], dims["nope"], dims["rope"], True,
                                      False)
    x_q = W.empty_mxfp8_rows_for_dense_gemm(big_m, GW, num_groups=G, device="cuda")
    rope_out = torch.empty_like(o)
    B.inv_rope_quant128(o, p, cs, x_q, groups=G, group_width=GW, nope_dim=dims["nope"],
                        rope_dim=dims["rope"], rope_variant=var, rope_out=rope_out)
    mq, _ = fp8_utils.per_token_quant_and_transform(ref.view(big_m, G, -1), need_permute102=True)
    nd = int((rope_out != ref).sum())
    nq = int((x_q.values.permute(2, 0, 1).contiguous().view(torch.uint8) != mq.view(torch.uint8)).sum())
    rope[f"variant{var}"] = {"rope_bf16_diff": nd, "x_q_diff": nq}
report["rope_large_m"] = rope
ok &= rope["variant0"]["rope_bf16_diff"] == 0 and rope["variant0"]["x_q_diff"] == 0
print("rope", rope, flush=True)

# ---- 2. repeated-execution sequence (graph, matches pair.py) ----
recorded = []
_orig_bases = W.empty_mxfp8_rows_bases
_orig_mnl = W.empty_dense_gemm_mnl_view


def _rec_bases(*a, **k):
    r = _orig_bases(*a, **k)
    if torch.cuda.is_current_stream_capturing():
        recorded.append(("x_q", r))
    return r


def _rec_mnl(*a, **k):
    r = _orig_mnl(*a, **k)
    if torch.cuda.is_current_stream_capturing():
        recorded.append(("tmp", r))
    return r


W.empty_mxfp8_rows_bases = _rec_bases
W.empty_dense_gemm_mnl_view = _rec_mnl

seq = {}
for M in BATCHES:
    o3, pos = C.make_inputs(dims, M, seed=100 + M)
    pristine = o3.view(M, -1).clone()
    # main reference (eager, deepseek_v4 chain) and its intermediates
    ref_out = project_sparse_attn_output(mods[0], [pristine.clone()], pos)
    roped = pristine.clone().view(M, dims["heads_tp"], -1)
    torch.ops.trtllm.mla_rope_inplace(roped, pos, cs, dims["heads_tp"], dims["nope"], dims["rope"], True,
                                      False)
    ref_xq, ref_xs = fp8_utils.per_token_quant_and_transform(roped.view(M, G, -1), need_permute102=True)
    ref_xs_bytes = ref_xs.contiguous().view(torch.uint8).view(G, M, -1)  # [G, M, GW/128]
    ref_lora = torch.empty([M, G, R], device="cuda", dtype=torch.bfloat16)
    fp8_block_scaling_bmm_out(roped.clone().view(M, G, -1), w0.wa_main, w0.sa_main,
                              ref_lora.transpose(0, 1), None, False)
    ref64, _ = C.fp64_reference(o3, pos, cs, w0, dims)
    bufs = [pristine.clone() for _ in range(ncopy)]
    for i in range(ncopy):
        Hn.project(mods[i], bufs[i], pos)
    torch.cuda.synchronize()
    for b in bufs:
        b.copy_(pristine)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            Hn.project(mods[i], bufs[i], pos)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    for b in bufs:
        b.copy_(pristine)
    recorded.clear()
    from b12x._lib import compiler as _bc  # noqa: E402
    c_before = (_bc._DISK_CACHE_HITS, _bc._COMPILE_MISSES)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [Hn.project(mods[i], bufs[i], pos) for i in range(ncopy)]
    c_after = (_bc._DISK_CACHE_HITS, _bc._COMPILE_MISSES)
    xqs = [r for k, r in recorded if k == "x_q"]
    tmps = [r for k, r in recorded if k == "tmp"]
    row = {"path": Hn.path_name(mods[0]), "scratch_recorded": [len(xqs), len(tmps)],
           "compiles_in_capture": {"disk_hits": c_after[0] - c_before[0],
                                   "misses": c_after[1] - c_before[1]},
           "per_iter": []}
    good = len(xqs) == ncopy and len(tmps) == ncopy and c_after == c_before
    for it in range(args.warmup + args.iters):
        for b in bufs:  # reset point (same as pair.py)
            b.copy_(pristine)
        for vb, sb, pb in xqs:  # poison scratch: e4m3 0xFF / e8m0 0xFF are NaN
            vb.view(torch.uint8).fill_(0xFF)
            sb.view(torch.uint8).fill_(0xFF)
            pb.view(torch.uint8).fill_(0xFF)
        for t in tmps:
            t.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        n_out = n_xq = n_xs = n_tmp = 0
        max_abs = 0.0
        for i in range(ncopy):
            o = outs[i]
            n_out += int((o != ref_out).sum()) + int((~torch.isfinite(o)).sum())
            max_abs = max(max_abs, float((o.float() - ref_out.float()).abs().max()))
            vb, sb, pb = xqs[i]
            n_xq += int((vb.view(torch.uint8) != ref_xq.view(torch.uint8)).sum())
            phys = pb.view(torch.uint8)  # (G, mt, kt, 32, 4, 4)
            logical = phys.permute(0, 1, 4, 3, 2, 5).reshape(G, -1, phys.shape[2] * 4)[:, :M]
            for j in range(4):
                n_xs += int((logical[:, :, j::4] != ref_xs_bytes).sum())
            n_tmp += int((tmps[i].permute(2, 0, 1) != ref_lora.transpose(0, 1)).sum())
        row["per_iter"].append({"it": it, "out_diff": n_out, "max_abs": max_abs, "x_q_diff": n_xq,
                                "x_scale_diff": n_xs, "tmp_diff": n_tmp})
        good &= n_out == 0 and n_xq == 0 and n_xs == 0 and n_tmp == 0
    row["all_bitwise"] = bool(good)
    row["out_vs_fp64"] = C.err(outs[0], ref64)
    row["main_vs_fp64"] = C.err(ref_out, ref64)
    # stock 1x32 b12x path (diagnostic, precision class)
    out_s = C.stock_chain(pristine.clone().view(M, dims["heads_tp"], -1), pos, cs, w0, dims)
    row["stock_1x32_vs_main"] = C.err(out_s, ref_out)
    row["stock_1x32_vs_fp64"] = C.err(out_s, ref64)
    seq[str(M)] = row
    ok &= good
    print(f"M={M} path={row['path']} all_bitwise={good} scratch={row['scratch_recorded']} "
          f"iters={len(row['per_iter'])} last={row['per_iter'][-1]}", flush=True)
    del g, outs
    torch.cuda.empty_cache()

W.empty_mxfp8_rows_bases = _orig_bases
W.empty_dense_gemm_mnl_view = _orig_mnl
report["sequence"] = seq
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/check.json", "w") as f:
    json.dump(report, f, indent=1)
summary = {M: {"path": r["path"], "all_bitwise": r["all_bitwise"], "replays": len(r["per_iter"]),
               "max_abs_vs_main": max(x["max_abs"] for x in r["per_iter"])} for M, r in seq.items()}
print(json.dumps({"passed": bool(ok), "compared_against": "main eager project_sparse_attn_output "
                  "(mla_rope_inplace + 1x128 quant + SM120 BMM + o_b_proj) bitwise, every replay",
                  "rope_large_m": rope, "summary": summary}))
