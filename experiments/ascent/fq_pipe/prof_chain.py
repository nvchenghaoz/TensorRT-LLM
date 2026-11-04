"""Idea 163: per-kernel view of main's WO chain with K3 off vs K3 arm (for nsys).

Graphs (40 layers, cold L2): mainK2_K3<arm> = K1 -> TMA-A wo_a -> K3 arm, and
K3only_<arm>. Each replayed REPS times inside an NVTX range. Prints CUDA-event
us/layer too. Usage: prof_chain.py <ckpt> M arms
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b12x_wo"))
import common as C  # noqa: E402
import torch  # noqa: E402

from b12x.gemm._shared import wo_mxfp8 as W  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2 import b12x_wo as B  # noqa: E402

ckpt = sys.argv[1]
M = int(sys.argv[2])
ARMS = {"off": (False, False, 0), "w_p2": (False, True, 2), "w": (False, True, 0)}
names = sys.argv[3].split(",")
LAYERS = 40
REPS = int(os.environ.get("REPS", "5"))
dims = C.dims_from_ckpt(ckpt)
G, R, GW, H = dims["groups_local"], dims["rank"], dims["group_width"], dims["hidden"]
w0 = C.make_weights(dims)
cs = C.make_cos_sin(dims)
ws = [w0] + [C.clone_weights(w0, dims) for _ in range(LAYERS - 1)]
o3, pos = C.make_inputs(dims, M, seed=100 + M)
xq = W.empty_mxfp8_rows_for_dense_gemm(M, GW, num_groups=G, device=C.DEV)
tmps = [W.empty_dense_gemm_mnl_view(M, R, G, device=C.DEV, dtype=torch.bfloat16) for _ in range(LAYERS)]
outs = [torch.empty((M, H, 1), device=C.DEV, dtype=torch.bfloat16) for _ in range(LAYERS)]
alpha = W._cached_alpha_one(C.DEV)


def k1():
    B.inv_rope_quant128(o3, pos, cs, xq, groups=G, group_width=GW, nope_dim=dims["nope"],
                        rope_dim=dims["rope"])


def k2(i):
    W.wo_a_dense_gemm_mxfp8(xq, ws[i].wa_b12x, out=tmps[i], alpha=alpha, expected_m=M,
                            sfb_k_replicated=True)


def k3(i, arm):
    W.wo_b_dense_gemm_fused_quant_mxfp8(tmps[i], ws[i].wb_b12x, out=outs[i], expected_m=M,
                                        sfb_k_replicated=False, fq_pipeline=arm)


def capture(fns):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:2]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    torch.cuda.synchronize()
    return g


graphs = {}
for n in names:
    arm = ARMS[n]
    graphs[f"chain_{n}"] = capture([lambda i=i, arm=arm: (k1(), k2(i), k3(i, arm)) for i in range(LAYERS)])
    graphs[f"K3only_{n}"] = capture([lambda i=i, arm=arm: k3(i, arm) for i in range(LAYERS)])
    graphs[f"K2K3_{n}"] = capture([lambda i=i, arm=arm: (k2(i), k3(i, arm)) for i in range(LAYERS)])
for gname, g in graphs.items():
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(gname)
    e0.record()
    for _ in range(REPS):
        g.replay()
    e1.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(gname, "us/layer", round(e0.elapsed_time(e1) * 1000 / REPS / LAYERS, 3), flush=True)
