"""Shared harness for idea 201 (fused fc1+fc2 FI SM12x decode kernel).

BENCH_SHAPE=ep4 (96 x 2304, default) or ep2tp2 (192 x 1152).  Builds synthetic MXFP4 weights
at one rank's shape, the production leaf state, and runs the production call sequence
(route_assign_decode -> q0_route_direct -> [fc1_act_q1 -> fc2_finalize | fused]) on buffers
laid out exactly as Sm12xW4a8DecodeState.run lays them out.
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w4a8_fi_chain"))

import torch  # noqa: E402

import common as _common  # noqa: E402

SHAPE = os.environ.get("BENCH_SHAPE", "ep4")
if SHAPE == "ep2tp2":
    _common.E, _common.I = 192, 1152

from common import E, E_GLOBAL, H, I, TOPK, log  # noqa: E402,F401


def routing(m, d, seed=0):
    """Exactly d distinct local experts in [0, E) over m tokens (d=0: an empty rank)."""
    g = random.Random(7000 + 31 * m + d + seed)
    assert d <= TOPK * m
    experts = g.sample(range(E), d)
    per_tok = [[] for _ in range(m)]
    for i, e in enumerate(experts):
        per_tok[i % m].append(e)
    if 0 < d < m:
        for t in range(d, m):
            per_tok[t].append(experts[t % d])
    ids = []
    for t in range(m):
        loc = per_tok[t]
        rem = [E + v for v in g.sample(range(E_GLOBAL - E), TOPK - len(loc))]
        row = loc + rem
        g.shuffle(row)
        ids.append(row)
    w = torch.rand(m, TOPK, generator=torch.Generator().manual_seed(d + 100 * m)) + 0.1
    w = (w / w.sum(-1, keepdim=True) * 1.5).float()
    return torch.tensor(ids, dtype=torch.int32).cuda(), w.cuda()


def uniform_routing(m, seed):
    """Uniform top-6 of 384, then rank 0 = the busiest rank (fact 1573's 'uniform' arm)."""
    g = random.Random(90000 + seed)
    ids = [g.sample(range(E_GLOBAL), TOPK) for _ in range(m)]
    ranks = E_GLOBAL // E
    cnt = [len({e for row in ids for e in row if e // E == r}) for r in range(ranks)]
    r0 = max(range(ranks), key=lambda r: cnt[r])
    ids = [[((e // E - r0) % ranks) * E + e % E for e in row] for row in ids]
    w = torch.rand(m, TOPK, generator=torch.Generator().manual_seed(seed)) + 0.1
    w = (w / w.sum(-1, keepdim=True) * 1.5).float()
    return torch.tensor(ids, dtype=torch.int32).cuda(), w.cuda(), cnt[r0]


def make_state(enable_pdl=True):
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as leaf

    torch.manual_seed(0)
    w1, s1, w2, s2 = _common.make_weights()
    w31, s31, w2v, s2s = _common.trtllm_layout(w1, s1, w2, s2)
    del w1, s1, w2, s2
    st = leaf.Sm12xW4a8DecodeState(
        w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v, w2_weight_scale=s2s,
        num_local_experts=E, slot_start=0, swiglu_limit=10.0, enable_pdl=enable_pdl)
    return leaf, st


class Runner:
    """mode 'base' = fc1_act_q1 + fc2_finalize (main), 'fused' = the fused kernel."""

    def __init__(self, leaf, st):
        self.leaf, self.st = leaf, st
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
            kernel_moe_mxfp8_mxfp4_fc12_fused as fk)
        self.fk = fk
        self.bufs = {}

    def buffers(self, x, ids):
        leaf, st = self.leaf, self.st
        m = x.shape[0]
        if m not in self.bufs:
            _, _, _, out_sf_shape, _, _ = leaf._load_ops()
            pairs = ids.numel()
            sf1_shape = out_sf_shape(pairs, st.inter, st.num_experts)
            n_q1 = pairs * st.inter
            n_sf1 = sf1_shape[0] * sf1_shape[1] * 4
            off_sf1 = leaf._align(n_q1, 256)
            off_out = leaf._align(off_sf1 + n_sf1, 256)
            buf = torch.empty(off_out + m * st.hidden * 2, dtype=torch.uint8, device=x.device)
            q1 = buf[:n_q1].view(torch.float8_e4m3fn).view(pairs, st.inter)
            sf1 = buf[off_sf1:off_sf1 + n_sf1].view(torch.int32).view(sf1_shape)
            out = buf[off_out:].view(torch.bfloat16).view(m, st.hidden)
            self.bufs[m] = (buf, q1, sf1, out)
        return self.bufs[m]

    def __call__(self, x, ids, w, mode):
        leaf, st = self.leaf, self.st
        route, _, fc1, _, fc2, fi_act = leaf._load_ops()
        m = x.shape[0]
        buf, q1, sf1, out = self.buffers(x, ids)
        ws = leaf._workspace(x, ids, st.num_experts)
        offsets, token_map, rw, a_q, a_sf = route(
            x, ids, w, st.num_experts, workspace=ws, enable_pdl=st.enable_pdl,
            expert_offset=st.slot_start, zero_buf=buf)
        if mode == "base":
            fc1(a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False, enable_pdl=st.enable_pdl,
                out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=st.swiglu_limit)
            fc2(q1, sf1, st.w2, st.w2_sf, offsets, token_map, rw, m, tune=False,
                enable_pdl=st.enable_pdl, out=out)
        else:
            self.fk.cute_dsl_sm12x_fc12_fused_mxfp8_mxfp4(
                a_q, a_sf, st.w1, st.w1_sf, offsets, st.w2, st.w2_sf, token_map, rw, m,
                out_q=q1, out_sf=sf1, out=out, swiglu_limit=st.swiglu_limit,
                enable_pdl=st.enable_pdl)
        return q1, sf1, out, offsets


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g
