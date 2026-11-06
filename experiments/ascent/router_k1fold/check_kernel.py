"""Idea 313 (K1 fold into dsv4FusedRouterKernel): kernel-level bitwise gate.

For P in {480, 224, 96, 256} and M = 1..4 (8192 tokens per P, in M-row launches), with fresh random mHC K1 inputs:
  * dsv4_fused_router_k1fold(x_out, K1 inputs, ...) writes x_out bitwise == mhc_lagged_layer_input(K1 inputs, P);
  * its logits / ids / weights bitwise == dsv4_fused_router(K1 output);
  * dsv4_fused_router_route_k1fold == dsv4_fused_router_route on the K1 output: logits, ids, weights and the whole
    route workspace (offsets, token_map, token_weights, dst_rows, scale_dst_rows, routed q_out rows, scale_out) and
    the zero-filled buffer;
  * counters back at 0.
x_out is NaN-poisoned before every fold launch.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fused_router"))
from common import HIDDEN, TOPK, biases, log, make_decode_state, router_weights  # noqa: E402

import torch  # noqa: E402

PS = (480, 224, 96, 256)
TOKENS = int(os.environ.get("K1FOLD_CHECK_TOKENS", "8192"))
EPS = 1e-6
SCALE = 1.5
FIELDS = ("offsets", "token_map", "token_weights", "dst_rows", "scale_dst_rows", "q_out", "scale_out")


def make_k1_inputs(m, gen, dev, scale=1.0):
    x = (scale * torch.randn(m, HIDDEN, generator=gen, device=dev)).to(torch.bfloat16)
    res = (scale * torch.randn(m, 4, HIDDEN, generator=gen, device=dev)).to(torch.bfloat16)
    post = (2.0 * torch.rand(m, 4, generator=gen, device=dev)).float().contiguous()
    comb = torch.rand(m, 4, 4, generator=gen, device=dev)
    comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
    pre = (torch.rand(m, 4, generator=gen, device=dev) + 1e-6).float().contiguous()
    nw = (1.0 + 0.1 * torch.randn(HIDDEN, generator=gen, device=dev)).to(torch.bfloat16)
    return x, res, post, comb, pre, nw


def bits(t):
    t = t.contiguous()
    if t.element_size() == 1:
        return t.view(torch.uint8)
    if t.element_size() == 2:
        return t.view(torch.int16)
    return t.view(torch.int32)


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules.mhc.mhc_lagged_split import mhc_lagged_layer_input

    ops = torch.ops.trtllm
    assert hasattr(ops, "dsv4_fused_router_k1fold"), "fold op not built"
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(31300)
    w = router_weights(gen, 1)[0]
    b = biases(gen, 1)[0]
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    counter_r = torch.zeros(1, dtype=torch.int32, device=dev)
    state, dec = make_decode_state(96, 2304, 0)
    report = {}
    ok_all = True
    for P in PS:
        rep = {}
        for m in (1, 2, 3, 4):
            n = TOKENS // 4 if m == 4 else 384  # M=4 launches carry the 8192-token sweep
            bad = {"x": 0, "logits": 0, "ids": 0, "weights": 0, "route_x": 0, "route_logits": 0, "route_ids": 0,
                   "route_weights": 0, "route_ws": 0, "route_buf": 0, "nonfinite": 0}
            # route workspace for this M (created by an eager run, as warmup does)
            x0 = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=dev)
            i0 = (torch.arange(m * TOPK, device=dev, dtype=torch.int32) % 96).view(m, TOPK)
            state.run(x0, i0, torch.full((m, TOPK), 1.0 / TOPK, device=dev))
            for it in range(n):
                sc = 1.0 if it % 3 else (0.02 if it % 2 else 30.0)
                xp, res, post, comb, pre, nw = make_k1_inputs(m, gen, dev, sc)
                ref_x = mhc_lagged_layer_input(xp, res, post, comb, pre, nw, EPS, P, enable_pdl=False)
                ref_ids, ref_w, ref_lg = ops.dsv4_fused_router(ref_x, w, b, counter, SCALE)
                xo = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
                ids, wts, lg = ops.dsv4_fused_router_k1fold(xo, xp, res, post, comb, pre, nw, EPS, P, w, b, counter,
                                                            SCALE)
                bad["x"] += int(not same(xo, ref_x))
                bad["logits"] += int(not same(lg, ref_lg))
                bad["ids"] += int(not torch.equal(ids, ref_ids))
                bad["weights"] += int(not same(wts, ref_w))
                bad["nonfinite"] += int(not bool(torch.isfinite(xo.float()).all()))
                if it % 16 == 0:
                    # Stage 2: route workspace of the fold vs main's route launch on the K1 output
                    pre_b = state.preroute_buffers(ref_x, TOPK)
                    assert pre_b is not None
                    ws, buf = pre_b
                    buf.fill_(0x5A)
                    r_ids, r_w, r_lg = ops.dsv4_fused_router_route(
                        ref_x, w, b, counter_r, SCALE, 0, 96, ws.offsets, ws.token_map, ws.token_weights,
                        ws.dst_rows, ws.scale_dst_rows, ws.q_out, ws.scale_out, buf)
                    nr = int(ws.offsets[96].item())
                    snap = {f: getattr(ws, f).clone() for f in FIELDS}
                    snap_buf = buf.clone()
                    for f in FIELDS:
                        getattr(ws, f).fill_(0x33 if getattr(ws, f).element_size() == 1 else 7)
                    buf.fill_(0x5A)
                    xo2 = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
                    f_ids, f_w, f_lg = ops.dsv4_fused_router_route_k1fold(
                        xo2, xp, res, post, comb, pre, nw, EPS, P, w, b, counter_r, SCALE, 0, 96, ws.offsets,
                        ws.token_map, ws.token_weights, ws.dst_rows, ws.scale_dst_rows, ws.q_out, ws.scale_out, buf)
                    bad["route_x"] += int(not same(xo2, ref_x))
                    bad["route_logits"] += int(not same(f_lg, r_lg))
                    bad["route_ids"] += int(not torch.equal(f_ids, r_ids))
                    bad["route_weights"] += int(not same(f_w, r_w))
                    for f in FIELDS:
                        a, c = getattr(ws, f), snap[f]
                        if f in ("token_map", "token_weights", "q_out"):
                            a, c = a[:nr], c[:nr]
                        if not same(a, c):
                            bad["route_ws"] += 1
                    bad["route_buf"] += int(not same(buf, snap_buf))
            torch.cuda.synchronize()
            cnt = int(counter.item()) + int(counter_r.item())
            ok = not any(bad.values()) and cnt == 0
            ok_all = ok_all and ok
            rep[m] = {"launches": n, "tokens": n * m, "mismatch": bad, "counters": cnt, "ok": ok}
            log(f"P={P} M={m} launches={n} tokens={n * m} mismatches={bad} counters={cnt} -> {'OK' if ok else 'FAIL'}")
        report[P] = rep
    print(json.dumps({"passed": ok_all, "compared": "fold x/logits/ids/weights/route ws vs main K1 -> router",
                      "report": report}))
    if not ok_all:
        sys.exit(1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
