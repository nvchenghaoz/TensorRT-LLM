"""Untimed correctness for the fa2_sm120 path in VanillaAttention (candidate tree).

Protected reference: torch SDPA forced onto its flash kernel (pytorch_flash::flash_fwd_kernel
<128,128,64,4>), same inputs/views. Exact class => bitwise.
 1. bitwise at S = 1000, 4096, 4097, 37769, 37966 (+ output strides equal SDPA's)
 2. error vs an independent FP32 reference (explicit softmax(QK^T)V on row slices) for both
 3. repeated-execution sequence with the timing loop's warmup/iteration counts: every call
    bitwise equal to call 1 and to the reference; q/k/v (and the qkv buffer) bitwise unchanged
 4. torch.compile(fullgraph=True) of the _attn_impl path ([B,S,H*D] views -> HND -> backend ->
    transpose/flatten): no graph break, one compile per S, no recompile on repeats, output
    bitwise equal to eager SDPA; the custom op is in the graph
 5. key_padding_mask calls stay on SDPA
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_inputs import D, H, make_inputs  # noqa: E402

from tensorrt_llm._torch.visual_gen.attention.backends import fa2_sm120  # noqa: E402
from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention  # noqa: E402

SCALE = 1.0 / math.sqrt(D)
WARMUP, ITERS = 10, 30


def ref_flash(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, scale=SCALE)


def fp32_rows(q, k, v, heads, rows):
    outs = []
    for h in heads:
        qs = q[0, h, rows].float()
        s = (qs @ k[0, h].float().T) * SCALE
        p = torch.softmax(s, -1)
        outs.append(p @ v[0, h].float())
    return torch.stack(outs)


def main():
    res = {"passed": True, "checks": {}}

    def fail(name, info):
        res["passed"] = False
        res["checks"][name] = {"ok": False, **info}
        print("FAIL", name, info, flush=True)

    t0 = time.time()
    attn = VanillaAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16)
    res["build_or_load_s"] = round(time.time() - t0, 1)
    res["fa2_sm120_enabled"] = bool(attn._fa2_sm120)
    if not attn._fa2_sm120:
        fail("enabled", {})

    # 1 + 2: bitwise per S, strides, FP32 error
    for S in [1000, 4096, 4097, 37769, 37966]:
        q, k, v, qkv = make_inputs(S, seed=S)
        ref = ref_flash(q, k, v)
        out = attn.forward(q, k, v)
        torch.cuda.synchronize()
        eq = torch.equal(out, ref)
        rec = {"ok": eq and out.stride() == ref.stride(), "bitwise": eq,
               "stride": list(out.stride()), "ref_stride": list(ref.stride()),
               "flatten_is_view": out.transpose(1, 2).flatten(2).data_ptr() == out.data_ptr()
               and out.transpose(1, 2).is_contiguous()}
        if not eq:
            d = (out.float() - ref.float()).abs()
            rec.update({"n_diff": int((out != ref).sum()), "max_abs": float(d.max())})
        rows = torch.cat([torch.arange(0, 64), torch.arange(S - 64, S)]).cuda()
        heads = [0, H - 1]
        r32 = fp32_rows(q, k, v, heads, rows)
        rec["maxabs_vs_fp32_candidate"] = float((out[0, heads][:, rows].float() - r32).abs().max())
        rec["maxabs_vs_fp32_sdpa"] = float((ref[0, heads][:, rows].float() - r32).abs().max())
        res["checks"][f"bitwise_S{S}"] = rec
        print("S", S, rec, flush=True)
        if not rec["ok"]:
            res["passed"] = False
        del q, k, v, qkv, ref, out
        torch.cuda.empty_cache()

    # 3: repeated-execution sequence (same warmup / iteration counts as bench_attn.py)
    S = 37966
    q, k, v, qkv = make_inputs(S, seed=S)
    pristine = [t.clone() for t in (q, k, v, qkv)]
    ref = ref_flash(q, k, v)
    first = None
    bad = []
    for it in range(WARMUP + ITERS):
        out = attn.forward(q, k, v)
        torch.cuda.synchronize()
        if first is None:
            first = out.clone()
        ok = torch.equal(out, ref) and torch.equal(out, first)
        inp_ok = all(torch.equal(a, b) for a, b in zip((q, k, v, qkv), pristine))
        if not (ok and inp_ok):
            bad.append({"iter": it, "out_ok": ok, "inputs_unchanged": inp_ok})
    res["checks"]["repeat_S37966"] = {"ok": not bad, "iterations": WARMUP + ITERS, "bad": bad[:5]}
    print("repeat", res["checks"]["repeat_S37966"], flush=True)
    if bad:
        res["passed"] = False
    del q, k, v, qkv, pristine, ref, first
    torch.cuda.empty_cache()

    # 4: torch.compile of the _attn_impl path
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    dynamo.reset()
    counters.clear()

    def attn_impl(qf, kf, vf):
        b = qf.shape[0]
        qh = qf.view(b, -1, H, D).transpose(1, 2)
        kh = kf.view(b, -1, H, D).transpose(1, 2)
        vh = vf.view(b, -1, H, D).transpose(1, 2)
        out = attn.forward(qh, kh, vh)
        return out.transpose(1, 2).flatten(2)

    compiled = torch.compile(attn_impl, fullgraph=True, dynamic=False)
    comp = {"ok": True}
    for S in [37966, 37769]:
        q, k, v, qkv = make_inputs(S, seed=S + 7)
        qf, kf = q.transpose(1, 2).flatten(2), k.transpose(1, 2).flatten(2)
        vf = qkv[..., 2 * H * D:]
        ref = ref_flash(q, k, v).transpose(1, 2).flatten(2)
        t1 = time.time()
        outs = [compiled(qf, kf, vf) for _ in range(3)]
        torch.cuda.synchronize()
        comp[f"S{S}_first3_s"] = round(time.time() - t1, 1)
        comp[f"S{S}_bitwise"] = all(torch.equal(o, ref) for o in outs)
        comp["ok"] = comp["ok"] and comp[f"S{S}_bitwise"]
        del q, k, v, qkv, qf, kf, vf, ref, outs
        torch.cuda.empty_cache()
    comp["unique_graphs"] = int(counters["stats"]["unique_graphs"])
    comp["graph_breaks"] = int(sum(counters["graph_break"].values()))
    comp["recompiles_beyond_one_per_S"] = comp["unique_graphs"] - 2
    comp["ok"] = comp["ok"] and comp["graph_breaks"] == 0 and comp["unique_graphs"] == 2
    # custom op in the compiled graph
    gm_ops = []
    try:
        from torch._dynamo.testing import CompileCounterWithBackend

        cnt = CompileCounterWithBackend("eager")
        dynamo.reset()
        q, k, v, qkv = make_inputs(1000, seed=3)
        torch.compile(attn_impl, backend=cnt, fullgraph=True)(
            q.transpose(1, 2).flatten(2), k.transpose(1, 2).flatten(2), qkv[..., 2 * H * D:])
        for g in cnt.graphs:
            gm_ops += [str(n.target) for n in g.graph.nodes if n.op == "call_function"]
    except Exception as e:  # noqa: BLE001
        gm_ops = [f"err {e!r}"]
    comp["custom_op_in_graph"] = any("fa2_sm120_fwd" in o for o in gm_ops)
    comp["sdpa_in_graph"] = any("scaled_dot_product" in o for o in gm_ops)
    comp["ok"] = comp["ok"] and comp["custom_op_in_graph"] and not comp["sdpa_in_graph"]
    res["checks"]["torch_compile"] = comp
    print("compile", comp, flush=True)
    if not comp["ok"]:
        res["passed"] = False

    # 5: masked path stays SDPA
    q, k, v, _ = make_inputs(1000, seed=5)
    kpm = torch.ones(1, 1000, dtype=torch.bool, device="cuda")
    kpm[:, 900:] = False
    out = attn.forward(q, k, v, key_padding_mask=kpm)
    refm = F.scaled_dot_product_attention(q, k, v, attn_mask=kpm[:, None, None, :], scale=SCALE)
    res["checks"]["masked_stays_sdpa"] = {"ok": torch.equal(out, refm)}
    if not torch.equal(out, refm):
        res["passed"] = False

    # kernel names on the candidate path
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        attn.forward(q, k, v)
        torch.cuda.synchronize()
    res["kernels"] = sorted({e.name[:140] for e in prof.events() if "flash" in e.name})
    res["compared_against"] = "torch SDPA flash backend (pytorch_flash::flash_fwd_kernel), bitwise"
    print(json.dumps(res))


if __name__ == "__main__":
    main()
