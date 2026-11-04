"""Correctness for the shared-expert row split (candidate tree only).

1. CPU partition check: GatedMLP's overridden mapping (Mapping(world 4, tp 2,
   pp 2, rank r), gated_mlp.py) for ranks 0-3 vs shared_expert_row_split_layout;
   for M in {128, 2047, 2048, 6145} the two ranks of each column shard own
   disjoint row ranges whose union is [0, M).
2. Real forward per rank (0-3, distinct routed outputs per rank) vs main's
   formula bf16(routed + bf16(0.5 * S)): owned rows = bf16(routed + S_rows),
   other rows = routed; owned-row S vs main's S/0.5 max abs diff; simulated
   4-rank all-reduce (fp32 sum) of both vs an fp32 reference.
3. Repeated execution: 8 iterations at M = 2048 and 6144 on ranks 0 and 2 with
   the bench's reset point (routed buffer restored), every iteration compared.
4. Main path kept: M < T (1, 2, 4, 8, 32, 127), CUDA-graph capture, kill switch.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_moe, build_shared_mlp, log  # noqa: E402

failures = []


def fail(**k):
    failures.append(k)
    log("FAIL", k)


def cpu_partition_check():
    from tensorrt_llm._torch.models.modeling_deepseekv4 import (shared_expert_row_range,
                                                                shared_expert_row_split_layout)
    from tensorrt_llm.mapping import Mapping

    tp, stp = 4, 2
    lay = {}
    for r in range(4):
        # Exactly gated_mlp.py's override: tp = overridden, pp = pp*tp // overridden.
        om = Mapping(world_size=stp * (1 * tp // stp), rank=r, gpus_per_node=8, tp_size=stp,
                     pp_size=1 * tp // stp)
        parts, idx, shard = shared_expert_row_split_layout(r, tp, stp)
        if not (shard == om.tp_rank and idx == om.pp_rank and parts == 2):
            fail(cpu="layout", rank=r, shard=shard, om_tp=om.tp_rank, idx=idx, om_pp=om.pp_rank)
        lay[r] = (parts, idx, shard)
    report = {"layout": {r: {"column_shard": v[2], "part": v[1]} for r, v in lay.items()}}
    for shard in range(stp):
        ranks = [r for r in range(4) if lay[r][2] == shard]
        if sorted(lay[r][1] for r in ranks) != [0, 1]:
            fail(cpu="parts", shard=shard, ranks=ranks)
    for M in (128, 2047, 2048, 6145):
        for shard in range(stp):
            ranks = [r for r in range(4) if lay[r][2] == shard]
            cover = torch.zeros(M, dtype=torch.int32)
            rngs = {}
            for r in ranks:
                s, e = shared_expert_row_range(M, lay[r][0], lay[r][1])
                rngs[r] = (s, e)
                cover[s:e] += 1
            ok = bool((cover == 1).all())
            if not ok:
                fail(cpu="cover", M=M, shard=shard, ranges=rngs)
            report[f"M{M}_shard{shard}"] = {"ranges": rngs, "each_row_once": ok}
    log("cpu partition", json.dumps(report))
    return report


def main_formula(mlp, scale, hs, routed):
    s = mlp(hs)
    s *= scale
    return s + routed


def main():
    os.environ.pop("TRTLLM_DSV4_SHARED_ROW_SPLIT", None)
    from tensorrt_llm._torch.models import modeling_deepseekv4 as m4

    T = m4.SHARED_ROW_SPLIT_MIN_TOKENS
    out = {"T": T, "cpu": cpu_partition_check()}
    torch.manual_seed(0)
    mlps = {}
    for shard in range(2):
        mlps[shard] = build_shared_mlp(shard, seed=1234 + shard)  # rank shard builds shard
    out["info"] = {s: v[3] for s, v in mlps.items()}
    moes = {}
    for r in range(4):
        mlp, stp, scale, info = mlps[r % 2]
        moe, stub, has = build_moe(r, mlp, stp, scale)
        assert has
        moes[r] = (moe, stub)
        log("rank", r, "parts", moe._shared_row_split_parts, "index", moe._shared_row_split_index)
        if moe._shared_row_split_parts != 2 or moe._shared_row_split_index != r // 2:
            fail(setup=r, parts=moe._shared_row_split_parts, idx=moe._shared_row_split_index)
    scale = mlps[0][2]
    hidden = mlps[0][3]["hidden"]
    gen = torch.Generator(device="cpu").manual_seed(11)

    # 2. numerics per rank + simulated all-reduce.
    numer = {}
    for M in (128, 2047, 2048, 6144, 6145):
        hs = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
        hs0 = hs.clone()
        routed = [torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda() for _ in range(4)]
        s_full = {sh: mlps[sh][0](hs) for sh in range(2)}
        base_pre, cand_pre, rep = [], [], {}
        for r in range(4):
            moe, stub = moes[r]
            mlp = mlps[r % 2][0]
            base = main_formula(mlp, scale, hs, routed[r])
            stub.buf = routed[r].clone()
            cand = moe.forward(hs)
            torch.cuda.synchronize()
            s, e = m4.shared_expert_row_range(M, 2, r // 2)
            s_own = mlp(hs[s:e].contiguous())
            d_s = (s_own.float() - s_full[r % 2][s:e].float()).abs().max().item()
            # main's S/0.5 (bf16(0.5*S) / 0.5 is exact for normal bf16)
            main_s = (s_full[r % 2] * scale) / scale
            d_main = (s_own.float() - main_s[s:e].float()).abs().max().item()
            exp_own = routed[r][s:e] + s_own
            own_ok = torch.equal(cand[s:e], exp_own)
            other = torch.cat([cand[:s], cand[e:]])
            other_ref = torch.cat([routed[r][:s], routed[r][e:]])
            other_ok = torch.equal(other, other_ref)
            if not (own_ok and other_ok and torch.isfinite(cand.float()).all()):
                fail(numer=M, rank=r, own_ok=own_ok, other_ok=other_ok)
            if not torch.equal(hs, hs0):
                fail(numer=M, rank=r, input_mutated=True)
            rep[r] = {"rows": [s, e], "owned_S_vs_fullM_S_maxabs": d_s,
                      "owned_S_vs_main_S_over_0.5_maxabs": d_main,
                      "owned_rows_bitwise_S": d_s == 0.0, "owned_eq_routed_plus_S": own_ok,
                      "other_rows_eq_routed": other_ok,
                      "pre_ar_vs_main_maxabs": (cand.float() - base.float()).abs().max().item()}
            base_pre.append(base.float())
            cand_pre.append(cand.float())
        ref32 = sum(x.float() for x in routed) + s_full[0].float() + s_full[1].float()
        ar_base = sum(base_pre)
        ar_cand = sum(cand_pre)
        mag = ref32.abs().max().item()
        rep["allreduce_sim"] = {
            "cand_vs_base_maxabs": (ar_cand - ar_base).abs().max().item(),
            "base_vs_fp32ref_maxabs": (ar_base - ref32).abs().max().item(),
            "cand_vs_fp32ref_maxabs": (ar_cand - ref32).abs().max().item(),
            "ref_absmax": mag}
        if not torch.allclose(ar_cand, ar_base, rtol=1.6e-2, atol=1.6e-2 * max(1.0, mag) / 8):
            fail(ar=M)
        numer[M] = rep
        log("numerics M", M, json.dumps(rep))
    out["numerics"] = numer

    # 3. repeated execution, 8 iterations, reset points as in bench.
    repeated = {}
    for M in (2048, 6144):
        hs = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
        hs0 = hs.clone()
        for r in (0, 2):
            moe, stub = moes[r]
            mlp = mlps[r % 2][0]
            routed0 = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
            s, e = m4.shared_expert_row_range(M, 2, r // 2)
            ref = routed0.clone()
            ref[s:e] = routed0[s:e] + mlp(hs[s:e].contiguous())
            buf = torch.empty_like(routed0)
            stub.buf = buf
            per_it = []
            for it in range(8):
                buf.copy_(routed0)  # reset point
                y = moe.forward(hs)
                torch.cuda.synchronize()
                d = (y.float() - ref.float()).abs().max().item()
                ok = torch.equal(y, ref) and torch.equal(hs, hs0) and y.data_ptr() == buf.data_ptr()
                per_it.append({"it": it, "maxabs": d, "rel": d / max(ref.abs().max().item(), 1e-9),
                               "bitwise": ok})
                if not ok:
                    fail(repeated=M, rank=r, it=it, d=d)
            repeated[f"M{M}_r{r}"] = per_it
            log("repeated", M, r, json.dumps(per_it))
    out["repeated"] = repeated

    # 4a. M < T keeps main's path (decode M <= 8 among them).
    small = {}
    for M in (1, 2, 4, 8, 32, T - 1):
        hs = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
        routed = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
        moe, stub = moes[2]
        took = moe._use_shared_row_split(hs, None, True)
        stub.buf = routed.clone()
        y = moe.forward(hs)
        ok = (not took) and torch.equal(y, main_formula(mlps[0][0], scale, hs, routed))
        small[M] = ok
        if not ok:
            fail(small=M, took=took)
    out["small_M_main_path"] = small
    log("small", small)

    # 4b. CUDA-graph capture never takes the split (predicate + captured forward).
    cap = {}
    M = 256
    hs = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
    routed = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
    moe, stub = moes[2]
    stub.buf = routed
    eager_took = moe._use_shared_row_split(hs, None, True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    flags = []
    try:
        with torch.cuda.stream(s):
            for _ in range(2):
                moe.forward(hs)  # main-like warm (eager split path; no in-place on graph buf)
                stub.buf = routed.clone()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        gbuf = routed.clone()
        stub.buf = gbuf
        with torch.cuda.graph(graph):
            flags.append(moe._use_shared_row_split(hs, None, True))
            gy = moe.forward(hs)
        gbuf.copy_(routed)
        graph.replay()
        torch.cuda.synchronize()
        ref = main_formula(mlps[0][0], scale, hs, routed)
        cap = {"eager_takes_split": eager_took, "capture_takes_split": flags[0],
               "graph_out_eq_main": torch.equal(gy, ref)}
    except Exception as ex:  # noqa: BLE001
        cap = {"eager_takes_split": eager_took, "capture_takes_split": flags[0] if flags else None,
               "capture_error": repr(ex)[:300]}
    if cap.get("capture_takes_split") is not False or cap.get("graph_out_eq_main") is False:
        fail(capture=cap)
    out["capture"] = cap
    log("capture", cap)

    # 4c. Kill switch.
    os.environ["TRTLLM_DSV4_SHARED_ROW_SPLIT"] = "0"
    mlp, stp, sc, _ = mlps[0]
    moe_off, stub_off, _ = build_moe(2, mlp, stp, sc)
    os.environ.pop("TRTLLM_DSV4_SHARED_ROW_SPLIT")
    M = 2048
    hs = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
    routed = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda()
    stub_off.buf = routed.clone()
    y = moe_off.forward(hs)
    ks = {"parts": moe_off._shared_row_split_parts,
          "out_eq_main": torch.equal(y, main_formula(mlp, sc, hs, routed))}
    if ks["parts"] != 0 or not ks["out_eq_main"]:
        fail(kill_switch=ks)
    out["kill_switch"] = ks
    log("kill switch", ks)

    print(json.dumps({"passed": not failures,
                      "compared": "real DeepseekV4MoE.forward (row split) vs main's formula "
                                  "bf16(routed + bf16(0.5*S)) and vs routed + S_rows; simulated "
                                  "4-rank AR vs fp32 reference; 8 repeated iterations with reset",
                      "failures": failures[:20], **out}, default=str))


if __name__ == "__main__":
    main()
