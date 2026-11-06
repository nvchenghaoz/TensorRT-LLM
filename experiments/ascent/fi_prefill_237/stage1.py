"""Idea 237 Stage 1: tile sweep of the FI SM12x prefill chain per M bucket (EP2xTP2 rank 0), vs CUTLASS.

Full fc1 x fc2 tile grid per (M, D), CUDA-graph replay with a read-only L2 flush, plus a host-us split
(route / fc1 / fc2 enqueue) of the heuristic chain.  Prints one JSON line with the table.
"""

import argparse
import json
import math
import time

import torch

import stage0 as S


def host_split(p, x, sel, sc, reps=20):
    st = p.st
    route, mkws, fc1, _, fc2, fi_act = S.dec._load_ops()
    m, pairs = x.shape[0], sel.numel()
    res = {"buf_ws": [], "route": [], "fc1": [], "fc2": []}
    for i in range(reps + 3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sf1_shape, n_q1, n_sf1, off_sf1, off_out, nbytes = st._buf_layout(m, pairs)
        ws = mkws(x, sel, st.num_experts)
        buf = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
        q1 = buf[:n_q1].view(torch.float8_e4m3fn).view(pairs, st.inter)
        sf1 = buf[off_sf1:off_sf1 + n_sf1].view(torch.int32).view(sf1_shape)
        out = buf[off_out:].view(torch.bfloat16).view(m, st.hidden)
        t1 = time.perf_counter()
        offsets, token_map, rw, a_q, a_sf = route(
            x, sel, sc, st.num_experts, workspace=ws, enable_pdl=st.enable_pdl,
            expert_offset=st.slot_start, zero_buf=buf)
        t2 = time.perf_counter()
        fc1(a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False, enable_pdl=st.enable_pdl,
            out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=st.swiglu_limit)
        t3 = time.perf_counter()
        fc2(q1, sf1, st.w2, st.w2_sf, offsets, token_map, rw, m, tune=False, enable_pdl=st.enable_pdl, out=out)
        t4 = time.perf_counter()
        if i >= 3:
            for k, a, b in (("buf_ws", t0, t1), ("route", t1, t2), ("fc1", t2, t3), ("fc2", t3, t4)):
                res[k].append((b - a) * 1e6)
    return {k: sorted(v)[len(v) // 2] for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ms", default="48,64,96,128,256,512,1024,1536,2048,3072,4096,6144,8192")
    ap.add_argument("--graph_reps", type=int, default=15)
    args = ap.parse_args()
    H, I_full, E_glob, topk, limit = S.read_cfg(args.ckpt)
    E, I = E_glob // 2, I_full // 2
    p = S.Problem(H, I, E, E_glob, topk, limit)
    S.FLUSH = torch.ones(128 * 1024 * 1024, dtype=torch.float32, device="cuda")
    ms = [int(v) for v in args.ms.split(",")]
    cases = []
    for m in ms:
        d_unif = min(E, max(1, round(E * (1 - math.exp(-3.0 * m / E)))))
        cases.append((m, d_unif))
        if m >= 1024 and d_unif > 120:
            cases.append((m, 120))
    inputs = {c: p.inputs(*c) for c in cases}
    with S.autotune():
        for c in cases:
            x, sel, sc, _, _ = inputs[c]
            p.cutlass(x, sel, sc)
    torch.cuda.synchronize()
    S.log("cases:", [(c, inputs[c][3], inputs[c][4]) for c in cases])
    BK1, BK2 = S.FC1.BK, S.FC2.BK
    table = {}
    for m, d in cases:
        x, sel, sc, n_active, n_loc = inputs[(m, d)]
        pairs = sel.numel()
        key = f"M{m}_D{d}"
        t1d = S.FC1.select_tile(total_rows=pairs, n=I, k=H, num_experts=E, num_sms=188)[:2]
        t2d = S.FC2.select_tile(total_rows=pairs, n=H, num_experts=E, num_sms=188)[:2]
        cg, oc = S.time_graph(lambda: p.cutlass(x, sel, sc), args.graph_reps)
        row = {"active": n_active, "local_pairs": n_loc, "heur": [list(t1d), list(t2d)],
               "cutlass_graph_us": cg["median_us"], "cutlass_noise": cg["noise"], "fi": {}}
        fc1_tiles = [(8, 128), (32, 128), (64, 128)]
        fc2_tiles = [(32, 128), (64, 128), (128, 128)]
        for a in fc1_tiles:
            for b in fc2_tiles:
                t1, t2 = (a[0], a[1], BK1), (b[0], b[1], BK2)
                vk = f"{a[0]}/{b[0]}"
                try:
                    vg, o = S.time_graph(lambda: p.fi(x, sel, sc, t1, t2), args.graph_reps)
                    e = S.err(o, oc)
                    row["fi"][vk] = {"us": vg["median_us"], "noise": vg["noise"],
                                     "vs_cutlass_absmax": e["max_abs_over_absmax"], "finite": e["finite"]}
                except Exception as ex:  # noqa: BLE001
                    row["fi"][vk] = {"error": repr(ex)[:200]}
        best = min(((k, v["us"]) for k, v in row["fi"].items() if "us" in v), key=lambda z: z[1])
        hk = f"{t1d[0]}/{t2d[0]}"
        row["best"] = best
        row["heur_us"] = row["fi"].get(hk, {}).get("us")
        row["host_split_us"] = host_split(p, x, sel, sc)
        table[key] = row
        S.log(f"{key} act={n_active} loc={n_loc} CUTLASS={cg['median_us']:.1f} heur {hk}={row['heur_us']:.1f} "
              f"best {best[0]}={best[1]:.1f} ({100 * (best[1] / cg['median_us'] - 1):+.1f}%) host={row['host_split_us']}")
        S.log("   grid:", {k: round(v.get("us", -1), 1) for k, v in row["fi"].items()})
        torch.cuda.empty_cache()
    print(json.dumps({"stage1": table}))


if __name__ == "__main__":
    main()
