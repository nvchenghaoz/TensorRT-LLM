"""Idea 237 Stage 0: prefill routed MoE (M > 32) on FlashInfer's SM12x MXFP8xMXFP4 chain vs main CUTLASS.

One EP2xTP2 rank of DeepSeek-V4.1-Flash: 192 local of 384 experts (rank 0 owns [0, 192)), hidden 5120,
moe_intermediate / 2 per rank, top-6, swiglu_limit from config.json. Synthetic MXFP4 weights
(random e2m1 codes, ue8m0 block exponents) in the TRT-LLM CUTLASS layout; the FlashInfer state is
built from them exactly as the leaf builds it (Sm12xW4a8DecodeState, SF re-laid out once).

Paths per (M, D = active local experts):
  CUTLASS : what main's leaf runs at M (M >= fold_min: fused_moe(bf16 x) with in-kernel MXFP8 quant;
            else mxfp8_quantize + fused_moe), autotuned, fused finalize.
  FI      : route (mxfp8_q0_route_triton, >256-pair prefill path, expert_offset remap + zero_buf) ->
            fc1_act_q1 (SwiGLU + MXFP8 quant epilogue) -> fc2_finalize (scatter), at the heuristic tiles
            and at explicit tile variants.

Timing: whole op, L2 flushed by a READ-only pass over 512 MiB before every timed call; eager (events)
and CUDA-graph replay; host us/call (perf_counter around the enqueue). Kineto per-kernel split (eager).
Accuracy: FP32 dequantized reference (BF16 x, unquantized), FI vs ref, CUTLASS vs ref, FI vs CUTLASS.
"""

import argparse
import json
import os
import subprocess
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402
from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8 as leaf  # noqa: E402
from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as dec  # noqa: E402

T0 = time.time()
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def read_cfg(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    if "text_config" in cfg:
        cfg = {**cfg, **cfg["text_config"]}
    return (cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"],
            cfg["num_experts_per_tok"], cfg.get("swiglu_limit", None))


def dequant(codes, sf_raw):
    n, kh = codes.shape
    k = kh * 2
    lut = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32, device=codes.device)
    vals = torch.empty(n, k, dtype=torch.float32, device=codes.device)
    vals[:, 0::2] = lut[(codes & 0x0F).long()]
    vals[:, 1::2] = lut[(codes >> 4).long()]
    scale = torch.pow(2.0, sf_raw.float() - 127.0)
    return (vals.view(n, k // 32, 32) * scale[:, :, None]).view(n, k)


class Problem:
    def __init__(self, H, I, E, E_glob, topk, limit, seed=0):
        self.H, self.I, self.E, self.E_glob, self.topk, self.limit = H, I, E, E_glob, topk, limit
        dev = "cuda"
        g = torch.Generator(device=dev).manual_seed(1234 + seed)
        self.w1 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
        self.w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=g)
        self.s1 = torch.randint(118, 123, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
        self.s2 = torch.randint(118, 123, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g)
        self.w31 = self.w1.view(torch.int64)
        self.w2v = self.w2.view(torch.int64)
        self.s31 = torch.ops.trtllm.block_scale_interleave(self.s1.contiguous()).view(torch.int32).reshape(
            E, 2 * I, H // 128)
        self.s2s = torch.ops.trtllm.block_scale_interleave(self.s2.contiguous()).view(torch.int32).reshape(
            E, H, I // 128)
        ones = torch.ones(E, dtype=torch.float32, device=dev)
        self.qs = [self.s31, ones, self.s2s, ones.clone()]
        self.limit_t = None if limit is None else torch.full((E,), float(limit), dtype=torch.float32, device=dev)
        from tensorrt_llm._torch.utils import ActivationType
        self.act = int(ActivationType.Swiglu)
        self.st = dec.Sm12xW4a8DecodeState(
            w3_w1_weight=self.w31, w3_w1_weight_scale=self.s31, w2_weight=self.w2v,
            w2_weight_scale=self.s2s, num_local_experts=E, slot_start=0, swiglu_limit=limit,
            enable_pdl=os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1")
        fmax = leaf._MXFP8_FOLD_MAX_TOKENS
        self.fold_min, self.fold_max = leaf._MXFP8_FOLD_MIN_TOKENS, (1 << 30) if fmax < 0 else fmax

    def inputs(self, m, d, seed=0):
        """Global top-k ids; local picks (rank 0 owns [0, E)) drawn from D active local experts."""
        gc = torch.Generator().manual_seed(17 + 31 * m + 7 * d + 1000 * seed)
        active = torch.randperm(self.E, generator=gc)[:d]
        nloc = (torch.rand(m, self.topk, generator=gc) < 0.5).sum(1)
        nloc = torch.clamp(nloc, max=min(self.topk, d))
        sel = torch.empty(m, self.topk, dtype=torch.int64)
        for t in range(m):
            nl = int(nloc[t])
            loc = active[torch.randperm(d, generator=gc)[:nl]]
            rem = torch.randperm(self.E_glob - self.E, generator=gc)[: self.topk - nl] + self.E
            row = torch.cat([loc, rem])
            sel[t] = row[torch.randperm(self.topk, generator=gc)]
        # every active expert gets a row: replace a local pick of an expert that has > 1 rows
        for _ in range(4):
            cnt = torch.bincount(sel[sel < self.E], minlength=self.E)
            missing = [int(e) for e in active if cnt[e] == 0]
            if not missing:
                break
            for e in missing:
                cnt = torch.bincount(sel[sel < self.E], minlength=self.E)
                done = False
                for t in torch.randperm(m, generator=gc).tolist():
                    for j in range(self.topk):
                        v = int(sel[t, j])
                        if v < self.E and cnt[v] > 1 and not (sel[t] == e).any():
                            sel[t, j] = e
                            done = True
                            break
                    if done:
                        break
                if not done:  # replace a remote pick
                    for t in torch.randperm(m, generator=gc).tolist():
                        js = [j for j in range(self.topk) if sel[t, j] >= self.E]
                        if js and not (sel[t] == e).any():
                            sel[t, js[0]] = e
                            break
        g = torch.Generator(device="cuda").manual_seed(99 + 7 * m + d + 1000 * seed)
        x = (torch.randn(m, self.H, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        sc = torch.softmax(torch.randn(m, self.topk, generator=gc), dim=-1).float()
        sel = sel.to(torch.int32).cuda()
        loc = sel[sel < self.E]
        n_active = int((torch.bincount(loc.long(), minlength=self.E) > 0).sum())
        return x.contiguous(), sel, sc.cuda(), n_active, int(loc.numel())

    def cutlass(self, x, sel, sc):
        m = x.shape[0]
        fold = self.fold_max > 0 and self.fold_min <= m <= self.fold_max
        if fold:
            xin, xsf = x, None
        else:
            xin, xsf = torch.ops.trtllm.mxfp8_quantize(x, True, alignment=128)
        return torch.ops.trtllm.fused_moe(
            xin, sel, sc, self.w31, None, self.w2v, None, torch.bfloat16,
            quant_scales=self.qs, input_sf=xsf, swizzled_input_sf=True,
            swiglu_alpha=None, swiglu_beta=None, swiglu_limit=self.limit_t,
            tp_size=2, tp_rank=0, ep_size=2, ep_rank=0, cluster_size=1, cluster_rank=0,
            enable_alltoall=False, use_deepseek_fp8_block_scale=False, use_w4_group_scaling=False,
            use_int8_woq_per_channel=False, use_mxfp8_act_scaling=True, min_latency_mode=False,
            use_fused_finalize=True, tune_max_num_tokens=8192, activation_type=self.act,
            unpadded_hidden_size=self.H)[0]

    def fi(self, x, sel, sc, t1=None, t2=None):
        """Sm12xW4a8DecodeState.run's two-kernel chain (fused route prologue path) with optional tiles."""
        st = self.st
        route, _, fc1, _, fc2, fi_act = dec._load_ops()
        m, pairs = x.shape[0], sel.numel()
        sf1_shape, n_q1, n_sf1, off_sf1, off_out, nbytes = st._buf_layout(m, pairs)
        ws = dec._workspace(x, sel, st.num_experts)
        buf = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
        q1 = buf[:n_q1].view(torch.float8_e4m3fn).view(pairs, st.inter)
        sf1 = buf[off_sf1:off_sf1 + n_sf1].view(torch.int32).view(sf1_shape)
        out = buf[off_out:].view(torch.bfloat16).view(m, st.hidden)
        offsets, token_map, rw, a_q, a_sf = route(
            x, sel, sc, st.num_experts, workspace=ws, enable_pdl=st.enable_pdl,
            expert_offset=st.slot_start, zero_buf=buf)
        kw1 = {}
        if t1 is not None:
            kw1["tile"] = t1
            tac = FC1.epi_tactics(t1)
            kw1["epi"] = FC1.DEFAULT_EPI if FC1.DEFAULT_EPI in tac else tac[0]
        fc1(a_q, a_sf, st.w1, st.w1_sf, offsets, tune=False, enable_pdl=st.enable_pdl,
            out_q=q1, out_sf=sf1, activation=fi_act.Swiglu, swiglu_limit=st.swiglu_limit, **kw1)
        kw2 = {} if t2 is None else {"tile": t2}
        fc2(q1, sf1, st.w2, st.w2_sf, offsets, token_map, rw, m, tune=False,
            enable_pdl=st.enable_pdl, out=out, **kw2)
        return out

    def reference(self, x, sel, sc):
        m = x.shape[0]
        out = torch.zeros(m, self.H, dtype=torch.float32, device=x.device)
        xf = x.float()
        L = self.limit
        for e in torch.unique(sel[sel < self.E]).tolist():
            t, j = (sel == e).nonzero(as_tuple=True)
            W1 = dequant(self.w1[e], self.s1[e])
            h = xf[t] @ W1.t()
            del W1
            lin, gate = h[:, :self.I], h[:, self.I:]
            if L is not None:
                gate = gate.clamp(max=L)
                lin = lin.clamp(-L, L)
            act = gate * torch.sigmoid(gate) * lin
            W2 = dequant(self.w2[e], self.s2[e])
            y = (act @ W2.t()) * sc[t, j][:, None]
            out.index_add_(0, t, y)
        return out


import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1 as FC1  # noqa: E402
import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize as FC2  # noqa: E402


def err(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    bmax = b.abs().max().item()
    rel = d / b.abs().clamp_min(0.05 * max(bmax, 1e-6))
    num = (a.double() * b.double()).sum().item()
    den = (a.double() ** 2 + b.double() ** 2).sum().item()
    return {"max_abs": d.max().item(), "max_abs_over_absmax": d.max().item() / max(bmax, 1e-12),
            "max_rel_floor5pct": rel.max().item(), "calc_diff": 0.0 if den == 0 else 1 - 2 * num / den,
            "finite": bool(torch.isfinite(a).all().item())}


FLUSH = None


def flush():
    FLUSH.sum()  # read-only L2 flush (one 4-byte write)


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    return {"median_us": med * 1e3, "p10_us": ts[int(0.1 * (n - 1))] * 1e3,
            "p90_us": ts[int(0.9 * (n - 1))] * 1e3,
            "noise": (ts[int(0.9 * (n - 1))] - ts[int(0.1 * (n - 1))]) / med, "n": n}


def time_eager(fn, reps):
    gpu, host = [], []
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    for _ in range(reps):
        flush()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        h0 = time.perf_counter()
        fn()
        h1 = time.perf_counter()
        b.record()
        b.synchronize()
        gpu.append(a.elapsed_time(b))
        host.append((h1 - h0) * 1e3)
    return stats(gpu), stats(host)


def time_graph(fn, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        flush()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    o = out.clone()
    del g
    return stats(ts), o


def kernel_split(fn, iters=5):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            flush()
            fn()
        torch.cuda.synchronize()
    rows = {}
    for ev in prof.key_averages():
        t = getattr(ev, "device_time_total", None)
        if t is None:
            t = getattr(ev, "cuda_time_total", 0)
        if t <= 0 or "reduce_kernel" in ev.key:
            continue
        rows[ev.key[:110]] = {"us_per_call": t / iters, "calls_per_call": ev.count / iters}
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]["us_per_call"]))


def main():
    global FLUSH
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cases", default="128:66,2048:120,2048:150,2048:192,6144:192")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--graph_reps", type=int, default=15)
    ap.add_argument("--variants", type=int, default=1)
    args = ap.parse_args()
    try:
        log(subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,clocks.sm",
                            "--format=csv"], capture_output=True, text=True, timeout=30).stdout)
        log(subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip())
    except Exception as e:  # noqa: BLE001
        log("nvidia-smi failed", e)
    H, I_full, E_glob, topk, limit = read_cfg(args.ckpt)
    E, I = E_glob // 2, I_full // 2
    log(f"EP2xTP2 rank0: H={H} I={I} E_local={E}/{E_glob} topk={topk} limit={limit} gpu={torch.cuda.get_device_name()}")
    ok, why = dec.fi_decode_available()
    log("fi available:", ok, why, "route_fuses_prologue:", dec._route_fuses_prologue())
    p = Problem(H, I, E, E_glob, topk, limit)
    log(f"fold_min={p.fold_min} fold_max={p.fold_max} pdl={p.st.enable_pdl}")
    FLUSH = torch.ones(128 * 1024 * 1024, dtype=torch.float32, device="cuda")
    cases = [tuple(int(v) for v in c.split(":")) for c in args.cases.split(",")]
    inputs = {}
    for m, d in cases:
        inputs[(m, d)] = p.inputs(m, d)
        log(f"case M={m} D={d}: active={inputs[(m, d)][3]} local_pairs={inputs[(m, d)][4]}")
    # CUTLASS autotune, as the server warmup does
    with autotune():
        for m, d in cases:
            x, sel, sc, _, _ = inputs[(m, d)]
            p.cutlass(x, sel, sc)
    torch.cuda.synchronize()
    try:
        picks = sorted(str((k[0], k[3] if len(k) > 3 else None, v[1]))[:160]
                       for k, v in AutoTuner.get().profiling_cache.cache.items()
                       if str(k[0]).startswith("trtllm::fused_moe"))
        log("cutlass tuned tactics:", picks)
    except Exception as e:  # noqa: BLE001
        log("tactic listing failed", repr(e))
    BK1, BK2 = FC1.BK, FC2.BK
    table = {}
    for m, d in cases:
        x, sel, sc, n_active, n_loc = inputs[(m, d)]
        pairs = sel.numel()
        key = f"M{m}_D{d}"
        t1d = FC1.select_tile(total_rows=pairs, n=I, k=H, num_experts=E, num_sms=188)
        t2d = FC2.select_tile(total_rows=pairs, n=H, num_experts=E, num_sms=188)
        log(f"{key}: FI heuristic tiles fc1={t1d[:2]} fc2={t2d[:2]}")
        row = {"active": n_active, "local_pairs": n_loc, "fi_tiles": {"fc1": list(t1d[:2]), "fc2": list(t2d[:2])}}
        ref = p.reference(x, sel, sc)
        torch.cuda.synchronize()
        # correctness first (eager)
        oc = p.cutlass(x, sel, sc).clone()
        of = p.fi(x, sel, sc).clone()
        torch.cuda.synchronize()
        row["err"] = {"cutlass_vs_ref": err(oc, ref), "fi_vs_ref": err(of, ref), "fi_vs_cutlass": err(of, oc)}
        log(key, "err", json.dumps(row["err"]))
        # timing
        ce, ch = time_eager(lambda: p.cutlass(x, sel, sc), args.reps)
        fe, fh = time_eager(lambda: p.fi(x, sel, sc), args.reps)
        cg, ocg = time_graph(lambda: p.cutlass(x, sel, sc), args.graph_reps)
        fg, ofg = time_graph(lambda: p.fi(x, sel, sc), args.graph_reps)
        row["cutlass"] = {"eager": ce, "host": ch, "graph": cg}
        row["fi"] = {"eager": fe, "host": fh, "graph": fg}
        row["graph_err"] = {"fi_graph_vs_ref": err(ofg, ref), "cutlass_graph_vs_ref": err(ocg, ref)}
        log(f"{key} CUTLASS eager={ce['median_us']:.1f}us (n{ce['noise']:.3f}) graph={cg['median_us']:.1f}us "
            f"(n{cg['noise']:.3f}) host={ch['median_us']:.1f}us")
        log(f"{key} FI      eager={fe['median_us']:.1f}us (n{fe['noise']:.3f}) graph={fg['median_us']:.1f}us "
            f"(n{fg['noise']:.3f}) host={fh['median_us']:.1f}us  delta_graph={fg['median_us'] - cg['median_us']:+.1f}us")
        row["split_cutlass"] = kernel_split(lambda: p.cutlass(x, sel, sc))
        row["split_fi"] = kernel_split(lambda: p.fi(x, sel, sc))
        for nm in ("split_cutlass", "split_fi"):
            for k, v in row[nm].items():
                log(f"{key} {nm}: {v['us_per_call']:8.1f} us x{v['calls_per_call']:.0f}  {k}")
        if args.variants:
            var = {}
            fc1_tiles = [(64, 128), (32, 128)] + ([(8, 128)] if pairs // E <= 64 else [])
            fc2_tiles = [(128, 128), (64, 128), (32, 128)]
            combos = [(a, t2d[:2]) for a in fc1_tiles] + [(t1d[:2], b) for b in fc2_tiles]
            seen = set()
            for a, b in combos:
                if (tuple(a), tuple(b)) in seen:
                    continue
                seen.add((tuple(a), tuple(b)))
                t1, t2 = (a[0], a[1], BK1), (b[0], b[1], BK2)
                vk = f"fc1_{a[0]}x{a[1]}_fc2_{b[0]}x{b[1]}"
                try:
                    o = p.fi(x, sel, sc, t1, t2).clone()
                    torch.cuda.synchronize()
                    e = err(o, ref)
                    vg, _ = time_graph(lambda: p.fi(x, sel, sc, t1, t2), args.graph_reps)
                    var[vk] = {"graph": vg, "err_vs_ref": e}
                    log(f"{key} FI variant {vk}: graph={vg['median_us']:.1f}us (n{vg['noise']:.3f}) "
                        f"max_abs/absmax={e['max_abs_over_absmax']:.3e} finite={e['finite']}")
                except Exception as ex:  # noqa: BLE001
                    var[vk] = {"error": repr(ex)[:300]}
                    log(f"{key} FI variant {vk} failed: {ex!r}"[:400])
            row["fi_variants"] = var
        table[key] = row
        del ref, oc, of, ocg, ofg
        torch.cuda.empty_cache()
    summ = {k: {"cutlass_graph_us": v["cutlass"]["graph"]["median_us"], "fi_graph_us": v["fi"]["graph"]["median_us"],
                "cutlass_eager_us": v["cutlass"]["eager"]["median_us"], "fi_eager_us": v["fi"]["eager"]["median_us"],
                "cutlass_host_us": v["cutlass"]["host"]["median_us"], "fi_host_us": v["fi"]["host"]["median_us"],
                "fi_tiles": v["fi_tiles"],
                "fi_vs_ref_absmax": v["err"]["fi_vs_ref"]["max_abs_over_absmax"],
                "cutlass_vs_ref_absmax": v["err"]["cutlass_vs_ref"]["max_abs_over_absmax"],
                "fi_vs_cutlass_absmax": v["err"]["fi_vs_cutlass"]["max_abs_over_absmax"],
                "best_fi_variant": min(((vk, vv["graph"]["median_us"]) for vk, vv in v.get("fi_variants", {}).items()
                                        if "graph" in vv), key=lambda z: z[1], default=None)}
            for k, v in table.items()}
    log("SUMMARY", json.dumps(summ, indent=1))
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    try:
        with open(os.path.join(out_dir, "stage0_237.json"), "w") as f:
            json.dump(table, f, indent=1)
    except OSError as e:
        log("artifact write failed", e)
    print(json.dumps({"stage0": summ}))


if __name__ == "__main__":
    main()
