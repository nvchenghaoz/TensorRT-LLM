"""Gemma4 SM120 prefill-attention census (analysis job, idea 10).

Times FlashInfer FA2 paged prefill exactly as TRT-LLM's FLASHINFER backend calls it
(HND paged FP8 KV, page 32, BF16 Q, causal, window_left = sliding_window - 1 on
sliding layers) and the alternatives:
  fa2          main: q bf16, kv fp8 (in-kernel dequant), o bf16
  fa2_fp8q     q fp8 e4m3, kv fp8, o bf16 (FP8 MMA path of the pinned FlashInfer)
  fa2_bf16kv   q bf16, kv bf16 (same values as the fp8 cache, exactly representable)
  cudnn_*      backend='cudnn' paged prefill
  fmhav2_*     flashinfer.trtllm_fmha_v2_prefill Q_PAGED_KV_HND (SM120 HMMA)
  prims_fp8    backend='cute-dsl-prims' (SM120 FP8 FMHA; d<=256, no window)
  b12x_fp8kv   b12x paged extend (BF16 Q + FP8 KV)
Each variant: plan once, untimed repeated-execution sequence (warmup+iters calls,
bitwise vs the first call), CUDA-event timing (median/p10/p90), and error vs an
independent FP32 torch reference on the dequantized KV.
Synthetic tensors only; dims read from <checkpoint>/config.json.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback

import torch

PAGE = 32


def log(*a):
    print(*a, flush=True)


def load_dims(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    nq = tc["num_attention_heads"]
    nkv = tc["num_key_value_heads"]
    hd = tc["head_dim"]
    ghd = tc.get("global_head_dim") or hd
    gkv = nkv
    if tc.get("attention_k_eq_v", False):
        gkv = tc.get("num_global_key_value_heads") or nkv
    sw = tc.get("sliding_window")
    lt = tc.get("layer_types") or []
    d = dict(num_heads=nq, sliding_kv_heads=nkv, sliding_head_dim=hd,
             global_kv_heads=gkv, global_head_dim=ghd, sliding_window=sw,
             attention_k_eq_v=tc.get("attention_k_eq_v"),
             num_layers=tc.get("num_hidden_layers"),
             n_sliding=sum(1 for t in lt if "sliding" in t),
             n_global=sum(1 for t in lt if "full" in t or "global" in t),
             hidden=tc.get("hidden_size"))
    qpath = os.path.join(ckpt, "hf_quant_config.json")
    if os.path.exists(qpath):
        q = json.load(open(qpath)).get("quantization", {})
        d["kv_cache_quant_algo"] = q.get("kv_cache_quant_algo")
        d["quant_algo"] = q.get("quant_algo")
    return d


class Case:
    def __init__(self, layer, batch, seqlen, dims, seed=0):
        self.layer = layer
        self.batch = batch
        self.seqlen = seqlen
        self.Hq = dims["num_heads"]
        if layer == "sliding":
            self.Hkv = dims["sliding_kv_heads"]
            self.D = dims["sliding_head_dim"]
            self.window_left = dims["sliding_window"] - 1
        else:
            self.Hkv = dims["global_kv_heads"]
            self.D = dims["global_head_dim"]
            self.window_left = -1
        self.sm_scale = 1.0 / math.sqrt(self.D)
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        lens = [seqlen] * batch
        self.lens = lens
        npages = [(L + PAGE - 1) // PAGE for L in lens]
        total_pages = sum(npages)
        self.total_q = sum(lens)
        qo = [0]
        kvp = [0]
        for L, n in zip(lens, npages):
            qo.append(qo[-1] + L)
            kvp.append(kvp[-1] + n)
        self.qo_indptr = torch.tensor(qo, dtype=torch.int32, device=dev)
        self.kv_indptr = torch.tensor(kvp, dtype=torch.int32, device=dev)
        perm = torch.randperm(total_pages, generator=g, device=dev).to(torch.int32)
        self.kv_indices = perm
        self.last_page_len = torch.tensor([(L - 1) % PAGE + 1 for L in lens],
                                          dtype=torch.int32, device=dev)
        maxp = max(npages)
        bt = torch.zeros(batch, maxp, dtype=torch.int32, device=dev)
        for r in range(batch):
            bt[r, :npages[r]] = perm[kvp[r]:kvp[r + 1]]
        self.block_tables = bt
        self.seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
        # HND pool [pages, 2, Hkv, P, D]
        pool = torch.randn(total_pages, 2, self.Hkv, PAGE, self.D, generator=g,
                           device=dev, dtype=torch.float32)
        self.kv_fp8 = pool.to(torch.float8_e4m3fn)
        self.kv_bf16 = self.kv_fp8.to(torch.bfloat16)  # exact
        self.q = torch.randn(self.total_q, self.Hq, self.D, generator=g, device=dev,
                             dtype=torch.float32).to(torch.bfloat16)
        self.q_fp8 = self.q.to(torch.float8_e4m3fn)
        # causal(+window) pair count
        cnt = 0
        for L in lens:
            if self.window_left < 0:
                cnt += L * (L + 1) // 2
            else:
                W = self.window_left + 1
                full = max(0, L - W)
                cnt += sum(min(i + 1, W) for i in range(min(L, W))) + full * W
        self.pairs = cnt
        self.flops = 4.0 * self.Hq * self.D * cnt

    def label(self):
        return f"{self.layer}_d{self.D}_{self.batch}x{self.seqlen}"

    def gather_kv(self, r):
        s, e = int(self.kv_indptr[r]), int(self.kv_indptr[r + 1])
        idx = self.kv_indices[s:e].long()
        pages = self.kv_fp8[idx].float()  # [n,2,H,P,D]
        L = self.lens[r]
        k = pages[:, 0].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:L]
        v = pages[:, 1].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:L]
        return k, v

    @torch.no_grad()
    def reference(self, q_src=None):
        """FP32 softmax(QK^T*s + mask)V on dequantized KV."""
        q_all = (self.q if q_src is None else q_src).float()
        out = torch.empty(self.total_q, self.Hq, self.D, dtype=torch.float32,
                          device="cuda")
        grp = self.Hq // self.Hkv
        for r in range(self.batch):
            s, e = int(self.qo_indptr[r]), int(self.qo_indptr[r + 1])
            L = e - s
            k, v = self.gather_kv(r)
            i = torch.arange(L, device="cuda").view(-1, 1)
            j = torch.arange(L, device="cuda").view(1, -1)
            mask = j <= i
            if self.window_left >= 0:
                mask &= (i - j) <= self.window_left
            for h in range(self.Hq):
                kh = k[:, h // grp]
                vh = v[:, h // grp]
                qh = q_all[s:e, h]
                sc = (qh @ kh.T) * self.sm_scale
                sc.masked_fill_(~mask, float("-inf"))
                p = torch.softmax(sc, dim=-1)
                out[s:e, h] = p @ vh
        return out


def err_stats(out, ref):
    o = out.float()
    d = (o - ref)
    return dict(max_abs=float(d.abs().max()),
                rel_fro=float(d.norm() / ref.norm()),
                finite=bool(torch.isfinite(o).all()))


WS = None
PLAN_KW = {}


def workspace():
    global WS
    if WS is None:
        WS = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return WS


def build_variant(name, c):
    """Returns (run_fn, out_tensor, info) or raises."""
    import flashinfer
    info = {}
    out = torch.empty(c.total_q, c.Hq, c.D, dtype=torch.bfloat16, device="cuda")
    if name.startswith("fa2") or name.startswith("cudnn") or name == "prims_fp8":
        backend = "fa2"
        if name.startswith("cudnn"):
            backend = "cudnn"
        if name == "prims_fp8":
            backend = "cute-dsl-prims"
        qdt, kvdt, odt = torch.bfloat16, torch.float8_e4m3fn, None
        q_in, kv_in = c.q, c.kv_fp8
        if name in ("fa2_fp8q", "prims_fp8", "cudnn_fp8q"):
            qdt, odt, q_in = torch.float8_e4m3fn, torch.bfloat16, c.q_fp8
        if name in ("fa2_bf16kv", "cudnn_bf16kv"):
            kvdt, kv_in = torch.bfloat16, c.kv_bf16
        wl = c.window_left
        if name == "prims_fp8":
            if c.window_left >= 0 and c.seqlen > c.window_left + 1:
                raise RuntimeError("prims has no sliding window; seq exceeds window")
            if c.D > 256:
                raise RuntimeError("prims supports head_dim<=256 only")
            wl = -1
        ws = workspace()
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend=backend)
        t0 = time.perf_counter()
        w.plan(c.qo_indptr, c.kv_indptr, c.kv_indices, c.last_page_len, c.Hq, c.Hkv,
               c.D, PAGE, causal=True, sm_scale=c.sm_scale, window_left=wl,
               q_data_type=qdt, kv_data_type=kvdt, o_data_type=odt, **PLAN_KW)
        torch.cuda.synchronize()
        info["plan_ms_first"] = (time.perf_counter() - t0) * 1e3
        # second plan (cached module) = the per-forward host cost
        t0 = time.perf_counter()
        w.plan(c.qo_indptr, c.kv_indptr, c.kv_indices, c.last_page_len, c.Hq, c.Hkv,
               c.D, PAGE, causal=True, sm_scale=c.sm_scale, window_left=wl,
               q_data_type=qdt, kv_data_type=kvdt, o_data_type=odt, **PLAN_KW)
        torch.cuda.synchronize()
        info["plan_ms"] = (time.perf_counter() - t0) * 1e3
        info["backend"] = getattr(w, "_backend", backend)
        pi = getattr(w, "_plan_info", None)
        if pi is not None:
            try:
                info["plan_info"] = [int(x) for x in pi]
            except Exception:
                info["plan_info"] = str(pi)[:300]

        def run():
            w.run(q_in, kv_in, out=out)
        return run, out, info
    if name.startswith("fmhav2"):
        from flashinfer.prefill import trtllm_fmha_v2_prefill
        kv_in = c.kv_bf16 if name == "fmhav2_bf16kv" else c.kv_fp8
        ws = workspace()
        cum_q = c.qo_indptr
        cum_kv = torch.tensor([0] + list(torch.cumsum(c.seq_lens, 0).tolist()),
                              dtype=torch.int32, device="cuda")
        mask_mode = "sliding_window" if c.window_left >= 0 else "causal"

        def run():
            trtllm_fmha_v2_prefill((c.q, kv_in), "Q_PAGED_KV_HND", ws, c.seq_lens,
                                   c.seqlen, c.seqlen, c.sm_scale, 1.0, c.batch,
                                   cum_q, cum_kv, block_tables=c.block_tables,
                                   out=out, mask_mode=mask_mode,
                                   window_left=c.window_left)
        return run, out, info
    if name == "b12x_fp8kv":
        import flashinfer.experimental.b12x  # noqa: F401  (aliases b12x)
        from b12x.attention import paged
        k_nhd = c.kv_fp8[:, 0].transpose(1, 2)  # [pages, P, H, D] strided view
        v_nhd = c.kv_fp8[:, 1].transpose(1, 2)
        if os.environ.get("B12X_CONTIG", "0") == "1":
            k_nhd = k_nhd.contiguous()
            v_nhd = v_nhd.contiguous()
        info["kv_contiguous"] = bool(k_nhd.is_contiguous())
        ones = torch.ones(c.batch, c.Hkv, dtype=torch.float32, device="cuda")
        caps = paged.Caps(device=torch.device("cuda"), mode="extend", dtype=c.q.dtype,
                          kv_dtype=torch.float8_e4m3fn, num_q_heads=c.Hq,
                          num_kv_heads=c.Hkv, head_dim_qk=c.D, head_dim_vo=c.D,
                          page_size=PAGE, max_total_q=c.total_q, max_batch=c.batch,
                          max_page_table_width=c.block_tables.shape[1],
                          max_work_items=4096, max_partial_rows=65536,
                          num_cache_pages=c.kv_fp8.shape[0], use_cuda_graph=False)
        splan = paged.plan(caps)
        specs = splan.scratch_specs()
        scratch = torch.empty(specs[0].shape, dtype=specs[0].dtype, device="cuda")
        binding = paged.bind(splan, scratch=scratch, q=c.q, k_cache=k_nhd,
                             v_cache=v_nhd, output=out, page_table=c.block_tables,
                             cache_seqlens=c.seq_lens, cu_seqlens_q=c.qo_indptr,
                             window_left=c.window_left, active_total_q=c.total_q,
                             k_descale=ones, v_descale=ones)
        info["note"] = "b12x default softmax scale assumed 1/sqrt(d)"

        def run():
            paged.run(binding=binding)
        return run, out, info
    raise ValueError(name)


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    med = statistics.median(ts)
    p10 = ts[max(0, int(0.1 * n))]
    p90 = ts[min(n - 1, int(0.9 * n))]
    return dict(median_ms=med, p10_ms=p10, p90_ms=p90,
                noise=(p90 - p10) / med if med > 0 else 0.0)


def repeat_check(fn, out, warmup, iters):
    fn()
    torch.cuda.synchronize()
    first = out.clone()
    bad = 0
    for _ in range(warmup + iters - 1):
        fn()
        torch.cuda.synchronize()
        if not torch.equal(out, first):
            bad += 1
    return dict(calls=warmup + iters, mismatches=bad), first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variants", default="fa2,fa2_fp8q,fa2_bf16kv")
    ap.add_argument("--layers", default="global,sliding")
    ap.add_argument("--cases", default="1x300,16x300,27x300,1x1000,8x1000,1x2000,"
                    "4x2000,16x512,1x4000,2x4000,1x8000")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--only-run", default="", help="variant:case to loop for ncu")
    ap.add_argument("--page", type=int, default=32)
    a = ap.parse_args()
    global PAGE
    PAGE = a.page
    log("PAGE", PAGE)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dims = load_dims(a.ckpt)
    log("DIMS", json.dumps(dims))
    p = torch.cuda.get_device_properties(0)
    log("GPU", p.name, "SMs", p.multi_processor_count, "cc", p.major, p.minor)
    try:
        import flashinfer
        log("flashinfer", getattr(flashinfer, "__version__", "?"), flashinfer.__file__)
    except Exception as ex:
        log("flashinfer import failed", ex)
    variants = a.variants.split(",")
    results = []
    all_ok = True
    for layer in a.layers.split(","):
        for cs in a.cases.split(","):
            b, s = (int(x) for x in cs.split("x"))
            c = Case(layer, b, s, dims, seed=1234)
            ref = None
            if not a.no_ref:
                ref = c.reference()
            main_out = None
            for vn in variants:
                rec = dict(case=c.label(), layer=layer, batch=b, seqlen=s, variant=vn,
                           Hq=c.Hq, Hkv=c.Hkv, D=c.D, window_left=c.window_left,
                           gflop=c.flops / 1e9)
                base, _, spec = vn.partition("@")
                envmap = {"cq": "FI_PF_CTA_TILE_Q", "kv": "FI_PF_NUM_MMA_KV",
                          "ctas": "FI_PF_CTAS_PER_SM"}
                for k in envmap.values():
                    os.environ.pop(k, None)
                PLAN_KW.clear()
                if spec:
                    for kv in spec.split("+"):
                        k, v = kv.split("=")
                        if k == "fs":
                            PLAN_KW["fixed_split_size"] = int(v)
                        elif k == "nosplit":
                            PLAN_KW["disable_split_kv"] = bool(int(v))
                        else:
                            os.environ[envmap[k]] = v
                try:
                    cq = os.environ.get("FI_PF_CTA_TILE_Q")
                    if cq is not None:
                        # instantiated set (batch_prefill_paged_kernel_inst.jinja); others
                        # abort the process with an undefined-symbol error.
                        allowed = (16, 32) if c.D >= 512 else (16, 64, 128)
                        if int(cq) not in allowed:
                            raise RuntimeError(f"CTA_TILE_Q={cq} not instantiated for d{c.D}")
                    fn, out, info = build_variant(base, c)
                    rec.update(info)
                    if a.only_run:
                        for _ in range(1):
                            fn()
                        torch.cuda.synchronize()
                        log("RAN", vn, c.label())
                        log("RESULT", json.dumps(rec))
                        continue
                    rc, first = repeat_check(fn, out, a.warmup, a.iters)
                    rec["repeat"] = rc
                    if rc["mismatches"]:
                        all_ok = False
                    t = time_fn(fn, a.warmup, a.iters)
                    rec.update(t)
                    rec["tflops"] = c.flops / (t["median_ms"] * 1e-3) / 1e12
                    if ref is not None:
                        rec["err_vs_fp32ref"] = err_stats(first, ref)
                    if vn == "fa2" or (main_out is None and vn == variants[0]):
                        main_out = first
                    elif main_out is not None:
                        rec["maxabs_vs_main"] = float(
                            (first.float() - main_out.float()).abs().max())
                    if vn == "fa2_fp8q" and ref is not None:
                        # also vs a reference that uses the fp8-rounded Q
                        rq = c.reference(q_src=c.q_fp8)
                        rec["err_vs_fp32ref_fp8q"] = err_stats(first, rq)
                        del rq
                except Exception as ex:
                    rec["error"] = f"{type(ex).__name__}: {str(ex)[:600]}"
                    tb = traceback.format_exc().splitlines()[-6:]
                    rec["tb"] = tb
                log("RESULT", json.dumps(rec))
                results.append(rec)
            del ref, main_out, c
            torch.cuda.empty_cache()
    if a.only_run:
        print(json.dumps({"passed": True, "mode": "only_run"}))
        return
    # summary
    summ = {}
    for r in results:
        if "median_ms" in r:
            summ.setdefault(r["variant"], {})[r["case"]] = dict(
                ms=round(r["median_ms"], 4), tflops=round(r["tflops"], 1),
                noise=round(r["noise"], 3),
                rel_fro=r.get("err_vs_fp32ref", {}).get("rel_fro"))
        else:
            summ.setdefault(r["variant"], {})[r["case"]] = r.get("error", "?")[:160]
    print(json.dumps({"passed": all_ok, "summary": summ}))


if __name__ == "__main__":
    main()
