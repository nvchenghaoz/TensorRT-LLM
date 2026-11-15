"""Layer-59 (last global layer) attention on read rows only: op pair + check.

Branch 28 / idea 45. Synthetic tensors at the real global-layer shapes read
from <checkpoint>/config.json (no model load). Paged FP8 E4M3 KV cache, HND,
page size 32 (TRT-LLM default tokens_per_block), shuffled page ids, BF16 Q.

Arms (same script in both trees):
  main      : FlashInfer FA2 BatchPrefillWithPagedKVCacheWrapper planned as
              TRT-LLM's prefill_plan() does (host indptrs), run on all rows.
  read_rows : FlashInferAttention._prefill_forward_read_rows (candidate tree
              only) with a stand-in metadata object; main's plan is the
              wrapper it checks.

usage:
  bench.py time  <ckpt>   -> one JSON line {"batch": {...}, "output": {...}}
  bench.py check <ckpt>   -> correctness JSON {"passed": ...}
"""
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

PAGE = 32
WARMUP = 5
ITERS = 25
# (label, batch, seq_len): workload case labels for the packed batches the
# scheduler forms under max_num_tokens=8192.
CASES = [
    ("isl300@1", 1, 300),
    ("isl300@4", 4, 300),
    ("isl300@16", 16, 300),
    ("isl300@64", 27, 300),
    ("isl1k@1", 1, 1000),
    ("isl1k@4", 4, 1000),
    ("isl1k@16", 8, 1000),
    ("isl2k@1", 1, 2000),
    ("isl2k@4", 4, 2000),
    ("isl4k@1", 1, 4000),
    ("isl4k@4", 2, 4000),
    ("isl8k@1", 1, 8000),
]


def log(*a):
    print(*a, flush=True)


def read_shapes(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    t = cfg.get("text_config", cfg)
    heads = t["num_attention_heads"]
    kv_heads = t.get("num_global_key_value_heads") or t["num_key_value_heads"]
    head_dim = t.get("global_head_dim") or t["head_dim"]
    layer_types = t.get("layer_types")
    log(f"config: num_attention_heads={heads} num_key_value_heads={t['num_key_value_heads']} "
        f"num_global_key_value_heads={t.get('num_global_key_value_heads')} head_dim={t['head_dim']} "
        f"global_head_dim={t.get('global_head_dim')} hidden={t['hidden_size']} "
        f"layers={t['num_hidden_layers']} sliding_window={t.get('sliding_window')} "
        f"last_layer_type={layer_types[-1] if layer_types else None} "
        f"attention_k_eq_v={t.get('attention_k_eq_v')}")
    hq = json.load(open(os.path.join(ckpt, "hf_quant_config.json")))
    log("hf_quant_config.quantization:", json.dumps(hq.get("quantization", hq))[:400])
    assert layer_types is None or layer_types[-1] == "full_attention"
    return heads, kv_heads, head_dim


class Case:
    """One forward's context batch: KV pool (sentinel elsewhere), page table,
    host plan inputs as FlashInferAttentionMetadata.prepare() builds them."""

    def __init__(self, fi_mod, heads, kv_heads, head_dim, batch, seq_len, seed, workspace):
        import flashinfer
        self.flashinfer = flashinfer
        self.fi_mod = fi_mod
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.batch, self.seq_len = batch, seq_len
        g = torch.Generator(device="cuda").manual_seed(seed)
        gc = np.random.default_rng(seed)
        seq_lens = np.full(batch, seq_len, dtype=np.int64)
        self.seq_lens = seq_lens
        npages = (seq_lens + PAGE - 1) // PAGE
        total_pages = int(npages.sum())
        self.pool_pages = total_pages + 37
        # Sentinel-filled FP8 pool: 0x3C bytes everywhere before the append.
        self.kv = torch.full((self.pool_pages, 2, kv_heads, PAGE, head_dim), 0x3C,
                             dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
        perm = gc.permutation(self.pool_pages)[:total_pages].astype(np.int32)
        self.page_ids = torch.from_numpy(perm).cuda()
        self.num_context_blocks = total_pages
        qo = np.concatenate([[0], np.cumsum(seq_lens)])
        kv_indptr = np.concatenate([[0], np.cumsum(npages)])
        last = seq_lens - (npages - 1) * PAGE
        pin = lambda a: torch.from_numpy(a.astype(np.int32)).pin_memory()
        self.host = (batch, pin(qo), pin(kv_indptr), pin(last))
        self.M = int(qo[-1])
        # q/k/v as after q/k/v RMSNorm + RoPE (unit RMS per head).
        self.q = torch.randn(self.M, heads, head_dim, device="cuda", generator=g).to(torch.bfloat16)
        k = torch.randn(self.M, kv_heads, head_dim, device="cuda", generator=g).to(torch.bfloat16)
        v = torch.randn(self.M, kv_heads, head_dim, device="cuda", generator=g).to(torch.bfloat16)
        self.k8, self.v8 = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
        # Append every row's K/V as main does (TRT-LLM's append helper).
        qo_d = torch.from_numpy(qo.astype(np.int32)).cuda()
        kvi_d = torch.from_numpy(kv_indptr.astype(np.int32)).cuda()
        last_d = torch.from_numpy(last.astype(np.int32)).cuda()
        seq_d = flashinfer.get_seq_lens(kvi_d, last_d, PAGE)
        bi, pos = flashinfer.get_batch_indices_positions(qo_d, seq_d, self.M)
        fi_mod._append_paged_kv_cache(append_key=self.k8, append_value=self.v8, batch_indices=bi,
                                      positions=pos, paged_kv_cache=self.kv, kv_indices=self.page_ids,
                                      kv_indptr=kvi_d, kv_last_page_len=last_d, kv_layout="HND")
        torch.cuda.synchronize()
        # Gemma4: q_scaling = 1/sqrt(head_dim) -> sm_scale 1.0
        self.sm_scale = 1.0 / (math.sqrt(head_dim) * (1.0 / math.sqrt(head_dim)))
        self.workspace = workspace
        self.main_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "HND", backend="fa2")
        self.plan_main()

    def plan_main(self, fixed_split_size=None):
        _, qo, kvi, last = self.host
        kw = {}
        if fixed_split_size is not None:
            kw["fixed_split_size"] = fixed_split_size
        self.main_wrapper.plan(qo, kvi, self.page_ids[:self.num_context_blocks], last, self.heads,
                               self.kv_heads, self.head_dim, PAGE, causal=True, sm_scale=self.sm_scale,
                               window_left=-1, q_data_type=torch.bfloat16,
                               kv_data_type=torch.float8_e4m3fn, o_data_type=None, custom_mask=None, **kw)

    def plan_params(self, window_left=-1):
        from tensorrt_llm.functional import AttentionMaskType
        return self.fi_mod.PlanParams(num_heads=self.heads, num_kv_heads=self.kv_heads,
                                      head_dim=self.head_dim, q_dtype=torch.bfloat16,
                                      kv_dtype=torch.float8_e4m3fn, sm_scale=self.sm_scale,
                                      window_left=window_left, attention_mask_type=AttentionMaskType.causal,
                                      attention_mask_data=None)

    def metadata(self, num_generations=0):
        m = types.SimpleNamespace()
        m.last_layer_read_rows = True
        m._host_prefill_plan_inputs = self.host
        m.num_contexts = self.batch
        m.num_generations = num_generations
        m.is_cuda_graph = False
        m.num_ctx_tokens = self.M
        m.workspace_buffer = self.workspace
        m.kv_layout = "HND"
        m._paged_kv_indices = self.page_ids
        m.num_context_blocks = self.num_context_blocks
        m.page_size = PAGE
        m.get_prefill_wrapper = lambda pp: self.main_wrapper
        return m

    def run_main(self, out):
        self.main_wrapper.run(self.q, self.kv, out=out.view(-1, self.heads, self.head_dim))

    def read_rows_fn(self):
        cls = self.fi_mod.FlashInferAttention
        if not hasattr(cls, "_prefill_forward_read_rows"):
            return None
        me = types.SimpleNamespace(flashinfer_backend="fa2")
        meta = self.metadata()
        pp = self.plan_params()

        def fn(out):
            ok = cls._prefill_forward_read_rows(me, meta, pp, self.q, self.kv, out)
            assert ok, "read-rows path fell back"
            return meta
        return fn

    def last_rows(self):
        return torch.from_numpy(np.cumsum(self.seq_lens) - 1).cuda()

    def fp32_ref_last_rows(self):
        """Naive FP32 softmax(q k^T * scale) v for each request's last row on the
        FP8-dequantized K/V."""
        outs = []
        qo = self.host[1].numpy()
        group = self.heads // self.kv_heads
        for b in range(self.batch):
            s, e = int(qo[b]), int(qo[b + 1])
            kk = self.k8[s:e].float()  # [L, kvh, d]
            vv = self.v8[s:e].float()
            qq = self.q[e - 1].float().view(self.kv_heads, group, self.head_dim)
            logits = torch.einsum("hgd,lhd->hgl", qq, kk) * self.sm_scale
            p = torch.softmax(logits, dim=-1)
            o = torch.einsum("hgl,lhd->hgd", p, vv).reshape(self.heads * self.head_dim)
            outs.append(o)
        return torch.stack(outs)


def events_time(fn, n, preroll):
    ts = []
    for _ in range(n):
        if preroll:
            torch.cuda._sleep(2_000_000)  # keep the GPU busy while the host enqueues
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        ts.append((s, e))
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in ts]


def stats(v):
    v = np.asarray(v)
    med = float(np.median(v))
    p10, p90 = float(np.percentile(v, 10)), float(np.percentile(v, 90))
    return med, p10, p90


def mode_time(ckpt):
    from tensorrt_llm._torch.attention.backends import flashinfer as fi_mod
    heads, kv_heads, head_dim = read_shapes(ckpt)
    ws = torch.empty(320 * 1024 * 1024 * 188 // 148, dtype=torch.uint8, device="cuda")
    candidate = hasattr(fi_mod.FlashInferAttention, "_prefill_forward_read_rows")
    log(f"arm={'read_rows' if candidate else 'main'} gpu={torch.cuda.get_device_name()}")
    batch = {}
    extra = {}
    for label, b, L in CASES:
        c = Case(fi_mod, heads, kv_heads, head_dim, b, L, seed=1000 + b * 7 + L, workspace=ws)
        out = torch.empty(c.M, heads * head_dim, dtype=torch.bfloat16, device="cuda")
        if candidate:
            rr = c.read_rows_fn()
            fn = lambda: rr(out)
        else:
            fn = lambda: c.run_main(out)
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        gpu = events_time(fn, ITERS, preroll=True)
        span = events_time(fn, ITERS, preroll=False)
        torch.cuda.synchronize()
        hs = []
        for _ in range(ITERS):
            torch.cuda._sleep(2_000_000)
            t0 = time.perf_counter()
            fn()
            hs.append((time.perf_counter() - t0) * 1e3)
        torch.cuda.synchronize()
        med, p10, p90 = stats(gpu)
        smed, sp10, sp90 = stats(span)
        hmed, _, _ = stats(hs)
        batch[label] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med, "p10": p10, "p90": p90,
                        "tokens": c.M}
        extra[label] = {"span_ms_no_preroll": smed, "span_p10": sp10, "span_p90": sp90,
                        "host_enqueue_ms": hmed}
        log(f"{label:10s} B={b:2d} L={L:5d} M={c.M:5d} gpu {med:8.4f} ms [p10 {p10:.4f} p90 {p90:.4f}] "
            f"span {smed:8.4f} host {hmed:.4f} ms")
        del c, out
        torch.cuda.empty_cache()
    print(json.dumps({"batch": batch, "output": {"class": "exact"},
                      "arm": "read_rows" if candidate else "main", "extra": extra}))


def mode_check(ckpt):
    from tensorrt_llm._torch.attention.backends import flashinfer as fi_mod
    heads, kv_heads, head_dim = read_shapes(ckpt)
    ws = torch.empty(320 * 1024 * 1024 * 188 // 148, dtype=torch.uint8, device="cuda")
    assert hasattr(fi_mod.FlashInferAttention, "_prefill_forward_read_rows")
    results = {}
    ok_all = True
    built = {}

    def get_case(label, b, L):
        if label not in built:
            built[label] = Case(fi_mod, heads, kv_heads, head_dim, b, L, seed=1000 + b * 7 + L,
                                workspace=ws)
        return built[label]

    order = list(CASES)
    # A, B, A permutation of the reused workspace / sub-plan wrapper.
    order += [CASES[-1], CASES[2], CASES[-1]]
    for label, b, L in order:
        c = get_case(label, b, L)
        H = heads * head_dim
        q_before = c.q.clone()
        kv_before = c.kv.view(torch.uint8).clone()
        ref_main = torch.empty(c.M, H, dtype=torch.bfloat16, device="cuda")
        c.run_main(ref_main)
        torch.cuda.synchronize()
        rows = c.last_rows()
        rr = c.read_rows_fn()
        out = torch.full((c.M, H), float("nan"), dtype=torch.bfloat16, device="cuda")
        n_calls = WARMUP + 2 * ITERS  # same calls as the timing run (warmup + 2 timed loops)
        first = None
        rep_equal = True
        meta = None
        for i in range(n_calls):
            meta = rr(out)
            if i == 0:
                first = out.clone()
            elif not torch.equal(out.view(torch.int16), first.view(torch.int16)):
                rep_equal = False
        torch.cuda.synchronize()
        read_bitwise = torch.equal(out[rows].view(torch.int16), ref_main[rows].view(torch.int16))
        mask = torch.ones(c.M, dtype=torch.bool, device="cuda")
        mask[rows] = False
        nonread_zero = bool((out[mask].view(torch.int16) == 0).all().item())
        ref32 = c.fp32_ref_last_rows()
        err_main = (ref_main[rows].float() - ref32)
        err_cand = (out[rows].float() - ref32)
        fro = lambda e: float(e.norm() / ref32.norm())
        q_same = torch.equal(q_before.view(torch.int16), c.q.view(torch.int16))
        kv_same = torch.equal(kv_before, c.kv.view(torch.uint8))
        used = torch.zeros(c.pool_pages, dtype=torch.bool, device="cuda")
        used[c.page_ids.long()] = True
        sentinel_ok = bool((c.kv.view(torch.uint8)[~used] == 0x3C).all().item())
        infos = meta._read_rows_plan_infos
        finite = bool(torch.isfinite(out[rows].float()).all().item())
        case_ok = (read_bitwise and nonread_zero and rep_equal and q_same and kv_same and sentinel_ok
                   and finite and infos["main"][0] == 32 and infos["main"][1] == 0
                   and infos["sub"][0] == 32 and infos["sub"][1] == 0)
        ok_all &= case_ok
        r = {"tokens": c.M, "read_rows_bitwise_vs_main": read_bitwise, "nonread_rows_zero": nonread_zero,
             "repeated_calls_equal": rep_equal, "calls": n_calls, "q_unchanged": q_same,
             "kv_pool_unchanged": kv_same, "sentinel_intact": sentinel_ok, "finite": finite,
             "main_plan(cta_tile_q,split_kv,padded_batch)": infos["main"],
             "sub_plan(cta_tile_q,split_kv,padded_batch)": infos["sub"], "sub_rows": infos["rows"],
             "maxabs_main_vs_fp32": float(err_main.abs().max()), "maxabs_cand_vs_fp32": float(err_cand.abs().max()),
             "relfro_main_vs_fp32": fro(err_main), "relfro_cand_vs_fp32": fro(err_cand),
             "maxabs_cand_vs_main": float((out[rows].float() - ref_main[rows].float()).abs().max()),
             "passed": case_ok}
        key = label if label not in results else label + "_again"
        results[key] = r
        log(key, json.dumps(r))
        del ref_main, out, kv_before, q_before

    # Guard-off cases: the method must return False before touching the output.
    guards = {}
    c = get_case("isl8k@1", 1, 8000)
    me = types.SimpleNamespace(flashinfer_backend="fa2")
    pp = c.plan_params()
    H = heads * head_dim
    cls = fi_mod.FlashInferAttention

    def untouched_false(case, meta, ppp, me_=me):
        o = torch.full((case.M, H), 7.0, dtype=torch.bfloat16, device="cuda")
        r = cls._prefill_forward_read_rows(me_, meta, ppp, case.q, case.kv, o)
        torch.cuda.synchronize()
        return (r is False) and bool((o == 7.0).all().item())

    saved = fi_mod._LAST_LAYER_ATTN_ROWS
    fi_mod._LAST_LAYER_ATTN_ROWS = False
    guards["kill_switch"] = untouched_false(c, c.metadata(), pp)
    fi_mod._LAST_LAYER_ATTN_ROWS = saved
    m = c.metadata()
    m.last_layer_read_rows = False
    guards["flag_off_non_last_layer"] = untouched_false(c, m, pp)
    guards["mixed_step_num_generations_1"] = untouched_false(c, c.metadata(num_generations=1), pp)
    m = c.metadata()
    m.is_cuda_graph = True
    guards["cuda_graph"] = untouched_false(c, m, pp)
    guards["trtllm_gen_backend"] = untouched_false(c, c.metadata(), pp,
                                                   me_=types.SimpleNamespace(flashinfer_backend="trtllm-gen"))
    guards["sliding_window"] = untouched_false(c, c.metadata(), c.plan_params(window_left=1023))
    # split main plan (1x1000, forced 256-token KV chunks; the workspace holds it)
    c1 = get_case("isl1k@1", 1, 1000)
    c1.plan_main(fixed_split_size=256 // PAGE)
    split_info = (int(c1.main_wrapper._plan_info[3]), int(c1.main_wrapper._plan_info[14]))
    log("forced-split main plan (cta_tile_q, split_kv):", split_info)
    guards["split_main_plan"] = split_info[1] == 1 and untouched_false(c1, c1.metadata(), c1.plan_params())
    c1.plan_main()
    # main plan with cta_tile_q 16: short requests (packed len <= 32)
    cs = Case(fi_mod, heads, kv_heads, head_dim, 3, 4, seed=5, workspace=ws)
    cs_info = (int(cs.main_wrapper._plan_info[3]), int(cs.main_wrapper._plan_info[14]))
    log("short-request main plan (cta_tile_q, split_kv):", cs_info)
    guards["main_cta_tile_q_16"] = cs_info[0] == 16 and untouched_false(cs, cs.metadata(), cs.plan_params())
    log("guards:", json.dumps(guards))
    ok_all &= all(guards.values())
    log("READ_ROWS_STATS", fi_mod.READ_ROWS_STATS)
    print(json.dumps({"passed": bool(ok_all),
                      "compared": "layer-59 read-rows attention vs main's full-row FA2 run (bitwise on "
                                  "rows cumsum(seq_lens)-1, zeros elsewhere), FP32 reference on FP8-dequantized "
                                  "KV, repeated calls, KV pool/sentinel, A-B-A plan reuse, guard-off cases",
                      "cases": results, "guards": guards}))


if __name__ == "__main__":
    {"time": mode_time, "check": mode_check}[sys.argv[1]](sys.argv[2])
