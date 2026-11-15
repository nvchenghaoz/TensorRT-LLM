"""Idea 39: Gemma4 global-layer (head_dim 512) FA2 paged prefill, main vs the 64-row kernel.

Modes
  --arm tree     run FlashInfer as the job's tree has it (baseline arm, or the candidate
                 once the kernel is committed into the FlashInfer tree).
  --arm overlay  build the experiments/ascent/idea39 overlay (fi_overlay.py) into a fresh
                 FLASHINFER_WORKSPACE_BASE and run it (stage-1 candidate).
  --driver       run 'tree' then 'overlay' in two subprocesses (fresh workspace base for the
                 overlay arm), then compare outputs.

Each arm, per case: plan exactly as TRT-LLM's FLASHINFER backend does (HND paged FP8 E4M3 KV,
page 32, BF16 Q, causal), an untimed repeated-execution sequence (warmup + iters calls, every
output bitwise vs the first), CUDA-event timing (median / p10 / p90), error vs an FP32
reference (softmax(QK^T*s + causal) V on the dequantized KV). Dims come from
<checkpoint>/config.json. Last stdout line: evidence JSON.
"""
import argparse
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
PAGE = 32
TILE64_FLAG = "FLASHINFER_FA2_D512_TILE64"
CASE_LABELS = {"1x300": "isl300@1", "16x300": "isl300@16", "27x300": "isl300@64",
               "1x1000": "isl1k@1", "8x1000": "isl1k@16", "1x2000": "isl2k@1",
               "4x2000": "isl2k@4", "16x512": "8192", "1x4000": "isl4k@1",
               "2x4000": "isl4k@4", "1x8000": "isl8k@1"}
CASES = "1x300,16x300,27x300,1x1000,8x1000,1x2000,4x2000,16x512,1x4000,2x4000,1x8000"
# ragged / prefix / split-KV numerics cases: name -> (q_lens, kv_lens, plan kwargs)
EXTRA = {
    "ragged5": ([37, 300, 1000, 513, 2048], None, {}),
    "ragged2": ([8000, 100], None, {}),
    "prefix2": ([500, 256], [1500, 700], {}),
    "split1x2000": ([2000], None, {"fixed_split_size": 16}),
    "splitprefix": ([700, 129], [2600, 1000], {"fixed_split_size": 8}),
}


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
    d = dict(num_heads=nq, global_kv_heads=gkv, global_head_dim=ghd,
             sliding_head_dim=hd, sliding_kv_heads=nkv,
             attention_k_eq_v=tc.get("attention_k_eq_v"))
    qpath = os.path.join(ckpt, "hf_quant_config.json")
    if os.path.exists(qpath):
        q = json.load(open(qpath)).get("quantization", {})
        d["kv_cache_quant_algo"] = q.get("kv_cache_quant_algo")
    return d


class Case:
    def __init__(self, label, q_lens, kv_lens, dims, plan_kw, seed=1234):
        import torch
        self.label = label
        self.plan_kw = dict(plan_kw)
        self.Hq = dims["num_heads"]
        self.Hkv = dims["global_kv_heads"]
        self.D = dims["global_head_dim"]
        self.sm_scale = 1.0 / math.sqrt(self.D)
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        self.q_lens = list(q_lens)
        self.kv_lens = list(kv_lens) if kv_lens is not None else list(q_lens)
        self.batch = len(self.q_lens)
        npages = [(L + PAGE - 1) // PAGE for L in self.kv_lens]
        total_pages = sum(npages)
        self.total_q = sum(self.q_lens)
        qo, kvp = [0], [0]
        for L, n in zip(self.q_lens, npages):
            qo.append(qo[-1] + L)
            kvp.append(kvp[-1] + n)
        self.qo_indptr = torch.tensor(qo, dtype=torch.int32, device=dev)
        self.kv_indptr = torch.tensor(kvp, dtype=torch.int32, device=dev)
        self.kv_indices = torch.randperm(total_pages, generator=g, device=dev).to(torch.int32)
        self.last_page_len = torch.tensor([(L - 1) % PAGE + 1 for L in self.kv_lens],
                                          dtype=torch.int32, device=dev)
        pool = torch.randn(total_pages, 2, self.Hkv, PAGE, self.D, generator=g, device=dev,
                           dtype=torch.float32)
        self.kv_fp8 = pool.to(torch.float8_e4m3fn)
        del pool
        self.q = torch.randn(self.total_q, self.Hq, self.D, generator=g, device=dev,
                             dtype=torch.float32).to(torch.bfloat16)
        pairs = 0
        for Lq, Lk in zip(self.q_lens, self.kv_lens):
            pairs += sum(min(Lk, Lk - Lq + i + 1) for i in range(Lq))
        self.flops = 4.0 * self.Hq * self.D * pairs

    def gather_kv(self, r):
        s, e = int(self.kv_indptr[r]), int(self.kv_indptr[r + 1])
        idx = self.kv_indices[s:e].long()
        pages = self.kv_fp8[idx].float()
        L = self.kv_lens[r]
        k = pages[:, 0].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:L]
        v = pages[:, 1].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:L]
        return k, v

    def reference(self):
        import torch
        out = torch.empty(self.total_q, self.Hq, self.D, dtype=torch.float32, device="cuda")
        grp = self.Hq // self.Hkv
        with torch.no_grad():
            for r in range(self.batch):
                s, e = int(self.qo_indptr[r]), int(self.qo_indptr[r + 1])
                Lq, Lk = e - s, self.kv_lens[r]
                k, v = self.gather_kv(r)
                i = torch.arange(Lq, device="cuda").view(-1, 1) + (Lk - Lq)
                j = torch.arange(Lk, device="cuda").view(1, -1)
                mask = j <= i
                for h in range(self.Hq):
                    sc = (self.q[s:e, h].float() @ k[:, h // grp].T) * self.sm_scale
                    sc.masked_fill_(~mask, float("-inf"))
                    out[s:e, h] = torch.softmax(sc, dim=-1) @ v[:, h // grp]
        return out


def err_stats(out, ref):
    o = out.float()
    d = o - ref
    return dict(max_abs=float(d.abs().max()), rel_fro=float(d.norm() / ref.norm()),
                finite=bool(o.isfinite().all()))


def time_fn(fn, warmup, iters):
    import torch
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
    return dict(median_ms=med, p10_ms=p10, p90_ms=p90, noise=(p90 - p10) / med)


def build_cases(a, dims):
    cases = []
    for cs in [c for c in a.cases.split(",") if c]:
        b, s = (int(x) for x in cs.split("x"))
        cases.append((cs, [s] * b, None, {}))
    for name in [c for c in a.extra.split(",") if c]:
        ql, kl, kw = EXTRA[name]
        cases.append((name, ql, kl, kw))
    return cases


def run_arm(a):
    if a.arm == "overlay":
        if not os.environ.get("FLASHINFER_WORKSPACE_BASE", "").count("idea39_ws_"):
            # every overlay arm needs its own JIT cache: all arms share one module URI
            os.environ["FLASHINFER_WORKSPACE_BASE"] = tempfile.mkdtemp(prefix="idea39_ws_")
        sys.path.insert(0, str(HERE))
        import fi_overlay
        ov = pathlib.Path(a.overlay_dir or tempfile.mkdtemp(prefix="idea39_ov_"))
        fi_overlay.build(ov)
        fi_overlay.apply(ov)
    import torch
    import flashinfer
    from flashinfer.jit import env as jit_env
    log("ARM", a.arm, "flashinfer", getattr(flashinfer, "__version__", "?"), flashinfer.__file__)
    log("JIT_DIR", jit_env.FLASHINFER_JIT_DIR, "INCLUDE", jit_env.FLASHINFER_INCLUDE_DIR)
    torch.backends.cuda.matmul.allow_tf32 = False
    dims = load_dims(a.ckpt)
    log("DIMS", json.dumps(dims))
    p = torch.cuda.get_device_properties(0)
    log("GPU", p.name, "SMs", p.multi_processor_count)
    ws = torch.zeros(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    save_dir = pathlib.Path(a.save_dir) if a.save_dir else None
    results = []
    all_ok = True
    for label, ql, kl, kw in build_cases(a, dims):
        c = Case(label, ql, kl, dims, kw)
        out = torch.empty(c.total_q, c.Hq, c.D, dtype=torch.bfloat16, device="cuda")
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")

        def plan():
            w.plan(c.qo_indptr, c.kv_indptr, c.kv_indices, c.last_page_len, c.Hq, c.Hkv, c.D,
                   PAGE, causal=True, sm_scale=c.sm_scale, window_left=-1,
                   q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, **c.plan_kw)
        t0 = time.perf_counter()
        plan()
        torch.cuda.synchronize()
        first_plan_s = time.perf_counter() - t0
        pi = [int(x) for x in w._plan_info]

        lse = None
        if a.lse:
            lse = torch.empty(c.total_q, c.Hq, dtype=torch.float32, device="cuda")

        def run():
            if lse is None:
                w.run(c.q, c.kv_fp8, out=out)
            else:
                w.run(c.q, c.kv_fp8, out=out, lse=lse, return_lse=True)
        rec = dict(case=label, q_lens=ql if len(ql) <= 8 else f"{len(ql)}x{ql[0]}",
                   cta_tile_q=pi[3], split_kv=bool(pi[14]), padded_batch=pi[0],
                   first_plan_s=round(first_plan_s, 2), gflop=c.flops / 1e9)
        ref32 = None
        if a.self_compare:
            # Same build, 32-row path (FLASHINFER_FA2_D512_TILE64=0 is read on every plan):
            # main's kernel. Then re-plan with the default (64-row where eligible).
            os.environ[TILE64_FLAG] = "0"
            plan()
            pi32 = [int(x) for x in w._plan_info]
            run()
            torch.cuda.synchronize()
            ref32 = dict(o=out.clone(), lse=None if lse is None else lse.clone(),
                         cta_tile_q=pi32[3], split_kv=bool(pi32[14]))
            if not a.no_time:
                t32 = time_fn(run, a.warmup, a.iters)
                ref32["timing"] = t32
            os.environ.pop(TILE64_FLAG, None)
            plan()
            pi = [int(x) for x in w._plan_info]
            rec.update(cta_tile_q=pi[3], split_kv=bool(pi[14]), padded_batch=pi[0])
        # untimed repeated-execution sequence: same call count as the timing run
        run()
        torch.cuda.synchronize()
        first = out.clone()
        mism = 0
        for _ in range(a.warmup + a.iters - 1):
            run()
            torch.cuda.synchronize()
            if not torch.equal(out, first):
                mism += 1
        rec["repeat"] = dict(calls=a.warmup + a.iters, mismatches=mism)
        all_ok &= mism == 0
        if not a.no_time:
            t = time_fn(run, a.warmup, a.iters)
            rec.update(t)
            rec["tflops"] = c.flops / (t["median_ms"] * 1e-3) / 1e12
        if not a.no_ref:
            ref = c.reference()
            rec["err_vs_fp32"] = err_stats(first, ref)
            all_ok &= rec["err_vs_fp32"]["finite"]
            del ref
        if ref32 is not None:
            nbad = int((first.view(torch.int16) != ref32["o"].view(torch.int16)).sum())
            sc = dict(path32_tile=ref32["cta_tile_q"], path32_split=ref32["split_kv"],
                      byte_mismatch_elems=nbad, elems=first.numel())
            if lse is not None:
                # the repeated sequence left the last call's lse in place: deterministic
                sc["lse_byte_mismatch"] = int(
                    (lse.view(torch.int32) != ref32["lse"].view(torch.int32)).sum())
            sc["bitwise"] = nbad == 0 and sc.get("lse_byte_mismatch", 0) == 0
            if "timing" in ref32:
                sc["path32_ms"] = ref32["timing"]["median_ms"]
                sc["path32_p10"] = ref32["timing"]["p10_ms"]
                sc["path32_p90"] = ref32["timing"]["p90_ms"]
            rec["vs_path32"] = sc
            all_ok &= sc["bitwise"]
            del ref32
        if save_dir is not None:
            lse_first = lse.clone() if lse is not None else None
            if a.arm == "tree":
                torch.save({"o": first.cpu(), "lse": None if lse_first is None else lse_first.cpu()},
                           save_dir / f"{label}.pt")
            else:
                bp = save_dir / f"{label}.pt"
                if bp.exists():
                    saved = torch.load(bp)
                    base = saved["o"].cuda()
                    d = (first.float() - base.float()).abs()
                    # raw byte compare (torch.equal would hide -0.0 vs +0.0)
                    nbad = int((first.view(torch.int16) != base.view(torch.int16)).sum())
                    rec["vs_base"] = dict(max_abs=float(d.max()), byte_mismatch_elems=nbad,
                                          elems=first.numel(), bitwise=nbad == 0)
                    if lse_first is not None and saved.get("lse") is not None:
                        bl = saved["lse"].cuda()
                        nl = int((lse_first.view(torch.int32) != bl.view(torch.int32)).sum())
                        rec["vs_base"]["lse_byte_mismatch"] = nl
                        rec["vs_base"]["bitwise"] = rec["vs_base"]["bitwise"] and nl == 0
                    if nbad:
                        bad = (first.view(torch.int16) != base.view(torch.int16))
                        idx = bad.nonzero()[:8].tolist()
                        rec["vs_base"]["first_bad"] = idx
                    del base
        log("RESULT", json.dumps(rec))
        results.append(rec)
        del c, out, first, w
        torch.cuda.empty_cache()
    # Evidence keys: the served case each op shape is (max_num_tokens 8192 packs the
    # concurrent requests: isl300@64 -> 27x300, isl1k@16 -> 8x1000, isl2k@4 -> 4x2000,
    # isl4k@4 -> 2x4000); (16,512) has no served label and is keyed by its total tokens.
    batch = {CASE_LABELS[r["case"]]: {"kernel_ms": r.get("median_ms"),
                                       "noise_floor": r.get("noise")}
             for r in results if r["case"] in CASE_LABELS}
    exact = all(r.get("vs_path32", {}).get("bitwise", False) for r in results) if \
        a.self_compare else False
    final = {"batch": batch, "output": {"class": "exact" if exact else "reordered"},
             "passed": bool(all_ok), "arm": a.arm, "results": results}
    print(json.dumps(final), flush=True)


def driver(a):
    save = pathlib.Path(tempfile.mkdtemp(prefix="idea39_out_"))
    common = ["--ckpt", a.ckpt, "--cases", a.cases, "--extra", a.extra, "--warmup",
              str(a.warmup), "--iters", str(a.iters), "--save-dir", str(save)]
    if a.no_ref:
        common.append("--no-ref")
    if a.no_time:
        common.append("--no-time")
    if a.lse:
        common.append("--lse")
    finals = {}
    # --variants "name:-DFLAG=1 -DOTHER=2;name2:..." : one overlay arm per variant
    variants = []
    for spec in [v for v in a.variants.split(";") if v.strip()]:
        name, _, flags = spec.partition(":")
        variants.append((name.strip(), flags.strip()))
    if not variants:
        variants = [("overlay", "")]
    arms = ([] if a.only_overlay else [("tree", "tree", "")]) + \
        [(name, "overlay", flags) for name, flags in variants]
    for arm_name, arm, flags in arms:
        env = dict(os.environ)
        if arm == "overlay":
            env["FLASHINFER_WORKSPACE_BASE"] = tempfile.mkdtemp(prefix="idea39_ws_")
            # VERBOSE=1 alone also turns on --device-debug -O0 (jit/core.py); DEBUG=0 keeps -O3
            env["FLASHINFER_JIT_VERBOSE"] = "1"
            env["FLASHINFER_JIT_DEBUG"] = "0"
            env["FLASHINFER_EXTRA_CUDAFLAGS"] = ("-Xptxas=-v " + flags).strip()
        cmd = [sys.executable, __file__, "--arm", arm] + common
        arm = arm_name
        log("DRIVER run", arm, "ws", env.get("FLASHINFER_WORKSPACE_BASE", "<default>"))
        t0 = time.time()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        last = None
        want_regs = 0
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("{"):
                last = line
                continue
            if want_regs > 0:
                log(f"[{arm}]", line[:400])
                want_regs -= 1
                continue
            if "ptxas" in line and "D512Tile64" in line:
                log(f"[{arm}]", line[:400])
                want_regs = 3
                continue
            if line.startswith("[") and "nvcc" in line:
                continue
            if ("Building JIT" in line or "pragma message" in line or "idea39" in line
                    or line.startswith(("RESULT", "ARM", "JIT_DIR", "DIMS", "GPU", "[overlay]",
                                        "Traceback", "  File", "RuntimeError", "Error", "error"))
                    or ("error" in line.lower()[:300] and "warning" not in line.lower())):
                log(f"[{arm}]", line[:2000])
        rc = proc.wait()
        log("DRIVER", arm, "rc", rc, "secs", round(time.time() - t0, 1))
        if last is not None:
            finals[arm] = json.loads(last)
    table = []
    base = {r["case"]: r for r in finals["tree"]["results"]} if "tree" in finals else {}
    for vname, _ in variants:
        if vname not in finals:
            continue
        for r in finals[vname]["results"]:
            b = base.get(r["case"], {})
            row = dict(variant=vname, case=r["case"], base_tile=b.get("cta_tile_q"),
                       cand_tile=r["cta_tile_q"],
                       base_split=b.get("split_kv"), cand_split=r["split_kv"],
                       base_ms=b.get("median_ms"), base_p10=b.get("p10_ms"),
                       base_p90=b.get("p90_ms"), cand_ms=r.get("median_ms"),
                       cand_p10=r.get("p10_ms"), cand_p90=r.get("p90_ms"),
                       base_err=b.get("err_vs_fp32"), cand_err=r.get("err_vs_fp32"),
                       vs_base=r.get("vs_base"), cand_repeat=r["repeat"])
            if row["base_ms"] and row["cand_ms"]:
                row["speedup"] = row["base_ms"] / row["cand_ms"]
            # gate: byte-equal to main's kernel output (and LSE when checked), deterministic
            vb = r.get("vs_base") or {}
            ok = bool(vb.get("bitwise", False))
            ce = row["cand_err"]
            if ce:
                ok &= ce["finite"]
            ok &= r["repeat"]["mismatches"] == 0
            row["passed"] = bool(ok)
            table.append(row)
            log("ROW", json.dumps(row))
    summary = {"passed": all(r["passed"] for r in table) if table else False,
               "table": table,
               "arms_passed": {k: v.get("passed") for k, v in finals.items()}}
    print(json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", default="tree", choices=["tree", "overlay"])
    ap.add_argument("--driver", action="store_true")
    ap.add_argument("--only-overlay", action="store_true")
    ap.add_argument("--variants", default="")
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--extra", default="")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--lse", action="store_true", help="also return and byte-compare LSE")
    ap.add_argument("--self-compare", action="store_true",
                    help="byte-compare each case against the 32-row path of the same build")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--overlay-dir", default="")
    a = ap.parse_args()
    if a.driver:
        driver(a)
    else:
        run_arm(a)


if __name__ == "__main__":
    main()
