"""Idea 65: hardware cvt.rn.bf16x2.e4m3x2 in vec_cast<nv_bfloat16, __nv_fp8_e4m3> (vec_dtypes.cuh),
stacked on branch 16's v2m repack remap (prefill.cuh repack_fp8_tile_to_bf16).

Gemma4 FA2 paged prefill as TRT-LLM's FLASHINFER backend runs it: HND paged FP8 E4M3 KV, page 32,
BF16 Q/O, causal; sliding layers (d256, window_left = sliding_window - 1) and global layers (d512,
64-row Tile64 kernel at main's planner defaults; FLASHINFER_FA2_D512_TILE64 is left unset = default).
Dims come from <checkpoint>/config.json.

Modes
  --worker              one arm in this process. --inc tree: FlashInfer as the job's tree has it
                        (run_gpu_pair arms); --inc <dir>: a text-patched copy (driver arms).
  (default) driver      stage 0 (256-code cvt table + ptxas check), then the arms main, v2m,
                        v2mA, v2mB as subprocesses, each with its own include copy and fresh
                        FLASHINFER_WORKSPACE_BASE, for --rounds interleaved rounds; then byte
                        compares of O and LSE vs the main arm and errors vs an FP32 reference.

Arms are built from the job's tree (which carries v2m + variant A):
  main  : v2m block reverted, vec_cast software path (#ifdef -> #if 0)  == main's sources
  v2m   : vec_cast software path
  v2mA  : the tree as is: hardware cvt (0x7F/0xFF give NaN like torch; main gives +-480)
  v2mB  : A + per-call NaN-code probe and software fallback (all 256 codes == main)
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
PAGE = 32
CASES = "1x300,16x300,27x300,1x1000,8x1000,1x2000,4x2000,16x512,1x4000,2x4000,1x8000"
# global-only read-rows cases (layer 59 sub-call): q rows 16 per request against kv_len
# 'rr<B>x<S>' cases (global layer only): layer 59's read-rows sub-call, B requests x the trailing
# read_rows_n(S) rows against kv_len S, planned with disable_split_kv=True (16 rows at S=8000).

# ------------------------------------------------------------------ text transforms
V2M_NEW = """  for (uint32_t idx = thread_id; idx < NUM_B128; idx += NUM_THREADS) {
    uint32_t row = idx / FP8_COLS, col = idx % FP8_COLS;
    if constexpr (FP8_COLS == 16 && SWIZZLE == SwizzleMode::k128B && NUM_THREADS % 32 == 0) {
      // Each warp converts 2 rows x 16 b128 columns. With the row-major map, the 8 lanes of a
      // quarter-warp write columns 2*col of one row, which land on 4 of the 8 16-byte bank groups
      // (2-way conflict on both staging stores). Here each quarter-warp takes 4 columns of the
      // even row and the paired 4 columns of the odd row: the k128B XOR (row % 8) then spreads
      // both the b128 reads and the two b128 writes over all 8 bank groups. Same data, same
      // layout, only the thread that moves each chunk changes.
      const uint32_t lane = idx % 32, quarter = lane / 8, half = (lane % 8) / 4;
      row = (idx / 32) * 2 + half;
      col = 8 * (quarter / 2) + 4 * ((quarter % 2) ^ half) + (lane % 4);
    }
"""
V2M_OLD = """  for (uint32_t idx = thread_id; idx < NUM_B128; idx += NUM_THREADS) {
    uint32_t row = idx / FP8_COLS, col = idx % FP8_COLS;
"""
HW_IFDEF = ("#ifdef FLASHINFER_HARDWARE_FP8_E4M3_TO_BF16_CONVERSION_ENABLED\n"
            "      // Hardware paired conversion")
HW_OFF = ("#if 0  // idea65 arm: software conversion (main)\n"
          "      // Hardware paired conversion")
# variant B (job 475's first landed form): A + a per-call NaN-code probe that re-runs the
# software path when any byte is 0x7F/0xFF, so all 256 codes match main
B1_OLD = """      // satfinite-quantized KV cache never holds them.
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        const uint32_t w = *(const uint32_t*)&src[i * 4];
"""
B1_NEW = """      // satfinite-quantized KV cache never holds them.
      uint32_t words[vec_size / 4];  // idea65 arm B: NaN-code fallback
      uint32_t nan_probe = 0;
#pragma unroll
      for (uint32_t i = 0; i < vec_size / 4; ++i) {
        const uint32_t w = *(const uint32_t*)&src[i * 4];
        words[i] = w;
        // bit 7 of a byte of (b & 0x7F) + 1 is set iff b is 0x7F or 0xFF (no carry across bytes)
        nan_probe |= (w & 0x7F7F7F7Fu) + 0x01010101u;
"""
B2_OLD = """        *(uint32_t*)&dst[i * 4 + 2] = hi;
      }
#else"""
B2_NEW = """        *(uint32_t*)&dst[i * 4 + 2] = hi;
      }
      if (__builtin_expect((nan_probe & 0x80808080u) != 0u, 0)) {
#pragma unroll
        for (uint32_t i = 0; i < vec_size / 4; ++i) {
          fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>(&words[i], (uint2*)&dst[i * 4]);
        }
      }
#else"""

ARMS = {
    "main": [("include/flashinfer/attention/prefill.cuh", V2M_NEW, V2M_OLD),
             ("include/flashinfer/vec_dtypes.cuh", HW_IFDEF, HW_OFF)],
    "v2m": [("include/flashinfer/vec_dtypes.cuh", HW_IFDEF, HW_OFF)],
    "v2mA": [],  # the tree as landed
    "v2mB": [("include/flashinfer/vec_dtypes.cuh", B1_OLD, B1_NEW),
             ("include/flashinfer/vec_dtypes.cuh", B2_OLD, B2_NEW)],
}


def log(*a):
    print(*a, flush=True)


def heartbeat(stop, tag):
    t0 = time.time()
    while not stop.wait(60):
        log(f"[hb {tag}] {time.time() - t0:.0f}s load={os.getloadavg()}")


def package_data():
    import importlib.util
    spec = importlib.util.find_spec("flashinfer")
    return pathlib.Path(spec.origin).resolve().parent / "data"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def make_arm_tree(arm, dst):
    data = package_data()
    dst = pathlib.Path(dst)
    for sub in ("include", "csrc"):
        shutil.copytree(data / sub, dst / sub, symlinks=False)
    for rel, old, new in ARMS[arm]:
        p = dst / rel
        t = p.read_text()
        n = t.count(old)
        if n != 1:
            raise RuntimeError(f"arm {arm}: anchor found {n}x in {rel}: {old[:70]!r}")
        p.write_text(t.replace(old, new))
    files = {rel: sha(dst / rel) for rel in ("include/flashinfer/attention/prefill.cuh",
                                             "include/flashinfer/vec_dtypes.cuh")}
    return dst, files


# ------------------------------------------------------------------ cases
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
    d = dict(num_heads=nq, sliding_kv_heads=nkv, sliding_head_dim=hd, global_kv_heads=gkv,
             global_head_dim=ghd, sliding_window=tc.get("sliding_window"),
             attention_k_eq_v=tc.get("attention_k_eq_v"), num_layers=tc.get("num_hidden_layers"),
             hidden=tc.get("hidden_size"))
    qpath = os.path.join(ckpt, "hf_quant_config.json")
    if os.path.exists(qpath):
        q = json.load(open(qpath)).get("quantization", {})
        d["kv_cache_quant_algo"] = q.get("kv_cache_quant_algo")
    return d


READ_ROWS_TILE = 64   # main's d512 CTA tile (Tile64) that the read-rows subset is aligned to
READ_ROWS_MIN = 512   # TRTLLM_GEMMA4_LAST_LAYER_ATTN_ROWS_MIN_TOKENS default


def read_rows_n(L, dims):
    """Trailing rows per request that layer 59's read-rows sub-call runs (pitfall 775)."""
    group = dims["num_heads"] // dims["global_kv_heads"]
    tile = READ_ROWS_TILE
    head = tile // group + 1
    return head + (L - head) % (tile // math.gcd(tile, group))


def case_list(layers, cases, dims):
    out = []
    for layer in layers.split(","):
        for cs in [c for c in cases.split(",") if c]:
            if cs.startswith("rr"):
                if layer != "global":
                    continue
                b, s = (int(x) for x in cs[2:].split("x"))
                n = read_rows_n(s, dims)
                out.append((layer, cs, [n] * b, [s] * b, {"disable_split_kv": True}))
            else:
                b, s = (int(x) for x in cs.split("x"))
                out.append((layer, cs, [s] * b, None, {}))
    return out


# Workload case -> the packed attention call one prefill forward makes for it (max_num_tokens
# 8192, fresh prompts, kv_len == qo_len), as in experiments/ascent/idea38/bench.py.
WORKLOAD = {"isl300@1": "1x300", "isl300@4": "4x300", "isl300@16": "16x300",
            "isl300@64": "27x300", "isl1k@1": "1x1000", "isl1k@4": "4x1000",
            "isl1k@16": "8x1000", "isl1k@64": "8x1000", "isl2k@1": "1x2000",
            "isl2k@4": "4x2000", "isl2k@16": "4x2000", "isl2k@64": "4x2000",
            "isl4k@1": "1x4000", "isl4k@4": "2x4000", "isl4k@16": "2x4000",
            "isl4k@64": "2x4000", "isl8k@1": "1x8000", "isl8k@4": "1x8000",
            "isl8k@16": "1x8000", "isl8k@64": "1x8000"}
PAIR_CASES = ("1x300,4x300,16x300,27x300,1x1000,4x1000,8x1000,1x2000,4x2000,16x512,1x4000,"
              "2x4000,1x8000,rr4x300,rr16x300,rr27x300,rr1x1000,rr4x1000,rr8x1000,rr1x2000,"
              "rr4x2000,rr1x4000,rr2x4000,rr1x8000")


def layer_counts(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    lt = tc.get("layer_types") or []
    ns = sum(1 for t in lt if "sliding" in t)
    ng = sum(1 for t in lt if "full" in t or "global" in t)
    return ns, ng


def composite(batch, dims, ckpt):
    """Per-forward FA2 attention kernel time of each workload case on the measured per-kernel
    medians: n_sliding x d256 + (n_global - 1) x d512 + layer 59's call (read-rows sub-call
    at >= 512 rows, else the full-row d512 call). Spread = the same weighted sum of p90 - p10."""
    ns, ng = layer_counts(ckpt)
    out = {}
    for label, spec in WORKLOAD.items():
        b, s = (int(x) for x in spec.split("x"))
        k256, k512 = batch.get(f"{spec}_d256"), batch.get(f"{spec}_d512")
        last = batch.get(f"rr{spec}_d512") if b * s >= READ_ROWS_MIN else k512
        if not (k256 and k512 and last):
            continue
        parts = [(ns, k256), (ng - 1, k512), (1, last)]
        med = sum(w * k["kernel_ms"] for w, k in parts)
        spread = sum(w * (k["p90_ms"] - k["p10_ms"]) for w, k in parts)
        out[label] = dict(kernel_ms=round(med, 4), noise_floor=round(spread / med, 4),
                          d256_ms=k256["kernel_ms"], d512_ms=k512["kernel_ms"],
                          last_layer_ms=last["kernel_ms"], n_sliding=ns, n_global=ng)
    return out


class Case:
    def __init__(self, layer, label, q_lens, kv_lens, dims, plan_kw, seed=1234):
        import torch
        self.layer, self.label, self.plan_kw = layer, label, dict(plan_kw)
        self.Hq = dims["num_heads"]
        if layer == "sliding":
            self.Hkv, self.D = dims["sliding_kv_heads"], dims["sliding_head_dim"]
            self.window_left = dims["sliding_window"] - 1
        else:
            self.Hkv, self.D = dims["global_kv_heads"], dims["global_head_dim"]
            self.window_left = -1
        self.sm_scale = 1.0 / math.sqrt(self.D)
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        self.q_lens = list(q_lens)
        self.kv_lens = list(kv_lens) if kv_lens is not None else list(q_lens)
        self.batch = len(self.q_lens)
        npages = [(L + PAGE - 1) // PAGE for L in self.kv_lens]
        qo, kvp = [0], [0]
        for L, n in zip(self.q_lens, npages):
            qo.append(qo[-1] + L)
            kvp.append(kvp[-1] + n)
        self.total_q = qo[-1]
        self.qo_indptr = torch.tensor(qo, dtype=torch.int32, device=dev)
        self.kv_indptr = torch.tensor(kvp, dtype=torch.int32, device=dev)
        self.kv_indices = torch.randperm(kvp[-1], generator=g, device=dev).to(torch.int32)
        self.last_page_len = torch.tensor([(L - 1) % PAGE + 1 for L in self.kv_lens],
                                          dtype=torch.int32, device=dev)
        pool = torch.randn(kvp[-1], 2, self.Hkv, PAGE, self.D, generator=g, device=dev,
                           dtype=torch.float32)
        self.kv_fp8 = pool.to(torch.float8_e4m3fn)
        del pool
        self.q = torch.randn(self.total_q, self.Hq, self.D, generator=g, device=dev,
                             dtype=torch.float32).to(torch.bfloat16)

    def gather_kv(self, r):
        s, e = int(self.kv_indptr[r]), int(self.kv_indptr[r + 1])
        pages = self.kv_fp8[self.kv_indices[s:e].long()].float()
        L = self.kv_lens[r]
        k = pages[:, 0].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:L]
        v = pages[:, 1].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:L]
        return k, v

    def reference(self):
        """FP32 softmax(QK^T*s + causal/sliding mask) V on the dequantized FP8 KV."""
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
                if self.window_left >= 0:
                    mask &= (i - j) <= self.window_left
                for h in range(self.Hq):
                    sc = (self.q[s:e, h].float() @ k[:, h // grp].T) * self.sm_scale
                    sc.masked_fill_(~mask, float("-inf"))
                    out[s:e, h] = torch.softmax(sc, dim=-1) @ v[:, h // grp]
        return out


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


def raw(t):
    import torch
    t = t.contiguous()
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


# ------------------------------------------------------------------ worker
def worker(a):
    if a.fresh_base:
        # build every module from this tree's sources (no .so shared with other jobs/trees)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = tempfile.mkdtemp(prefix=f"idea65_ws_{a.arm}_")
        os.environ["FLASHINFER_JIT_DEBUG"] = "0"
    log("FLASHINFER_WORKSPACE_BASE", os.environ.get("FLASHINFER_WORKSPACE_BASE", "<default>"))
    from flashinfer.jit import env as jit_env
    if a.inc != "tree":
        jit_env.FLASHINFER_INCLUDE_DIR = pathlib.Path(a.inc) / "include"
        jit_env.FLASHINFER_CSRC_DIR = pathlib.Path(a.inc) / "csrc"
    jit_env.get_aot_artifacts = lambda name: ()
    import torch
    import flashinfer
    from flashinfer.jit import env as jit_env
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, a.arm), daemon=True).start()
    log("ARM", a.arm, "flashinfer", getattr(flashinfer, "__version__", "?"), flashinfer.__file__)
    log("JIT_DIR", jit_env.FLASHINFER_JIT_DIR, "INCLUDE", jit_env.FLASHINFER_INCLUDE_DIR)
    inc = pathlib.Path(jit_env.FLASHINFER_INCLUDE_DIR)
    for rel in ("flashinfer/attention/prefill.cuh", "flashinfer/vec_dtypes.cuh"):
        txt = (inc / rel).read_text()
        log("SRC", rel, "sha", hashlib.sha256(txt.encode()).hexdigest()[:16],
            "v2m" if "quarter-warp takes 4 columns" in txt else "",
            "hwcvt_block" if "FLASHINFER_HARDWARE_FP8_E4M3_TO_BF16_CONVERSION_ENABLED" in txt
            else "", "armB_fallback" if "idea65 arm B" in txt else "",
            "sw_forced" if "idea65 arm: software" in txt else "")
    log("ENV FLASHINFER_FA2_D512_TILE64", os.environ.get("FLASHINFER_FA2_D512_TILE64", "<unset>"))
    dims = load_dims(a.ckpt)
    log("DIMS", json.dumps(dims))
    p = torch.cuda.get_device_properties(0)
    log("GPU", p.name, "SMs", p.multi_processor_count)
    ws = torch.zeros(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    save = pathlib.Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    batch, hashes, recs = {}, {}, []
    ok = True
    # Warmup phase (untimed, not recorded): build/load every JIT module the cases use, then keep
    # the GPU busy ~3 s so clocks are up before the first timed case (the first case after a
    # long CPU-bound JIT build otherwise times low-clock).
    t0 = time.time()
    for layer in a.layers.split(","):
        c = Case(layer, "warm", [300], None, dims, {})
        wo = torch.empty(c.total_q, c.Hq, c.D, dtype=torch.bfloat16, device="cuda")
        ww = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
        ww.plan(c.qo_indptr, c.kv_indptr, c.kv_indices, c.last_page_len, c.Hq, c.Hkv, c.D,
                PAGE, causal=True, sm_scale=c.sm_scale, window_left=c.window_left,
                q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)
        ww.run(c.q, c.kv_fp8, out=wo)
        torch.cuda.synchronize()
        del c, wo, ww
    log("WARM jit/load done", round(time.time() - t0, 1), "s")
    xa = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t1 = time.time()
    while time.time() - t1 < 3.0:
        for _ in range(20):
            xa = (xa @ xa).clamp_(-1, 1)
        torch.cuda.synchronize()
    del xa
    torch.cuda.empty_cache()
    log("WARM gpu spin done")
    for layer, label, ql, kl, kw in case_list(a.layers, a.cases, dims):
        key = f"{label}_{'d256' if layer == 'sliding' else 'd512'}"
        t0 = time.time()
        c = Case(layer, label, ql, kl, dims, kw)
        out = torch.empty(c.total_q, c.Hq, c.D, dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(c.total_q, c.Hq, dtype=torch.float32, device="cuda")
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
        w.plan(c.qo_indptr, c.kv_indptr, c.kv_indices, c.last_page_len, c.Hq, c.Hkv, c.D, PAGE,
               causal=True, sm_scale=c.sm_scale, window_left=c.window_left,
               q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, **c.plan_kw)
        pi = [int(x) for x in w._plan_info]
        q0, kv0 = c.q.clone(), c.kv_fp8.clone()

        def run():
            w.run(c.q, c.kv_fp8, out=out)

        # untimed repeated-execution sequence: same call count as the timing run
        run()
        torch.cuda.synchronize()
        first = out.clone()
        mism = 0
        for _ in range(a.warmup + a.iters - 1):
            run()
            torch.cuda.synchronize()
            mism += int(not torch.equal(raw(out), raw(first)))
        # O + LSE call (untimed), O must equal the no-LSE call's O
        w.run(c.q, c.kv_fp8, out=out, lse=lse, return_lse=True)
        torch.cuda.synchronize()
        o_lse_same = bool(torch.equal(raw(out), raw(first)))
        rec = dict(case=key, cta_tile_q=pi[3], split_kv=bool(pi[14]), padded_batch=pi[0],
                   repeat=dict(calls=a.warmup + a.iters, mismatches=mism),
                   o_lse_call_equal=o_lse_same,
                   q_unchanged=bool(torch.equal(raw(q0), raw(c.q))),
                   kv_unchanged=bool(torch.equal(raw(kv0), raw(c.kv_fp8))),
                   finite=bool(first.float().isfinite().all()))
        ok &= (mism == 0 and o_lse_same and rec["q_unchanged"] and rec["kv_unchanged"]
               and rec["finite"])
        h = hashlib.sha256(raw(first).cpu().numpy().tobytes())
        h.update(raw(lse).cpu().numpy().tobytes())
        hashes[key] = h.hexdigest()[:24]
        rec["o_lse_sha"] = hashes[key]
        if save:
            torch.save({"o": first.cpu(), "lse": lse.clone().cpu()}, save / f"{a.arm}_{key}.pt")
        if not a.no_time:
            t = time_fn(run, a.warmup, a.iters)
            rec.update({k: round(v, 5) for k, v in t.items()})
            batch[key] = dict(kernel_ms=round(t["median_ms"], 5), noise_floor=round(t["noise"], 4),
                              p10_ms=round(t["p10_ms"], 5), p90_ms=round(t["p90_ms"], 5))
        rec["wall_s"] = round(time.time() - t0, 1)
        log("RESULT", json.dumps(rec))
        recs.append(rec)
        del c, out, lse, first, w, q0, kv0
        torch.cuda.empty_cache()
    # SASS census of the built prefill modules (best effort)
    if a.sass:
        try:
            jd = pathlib.Path(jit_env.FLASHINFER_JIT_DIR)
            cuobj = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
            for so in sorted(jd.rglob("*.so")):
                if "prefill" not in so.name:
                    continue
                txt = subprocess.run([cuobj, "-sass", str(so)], capture_output=True, text=True,
                                     timeout=900).stdout
                cnt = {k: txt.count(k) for k in ("F2FP", "HMUL2", "PRMT", "HMMA", "BRA")}
                log("SASS", so.name, "age_s", round(time.time() - so.stat().st_mtime),
                    json.dumps(cnt))
        except Exception as ex:  # noqa: BLE001
            log("SASS census failed", ex)
    stop.set()
    kernels = batch
    if a.workload_batch:
        batch = composite(kernels, dims, a.ckpt)
        for label, v in batch.items():
            log("WORKLOAD", label, json.dumps(v))
    print(json.dumps({"batch": batch, "kernels": kernels, "output": {"class": "exact"},
                      "passed": bool(ok),
                      "arm": a.arm, "hashes": hashes, "results": recs}), flush=True)


# ------------------------------------------------------------------ stage 0
STAGE0_CU = r"""
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include "flashinfer/vec_dtypes.cuh"
using namespace flashinfer;

__global__ void conv(const uint32_t* in, uint32_t* o_main, uint32_t* o_b4, uint32_t* o_b16,
                     uint32_t* o_a, int nwords, int* flags) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t == 0) {
#ifdef FLASHINFER_HARDWARE_FP8_E4M3_TO_BF16_CONVERSION_ENABLED
    flags[0] = 1;
#else
    flags[0] = 0;
#endif
#ifdef __CUDA_ARCH__
    flags[1] = __CUDA_ARCH__;
#endif
#ifdef __CUDA_ARCH_FAMILY_SPECIFIC__
    flags[2] = __CUDA_ARCH_FAMILY_SPECIFIC__;
#endif
#ifdef __CUDA_ARCH_SPECIFIC__
    flags[3] = __CUDA_ARCH_SPECIFIC__;
#endif
  }
  if (4 * t >= nwords) return;
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) w[i] = in[4 * t + i];
  // main's vec_cast<bf16, e4m3>::cast<4k> body: fast_dequant_f8f16x4 per word
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint2 o;
    fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>(&w[i], &o);
    o_main[8 * t + 2 * i] = o.x;
    o_main[8 * t + 2 * i + 1] = o.y;
  }
  // the header's vec_cast, cast<4> (d512 fp8x4_to_16b) and cast<16> (d256 repack)
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    alignas(8) nv_bfloat16 o[4];
    vec_cast<nv_bfloat16, __nv_fp8_e4m3>::cast<4>(o, (const __nv_fp8_e4m3*)&w[i]);
    o_b4[8 * t + 2 * i] = *(uint32_t*)&o[0];
    o_b4[8 * t + 2 * i + 1] = *(uint32_t*)&o[2];
  }
  {
    alignas(16) nv_bfloat16 o[16];
    vec_cast<nv_bfloat16, __nv_fp8_e4m3>::cast<16>(o, (const __nv_fp8_e4m3*)w);
#pragma unroll
    for (int i = 0; i < 8; ++i) o_b16[8 * t + i] = *(uint32_t*)&o[2 * i];
  }
#ifdef FLASHINFER_HARDWARE_FP8_E4M3_TO_BF16_CONVERSION_ENABLED
  // variant A: raw hardware conversion, no NaN-code fallback
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t lo, hi;
    asm("{\n .reg .b16 e01, e23;\n mov.b32 {e01, e23}, %2;\n"
        " cvt.rn.bf16x2.e4m3x2 %0, e01;\n cvt.rn.bf16x2.e4m3x2 %1, e23;\n}\n"
        : "=r"(lo), "=r"(hi) : "r"(w[i]));
    o_a[8 * t + 2 * i] = lo;
    o_a[8 * t + 2 * i + 1] = hi;
  }
#endif
}

int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb");
  fseek(f, 0, SEEK_END);
  long nbytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  int nwords = (int)(nbytes / 4);
  uint32_t* h_in = (uint32_t*)malloc(nbytes);
  if (fread(h_in, 1, nbytes, f) != (size_t)nbytes) return 2;
  fclose(f);
  uint32_t *d_in, *d_out;
  int* d_flags;
  size_t ob = (size_t)nwords * 2 * 4;
  cudaMalloc(&d_in, nbytes);
  cudaMalloc(&d_out, 4 * ob);
  cudaMalloc(&d_flags, 16);
  cudaMemset(d_out, 0, 4 * ob);
  cudaMemset(d_flags, 0, 16);
  cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice);
  int threads = nwords / 4;
  conv<<<(threads + 127) / 128, 128>>>(d_in, d_out, d_out + nwords * 2, d_out + nwords * 4,
                                       d_out + nwords * 6, nwords, d_flags);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA error %s\n", cudaGetErrorString(e)); return 3; }
  uint32_t* h_out = (uint32_t*)malloc(4 * ob);
  int h_flags[4];
  cudaMemcpy(h_out, d_out, 4 * ob, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_flags, d_flags, 16, cudaMemcpyDeviceToHost);
  FILE* g = fopen(argv[2], "wb");
  fwrite(h_out, 1, 4 * ob, g);
  fclose(g);
  printf("FLAGS hw=%d arch=%d family=%d specific=%d\n", h_flags[0], h_flags[1], h_flags[2],
         h_flags[3]);
  return 0;
}
"""


def stage0(a, workdir, inc=None, tag="tree", expect="A"):
    import numpy as np
    import torch
    from flashinfer.jit.cpp_ext import get_cuda_path
    inc = pathlib.Path(inc) if inc else package_data() / "include"
    workdir = pathlib.Path(workdir) / f"stage0_{tag}"
    workdir.mkdir(parents=True, exist_ok=True)
    log("STAGE0", tag, "include", inc, "expect", expect)
    nvcc = os.path.join(get_cuda_path(), "bin", "nvcc")
    cuobj = os.path.join(get_cuda_path(), "bin", "cuobjdump")
    log("STAGE0 nvcc", nvcc, subprocess.run([nvcc, "--version"], capture_output=True,
                                            text=True).stdout.strip().splitlines()[-1])
    vtxt = (inc / "flashinfer/vec_dtypes.cuh").read_text()
    log("STAGE0 vec_dtypes sha", hashlib.sha256(vtxt.encode()).hexdigest()[:16],
        "hw block present:", "FLASHINFER_HARDWARE_FP8_E4M3_TO_BF16_CONVERSION_ENABLED" in vtxt)
    src = workdir / "stage0.cu"
    src.write_text(STAGE0_CU)
    # inputs: (1) all 256 codes in order (groups with 0x7F / 0xFF take the fallback),
    # (2) the 254 finite codes shuffled + 2 zeros (no fallback anywhere), (3) 64 Ki random bytes
    rng = np.random.default_rng(65)
    all_codes = np.arange(256, dtype=np.uint8)
    finite = np.array([c for c in range(256) if c & 0x7F != 0x7F] + [0, 0], dtype=np.uint8)
    rng.shuffle(finite)
    rnd = rng.integers(0, 256, size=65536, dtype=np.uint8)
    buf = np.concatenate([all_codes, finite, rnd])
    pad = (-len(buf)) % 16
    buf = np.concatenate([buf, np.zeros(pad, dtype=np.uint8)])
    inp = workdir / "in.bin"
    buf.tofile(inp)
    flag_sets = {
        "sm120f": ["-gencode=arch=compute_120f,code=sm_120f"],
        "sm120a": ["-gencode=arch=compute_120a,code=sm_120a"],
        "sm120_plain": ["-gencode=arch=compute_120,code=sm_120"],
    }
    res = {}
    torch_ref = torch.from_numpy(buf.copy()).view(torch.float8_e4m3fn).to(torch.bfloat16)
    torch_ref = torch_ref.view(torch.int16).numpy().view(np.uint16)
    for name, arch in flag_sets.items():
        exe = workdir / f"stage0_{name}"
        cmd = [nvcc, "-std=c++17", "-O3", "-use_fast_math", "-DNDEBUG",
               "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_FP8_E4M3", *arch,
               "-Xptxas=-v", f"-I{inc}", str(src), "-o", str(exe)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        rec = dict(build_rc=p.returncode)
        if p.returncode != 0:
            rec["build_err"] = (p.stdout + p.stderr)[-1500:]
            res[name] = rec
            log("STAGE0", name, json.dumps(rec))
            continue
        ptxas = [ln for ln in (p.stdout + p.stderr).splitlines() if "registers" in ln]
        rec["ptxas"] = ptxas[:2]
        out = workdir / f"out_{name}.bin"
        r = subprocess.run([str(exe), str(inp), str(out)], capture_output=True, text=True)
        rec["run_rc"] = r.returncode
        rec["run_out"] = r.stdout.strip()
        if r.returncode == 0:
            o = np.fromfile(out, dtype=np.uint16)
            n = len(buf)
            o_main, o_b4, o_b16, o_a = o[:n], o[n:2 * n], o[2 * n:3 * n], o[3 * n:4 * n]
            seg = {"all256": slice(0, 256), "finite_shuffled": slice(256, 512),
                   "random64k": slice(512, n)}
            nanpos = (buf & 0x7F) == 0x7F
            tab = {}
            for sname, sl in seg.items():
                fin = ~nanpos[sl]
                tab[sname] = dict(
                    b4_eq_main=int((o_b4[sl] == o_main[sl]).sum()),
                    b16_eq_main=int((o_b16[sl] == o_main[sl]).sum()),
                    a_eq_main=int((o_a[sl] == o_main[sl]).sum()),
                    main_eq_torch=int((o_main[sl] == torch_ref[sl]).sum()),
                    n=int(sl.stop - sl.start), n_nan_codes=int(nanpos[sl].sum()),
                    b4_eq_main_finite=int((o_b4[sl] == o_main[sl])[fin].sum()),
                    b16_eq_main_finite=int((o_b16[sl] == o_main[sl])[fin].sum()),
                    n_finite=int(fin.sum()),
                    b4_nan_is_7fff=int((o_b4[sl][~fin] == 0x7FFF).sum()),
                    b16_nan_is_7fff=int((o_b16[sl][~fin] == 0x7FFF).sum()))
            rec["table"] = tab
            # per-code table for all 256 codes: main, B(cast<4>), B(cast<16>), A, torch
            codes = {}
            for c in range(256):
                codes[f"{c:02x}"] = [f"{int(x):04x}" for x in
                                     (o_main[c], o_b4[c], o_b16[c], o_a[c], torch_ref[c])]
            diff = {k: v for k, v in codes.items() if len(set(v[:3])) != 1 or v[3] != v[0]
                    or v[4] != v[0]}
            rec["codes_differing_any_variant_or_torch"] = diff
            if name == "sm120f":
                rec["code_table_256 [main,B4,B16,A,torch]"] = codes
        # SASS census of the conversion kernel
        sass = subprocess.run([cuobj, "-sass", str(exe)], capture_output=True, text=True).stdout
        rec["sass_counts"] = {k: sass.count(k) for k in ("F2FP", "HMUL2", "PRMT", "BRA",
                                                           "LOP3")}
        f2fp = sorted({ln.strip()[:90] for ln in sass.splitlines() if "F2FP" in ln})
        rec["sass_f2fp_examples"] = f2fp[:4]
        res[name] = rec
        log("STAGE0", name, json.dumps(rec))
    def ok_table(t, mode):
        for sname in ("all256", "finite_shuffled", "random64k"):
            s = t.get(sname)
            if not s:
                return False
            if mode == "exact256":
                if s["b4_eq_main"] != s["n"] or s["b16_eq_main"] != s["n"]:
                    return False
            else:  # A: finite codes bit-equal, NaN codes -> 0x7FFF
                if (s["b4_eq_main_finite"] != s["n_finite"]
                        or s["b16_eq_main_finite"] != s["n_finite"]
                        or s["b4_nan_is_7fff"] != s["n_nan_codes"]
                        or s["b16_nan_is_7fff"] != s["n_nan_codes"]):
                    return False
        return True

    f = res.get("sm120f", {})
    plain = res.get("sm120_plain", {})
    passed = (f.get("build_rc") == 0 and "hw=1" in f.get("run_out", "")
              and ok_table(f.get("table", {}), "exact256" if expect == "B" else "A")
              # plain compute_120 keeps the software path: all 256 codes == main
              and plain.get("build_rc") == 0 and "hw=0" in plain.get("run_out", "")
              and ok_table(plain.get("table", {}), "exact256"))
    log("STAGE0_RESULT", tag, "expect", expect, "passed", passed)
    return dict(passed=bool(passed), tag=tag, expect=expect, results=res)


# ------------------------------------------------------------------ driver
def run_worker(arm, inc, a, base, save_dir):
    cmd = [sys.executable, __file__, "--worker", "--arm", arm, "--inc", str(inc), "--ckpt", a.ckpt,
           "--layers", a.layers, "--cases", a.cases, "--warmup", str(a.warmup), "--iters",
           str(a.iters)]
    if save_dir:
        cmd += ["--save-dir", str(save_dir)]
    if a.sass and save_dir:
        cmd += ["--sass"]
    env = dict(os.environ)
    env["FLASHINFER_WORKSPACE_BASE"] = str(base)
    env["FLASHINFER_JIT_DEBUG"] = "0"
    env.pop("FLASHINFER_FA2_D512_TILE64", None)
    log("SPAWN", arm, "base", base)
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        line = line.rstrip("\n")
        if line.startswith("{"):
            last = line
            continue
        if line.startswith(("[", " ")) and "nvcc" in line:
            continue
        print(f"[{arm}] {line[:1500]}", flush=True)
    rc = p.wait()
    log("ARM_DONE", arm, "rc", rc, "wall_s", round(time.time() - t0, 1))
    return json.loads(last) if (rc == 0 and last) else None


def driver(a):
    import torch
    work = pathlib.Path(tempfile.mkdtemp(prefix="idea65_"))
    log("DRIVER work", work, "load", os.getloadavg())
    summary = {}
    arms = [x for x in a.arms.split(",") if x]
    trees, bases = {}, {}
    for arm in arms:
        d, files = make_arm_tree(arm, work / f"tree_{arm}")
        trees[arm] = d
        bases[arm] = pathlib.Path(tempfile.mkdtemp(prefix=f"idea65_ws_{arm}_"))
        log("ARM_TREE", arm, json.dumps(files))
    if a.stage0:
        s0 = stage0(a, work, tag="tree_A", expect="A")
        s0b = None
        if "v2mB" in trees:
            s0b = stage0(a, work, inc=trees["v2mB"] / "include", tag="armB", expect="B")
        summary["stage0"] = dict(passed=s0["passed"] and (s0b is None or s0b["passed"]),
                                 tree_A=s0, armB=s0b)
        log("STAGE0_PASSED", summary["stage0"]["passed"])
    save = work / "out"
    res = {arm: [] for arm in arms}
    for r in range(a.rounds):
        order = arms if r % 2 == 0 else list(reversed(arms))
        for arm in order:
            res[arm].append(run_worker(arm, trees[arm], a, bases[arm], save if r == 0 else None))
    passed = all(all(j is not None and j.get("passed") for j in res[arm]) for arm in arms)
    # numerics: raw bytes of O and LSE vs the main arm; errors vs FP32 reference
    dims = load_dims(a.ckpt)
    numer = {}
    ref_arm = arms[0]
    for layer, label, ql, kl, kw in case_list(a.layers, a.cases, dims):
        key = f"{label}_{'d256' if layer == 'sliding' else 'd512'}"
        outs = {}
        for arm in arms:
            pth = save / f"{arm}_{key}.pt"
            if pth.exists():
                outs[arm] = torch.load(pth)
        row = {}
        ref = None
        if not a.no_ref:
            c = Case(layer, label, ql, kl, dims, kw)
            ref = c.reference()
            del c
        m = outs.get(ref_arm)
        for arm, d in outs.items():
            o = d["o"].cuda()
            cell = {}
            if ref is not None:
                diff = o.float() - ref
                cell["max_abs_ref"] = float(diff.abs().max())
                cell["rel_fro_ref"] = float(diff.norm() / ref.norm())
            if m is not None and arm != ref_arm:
                mo = m["o"].cuda()
                cell["o_byte_mismatch"] = int((raw(o) != raw(mo)).sum())
                cell["lse_byte_mismatch"] = int((raw(d["lse"]) != raw(m["lse"])).sum())
                cell["maxabs_vs_main"] = float((o.float() - mo.float()).abs().max())
                passed &= cell["o_byte_mismatch"] == 0 and cell["lse_byte_mismatch"] == 0
            row[arm] = cell
        numer[key] = row
        log("NUMERICS", key, json.dumps(row))
        del ref, outs, m
        torch.cuda.empty_cache()
    # timing table
    table = {}
    for layer, label, ql, kl, kw in case_list(a.layers, a.cases, dims):
        key = f"{label}_{'d256' if layer == 'sliding' else 'd512'}"
        table[key] = {arm: [None if j is None else j["kernels"].get(key) for j in res[arm]]
                      for arm in arms}
    log("TABLE case | " + " | ".join(f"{arm} r{r} med(p10,p90)" for arm in arms
                                     for r in range(a.rounds)))
    for key, row in table.items():
        cells = []
        for arm in arms:
            for t in row[arm]:
                cells.append("-" if t is None else
                             f"{t['kernel_ms']:.4f}({t['p10_ms']:.4f},{t['p90_ms']:.4f})")
        meds = {arm: statistics.median([t["kernel_ms"] for t in row[arm] if t])
                for arm in arms if any(row[arm])}
        rel = ""
        if "main" in meds:
            rel = " ".join(f"{arm}/main={meds[arm] / meds['main']:.4f}" for arm in meds
                           if arm != "main")
        if "v2m" in meds and "v2mB" in meds:
            rel += f" v2mB/v2m={meds['v2mB'] / meds['v2m']:.4f}"
        if "v2mA" in meds and "v2mB" in meds:
            rel += f" v2mB/v2mA={meds['v2mB'] / meds['v2mA']:.4f}"
        log("TABLE", key, "|", " | ".join(cells), "|", rel)
    if a.stage0:
        passed &= summary["stage0"]["passed"]
    log("DRIVER END load", os.getloadavg())
    print(json.dumps({"passed": bool(passed), "stage0_passed": summary.get("stage0", {})
                      .get("passed"), "table": table, "numerics": numer}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", default="tree")
    ap.add_argument("--inc", default="tree")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layers", default="sliding,global")
    ap.add_argument("--cases", default=CASES + ",rr1x8000,rr4x2000")
    ap.add_argument("--arms", default="main,v2m,v2mA,v2mB")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--stage0", action="store_true")
    ap.add_argument("--stage0-only", action="store_true")
    ap.add_argument("--sass", action="store_true")
    ap.add_argument("--fresh-base", action="store_true")
    ap.add_argument("--workload-batch", action="store_true",
                    help="report batch keyed by workload case (per-forward FA2 composite)")
    ap.add_argument("--pair-cases", action="store_true", help=f"--cases {PAIR_CASES}")
    a = ap.parse_args()
    if a.pair_cases:
        a.cases = PAIR_CASES
    if a.worker:
        worker(a)
    elif a.stage0_only:
        work = pathlib.Path(tempfile.mkdtemp(prefix="idea65_s0_"))
        s0 = stage0(a, work)
        print(json.dumps({"passed": s0["passed"], "stage0": s0}), flush=True)
    else:
        driver(a)


if __name__ == "__main__":
    main()
