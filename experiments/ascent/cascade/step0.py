#!/usr/bin/env python3
# STEP 0 for idea 20 (shared-prefix cascade decode). No model load. Cold L2 (384 MiB read flush before every call).
#  (a) main's kernel_mha over the full 34944-token sequence (32768 shared-prefix pages aliased) at B=8/16/32 for
#      main's one-wave counts (5/2/1) and the stock counts (47/23/11) and a few others; CUDA-graph replay.
#  (b) kernel_mha over a 2176-token unique suffix only (2048 + 128 decoded) at B=8/16/32, sub-seq 1/2/4.
#  (c) prefix-pass stand-in: FlashInfer FA2 BatchPrefillWithPagedKVCacheWrapper, one "request" whose qo_len=B
#      (B*24 q heads -> B*6 GQA rows per kv head) over the 1024 shared prefix pages, causal=False, return_lse.
#  (d) merge stand-in: flashinfer merge_state over [B, 24, 256].
import importlib.util
import json
import os
import sys

os.environ.setdefault("XQA_KV_DTYPE", "fp8")
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("xe", os.path.join(HERE, "..", "xqa_merge", "xqa_evidence.py"))
xe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xe)
log = xe.log


class A:
    warmup, iters, rounds = 10, 60, 2


def time_fn(fn, flush, n_warm=10, n=60):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n_warm + n)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n_warm + n)]
    for i in range(n_warm + n):
        flush()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    ts = np.array([st[i].elapsed_time(en[i]) for i in range(n_warm, n_warm + n)])
    return xe.stats(ts)


def main():
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    xe.ck(xe.cu.cuInit(0), "cuInit")
    log(f"device {torch.cuda.get_device_name()}")
    kern = xe.Kernel(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
    log(f"kernel_mha smem {kern.smem} regs {kern.regs}")
    flush = xe.Flusher(384)
    res = {"full": {}, "suffix": {}, "prefix_fi": {}, "merge_fi": {}}
    a = A()
    # (a) full sequence
    inp = xe.build_inputs(32)
    for b, ms in ((8, (5, 3, 10, 47)), (16, (2, 3, 4, 23)), (32, (1, 2, 3, 11))):
        for m in ms:
            s = xe.time_config(kern, inp, flush, b, m, a)
            res["full"][f"{b}x{m}"] = s["kernel_ms"] * 1e3
            log(f"FULL B={b} m={m}: {s['kernel_ms'] * 1e3:.1f} us (noise {s['noise_floor']:.3f})")
    pool_prefix_k = inp["pool"][0:2048:2].clone()  # [1024, 4, 32, 256] e4m3, HND
    pool_prefix_v = inp["pool"][1:2048:2].clone()
    q_all = inp["q"].clone()
    del inp
    torch.cuda.empty_cache()
    # (b) suffix only
    xe.SEQ_LEN, xe.PREFIX = 2048 + 128, 0
    inp = xe.build_inputs(32)
    for b in (8, 16, 32):
        for m in (1, 2, 4):
            s = xe.time_config(kern, inp, flush, b, m, a)
            res["suffix"][f"{b}x{m}"] = s["kernel_ms"] * 1e3
            log(f"SUFFIX B={b} m={m}: {s['kernel_ms'] * 1e3:.1f} us (noise {s['noise_floor']:.3f})")
    del inp
    torch.cuda.empty_cache()
    # (c) FlashInfer prefix-pass stand-in
    try:
        import flashinfer
        ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        for kvdt in ("fp8", "bf16"):
            if kvdt == "fp8":
                kc, vc = pool_prefix_k, pool_prefix_v
                kvt = torch.float8_e4m3fn
            else:
                kc, vc = pool_prefix_k.to(torch.bfloat16), pool_prefix_v.to(torch.bfloat16)
                kvt = torch.bfloat16
            for b in (8, 16, 32):
                q = q_all[:b].contiguous()
                for split in (None, 32, 64, 128, 256):
                    key = f"{kvdt}_B{b}_split{split}"
                    try:
                        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
                        w.plan(torch.tensor([0, b], dtype=torch.int32, device="cuda"),
                               torch.tensor([0, 1024], dtype=torch.int32, device="cuda"),
                               torch.arange(1024, dtype=torch.int32, device="cuda"),
                               torch.tensor([32], dtype=torch.int32, device="cuda"), 24, 4, 256, 32, causal=False,
                               q_data_type=torch.bfloat16, kv_data_type=kvt, o_data_type=torch.bfloat16,
                               sm_scale=0.5 / 16.0, fixed_split_size=split)
                        out = w.run(q, (kc, vc), return_lse=True)
                        torch.cuda.synchronize()
                        s = time_fn(lambda: w.run(q, (kc, vc), return_lse=True), flush)
                        res["prefix_fi"][key] = s["kernel_ms"] * 1e3
                        log(f"FI prefix {key}: {s['kernel_ms'] * 1e3:.1f} us (noise {s['noise_floor']:.3f}) "
                            f"finite {bool(torch.isfinite(out[0].float()).all())}")
                    except Exception as e:  # noqa: BLE001
                        log(f"FI prefix {key} failed: {type(e).__name__}: {str(e)[:300]}")
                        res["prefix_fi"][key] = None
        for b in (8, 16, 32):
            va = torch.randn(b, 24, 256, device="cuda", dtype=torch.bfloat16)
            vb = torch.randn(b, 24, 256, device="cuda", dtype=torch.bfloat16)
            la = torch.randn(b, 24, device="cuda")
            lb = torch.randn(b, 24, device="cuda")
            s = time_fn(lambda: flashinfer.merge_state(va, la, vb, lb), flush)
            res["merge_fi"][str(b)] = s["kernel_ms"] * 1e3
            log(f"FI merge B={b}: {s['kernel_ms'] * 1e3:.1f} us")
    except Exception as e:  # noqa: BLE001
        log(f"flashinfer import/run failed: {type(e).__name__}: {e}")
    # gate
    gate = {}
    for b in (8, 16, 32):
        best_full = min(v for k, v in res["full"].items() if k.startswith(f"{b}x"))
        suf = min(v for k, v in res["suffix"].items() if k.startswith(f"{b}x"))
        pre = [v for k, v in res["prefix_fi"].items() if v is not None and f"_B{b}_" in k and k.startswith("fp8")]
        pre = min(pre) if pre else None
        mg = res["merge_fi"].get(str(b), 0.0)
        tot = None if pre is None else pre + suf + mg
        gate[str(b)] = dict(best_full_us=best_full, main_full_us=res["full"][f"{b}x{ {8: 5, 16: 2, 32: 1}[b]}"],
                            suffix_us=suf, prefix_us=pre, merge_us=mg, cascade_us=tot,
                            ratio=None if tot is None else tot / best_full)
        log(f"GATE B={b}: {gate[str(b)]}")
    res["gate"] = gate
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
