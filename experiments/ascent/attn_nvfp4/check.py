# Untimed correctness for the gated-attention decode NVFP4 W4A16 copies
# (qkv_proj (14336,5120), o_proj (5120,6144)). Runs in the candidate tree only.
#  1. Decode rows (M=1/2/4/8): eager call 1, then the same warmup (10) + timed (60)
#     CUDA-graph replays as bench.py; every replay is compared bitwise with call 1,
#     and its error vs the FP32 reference (dequantized FP8 weight x BF16 activation)
#     is recorded per iteration (drift check), next to main's FP8 path error.
#  2. Rows > 16 (17/32/2048/8192): copy-flagged module output bitwise equal to the
#     unflagged module (= main's FP8 path, FP8QDQLinearMethod.apply unchanged).
#  3. Kill switches: TRTLLM_ATTN_QKV_NVFP4_DECODE=0 / TRTLLM_ATTN_O_NVFP4_DECODE=0
#     (and the shared TRTLLM_GDN_INPROJ_NVFP4_DECODE=0) build no copy, and decode
#     output is then bitwise equal to the FP8 path.
import json
import os

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

from tensorrt_llm._torch.attention import attention as attn_mod  # noqa: E402
from tensorrt_llm._torch.modules.linear import Linear  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

WARM, ITERS = 10, 60
dev = torch.device("cuda")
SHAPES = {"qkv": (14336, 5120, 0.02, 1.0, "TRTLLM_ATTN_QKV_NVFP4_DECODE"),
          "o": (5120, 6144, 0.01, 0.5, "TRTLLM_ATTN_O_NVFP4_DECODE")}


def log(*a):
    print(*a, flush=True)


def weights(seed, n, k):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=dev) * 0.02
    w = w * (torch.rand(n, 1, generator=g, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    return (w / s).to(torch.float8_e4m3fn), s


def make(w8, s, xs, flag):
    n, k = w8.shape
    lin = Linear(k, n, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin._nvfp4_decode_copy = flag
    lin.load_weights([{"weight": w8, "weight_scale": s.reshape(()),
                       "input_scale": torch.tensor(xs, device=dev)}])
    lin.post_load_weights()
    return lin


def err(o, r):
    d = (o.float() - r).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "rel_rms": ((o.float() - r).norm() / r.norm()).item(),
            "finite": bool(torch.isfinite(o.float()).all().item())}


def main():
    results = {}
    ok_all = True
    for name, (n, k, xs, amp, env) in SHAPES.items():
        w8, s = weights(4242 + n, n, k)
        wref = w8.float() * s
        os.environ.pop(env, None)
        flag = attn_mod._attn_nvfp4_decode_requested(env)
        cand = make(w8, s, xs, flag)
        fp8 = make(w8, s, xs, False)
        has_copy = getattr(cand, "_nvfp4_decode_weight", None) is not None
        log(f"[{name}] N={n} K={k} flag={flag} copy={has_copy}")
        r = {"copy_built": has_copy, "decode": {}, "rows_gt16": {}, "kill_switch": {}}
        ok_all &= has_copy and getattr(fp8, "_nvfp4_decode_weight", None) is None
        # 1. decode rows: replays vs call 1, per-iteration error vs FP32 reference.
        for m in (1, 2, 4, 8):
            g = torch.Generator(device=dev).manual_seed(77 + m)
            x = (torch.randn(m, k, generator=g, device=dev) * amp).to(torch.bfloat16)
            ref = x.float() @ wref.t()
            first = cand(x).clone()
            fp8_out = fp8(x)
            e_first, e_fp8 = err(first, ref), err(fp8_out, ref)
            eager_rep = all(torch.equal(cand(x), first) for _ in range(5))
            s0 = torch.cuda.Stream()
            s0.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s0):
                for _ in range(3):
                    cand(x)
            torch.cuda.current_stream().wait_stream(s0)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                gout = cand(x)
            mism, max_rel = 0, 0.0
            for it in range(WARM + ITERS):
                gr.replay()
                torch.cuda.synchronize()
                if not torch.equal(gout, first):
                    mism += 1
                max_rel = max(max_rel, err(gout, ref)["rel_rms"])
            ok = (eager_rep and mism == 0 and e_first["finite"]
                  and abs(max_rel - e_first["rel_rms"]) == 0.0)
            ok_all &= ok
            r["decode"][str(m)] = {"replay_mismatches_vs_call1": mism, "replays": WARM + ITERS,
                                   "eager_repeat_bitwise": eager_rep,
                                   "err_vs_fp32_ref_nvfp4": e_first, "err_vs_fp32_ref_fp8_main": e_fp8,
                                   "err_nvfp4_vs_fp8_main": err(first, fp8_out.float()),
                                   "per_iter_max_rel_rms": max_rel, "ok": ok}
            log(f"[{name}] M={m} replays {WARM + ITERS} mismatches {mism} eager_rep {eager_rep} "
                f"rel_rms nvfp4 {e_first['rel_rms']:.5f} fp8 {e_fp8['rel_rms']:.5f}")
            del gr
        # 2. rows > 16: bitwise equal to the unflagged (main) FP8 path.
        for m in (17, 32, 2048, 8192):
            g = torch.Generator(device=dev).manual_seed(99 + m)
            x = (torch.randn(m, k, generator=g, device=dev) * amp).to(torch.bfloat16)
            eq = bool(torch.equal(cand(x), fp8(x)))
            ok_all &= eq
            r["rows_gt16"][str(m)] = eq
            log(f"[{name}] rows={m} bitwise vs FP8 path {eq}")
        # 3. kill switches.
        x = (torch.randn(4, k, device=dev) * amp).to(torch.bfloat16)
        for sw, val in ((env, "0"), ("TRTLLM_GDN_INPROJ_NVFP4_DECODE", "0")):
            os.environ[sw] = val
            fl = attn_mod._attn_nvfp4_decode_requested(env)
            lin = make(w8, s, xs, fl if sw == env else True)
            built = getattr(lin, "_nvfp4_decode_weight", None) is not None
            eq = bool(torch.equal(lin(x), fp8(x)))
            os.environ.pop(sw)
            good = (not built) and eq
            ok_all &= good
            r["kill_switch"][f"{sw}=0"] = {"flag": fl, "copy_built": built,
                                           "decode_bitwise_fp8": eq, "ok": good}
            log(f"[{name}] {sw}=0 flag {fl} copy_built {built} decode==fp8 {eq}")
            del lin
        results[name] = r
        del cand, fp8
        torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(ok_all),
                      "compared": "decode replays vs call 1 (bitwise) and vs FP32 reference "
                                  "(dequantized FP8 weight x BF16 act); rows>16 and kill switches "
                                  "bitwise vs main's FP8 path",
                      "results": results}))


if __name__ == "__main__":
    main()
