"""Idea 17 STEP 0b: the 256x128x128 large-M variants vs main's served kernels.

Variants (add-only FlashInfer overlay modules, JIT-built here under their own names):
  qkv3_t256_dyn / qkv3_t256_static  csrc/flux3_fp4_qkv3_t256_sm120.cu (Cooperative 256x128x128,
                                    L=3 planar, CLC dynamic / static persistent scheduler)
  lpt_t256_dyn / lpt_t256_static    csrc/flux3_fp4_lpt_pair_t256_sm120.cu (Cooperative 256x128x128
                                    grouped pair, longest-first counter / static round robin)
Main: flux3_sm120_strided_gemm.gemm_qkv3 (Pingpong 128x128x128 CLC), flux3_sm120_lpt_pair.run.
Bitwise: each variant output vs main's output on the same inputs (q/k/v planes; down and attn_out).
Env: STEP0_MS (default 16896,33280,4176). Last stdout line JSON.
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step0  # noqa: E402

MS = [int(x) for x in os.environ.get("STEP0_MS", "16896,33280,4176").split(",")]
log = step0.log


def build(name, src):
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    t0 = time.time()
    spec = gen_jit_spec(name, [jit_env.FLASHINFER_CSRC_DIR / src],
                        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16", "-DENABLE_FP4",
                                                        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
                        extra_cflags=["-DFAST_BUILD"])
    mod = spec.build_and_load()
    log(f"built {name} in {time.time() - t0:.0f}s")
    return mod


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda")
    strided.preload(dev)
    lpt.preload(dev)
    q256 = build("flux3_fp4_qkv3_t256_sm120_s0b", "flux3_fp4_qkv3_t256_sm120.cu")
    l256 = build("flux3_fp4_lpt_pair_t256_sm120_s0b", "flux3_fp4_lpt_pair_t256_sm120.cu")
    log(f"qkv3_t256 info dyn={q256.t256_info(1)} static={q256.t256_info(0)} "
        f"ws={q256.qkv3_t256_workspace_size(1 << 16, 3072, 3072, 1)} "
        f"lpt_t256 info={l256.lpt_pair_info()} main lpt info={lpt.module().lpt_pair_info()}")
    qws = strided.workspace(dev, 0)
    lws = lpt.workspace(dev)
    lws2 = torch.empty_like(lws)
    g = torch.Generator(device=dev)
    g.manual_seed(1717)
    table = {}
    NS = step0.NSETS
    C = step0.CALLS
    for m in MS:
        # ---- QKV3 ----
        sets = [step0.fp4_set(m, 9216, 3072, g, dev) for _ in range(NS)]
        a3 = torch.tensor([1.0 / 64, 1.0 / 32, 1.0 / 128], dtype=torch.float32, device=dev)
        outs = [torch.empty(m * 9216, dtype=torch.bfloat16, device=dev) for _ in range(NS)]
        ref = torch.empty(m * 9216, dtype=torch.bfloat16, device=dev)
        strided.gemm_qkv3(*sets[0], a3, ref)
        st = {}
        for lab, dyn in (("qkv3_t256_dyn", 1), ("qkv3_t256_static", 0)):
            o = torch.full_like(ref, float("nan"))
            q256.gemm_qkv3_t256(*sets[0], a3, o, qws, 0, 1, dyn)
            torch.cuda.synchronize()
            st[lab] = "bitwise" if step0.bits_equal(o, ref) else "DIFFERS"
        arms = {"main_qkv3": [(lambda i=i: strided.gemm_qkv3(*sets[i % NS], a3, outs[i % NS]))
                              for i in range(C)]}
        for lab, dyn in (("qkv3_t256_dyn", 1), ("qkv3_t256_static", 0)):
            for r, s in ((0, 1), (1, 1), (2, 1), (0, 4)):
                arms[f"{lab}_r{r}s{s}"] = [
                    (lambda i=i, dyn=dyn, r=r, s=s: q256.gemm_qkv3_t256(
                        *sets[i % NS], a3, outs[i % NS], qws, r, s, dyn)) for i in range(C)]
        arms["main_qkv3_AA"] = arms["main_qkv3"]
        res = step0.time_arms(arms, rounds=20)
        for k in res:
            res[k]["status"] = "served" if k.startswith("main") else st[k.rsplit("_r", 1)[0]]
        table[f"M{m}/qkv3"] = res
        base = res["main_qkv3"]["us"]
        for k, v in sorted(res.items(), key=lambda kv: kv[1]["us"]):
            log(f"M{m} qkv3 {k:26s} {v['us']:9.2f} us nf {v['nf']:.4f} "
                f"({(v['us'] / base - 1) * 100:+.2f}%) {v['status']}")
        del sets, outs, ref
        torch.cuda.empty_cache()
        # ---- LPT pair ----
        alpha_d = torch.tensor([1.0 / 64], dtype=torch.float32, device=dev)
        alpha_a = torch.tensor([1.0 / 32], dtype=torch.float32, device=dev)
        ds = [step0.fp4_set(m, 3072, 9216, g, dev) for _ in range(NS)]
        at = [step0.fp4_set(m, 3072, 3072, g, dev) for _ in range(NS)]
        od = [torch.empty((m, 3072), dtype=torch.bfloat16, device=dev) for _ in range(NS)]
        oa = [torch.empty((m, 3072), dtype=torch.bfloat16, device=dev) for _ in range(NS)]
        rd = torch.empty((m, 3072), dtype=torch.bfloat16, device=dev)
        ra = torch.empty((m, 3072), dtype=torch.bfloat16, device=dev)
        lpt.run(*ds[0], alpha_d, rd, *at[0], alpha_a, ra)
        st = {}
        for lab, mode in (("lpt_t256_dyn", 1), ("lpt_t256_static", 0), ("lpt_t256_ondemand", 2)):
            xd = torch.full_like(rd, float("nan"))
            xa = torch.full_like(ra, float("nan"))
            l256.lpt_pair(*ds[0], alpha_d, xd, *at[0], alpha_a, xa, lws2, mode, 0)
            torch.cuda.synchronize()
            st[lab] = ("bitwise" if step0.bits_equal(xd, rd) and step0.bits_equal(xa, ra)
                       else "DIFFERS")

        def mainpair(i):
            j = i % NS
            lpt.run(*ds[j], alpha_d, od[j], *at[j], alpha_a, oa[j])

        def var(i, mode):
            j = i % NS
            l256.lpt_pair(*ds[j], alpha_d, od[j], *at[j], alpha_a, oa[j], lws2, mode, 0)

        arms = {"main_lpt": [(lambda i=i: mainpair(i)) for i in range(C)]}
        for lab, mode in (("lpt_t256_dyn", 1), ("lpt_t256_static", 0), ("lpt_t256_ondemand", 2)):
            arms[lab] = [(lambda i=i, mode=mode: var(i, mode)) for i in range(C)]
        arms["main_lpt_AA"] = arms["main_lpt"]
        res = step0.time_arms(arms, rounds=20)
        for k in res:
            res[k]["status"] = "served" if k.startswith("main") else st[k]
        table[f"M{m}/lpt_pair"] = res
        base = res["main_lpt"]["us"]
        for k, v in sorted(res.items(), key=lambda kv: kv[1]["us"]):
            log(f"M{m} pair {k:26s} {v['us']:9.2f} us nf {v['nf']:.4f} "
                f"({(v['us'] / base - 1) * 100:+.2f}%) {v['status']}")
        del ds, at, od, oa, rd, ra
        torch.cuda.empty_cache()
    print(json.dumps({"table": table}), flush=True)


if __name__ == "__main__":
    main()
