#!/usr/bin/env python3
# Exploration-only diagnostics for the XQA multi-block merge (never used as evidence). Reuses xqa_bench.py and swaps
# in more variants (in-memory patches of the ORIGINAL mha.cu from the base commit).
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xqa_bench as xb  # noqa: E402
import xqa_newmerge as nm  # noqa: E402
from xqa_bench import _rep, compile_xqa  # noqa: E402

LOOP_HDR = "                for (uint32_t n = warpGrpIdx; n < nbSubSeqPerSeq; n += gemm1NbWarpGrps)\n                {\n"
WAIT = "                    ldgsts::commitGroup();\n                    ldgsts::waitGroup<1>();\n"
SYNC = "synchronize before consuming the merge tile.\n                    __syncwarp();\n"
COMPUTE_BEG = "                    WarpAcc tile = toWarpAcc("
COMPUTE_END = "                    sumAcc = sumAcc + tile;\n"


def patch_timing2(src):
    src = xb.patch_timing(src)
    src = _rep(src, LOOP_HDR,
               "                long long dbgIssue = 0, dbgWait = 0;\n" + LOOP_HDR +
               "                    long long const dbgL0 = clock64();\n")
    src = _rep(src, WAIT, "                    ldgsts::commitGroup();\n                    long long const dbgL1 = clock64();\n"
               "                    ldgsts::waitGroup<1>();\n")
    src = _rep(src, SYNC, SYNC + "                    long long const dbgL2 = clock64();\n"
               "                    dbgIssue += dbgL1 - dbgL0;\n                    dbgWait += dbgL2 - dbgL1;\n")
    a = "                    dbgMerge[3] = dbgT2;\n"
    src = _rep(src, a, a + "                    dbgMerge[4] = dbgIssue;\n                    dbgMerge[5] = dbgWait;\n")
    return src


def patch_timing_common(src):
    """CTA timeline + merge end stamps; works on both the original and the restructured merge."""
    src = _rep(src, "    uint32_t const idxReq = blockIdx.z;\n",
               "    uint32_t const idxReq = blockIdx.z;\n    uint64_t dbgT0;\n    " + xb.GT.format("dbgT0") + "\n")
    src = _rep(src, "            // increase the semaphore by 1\n", xb.DBG_CTA + "            // increase the semaphore by 1\n")
    a = "                MultiBlockSMem::MBBuf& mbbuf = mbsmem.storage[warpIdx.y];\n"
    src = _rep(src, a, a + "                long long const dbgC0 = clock64();\n")
    src = _rep(src, xb.LOOP_END, "\n                long long const dbgC2 = clock64();" + xb.LOOP_END)
    a = "                smemOutTile = mergeAndSaveOutTile(mergedOutTile, false);\n"
    src = _rep(src, a, a + xb.DBG_MERGE)
    return src


def patch_newmerge_timing(src, d, c):
    src = nm.patch_newmerge(src, d, c)
    src = patch_timing_common(src)
    a = "                // rescale and accumulate, in the order of the sub-sequences\n"
    return _rep(src, a, "                long long const dbgC1 = clock64();\n" + a)


def patch_noload(src):
    return _rep(src, "                        loadBufAsync(n + gemm1NbWarpGrps);\n", "")


def patch_nocompute(src):
    i0 = src.index(COMPUTE_BEG)
    i1 = src.index(COMPUTE_END) + len(COMPUTE_END)
    assert src.count(COMPUTE_BEG) == 1 and src.count(COMPUTE_END) == 1 and i0 < i1
    return src[:i0] + ("                    ThrdRegRowMax const tileRowSum = getTileBuf(mbbuf.tileRowSums, d)"
                       ".loadToReg<false>(warp);\n"
                       "                    partialMergedRowSum = partialMergedRowSum + tileRowSum;\n") + src[i1:]


_orig_make_variant = xb.make_variant


def _nm_params(spec):
    parts = spec.split("_")
    d = int(parts[1][1:]) if len(parts) > 1 else 8
    c = int(parts[2][1:]) if len(parts) > 2 else 64
    return d, c


def variant_patch(spec):
    """Returns (use_tree, patch) for a variant spec."""
    if spec == "timing2":
        return False, patch_timing2
    if spec == "noload":
        return False, patch_noload
    if spec == "nocompute":
        return False, patch_nocompute
    if spec == "treetiming":  # timing instrumentation of the tree's (committed) merge
        return True, lambda s: _rep(patch_timing_common(s), "                // rescale and accumulate, in the order",
                                    "                long long const dbgC1 = clock64();\n"
                                    "                // rescale and accumulate, in the order")
    if spec.startswith("nmt"):  # nmt_dD_cC : restructured merge with timing instrumentation
        d, c = _nm_params(spec)
        return False, lambda s: patch_newmerge_timing(s, d, c)
    if spec.startswith("nm"):  # nm_dD_cC : restructured merge
        d, c = _nm_params(spec)
        return False, lambda s: nm.patch_newmerge(s, d, c)
    return None


def make_variant(spec, tree_dir, base_dir):
    vp = variant_patch(spec)
    if vp is not None:
        return compile_xqa(tree_dir if vp[0] else base_dir, patch=vp[1])
    return _orig_make_variant(spec, tree_dir, base_dir)


xb.make_variant = make_variant


def mode_timeline2(a):
    base = xb.extract_base_tree(a.base_commit)
    tree = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    configs = xb.parse_configs(a.configs)
    inp = xb.build_inputs(max(b for b, _ in configs))
    flush = xb.Flusher(a.flush, a.flush_mb)
    off, raw = inp["scratch_off"], inp["scratch_raw"]
    out = {}
    for spec in a.timing_variant.split(","):
        kern = xb.Kernel(make_variant(spec, tree, base))
        for (b, m) in configs:
            r = xb.Runner(kern, inp, b, m)
            nseq = xb.NB_K_HEADS * b
            recs = []
            for _ in range(a.iters):
                raw[off + xb.DBG_MERGE_OFF:off + xb.DBG_MERGE_OFF + 64 * nseq].zero_()
                flush()
                r.graph.replay()
                torch.cuda.synchronize()
                o1 = off + xb.DBG_CTA_OFF
                o2 = off + xb.DBG_MERGE_OFF
                cta = raw[o1:o1 + 16 * nseq * m].view(torch.int64).view(nseq, m, 2).cpu()
                mg = raw[o2:o2 + 64 * nseq].view(torch.int64).view(nseq, 8).cpu()
                t0 = int(cta[:, :, 0].min())
                end = (cta[:, :, 1] - t0).double() / 1e3
                mend = (mg[:, 3] - t0).double() / 1e3
                recs.append(dict(last_arrival=float(end.max()), merge_end=float(mend.max()),
                                 rowstats_cyc=float(mg[:, 0].double().mean()),
                                 loop_cyc=float(mg[:, 1].double().mean()), tail_cyc=float(mg[:, 2].double().mean()),
                                 issue_cyc=float(mg[:, 4].double().mean()), wait_cyc=float(mg[:, 5].double().mean())))
            med = {k: float(np.median([x[k] for x in recs])) for k in recs[0]}
            out[f"{spec}:{b}x{m}"] = med
            xb.log(f"{spec} B={b} m={m}: " + " ".join(f"{k}={v:.2f}" for k, v in med.items()))
    print(json.dumps({"timeline2": out}), flush=True)


def mode_sass(a):
    base = xb.extract_base_tree(a.base_commit)
    tree = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    for spec in a.variants.split(","):
        cubin = make_variant(spec, tree, base)
        path = os.path.join(tempfile.mkdtemp(), f"{spec}.cubin")
        with open(path, "wb") as f:
            f.write(cubin)
        sass = subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True).stdout
        lines = [l for l in sass.splitlines() if "/*" in l and "*/" in l and l.strip().startswith("/*")]
        idx = [i for i, l in enumerate(lines) if "ATOM" in l and "INC" in l]
        start = idx[-1] if idx else max(0, len(lines) - a.sass_lines)
        xb.log(f"===== {spec}: {len(lines)} SASS instructions, merge section from {start} =====")
        for l in lines[start:start + a.sass_lines]:
            xb.log(l.split(";")[0].strip())


CONFIG_MATRIX = [
    ("serving", {}),
    ("h128_fp8", {"HEAD_ELEMS": "128"}),
    ("h64_fp16kv_fp16", {"HEAD_ELEMS": "64", "HEAD_GRP_SIZE": "4", "M_TILESIZE": "4", "CACHE_ELEM_ENUM": "0",
                         "INPUT_FP16": "1", "DTYPE": "__half", "ROPE_ELEMS": "64"}),
    ("h128_int8_g16", {"HEAD_ELEMS": "128", "HEAD_GRP_SIZE": "16", "M_TILESIZE": "16", "CACHE_ELEM_ENUM": "1"}),
    ("h128_bf16kv_g1", {"HEAD_ELEMS": "128", "HEAD_GRP_SIZE": "1", "M_TILESIZE": "1", "CACHE_ELEM_ENUM": "0"}),
    ("h256_g32", {"HEAD_GRP_SIZE": "32", "M_TILESIZE": "32"}),
    ("h256_beam4", {"BEAM_WIDTH": "4", "HEAD_GRP_SIZE": "4", "M_TILESIZE": "4"}),
    ("h128_specdec16", {"HEAD_ELEMS": "128", "SPEC_DEC": "1", "M_TILESIZE": "16", "SLIDING_WINDOW": "0",
                        "IS_SPEC_DEC_TREE": "1"}),
    ("h128_specdec32", {"HEAD_ELEMS": "128", "SPEC_DEC": "1", "M_TILESIZE": "32", "SLIDING_WINDOW": "1",
                        "IS_SPEC_DEC_TREE": "0"}),
    ("h256_nonpaged", {"TOKENS_PER_PAGE": "0"}),
    ("h256_page64_fp8out", {"TOKENS_PER_PAGE": "64", "LOW_PREC_OUTPUT": "1"}),
]
ARCHS = ["sm_120f", "sm_89", "sm_86", "sm_80", "sm_90a"]


def mode_compile_matrix(a):
    base = xb.extract_base_tree(a.base_commit)
    tree = os.path.join(os.getcwd(), "cpp/kernels/xqa")
    results = {}
    orig_macros = xb.serving_macros
    for spec in a.variants.split(","):
        if spec == "base":
            src_dir, patch = base, None
        elif spec == "tree":
            src_dir, patch = tree, None
        else:
            vp = variant_patch(spec)
            src_dir, patch = (tree if vp[0] else base), vp[1]
        for name, over in CONFIG_MATRIX:
            for arch in ARCHS:
                macros = dict(orig_macros())
                macros.update(over)
                xb.serving_macros = lambda m=macros: m
                key = f"{spec}:{name}:{arch}"
                try:
                    src = open(os.path.join(src_dir, "mha.cu")).read()
                    if patch is not None:
                        src = patch(src)
                    ok, msg = compile_arch(src_dir, src, arch)
                except Exception as e:  # noqa: BLE001
                    ok, msg = False, str(e)[-800:]
                results[key] = ok
                xb.log(f"{key}: {'OK' if ok else 'FAIL ' + msg}")
        xb.serving_macros = orig_macros
    print(json.dumps({"compile_matrix": results, "all_ok": all(results.values())}), flush=True)


def compile_arch(src_dir, src, arch):
    names, contents = [], []
    for h in xb.XQA_HEADERS:
        names.append(h.encode())
        with open(os.path.join(src_dir, h), "rb") as f:
            contents.append(f.read())
    for h in xb.CUDA_HEADERS:
        names.append(h.encode())
        with open(os.path.join(xb.CUDA_INC, h), "rb") as f:
            contents.append(f.read())
    prog = xb.ck(xb.nvrtc.nvrtcCreateProgram(src.encode(), b"default_program", len(names), contents, names), "create")
    opts = [b"-dw", b"--use_fast_math", b"-default-device", f"-arch={arch}".encode()]
    opts += [f"-D{k}={v}".encode() for k, v in xb.serving_macros().items()]
    err = xb.nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
    msg = ""
    if int(err) != 0:
        size = xb.ck(xb.nvrtc.nvrtcGetProgramLogSize(prog), "logsize")
        buf = b" " * size
        xb.nvrtc.nvrtcGetProgramLog(prog, buf)
        msg = buf.decode(errors="replace")[-1500:]
    xb.nvrtc.nvrtcDestroyProgram(prog)
    return int(err) == 0, msg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["explore", "timeline2", "sass", "compile_matrix"], default="explore")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--flush", choices=["read", "write", "none"], default="read")
    p.add_argument("--flush-mb", type=int, default=384)
    p.add_argument("--base-commit", default=xb.BASE_COMMIT)
    p.add_argument("--variants", default="base")
    p.add_argument("--timing-variant", default="timing2")
    p.add_argument("--configs", default="1x48")
    p.add_argument("--sass-lines", type=int, default=400)
    a = p.parse_args()
    if a.mode == "sass":
        mode_sass(a)
        return
    if a.mode == "compile_matrix":
        mode_compile_matrix(a)
        return
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    xb.ck(xb.cu.cuInit(0), "cuInit")
    if a.mode == "explore":
        xb.mode_explore(a)
    else:
        mode_timeline2(a)


if __name__ == "__main__":
    main()
