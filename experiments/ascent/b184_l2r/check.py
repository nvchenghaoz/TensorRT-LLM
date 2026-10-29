"""Branch 184 (idea 244): untimed correctness of stage D (L2 discard in VEC mode 2) and stage R
(FP8R QKV -> qknorm_rope_fp8 on packed views -> prequant FMHA), synthetic tensors, no model.

  op    : qkv8 + scales[8] bitwise, discard on (q/k, and q/k/v) vs L2DISCARD off (main's kernel),
          planar [3, M, 3072] and packed [M, 9216] sources, M 4096/4176/4256/4100, plain, inf/NaN
          and zero-plane cases; keep_v=True leaves v bitwise unchanged; 50 repeats == call 0 with
          the source restored before every call (as the GEMM rewrites it); inputs other than the
          dead planes unchanged; CUDA graph (source restore + op) x3 == eager.
  r     : image.0 chain, S4096 and S4176: FMHA output of B (direct, discard off/on) bitwise vs
          main's chain (fp8r_qkv_qknorm_rope + visual_gen_fp8_attention_sm120); qkv8 / scales[0:4]
          bitwise vs fp8_sm12x.quantize_qkv_e4m3 on main's normed BF16 planes; v column
          bitwise with keep_v; 50 repeats == call 0; graph x3 == eager.
  res   : cuobjdump resource usage (REG/STACK/LOCAL, STL/LDL) of the VEC kernels (CUTE_DSL
          cubin dump) per (mode, discard).
  sass  : diagnostic for run_gpu_pair: sha of the VEC mode 1/2 SASS with the levers off
          (base tree and branch tree must print the same hashes).
Last stdout line: JSON with passed.
"""

import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

DUMP = tempfile.mkdtemp(prefix="b184_cute_")
os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", DUMP)

import torch  # noqa: E402

from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x  # noqa: E402
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)

H, HEADS, D = 3072, 24, 128
SM = 1.0 / math.sqrt(D)
fails = []


def log(*a):
    print(*a, flush=True)


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)
    return ok


def same(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype in (torch.float8_e4m3fn,):
        return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def make_src(M, packed, special, g):
    if packed:
        buf = torch.randn(M, 3 * H, device="cuda", generator=g).bfloat16() * 3
        q, k, v = buf[:, :H], buf[:, H:2 * H], buf[:, 2 * H:]
    else:
        buf = (torch.randn(3, M, H, device="cuda", generator=g) * 3).bfloat16()
        q, k, v = buf[0], buf[1], buf[2]
    if special == "infnan":
        q[5, 7] = float("inf")
        k[M - 1, 100] = float("nan")
        v[3, 2000] = float("-inf")
        v[M // 2, 11] = float("nan")
    elif special == "zero_k":
        k.zero_()
    elif special == "zero_v":
        v.zero_()
    elif special == "nan_q":
        q.fill_(float("nan"))
    theta = torch.rand(M, 64, device="cuda", generator=g) * 6.28
    cs = torch.cat((theta.cos(), theta.sin()), -1).float().contiguous()
    qn = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).bfloat16()
    kn = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).bfloat16()
    pos = torch.arange(M, device="cuda", dtype=torch.int64)
    return buf, (q, k, v, qn, kn, cs, pos)


def run_op(ins, discard, keep_v):
    old = qr8.L2DISCARD
    qr8.L2DISCARD = discard
    try:
        return qr8.qknorm_rope_fp8(*ins, SM, discard_qk=True, keep_v=keep_v)
    finally:
        qr8.L2DISCARD = old


def op_suite(results):
    """Run the op cases under main's default STATIC_QK (on) and with it off."""
    default = qr8.STATIC_QK
    try:
        for static in (default, not default):
            qr8.STATIC_QK = static
            sub = {}
            op_suite_one(sub)
            for k, v in sub.items():
                results[f"{k}{'' if static else '__static_off'}"] = v
    finally:
        qr8.STATIC_QK = default


def op_suite_one(results):
    g = torch.Generator(device="cuda").manual_seed(244)
    recs = []
    cases = []
    for packed in (False, True):
        for M in (4096, 4176, 4256, 4100):
            cases.append((M, packed, "plain"))
        for special in ("infnan", "zero_k", "zero_v", "nan_q"):
            cases.append((4176, packed, special))
    for M, packed, special in cases:
        buf, ins = make_src(M, packed, special, g)
        pristine = buf.clone()
        aux = [t.clone() for t in ins[3:]]
        r8, rs = run_op(ins, False, True)  # main's kernel (lever off)
        note(same(buf, pristine), f"lever-off leaves source unchanged M{M} {special}")
        rec = {"M": M, "packed": packed, "special": special}
        for keep_v in (True, False):
            buf.copy_(pristine)
            c8, cs8 = run_op(ins, True, keep_v)
            tag = "keep_v" if keep_v else "drop_v"
            rec[f"{tag}_qkv8"] = same(c8, r8)
            rec[f"{tag}_scales8"] = same(cs8, rs)
            if keep_v:
                rec["keep_v_v_unchanged"] = same(ins[2], pristine[..., 2 * H:] if packed
                                                 else pristine[2])
            rec[f"{tag}_aux_unchanged"] = all(same(a, b) for a, b in zip(ins[3:], aux))
        ok = all(v for k, v in rec.items() if isinstance(v, bool) and k != "packed")
        rec["ok"] = note(ok, f"op case {rec}")
        recs.append(rec)
        log("op case", json.dumps(rec))
    results["op_cases"] = recs
    # 50 repeats == call 0 (source restored before each call, as the GEMM rewrites it)
    for packed in (False, True):
        buf, ins = make_src(4176, packed, "plain", g)
        pristine = buf.clone()
        ref8, refs = run_op(ins, False, True)
        n_ref = n_first = 0
        first = None
        for it in range(52):
            buf.copy_(pristine)
            o8, os_ = run_op(ins, True, False)
            if it < 2:
                continue
            if first is None:
                first = (o8.clone(), os_.clone())
            n_ref += same(o8, ref8) and same(os_, refs)
            n_first += same(o8, first[0]) and same(os_, first[1])
        key = f"repeats50_{'packed' if packed else 'planar'}"
        results[key] = {"vs_lever_off": f"{n_ref}/50", "vs_call0": f"{n_first}/50"}
        note(n_ref == 50 and n_first == 50, key)
        log(key, results[key])
        # graph: restore + op, replay x3 vs eager
        torch.cuda.synchronize()
        qr8.L2DISCARD = True
        buf.copy_(pristine)
        qr8.qknorm_rope_fp8(*ins, SM, discard_qk=True, keep_v=False)  # warm compile
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            buf.copy_(pristine)
            g8, gs = qr8.qknorm_rope_fp8(*ins, SM, discard_qk=True, keep_v=False)
        torch.cuda.synchronize()
        gok = True
        for _ in range(3):
            g8.view(torch.uint8).fill_(0xFF)
            gs.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            gok &= same(g8, ref8) and same(gs, refs)
        del graph
        qr8.L2DISCARD = False
        key = f"graph_x3_{'packed' if packed else 'planar'}"
        results[key] = gok
        note(gok, key)
        log(key, gok)


def r_suite(results):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import step0 as s0

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv as f8r  # noqa: F401

    for S, seed in ((4096, 31), (4176, 32)):
        c = s0.make_fp8r_case(S, seed)
        pristine = {k: v.clone() for k, v in c.items() if isinstance(v, torch.Tensor)}
        qr8.L2DISCARD = False
        ref_o, ref_v = s0.fp8r_main(c)
        ref_o, ref_v = ref_o.clone(), ref_v.clone()
        normed = torch.ops.trtllm.flux3_sm120_fp8r_qkv_qknorm_rope(
            c["act"], c["w"], c["a_s"], c["w_s"], c["qn"], c["kn"], c["cs"], False)
        q, k, v = (t.reshape(1, S, H).view(1, S, HEADS, D) for t in normed.split(H, -1))
        q8, k8, v8, sc4 = fp8_sm12x.quantize_qkv_e4m3(q, k, v, SM)
        rec = {"S": S}
        for disc in (False, True):
            qr8.L2DISCARD = disc
            o, vv, qkv8, sc = s0.fp8r_direct(c, keep_v=True)
            tag = "discard" if disc else "nodiscard"
            rec[f"{tag}_fmha_out_bitwise_vs_main"] = same(o, ref_o)
            rec[f"{tag}_qkv8_vs_K1K2"] = (same(qkv8[0], q8.reshape(S, H)) and
                                          same(qkv8[1], k8.reshape(S, H)) and
                                          same(qkv8[2], v8.reshape(S, H)))
            rec[f"{tag}_scales4_vs_K1K2"] = same(sc[:4], sc4)
            rec[f"{tag}_v_bitwise_keep_v"] = same(vv, ref_v)
        qr8.L2DISCARD = True
        n_first = n_ref = 0
        first = None
        for it in range(52):
            o = s0.fp8r_direct(c, keep_v=False)[0]
            if it < 2:
                continue
            if first is None:
                first = o.clone()
            n_ref += same(o, ref_o)
            n_first += same(o, first)
        rec["repeats50_vs_main"] = f"{n_ref}/50"
        rec["repeats50_vs_call0"] = f"{n_first}/50"
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            go = s0.fp8r_direct(c, keep_v=False)[0]
        torch.cuda.synchronize()
        gok = True
        for _ in range(3):
            go.zero_()
            graph.replay()
            torch.cuda.synchronize()
            gok &= same(go, ref_o)
        del graph
        qr8.L2DISCARD = False
        rec["graph_x3_vs_main"] = gok
        rec["inputs_unchanged"] = all(same(pristine[k], c[k]) for k in pristine)
        ok = all(v for k, v in rec.items() if isinstance(v, bool))
        ok &= n_ref == 50 and n_first == 50
        rec["ok"] = note(ok, f"r case S{S}")
        results[f"r_S{S}"] = rec
        log("r case", json.dumps(rec))


def cubins():
    return sorted(glob.glob(os.path.join(DUMP, "**", "*.cubin"), recursive=True))


def resource(path):
    out = {"cubin": os.path.basename(path)}
    try:
        r = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", path],
                           capture_output=True, text=True, timeout=120)
        out["functions"] = [m[:90] for m in re.findall(r"Function ([^\s:]+)", r.stdout)]
        out["REG"] = [int(m) for m in re.findall(r"REG:(\d+)", r.stdout)]
        out["STACK"] = [int(m) for m in re.findall(r"STACK:(\d+)", r.stdout)]
        out["LOCAL"] = [int(m) for m in re.findall(r"LOCAL:(\d+)", r.stdout)]
        s = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", path], capture_output=True,
                           text=True, timeout=300)
        out["STL"] = len(re.findall(r"\bSTL\b", s.stdout))
        out["LDL"] = len(re.findall(r"\bLDL\b", s.stdout))
        out["DISCARD_SASS"] = len(re.findall(r"CCTL|DISCARD", s.stdout))
        body = [ln.split("*/", 1)[-1].strip() for ln in s.stdout.splitlines() if "/*" in ln
                and re.search(r"/\*[0-9a-f]{4}\*/", ln)]
        body = [re.sub(r"0x[0-9a-f]+", "X", b) for b in body]
        out["sass_lines"] = len(body)
        out["sass_sha"] = hashlib.sha256("\n".join(body).encode()).hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    return out


def compile_variants(variants):
    """Compile (heads 24) VEC mode 1/2 for planar and packed src_ld; returns {label: cubin}."""
    g = torch.Generator(device="cuda").manual_seed(1)
    seen = {}
    for packed in (False, True):
        buf, ins = make_src(4176, packed, "plain", g)
        for label, discard, keep_v in variants:
            before = set(cubins())
            if discard is None:
                qr8.qknorm_rope_fp8(*ins, SM)
            else:
                run_op(ins, discard, keep_v)
            torch.cuda.synchronize()
            new = sorted(set(cubins()) - before)
            seen[f"{'packed' if packed else 'planar'}_{label}"] = new
    return seen


def res_suite(results):
    compile_variants([("off", False, True), ("qk", True, True), ("qkv", True, False)])
    results["resource"] = [resource(p) for p in cubins()]
    results["dump_dir_files"] = sorted(os.path.relpath(p, DUMP) for p in glob.glob(
        os.path.join(DUMP, "**", "*"), recursive=True))[:60]
    log("resource", json.dumps(results["resource"]))


def sass():
    # Lever-off compile in either tree; print per-cubin sha (function-name-free SASS).
    seen = compile_variants([("default", None, None)])
    out = {k: [resource(p) for p in v] for k, v in seen.items()}
    print(json.dumps({"passed": True, "diagnostic": "sass", "levers_env":
                      os.environ.get("TRTLLM_FLUX3_QKV_FP8_L2DISCARD"), "sass": out}))


if __name__ == "__main__":
    torch.manual_seed(0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "sass":
        sass()
        sys.exit(0)
    results = {}
    t0 = time.time()
    if mode in ("op", "all"):
        op_suite(results)
        log(f"op suite {time.time() - t0:.0f}s")
    if mode in ("r", "all"):
        r_suite(results)
        log(f"r suite {time.time() - t0:.0f}s")
    if mode in ("res", "all"):
        res_suite(results)
    print(json.dumps({
        "passed": not fails, "failures": fails,
        "reference": "lever off in-process (main's VEC kernel; main's image.0 chain "
                     "fp8r_qkv_qknorm_rope + visual_gen_fp8_attention_sm120; fp8_sm12x K1/K2)",
        "results": results,
    }))
