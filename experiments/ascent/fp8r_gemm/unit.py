"""Idea 122: image.0's FP8R block-step GEMM unit, routed (candidate) vs torch._scaled_mm (base).

Unit = gate_up (PreparedFP8Sm120.gemm, N18432 K3072) + QKV (flux3_fp8r_qkv.fp8r_qkv_qknorm_rope,
N9216 K3072, + its QK-norm/RoPE pass) + trtllm::flux3_sm120_fp8r_out_pair_q (down N3072 K9216 on
the current stream || attn_out quantize + N3072 K3072 on the side stream), at M=4096.
Every op is called exactly as the served code calls it. The candidate tree routes the 4 GEMMs
through flux3_fp8r_gemm (install() at "load"); the base tree has no such module.

  python unit.py bench   -> evidence JSON {"batch": {"1": {"kernel_ms", "noise_floor"}}, ...}
  python unit.py check   -> {"passed": ...}: routed vs torch._scaled_mm reference, bitwise, at
                            M=4096/4176/4256, poisoned (0xFFFF) allocator blocks, >=3 calls vs
                            call 0, CUDA-graph capture + 3 replays, kill switch in a subprocess.
"""

import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step0  # noqa: E402

log = step0.log
CALLS = 10
ROUNDS = int(os.environ.get("UNIT_ROUNDS", "100"))
HEADS = 24
HEAD_DIM = 128


def load_mods():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv, flux3_primsts, primsts

    try:
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg
    except ImportError:
        fg = None
    return primsts, flux3_primsts, flux3_fp8r_qkv, fg


class Block:
    """One FP8R block's prepared GEMM operands (as prepare_sm120_fp8_excluded_block builds them)."""

    def __init__(self, primsts, gen):
        dev = "cuda"
        w = lambda n, k: (torch.randn((n, k), device=dev, generator=gen) * 0.02).to(torch.bfloat16)
        self.qkv = primsts.PreparedFP8Sm120(w(3 * HEADS * HEAD_DIM, 3072))
        self.out = primsts.PreparedFP8Sm120(w(3072, 3072))
        self.up = primsts.PreparedFP8Sm120(w(18432, 3072), swiglu=True)
        self.down = primsts.PreparedFP8Sm120(w(3072, 9216))
        self.q_norm = (1 + 0.1 * torch.randn(HEAD_DIM, device=dev, generator=gen)).to(torch.bfloat16)
        self.k_norm = (1 + 0.1 * torch.randn(HEAD_DIM, device=dev, generator=gen)).to(torch.bfloat16)


class Inputs:
    def __init__(self, m, gen):
        from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r

        dev = "cuda"
        x = (torch.randn((m, 3072), device=dev, generator=gen) * 1.5).to(torch.bfloat16)
        x[:, :8] *= 20
        self.x_data, s = quantize_fp8r(x)
        self.x_data = self.x_data.contiguous()
        self.x_scales = s.float().reshape(-1).contiguous()
        h = (torch.randn((m, 9216), device=dev, generator=gen)).to(torch.bfloat16)
        self.h_data, s = quantize_fp8r(h)
        self.h_data = self.h_data.contiguous()
        self.h_scales = s.float().reshape(-1).contiguous()
        self.attn = (torch.randn((m, 3072), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
        ang = torch.rand((m, 64), device=dev, generator=gen) * 6.28
        self.cos_sin = torch.cat((ang.cos(), ang.sin()), dim=-1).float().contiguous()


def unit_fns(mods, blk, inp):
    primsts, flux3_primsts, flux3_fp8r_qkv, _ = mods

    def gate_up():
        return blk.up.gemm(inp.x_data, inp.x_scales)

    def qkv():
        # Looked up at call time, like _sm120_fp8r_qkv_forward does (rebound by install()).
        return flux3_fp8r_qkv.fp8r_qkv_qknorm_rope(
            inp.x_data, blk.qkv.weight, inp.x_scales, blk.qkv.scale,
            blk.q_norm, blk.k_norm, inp.cos_sin, flux3_fp8r_qkv._FP8R_QKV_PLANAR)

    def pair():
        return flux3_primsts.fp8r_out_pair_q(
            inp.h_data, inp.h_scales, blk.down.weight, blk.down.scale,
            inp.attn, blk.out.weight, blk.out.scale)

    return {"gate_up": gate_up, "qkv": qkv, "pair": pair}


def ref_fns(mods, blk, inp):
    """Protected reference: main's exact torch._scaled_mm calls (original ops, env-independent)."""
    _, flux3_primsts, _, _ = mods

    def mm(a, w, sa, sb):
        return torch._scaled_mm(a, w.t(), scale_a=sa.reshape(-1, 1), scale_b=sb.reshape(1, -1),
                                out_dtype=torch.bfloat16)

    def gate_up():
        return mm(inp.x_data, blk.up.weight, inp.x_scales, blk.up.scale)

    def qkv():
        return torch.ops.trtllm.flux3_sm120_fp8r_qkv_qknorm_rope(
            inp.x_data, blk.qkv.weight, inp.x_scales, blk.qkv.scale,
            blk.q_norm, blk.k_norm, inp.cos_sin, False)

    def pair():
        from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r

        d, s = quantize_fp8r(inp.attn)
        a = mm(d.contiguous(), blk.out.weight, s.float().reshape(-1).contiguous(), blk.out.scale)
        y = mm(inp.h_data, blk.down.weight, inp.h_scales, blk.down.scale)
        return a, y

    return {"gate_up": gate_up, "qkv": qkv, "pair": pair}


def install(mods):
    fg = mods[3]
    if fg is not None:
        routed = fg.install(torch.device("cuda", torch.cuda.current_device()))
        log(f"[install] flux3_fp8r_gemm routed={routed} arm={fg.ARM}")
    else:
        log("[install] base tree: no flux3_fp8r_gemm module")


def as_tuple(o):
    return o if isinstance(o, tuple) else (o,)


def bits_equal(a, b):
    return all(x.shape == y.shape and torch.equal(x.view(torch.int16), y.view(torch.int16))
               for x, y in zip(as_tuple(a), as_tuple(b)))


def time_unit(fn_sets, nsets):
    """Graph of CALLS calls of fn (rotating over nsets input sets), per-call us stats."""
    return step0.time_graph([fn_sets[i % nsets] for i in range(CALLS)], rounds=ROUNDS)


def bench():
    step0.header()
    mods = load_mods()
    install(mods)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(122)
    nsets = 6
    blocks = [Block(mods[0], gen) for _ in range(nsets)]
    inps = [Inputs(4096, gen) for _ in range(nsets)]
    fns = [unit_fns(mods, b, i) for b, i in zip(blocks, inps)]
    res = {}
    # A/B/A-style: each op timed in two interleaved repeats; the unit in 3 repeats.
    for rep in range(2):
        for name in ("gate_up", "qkv", "pair"):
            t = time_unit([f[name] for f in fns], nsets)
            res.setdefault(name, []).append(t)
            log(f"[bench] {name} rep{rep} med {t['med']:.1f} p10 {t['p10']:.1f} p90 {t['p90']:.1f} us")

    def unit_call(f):
        def run():
            return f["gate_up"](), f["qkv"](), f["pair"]()
        return run

    units = []
    for rep in range(3):
        t = time_unit([unit_call(f) for f in fns], nsets)
        units.append(t)
        log(f"[bench] unit rep{rep} med {t['med']:.1f} p10 {t['p10']:.1f} p90 {t['p90']:.1f} us")
    med = sorted(u["med"] for u in units)[1]
    noise = max((u["p90"] - u["p10"]) / u["med"] for u in units)
    fg = mods[3]
    if fg is not None:
        log(f"[bench] routes by N {dict(fg.ROUTES)}")
    out = {"batch": {"1": {"kernel_ms": med / 1000.0, "noise_floor": noise}},
           "output": {"class": "exact"},
           "per_op_us": {k: [round(t["med"], 2) for t in v] for k, v in res.items()},
           "unit_us": [round(u["med"], 2) for u in units],
           "routed": fg is not None and fg.enabled()}
    print(json.dumps(out), flush=True)


def poison():
    """Fill freed caching-allocator blocks with 0xFF so a fresh output that is not fully written
    shows NaN-pattern bytes."""
    junk = torch.full((512 << 20,), 0xFF, dtype=torch.uint8, device="cuda")
    del junk
    sizes = [4096 * 18432 * 2, 4096 * 9216 * 2, 4256 * 18432 * 2, 4256 * 9216 * 2]
    for s in sizes:
        t = torch.full((s,), 0xFF, dtype=torch.uint8, device="cuda")
        del t


def check():
    step0.header()
    mods = load_mods()
    fg = mods[3]
    if fg is None:
        print(json.dumps({"passed": False, "error": "no flux3_fp8r_gemm module"}))
        return
    install(mods)
    _, _, flux3_fp8r_qkv, _ = mods
    ok = flux3_fp8r_qkv.fp8r_qkv_qknorm_rope is fg.fp8r_qkv_qknorm_rope_cg
    report = {"qkv_rebound": ok}
    passed = ok
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2122)
    blk = Block(mods[0], gen)
    for m in (4096, 4176, 4256):
        inp = Inputs(m, gen)
        cand, ref = unit_fns(mods, blk, inp), ref_fns(mods, blk, inp)
        for name in ("gate_up", "qkv", "pair"):
            r = ref[name]()
            torch.cuda.synchronize()
            outs = []
            before = fg.ROUTES.copy()
            for it in range(4):  # call 0 + 3 repeats, each on poisoned allocator blocks
                poison()
                outs.append(tuple(t.clone() for t in as_tuple(cand[name]())))
            torch.cuda.synchronize()
            routes = {k: fg.ROUTES[k] - before.get(k, 0) for k in fg.ROUTES}
            vs_ref = [bits_equal(o, r) for o in outs]
            vs_0 = [bits_equal(o, outs[0]) for o in outs[1:]]
            nan = any(torch.isnan(t.float()).any().item() for o in outs for t in o)
            # Graph capture + 3 replays.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                cand[name]()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                go = cand[name]()
            graph_ok = []
            for _ in range(3):
                g.replay()
                torch.cuda.synchronize()
                graph_ok.append(bits_equal(go, r))
            del g, go
            row = {"vs_ref": vs_ref, "vs_call0": vs_0, "graph_replays": graph_ok, "nan": nan,
                   "routes_by_N": routes}
            row_ok = all(vs_ref) and all(vs_0) and all(graph_ok) and not nan
            report[f"{name}/M{m}"] = row
            passed = passed and row_ok
            log(f"[check] {name} M{m} ok={row_ok} {row}")
    # Kill switch: a fresh process with TRTLLM_FLUX3_FP8R_GEMM=0.
    env = dict(os.environ, TRTLLM_FLUX3_FP8R_GEMM="0")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "killswitch"], env=env,
                       capture_output=True, text=True, timeout=900)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
    last = (lines or ["{}"])[-1]
    try:
        ks = json.loads(last)
    except json.JSONDecodeError:
        ks = {"passed": False, "stdout_tail": r.stdout[-2000:], "stderr_tail": r.stderr[-2000:]}
    report["killswitch"] = ks
    passed = passed and bool(ks.get("passed"))
    log(f"[check] killswitch {ks}")
    print(json.dumps({"passed": bool(passed), "compared": "routed FP8R ops vs main's "
                      "torch._scaled_mm path (bitwise)", "report": report}), flush=True)


def killswitch():
    mods = load_mods()
    fg = mods[3]
    install(mods)
    _, _, flux3_fp8r_qkv, _ = mods
    orig = flux3_fp8r_qkv.fp8r_qkv_qknorm_rope is not fg.fp8r_qkv_qknorm_rope_cg
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    blk = Block(mods[0], gen)
    inp = Inputs(4096, gen)
    cand, ref = unit_fns(mods, blk, inp), ref_fns(mods, blk, inp)
    eq = {n: bits_equal(cand[n](), ref[n]()) for n in cand}
    torch.cuda.synchronize()
    built = fg.module.cache_info().currsize
    res = {"passed": orig and all(eq.values()) and sum(fg.ROUTES.values()) == 0 and built == 0,
           "enabled": fg.enabled(), "qkv_op_original": orig, "bitwise_vs_torch": eq,
           "routes": dict(fg.ROUTES), "module_built": built}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    {"bench": bench, "check": check, "killswitch": killswitch}[sys.argv[1]]()
    log(f"done {time.time() - t0:.1f}s")
