"""Idea 89: SwiGLU-FP4 Pingpong-128 gate_up epilogue v2 (flux3_fp4_gemm_swiglu_fp4_sm120.cu).

FLUX3_EPI_V2=1 (on top of FLUX3_EPI_VEC=1) uses the hierarchical N permutation
((2,4),2,8):((1,16),64,2) so lane q owns the consecutive output pairs 8q..8q+7 of its warp's
32 outputs per row: amax = 1 shuffle, lane-local e2m1 packing, 1 word shuffle -> one STG.64 and one
scale byte per lane per (mi). Env TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2=0 = the served _dyn_ev module.

Synthetic tensors at served shapes (gate_up N=18432, K=3072; M=16464/16384/32848, eager 4176).
  step0 : ablation table, one process, seeded random arm order per round: served A/A (_dyn_ev
          twice), NOEPI, NOMATH, NOSHFL, NOSTORE, and v2 (+ v2 NOSHFL / NOSTORE); per arm
          registers, local bytes and epilogue SASS counts.
  check : bitwise v2 vs _dyn_ev (0xFF-poisoned whole padded out/out_sf), repeated calls vs
          call 0, inputs unchanged, graph t2i->i2i->t2i sequence with eager 4176 between, env.
  pair  : evidence: the tree's default module (get_module()) us/call at the three served Ms.
"""

import concurrent.futures as cf
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as swc

H, FF = 3072, 9216
N = 2 * FF
SERVED_MS = (16464, 16384, 32848)
CALLS = int(os.environ.get("EPI_V2_CALLS", "5"))
ROUNDS = int(os.environ.get("EPI_V2_ROUNDS", "20"))

BASE = ("-DFLUX3_TILESCHED=1", "-DFLUX3_EPI_VEC=1")
V2 = BASE + ("-DFLUX3_EPI_V2=1",)
VARIANTS = {
    "served": ("_dyn_ev", BASE),
    "noepi": ("_dyn_ev_d_noepi", BASE + ("-DFLUX3_DIAG_NOEPI",)),
    "nomath": ("_dyn_ev_d_nomath", BASE + ("-DFLUX3_DIAG_NOMATH",)),
    "noshfl": ("_dyn_ev_d_noshfl", BASE + ("-DFLUX3_DIAG_NOSHFL",)),
    "nostore": ("_dyn_ev_d_nostore", BASE + ("-DFLUX3_DIAG_NOSTORE",)),
    "v2": ("_dyn_ev2", V2),
    "v2_nomath": ("_dyn_ev2_d_nomath", V2 + ("-DFLUX3_DIAG_NOMATH",)),
    "v2_noshfl": ("_dyn_ev2_d_noshfl", V2 + ("-DFLUX3_DIAG_NOSHFL",)),
    "v2_nostore": ("_dyn_ev2_d_nostore", V2 + ("-DFLUX3_DIAG_NOSTORE",)),
}


def log(*a):
    print(*a, flush=True)


def git_log():
    wt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    r = subprocess.run(["git", "-C", wt, "log", "-3", "--oneline"], capture_output=True, text=True)
    log("git log -3:\n" + r.stdout.strip())
    return r.stdout.strip().splitlines()


def prebuild(pairs):
    def _b(p):
        try:
            swc.gen_spec(*p).build()
            return p[0], "ok"
        except Exception as e:  # noqa: BLE001
            return p[0], f"{type(e).__name__}: {str(e)[-3000:]}"

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for n, msg in ex.map(_b, pairs):
            log(f"prebuilt {n}: {msg}")


def build(names):
    """Build the named variants concurrently (separate JIT names), then load them."""
    specs = {n: swc.gen_spec(*VARIANTS[n]) for n in names}

    def _b(n):
        try:
            specs[n].build()
            return n, None
        except Exception as e:  # noqa: BLE001
            return n, f"{type(e).__name__}: {str(e)[-2000:]}"

    with cf.ThreadPoolExecutor(max_workers=min(4, len(names))) as ex:
        for n, err in ex.map(_b, names):
            log(f"built {n}: {'ok' if err is None else err}")
    mods = {}
    for n in names:
        mods[n] = swc._build(*VARIANTS[n])
        log(f"loaded {n}: stages={mods[n].flux3_swiglu_fp4_stages()} smem={mods[n].flux3_swiglu_fp4_smem_bytes()}")
    return mods, specs


def _lib_path(spec):
    for attr in ("jit_library_path", "library_path"):
        p = getattr(spec, attr, None)
        if p is not None:
            return str(p)
    from flashinfer.jit import env as jit_env
    hits = glob.glob(str(jit_env.FLASHINFER_JIT_DIR / spec.name / "*.so"))
    return hits[0] if hits else None


CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
SASS_OPS = ("SHFL", "STG.E.128", "STG.E.64", "STG.E.U16", "STG.E.U8", "STG.E ", "MUFU", "F2FP",
            "OMMA", "HMMA", "QMMA", "STL", "LDL", "FMUL", "FFMA", "FMNMX", "PRMT", "SEL", "BRA")


def resources(spec):
    so = _lib_path(spec)
    row = {"so": so}
    if not so or not os.path.exists(so):
        return row
    r = subprocess.run([CUOBJDUMP, "--dump-resource-usage", so], capture_output=True, text=True)
    lines = r.stdout.splitlines()
    for i, ln in enumerate(lines):
        if "device_kernel" in ln and i + 1 < len(lines):
            res = lines[i + 1]
            for key in ("REG", "STACK", "SHARED", "LOCAL"):
                m = re.search(rf"{key}:(\d+)", res)
                if m:
                    row[key] = int(m.group(1))
    s = subprocess.run([CUOBJDUMP, "-sass", so], capture_output=True, text=True).stdout
    cur, body = None, {}
    for ln in s.splitlines():
        if "Function :" in ln:
            cur = ln.split("Function :")[1].strip()
            body[cur] = []
        elif cur is not None and "/*" in ln:
            body[cur].append(ln)
    kern = [k for k in body if "device_kernel" in k]
    if kern:
        lines = body[kern[0]]
        counts = {op: 0 for op in SASS_OPS}
        for ln in lines:
            text = ln.split("*/", 1)[-1] if ln.strip().startswith("/*") else ln
            for op in SASS_OPS:
                if f" {op}" in text or f"\t{op}" in text:
                    counts[op] += 1
        row["sass"] = counts
        row["sass_instr"] = len(lines)
        # Normalized SASS hash (instruction text only, no addresses/comments).
        norm = "\n".join(re.sub(r"/\*[0-9a-f]{4,}\*/", "", ln).strip() for ln in lines)
        row["sass_sha256"] = hashlib.sha256(norm.encode()).hexdigest()[:16]
    return row


def make_case(M, seed=1234):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(N, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    s_x = float(x.abs().amax().float() / (448.0 * 6.0))
    s_w = float(w.abs().amax().float() / (448.0 * 6.0))
    a, sfa = fp.nvfp4_quantize(x, torch.tensor([1.0 / s_x], device="cuda", dtype=torch.float32))
    b, sfb = fp.nvfp4_quantize(w, torch.tensor([1.0 / s_w], device="cuda", dtype=torch.float32))
    alpha = torch.tensor([s_x * s_w], device="cuda", dtype=torch.float32)
    inv = torch.tensor([1.0 / 0.05], device="cuda", dtype=torch.float32)
    del x, w
    return {"M": M, "a": a, "b": b, "sfa": sfa, "sfb": sfb, "alpha": alpha, "inv": inv}


KEYS = ("a", "b", "sfa", "sfb", "alpha", "inv")


def call(c, module, bufs, poison=True):
    o, s = bufs
    if poison:
        o.fill_(0xFF)
        s.fill_(0xFF)
    swc.swiglu_packed(c["a"], c["b"], c["sfa"], c["sfb"], c["alpha"], c["inv"], module=module,
                      output=o, output_scale=s)
    return o, s


_STREAM = []


def capture(fn):
    if not _STREAM:
        _STREAM.append(torch.cuda.Stream())
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=_STREAM[0], capture_error_mode="thread_local"):
        fn()
    torch.cuda.synchronize()
    return g


def time_arms(c, mods, rounds, seed):
    """{arm: [us/call per round]}: one CALLS-call graph per arm, seeded random order per round."""
    graphs = {}
    for arm, mod in mods.items():
        bufs = swc.allocate_outputs(c["M"], N, c["a"].device)
        graphs[arm] = capture(lambda mod=mod, bufs=bufs: [call(c, mod, bufs, poison=False)
                                                          for _ in range(CALLS)])
    rng = random.Random(seed)
    res = {arm: [] for arm in mods}
    for g in graphs.values():  # warm
        g.replay()
    torch.cuda.synchronize()
    for r in range(rounds):
        order = list(mods)
        rng.shuffle(order)
        for arm in order:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[arm].replay()
            e1.record()
            e1.synchronize()
            res[arm].append(e0.elapsed_time(e1) * 1000.0 / CALLS)
    del graphs
    torch.cuda.synchronize()
    return res


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    p10, p90 = xs[int(0.1 * (n - 1))], xs[int(round(0.9 * (n - 1)))]
    return {"median_us": round(med, 2), "p10": round(p10, 2), "p90": round(p90, 2),
            "noise": round((p90 - p10) / med, 4)}


def clocks():
    r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def step0():
    gl = git_log()
    names = list(VARIANTS)
    mods, specs = build(names)
    res_rows = {n: resources(specs[n]) for n in names}
    for n in names:
        log(f"resources {n}: {json.dumps(res_rows[n])}")
    arms = dict(mods)
    arms["served_aa"] = mods["served"]
    table = {}
    for M in SERVED_MS:
        c = make_case(M)
        log(f"M={M} clocks before: {clocks()}")
        t = time_arms(c, arms, ROUNDS, seed=M)
        log(f"M={M} clocks after: {clocks()}")
        st = {arm: stats(v) for arm, v in t.items()}
        base = st["served"]["median_us"]
        for arm in st:
            st[arm]["vs_served"] = round(st[arm]["median_us"] / base - 1.0, 4)
        table[str(M)] = st
        log(f"M={M}: " + json.dumps(st))
        del c
        torch.cuda.empty_cache()
    print(json.dumps({"git": gl, "table_us_per_call": table, "resources": res_rows}))


def bitwise(x, y):
    return torch.equal(x, y)


def check():
    gl = git_log()
    prebuild([("_dyn_ev", BASE), ("_dyn_ev2", V2), ("_ev", ("-DFLUX3_EPI_VEC=1",)),
              ("_ev2", ("-DFLUX3_EPI_VEC=1", "-DFLUX3_EPI_V2=1"))])
    ref_mod = swc.get_module_dynamic(True, False)  # served _dyn_ev
    cand = swc.get_module_dynamic(True, True)       # _dyn_ev2
    st_ref = swc.get_module_static(True, False)
    st_cand = swc.get_module_static(True, True)
    assert cand is not ref_mod
    passed, results = True, {}
    WARM, ITERS = 2, 5
    cases = {}
    for M in SERVED_MS + (4176,):
        c = make_case(M, seed=M + 7)
        cases[M] = c
        pristine = {k: c[k].clone() for k in KEYS}
        bufs = swc.allocate_outputs(M, N, c["a"].device)
        ref = [t.clone() for t in call(c, ref_mod, bufs)]
        torch.cuda.synchronize()
        c["ref"] = ref
        row = {}
        for arm, mod, r in (("v2_dynamic", cand, ref), ("v2_static", st_cand, None),
                            ("ref_dynamic_repeat", ref_mod, ref)):
            if r is None:
                r = [t.clone() for t in call(c, st_ref, bufs)]
                torch.cuda.synchronize()
                row["static_ev_vs_dyn_ev"] = all(bitwise(x, y) for x, y in zip(r, ref))
            for k in KEYS:
                c[k].copy_(pristine[k])
            for _ in range(WARM):
                call(c, mod, bufs)
            first, n_ref, n_first = None, 0, 0
            for _ in range(ITERS):
                out = [t.clone() for t in call(c, mod, bufs)]
                torch.cuda.synchronize()
                if first is None:
                    first = out
                n_ref += all(bitwise(x, y) for x, y in zip(out, r))
                n_first += all(bitwise(x, y) for x, y in zip(out, first))
            inputs_ok = all(torch.equal(pristine[k], c[k]) for k in KEYS)
            ok = n_ref == ITERS and n_first == ITERS and inputs_ok
            if not ok and n_ref < ITERS:
                o = out[0].view(-1) != r[0].view(-1)
                s = out[1].view(-1) != r[1].view(-1)
                row[f"{arm}_mismatch"] = {"out_bytes": int(o.sum()), "sf_bytes": int(s.sum()),
                                          "first_out_idx": int(o.nonzero()[0]) if o.any() else -1}
            passed &= ok
            row[arm] = {"bitwise_vs_dyn_ev": f"{n_ref}/{ITERS}", "vs_call0": f"{n_first}/{ITERS}",
                        "inputs_unchanged": inputs_ok, "ok": ok}
        passed &= row.get("static_ev_vs_dyn_ev", False)
        # Sanity: the poisoned regions are fully written (no 0xFF-only rows in the real rows).
        row["out_all_written"] = bool((ref[0][:M] != 0xFF).any(dim=1).all())
        results[f"M{M}"] = row
        log(f"M={M}: {json.dumps(row)}")

    # Graph sequence like serving: t2i graph (16464 single + 16384 mode), i2i graph
    # (32848 single + 16384 mode), alternating t2i -> i2i -> t2i x2, eager 4176 between,
    # outputs poisoned before every replay, compared with _dyn_ev's references.
    pristine = {M: {k: cases[M][k].clone() for k in KEYS} for M in cases}
    gbufs = {M: swc.allocate_outputs(M, N, cases[M]["a"].device) for M in cases}

    def run_graph_fn(Ms):
        return lambda: [call(cases[M], cand, gbufs[M], poison=False) for M in Ms]

    graphs = {"t2i": (capture(run_graph_fn((16464, 16384))), (16464, 16384)),
              "i2i": (capture(run_graph_fn((32848, 16384))), (32848, 16384))}
    seq = ["t2i", "i2i", "t2i", "i2i", "t2i"]
    mism, n_cmp = 0, 0
    for step, name in enumerate(seq):
        g, Ms = graphs[name]
        for M in Ms:
            gbufs[M][0].fill_(0xFF)
            gbufs[M][1].fill_(0xFF)
        g.replay()
        torch.cuda.synchronize()
        for M in Ms:
            n_cmp += 1
            ok = all(bitwise(x, y) for x, y in zip(gbufs[M], cases[M]["ref"]))
            mism += 0 if ok else 1
        e = call(cases[4176], cand, gbufs[4176])
        torch.cuda.synchronize()
        n_cmp += 1
        mism += 0 if all(bitwise(x, y) for x, y in zip(e, cases[4176]["ref"])) else 1
        for M in cases:
            if not all(torch.equal(pristine[M][k], cases[M][k]) for k in KEYS):
                mism += 1
                log(f"inputs changed at step {step} M={M}")
    del graphs
    results["graph_sequence"] = {"sequence": seq, "comparisons": n_cmp, "mismatches": mism}
    passed &= mism == 0
    log(f"graph sequence: {results['graph_sequence']}")

    # Env selection: EPI_V2=0 -> the _dyn_ev module (same name/flags as main's served module).
    env_rows = {}
    for v2 in ("0", "1", None, "bogus"):
        env = dict(os.environ)
        env.pop("TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2", None)
        env.pop("TRTLLM_FLUX3_SM120_SWIGLU_EPI_VEC", None)
        env.pop("TRTLLM_FLUX3_SM120_SWIGLU_SCHED", None)
        if v2 is not None:
            env["TRTLLM_FLUX3_SM120_SWIGLU_EPI_V2"] = v2
        code = ("import json,tensorrt_llm\n"
                "from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as s\n"
                "try:\n m=s.get_module()\n names={n: s._build(*a) is m for n,a in "
                "{'dyn_ev':('_dyn_ev',('-DFLUX3_TILESCHED=1','-DFLUX3_EPI_VEC=1')),"
                "'dyn_ev2':('_dyn_ev2',('-DFLUX3_TILESCHED=1','-DFLUX3_EPI_VEC=1','-DFLUX3_EPI_V2=1'))}.items()}\n"
                " print(json.dumps({'module': [n for n,v in names.items() if v]}))\n"
                "except ValueError as e:\n print(json.dumps({'error': str(e)[:120]}))\n")
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        line = (r.stdout.strip().splitlines() or ["{}"])[-1]
        env_rows[str(v2)] = json.loads(line) if line.startswith("{") else {"raw": r.stderr[-300:]}
        log(f"env EPI_V2={v2}: {env_rows[str(v2)]}")
    env_ok = (env_rows["0"].get("module") == ["dyn_ev"] and env_rows["1"].get("module") == ["dyn_ev2"]
              and env_rows["None"].get("module") == ["dyn_ev2"] and "error" in env_rows["bogus"])
    results["env_selection"] = {"rows": env_rows, "ok": env_ok}
    passed &= env_ok
    spec_ev = swc.gen_spec("_dyn_ev", BASE)
    results["dyn_ev_resources"] = resources(spec_ev)
    results["dyn_ev2_resources"] = resources(swc.gen_spec("_dyn_ev2", V2))
    print(json.dumps({"passed": bool(passed), "git": gl,
                      "reference": "served _dyn_ev module (FLUX3_TILESCHED=1, FLUX3_EPI_VEC=1, same "
                                   "module name and flags as main) on pristine inputs; whole padded "
                                   "out/out_sf after 0xFF poison, bitwise",
                      "results": results}))


def pair():
    gl = git_log()
    mod = swc.get_module()
    spec_ev = swc.gen_spec("_dyn_ev", BASE)
    swc._build("_dyn_ev", BASE)
    ev_res = resources(spec_ev)
    log(f"_dyn_ev resources in this tree: {json.dumps(ev_res)}")
    batch = {}
    for M in SERVED_MS:
        c = make_case(M)
        t = time_arms(c, {"default": mod, "default_aa": mod}, ROUNDS * 2, seed=M)
        s, s2 = stats(t["default"]), stats(t["default_aa"])
        log(f"M={M}: {s} A/A {s2} clocks {clocks()}")
        batch[str(M)] = {"kernel_ms": round(s["median_us"] / 1000.0, 5), "noise_floor": s["noise"],
                         "aa_kernel_ms": round(s2["median_us"] / 1000.0, 5)}
        del c
        torch.cuda.empty_cache()
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "git": gl,
                      "dyn_ev_sass_sha256": ev_res.get("sass_sha256"),
                      "dyn_ev_reg": ev_res.get("REG")}))


if __name__ == "__main__":
    {"step0": step0, "check": check, "pair": pair}[sys.argv[1]]()
