"""Embed + mHC stream seed microbenchmark / correctness check (DeepSeek-V4.1, one rank).

Runs the model's own forward fragment: DeepseekV4Model.forward on a stub self, from the embed_tokens
lookup through the mHC seed and DeepseekV41Model._init_hc_state, stopped right after the seed.
Base tree: masked 9-launch F.embedding + unsqueeze(1).repeat + zeros/fill pre_mix.
Candidate tree: one Triton gather writing [T, 4, 5120] (+ the fp32 one-hot pre_mix).

  --mode bench : decode CUDA-graph T=1/2/4/8, prefill eager T=2048/8192 (256 MB L2 flush),
                 prefill T=2048 with a concurrent num_sms//2-CTA _engram_lookup_kernel background.
  --mode check : untimed repeated poisoned iterations, fused vs masked reference, bitwise.
"""

import argparse
import inspect
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import torch

V, D, HC = 129280, 5120, 4
WARMUP, ITERS = 10, 200
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def log(*a):
    print(*a, flush=True)


def read_dims(ckpt):
    if not ckpt:
        return None
    path = os.path.join(ckpt, "config.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except Exception as e:  # noqa: BLE001
        log(f"[dims] cannot read {path}: {e}")
        return None
    found = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("vocab_size", "hidden_size", "hc_mult") and isinstance(v, int):
                    found.setdefault(k, v)
                walk(v)

    walk(cfg)
    log(f"[dims] checkpoint config: {found}")
    return found


from tensorrt_llm._torch.models import modeling_deepseekv4 as m4  # noqa: E402
from tensorrt_llm._torch.models import modeling_deepseekv41 as m41  # noqa: E402
from tensorrt_llm._torch.modules import embedding as emb_mod  # noqa: E402
from tensorrt_llm._torch.modules.embedding import Embedding  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

V4 = m4.DeepseekV4Model
V41 = m41.DeepseekV41Model
CTOR_SRC = inspect.getsource(V4.__init__)
TREE_HAS_EMBED_GATHER = "TRTLLM_DSV4_EMBED_GATHER" in CTOR_SRC
TREE_HAS_SEED_FAST = "TRTLLM_DSV4_HC_SEED_FAST" in CTOR_SRC
TREE = "candidate" if TREE_HAS_SEED_FAST else "base"


class _Stop(Exception):
    pass


def make_embedding(table, use_gather):
    e = Embedding(table.shape[0], table.shape[1], dtype=table.dtype, use_gather_kernel=use_gather)
    e.weight = torch.nn.Parameter(table, requires_grad=False)
    return e


def make_stub(table, env=None):
    """A stub self for DeepseekV4Model.forward, configured exactly as the tree's ctor would."""
    env = os.environ if env is None else env
    use_gather = TREE_HAS_EMBED_GATHER and env.get("TRTLLM_DSV4_EMBED_GATHER", "1") == "1"
    s = SimpleNamespace()
    s.embed_tokens = make_embedding(table, use_gather)
    s.hc_mult = HC
    s.model_config = SimpleNamespace(mapping=Mapping())
    s.use_engram = False
    s.engram_hash_provider = None
    if TREE_HAS_SEED_FAST:
        s._hc_seed_fast = (
            s.embed_tokens.use_gather_kernel
            and s.embed_tokens.tp_size == 1
            and env.get("TRTLLM_DSV4_HC_SEED_FAST", "1") == "1"
        )
    s._hc_seed_writes_pre_mix = getattr(V41, "_hc_seed_writes_pre_mix", False)
    s.result = None

    def init(hidden_states, **kw):
        s.result = V41._init_hc_state(s, hidden_states, **kw)
        raise _Stop

    s._init_hc_state = init
    return s


def fragment(stub, ids):
    try:
        V4.forward(stub, None, input_ids=ids)
    except _Stop:
        pass
    st = stub.result
    return st.residual, st.pre_mix


def reference(table, ids):
    """Protected reference: the unmodified masked lookup + repeat + one-hot seed."""
    x = emb_mod.pre_comm_embedding_ops(ids, table, 1, 0, None, 0, table.shape[0], False, 0)
    hs = x.unsqueeze(1).repeat(1, HC, 1)
    pre = hs.new_zeros((hs.shape[0], HC, 1), dtype=torch.float32)
    pre[:, 0, :] = 1.0
    return hs, pre


def sm_clock_sampler():
    samples = []
    stop = threading.Event()
    try:
        import pynvml

        pynvml.nvmlInit()
        h = None
        try:
            uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
            h = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        except Exception as e:  # noqa: BLE001
            log(f"[clock] uuid lookup failed ({e}); using nvml index 0")
            h = pynvml.nvmlDeviceGetHandleByIndex(0)

        def run():
            while not stop.is_set():
                try:
                    samples.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.05)
    except Exception as e:  # noqa: BLE001
        log(f"[clock] pynvml unavailable: {e}")

        def run():
            return

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return samples, stop


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1))]
    return {"median_us": med * 1e3, "p10_us": p10 * 1e3, "p90_us": p90 * 1e3,
            "noise_floor": (p90 - p10) / med}


def gen_ids(gen, n, t, dtype):
    return torch.randint(0, V, (n, t), generator=gen, dtype=torch.int64).to(dtype).cuda()


# ----------------------------------------------------------------------------------------------
def bench(args):
    torch.manual_seed(0)
    table = torch.randn((V, D), dtype=torch.bfloat16, device="cuda")
    stub = make_stub(table)
    log(f"[bench] tree={TREE} embed_gather={stub.embed_tokens.use_gather_kernel} "
        f"seed_fast={getattr(stub, '_hc_seed_fast', None)}")
    gen = torch.Generator().manual_seed(1)
    res = {}
    clocks, stop = sm_clock_sampler()

    # Decode: CUDA graph [elementwise dummy -> embed+seed -> elementwise consumer], PDL as served.
    for t in (1, 2, 4, 8):
        pristine = gen_ids(gen, 64, t, torch.int32)
        ids = pristine[0].clone()
        dummy = torch.randn(4096, device="cuda")
        for _ in range(3):  # Triton JIT + allocator warm before capture
            fragment(stub, ids)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fragment(stub, ids)
            with torch.cuda.graph(g, stream=s):
                dummy.mul_(1.0001)
                hs, pre = fragment(stub, ids)
                cons = hs * 2
                cons2 = pre + 1
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        ts = []
        for i in range(WARMUP + ITERS):
            ids.copy_(pristine[i % 64])
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            if i >= WARMUP:
                ts.append(a.elapsed_time(b))
        rh, rp = reference(table, ids)
        ok = (torch.equal(hs, rh) and torch.equal(pre, rp) and torch.equal(cons, rh * 2)
              and torch.equal(cons2, rp + 1))
        st = stats(ts)
        st["bitwise_ok"] = ok
        res[str(t)] = st
        log(f"[decode T={t}] {st}")
        del g

    # Prefill: eager, CUDA events, 256 MB L2 flush between iterations.
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for t in (2048, 8192):
        pristine = gen_ids(gen, 8, t, torch.int32)
        for _ in range(3):
            fragment(stub, pristine[0])
        torch.cuda.synchronize()
        ts = []
        for i in range(WARMUP + ITERS):
            ids = pristine[i % 8]
            flush.zero_()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            hs, pre = fragment(stub, ids)
            b.record()
            b.synchronize()
            if i >= WARMUP:
                ts.append(a.elapsed_time(b))
        rh, rp = reference(table, ids)
        st = stats(ts)
        st["bitwise_ok"] = torch.equal(hs, rh) and torch.equal(pre, rp)
        res[str(t)] = st
        log(f"[prefill T={t}] {st}")

    # Contention (context): the real _engram_lookup_kernel at num_sms//2 CTAs (background=True grid)
    # on a 2nd stream, on device-memory tables (stand-in for the UVA host tables). Span = embed
    # start -> both done.
    from tensorrt_llm._torch.modules.engram import engram as eng

    t = 2048
    heads, dim, qb, rows_tab = 8, 512, 128, 1 << 20
    w8 = torch.randn((rows_tab, dim), device="cuda").to(torch.float8_e4m3fn)
    sc = torch.full((rows_tab, dim // qb), 127, dtype=torch.uint8, device="cuda")
    eids = torch.randint(0, rows_tab // heads, (t, heads), device="cuda", dtype=torch.int64)
    offs = torch.arange(heads, device="cuda", dtype=torch.int64) * (rows_tab // heads)
    eout = torch.empty((t, heads, dim), dtype=torch.bfloat16, device="cuda")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    nrows = t * heads
    grid = min((nrows + eng._LOOKUP_BLOCK_R - 1) // eng._LOOKUP_BLOCK_R, max(1, num_sms // 2))

    def lookup():
        eng._engram_lookup_kernel[(grid,)](
            w8, sc, eids, offs, eout, 0, rows_tab, nrows, eids.stride(0), eids.stride(1), grid,
            HEAD_START=0, LOCAL_HEADS=heads, DIM=dim, DIM_PAD=dim, QB=qb,
            BLOCK_R=eng._LOOKUP_BLOCK_R, MASKED=False)

    bg = torch.cuda.Stream()
    pristine = gen_ids(gen, 8, t, torch.int32)
    lookup()
    torch.cuda.synchronize()
    ts = []
    for i in range(WARMUP + ITERS):
        ids = pristine[i % 8]
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        bdone = torch.cuda.Event()
        a.record()
        bg.wait_event(a)
        with torch.cuda.stream(bg):
            lookup()
            bdone.record()
        hs, pre = fragment(stub, ids)
        torch.cuda.current_stream().wait_event(bdone)
        b.record()
        b.synchronize()
        if i >= WARMUP:
            ts.append(a.elapsed_time(b))
    st = stats(ts)
    st["grid"] = grid
    rh, rp = reference(table, ids)
    st["bitwise_ok"] = torch.equal(hs, rh) and torch.equal(pre, rp)
    res[f"{t}_contended"] = st
    log(f"[contention T={t}] {st}")

    stop.set()
    clk = ({"min": min(clocks), "median": sorted(clocks)[len(clocks) // 2],
            "max": max(clocks), "n": len(clocks)} if clocks else None)
    log(f"[clock] sm MHz {clk}")
    out = {
        "tree": TREE,
        "dims": {"V": V, "D": D, "hc": HC},
        "pdl": os.environ.get("TRTLLM_ENABLE_PDL", "1"),
        "sm_clock_mhz": clk,
        "batch": {k: {"kernel_ms": v["median_us"] / 1e3, "noise_floor": v["noise_floor"],
                      "p10_ms": v["p10_us"] / 1e3, "p90_ms": v["p90_us"] / 1e3}
                  for k, v in res.items()},
        "all_bitwise_ok": all(v.get("bitwise_ok", True) for v in res.values()),
        "output": {"class": "exact"},
    }
    print(json.dumps(out), flush=True)


# ----------------------------------------------------------------------------------------------
def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in p.events() if e.device_type.name == "CUDA"]


def check(args):
    torch.manual_seed(0)
    table = torch.randn((V, D), dtype=torch.bfloat16, device="cuda")
    table_copy = table.clone()
    gen = torch.Generator().manual_seed(2)
    failures = []
    nchecked = 0
    max_abs = 0.0
    specials = [-1, V, V + 7, INT32_MIN, INT32_MAX, 0, V - 1]

    def ids_for(t, dtype, i):
        x = torch.randint(0, V, (t,), generator=gen, dtype=torch.int64)
        k = max(1, t // 4)
        pos = torch.randint(0, t, (k,), generator=gen)
        vals = torch.tensor(specials, dtype=torch.int64)[
            torch.randint(0, len(specials), (k,), generator=gen)]
        x[pos] = vals
        if t >= len(specials) and i == 0:
            x[: len(specials)] = torch.tensor(specials)
        return x.to(dtype).cuda()

    def compare(tag, hs, pre, rh, rp):
        nonlocal nchecked, max_abs
        nchecked += 1
        ok = (hs.dtype == rh.dtype and hs.shape == rh.shape and hs.stride() == rh.stride()
              and hs.is_contiguous() and torch.equal(hs, rh)
              and pre.dtype == rp.dtype and pre.shape == rp.shape and pre.stride() == rp.stride()
              and torch.equal(pre, rp))
        if not ok:
            err = float("inf")
            if hs.shape == rh.shape:
                d = (hs.float() - rh.float()).abs()
                err = float(torch.nan_to_num(d, nan=float("inf")).max())
            max_abs = max(max_abs, err)
            failures.append(f"{tag}: hs {tuple(hs.shape)}/{hs.stride()} vs {tuple(rh.shape)}/{rh.stride()}"
                            f" pre {tuple(pre.shape)}/{pre.stride()} err {err}")

    def poison_alloc(t, i):
        val = -7.25 if i % 2 == 0 else float("nan")
        a = torch.empty((t, HC, D), dtype=torch.bfloat16, device="cuda").fill_(val)
        b = torch.empty((t, HC, 1), dtype=torch.float32, device="cuda").fill_(val)
        c = torch.empty((t, D), dtype=torch.bfloat16, device="cuda").fill_(val)
        del a, b, c

    stub = make_stub(table)
    log(f"[check] tree={TREE} embed_gather={stub.embed_tokens.use_gather_kernel} "
        f"seed_fast={getattr(stub, '_hc_seed_fast', None)} dims V={V} D={D} hc={HC}")

    # 1) Eager: harness sequence (10 warmup + 200 iterations), poisoned outputs, fresh ids each
    #    iteration, PDL on/off, int32/int64.
    for pdl in ("1", "0"):
        os.environ["TRTLLM_ENABLE_PDL"] = pdl
        for dtype in (torch.int32, torch.int64):
            for t in (1, 2, 3, 4, 8, 37, 2048):
                for i in range(WARMUP + ITERS):
                    ids = ids_for(t, dtype, i)
                    poison_alloc(t, i)
                    hs, pre = fragment(stub, ids)
                    rh, rp = reference(table, ids)
                    compare(f"eager pdl={pdl} {dtype} T={t} it={i}", hs, pre, rh, rp)
                log(f"[check] eager pdl={pdl} {dtype} T={t} done, failures={len(failures)}")

    # 2) CUDA graph: capture (Triton JIT warmed before capture), poisoned static outputs before each
    #    replay, fresh ids each replay, PDL on/off.
    for pdl in ("1", "0"):
        os.environ["TRTLLM_ENABLE_PDL"] = pdl
        for dtype in (torch.int32, torch.int64):
            for t in (1, 2, 3, 4, 8, 37):
                ids = ids_for(t, dtype, 0)
                for _ in range(3):
                    fragment(stub, ids)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    fragment(stub, ids)
                    with torch.cuda.graph(g, stream=s):
                        hs, pre = fragment(stub, ids)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                for i in range(WARMUP + ITERS):
                    ids.copy_(ids_for(t, dtype, i))
                    hs.fill_(-7.25 if i % 2 == 0 else float("nan"))
                    pre.fill_(-7.25 if i % 2 == 0 else float("nan"))
                    g.replay()
                    rh, rp = reference(table, ids)
                    compare(f"graph pdl={pdl} {dtype} T={t} it={i}", hs, pre, rh, rp)
                del g
                log(f"[check] graph pdl={pdl} {dtype} T={t} done, failures={len(failures)}")
    os.environ["TRTLLM_ENABLE_PDL"] = "1"

    # 3) Kill switches: each must run its own path (kernel names) with bitwise outputs.
    switches = {}
    ids = ids_for(8, torch.int32, 0)
    rh, rp = reference(table, ids)
    for name, env in (("default", {}),
                      ("seed_fast_0", {"TRTLLM_DSV4_HC_SEED_FAST": "0"}),
                      ("embed_gather_0", {"TRTLLM_DSV4_EMBED_GATHER": "0"})):
        st = make_stub(table, env=dict(os.environ, **env))
        fragment(st, ids)
        names = kernel_names(lambda: fragment(st, ids))
        hs, pre = fragment(st, ids)
        compare(f"switch {name}", hs, pre, rh, rp)
        gather = sum("embedding_gather" in n for n in names)
        switches[name] = {"launches": len(names), "gather_launches": gather,
                          "use_gather": st.embed_tokens.use_gather_kernel,
                          "seed_fast": getattr(st, "_hc_seed_fast", None)}
        log(f"[check] switch {name}: {switches[name]} kernels={names}")
    if TREE == "candidate":
        if switches["default"]["launches"] != 1 or switches["default"]["gather_launches"] != 1:
            failures.append(f"fused path is not one gather launch: {switches['default']}")
        if switches["seed_fast_0"]["gather_launches"] != 1 or switches["seed_fast_0"]["launches"] <= 1:
            failures.append(f"HC_SEED_FAST=0 did not run gather + separate seed: {switches['seed_fast_0']}")
        if switches["embed_gather_0"]["gather_launches"] != 0:
            failures.append(f"EMBED_GATHER=0 still ran the gather: {switches['embed_gather_0']}")

    # 4) Other widths (sanity): D=7168 (multiple of 1024) and D=4100 (masked tail), direct call.
    if hasattr(emb_mod, "embedding_gather_hc"):
        for d, vv in ((7168, 32768), (4100, 4096)):
            tab = torch.randn((vv, d), dtype=torch.bfloat16, device="cuda")
            for t in (1, 3, 37, 2048):
                x = torch.randint(0, vv, (t,), generator=gen)
                x[0] = -1
                if t > 2:
                    x[1], x[2] = vv, INT32_MIN
                x = x.to(torch.int32).cuda()
                hs, pre = emb_mod.embedding_gather_hc(x, tab, HC, with_pre_mix=True)
                r = emb_mod.pre_comm_embedding_ops(x, tab, 1, 0, None, 0, vv, False, 0)
                rh2 = r.unsqueeze(1).repeat(1, HC, 1)
                rp2 = rh2.new_zeros((t, HC, 1), dtype=torch.float32)
                rp2[:, 0, :] = 1.0
                compare(f"D={d} T={t}", hs, pre, rh2, rp2)
                hs2, none = emb_mod.embedding_gather_hc(x, tab, HC, with_pre_mix=False)
                if none is not None or not torch.equal(hs2, rh2):
                    failures.append(f"D={d} T={t} no-pre_mix variant mismatch")
            log(f"[check] D={d} done, failures={len(failures)}")
        for t in (1, 37):
            x = torch.randint(0, V, (t,), generator=gen)
            x[0] = -1
            x = x.cuda()
            if not torch.equal(emb_mod.embedding_gather(x, table),
                               emb_mod.pre_comm_embedding_ops(x, table, 1, 0, None, 0, V, False, 0)):
                failures.append(f"embedding_gather hc=1 mismatch T={t}")

    table_ok = torch.equal(table, table_copy)
    if not table_ok:
        failures.append("embedding table modified")
    passed = (not failures) and nchecked > 0
    out = {"passed": passed, "tree": TREE,
           "compared_against": "masked F.embedding (pre_comm_embedding_ops) + unsqueeze(1).repeat(1,4,1) "
           "+ zeros/[:,0]=1 pre_mix, bitwise incl dtype/shape/strides",
           "comparisons": nchecked, "failures": failures[:20], "num_failures": len(failures),
           "max_abs_err": max_abs, "table_unchanged": table_ok, "switches": switches,
           "dims": {"V": V, "D": D, "hc": HC}}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "check"], required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    dims = read_dims(args.checkpoint)
    if dims:
        assert dims.get("vocab_size", V) == V, dims
        assert dims.get("hidden_size", D) == D, dims
        assert dims.get("hc_mult", HC) == HC, dims
    assert D == 5120
    log(f"[dims] running V={V} D={D} hc={HC} tree={TREE} torch={torch.__version__} "
        f"gpu={torch.cuda.get_device_name(0)}")
    if args.mode == "bench":
        bench(args)
    else:
        check(args)
        sys.exit(0)
