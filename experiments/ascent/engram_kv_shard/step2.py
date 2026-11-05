"""Idea 224 step 2 (4 GPUs): the Engram WKV decode path through the real module code.

Each of 4 MPI ranks (mpi4py.futures) builds EngramFp8Projection(6144, 25600) loaded through
its own load_weights from a fixed-seed E4M3 weight and [800,192] UE8M0 tile scale (identical
on every rank, as the checkpoint is), and a DeepseekV41Engram stand-in that carries only the
attributes the kv path reads (mapping, tp_size/rank, kv_proj). The kv path is
``engram._project_kv`` when the tree has it (candidate: sharded GEMM + TP all-gather at
T<=4) and ``engram.kv_proj`` otherwise (main: replicated).

Modes:
  pair   timing on either tree: per T in 1,2,4 one CUDA graph of the kv path over 8 module
         copies (1.26 GB of weights per rank, so every call streams cold from DRAM),
         30 replays x 3 sequences behind an MPI barrier; per-call median (max over ranks).
         Last line: evidence JSON {"batch": {T: {"kernel_ms", "noise_floor"}}}.
  check  candidate tree: for T in 1,2,4,8 graph capture of the kv path, outputs NaN-poisoned
         before each replay, every replay compared bitwise with the replicated kv_proj
         (30 replays on the fixed input + 30 replays with a new input copied in each time);
         a 30-iteration eager repeated-execution sequence; FlashInfer call counts
         (Mock wraps method._flashinfer_mxfp8) and all-gather call counts per T (T=8 and
         prefill T=2048 must not all-gather and must equal kv_proj bitwise); b12x kernel
         cache size before vs after the captures (compile-miss delta); in-graph split of the
         shard GEMM and the all-gather (+cat); persistent memory of the shard path.
"""

import json
import os
import pickle
import sys
import time

WORLD = 4
N_FULL, K = 25600, 6144
COPIES = int(os.environ.get("COPIES", "8"))
REPLAYS = 30
SEQS = 3
KEEP = []


def log(*a):
    print(*a, flush=True)


def _setup():
    import torch

    from tensorrt_llm._utils import mpi_rank
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm.mapping import Mapping
    mapping = Mapping(world_size=WORLD, rank=rank, gpus_per_node=WORLD, tp_size=WORLD)
    return torch, rank, mapping


def _barrier():
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()


def _make_engram(torch, mapping, rank, seed):
    from torch import nn

    from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41Engram
    from tensorrt_llm._torch.modules.engram.projection import EngramFp8Projection
    dev = torch.device("cuda")
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    w = (torch.randn((N_FULL, K), generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    tile = torch.randint(118, 127, (N_FULL // 32, K // 32), generator=g, device=dev,
                         dtype=torch.int32).to(torch.uint8)
    with torch.device(dev):
        proj = EngramFp8Projection(K, N_FULL, torch.bfloat16)
    proj.load_weights([{"weight": w, "scale": tile}])
    del w
    eng = DeepseekV41Engram.__new__(DeepseekV41Engram)
    nn.Module.__init__(eng)
    eng.mapping = mapping
    eng.tp_size = WORLD
    eng.tp_rank = rank
    eng.enable_attention_dp = False
    eng.layer_id = 1
    eng.kv_proj = proj
    return eng


def _kv_fn(eng):
    fn = getattr(eng, "_project_kv", None)
    return fn if fn is not None else eng.kv_proj


def _emb(torch, T, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(5000 + 31 * T + seed)
    return torch.randn((T, K), generator=g, device="cuda").to(torch.bfloat16)


def _capture(torch, body):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()  # warm on the capture stream
        torch.cuda.synchronize()
        _barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g, out


def _time_graph(torch, g, calls):
    seq_meds, flat = [], []
    for _ in range(SEQS):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPLAYS):
            _barrier()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) / calls)
        ts.sort()
        seq_meds.append(ts[len(ts) // 2])
        flat += ts
    flat.sort()
    n = len(flat)
    med = sorted(seq_meds)[len(seq_meds) // 2]
    return {"median_ms": med, "seq_medians_us": [round(x * 1e3, 3) for x in seq_meds],
            "noise_floor": (flat[(9 * n) // 10] - flat[n // 10]) / med}


def rank_pair(env):
    torch, rank, mapping = _setup()
    engs = [_make_engram(torch, mapping, rank, s) for s in range(COPIES)]
    KEEP.extend(engs)
    sharded = hasattr(engs[0], "_project_kv")
    res = {"rank": rank, "sharded_tree": sharded, "timing": {}}
    for T in (1, 2, 4):
        emb = _emb(torch, T, 0)
        KEEP.append(emb)
        fns = [_kv_fn(e) for e in engs]
        g, outs = _capture(torch, lambda: [f(emb) for f in fns])
        KEEP.append(outs)
        res["timing"][str(T)] = _time_graph(torch, g, len(fns))
        if rank == 0:
            t = res["timing"][str(T)]
            log(f"[pair sharded={sharded}] T={T} per-call {t['median_ms']*1e3:.2f} us "
                f"seqs {t['seq_medians_us']} noise {t['noise_floor']:.4f}")
        del g
    _barrier()
    return res


def rank_check(env):
    from unittest import mock

    torch, rank, mapping = _setup()
    import tensorrt_llm._torch.models.modeling_deepseekv41 as m41
    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _B12X_MM_MXFP8_KERNEL_CACHE
    mem0 = torch.cuda.memory_allocated()
    eng = _make_engram(torch, mapping, rank, 0)
    KEEP.append(eng)
    mem_load = torch.cuda.memory_allocated() - mem0
    method = eng.kv_proj.quant_method
    res = {"rank": rank, "backend": method.backend, "T": {}, "ok": True,
           "weight_bytes_loaded": mem_load}
    fi_mock = mock.Mock(wraps=method._flashinfer_mxfp8)
    method._flashinfer_mxfp8 = fi_mock
    ag_mock = mock.Mock(wraps=m41.allgather)
    m41.allgather = ag_mock
    ok = True
    # eager prewarm of every size (as the engine warmup does), then freeze the cache size
    for T in (1, 2, 4, 8):
        eng._project_kv(_emb(torch, T, 0))
        eng.kv_proj(_emb(torch, T, 0))
    torch.cuda.synchronize()
    cache0 = len(_B12X_MM_MXFP8_KERNEL_CACHE)
    res["shard_state"] = getattr(eng, "_kv_shard_state", None)
    for T in (1, 2, 4, 8):
        row = {}
        emb = _emb(torch, T, 0)
        KEEP.append(emb)
        ref = eng.kv_proj(emb)
        torch.cuda.synchronize()
        fi0, ag0 = fi_mock.call_count, ag_mock.call_count
        out_e = eng._project_kv(emb)
        torch.cuda.synchronize()
        row["eager_fi_calls"] = fi_mock.call_count - fi0
        row["eager_allgather_calls"] = ag_mock.call_count - ag0
        row["eager_bad"] = int((out_e.view(torch.int16) != ref.view(torch.int16)).sum().item())
        row["shape_ok"] = tuple(out_e.shape) == (T, N_FULL) and out_e.is_contiguous()
        expect_ag = 1 if T <= 4 else 0
        ok = ok and row["eager_bad"] == 0 and row["shape_ok"] and row["eager_fi_calls"] == 1 \
            and row["eager_allgather_calls"] == expect_ag
        # graph: fixed input
        mem_g0 = torch.cuda.memory_allocated()
        g, out = _capture(torch, lambda: eng._project_kv(emb))
        KEEP.append(out)
        row["graph_pool_bytes"] = torch.cuda.memory_allocated() - mem_g0
        bad_fixed, max_bad = 0, 0
        for _ in range(REPLAYS):
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            b = int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())
            bad_fixed += int(b > 0)
            max_bad = max(max_bad, b)
        # graph: new input every replay (reference recomputed with the replicated kv_proj)
        bad_new = 0
        max_abs = 0.0
        for i in range(REPLAYS):
            x = _emb(torch, T, 100 + i)
            emb.copy_(x)
            ref_i = eng.kv_proj(x)
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            b = int((out.view(torch.int16) != ref_i.view(torch.int16)).sum().item())
            if b:
                d = torch.nan_to_num((out.float() - ref_i.float()).abs(), nan=float("inf"))
                max_abs = max(max_abs, float(d.max().item()))
            bad_new += int(b > 0)
            max_bad = max(max_bad, b)
        emb.copy_(_emb(torch, T, 0))
        del g
        # repeated-execution (eager) sequence: 3 warmup + 30 iterations, each compared
        bad_rep = 0
        for i in range(33):
            o = eng._project_kv(emb)
            torch.cuda.synchronize()
            bad_rep += int(not torch.equal(o.view(torch.int16), ref.view(torch.int16)))
        row.update({"graph_replays_fixed": REPLAYS, "graph_bad_fixed": bad_fixed,
                    "graph_replays_new_input": REPLAYS, "graph_bad_new_input": bad_new,
                    "graph_max_bad_elems": max_bad, "graph_max_abs": max_abs,
                    "repeated_iters": 33, "repeated_bad": bad_rep})
        ok = ok and bad_fixed == 0 and bad_new == 0 and bad_rep == 0
        res["T"][str(T)] = row
        if rank == 0:
            log(f"[check] T={T}: {row}")
    res["b12x_cache_before_capture"] = cache0
    res["b12x_cache_after_capture"] = len(_B12X_MM_MXFP8_KERNEL_CACHE)
    ok = ok and res["b12x_cache_after_capture"] == cache0
    # prefill: replicated path, no all-gather, bitwise kv_proj
    x = _emb(torch, 2048, 0)
    ag0 = ag_mock.call_count
    o = eng._project_kv(x)
    r = eng.kv_proj(x)
    torch.cuda.synchronize()
    res["prefill_2048"] = {"allgather_calls": ag_mock.call_count - ag0,
                           "bad": int((o.view(torch.int16) != r.view(torch.int16)).sum().item())}
    ok = ok and res["prefill_2048"]["allgather_calls"] == 0 and res["prefill_2048"]["bad"] == 0
    # in-graph split at T=1/2/4: shard GEMM (quant+gemm) vs all-gather (+cat), cold weights
    engs = [eng] + [_make_engram(torch, mapping, rank, s) for s in range(1, COPIES)]
    KEEP.extend(engs)
    split = {}
    for T in (1, 2, 4):
        emb = _emb(torch, T, 0)
        KEEP.append(emb)
        tp = WORLD
        rows = N_FULL // tp

        def gemm_only(e):
            proj = e.kv_proj
            per = proj.weight_scale.numel() // N_FULL
            wv = proj.weight.narrow(0, rank * rows, rows)
            sv = proj.weight_scale.narrow(0, rank * rows * per, rows * per)
            a, a_sf = torch.ops.trtllm.mxfp8_quantize(emb, True)
            return proj.quant_method._flashinfer_mxfp8(a, a_sf, wv, sv, torch.bfloat16)

        locs = [gemm_only(e) for e in engs]
        torch.cuda.synchronize()
        KEEP.append(locs)

        def gather_only():
            outs = []
            for loc in locs:
                gth = m41.allgather(loc, mapping, dim=0)
                outs.append(gth.view(1, -1) if T == 1 else
                            gth.view(tp, T, rows).transpose(0, 1).reshape(T, tp * rows))
            return outs

        g1, o1 = _capture(torch, lambda: [gemm_only(e) for e in engs])
        g2, o2 = _capture(torch, gather_only)
        g3, o3 = _capture(torch, lambda: [e._project_kv(emb) for e in engs])
        KEEP.extend([o1, o2, o3])
        split[str(T)] = {"shard_gemm_us": round(_time_graph(torch, g1, COPIES)["median_ms"] * 1e3, 3),
                         "allgather_cat_us": round(_time_graph(torch, g2, COPIES)["median_ms"] * 1e3, 3),
                         "kv_path_us": round(_time_graph(torch, g3, COPIES)["median_ms"] * 1e3, 3)}
        if rank == 0:
            log(f"[check] split T={T}: {split[str(T)]}")
        del g1, g2, g3
    res["split"] = split
    res["ok"] = bool(ok)
    _barrier()
    return res


def run_ranks(fn, *args):
    import cloudpickle
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
    t0 = time.time()
    with MPIPoolExecutor(max_workers=WORLD) as ex:
        out = list(ex.map(fn, *[[a] * WORLD for a in args]))
    log(f"[driver] {fn.__name__} done in {time.time()-t0:.0f}s")
    return sorted(out, key=lambda r: r["rank"])


def main():
    import subprocess
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    log(subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,clocks.sm,temperature.gpu",
                        "--format=csv,noheader"], stdout=subprocess.PIPE, text=True).stdout)
    if mode == "pair":
        res = run_ranks(rank_pair, {})
        batch = {}
        for T in res[0]["timing"]:
            per = [r["timing"][T] for r in res]
            batch[T] = {"kernel_ms": max(p["median_ms"] for p in per),
                        "noise_floor": max(p["noise_floor"] for p in per),
                        "rank_medians_us": [round(p["median_ms"] * 1e3, 3) for p in per]}
        print(json.dumps({"batch": batch, "output": {"class": "exact"},
                          "sharded_tree": res[0]["sharded_tree"]}))
        return
    res = run_ranks(rank_check, {})
    ok = all(r["ok"] for r in res)
    summary = {
        "passed": ok,
        "compared": "sharded _project_kv (GEMM on rank rows + TP all-gather) vs the replicated "
                    "kv_proj on the same inputs, bitwise, every graph replay and eager iteration",
        "backend": res[0]["backend"], "shard_state": [r["shard_state"] for r in res],
        "T": {T: {k: (sum(r["T"][T][k] for r in res) if isinstance(v, int) and not isinstance(v, bool)
                      else v)
                  for k, v in res[0]["T"][T].items()} for T in res[0]["T"]},
        "b12x_cache": [(r["b12x_cache_before_capture"], r["b12x_cache_after_capture"]) for r in res],
        "prefill_2048": [r["prefill_2048"] for r in res],
        "split_us_rank0": res[0]["split"],
        "split_us_max_rank": {T: {k: max(r["split"][T][k] for r in res) for k in res[0]["split"][T]}
                              for T in res[0]["split"]},
        "weight_bytes_loaded_per_rank": res[0]["weight_bytes_loaded"],
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
