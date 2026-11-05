"""Branch 129 Stage 0: Cake DSv4.1 mixed-cache decode on CSA2's own pools (1 GPU, no model).

Parts (all in one job):
  census : job 1063 decode traces (c1, c4): per-layer-type us/launch of StageDequant,
           _split_sink_main, _split_sink_merge, StageCompact (layer type from the
           StageDequant grid: y = 640 -> width 640, y = 129 -> SWA-only width 128).
  build  : JIT build time of the Cake dsv41-mixed module, format info.
  format : CSA2-written pools; FI (Cake) vs FP64 and main split-sink chain vs FP64,
           counts 1/2/4/8 x width 640/128, holes, out-of-pool, empty rows, extremes.
           FlashInfer's own dsv41 pack writers vs CSA2's bytes (informative).
  timing : cold-L2 CUDA-graph replay per layer: main (StageDequant + main + merge,
           with/without StageCompact) vs Cake (with / without index prep).
"""

import argparse
import gzip
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ROOT = ("/home/scratch.chengzhang/trtllm-dev-dsv41-1017-c124-20261102-145209/runner/slots/0/"
        "jobs/90327323a0d447eda2bff999accf58c9/artifacts/profile")
NAMES = ("StageDequant", "_split_sink_main", "_split_sink_merge", "StageCompact")


def census():
    import glob

    res = {}
    for case in ("decode_c1", "decode_c4"):
        files = sorted(glob.glob(f"{ROOT}/traces/{case}/*rank-0.json.gz"))
        if not files:
            res[case] = {"error": f"no trace under {ROOT}/traces/{case}", "ls": _ls(ROOT)}
            continue
        with gzip.open(files[0], "rt") as f:
            data = json.load(f)
        events = data["traceEvents"] if isinstance(data, dict) else data
        kern = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op",
                                                                          "Kernel")]
        picked = {n: [] for n in NAMES}
        for e in kern:
            name = e.get("name", "")
            for n in NAMES:
                if n in name:
                    picked[n].append(e)
                    break
        for n in NAMES:
            picked[n].sort(key=lambda e: e["ts"])
        sd = picked["StageDequant"]
        out = {"file": os.path.basename(files[0]), "counts": {n: len(v) for n, v in picked.items()}}
        if not sd:
            res[case] = out
            continue
        out["example_names"] = {n: (v[0]["name"][:160] if v else None) for n, v in picked.items()}
        out["example_args"] = {n: ({k: v[0].get("args", {}).get(k) for k in ("grid", "block")}
                                   if v else None) for n, v in picked.items()}

        def width_of(e):
            grid = e.get("args", {}).get("grid")
            return None if grid is None else int(grid[1])

        # layer type for each StageDequant, then assign the others to the latest
        # StageDequant at or before them (StageCompact: the next one).
        sd_ts = [e["ts"] for e in sd]
        import bisect

        types = [width_of(e) for e in sd]
        per = {}
        for n in NAMES:
            for e in picked[n]:
                if n == "StageCompact":
                    i = bisect.bisect_left(sd_ts, e["ts"])
                else:
                    i = bisect.bisect_right(sd_ts, e["ts"]) - 1
                if i < 0 or i >= len(sd):
                    continue
                key = f"{n}/w{types[i]}"
                per.setdefault(key, []).append(float(e["dur"]))
        out["per_type_us"] = {
            k: dict(n=len(v), median=statistics.median(v), p10=sorted(v)[len(v) // 10],
                    p90=sorted(v)[(9 * len(v)) // 10]) for k, v in sorted(per.items())}
        # chain span per layer: StageDequant start -> merge end.
        merges = picked["_split_sink_merge"]
        m_ts = [e["ts"] for e in merges]
        spans = {}
        for i, e in enumerate(sd):
            j = bisect.bisect_left(m_ts, e["ts"])
            if j < len(merges) and (i + 1 >= len(sd) or merges[j]["ts"] < sd[i + 1]["ts"]):
                spans.setdefault(f"w{types[i]}", []).append(
                    merges[j]["ts"] + merges[j]["dur"] - e["ts"])
        out["chain_span_us"] = {k: dict(n=len(v), median=statistics.median(v))
                                for k, v in spans.items()}
        out["layers_per_step"] = {f"w{t}": types.count(t) / 40.0 for t in set(types)}
        res[case] = out
        C.log(case, json.dumps(out)[:3000])
    return res


def _ls(path):
    try:
        return os.listdir(path)[:20]
    except Exception as e:  # noqa: BLE001
        return str(e)


def build():
    from flashinfer.jit.cake_sparse_mla_sm120_dsv41_mixed import (
        cake_sparse_mla_sm120_dsv41_mixed_available,
    )
    from flashinfer.mla._sparse_mla_sm120 import cake_dsv41_mixed as cdm

    avail = bool(cake_sparse_mla_sm120_dsv41_mixed_available())
    t0 = time.time()
    cdm.get_cake_sparse_mla_sm120_dsv41_mixed_module()
    dt = time.time() - t0
    info = cdm.cake_sparse_mla_sm120_dsv41_mixed_format_info()
    info = {k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()}
    C.log(f"cake build/load {dt:.1f}s available={avail} info={info}")
    return dict(build_s=dt, kernels_available=avail, info=info)


def errs(x, ref):
    d = (x.double() - ref).abs()
    scale = ref.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    return dict(abs=d.max().item(), rel=(d / scale).max().item())


_POOLS = {}


def fmt_case(name, count, width, seed, extremes=False, empty_rows=()):
    if extremes not in _POOLS:
        _POOLS[extremes] = C.build_pools(31 + int(extremes), extremes=extremes)
    pools = _POOLS[extremes]
    sel = C.build_selection(count, seed + 1, width=width, empty_rows=empty_rows,
                            force_rows=list(range(36)) if extremes else None)
    q, sink = C.make_q(count, seed + 2)
    ref = C.reference(q, sink, *pools, sel)
    main = C.MainArm(count, width)
    cake = C.CakeArm(count, width)
    o_main = torch.empty_like(q)
    o_cake = torch.empty_like(q)
    main(q, sink, *pools, sel, o_main)
    cake(q, sink, *pools, sel, o_cake)
    o_cake2 = torch.empty_like(q)
    cake(q, sink, *pools, sel, o_cake2)
    torch.cuda.synchronize()
    e_main, e_cake = errs(o_main, ref), errs(o_cake, ref)
    live = torch.tensor([r not in empty_rows for r in range(count)], device=q.device)
    empty_zero = bool((o_cake[~live] == 0).all()) if (~live).any() else None
    r = dict(case=name, count=count, width=width, err_main=e_main, err_cake=e_cake,
             diff=(o_cake.float() - o_main.float()).abs().max().item(),
             finite=bool(torch.isfinite(o_cake).all()), empty_rows_zero=empty_zero,
             deterministic=bool(torch.equal(o_cake, o_cake2)),
             le_main_abs=e_cake["abs"] <= e_main["abs"], le_main_rel=e_cake["rel"] <= e_main["rel"],
             tol_1p6e2=bool(torch.allclose(o_cake.double(), ref, rtol=1.6e-2, atol=1.6e-2)))
    C.log("format", json.dumps(r))
    return r


def fi_writers(seed):
    """Informative: FlashInfer's dsv41 pack writers vs CSA2's QuantizeScatter bytes."""
    out = {}
    try:
        from flashinfer.mla import (
            dsv41_fp4_quantize_pack_sparse_mla_cache,
            dsv41_fp8_quantize_pack_sparse_mla_cache,
        )
        from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import store_rows
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}
    gen = torch.Generator().manual_seed(seed)
    n = 1024
    for fmt, fn, nbytes in (("swa", dsv41_fp8_quantize_pack_sparse_mla_cache, 528),
                            ("main", dsv41_fp4_quantize_pack_sparse_mla_cache, 288)):
        for large in (False, True):
            v = C.make_values(n, gen, large=large).cuda()
            if not large:
                v = (v.float() / 10).clamp(-1, 1).to(torch.bfloat16)
            pool = torch.zeros((n, nbytes), dtype=torch.uint8, device="cuda")
            store_rows(pool, torch.arange(n, device="cuda"), v, fmt)
            try:
                fi = fn(v.view(n, 1, 512)).reshape(n, nbytes)
                eq = (fi == pool)
                data_b = 512 if fmt == "swa" else 256
                out[f"{fmt}/large={large}"] = dict(
                    bytes_equal_frac=eq.float().mean().item(),
                    data_equal_frac=eq[:, :data_b].float().mean().item(),
                    scale_equal_frac=eq[:, data_b:].float().mean().item(),
                    fi_shape=list(fi.shape))
            except Exception as e:  # noqa: BLE001
                out[f"{fmt}/large={large}"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            C.log("fi_writer", fmt, large, out[f"{fmt}/large={large}"])
    return out


def timing(counts, widths):
    res = {}
    pools = C.build_pools(7)
    for width in widths:
        for count in counts:
            sel = C.build_selection(count, 100 + count, width=width)
            q, sink = C.make_q(count, 200 + count)
            out = torch.empty_like(q)
            row = {}
            main = C.MainArm(count, width)
            main(q, sink, *pools, sel, out)  # compaction for the no-compact variant
            st, g = C.time_graph(lambda: main(q, sink, *pools, sel, out, compact=False))
            row["main_no_compact"] = st
            del g
            st, g = C.time_graph(lambda: main(q, sink, *pools, sel, out, compact=True))
            row["main_with_compact"] = st
            del g
            cake = C.CakeArm(count, width)
            st, g = C.time_graph(lambda: cake(q, sink, *pools, sel, out))
            row["cake_with_prep"] = st
            del g
            nop = C.CakeNoPrepArm(count, width, sel)
            st, g = C.time_graph(lambda: nop(q, sink, *pools, sel, out))
            row["cake_no_prep"] = st
            row["cake_plan"] = nop.plan
            del g
            row["clock"] = C.sm_clock()
            key = f"w{width}/c{count}"
            res[key] = row
            C.log("timing", key, json.dumps(row))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", default="census,build,format,writers,timing")
    args = p.parse_args()
    parts = args.parts.split(",")
    C.log(f"device={torch.cuda.get_device_name()} parts={parts}")
    result = {}
    if "census" in parts:
        try:
            result["census"] = census()
        except Exception as e:  # noqa: BLE001
            result["census"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            C.log("census failed", result["census"])
    if "build" in parts:
        result["build"] = build()
    if "format" in parts:
        rows = []
        seed = 11
        for width in (640, 128):
            for count in (1, 2, 4, 8):
                empty = (1,) if count >= 2 else ()
                rows.append(fmt_case("normal", count, width, seed, empty_rows=empty))
                seed += 10
        rows.append(fmt_case("empty_c1", 1, 640, 901, empty_rows=(0,)))
        rows.append(fmt_case("empty_c1", 1, 128, 902, empty_rows=(0,)))
        for count in (1, 4):
            for width in (640, 128):
                rows.append(fmt_case("extremes", count, width, 950 + count + width, extremes=True))
        result["format"] = rows
        result["format_summary"] = dict(
            all_le_main_rel=all(r["le_main_rel"] for r in rows),
            all_le_main_abs=all(r["le_main_abs"] for r in rows),
            all_finite=all(r["finite"] for r in rows),
            all_deterministic=all(r["deterministic"] for r in rows),
            all_empty_zero=all(r["empty_rows_zero"] in (None, True) for r in rows),
            worst_rel_ratio=max(r["err_cake"]["rel"] / max(r["err_main"]["rel"], 1e-12)
                                for r in rows),
        )
        C.log("format_summary", result["format_summary"])
    if "writers" in parts:
        result["writers"] = fi_writers(5)
    if "timing" in parts:
        result["timing"] = timing((1, 2, 4, 8), (640, 128))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
