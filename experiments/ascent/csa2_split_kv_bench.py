"""CSA2 SM120 decode attention (CSA2FlashInfer._forward_pools): split-KV microbenchmark and check.

Shapes: one TP4 rank. q [count, 16, 512] bf16, SWA pool [count, 128, 512] and extra
(top-k) pool [count, 512, 512] bf16 (width 640), prepared lengths int32 in [1, 640]
(dense prefix), sinks [16] fp32, sm_scale 1/sqrt(512). Counts 1/2/4/8 (the server's
padded decode graphs).

Modes
  time   : production _forward_pools (Triton stage_rows_and_mask + FA2 [+ merge] + Triton
           sink_epilogue) and wrapper.run only (FA2 [+ merge]), each under CUDA-graph
           replay with cold L2. Runs in either tree; prints the evidence JSON
           (kernel_ms = full _forward_pools per call).
  sweep  : candidate tree only. Same timing for split rows off/64/128/320/auto, plus
           max abs error vs an FP64 sink-attention reference.
  check  : candidate tree only. Untimed correctness: FP64 reference vs split path and
           vs the disable_split_kv=True path at lengths incl. 0,1,63,64,65,127,128,129,
           639,640; packed mask / mask_indptr byte identity between split and off plans;
           merged LSE finite (no NaN) for len > 0; repeated graph replays (same warmup/
           iters/reset points as timing) and replays with changing inputs vs eager.
"""

import argparse
import json
import math
import sys

import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer

HEADS, DIM, SWA, EXTRA = 16, 512, 128, 512
WIDTH = SWA + EXTRA
SCALE = 1.0 / math.sqrt(DIM)
COUNTS = (1, 2, 4, 8)
WARMUP, ITERS = 10, 100
HAS_SPLIT = hasattr(CSA2FlashInfer, "_SPLIT_KV_ROWS")
SPECIAL_LENS = (1, 640, 64, 65, 63, 127, 128, 129, 639)


def log(*a):
    print(*a, flush=True)


def make_inputs(count, seed, lens=None, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((count, HEADS, DIM), generator=g).to(torch.bfloat16)
    swa = torch.randn((count, SWA, DIM), generator=g).to(torch.bfloat16)
    extra = torch.randn((count, EXTRA, DIM), generator=g).to(torch.bfloat16)
    if lens is None:
        lens = torch.randint(1, WIDTH + 1, (count,), generator=g)
        lens[0] = 1
        if count > 1:
            lens[-1] = WIDTH
    else:
        lens = torch.tensor(lens, dtype=torch.int64)
    sink = torch.randn((HEADS,), generator=g) * 2.0
    return [
        t.to(device)
        for t in (q, swa, extra, lens.to(torch.int32), sink.float())
    ]


def reference(q, swa, extra, lens, sink):
    kv = torch.cat((swa, extra), dim=1).double()
    valid = torch.arange(WIDTH, device=q.device)[None, :] < lens[:, None]
    s = torch.einsum("bhd,bkd->bhk", q.double(), kv) * SCALE
    s = s.masked_fill(~valid[:, None, :], -math.inf)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
    o = torch.einsum("bhk,bkd->bhd", p, kv)
    out = o * torch.sigmoid(lse - sink.double()[None, :])[..., None]
    return torch.where(valid.any(-1)[:, None, None], out, 0.0)


def new_fi(split_rows="default"):
    fi = CSA2FlashInfer()
    if split_rows != "default":
        if not HAS_SPLIT:
            raise RuntimeError("split rows need the candidate tree")
        fi._SPLIT_KV_ROWS = split_rows  # instance override of the class policy
    return fi


def split_rows_of(fi, count):
    if not HAS_SPLIT:
        return None
    return fi.split_kv_rows(count, HEADS, WIDTH, DIM)


def run_once(fi, q, swa, extra, lens, sink, out=None):
    return fi._forward_pools(q, swa, extra, lens, sink, SCALE, out=out)


FLUSH = None


def time_graph(fn, iters=ITERS, warmup=WARMUP):
    global FLUSH
    if FLUSH is None:
        FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup):
        FLUSH.zero_()
        graph.replay()
    for i in range(iters):
        FLUSH.zero_()
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
    med = times[len(times) // 2]
    p10, p90 = times[len(times) // 10], times[(9 * len(times)) // 10]
    return med, (p90 - p10) / med, graph, out


def time_count(fi, count, seed):
    # Re-create pools and lengths from the same seed before each timed sequence.
    q, swa, extra, lens, sink = make_inputs(count, seed)
    out = torch.empty_like(q)
    full_ms, full_noise, g1, res = time_graph(lambda: run_once(fi, q, swa, extra, lens, sink, out))
    plan = fi._plans[(count, HEADS, WIDTH, q.device, SCALE)]
    kv = torch.cat((swa, extra), dim=1)
    valid = torch.arange(WIDTH, device=q.device)[None, :] < lens[:, None]
    rows = torch.where(valid[..., None], kv, 0).reshape(-1, 1, DIM).contiguous()
    plan.packed_mask.copy_(fi.pack_query_masks(valid))
    k_ms, k_noise, g2, _ = time_graph(lambda: plan.wrapper.run(q, rows, rows, return_lse=True))
    ref = reference(q, swa, extra, lens, sink)
    g1.replay()
    torch.cuda.synchronize()
    err = (res.double() - ref).abs().max().item()
    finite = bool(torch.isfinite(res).all())
    del g1, g2
    return dict(
        full_ms=full_ms,
        full_noise=full_noise,
        kernel_ms=k_ms,
        kernel_noise=k_noise,
        max_abs_err=err,
        finite=finite,
        split_rows=split_rows_of(fi, count),
    )


def mode_time(args):
    batch = {}
    for count in COUNTS:
        r = time_count(new_fi(), count, args.seed + count)
        log(f"count {count}: {r}")
        batch[str(count)] = {
            "kernel_ms": r["full_ms"],
            "noise_floor": r["full_noise"],
            "fa2_plus_merge_ms": r["kernel_ms"],
            "fa2_plus_merge_noise": r["kernel_noise"],
            "max_abs_err_vs_fp64": r["max_abs_err"],
            "finite": r["finite"],
            "split_rows": r["split_rows"],
        }
    print(json.dumps({"batch": batch, "output": {"class": "reordered" if HAS_SPLIT else "exact"}}))


def mode_sweep(args):
    res = {}
    for rows in (None, 64, 128, 320, 0):
        for count in COUNTS:
            fi = new_fi(rows)
            fi._SPLIT_KV_MAX_COUNT = 64
            r = time_count(fi, count, args.seed + count)
            log(f"rows {rows} count {count}: {r}")
            res[f"{rows}/{count}"] = r
    print(json.dumps({"sweep": res}))


def mode_check(args):
    ok = True
    report = {}
    # (0) plan layout: the packed-mask buffer and mask_indptr are byte-identical
    # between the split and the disable_split_kv=True plans (stage_rows_and_mask
    # writes them after plan()).
    for count in COUNTS:
        a = new_fi(None)._ragged_plan(count, HEADS, WIDTH, DIM, torch.device("cuda"), SCALE)
        b = new_fi()._ragged_plan(count, HEADS, WIDTH, DIM, torch.device("cuda"), SCALE)
        same = (
            torch.equal(a.packed_mask, b.packed_mask)
            and a.packed_mask.shape == b.packed_mask.shape
            and torch.equal(a.wrapper._mask_indptr_buf.cpu(), b.wrapper._mask_indptr_buf.cpu())
            and b.wrapper._custom_mask_buf.data_ptr() == b.packed_mask.data_ptr()
        )
        ok &= same
        report[f"layout/{count}"] = dict(
            mask_bytes=b.packed_mask.numel(), split_rows=b.split_rows, off_rows=a.split_rows,
            identical=same,
        )
        log(f"layout/{count}", report[f"layout/{count}"])
    # (1) accuracy at special lengths incl. fully masked chunks and empty selections.
    for count in COUNTS:
        for trial in range(4):
            if trial == 0:
                lens = [SPECIAL_LENS[i % len(SPECIAL_LENS)] for i in range(count)]
            elif trial == 1:
                lens = [SPECIAL_LENS[(i + 2) % len(SPECIAL_LENS)] for i in range(count)]
                lens[0] = 0  # empty selection
            elif trial == 2:
                lens = [64 if i % 2 == 0 else 65 for i in range(count)]
            else:
                lens = None
            q, swa, extra, ln, sink = make_inputs(count, args.seed + 100 * count + trial, lens)
            ref = reference(q, swa, extra, ln, sink)
            off = run_once(new_fi(None), q, swa, extra, ln, sink)
            fi = new_fi()
            new = run_once(fi, q, swa, extra, ln, sink)
            # Merged LSE of the split plan (staged by the last call).
            plan = fi._plans[(count, HEADS, WIDTH, q.device, SCALE)]
            kv = torch.cat((swa, extra), dim=1)
            valid = torch.arange(WIDTH, device=q.device)[None, :] < ln[:, None]
            rows = torch.where(valid[..., None], kv, 0).reshape(-1, 1, DIM).contiguous()
            o_raw, lse = plan.wrapper.run(q, rows, rows, return_lse=True)
            torch.cuda.synchronize()
            live = ln > 0
            lse_ok = bool(torch.isfinite(lse[live]).all()) and not bool(torch.isnan(lse).any())
            raw_ok = bool(torch.isfinite(o_raw[live]).all())
            # base-2 LSE: compare with the FP64 natural-log LSE / ln2.
            s = torch.einsum("bhd,bkd->bhk", q.double(), kv.double()) * SCALE
            s = s.masked_fill(~valid[:, None, :], -math.inf)
            ref_lse2 = torch.logsumexp(s, -1) / math.log(2.0)
            lse_err = (lse.double()[live] - ref_lse2[live]).abs().max().item() if live.any() else 0.0
            e_off = (off.double() - ref).abs().max().item()
            e_new = (new.double() - ref).abs().max().item()
            d = (new.float() - off.float()).abs().max().item()
            fin = bool(torch.isfinite(new).all())
            zero_ok = bool((new[~live] == 0).all())
            tol_ok = torch.allclose(new.double(), ref, rtol=1.6e-2, atol=1.6e-2)
            # "no worse": allow 25% / one BF16 ulp of slack on the reference error.
            good = (
                fin and tol_ok and lse_ok and raw_ok and zero_ok and lse_err < 1e-3
                and e_new <= max(e_off * 1.25, e_off + 2 ** -8)
                and new.shape == q.shape and new.stride() == q.stride()
            )
            ok &= good
            k = f"acc/{count}/{trial}"
            report[k] = dict(
                lens=ln.tolist(), split_rows=split_rows_of(fi, count), err_split=e_new,
                err_off=e_off, diff_vs_off=d, finite=fin, merged_lse_finite=lse_ok,
                merged_o_finite=raw_ok, lse_base2_max_err=lse_err, empty_rows_zero=zero_ok,
                passed=good,
            )
            log(k, report[k])
    # (2) repeated graph replays with the timing loop's warmup/iters and reset points.
    for count in COUNTS:
        fi = new_fi()
        q, swa, extra, ln, sink = make_inputs(count, args.seed + count)
        ref = reference(q, swa, extra, ln, sink)
        out = torch.empty_like(q)
        eager = run_once(fi, q, swa, extra, ln, sink).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_once(fi, q, swa, extra, ln, sink, out)
        errs, rels = [], []
        bit = True
        for _ in range(WARMUP + ITERS):
            graph.replay()
            torch.cuda.synchronize()
            diff = (out.double() - ref).abs()
            errs.append(diff.max().item())
            rels.append((diff / ref.abs().clamp_min(1e-2)).max().item())
            bit &= torch.equal(out, eager)
        good = bit and max(errs) <= 1.6e-2 and bool(torch.isfinite(out).all())
        ok &= good
        report[f"replay/{count}"] = dict(
            iters=len(errs), max_abs_err_first=errs[0], max_abs_err_last=errs[-1],
            max_abs_err_all=max(errs), max_rel_err_all=max(rels),
            bitwise_vs_eager_all_iters=bit, passed=good,
        )
        log(f"replay/{count}", report[f"replay/{count}"])
        # (3) replays with changing pools/lengths vs eager (no carried workspace state).
        bit2 = True
        maxerr = 0.0
        for step in range(12):
            lens = None if step % 3 else [SPECIAL_LENS[(step + i) % len(SPECIAL_LENS)] for i in range(count)]
            if step == 4:
                lens = [0] + [640] * (count - 1)
            nq, nswa, nextra, nln, nsink = make_inputs(count, args.seed + 1000 + step, lens)
            for dst, src in ((q, nq), (swa, nswa), (extra, nextra), (ln, nln), (sink, nsink)):
                dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            got = out.clone()
            e = run_once(fi, nq, nswa, nextra, nln, nsink)
            torch.cuda.synchronize()
            bit2 &= torch.equal(got, e)
            maxerr = max(maxerr, (got.double() - reference(nq, nswa, nextra, nln, nsink)).abs().max().item())
        good = bit2 and maxerr <= 1.6e-2
        ok &= good
        report[f"changing/{count}"] = dict(steps=12, bitwise_vs_eager=bit2, max_abs_err=maxerr, passed=good)
        log(f"changing/{count}", report[f"changing/{count}"])
        del graph
    print(json.dumps({
        "passed": bool(ok),
        "compared": "CSA2FlashInfer._forward_pools split-KV vs FP64 sink-attention reference and vs "
        "disable_split_kv=True path; mask layout identity; graph replays vs eager",
        "report": report,
    }))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("time", "sweep", "check"))
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    log(f"has_split={HAS_SPLIT} device={torch.cuda.get_device_name()}")
    {"time": mode_time, "sweep": mode_sweep, "check": mode_check}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
