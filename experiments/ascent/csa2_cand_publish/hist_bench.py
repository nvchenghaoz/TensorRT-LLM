"""CandidatePublishKernel wide path: 8-bit histogram select (this branch) vs branch 140.

The shipped kernel (design A) is compared with branch 140's kernel, rebuilt here as
a subclass whose ``_select_wide`` is 140's code verbatim (16-round 2-bit radix
select, keys re-read from shared memory). ``stage0`` is 140's kernel with the
select skipped (fixed -inf threshold) - bench-only, never shipped - to split the
row cost into block-max/load + scan/publish vs select.

Modes:
  variants - graph replay, 256 MB L2 flush before each replay, the variants
             interleaved replay by replay; per-case median/p10/p90 per variant.
  check    - untimed: the bench's warmup + iters graph-replay sequence of the
             shipped kernel, outputs reset to garbage before every replay; every
             replay torch.equal to branch 140's kernel on EVERY row (ties
             included); 140's kernel also checked against the PyTorch fallback on
             non-tie rows; no allocation during replay; decode capture compiles
             only in eager warmup, one launch per call.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402

import tensorrt_llm._torch.attention.backends.sparse.csa2.indexer as ix  # noqa: E402
import tensorrt_llm._torch.attention.backends.sparse.csa2.kernel as K  # noqa: E402

log = B.log
_orig_launch = K._launch
VARIANT = {"cls": None, "tag": None}


def _patched_launch(key, factory, *args):
    if VARIANT["cls"] is not None and key[0] == "candidate_publish":
        cls = VARIANT["cls"]
        key = key + (VARIANT["tag"],)

        def factory():
            return cls(key[1], key[2], key[3])

    return _orig_launch(key, factory, *args)


K._launch = _patched_launch


def build_variants():
    ns = K._glue_kernels()
    cute, cutlass = ns.cute, ns.cutlass
    Int32, Int64, Float32 = cutlass.Int32, cutlass.Int64, cutlass.Float32
    llvm, T, dsl_user_op = ns.llvm, ns.T, ns.dsl_user_op
    INT32_MAX = K._INT32_MAX

    @dsl_user_op
    def bitcast_f32_i32(value, *, loc=None, ip=None):
        return Int32(llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip)))

    class Ref140(ns.CandidatePublishKernel):
        """Branch 140's wide select (landing 47), verbatim apart from ``skip_select``."""

        skip_select = False

        @cute.jit
        def _select_wide(
            self, score_t, keys, ids, partial, hist, q, width, blocks, selected_blocks, latest, tid
        ):
            lane = tid % 32
            warp = tid // 32
            warps = self.threads // 32
            per_thread = self.padded // self.threads
            bs = self.block_size
            pad_key = Int32(-(2**31))
            bias = Int64(2**31)
            neg_inf_u = Int64(0x007FFFFF)  # key of -inf, biased to [0, 2**32)
            for b in cutlass.range(tid, self.padded, self.threads):
                ids[b] = Int32(INT32_MAX)
            for b in cutlass.range(blocks + tid, self.padded, self.threads):
                keys[b] = pad_key
            if cutlass.const_expr(self.coalesced):
                unroll = 8
                span = blocks * Int32(bs)
                for base in cutlass.range(0, span, self.threads * unroll):
                    loaded = []
                    for u in cutlass.range_constexpr(unroll):
                        column = base + Int32(u * self.threads) + tid
                        clamped = column
                        if clamped >= width:
                            clamped = width - 1
                        value = Float32(score_t[q, clamped])
                        if column >= width:
                            value = Float32(float("-inf"))
                        loaded.append(value)
                    for u in cutlass.range_constexpr(unroll):
                        value = loaded[u]
                        offset = 1
                        while offset < bs:
                            value = cute.arch.fmax(
                                value, cute.arch.shuffle_sync_bfly(value, offset=offset)
                            )
                            offset = offset * 2
                        column = base + Int32(u * self.threads) + tid
                        b = column // Int32(bs)
                        if (column % Int32(bs) == 0) & (b < blocks):
                            if Int64(b) == latest:
                                value = Float32(float("inf"))
                            bits = bitcast_f32_i32(value + Float32(0.0))
                            if bits < 0:
                                bits = bits ^ Int32(0x7FFFFFFF)
                            keys[b] = bits
            else:
                for b in cutlass.range(tid, blocks, self.threads):
                    value = Float32(float("-inf"))
                    first = Int64(b) * Int64(bs)
                    last = first + Int64(bs)
                    if last > Int64(width):
                        last = Int64(width)
                    for column in cutlass.range(first, last, 1):
                        value = cute.arch.fmax(value, Float32(score_t[q, Int32(column)]))
                    if Int64(b) == latest:
                        value = Float32(float("inf"))
                    bits = bitcast_f32_i32(value + Float32(0.0))
                    if bits < 0:
                        bits = bits ^ Int32(0x7FFFFFFF)
                    keys[b] = bits
            cute.arch.barrier()
            threshold = neg_inf_u
            if cutlass.const_expr(not self.skip_select):
                if selected_blocks < blocks:
                    threshold = Int64(0)
                    for r in cutlass.range_constexpr(16):
                        shift = 30 - 2 * r
                        c1 = threshold + Int64(1 << shift)
                        c2 = threshold + Int64(2 << shift)
                        c3 = threshold + Int64(3 << shift)
                        n12 = Int32(0)
                        n3 = Int32(0)
                        for i in cutlass.range_constexpr(per_thread):
                            key = Int64(keys[tid + i * self.threads]) + bias
                            if key >= c1:
                                n12 = n12 + 1
                            if key >= c2:
                                n12 = n12 + 65536
                            if key >= c3:
                                n3 = n3 + 1
                        t12, t3 = self._block_sum2(
                            n12, n3, partial, (r % 2) * 2 * warps, lane, warp
                        )
                        chosen = threshold
                        if (t12 & 65535) >= selected_blocks:
                            chosen = c1
                        if (t12 >> 16) >= selected_blocks:
                            chosen = c2
                        if t3 >= selected_blocks:
                            chosen = c3
                        threshold = chosen
            local = Int32(0)
            for i in cutlass.range_constexpr(per_thread):
                key = Int64(keys[tid * per_thread + i]) + bias
                if key > threshold:
                    local = local + 1
                if key == threshold:
                    local = local + 65536
            inclusive = local
            offset = 1
            while offset < 32:
                other = cute.arch.shuffle_sync(inclusive, lane - offset)
                if lane >= offset:
                    inclusive = inclusive + other
                offset = offset * 2
            scan_base = 4 * warps
            if lane == 31:
                partial[scan_base + warp] = inclusive
            cute.arch.barrier()
            warp_total = Int32(0)
            if lane < warps:
                warp_total = partial[scan_base + lane]
            warp_inclusive = warp_total
            offset = 1
            while offset < 32:
                other = cute.arch.shuffle_sync(warp_inclusive, lane - offset)
                if lane >= offset:
                    warp_inclusive = warp_inclusive + other
                offset = offset * 2
            total = cute.arch.shuffle_sync(warp_inclusive, Int32(31))
            before_warp = cute.arch.shuffle_sync(warp_inclusive, warp) - partial[scan_base + warp]
            prefix = before_warp + inclusive - local
            above = prefix & 65535
            equal = prefix >> 16
            take = selected_blocks - (total & 65535)
            if threshold == neg_inf_u:
                take = Int32(0)  # ties at -inf are unreachable
            for i in cutlass.range_constexpr(per_thread):
                b = tid * per_thread + i
                key = Int64(keys[b]) + bias
                taken = equal
                if taken > take:
                    taken = take
                if (key > threshold) | ((key == threshold) & (equal < take)):
                    ids[above + taken] = Int32(b)
                if key > threshold:
                    above = above + 1
                if key == threshold:
                    equal = equal + 1
            cute.arch.barrier()

    class Stage0(Ref140):
        skip_select = True

    return {"ref140": Ref140, "stage0": Stage0, "A": None}


def use(variants, name):
    VARIANT["cls"] = variants[name]
    VARIANT["tag"] = None if variants[name] is None else name


def stats(times):
    t = sorted(times)
    n = len(t)
    med = t[n // 2]
    p10 = t[int(0.1 * (n - 1))]
    p90 = t[int(0.9 * (n - 1))]
    return med, p10, p90


def variants_mode(args):
    variants = build_variants()
    names = args.names.split(",")
    ind = B.make_indexer()
    table = []
    iters = args.iters
    for count, width in B.CASES:
        scores, visible, wp = B.make_inputs(count, width, 1234 + count)
        graphs, outs = {}, {}
        for name in names:
            use(variants, name)
            fn, out, _ = B.make_call(ind, scores, visible, wp)
            graphs[name] = B.capture(fn)
            outs[name] = out
        use(variants, "A")
        for _ in range(B.WARMUP):
            for name in names:
                B.l2_flush()
                graphs[name].replay()
        torch.cuda.synchronize()
        times = {name: [] for name in names}
        clocks = []
        for i in range(iters):
            for name in names:
                B.l2_flush()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[name].replay()
                b.record()
                b.synchronize()
                times[name].append(a.elapsed_time(b) * 1000.0)
            if i % 20 == 10:
                clocks.append(B.sm_clock())
        row = dict(count=count, width=width, sm_clock_min=min(clocks))
        for name in names:
            med, p10, p90 = stats(times[name])
            row[name] = dict(us=round(med, 2), p10=round(p10, 2), p90=round(p90, 2))
        same = {}
        if "ref140" in names:
            ref = [t.clone() for t in outs["ref140"]()]
            for name in names:
                if name not in ("ref140", "stage0"):
                    same[name] = all(torch.equal(x, y) for x, y in zip(outs[name](), ref))
        row["equal_to_ref140"] = same
        log(json.dumps(row))
        table.append(row)
        del graphs
    print(json.dumps({"variants": table}))


KINDS = ("normal", "special", "ties", "narrow", "constant_hi")


def make_case(count, width, seed, kind):
    if kind in ("normal", "special", "ties"):
        return B.make_inputs(count, width, seed, kind)
    scores, visible, wp = B.make_inputs(count, width, seed, "normal")
    cols = torch.arange(width, device="cuda")
    mask = cols[None, :] >= visible[:, None].long()
    if kind == "narrow":  # values differing only in low mantissa bits: rounds 2-3 decide
        g = torch.Generator(device="cuda").manual_seed(seed + 1)
        s = 1.0 + torch.randint(0, 1 << 12, (count, width), generator=g, device="cuda") * 2.0**-23
        s = torch.where(torch.rand(count, width, generator=g, device="cuda") < 0.5, s, -s)
        scores = s.float().masked_fill(mask, -torch.inf).contiguous()
    elif kind == "constant_hi":  # many equal keys around K plus +/-0 and denormals
        s = torch.zeros(count, width, device="cuda")
        s[:, ::3] = -0.0
        s[:, 1::7] = 1e-40
        s[:, 2::11] = 3.0
        scores = s.masked_fill(mask, -torch.inf).contiguous()
    return scores, visible, wp


def check_mode(args):
    variants = build_variants()
    ind = B.make_indexer()
    ok = True
    report = []
    for count, width in B.CASES:
        for kind in KINDS:
            scores, visible, wp = make_case(count, width, 99 + count, kind)
            sel = min(B.TOPK_BLOCKS, -(-width // B.BLOCK))
            ties, _ = B.tie_rows(scores, visible, sel)
            # Reference 1: branch 140's kernel (eager, twice: deterministic).
            use(variants, "ref140")
            fn, out, reset = B.make_call(ind, scores, visible, wp)
            fn()
            torch.cuda.synchronize()
            ref = [t.clone() for t in out()]
            reset()
            fn()
            torch.cuda.synchronize()
            ref_det = all(torch.equal(x, y) for x, y in zip(out(), ref))
            # Reference 2: PyTorch fallback on non-tie rows (sanity of the reference).
            fb_results, fb_blocks, fb_counts = {}, {}, {}
            saved = ix.dsl_available
            ix.dsl_available = lambda: False
            try:
                ind._publish_candidates(scores, visible, fb_results, 0, count, wp,
                                        fb_blocks, fb_counts)
            finally:
                ix.dsl_available = saved
            torch.cuda.synchronize()
            fb = (fb_results[B.LAYER], fb_blocks[B.LAYER], fb_counts[B.LAYER])
            nontie = torch.ones(count, dtype=torch.bool, device="cuda")
            if ties:
                nontie[list(ties)] = False
            ref_vs_fb = all(torch.equal(a[nontie], b[nontie]) for a, b in zip(ref, fb))
            # Candidate: shipped kernel, the bench's warmup + iters replay sequence.
            use(variants, "A")
            fn, out, reset = B.make_call(ind, scores, visible, wp)
            g = B.capture(fn)
            torch.cuda.synchronize()
            bad_iters = []
            bad_rows = 0
            for it in range(B.WARMUP + B.ITERS):
                reset()
                g.replay()
                torch.cuda.synchronize()
                got = out()
                if not all(torch.equal(x, y) for x, y in zip(got, ref)):
                    bad_iters.append(it)
                    rows = (got[0] != ref[0]).any(1) | (got[1] != ref[1]).any(1) | (
                        got[2] != ref[2]).any(1)
                    bad_rows = max(bad_rows, int(rows.sum()))
            torch.cuda.synchronize()
            n0 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
            m0 = torch.cuda.memory_allocated()
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            n1 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
            m1 = torch.cuda.memory_allocated()
            alloc = (n1 - n0) + (m1 - m0)
            passed = (not bad_iters) and ref_det and ref_vs_fb and alloc == 0
            ok &= passed
            line = dict(count=count, width=width, kind=kind, tie_rows=len(ties),
                        replays=B.WARMUP + B.ITERS, bad_iters=len(bad_iters),
                        max_bad_rows=bad_rows, ref_deterministic=ref_det,
                        ref140_eq_fallback_nontie=ref_vs_fb, alloc_delta=alloc, passed=passed)
            log(json.dumps(line))
            report.append(line)
            del g
    use(variants, "A")
    keys = [key for key in K._COMPILED if key[0][0] == "candidate_publish"]
    shipped = [key for key in keys if len(key[0]) == 5]
    log("compiled candidate_publish keys:", keys)
    launches = {}
    for count in (1, 2, 4, 8):
        scores, visible, wp = B.make_inputs(count, 36864, 7)
        fn, out, reset = B.make_call(ind, scores, visible, wp)
        names = B.kernel_names(fn)
        before = len(K._COMPILED)
        g = B.capture(fn)  # compile happened in eager warmup; capture adds nothing
        compiled_in_capture = len(K._COMPILED) - before
        torch.cuda.synchronize()
        n0 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        n1 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
        launches[count] = dict(launches=len(names), kernels=sorted(set(n[:60] for n in names)),
                               compiled_in_capture=compiled_in_capture,
                               replay_allocs=n1 - n0)
        log(f"decode count={count} {launches[count]}")
        if len(names) != 1 or compiled_in_capture or n1 != n0:
            ok = False
        del g
    print(json.dumps({"passed": bool(ok), "compared_against":
                      "branch 140 CandidatePublishKernel (landing 47) on every row incl. ties; "
                      "140 vs PyTorch fallback on non-tie rows",
                      "shipped_keys": len(shipped), "cases": report,
                      "decode": launches}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("variants", "check"))
    p.add_argument("--names", default="ref140,stage0,A")
    p.add_argument("--iters", type=int, default=60)
    a = p.parse_args()
    log("device", torch.cuda.get_device_name())
    {"variants": variants_mode, "check": check_mode}[a.mode](a)
