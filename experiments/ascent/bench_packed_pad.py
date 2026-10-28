"""Microbenchmark + bitwise check for flux3_sm120_nvfp4.run('packed') (idea 70).

Single-block gate_up: N=18432 (pair-interleaved value/gate), K=3072, M=4096+T.
The activation is a view of the first M rows of a larger buffer whose tail is
0xFF, so any read past [M, K/2] would change the output.

Usage: python bench_packed_pad.py {bench|check}
"""

import hashlib
import json
import statistics
import sys

import torch

from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4 as mod

N, K = 18432, 3072
PK = K // 2
MS = (4096, 4176, 4200, 4256)
IN_GS, W_GS = 1.0 / 16, 1.0 / 16

_compiles = [0]
_orig_compile = mod.cute.compile


def _counting_compile(*args, **kwargs):
    _compiles[0] += 1
    return _orig_compile(*args, **kwargs)


mod.cute.compile = _counting_compile


def log(*a):
    print(*a, flush=True)


def scale_size(rows, cols):
    return ((rows + 127) // 128) * 128 * ((cols // 16 + 3) // 4) * 4


def rand_e4m3(n, gen, dev):
    v = torch.rand(n, generator=gen, device=dev) * 1.75 + 0.25
    return v.to(torch.float8_e4m3fn).view(torch.uint8)


def make_inputs(dev):
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    weight = torch.randint(0, 256, (N, PK), generator=gen, device=dev, dtype=torch.uint8)
    weight_scale = rand_e4m3(scale_size(N, K), gen, dev)
    out_inv = torch.full((1,), 0.5, device=dev, dtype=torch.float32)
    per_m = {}
    for m in MS:
        buf = torch.full((m + 256, PK), 0xFF, device=dev, dtype=torch.uint8)
        buf[:m] = torch.randint(0, 256, (m, PK), generator=gen, device=dev, dtype=torch.uint8)
        a_scale = rand_e4m3(scale_size(m, K), gen, dev)
        per_m[m] = (buf, a_scale)
    return weight, weight_scale, out_inv, per_m


def call(m, weight, weight_scale, out_inv, per_m):
    buf, a_scale = per_m[m]
    return mod.run(
        "packed",
        buf[:m],
        weight,
        a_scale,
        weight_scale,
        IN_GS,
        W_GS,
        output_inverse_scale=out_inv,
    )


def digest(fp4, sf):
    h = hashlib.sha256()
    h.update(fp4.contiguous().cpu().numpy().tobytes())
    h.update(sf.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    dev = torch.device("cuda")
    weight, weight_scale, out_inv, per_m = make_inputs(dev)
    pristine = {m: (b.clone(), s.clone()) for m, (b, s) in per_m.items()}
    w_pristine = (weight.clone(), weight_scale.clone(), out_inv.clone())

    # Warmup: compile every key once (4176 and 4200 share padded_rows=4224).
    compiles_by_m = {}
    for m in MS:
        before = _compiles[0]
        call(m, weight, weight_scale, out_inv, per_m)
        torch.cuda.synchronize()
        compiles_by_m[m] = _compiles[0] - before
    log("warmup compiles per M:", compiles_by_m, "total", _compiles[0], "keys", len(mod._COMPILED))
    for _ in range(5):
        for m in MS:
            call(m, weight, weight_scale, out_inv, per_m)
    torch.cuda.synchronize()
    warm_compiles = _compiles[0]

    WARM, ITERS = 10, 200
    if mode == "bench":
        # Interleaved series: 4176, 4096, 4176 again (A/A), 4200.
        series = [("4176", 4176), ("4096", 4096), ("4176_aa", 4176), ("4200", 4200)]
        times = {name: [] for name, _ in series}
        for _ in range(WARM):
            for _, m in series:
                call(m, weight, weight_scale, out_inv, per_m)
        torch.cuda.synchronize()
        for it in range(ITERS):
            for name, m in series:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                call(m, weight, weight_scale, out_inv, per_m)
                e.record()
                e.synchronize()
                times[name].append(s.elapsed_time(e))
            if it % 50 == 0:
                log("iter", it)
        table = {}
        for name, ts in times.items():
            ts = sorted(ts)
            med = statistics.median(ts)
            p10 = ts[int(0.1 * len(ts))]
            p90 = ts[int(0.9 * len(ts)) - 1]
            table[name] = {
                "median_ms": med,
                "p10_ms": p10,
                "p90_ms": p90,
                "noise": (p90 - p10) / med,
                "n": len(ts),
            }
            log(name, table[name])
        aa = table["4176_aa"]["median_ms"] / table["4176"]["median_ms"] - 1
        log("A/A rel delta 4176 vs 4176_aa:", aa)
        result = {
            "batch": {
                "1": {
                    "kernel_ms": table["4176"]["median_ms"],
                    "noise_floor": max(table["4176"]["noise"], table["4176_aa"]["noise"]),
                }
            },
            "output": {"class": "exact"},
            "table": table,
            "aa_rel_delta": aa,
            "compiles_warmup": warm_compiles,
            "compiles_in_timed": _compiles[0] - warm_compiles,
            "compiles_by_m": compiles_by_m,
            "keys": len(mod._COMPILED),
        }
        print(json.dumps(result), flush=True)
        return

    # Untimed correctness sequence: same warmup and iteration counts, reset
    # (restore inputs from pristine) before the loop.
    for m in MS:
        b, s = pristine[m]
        per_m[m][0].copy_(b)
        per_m[m][1].copy_(s)
    for _ in range(WARM):
        for m in MS:
            call(m, weight, weight_scale, out_inv, per_m)
    torch.cuda.synchronize()
    ok = True
    report = {}
    ref = {}
    for m in MS:
        # Explicit reference with main's semantics: a zero-padded activation at
        # padded_rows (run() needs no padding for it in either tree).
        pr = ((m + 127) // 128) * 128
        buf, a_scale = per_m[m]
        a_pad = torch.zeros((pr, PK), device=dev, dtype=torch.uint8)
        a_pad[:m] = buf[:m]
        r_fp4, r_sf = mod.run(
            "packed", a_pad, weight, a_scale, weight_scale, IN_GS, W_GS, output_inverse_scale=out_inv
        )
        ref[m] = (r_fp4[:m].clone(), r_sf.clone())
    torch.cuda.synchronize()
    for m in MS:
        report[m] = {"iters": 0, "mismatch_ref": 0, "mismatch_it0": 0}
    it0 = {}
    for it in range(ITERS):
        for m in MS:
            fp4, sf = call(m, weight, weight_scale, out_inv, per_m)
            if it == 0:
                it0[m] = (fp4.clone(), sf.clone())
                report[m]["digest"] = digest(fp4, sf)
                report[m]["nonzero_fp4"] = int((fp4 != 0).sum())
                report[m]["nonzero_sf"] = int((sf != 0).sum())
                report[m]["sf_bytes"] = sf.numel()
            r = report[m]
            r["iters"] += 1
            if not (torch.equal(fp4, ref[m][0]) and torch.equal(sf, ref[m][1])):
                r["mismatch_ref"] += 1
            if not (torch.equal(fp4, it0[m][0]) and torch.equal(sf, it0[m][1])):
                r["mismatch_it0"] += 1
        if it % 50 == 0:
            for m in MS:
                b, s = pristine[m]
                if not (torch.equal(per_m[m][0], b) and torch.equal(per_m[m][1], s)):
                    ok = False
                    report[m]["input_changed"] = True
            log("check iter", it)
    for m in MS:
        b, s = pristine[m]
        if not (torch.equal(per_m[m][0], b) and torch.equal(per_m[m][1], s)):
            ok = False
            report[m]["input_changed"] = True
        tail_ff = bool((per_m[m][0][m:] == 0xFF).all())
        report[m]["tail_0xff"] = tail_ff
        ok = ok and tail_ff and report[m]["mismatch_ref"] == 0 and report[m]["mismatch_it0"] == 0
        ok = ok and report[m]["nonzero_fp4"] > 0
    w_ok = (
        torch.equal(weight, w_pristine[0])
        and torch.equal(weight_scale, w_pristine[1])
        and torch.equal(out_inv, w_pristine[2])
    )
    ok = ok and w_ok
    compiles_ok = compiles_by_m[4176] + compiles_by_m[4200] == 1 and _compiles[0] == warm_compiles
    result = {
        "passed": bool(ok and compiles_ok),
        "compared": "run('packed') fp4[:M] + full output_scale vs zero-padded-activation reference and vs iteration 0, 200 iters x M=4096/4176/4200/4256; activation tail past [M,K/2] is 0xFF; digests compare across trees",
        "per_m": {str(k): v for k, v in report.items()},
        "compiles_by_m": compiles_by_m,
        "compiles_after_warmup": _compiles[0] - warm_compiles,
        "keys": len(mod._COMPILED),
        "weights_unchanged": w_ok,
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
