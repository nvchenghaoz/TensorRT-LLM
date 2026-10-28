"""FLUX.3 stage-boundary GPU idle microbenchmark (idea 54, branch 42). No model load.

Both trees run this script. "module" is the tree's pipeline_flux3 (main in the base arm,
the branch in the candidate arm); "ref" / "ref2" are verbatim copies of main's
sample_image_latents / scatter_ids / _compress_time / _as_image_grid (an A/A pair).

Boundary a (TE -> denoise): a synthetic GPU tail (torch.cuda._sleep, ~8.8 ms like the
graphed TE) is queued, then sample_image_latents runs and a marker kernel consumes x and
x_ids. Boundary b (denoise -> VAE): a synthetic tail, then _as_image_grid + the VAE's
inv_normalize/rearrange entry and a marker. Each boundary's idle = elapsed time from the
tail's end event to the marker's end event (GPU idle + the boundary's own tiny kernels).

Usage: python flux3_boundary_bench.py bench|check [iters]
bench: last line {"batch": {"1": {"step_ms": a+b idle of module, "noise_floor": ...}}, ...}
check: untimed, same iterations/reset points; x, x_ids, orig_ids, grid z (values and
strides) of module == ref for every iteration and == iteration 0; last line {"passed": ...}.
"""

import json
import statistics
import sys

import torch
from einops import rearrange

import tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 as pf

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
WARMUP = 5
dev = torch.device("cuda")


# ---- verbatim main (a5fbf7d3) copies -------------------------------------------------
def ref_compress_time(t_ids):
    t_max = torch.max(t_ids)
    remap = torch.zeros((t_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    uniq = torch.unique(t_ids, sorted=True)
    remap[uniq] = torch.arange(len(uniq), device=t_ids.device, dtype=t_ids.dtype)
    return remap[t_ids]


def ref_scatter_ids(x, x_ids):
    out_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        t_ids_c = ref_compress_time(t_ids)
        t = int(torch.max(t_ids_c)) + 1
        h = int(torch.max(h_ids)) + 1
        w = int(torch.max(w_ids)) + 1
        flat_ids = t_ids_c * w * h + h_ids * w + w_ids
        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out_list.append(out.reshape(1, t, h, w, ch).permute(0, 4, 1, 2, 3))
    return out_list


def ref_sample_image_latents(*, height, width, seed, device, image_rope_scale=0.5,
                             dtype=torch.bfloat16, copies=2):
    h, w = height // 16, width // 16
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn((h * w, 128), generator=rng).unsqueeze(0)
    x_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1)
    ).unsqueeze(0)
    orig_ids = x_ids.to(device)
    x_ids = x_ids.to(dtype=torch.float32)
    x_ids[..., 1:3] = x_ids[..., 1:3] * image_rope_scale
    x = x.repeat(copies, 1, 1).to(device=device, dtype=dtype)
    x_ids = x_ids.repeat(copies, 1, 1).to(device=device)
    return x, x_ids, orig_ids


def ref_as_image_grid(tokens, orig_ids):
    return ref_scatter_ids(tokens.float(), orig_ids[:1])[0].squeeze(2)


# ---- harness -------------------------------------------------------------------------
IMPL = {
    "module": (pf.sample_image_latents, pf._as_image_grid),
    "ref": (ref_sample_image_latents, ref_as_image_grid),
    "ref2": (ref_sample_image_latents, ref_as_image_grid),
}
# VAE entry stand-in (Flux3ImageVAE.inv_normalize + rearrange) with fixed bn stats.
g = torch.Generator(device="cpu").manual_seed(1234)
BN_VAR = (torch.rand(128, generator=g) + 0.5).to(dev)
BN_MEAN = torch.randn(128, generator=g).to(dev)
LATENT = torch.randn((4096, 128), generator=g).to(dev, torch.bfloat16).unsqueeze(0)
LATENT_PRISTINE = LATENT.clone()


def vae_entry(grid):
    s = torch.sqrt(BN_VAR.view(1, -1, 1, 1) + 1e-4)
    z = grid * s + BN_MEAN.view(1, -1, 1, 1)
    return rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2)


def calibrate_sleep(target_ms):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cyc = 10_000_000
    for _ in range(3):
        torch.cuda._sleep(cyc)
    torch.cuda.synchronize()
    s.record()
    torch.cuda._sleep(cyc)
    e.record()
    e.synchronize()
    return int(cyc * target_ms / s.elapsed_time(e))


TE_CYC = TAIL_CYC = None  # calibrated in __main__ (importers only use the ref copies)


def run_a(name, seed):
    sample, _ = IMPL[name]
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    torch.cuda._sleep(TE_CYC)
    e0.record()
    x, x_ids, oid = sample(height=1024, width=1024, seed=seed, device=dev,
                           image_rope_scale=0.5, copies=2)
    marker = x.float().sum() + x_ids.sum()  # first consumer of the prepared inputs
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1), (x, x_ids, oid, marker)


def run_b(name, oid):
    _, grid_fn = IMPL[name]
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    torch.cuda._sleep(TAIL_CYC)
    tokens = LATENT * 1  # the last denoise step's output, produced behind the tail
    e0.record()
    grid = grid_fn(tokens, oid)
    z = vae_entry(grid)
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1), (grid, z)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def bench():
    order = ["module", "ref", "ref2"]
    samples = {n: {"a": [], "b": []} for n in order}
    oids = {n: run_a(n, 0)[1][2] for n in order}
    for it in range(WARMUP + ITERS):
        rot = order[it % 3:] + order[:it % 3]
        for n in rot:
            ta, _ = run_a(n, it)
            tb, _ = run_b(n, oids[n])
            if it >= WARMUP:
                samples[n]["a"].append(ta)
                samples[n]["b"].append(tb)
        if it % 20 == 0:
            print(f"iter {it}", flush=True)
    stats = {}
    for n in order:
        tot = [a + b for a, b in zip(samples[n]["a"], samples[n]["b"])]
        med = statistics.median(tot)
        stats[n] = {
            "a_med_ms": statistics.median(samples[n]["a"]),
            "b_med_ms": statistics.median(samples[n]["b"]),
            "total_med_ms": med,
            "total_p10": pct(tot, 0.1),
            "total_p90": pct(tot, 0.9),
            "noise_floor": (pct(tot, 0.9) - pct(tot, 0.1)) / med,
        }
        print(n, json.dumps(stats[n]), flush=True)
    aa = abs(stats["ref"]["total_med_ms"] - stats["ref2"]["total_med_ms"])
    print(f"A/A |ref - ref2| total median = {aa:.4f} ms; module - ref = "
          f"{stats['module']['total_med_ms'] - stats['ref']['total_med_ms']:.4f} ms", flush=True)
    m = stats["module"]
    print(json.dumps({
        "batch": {"1": {"step_ms": m["total_med_ms"], "noise_floor": m["noise_floor"]}},
        "output": {"class": "exact"},
        "boundary_a_ms": m["a_med_ms"], "boundary_b_ms": m["b_med_ms"],
        "ref_main_in_process": stats["ref"], "ref2_main_in_process": stats["ref2"],
        "aa_abs_ms": aa, "iters": ITERS,
    }))


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride() and \
        torch.equal(a.cpu(), b.cpu())


def check():
    ok = True
    details = []
    first = {}
    for it in range(WARMUP + ITERS):
        assert torch.equal(LATENT, LATENT_PRISTINE), "input latent mutated"
        seed = it % 4
        _, (x, x_ids, oid, _) = run_a("module", seed)
        _, (rx, rx_ids, roid, _) = run_a("ref", seed)
        _, (grid, z) = run_b("module", oid)
        _, (rgrid, rz) = run_b("ref", roid)
        r = {
            "x": same(x, rx), "x_ids": same(x_ids, rx_ids),
            "orig_ids_values": torch.equal(oid.cpu(), roid.cpu()) and oid.shape == roid.shape
            and oid.dtype == roid.dtype,
            "grid": same(grid, rgrid), "z": same(z, rz),
            "z_strides": list(z.stride()), "ref_z_strides": list(rz.stride()),
            "orig_ids_device": str(oid.device),
        }
        if seed not in first:
            first[seed] = (x.clone(), x_ids.clone(), grid.clone(), z.clone())
        else:
            f = first[seed]
            r["repeat_vs_iter0"] = all(torch.equal(p, q) for p, q in zip(f, (x, x_ids, grid, z)))
        good = all(v for k, v in r.items() if isinstance(v, bool))
        ok &= good
        if it < 4 or not good:
            details.append({"iter": it, **r})
        if it % 20 == 0:
            print(f"check iter {it}: {good}", flush=True)
    # Non-raster ids still go through the scatter path (two time slices, shuffled).
    perm = torch.randperm(2 * 4096, generator=torch.Generator().manual_seed(7))
    ids = torch.cartesian_prod(torch.tensor([0, 3]), torch.arange(64), torch.arange(64),
                               torch.arange(1))[perm].unsqueeze(0)
    toks = torch.randn((1, 2 * 4096, 128), generator=g).to(dev, torch.bfloat16)
    nr_mod = pf._as_image_grid(toks, ids.to(dev))
    nr_mod_cpu = pf._as_image_grid(toks, ids)
    nr_ref = ref_as_image_grid(toks, ids.to(dev))
    nonraster = same(nr_mod, nr_ref) and same(nr_mod_cpu, nr_ref)
    ok &= nonraster
    print(json.dumps({
        "passed": bool(ok),
        "compared": "tree pipeline_flux3 sample_image_latents/_as_image_grid (+VAE entry) vs "
                    "verbatim main copies: x, x_ids, orig_ids, grid and z bitwise with equal "
                    "strides, every iteration and vs iteration 0 per seed; non-raster scatter path",
        "iters": WARMUP + ITERS, "nonraster_scatter_equal": nonraster, "details": details,
    }, default=str))


if __name__ == "__main__":
    TE_CYC = calibrate_sleep(8.8)
    TAIL_CYC = calibrate_sleep(8.0)
    bench() if MODE == "bench" else check()
