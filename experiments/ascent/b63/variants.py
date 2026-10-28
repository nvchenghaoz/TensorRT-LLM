"""Branch 63 diagnostics: build module variants in parallel, check bitwise, time at M=4176/4096.

Variants (same .cu, macros):
  coop     : cooperative 256x128x128, div.rn          (default module flags)
  nomath   : coop with the SwiGLU math removed (diagnostic floor: mainloop + stores)
  pingpong : Pingpong 128x128x128, div.rn
"""

import json
import sys
import threading

import torch

sys.path.insert(0, "experiments/ascent/b63")
import b63  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as c  # noqa: E402

VARIANTS = {
    "coop": ("_diag_coop", ("-DFLUX3_SCHED=0",)),
    "nomath": ("_diag_nomath", ("-DFLUX3_DIAG_NOMATH", "-DFLUX3_SCHED=0")),
    "pingpong": ("", ()),  # the default module flags
}


def main():
    names = sys.argv[1:] or list(VARIANTS)
    specs = {n: c.gen_spec(*VARIANTS[n]) for n in names}
    errors = {}

    def build(n):
        try:
            specs[n].build()
        except Exception as e:  # noqa: BLE001
            errors[n] = repr(e)[-3000:]

    threads = [threading.Thread(target=build, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for n, e in errors.items():
        b63.log("BUILD FAILED", n, e)
    mods = {}
    for n in names:
        if n in errors:
            continue
        mods[n] = specs[n].build_and_load()
        b63.log(n, "stages", mods[n].flux3_swiglu_fp4_stages(), "smem",
                mods[n].flux3_swiglu_fp4_smem_bytes())

    def fn_for(mod):
        def f(args, kw):
            a, w, sfa, sfw, s_in, s_w = args
            alpha = b63.m._scalar(a.device, s_in * s_w)
            return c.swiglu_packed(a, w, sfa, sfw, alpha, kw["output_inverse_scale"], module=mod)

        return f

    res = {}
    for n, mod in mods.items():
        if n == "nomath":
            continue
        for M in (4176, 4096):
            r = b63.compare(M, M + 1, fn_for(mod))
            res[f"{n}_M{M}"] = r
            b63.log(n, M, r)
    timing = {}
    runner = b63.fp._sm120_fp4_gemm_runner()
    for M in (4176, 4096):
        args, kw, _ = b63.make_case(M)
        a, w, sfa, sfw, s_in, s_w = args
        alpha = b63.m._scalar(a.device, s_in * s_w)
        calls = [("ref_packed", lambda: b63.ref_packed(args, kw)),
                 ("cutlass_t2", lambda: runner.run_gemm(a, w, sfa, sfw, alpha, 0, 2, None, None))]
        for n, mod in mods.items():
            calls.append((n, (lambda f: (lambda: f(args, kw)))(fn_for(mod))))
        for rnd in range(2):
            for name, fn in calls:
                med, noise = b63.time_fn(fn)
                timing.setdefault(f"M{M}", {}).setdefault(name, []).append(round(med * 1e3, 1))
                b63.log(f"M={M} r{rnd} {name}: {med * 1e3:.1f} us ({noise:.3f})")
    print(json.dumps({"errors": errors, "bitwise": res, "us": timing}))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
