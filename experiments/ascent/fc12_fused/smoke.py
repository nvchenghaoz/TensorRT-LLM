"""Compile the fused kernel and compare it with main's fc1+fc2 on a few (M, D) cases (eager)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common12 import H, SHAPE, Runner, log, make_state, routing  # noqa: E402

CASES = [tuple(int(v) for v in c.split("x")) for c in
         os.environ.get("SMOKE_CASES", "1x2,1x0,4x6,8x12").split(",")]


def main():
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    leaf, st = make_state()
    run = Runner(leaf, st)
    sync = run.fk.sync_workspace("cuda")
    ok = True
    rows = []
    for m, d in CASES:
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids, w = routing(m, d)
        q1b, sf1b, outb, _ = run(x, ids, w, "base")
        torch.cuda.synchronize()
        q1b, sf1b, outb = q1b.clone(), sf1b.clone(), outb.clone()
        log(f"{SHAPE} M{m} D{d}: base done; compiling/running fused")
        q1f, sf1f, outf, off = run(x, ids, w, "fused")
        torch.cuda.synchronize()
        q1_eq = bool(torch.equal(q1f.view(torch.uint8), q1b.view(torch.uint8)))
        sf_eq = bool(torch.equal(sf1f, sf1b))
        dmax = (outf.float() - outb.float()).abs().max().item()
        amax = outb.float().abs().max().item()
        cnt = sync[:8].tolist()
        nz = int((sync != 0).sum().item())
        fin = bool(torch.isfinite(outf.float()).all().item())
        r = {"M": m, "D": d, "q1_bitwise": q1_eq, "sf1_bitwise": sf_eq, "out_maxabs": dmax,
             "out_absmax": amax, "sync_nonzero": nz, "finite": fin}
        rows.append(r)
        log(json.dumps(r), "sync head", cnt)
        ok = ok and q1_eq and sf_eq and nz == 0 and fin and dmax <= 2e-2 * max(amax, 1e-3)
    print(json.dumps({"passed": ok, "rows": rows}))


if __name__ == "__main__":
    main()
