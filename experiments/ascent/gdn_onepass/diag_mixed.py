"""q/k reduce-order probe over specializations (in-process, diagnostic only)."""
import sys

import torch

sys.path.insert(0, "experiments/ascent/gdn_onepass")
import proto_lib as P  # noqa: E402

VARS = {
    "auto": dict(),
    "r1": dict(reduce=1),
    "r3": dict(reduce=3),
    "r4": dict(reduce=4),
    "r5": dict(reduce=5),
    "r5ws": dict(reduce=5, warp_seq=True),
}
for seed in (0, 1, 2):
    for name, lens, nd in [("2048+d1", [2048], 1), ("ragged+d2", [1000, 37, 2], 2), ("85+d3", [85], 3),
                           ("1039", [1000, 37, 2], 0), ("85", [85], 0), ("2050", [2050], 0),
                           ("1", [1], 0), ("2", [2], 0), ("3", [3], 0), ("2048", [2048], 0),
                           ("2x2048", [2048, 2048], 0), ("4100", [4100], 0), ("17", [17], 0)]:
        c = P.make(lens, nd, seed=seed)
        c["pool"].copy_(c["pool0"])
        ref = P.run_main(c)
        line = []
        for vn, kw in VARS.items():
            c["pool"].copy_(c["pool0"])
            out = P.run_one(c, **kw)
            torch.cuda.synchronize()
            nb = sum(int((o.view(torch.int16) != r.view(torch.int16)).sum().item())
                     for o, r in zip(out[:3], ref[:3]))
            line.append("%s=%d" % (vn, nb))
        print("seed", seed, name, " ".join(line), flush=True)
        del c
        torch.cuda.empty_cache()
print('{"done": true}')
