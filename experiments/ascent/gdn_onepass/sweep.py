"""Tile / warp / reduce sweep of the one-pass kernel at 1x2048 (L2-warm, producer rewrite), vs main chain."""
import sys

import torch

sys.path.insert(0, "experiments/ascent/gdn_onepass")
import proto_lib as P  # noqa: E402

c = P.make([2048], 0)
flush = torch.empty(32 * 1024 * 1024, dtype=torch.float32, device="cuda")


def timeit(fn, iters=60, warm=10):
    ts = []
    for i in range(warm + iters):
        c["pool"].copy_(c["pool0"])
        flush.fill_(1.0)
        c["qkvz"].copy_(c["qkvz"])  # producer stand-in: all of in_proj_qkvz L2-warm (64 MB)
        torch.cuda._sleep(30000)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        if i >= warm:
            ts.append((s, e))
    torch.cuda.synchronize()
    v = sorted(a.elapsed_time(b) * 1e3 for a, b in ts)
    return v[len(v) // 2], v[len(v) // 10], v[(9 * len(v)) // 10]


ref = P.run_main(c)
print("main chain %.2f us (p10 %.2f p90 %.2f)" % timeit(lambda: P.run_main(c)), flush=True)
for bt in (16, 32, 64):
    for nw in (2, 4, 8):
        for red in (None, 0):
            kw = dict(block_t=bt, num_warps=nw)
            if red is not None:
                kw["reduce"] = red
            try:
                c["pool"].copy_(c["pool0"])
                out = P.run_one(c, **kw)
                torch.cuda.synchronize()
                nb = sum(int((o.view(torch.int16) != r.view(torch.int16)).sum().item())
                         for o, r in zip(out[:3], ref[:3]))
                t = timeit(lambda: P.run_one(c, **kw))
                print("bt=%d nw=%d reduce=%s: %.2f us (p10 %.2f p90 %.2f) mismatches=%d"
                      % (bt, nw, red, t[0], t[1], t[2], nb), flush=True)
            except Exception as e:  # noqa: BLE001
                print("bt=%d nw=%d reduce=%s failed %r" % (bt, nw, red, repr(e)[:200]), flush=True)
print('{"done": true}')
