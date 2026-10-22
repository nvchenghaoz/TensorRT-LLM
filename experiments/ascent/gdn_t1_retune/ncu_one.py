# ncu driver: main T1 config (or BV/warps from argv) at N in argv[1] (comma
# list), cold L2 (256 MiB read before each launch), 2 launches per N.
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common  # noqa: E402

batches = [int(x) for x in sys.argv[1].split(",")]
cfg = None
if len(sys.argv) > 2:
    bv, w = sys.argv[2].split("x")
    cfg = (int(bv), int(w))
for n in batches:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1000 + n)
    lay = common.make_layer(n, gen)
    call = common.make_call(*(cfg or (None, None)))
    for _ in range(3):
        common.reset([lay])
        call(lay)
    torch.cuda.synchronize()
    print("N", n, "done", flush=True)
