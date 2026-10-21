"""Write the PTX of main's post_conv kernel (and the one-pass kernel) per specialization to artifacts."""
import os
import sys

import torch

sys.path.insert(0, "experiments/ascent/gdn_onepass")
import proto_lib as P  # noqa: E402

from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (  # noqa: E402
    _fused_gdn_post_conv_kernel,
)
from tensorrt_llm._torch.modules.mamba.gdn_prefill_onepass import (  # noqa: E402
    _gdn_prefill_onepass_kernel,
)

art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
for name, lens, nd in [("t2048", [2048], 0), ("t2048d1", [2048], 1), ("t85", [85], 0)]:
    c = P.make(lens, nd)
    P.run_main(c)
    P.run_one(c)
    torch.cuda.synchronize()
    for tag, kfn in (("main", _fused_gdn_post_conv_kernel), ("one", _gdn_prefill_onepass_kernel)):
        for cc in kfn.device_caches.values():
            k, kern = list(cc[0].items())[-1]
        with open(os.path.join(art, "%s_%s.ptx" % (tag, name)), "w") as f:
            f.write(kern.asm["ptx"])
        with open(os.path.join(art, "%s_%s.ttgir" % (tag, name)), "w") as f:
            f.write(kern.asm["ttgir"])
        print("wrote", tag, name, flush=True)
print('{"done": true}')
