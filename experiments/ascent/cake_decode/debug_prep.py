"""Isolate prep-kernel vs fast-path errors (branch 129 debugging)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2.cake_decode import (  # noqa: E402
    CakeDecode,
    prepare_cake_indices,
)


def main():
    pools = C.build_pools(31)
    res = {}
    for width in (640, 128):
        for count in (1, 2, 4):
            sel = C.build_selection(count, 7 + count, width=width)
            swa_ref, main_ref = C.map_reference(sel)
            swa_out = torch.full((count, 128), -9, dtype=torch.int32, device="cuda")
            main_out = torch.full((count, 512), -9, dtype=torch.int32, device="cuda")
            has_main = width > 128
            prepare_cake_indices(sel["swa"], C.SWA_ROWS, swa_out,
                                 sel["logical"] if has_main else None, C.MAIN_ROWS,
                                 main_out if has_main else None,
                                 sel["mapping"] if has_main else None)
            torch.cuda.synchronize()
            swa_ok = torch.equal(swa_out.long(), swa_ref)
            main_ok = torch.equal(main_out.long(), main_ref) if has_main else None
            q, sink = C.make_q(count, 3)
            ref = C.reference(q, sink, *pools, sel)
            dec = CakeDecode()
            outs = []
            for i in range(10):
                # Poison every scratch buffer the merge reads (a load hoisted above
                # griddepcontrol.wait would read NaN / stale values here).
                for buf in dec._buffers.values():
                    buf["mid_out"].fill_(float("nan"))
                    buf["mid_lse"].fill_(float("nan"))
                o = torch.full_like(q, float("nan"))
                dec(q, pools[0], sel["swa"], pools[1] if has_main else None,
                    sel["logical"] if has_main else None, sink, C.SCALE, o,
                    mapping=sel["mapping"] if has_main else None)
                torch.cuda.synchronize()
                outs.append(o)
            errs = [(o.double() - ref).abs().max().item() for o in outs]
            key = f"w{width}/c{count}"
            res[key] = dict(swa_prep_ok=swa_ok, main_prep_ok=main_ok, errs=errs,
                            swa_mismatch=int((swa_out.long() != swa_ref).sum()),
                            main_mismatch=None if not has_main else int((main_out.long() != main_ref).sum()))
            C.log(key, res[key])
    print(json.dumps(res))


if __name__ == "__main__":
    main()
