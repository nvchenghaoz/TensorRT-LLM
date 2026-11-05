"""Which split-scratch entries does the Cake decode leave unwritten? (branch 129 debugging)

The merge skips a split whose mid_lse is NaN (NaN > -1e29 is false), so NaN poisoning
hides unwritten splits. Here mid_lse is poisoned with a FINITE sentinel and mid_out with
another, then we count survivors per split after one call, for the public and the direct
binding path, fp32 and bf16 partials.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2.cake_decode import CakeDecode  # noqa: E402

LSE_POISON = 12345.0
OUT_POISON = 7777.0


def one(dec, q, sink, pools, sel, has_main, path):
    count = q.shape[0]
    (buf,) = list(dec._buffers.values())
    if path == "public":
        buf.pop("launch", None)
    buf["mid_out"].fill_(OUT_POISON)
    buf["mid_lse"].fill_(LSE_POISON)
    o = torch.full_like(q, float("nan"))
    dec(q, pools[0], sel["swa"], pools[1] if has_main else None,
        sel["logical"] if has_main else None, sink, C.SCALE, o,
        mapping=sel["mapping"] if has_main else None)
    torch.cuda.synchronize()
    took_direct = path == "direct" and buf.get("launch") is not None
    lse = buf["mid_lse"]
    mo = buf["mid_out"].float()
    lse_left = (lse == LSE_POISON).sum(dim=(0, 1)).tolist()  # per split
    out_left = (mo == OUT_POISON).sum(dim=(0, 1, 3)).tolist()  # per split
    # per-head survivors for split 0
    lse_left_heads = (lse == LSE_POISON).sum(dim=(0, 2)).tolist()
    return o, dict(path=path, took_direct=took_direct, lse_unwritten_per_split=lse_left,
                   out_unwritten_per_split=out_left, lse_unwritten_per_head=lse_left_heads,
                   splits_alloc=int(lse.shape[2]),
                   out_nan=int(torch.isnan(o).sum()))


def main():
    pools = C.build_pools(31)
    res = {}
    for f32 in (True, False):
        for width in (640, 128):
            for count in (1, 2, 4):
                sel = C.build_selection(count, 7 + count, width=width)
                has_main = width > 128
                q, sink = C.make_q(count, 3)
                ref = C.reference(q, sink, *pools, sel)
                dec = CakeDecode(f32_partials=f32)
                o0 = torch.empty_like(q)
                dec(q, pools[0], sel["swa"], pools[1] if has_main else None,
                    sel["logical"] if has_main else None, sink, C.SCALE, o0,
                    mapping=sel["mapping"] if has_main else None)
                torch.cuda.synchronize()
                rows = []
                for path in ("public", "direct", "public"):
                    o, info = one(dec, q, sink, pools, sel, has_main, path)
                    info["err"] = float((o.double() - ref).abs().nan_to_num(1e30).max())
                    rows.append(info)
                key = f"f32={int(f32)}/w{width}/c{count}"
                res[key] = rows
                C.log(key, json.dumps(rows))
    print(json.dumps({"passed": True, "report": res}))


if __name__ == "__main__":
    main()
