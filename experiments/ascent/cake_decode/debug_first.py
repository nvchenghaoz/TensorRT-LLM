"""Why is the FIRST call of a fresh CakeDecode wrong? (branch 129 debugging)"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2.cake_decode import CakeDecode  # noqa: E402


def call(dec, q, sink, pools, sel, has_main):
    o = torch.full_like(q, float("nan"))
    dec(q, pools[0], sel["swa"], pools[1] if has_main else None,
        sel["logical"] if has_main else None, sink, C.SCALE, o,
        mapping=sel["mapping"] if has_main else None)
    torch.cuda.synchronize()
    return o


def err(o, ref):
    d = (o.double() - ref).abs().nan_to_num(1e30)
    bad = d.amax(-1) > 1.0  # [count, heads]
    return dict(err=float(d.max()), bad_rows=int(bad.sum()), bad_tokens=bad.any(-1).nonzero().flatten().tolist(),
                bad_heads=bad.any(0).nonzero().flatten().tolist())


def main():
    pools = C.build_pools(31)
    res = {}
    for width in (640, 128):
        for count in (2, 4):
            sel = C.build_selection(count, 7 + count, width=width)
            has_main = width > 128
            q, sink = C.make_q(count, 3)
            ref = C.reference(q, sink, *pools, sel)
            main_w = 512 if has_main else 0
            r = {}
            # A: fresh instance, first call
            dec = CakeDecode()
            r["A_first"] = err(call(dec, q, sink, pools, sel, has_main), ref)
            r["A_second"] = err(call(dec, q, sink, pools, sel, has_main), ref)
            # B: pre-allocated scratch, synced, finite poison, then first call
            dec = CakeDecode()
            buf = dec._scratch(count, 16, 128, main_w, q.device)
            buf["mid_out"].fill_(7777.0)
            buf["mid_lse"].fill_(12345.0)
            torch.cuda.synchronize()
            o = call(dec, q, sink, pools, sel, has_main)
            r["B_prealloc_first"] = err(o, ref)
            r["B_lse_unwritten"] = int((buf["mid_lse"] == 12345.0).sum())
            r["B_out_unwritten"] = int((buf["mid_out"].float() == 7777.0).sum())
            # C: pre-allocated scratch, zero-filled, first call
            dec = CakeDecode()
            buf = dec._scratch(count, 16, 128, main_w, q.device)
            buf["mid_out"].zero_()
            buf["mid_lse"].zero_()
            buf["lse"].zero_()
            torch.cuda.synchronize()
            r["C_zero_first"] = err(call(dec, q, sink, pools, sel, has_main), ref)
            # D: first call, but the index buffers checked against the reference mapping
            dec = CakeDecode()
            o = call(dec, q, sink, pools, sel, has_main)
            (buf,) = list(dec._buffers.values())
            swa_ref, main_ref = C.map_reference(sel)
            r["D_first"] = err(o, ref)
            r["D_swa_ok"] = bool(torch.equal(buf["swa"].long(), swa_ref))
            r["D_main_ok"] = bool(torch.equal(buf["main"].long(), main_ref)) if has_main else None
            r["D_lse_finite"] = bool(torch.isfinite(buf["lse"]).all())
            key = f"w{width}/c{count}"
            res[key] = r
            C.log(key, json.dumps(r))
    print(json.dumps({"passed": True, "report": res}))


if __name__ == "__main__":
    main()
