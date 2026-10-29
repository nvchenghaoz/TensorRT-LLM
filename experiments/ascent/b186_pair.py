"""Branch 186: run_gpu_pair arm = b165_g3.py --mode pair (the tree's own routing, graph of the
per-pass d128 set 5xS4096 + 28xS4176, kernel_ms = graph/33), plus the SASS hash / REG / STACK of
the tree's routed module WITHOUT g3 (== main's routed 4-warp kernel when G3=0) and, if the tree has
the lever, of the G3 kernel. The merged JSON is the last stdout line."""

import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import b165_g3  # noqa: E402


class _A:
    soak_s, blocks, block = 10.0, 16, 25


def main():
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            sys.__stdout__.flush()
            buf.write(s)
            return len(s)

    with contextlib.redirect_stdout(Tee()):
        b165_g3.mode_pair(_A())
    res = json.loads([ln for ln in buf.getvalue().splitlines() if ln.startswith("{")][-1])
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    lev = tuple(lib.ROUTED_LEVERS)
    lev_a = tuple(x for x in lev if x != "g3")
    lib.get_flux3_d128_module(lev_a)
    ra = b165_g3.resource_usage(lib, lev_a)
    sass = {"routed_no_g3": {"uri": ra.get("uri"), "sass_sha": ra.get("sass_sha"),
                             "functions": {k: v for k, v in ra.get("functions", {}).items()
                                           if "flux3" in k}}}
    if "g3" in lev:
        rb = b165_g3.resource_usage(lib, lev)
        sass["routed_g3"] = {"uri": rb.get("uri"), "sass_sha": rb.get("sass_sha"),
                             "functions": {k: v for k, v in rb.get("functions", {}).items()
                                           if "flux3" in k},
                             "stl_ldl": rb.get("sass_stl_ldl")}
    res["sass"] = sass
    print(json.dumps(res))


if __name__ == "__main__":
    main()
