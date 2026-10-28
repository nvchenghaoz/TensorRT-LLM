"""Profile one served-like denoise with the tail pair on or off (profiler region = 2 requests).

usage: python prof.py <checkpoint> <on|off>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e2e_check import PROMPT_A, Runner, log  # noqa: E402


def main():
    import torch

    ckpt, mode = sys.argv[1], sys.argv[2]
    r = Runner(ckpt)
    r.set_flag(mode == "on")
    for _ in range(2):
        log(f"[warm] denoise {r.gen(PROMPT_A)[2]:.1f} ms")
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(2):
        torch.cuda.nvtx.range_push("request")
        log(f"[prof] denoise {r.gen(PROMPT_A)[2]:.1f} ms")
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
