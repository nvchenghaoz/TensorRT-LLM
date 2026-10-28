"""Branch 18: time/profile the q/k norm+RoPE pass alone at M=4176, with yardsticks."""

import json
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope import (
    qknorm_rope,
)

M, W = 4176, 3072


def time_fn(fn, iters=50):
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) * 1000 for a, b in zip(s, e))
    return round(t[len(t) // 2], 1)


def main():
    torch.manual_seed(0)
    buf = torch.randn(M, 3 * W, device="cuda").bfloat16()
    qn = (1 + 0.1 * torch.randn(128, device="cuda")).bfloat16()
    kn = (1 + 0.1 * torch.randn(128, device="cuda")).bfloat16()
    th = torch.rand(M, 64, device="cuda") * 6.28
    cs = torch.cat([th.cos(), th.sin()], -1).contiguous()
    pos = torch.arange(M, device="cuda", dtype=torch.int64)
    qk = buf[:, : 2 * W]

    def run():
        qknorm_rope(buf[:, :W], buf[:, W:2 * W], buf[:, 2 * W:], buf, qn, kn, cs, pos)

    res = dict(
        pass_inplace=time_fn(run),
        torch_mul_inplace_qk=time_fn(lambda: qk.mul_(1.0)),
        torch_copy_51MB=time_fn(lambda: qk.clone()),
    )
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        torch.cuda.synchronize()
        for _ in range(3):
            run()
        torch.cuda.synchronize()
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
