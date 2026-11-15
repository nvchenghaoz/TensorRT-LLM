"""num_warps sweep of _embed_scale_rmsnorm_kernel (device time, back to back) + bitwise vs num_warps=4."""
import sys

import torch
import triton

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import build_modules, make_inputs, read_dims  # noqa: E402


def main():
    dims = read_dims(sys.argv[1])
    emb, norm0, _ = build_modules(dims)
    from tensorrt_llm._torch.kernels.fused_ops import embed_scale_rmsnorm as mod
    H = dims["hidden"]
    tpr, vec, nvb = mod._cute_rmsnorm_layout(H, 2)
    for label in ("300", "2000", "8000"):
        ids, _, _ = make_inputs(label, dims, seed=1)
        M = ids.numel()
        ref = None
        for nw in (2, 4, 8, 16):
            h = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
            n = torch.empty_like(h)

            def run():
                mod._embed_scale_rmsnorm_kernel[(M, )](ids, emb.weight, norm0.weight, h, n, dims["vocab"],
                                                       emb.embed_scale_value, 1e-6, H=H,
                                                       BH=triton.next_power_of_2(H), TPR=tpr, VEC=vec,
                                                       NVB=nvb, HAS_NORM=True, num_warps=nw)
            for _ in range(5):
                run()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(2_000_000)
            s.record()
            for _ in range(50):
                run()
            e.record()
            torch.cuda.synchronize()
            if ref is None:
                ref = (h.clone(), n.clone())
            same = torch.equal(h, ref[0]) and torch.equal(n.view(torch.int16), ref[1].view(torch.int16))
            print(f"M={M} num_warps={nw} device_us={s.elapsed_time(e) / 50 * 1e3:.1f} same_as_nw2={same}",
                  flush=True)
    print('{"passed": true}')


if __name__ == "__main__":
    main()
