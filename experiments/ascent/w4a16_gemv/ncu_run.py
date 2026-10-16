# ncu driver: one (N,K) NVFP4 GEMV copy, trtllm::nvfp4_w4a16_gemv and Marlin at M=1..4.
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401

dev = torch.device("cuda")
N, K = (int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "5120,6144").split(","))
wq = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
sf = (torch.rand(N, K // 16, device=dev) + 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
for m in (1, 2, 3, 4):
    x = torch.randn(m, K, device=dev).to(torch.bfloat16)
    for _ in range(2):
        torch.ops.trtllm.nvfp4_w4a16_gemv(x, wq, sf, 1.0)
torch.cuda.synchronize()
print("done", flush=True)
