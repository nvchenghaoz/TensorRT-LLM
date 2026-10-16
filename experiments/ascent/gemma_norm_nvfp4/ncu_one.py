"""ncu target: a few eager calls of the fused norm+quant kernel and of main's
FlashInfer gemma add+RMSNorm + fp4_quantize at M=1 (branch 57)."""
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401
from tensorrt_llm._torch.modules import fused_gemma_add_rmsnorm_nvfp4  # noqa

H, EPS = 5120, 1e-6
dev = "cuda"
print("TREE_COMMIT=see job receipt", flush=True)
gs = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device=dev)
w = (torch.randn(H, device=dev) * 0.3).to(torch.bfloat16)
for m in (1, 4):
    x = (torch.randn(m, H, device=dev) * 0.7).to(torch.bfloat16)
    r = (torch.randn(m, H, device=dev) * 3.0).to(torch.bfloat16)
    for _ in range(3):
        torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(x, r.clone(), w, EPS,
                                                       gs)
        x2 = x.clone()
        torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(
            x2, r.clone(), w, EPS)
        torch.ops.trtllm.fp4_quantize(x2, gs, 16, False)
torch.cuda.synchronize()
print("done", flush=True)
