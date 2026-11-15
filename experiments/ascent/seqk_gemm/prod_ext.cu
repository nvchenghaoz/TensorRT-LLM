// Test-only torch binding that compiles the production kernel source
// (cpp/tensorrt_llm/kernels/sm120Bf16SeqKGemm/) so sweeps and checks can run without a full
// TensorRT-LLM rebuild. The model path uses torch.ops.trtllm.sm120_bf16_seqk_gemm instead.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "tensorrt_llm/kernels/sm120Bf16SeqKGemm/sm120Bf16SeqKGemm.cu"

torch::Tensor seqk(torch::Tensor x, torch::Tensor w, int64_t cfg)
{
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
    TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && x.size(1) == w.size(1));
    auto out = at::empty({x.size(0), w.size(0)}, x.options());
    int st = tensorrt_llm::kernels::sm120_seqk_gemm::run(x.data_ptr(), w.data_ptr(), out.data_ptr(), x.size(0),
        w.size(0), x.size(1), x.stride(0), w.stride(0), out.stride(0), cfg,
        at::cuda::getCurrentCUDAStream(x.get_device()));
    TORCH_CHECK(st == 0, "seqk status ", st);
    return out;
}

std::vector<int64_t> tile(int64_t cfg)
{
    int t = 0, w = 0;
    tensorrt_llm::kernels::sm120_seqk_gemm::tileShape(cfg, &t, &w);
    return {t, w};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("seqk", &seqk);
    m.def("tile", &tile);
}
