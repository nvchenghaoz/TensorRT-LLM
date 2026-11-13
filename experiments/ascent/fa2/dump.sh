set -x
P=/opt/pytorch/pytorch
S=$P/third_party/flash-attention/csrc/flash_attn/src
D=${AUTO_DEV_ARTIFACTS:-/tmp/a}/src
mkdir -p $D
for f in flash_fwd_kernel.h flash_fwd_launch_template.h kernel_traits.h softmax.h utils.h block_info.h flash.h mask.h static_switch.h namespace_config.h hardware_info.h rotary.h philox_unpack.cuh flash_fwd_hdim128_bf16_sm80.cu dropout.h alibi.h; do cp $S/$f $D/ 2>&1; done
ls $S | grep -v '_sm80.cu' 
sed -n 180,200p $P/aten/src/ATen/CMakeLists.txt
sed -n 330,370p $P/aten/src/ATen/CMakeLists.txt > $D/aten_cmake_flash.txt
sed -n 1110,1140p $P/caffe2/CMakeLists.txt > $D/caffe2_cmake_flash.txt
grep -rn 'use_fast_math\|CMAKE_CUDA_FLAGS' $P/CMakeLists.txt $P/cmake/*.cmake $P/cmake/public/*.cmake $P/aten/src/ATen/CMakeLists.txt $P/caffe2/CMakeLists.txt | head -60 > $D/cuda_flags.txt
ls $P/third_party/flash-attention/csrc/cutlass/include/cutlass/version.h $P/third_party/cutlass/include/cutlass/version.h
grep -n 'define CUTLASS_M' $P/third_party/flash-attention/csrc/cutlass/include/cutlass/version.h $P/third_party/cutlass/include/cutlass/version.h
cp $P/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp $D/ 2>&1
cp $P/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.h $D/ 2>&1
ls $P/third_party/flash-attention/include 2>&1
python - <<'EOF'
import torch
print(torch.version.cuda, torch._C._cuda_getCompiledVersion() if hasattr(torch._C,'_cuda_getCompiledVersion') else '')
print(torch.__config__.show())
EOF
