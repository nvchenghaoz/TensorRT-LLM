#!/bin/bash
# Diff branch 23's FlashInfer overlay against pinned FlashInfer, and check main's copies of the files branch 23 edits.
R=/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048
OUT=${AUTO_DEV_ARTIFACTS:-/tmp}
T=$R/runner/repos/trtllm.git
FI=$R/runner/repos/flashinfer.git
S=b5fd9b64b59ac796ee26b972d561ea7a46505069
git -C $T rev-parse fbc5e8d663d1
git -C $T show $S:experiments/ascent-flashinfer/flashinfer/jit/gemm/core.py > $OUT/b23_core.py
git -C $FI show 775207deb34055667df89ed618fb4b06fc84de74:flashinfer/jit/gemm/core.py > $OUT/pinned_core.py
diff -u $OUT/pinned_core.py $OUT/b23_core.py > $OUT/core_overlay.diff; echo "core diff lines: $(wc -l < $OUT/core_overlay.diff)"
cat $OUT/core_overlay.diff
for f in tensorrt_llm/_torch/models/modeling_gemma4.py tensorrt_llm/_torch/kernels/fused_ops/gate_up_gelu_fp4_chunked.py tensorrt_llm/_torch/kernels/fused_ops/gelu_tanh_mul_fp4_quant.py tensorrt_llm/_torch/modules/linear.py; do
  a=$(git -C $T show 8c2c2cb5:$f | sha256sum | cut -c1-16); b=$(sha256sum < $f | cut -c1-16); echo "$f 8c2c2cb5=$a worktree(main)=$b $([ $a = $b ] && echo SAME || echo DIFF)"
done
ls experiments/ascent-flashinfer -R | head -40
# Files under the FI include/csrc paths branch 23 adds: do they exist in pinned FlashInfer?
for p in csrc/nvfp4_gelu_gemm_sm120.cu include/flashinfer/gemm/nvfp4_gelu_gemm_sm120.cuh; do git -C $FI cat-file -e 775207deb34055667df89ed618fb4b06fc84de74:$p 2>/dev/null && echo "$p exists in pinned" || echo "$p new"; done
echo '{"passed": true}'
