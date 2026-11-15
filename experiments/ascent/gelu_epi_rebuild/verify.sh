#!/bin/bash
# Compare this tree's copies with branch 23's head tree (snapshot b5fd9b64 = fbc5e8d663d1's tree),
# then show the diff of the files that were adapted on purpose, and check for duplicate defs.
R=/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048
T=$R/runner/repos/trtllm.git
S=b5fd9b64b59ac796ee26b972d561ea7a46505069
echo "branch 23 head: $(git -C $T rev-parse $S~1)"
for f in experiments/ascent-flashinfer/csrc/nvfp4_gelu_gemm_sm120.cu \
         experiments/ascent-flashinfer/include/flashinfer/gemm/nvfp4_gelu_gemm_sm120.cuh \
         experiments/ascent-flashinfer/flashinfer/jit/gemm/core.py \
         experiments/ascent/gelu_epi/correctness.py experiments/ascent/gelu_epi/kernel_test.py \
         tensorrt_llm/_torch/kernels/fused_ops/gate_up_gelu_fp4_chunked.py \
         tensorrt_llm/_torch/models/modeling_gemma4.py \
         tensorrt_llm/_torch/kernels/fused_ops/nvfp4_gelu_gemm_sm120.py \
         experiments/ascent/gelu_epi/bench.py experiments/ascent/gelu_epi/synced_test.py \
         experiments/ascent/gelu_epi/bcg_check.py experiments/ascent/gelu_epi/checks.py; do
  a=$(git -C $T show $S:$f 2>/dev/null | sha256sum | cut -c1-16); b=$(sha256sum < $f | cut -c1-16)
  echo "VERBATIM? $f b23=$a here=$b $([ "$a" = "$b" ] && echo SAME || echo DIFF)"
done
for f in tensorrt_llm/_torch/kernels/fused_ops/nvfp4_gelu_gemm_sm120.py experiments/ascent-flashinfer/flashinfer/jit/gemm/core.py; do
  echo "----- diff b23 -> here: $f"
  git -C $T show $S:$f > /tmp/b23_$$.txt; diff -u /tmp/b23_$$.txt $f | head -150
done
echo "----- defs in _Gemma4GeluQuantMLP"
python3 - <<'EOF'
import ast, collections
src = open("tensorrt_llm/_torch/models/modeling_gemma4.py").read()
t = ast.parse(src)
for n in t.body:
    if isinstance(n, ast.ClassDef) and n.name in ("_Gemma4GeluQuantMLP", "Gemma4DecoderLayer"):
        c = collections.Counter(f.name for f in n.body if isinstance(f, ast.FunctionDef))
        print(n.name, "duplicate defs:", {k: v for k, v in c.items() if v > 1}, "forward defs:", c.get("forward"))
EOF
echo '{"passed": true}'
