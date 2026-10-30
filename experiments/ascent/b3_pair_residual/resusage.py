"""Print cuobjdump resource usage (REG / STACK / LOCAL) of the pair_residual kernels."""

import os
import subprocess

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_tail_one_launch as pr

for staged in ("0", "1"):
    os.environ["TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH_STAGED"] = staged
    os.environ.pop("TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH_STAGES", None)
    spec = pr.jit_spec()
    spec.build_and_load()
    path = spec.get_library_path()
    print("==", staged, path, flush=True)
    out = subprocess.run(["cuobjdump", "--dump-resource-usage", str(path)], capture_output=True,
                         text=True).stdout
    for line in out.splitlines():
        if "Function" in line or "REG" in line:
            print(line[:400], flush=True)
