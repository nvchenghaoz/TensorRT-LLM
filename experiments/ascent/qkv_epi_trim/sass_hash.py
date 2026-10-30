"""Branch 54 / idea 84: hash the SASS of the fused-QKV GEMM kernel that the levers-off route
launches (flux3_qkv_epi_fp8.module().qkv3_epi_fp8), to show it is landing 22's kernel byte for
byte. Run in both trees (run_gpu_pair): base has one QkvFp8Epilogue GEMM kernel; the branch has
the kCompact=false one plus the compact one. Instruction text is hashed with addresses and the
function name stripped. Last stdout line: JSON."""

import hashlib
import json
import os
import re
import shutil
import subprocess


def main():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi

    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    epi.module()  # build (same spec as below)
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        "flux3_qkv_epi_fp8_sm120",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_qkv_epi_fp8_sm120.cu"],
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_cflags=["-DFAST_BUILD"],
    )
    so = str(spec.jit_library_path)
    assert os.path.exists(so), so
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    out = subprocess.run([cuobjdump, "-sass", so], capture_output=True, text=True).stdout
    funcs = re.split(r"\n\s*Function : ", out)
    res = {}
    for f in funcs[1:]:
        name, body = f.split("\n", 1)
        if "QkvFp8Epilogue" not in name:
            continue
        compact = "QkvFp8EpilogueI" in name and re.search(r"NoSmemWarpSpecializedLb1E", name)
        lines = []
        for ln in body.splitlines():
            m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(.*?);", ln)
            if m:
                lines.append(m.group(1).strip())
        h = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        res["compact" if compact else "full"] = {"sha256": h, "instructions": len(lines),
                                                 "name_tail": name[-120:]}
    print(json.dumps({"so": so, "kernels": res}))


if __name__ == "__main__":
    main()
