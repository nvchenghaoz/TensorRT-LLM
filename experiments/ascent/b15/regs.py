"""Branch 15: NUM_REGS and local bytes of the warp adaLN+NVFP4 kernel (both served variants,
LN_ID=0/1), built by THIS tree. Run in both trees (run_gpu_pair) to show the index edit leaves
the register budget unchanged. Also runs one bool-select call per variant at M=16464 and prints
the output sha256 (t2i form, comparable across trees)."""

import hashlib
import json

import torch
from cuda.bindings import driver

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as warp


def main():
    d, m = 3072, 16464
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(1, m, d, generator=g, device="cuda").to(torch.bfloat16)
    sc = (torch.randn(1, 2, d, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    sh = (torch.randn(1, 2, d, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    sel = (torch.arange(m, device="cuda") >= 16384).view(1, m, 1)
    gs = torch.tensor([168.0], device="cuda")
    w1 = torch.ones(d, dtype=torch.bfloat16, device="cuda")
    b0 = torch.zeros(d, dtype=torch.bfloat16, device="cuda")
    w2 = w1.clone()
    b2 = b0.clone()
    warp.register_identity_affine(w1, b0)  # LN_ID=1 variant; (w2, b2) unregistered -> LN_ID=0
    out = {}
    for tag, (w, b) in (("ln_id1", (w1, b0)), ("ln_id0", (w2, b2))):
        xm, fp4, sf = warp.launch(x, sel, sc, sh, w, b, gs, 1e-6, store_x_mod=True)
        torch.cuda.synchronize()
        h = hashlib.sha256()
        for t in (xm, fp4, sf):
            h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        out[tag] = h.hexdigest()
    regs = {}
    for key, fn in warp._MODULES.items():
        n = driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)[1]
        lb = driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn)[1]
        regs[str(key)] = {"regs": int(n), "local_bytes": int(lb)}
    print(json.dumps({"passed": True, "warp_kernel": regs, "t2i_form_sha256": out}), flush=True)


if __name__ == "__main__":
    main()
