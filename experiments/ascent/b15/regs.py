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
    # u8 K=3 index variant (candidate tree only; the base tree has no SEL_INDEX build)
    if "SEL_INDEX" in warp._SRC:
        ar = torch.arange(m, device="cuda")
        idx = ((ar >= 8192).to(torch.uint8) + (ar >= 16384).to(torch.uint8)).view(1, m, 1)
        sc3 = torch.cat([sc, sc[:, :1] * 0.5], 1).contiguous()
        sh3 = torch.cat([sh, sh[:, :1] * 0.5], 1).contiguous()
        for ww, bb in ((w1, b0), (w2, b2)):
            warp.launch(x, idx, sc3, sh3, ww, bb, gs, 1e-6, store_x_mod=False)
        torch.cuda.synchronize()
    # cubin sha256 of the bool-select builds (same NVRTC options as _compile)
    from cuda.bindings import nvrtc

    cubins = {}
    for tag, defs in (("ln_id0", ("MINB=12", "X_POLICY=2")),
                      ("ln_id1", ("MINB=12", "X_POLICY=2", "LN_ID=1"))):
        prog = warp._check(nvrtc.nvrtcCreateProgram(warp._SRC.encode(), b"flux3_adaln_nvfp4.cu",
                                                    0, [], []))
        opts = [b"--gpu-architecture=sm_120a", b"--fmad=false", b"--prec-div=true",
                b"--prec-sqrt=true", b"--ftz=false", b"-std=c++17", b"-DWARPS=1",
                b"-DUSE_PDL=0"] + [f"-D{d_}".encode() for d_ in defs]
        nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        size = warp._check(nvrtc.nvrtcGetCUBINSize(prog))
        cub = b" " * size
        warp._check(nvrtc.nvrtcGetCUBIN(prog, cub))
        nvrtc.nvrtcDestroyProgram(prog)
        cubins[tag] = hashlib.sha256(cub).hexdigest()
    regs = {}
    for key, fn in warp._MODULES.items():
        n = driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)[1]
        lb = driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn)[1]
        regs[str(key)] = {"regs": int(n), "local_bytes": int(lb)}
    print(json.dumps({"passed": True, "warp_kernel": regs, "t2i_form_sha256": out,
                      "bool_cubin_sha256": cubins}), flush=True)


if __name__ == "__main__":
    main()
