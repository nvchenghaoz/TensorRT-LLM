"""Branch 54 / idea 84: diagnostic builds of flux3_qkv_epi_fp8_sm120.cu (STEP 0 split probes).

Each variant is the same source compiled with extra -D flags under its own module name; the served
module (flux3_qkv_epi_fp8.module()) is built with none of them.
"""

import time
from concurrent.futures import ThreadPoolExecutor

VARIANTS = {
    "noop": ["-DFLUX3_EPI_DIAG_NOOP"],
    "nocs": ["-DFLUX3_EPI_DIAG_NOCS"],
    "nox": ["-DFLUX3_EPI_DIAG_NOX"],
    "novst": ["-DFLUX3_EPI_DIAG_NOVST"],
    "noqkst": ["-DFLUX3_EPI_DIAG_NOQKST"],
    "nosf": ["-DFLUX3_EPI_DIAG_NOSF"],
}


def spec_for(name, defines):
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    return gen_jit_spec(
        f"flux3_qkv_epi_b54_{name}",
        [jit_env.FLASHINFER_CSRC_DIR / "flux3_qkv_epi_fp8_sm120.cu"],
        extra_cuda_cflags=nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"] + list(defines),
        extra_cflags=["-DFAST_BUILD"],
    )


def build_all(names=None, log=print, workers=4, variants=None):
    table = VARIANTS if variants is None else variants
    names = list(table) if names is None else names
    specs = {n: spec_for(n, table[n]) for n in names}
    t0 = time.time()

    def b(n):
        specs[n].build()
        log(f"built {n} {time.time() - t0:.0f}s")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(b, names))
    return {n: specs[n].load() for n in names}
