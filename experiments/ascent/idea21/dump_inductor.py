"""Idea 21 step 0: dump main's Inductor softmax kernels for _attention_tc at T=65536.

Compiles Flux3ImageAttnBlock._attention_tc exactly as the served VAEs compile it
(decoder: bf16, fp8_pv=True, options layout_optimization=False + emulate_precision_casts;
encoder: fp32 inputs, fp8_pv=False, layout_optimization=False) on synthetic q/k/v at the
2048^2 mid-block shape [1, 512, 256, 256] channels_last, and writes the generated code.
"""

import os
import subprocess
import sys
import time

import torch
from torch._inductor.utils import run_and_get_code

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae.image_vae import (
    Flux3ImageAttnBlock,
)

T0 = time.time()
OUT = os.environ.get("AUTO_DEV_ARTIFACTS", ".")
H = int(sys.argv[1]) if len(sys.argv) > 1 else 256


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


try:
    log(subprocess.run(["git", "log", "-3", "--oneline"], capture_output=True, text=True).stdout)
except Exception as ex:  # noqa: BLE001
    log("git log failed", ex)


def mk(dt, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(1, 512, H, H, device="cuda", generator=g) * 2.0
    return x.to(dt).contiguous(memory_format=torch.channels_last)


fn = Flux3ImageAttnBlock._attention_tc
with torch.inference_mode():
    for name, dt, fp8, opts in [
        ("decoder", torch.bfloat16, True, {"layout_optimization": False, "emulate_precision_casts": True}),
        ("encoder", torch.float32, False, {"layout_optimization": False}),
    ]:
        torch._dynamo.reset()
        q, k, v = mk(dt, 1), mk(dt, 2), mk(dt, 3)
        cf = torch.compile(lambda a, b, c: fn(a, b, c, fp8), dynamic=False, options=opts)
        _, codes = run_and_get_code(cf, q, k, v)
        path = os.path.join(OUT, f"inductor_{name}.py")
        with open(path, "w") as f:
            f.write("\n\n#######\n\n".join(codes))
        log(name, "wrote", path, [len(c) for c in codes])
        for c in codes:
            parts = c.split("async_compile.triton(")
            for p in parts[1:]:
                body = p.split("''', device_str")[0]
                print(f"===== {name} kernel =====")
                print(body[:6000])
            for line in c.splitlines():
                if ".run(" in line or "extern_kernels" in line or "call(" in line:
                    print("CALL:", line.strip()[:300])
        del q, k, v
        torch.cuda.empty_cache()
print('{"passed": true}')
