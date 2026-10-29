"""Branch 186 (idea 270): N fresh pipeline starts of this tree, as served.

Each start is a fresh child process: PipelineLoader(...).load() with the engine.json overlay
(torch.compile + autotune, CUDA graphs on, warmup [[1024,1024]] t2i: the autotune pass, then the
capture pass -- exactly the server's warmup that died in merge job 1987), then 2 requests (the
second is compared bitwise with the first). CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 (lightweight) so a
CUDA 700 leaves a coredump naming the faulting kernel. The child also counts G3 launches seen by
the router (fmha_v2_flux3 module calls).

usage: python b186_starts.py <checkpoint> <n_starts> [ENV=VAL ...]
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


def child(ckpt, out_path):
    import hashlib

    import torch

    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    calls = {}
    orig = lib.fmha_v2_flux3_d128
    orig_nfo = getattr(lib, "fmha_v2_flux3_d128_nvfp4_out", None)
    tags = {}

    def mod_tag(module):
        if not tags:
            base = tuple(lib.ROUTED_LEVERS)
            for lev in (base, (*base, "nvfp4_out")):
                try:
                    tags[id(lib.get_flux3_d128_module(lev))] = lib._tag(lev, 0)
                except Exception as exc:  # noqa: BLE001
                    log(f"[child] tag map {lev}: {exc}")
        return tags.get(id(module), str(id(module)))

    def counted(query, *a, module=None, **kw):
        key = str(("bf16", mod_tag(module), tuple(query.shape)))
        calls[key] = calls.get(key, 0) + 1
        return orig(query, *a, module=module, **kw)

    def counted_nfo(query, *a, module=None, **kw):
        key = str(("nfo", mod_tag(module), tuple(query.shape)))
        calls[key] = calls.get(key, 0) + 1
        return orig_nfo(query, *a, module=module, **kw)

    lib.fmha_v2_flux3_d128 = counted
    if orig_nfo is not None:
        lib.fmha_v2_flux3_d128_nvfp4_out = counted_nfo

    engine = json.load(open(os.path.join(HERE, "..", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    args = VisualGenArgs(
        model=ckpt,
        pipeline_config=pipeline_config,
        compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
        **engine,
    )
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    t_load = time.time() - t0
    log(f"[child] loaded+warmup {t_load:.0f}s ROUTED_LEVERS={lib.ROUTED_LEVERS} calls={calls}")

    def generate(prompt, seed):
        with torch.inference_mode():
            ctx = pipe._encode_prompt(prompt, None)
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            latent, ids = pipe._prepare_t2i(
                ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
            )
            img = pipe._package_output(pipe._decode_latents((latent, ids))).image
        torch.cuda.synchronize()
        return img.clone().cpu()

    shas = []
    for i in range(2):
        t = time.time()
        img = generate("A red fox standing in fresh snow at sunrise", 11)
        shas.append(hashlib.sha256(img.contiguous().view(-1).numpy().tobytes()).hexdigest()[:16])
        log(f"[child] request {i}: {time.time() - t:.2f}s sha {shas[-1]}")
    json.dump({"ok": True, "load_s": t_load, "shas": shas, "repeat_equal": shas[0] == shas[1],
               "routed_levers": list(lib.ROUTED_LEVERS), "router_calls": calls},
              open(out_path, "w"))


def parent(ckpt, n, extra_env):
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    os.makedirs(art, exist_ok=True)
    rows = []
    for i in range(n):
        out = os.path.join(art, f"start_{i}.json")
        env = dict(os.environ, CUDA_ENABLE_COREDUMP_ON_EXCEPTION="1",
                   CUDA_ENABLE_LIGHTWEIGHT_COREDUMP="1",
                   CUDA_COREDUMP_FILE=os.path.join(art, f"core_{i}_%p"), **extra_env)
        t = time.time()
        log(f"=== start {i + 1}/{n} env {extra_env}")
        p = subprocess.run([sys.executable, __file__, "--child", ckpt, out], env=env,
                           capture_output=True, text=True, timeout=1500)
        with open(os.path.join(art, f"start_{i}.log"), "w") as f:
            f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
        tail = [ln for ln in (p.stdout + p.stderr).splitlines()
                if "[child]" in ln or "CUDA" in ln or "rror" in ln][-12:]
        res = json.load(open(out)) if os.path.exists(out) else {"ok": False}
        res.update(rc=p.returncode, wall_s=round(time.time() - t, 1),
                   cuda_error=("illegal memory access" in p.stderr or "CUDA error" in p.stderr
                               or "cudaError" in p.stderr))
        rows.append(res)
        log(f"[start {i}] rc {p.returncode} ok {res.get('ok')} cuda_error {res['cuda_error']} "
            f"wall {res['wall_s']}s repeat_equal {res.get('repeat_equal')}")
        for ln in tail:
            log("   ", ln[:400])
    cores = [f for f in os.listdir(art) if f.startswith("core_")]
    good = sum(1 for r in rows if r.get("ok") and r["rc"] == 0 and not r["cuda_error"]
               and r.get("repeat_equal"))
    print(json.dumps({"passed": good == n, "starts": n, "clean_starts": good, "coredumps": cores,
                      "env": extra_env, "rows": rows}))


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        extra = dict(kv.split("=", 1) for kv in sys.argv[3:])
        parent(sys.argv[1], int(sys.argv[2]), extra)
