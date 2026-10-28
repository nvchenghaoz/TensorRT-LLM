"""Localize the repeat-determinism break seen in job 117 (FP8R on, prompts at T=80 change after a T=160 request).

Each case runs one VisualGen process with order: P0 (T=80), P0, P1 (T=160), P0, P0, and reports whether
the P0 images are bitwise identical before and after the T=160 request.
Cases (env of the child):
  on            FP8R on, default Dynamo config
  on_nodyn      FP8R on, torch._dynamo.config.automatic_dynamic_shapes=False in the worker
  off           FP8R off (main's BF16 path), default
The worker is started with multiprocessing spawn, which re-imports this module as __mp_main__, so the
module-level config below also applies inside the worker.
"""
import json
import os
import subprocess
import sys

if os.environ.get("FP8X_NO_AUTODYN") == "1":
    import torch._dynamo

    torch._dynamo.config.automatic_dynamic_shapes = False

P0 = "A red fox standing in fresh snow at sunrise"
P1 = (
    "An isometric illustration of a small futuristic harbor town at dusk: terraced houses with "
    "solar roofs climbing a green hill, a lighthouse with a rotating beam, fishing boats with "
    "glowing lanterns moored along a curved stone pier, a cable car crossing the bay, market "
    "stalls selling fruit under striped awnings, people walking dogs on the promenade, seagulls, "
    "soft pastel colors"
)
ORDER = [P0, P0, P1, P0, P0]


def child(out_dir, ckpt):
    import torch

    from tensorrt_llm import VisualGen, VisualGenArgs

    yaml_path = os.path.join(out_dir, "args.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            "parallel_config:\n  tp_size: 1\nattention_config:\n  backend: VANILLA\n"
            "cuda_graph_config:\n  enable: false\ncompilation_config:\n"
            "  resolutions: [[1024, 1024]]\n  num_frames: [1]\n  modes: [t2i]\n"
        )
    config = VisualGenArgs.from_yaml(yaml_path)
    config.pipeline_config["variant"] = "nvfp4"
    model = VisualGen(model=ckpt, args=config)
    imgs = []
    for i, prompt in enumerate(ORDER):
        params = model.default_params
        params.height, params.width, params.seed = 1024, 1024, 11
        out = model.generate(inputs=prompt, params=params)
        imgs.append(out.image.detach().cpu())
        print("REQ", i, "denoise", getattr(out.metrics, "denoise", None), flush=True)
    p0 = [imgs[i] for i in (0, 1, 3, 4)]
    res = {
        "p0_before_equal": torch.equal(p0[0], p0[1]),
        "p0_after_equal": torch.equal(p0[2], p0[3]),
        "p0_before_vs_after_equal": torch.equal(p0[0], p0[2]),
        "p0_before_vs_after_maxdiff": (p0[0].float() - p0[2].float()).abs().max().item(),
    }
    print("RESULT", json.dumps(res), flush=True)


def parent(ckpt):
    base = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/fp8x_diag")
    cases = {
        "on": {"TRTLLM_FLUX3_SM120_FP8_EXCLUDED": "1"},
        "on_nodyn": {"TRTLLM_FLUX3_SM120_FP8_EXCLUDED": "1", "FP8X_NO_AUTODYN": "1"},
        "off": {"TRTLLM_FLUX3_SM120_FP8_EXCLUDED": "0"},
    }
    only = os.environ.get("FP8X_CASES")
    summary = {}
    for name, extra in cases.items():
        if only and name not in only.split(","):
            continue
        d = os.path.join(base, name)
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, TORCH_LOGS="recompiles", **extra)
        print("case", name, flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "child", d, ckpt],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        open(os.path.join(d, "log.txt"), "w").write(proc.stdout)
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith(("REQ", "RESULT")):
                print(name, line, flush=True)
            if line.startswith("RESULT"):
                result = json.loads(line[len("RESULT "):])
        summary[name] = {
            "rc": proc.returncode,
            "result": result,
            "recompile_msgs": proc.stdout.count("Recompiling function"),
        }
    print(json.dumps({"passed": True, "cases": summary}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    else:
        parent(sys.argv[1])
