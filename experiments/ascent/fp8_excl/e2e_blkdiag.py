"""Localize the FP8R repeat break to a block: per-block input/output checksums in the VisualGen worker.

The worker is started with multiprocessing spawn and re-imports this module as __mp_main__, so the
patch below (applied at import) runs inside the worker: after Flux3Pipeline.torch_compile, forward
hooks are registered on every transformer block (outside the compiled region); run_inference is
wrapped to dump the per-request records to $FP8X_DIAG_DIR/req_<n>.json.
Order: P0, P1, P2 (T=160), P3, P0, P1. The parent compares P0 (req 0 vs 4) and P1 (req 1 vs 5).
"""
import json
import os
import subprocess
import sys

DIAG_DIR = os.environ.get("FP8X_DIAG_DIR")

if DIAG_DIR and os.environ.get("FP8X_DIAG_CHILD") == "1":
    import torch

    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as _pf

    _records = []
    _state = {"req": 0}

    def _sig(t):
        if isinstance(t, (tuple, list)):
            t = t[0]
        if not isinstance(t, torch.Tensor):
            return None
        f = t.detach().float()
        return [f.sum().item(), f.abs().sum().item(), list(t.shape)]

    def _hook(name):
        def fn(mod, args, kwargs, out):
            x = args[0] if args else next(iter(kwargs.values()))
            _records.append([name, _sig(x), _sig(out)])

        return fn

    _orig_compile = _pf.Flux3Pipeline.torch_compile

    def _patched_compile(self):
        _orig_compile(self)
        for lname, blocks in (
            ("single", self.transformer.single_blocks),
            ("image", self.transformer.content_mode_blocks["image"]),
            ("txt", self.transformer.txt_mode_blocks),
        ):
            for i, b in enumerate(blocks):
                b.register_forward_hook(_hook(f"{lname}.{i}"), with_kwargs=True)

    _pf.Flux3Pipeline.torch_compile = _patched_compile
    _orig_run = _pf.Flux3Pipeline.run_inference

    def _patched_run(self, req):
        _records.clear()
        out = _orig_run(self, req)
        with open(os.path.join(DIAG_DIR, f"req_{_state['req']}.json"), "w") as f:
            json.dump(_records, f)
        _state["req"] += 1
        return out

    _pf.Flux3Pipeline.run_inference = _patched_run

P = [
    "A red fox standing in fresh snow at sunrise",
    "A cozy reading nook with a green velvet armchair, a brass floor lamp, stacks of old books "
    "on a wooden side table, a sleeping tabby cat curled on a knitted blanket, rain streaking "
    "down a tall window, warm evening light, photorealistic interior photograph",
    "An isometric illustration of a small futuristic harbor town at dusk: terraced houses with "
    "solar roofs climbing a green hill, a lighthouse with a rotating beam, fishing boats with "
    "glowing lanterns moored along a curved stone pier, a cable car crossing the bay, market "
    "stalls selling fruit under striped awnings, people walking dogs on the promenade, seagulls, "
    "soft pastel colors, clean vector lines, high detail, wide establishing view, gentle haze "
    "over the water and the first stars appearing in a violet sky above distant mountains",
    "Close-up macro photo of a dewdrop on a fern leaf",
]
ORDER = [(0, 11), (1, 23), (2, 37), (3, 41), (0, 11), (1, 23)]


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
    for i, (p, seed) in enumerate(ORDER):
        params = model.default_params
        params.height, params.width, params.seed = 1024, 1024, seed
        out = model.generate(inputs=P[p], params=params)
        imgs.append(out.image.detach().cpu())
        print("REQ", i, flush=True)
    print(
        "IMG",
        json.dumps({"p0": torch.equal(imgs[0], imgs[4]), "p1": torch.equal(imgs[1], imgs[5])}),
        flush=True,
    )


def compare(d, a, b):
    ra = json.load(open(os.path.join(d, f"req_{a}.json")))
    rb = json.load(open(os.path.join(d, f"req_{b}.json")))
    out = {"n_records": [len(ra), len(rb)], "first_diff": None}
    for k, (x, y) in enumerate(zip(ra, rb)):
        if x != y:
            out["first_diff"] = {"index": k, "a": x, "b": y, "prev": ra[k - 1] if k else None}
            break
    return out


def parent(ckpt):
    base = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/fp8x_blk")
    res = {}
    for mode in os.environ.get("FP8X_MODES", "on").split(","):
        d = os.path.join(base, mode)
        os.makedirs(d, exist_ok=True)
        env = dict(
            os.environ,
            FP8X_DIAG_DIR=d,
            FP8X_DIAG_CHILD="1",
            TRTLLM_FLUX3_SM120_FP8_EXCLUDED="1" if mode == "on" else "0",
        )
        proc = subprocess.run(
            [sys.executable, __file__, "child", d, ckpt],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        open(os.path.join(d, "log.txt"), "w").write(proc.stdout)
        img = None
        for line in proc.stdout.splitlines():
            if line.startswith(("REQ", "IMG")):
                print(mode, line, flush=True)
            if line.startswith("IMG"):
                img = json.loads(line[4:])
        res[mode] = {"rc": proc.returncode, "img_equal": img}
        if proc.returncode == 0:
            res[mode]["p0"] = compare(d, 0, 4)
            res[mode]["p1"] = compare(d, 1, 5)
        print(mode, json.dumps(res[mode])[:3000], flush=True)
    print(json.dumps({"passed": True, "res": res})[:6000])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    else:
        parent(sys.argv[1])
