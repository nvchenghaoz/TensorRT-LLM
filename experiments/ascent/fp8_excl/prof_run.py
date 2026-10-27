"""Small VisualGen run for nsys: benchmark engine defaults, warmup [[1024,1024]] t2i, then N requests of
the T=80 prompt. FP8R on/off via TRTLLM_FLUX3_SM120_FP8_EXCLUDED (1 = branch default, 0 = main's BF16 path)."""
import os
import sys

from tensorrt_llm import VisualGen, VisualGenArgs


def main(ckpt, n):
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/fp8x_prof")
    os.makedirs(out_dir, exist_ok=True)
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
    for i in range(n):
        params = model.default_params
        params.height, params.width, params.seed = 1024, 1024, 100 + i
        out = model.generate(inputs="A red fox standing in fresh snow at sunrise", params=params)
        print("REQ", i, "denoise", getattr(out.metrics, "denoise", None), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
