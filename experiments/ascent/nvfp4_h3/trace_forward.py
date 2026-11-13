"""Load MiniMax-H3 on 2 GPUs with the harness-like serve settings and run one
short request at the benchmark shape, so a profiler can capture denoise
forwards (use TLLM_PROFILE_VISUAL_GEN_START_STOP=<a>-<b> + capture=region).

  trace_forward.py <checkpoint> [--engine-json path] [--steps N]

The quant_config (if any) comes from the engine.json overlay, as in serving.
"""
import argparse
import json
import os
import time


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--engine-json", default="experiments/ascent-config/engine.json")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()

    from tensorrt_llm.visual_gen import VisualGen
    from tensorrt_llm.visual_gen.args import VisualGenArgs
    from tensorrt_llm.visual_gen.params import VisualGenParams

    overlay = {}
    if a.engine_json and os.path.exists(a.engine_json):
        with open(a.engine_json) as f:
            overlay = json.load(f)
    log("engine overlay:", overlay)
    cfg = dict(
        parallel_config=dict(tp_size=2, ulysses_size=1, text_encoder_tp_size=2,
                             parallel_vae_size=2),
        attention_config=dict(backend="VANILLA"),
        torch_compile_config=dict(enable=True, enable_autotune=True),
        cuda_graph_config=dict(enable=False),
        compilation_config=dict(resolutions=[(768, 1344)], num_frames=[124]),
    )
    if a.no_compile:
        cfg["torch_compile_config"] = dict(enable=False, enable_autotune=True)
    cfg.update(overlay)
    args = VisualGenArgs(**cfg)
    t0 = time.time()
    vg = VisualGen(model=a.checkpoint, args=args)
    log(f"loaded + warmed up in {time.time() - t0:.1f}s")
    params = VisualGenParams(height=768, width=1344, num_frames=124,
                             num_inference_steps=a.steps, seed=42)
    t0 = time.time()
    out = vg.generate(inputs="A woman sits on a park bench in autumn light, portrait.",
                      params=params)
    log(f"generate ({a.steps} steps) in {time.time() - t0:.1f}s")
    del out
    vg.shutdown()
    print(json.dumps({"done": True, "overlay": overlay, "steps": a.steps}))


if __name__ == "__main__":
    main()
