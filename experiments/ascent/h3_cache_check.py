"""MiniMax-H3 Cache-DiT functional check on TP2 (real checkpoint, VisualGen API).

One VisualGen instance (2 GPUs, tp_size 2), an untimed 2-step warmup request, then
the requested seed sequence (default A, B, A) back to back with nothing reset in
between. Saves each output (uint8 video + audio) to --out for cross-arm comparison
and prints per-request wall time.

Usage:
  python h3_cache_check.py --tag cache --cache '{"cache_backend": "cache_dit", ...}' \
      --out /tmp/h3 --seeds 42 7 42
  python h3_cache_check.py --tag base --cache none --out /tmp/h3 --seeds 42
"""

import argparse
import json
import os
import time

import torch

PROMPT = (
    "A woman with long brown hair and light skin smiles at the camera while standing in a "
    "sunlit park, her hair gently blowing in the breeze as she tilts her head slightly to "
    "the side."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--cache", default="none")
    p.add_argument("--out", required=True)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 42])
    p.add_argument("--prompt", default=PROMPT)
    args = p.parse_args()

    from tensorrt_llm import VisualGen, VisualGenArgs

    cfg = dict(
        parallel_config=dict(tp_size=2, text_encoder_tp_size=2, parallel_vae_size=2),
        attention_config=dict(backend="VANILLA"),
        compilation_config=dict(
            resolutions=[[args.height, args.width]], num_frames=[args.frames]
        ),
    )
    if args.cache != "none":
        cfg["cache_config"] = json.loads(args.cache)
    print(f"[{args.tag}] VisualGenArgs: {json.dumps(cfg)}", flush=True)
    # The client receives outputs through CUDA IPC; create its context before the
    # workers fill the GPU (trtllm-serve does this at startup too).
    torch.zeros(1, device="cuda")
    t0 = time.time()
    vg = VisualGen(model=args.model, args=VisualGenArgs(**cfg))
    print(f"[{args.tag}] loaded in {time.time() - t0:.1f}s", flush=True)

    def run(seed: int, steps: int):
        params = vg.default_params
        params.height = args.height
        params.width = args.width
        params.num_frames = args.frames
        params.num_inference_steps = steps
        params.seed = seed
        t = time.time()
        out = vg.generate(inputs=args.prompt, params=params)
        return out, time.time() - t

    _, dt = run(args.seeds[0], 2)
    print(f"[{args.tag}] warmup request (2 steps) {dt:.1f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    results = []
    for i, seed in enumerate(args.seeds):
        out, dt = run(seed, args.steps)
        video = out.video.cpu() if out.video is not None else None
        audio = out.audio.cpu() if out.audio is not None else None
        path = os.path.join(args.out, f"{args.tag}_{i}_seed{seed}.pt")
        torch.save({"video": video, "audio": audio, "seconds": dt}, path)
        print(
            f"[{args.tag}] request {i} seed={seed} {dt:.2f}s video={tuple(video.shape)} "
            f"{video.dtype} audio={None if audio is None else tuple(audio.shape)} -> {path}",
            flush=True,
        )
        results.append(dict(index=i, seed=seed, seconds=round(dt, 3), path=path))
    print(json.dumps({"tag": args.tag, "results": results}), flush=True)
    if hasattr(vg, "shutdown"):
        vg.shutdown()


if __name__ == "__main__":
    main()
