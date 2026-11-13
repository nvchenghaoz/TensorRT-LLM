"""End-to-end probe for MiniMax-H3 at the campaign layout (TP2, text-encoder TP2, parallel VAE 2,
VANILLA attention, torch.compile + autotune, no CUDA graphs, warmup at 768x1344x124).

Generates one short request (--steps denoising steps, seed 42) twice in the same process and prints
SHA-256 hashes of the video and audio tensors, so a baseline arm and a candidate arm can be compared
bitwise, plus wall time per request. Not the frozen benchmark: a correctness/trace probe only.
"""
import argparse
import hashlib
import json
import time


def h(t):
    if t is None:
        return None
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()

    from tensorrt_llm import VisualGen, VisualGenArgs

    vg_args = VisualGenArgs(
        parallel_config={"tp_size": 2, "text_encoder_tp_size": 2, "parallel_vae_size": 2},
        attention_config={"backend": "VANILLA"},
        cuda_graph_config={"enable": False},
        torch_compile_config={"enable": True, "enable_autotune": True},
        compilation_config={
            "resolutions": [[768, 1344]],
            "num_frames": [124],
            "skip_warmup": bool(args.no_warmup),
        },
    )
    t0 = time.time()
    vg = VisualGen(model=args.checkpoint, args=vg_args)
    print(f"[probe] loaded in {time.time() - t0:.1f}s", flush=True)
    prompt = (
        "A woman with long brown hair and light skin smiles at the camera while standing in a sunlit "
        "park, her hair gently blowing in the breeze as she tilts her head slightly to the side."
    )
    results = []
    for r in range(args.repeats):
        params = vg.default_params
        params.height = 768
        params.width = 1344
        params.num_frames = 124
        params.num_inference_steps = args.steps
        params.seed = 42
        t = time.time()
        out = vg.generate(inputs=prompt, params=params)
        dt = time.time() - t
        rec = {
            "repeat": r,
            "seconds": dt,
            "video_sha256": h(out.video),
            "audio_sha256": h(out.audio),
            "video_shape": list(out.video.shape) if out.video is not None else None,
            "error": out.error,
        }
        print("[probe]", json.dumps(rec), flush=True)
        results.append(rec)
    same = len({(x["video_sha256"], x["audio_sha256"]) for x in results}) == 1
    print(json.dumps({"passed": same and all(x["error"] is None for x in results),
                      "repeats_identical": same, "results": results}), flush=True)


if __name__ == "__main__":
    main()
