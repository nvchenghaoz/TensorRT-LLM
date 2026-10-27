"""End-to-end accuracy probe for FP8 attention on FLUX.3 klein (nvfp4, 1024x1024, 16 steps).

Generates the same prompts/seeds under several attention configs (each in a fresh
subprocess, fresh VisualGen) and compares every image with the VANILLA (main default)
image by PSNR and LPIPS(alex) when lpips is importable. A second VANILLA-like run of the
same config (repeat of prompt 0) checks repeated-request determinism.

usage: python acc_e2e.py <checkpoint> <outdir> [modes=vanilla,fi_bf16,fp8] [n_prompts]
"""
import json
import math
import os
import subprocess
import sys
import time

PROMPTS = [
    "A red fox standing in fresh snow at sunrise, photorealistic",
    "A bustling night market in Taipei with neon signs and steam rising from food stalls",
    "A watercolor painting of a lighthouse on a rocky cliff during a storm",
    "Close-up portrait of an elderly fisherman with a weathered face, dramatic lighting",
    "A futuristic city skyline with flying cars at dusk, highly detailed digital art",
    "A bowl of ramen with a soft-boiled egg, chopsticks, top-down food photography",
    "A cozy reading nook with a cat sleeping on a stack of books, warm afternoon light",
    "A vintage poster that says 'VISIT MARS' in bold retro typography",
    "An astronaut riding a horse across a desert under a starry sky",
    "A macro photo of a dew-covered spider web in the morning",
    "A medieval castle surrounded by autumn forest, aerial view",
    "A glass of orange juice on a wooden table next to sliced oranges, studio lighting",
]


def mode_args(mode):
    from tensorrt_llm import VisualGenArgs
    from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

    cfg = VisualGenArgs()
    if mode == "vanilla":
        att = AttentionConfig()
    elif mode == "fi_bf16":
        att = AttentionConfig(backend="FLASHINFER")
    elif mode == "fp8":
        att = AttentionConfig(
            backend="FLASHINFER",
            quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
        )
    else:
        raise ValueError(mode)
    cfg = cfg.model_copy(update={"attention_config": att})
    cfg.pipeline_config["variant"] = "nvfp4"
    return cfg


def run_mode(ckpt, outdir, mode, n):
    import torch
    from tensorrt_llm import VisualGen

    t0 = time.time()
    model = VisualGen(model=ckpt, args=mode_args(mode))
    print(f"[{mode}] loaded in {time.time() - t0:.0f}s", flush=True)
    imgs = []
    times = []
    for i in range(n):
        params = model.default_params
        params.height, params.width, params.seed = 1024, 1024, 1000 + i
        params.num_inference_steps = 16
        t1 = time.time()
        out = model.generate(inputs=PROMPTS[i], params=params)
        times.append(time.time() - t1)
        if out.error:
            raise RuntimeError(out.error)
        imgs.append(out.image.cpu())
        print(f"[{mode}] prompt {i} {times[-1]:.2f}s shape={tuple(out.image.shape)} "
              f"dtype={out.image.dtype}", flush=True)
    # Repeat prompt 0 on the same server: must be bitwise identical.
    params = model.default_params
    params.height, params.width, params.seed = 1024, 1024, 1000
    params.num_inference_steps = 16
    rep = model.generate(inputs=PROMPTS[0], params=params).image.cpu()
    repeat_ok = bool(torch.equal(rep, imgs[0]))
    print(f"[{mode}] repeat bitwise: {repeat_ok}", flush=True)
    torch.save({"imgs": imgs, "times": times, "repeat_ok": repeat_ok},
               os.path.join(outdir, f"{mode}.pt"))
    model.shutdown()


def to_float(img):
    import torch

    x = img.float()
    if x.dim() == 4:
        x = x[0]
    if x.shape[-1] in (3, 4) and x.shape[0] not in (3, 4):
        x = x[..., :3].permute(2, 0, 1)
    if x.max() > 1.5:
        x = x / 255.0
    return x.clamp(0, 1)


def compare(outdir, modes):
    import torch

    data = {m: torch.load(os.path.join(outdir, f"{m}.pt")) for m in modes}
    lp = None
    try:
        import lpips

        lp = lpips.LPIPS(net="alex", verbose=False).cuda()
    except Exception as ex:  # noqa: BLE001
        print("lpips unavailable:", repr(ex)[:300], flush=True)
    ref = data["vanilla"]["imgs"]
    res = {}
    for m in modes:
        per = []
        for i, img in enumerate(data[m]["imgs"]):
            a, b = to_float(img), to_float(ref[i])
            mse = torch.mean((a - b) ** 2).item()
            psnr = 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)
            entry = {"psnr": round(psnr, 3)}
            if lp is not None:
                with torch.no_grad():
                    entry["lpips"] = round(
                        lp(a[None].cuda() * 2 - 1, b[None].cuda() * 2 - 1).item(), 4)
            per.append(entry)
        summ = {
            "mean_psnr": round(sum(e["psnr"] for e in per) / len(per), 3),
            "min_psnr": min(e["psnr"] for e in per),
            "repeat_bitwise": data[m]["repeat_ok"],
            "mean_gen_s": round(sum(data[m]["times"][1:]) / max(1, len(data[m]["times"]) - 1), 3),
        }
        if lp is not None:
            summ["mean_lpips"] = round(sum(e["lpips"] for e in per) / len(per), 4)
            summ["max_lpips"] = max(e["lpips"] for e in per)
        res[m] = {"summary": summ, "per_prompt": per}
        print(m, json.dumps(summ), flush=True)
    return res


def main():
    if sys.argv[1] == "--mode":
        _, _, mode, ckpt, outdir, n = sys.argv
        run_mode(ckpt, outdir, mode, int(n))
        return
    ckpt, outdir = sys.argv[1], sys.argv[2]
    modes = (sys.argv[3] if len(sys.argv) > 3 else "vanilla,fi_bf16,fp8").split(",")
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(outdir, exist_ok=True)
    for m in modes:
        print(f"=== mode {m}", flush=True)
        subprocess.run([sys.executable, __file__, "--mode", m, ckpt, outdir, str(n)], check=True)
    res = compare(outdir, modes)
    passed = all(r["summary"]["repeat_bitwise"] for r in res.values())
    print(json.dumps({"passed": passed, "compared_against": "VANILLA (main default) images, "
                      "same prompts/seeds", "results": {m: r["summary"] for m, r in res.items()},
                      "per_prompt": {m: r["per_prompt"] for m, r in res.items()}}))


if __name__ == "__main__":
    main()
