"""Idea 55 end-to-end check: FLUX.3 klein nvfp4, served engine settings (engine.json overlay:
FLASHINFER FP8 attention, BF16 VAE decode; per-block compile; warmup 1024x1024 t2i).

Two fresh processes on one GPU, TORCH_LOGS=recompiles:
  on : branch default (text stream step-batched)
  off: TRTLLM_FLUX3_TEXT_STEP_BATCH=0 (main's per-step text stream, same tree)
Each: load (warmup) -> marker -> requests A (T=80), B (T=160), A, C, D (T=80), then C twice
more (consecutive repeat). Reports Dynamo recompiles per request segment (warmup and timed),
A vs A' and C repeats bitwise, per-request wall time, and on-vs-off image PSNR / LPIPS(alex)
against gate v2 (LPIPS <= 0.265977, PSNR >= 17.658 dB).
usage: python b33_e2e_check.py <checkpoint> [outdir] [modes=off,on]
"""

import json
import math
import os
import subprocess
import sys
import time

A = "A red fox standing in fresh snow at sunrise, photorealistic"
FILLER = (
    " The scene is rendered with careful attention to texture, light and atmosphere,"
    " with soft shadows, crisp details on fur and snow crystals, a pale pink sky,"
    " distant pine trees fading into mist, and a calm, quiet winter mood."
)
B = A + FILLER * 3
C = "A cozy reading nook with a green velvet armchair and a sleeping tabby cat"
D = "Close-up macro photo of a dewdrop on a fern leaf"
REQS = [("A", A, 11), ("B", B, 22), ("A", A, 11), ("C", C, 33), ("D", D, 44), ("C", C, 33), ("C", C, 33)]
MARK = "=== B33 REQ"


def run_mode(ckpt, outdir, mode):
    import torch

    from tensorrt_llm import VisualGen, VisualGenArgs
    from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

    cfg = VisualGenArgs()
    att = AttentionConfig(
        backend="FLASHINFER",
        quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
    )
    cfg = cfg.model_copy(update={"attention_config": att})
    cfg.pipeline_config["variant"] = "nvfp4"
    cfg.pipeline_config["vae_decode_dtype"] = "bfloat16"
    t0 = time.time()
    model = VisualGen(model=ckpt, args=cfg)
    print(f"[{mode}] loaded in {time.time() - t0:.0f}s", flush=True)
    imgs, times = [], []
    for i, (name, prompt, seed) in enumerate(REQS):
        print(f"{MARK} {i} {name}", flush=True)
        params = model.default_params
        params.height, params.width, params.seed = 1024, 1024, seed
        params.num_inference_steps = 16
        t1 = time.time()
        out = model.generate(inputs=prompt, params=params)
        if out.error:
            raise RuntimeError(out.error)
        times.append(time.time() - t1)
        imgs.append(out.image.cpu())
        print(f"[{mode}] req {i} {name} {times[-1]:.3f}s", flush=True)
    print(f"{MARK} end", flush=True)
    torch.save({"imgs": imgs, "times": times}, os.path.join(outdir, f"b33_{mode}.pt"))
    model.shutdown()


def to_float(img):
    x = img.float()
    if x.dim() == 4:
        x = x[0]
    if x.shape[-1] in (3, 4) and x.shape[0] not in (3, 4):
        x = x[..., :3].permute(2, 0, 1)
    if x.max() > 1.5:
        x = x / 255.0
    return x.clamp(0, 1)


def main():
    if sys.argv[1] == "--mode":
        _, _, mode, ckpt, outdir = sys.argv
        run_mode(ckpt, outdir, mode)
        return
    ckpt = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b33")
    modes = (sys.argv[3] if len(sys.argv) > 3 else "off,on").split(",")
    os.makedirs(outdir, exist_ok=True)
    summary = {}
    for m in modes:
        env = dict(os.environ, TORCH_LOGS="recompiles")
        env["TRTLLM_FLUX3_TEXT_STEP_BATCH"] = "0" if m == "off" else "1"
        print(f"=== mode {m}", flush=True)
        log = os.path.join(outdir, f"b33_{m}.log")
        with open(log, "w") as f:
            proc = subprocess.Popen(
                [sys.executable, __file__, "--mode", m, ckpt, outdir],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
            for line in proc.stdout:
                f.write(line)
                if line.startswith(("[", "===")) or "ecompil" in line:
                    print(line.rstrip()[:300], flush=True)
            proc.wait()
        if proc.returncode != 0:
            print(json.dumps({"passed": False, "error": f"mode {m} rc={proc.returncode}"}))
            raise SystemExit(1)
        text = open(log).read()
        parts = text.split(MARK)
        summary[m] = {
            "recompiles_warmup": parts[0].count("Recompiling function"),
            "recompiles_per_request": [
                p.count("Recompiling function") for p in parts[1 : 1 + len(REQS)]
            ],
            "requests": [r[0] for r in REQS],
        }
    import torch

    data = {m: torch.load(os.path.join(outdir, f"b33_{m}.pt")) for m in modes}
    lp = None
    try:
        import lpips

        lp = lpips.LPIPS(net="alex", verbose=False).cuda()
    except Exception as ex:  # noqa: BLE001
        print("lpips unavailable:", repr(ex)[:200], flush=True)
    for m in modes:
        imgs = data[m]["imgs"]
        summary[m].update(
            A_B_A_bitwise=bool(torch.equal(imgs[0], imgs[2])),
            C_repeats_bitwise=bool(torch.equal(imgs[3], imgs[5]) and torch.equal(imgs[3], imgs[6])),
            wall_s=[round(t, 3) for t in data[m]["times"]],
        )
    if "on" in data and "off" in data:
        per = []
        for i, (a_img, b_img) in enumerate(zip(data["on"]["imgs"], data["off"]["imgs"])):
            a, b = to_float(a_img), to_float(b_img)
            mse = torch.mean((a - b) ** 2).item()
            e = {"req": REQS[i][0], "psnr": 99.0 if mse == 0 else round(10 * math.log10(1.0 / mse), 3)}
            if lp is not None:
                with torch.no_grad():
                    e["lpips"] = round(
                        lp(a[None].cuda() * 2 - 1, b[None].cuda() * 2 - 1).item(), 4
                    )
            e["gate_v2"] = e["psnr"] >= 17.658 and e.get("lpips", 0.0) <= 0.265977
            per.append(e)
        summary["on_vs_off"] = per
    on = summary.get("on", {})
    t80_idx = [i for i, r in enumerate(REQS) if r[0] != "B"]
    passed = (
        all(summary[m]["A_B_A_bitwise"] and summary[m]["C_repeats_bitwise"] for m in modes)
        and all(e["gate_v2"] for e in summary.get("on_vs_off", []))
        and all(on.get("recompiles_per_request", [0] * len(REQS))[i] == 0 for i in t80_idx)
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "compared_against": "same tree with TRTLLM_FLUX3_TEXT_STEP_BATCH=0 (main's per-step "
                "text stream) in a separate fresh process, same prompts/seeds; gate v2 thresholds; "
                "bitwise repeats within each process",
                "summary": summary,
            }
        )
    )


if __name__ == "__main__":
    main()
