"""Shape-keyed-state + repeated-execution check for FP8 attention on FLUX.3 klein.

For each mode (fresh process, TORCH_LOGS=recompiles):
  load VisualGen (its own warmup runs at load) -> print marker ->
  generate prompts whose text lengths T cover several pad_multiple buckets
  (short, ~160, ~240, ~320 tokens) -> repeat the short prompt (A, B..., A) ->
  run 2 consecutive identical requests.
Counts Dynamo 'Recompiling' log lines after the marker (i.e. inside "timed" requests) and
checks A == A' and the two consecutive requests bitwise. Compares FP8 images to VANILLA
images (same prompt/seed) by PSNR / LPIPS(alex) for information.

usage: python recompile_check.py <checkpoint> <outdir> [modes=vanilla,fp8]
"""
import json
import math
import os
import subprocess
import sys
import time

BASE = "A red fox standing in fresh snow at sunrise, photorealistic"
FILLER = (" The scene is rendered with careful attention to texture, light and atmosphere,"
          " with soft shadows, crisp details on fur and snow crystals, a pale pink sky,"
          " distant pine trees fading into mist, and a calm, quiet winter mood.")


def prompts():
    return [BASE, BASE + FILLER * 3, BASE + FILLER * 6, BASE + FILLER * 9, BASE]


MARK = "=== WARMUP DONE: timed region starts"


def mode_args(mode):
    from tensorrt_llm import VisualGenArgs
    from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

    cfg = VisualGenArgs()
    if mode == "vanilla":
        att = AttentionConfig()
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


def gen(model, prompt, seed):
    params = model.default_params
    params.height, params.width, params.seed = 1024, 1024, seed
    params.num_inference_steps = 16
    out = model.generate(inputs=prompt, params=params)
    if out.error:
        raise RuntimeError(out.error)
    return out.image.cpu()


def run_mode(ckpt, outdir, mode):
    import torch
    from tensorrt_llm import VisualGen

    t0 = time.time()
    model = VisualGen(model=ckpt, args=mode_args(mode))
    print(f"[{mode}] loaded in {time.time() - t0:.0f}s", flush=True)
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    print(MARK, flush=True)
    imgs, times = [], []
    for i, p in enumerate(prompts()):
        t1 = time.time()
        imgs.append(gen(model, p, 1000))
        times.append(time.time() - t1)
        print(f"[{mode}] prompt {i} chars={len(p)} {times[-1]:.2f}s", flush=True)
    aa_ok = bool(torch.equal(imgs[0], imgs[-1]))
    r1 = gen(model, prompts()[2], 2000)
    r2 = gen(model, prompts()[2], 2000)
    consec_ok = bool(torch.equal(r1, r2))
    print(f"[{mode}] A-B-A bitwise {aa_ok}; consecutive bitwise {consec_ok}", flush=True)
    torch.save({"imgs": imgs, "times": times, "aa_ok": aa_ok, "consec_ok": consec_ok},
               os.path.join(outdir, f"rc_{mode}.pt"))
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
    ckpt, outdir = sys.argv[1], sys.argv[2]
    modes = (sys.argv[3] if len(sys.argv) > 3 else "vanilla,fp8").split(",")
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, TORCH_LOGS="recompiles")
    summary = {}
    for m in modes:
        print(f"=== mode {m}", flush=True)
        log = os.path.join(outdir, f"rc_{m}.log")
        with open(log, "w") as f:
            proc = subprocess.Popen([sys.executable, __file__, "--mode", m, ckpt, outdir],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                    text=True)
            for line in proc.stdout:
                f.write(line)
                if line.startswith("[") or line.startswith("===") or "ecompil" in line:
                    print(line.rstrip()[:300], flush=True)
            proc.wait()
        if proc.returncode != 0:
            raise SystemExit(f"mode {m} failed rc={proc.returncode}")
        text = open(log).read()
        pre, _, post = text.partition(MARK)
        summary[m] = {"recompiles_warmup": pre.count("Recompiling function"),
                      "recompiles_timed": post.count("Recompiling function")}
    import torch

    data = {m: torch.load(os.path.join(outdir, f"rc_{m}.pt")) for m in modes}
    lp = None
    try:
        import lpips

        lp = lpips.LPIPS(net="alex", verbose=False).cuda()
    except Exception as ex:  # noqa: BLE001
        print("lpips unavailable:", repr(ex)[:200], flush=True)
    for m in modes:
        summary[m].update(aa_bitwise=data[m]["aa_ok"], consecutive_bitwise=data[m]["consec_ok"],
                          gen_s=[round(t, 3) for t in data[m]["times"]])
        if m != "vanilla" and "vanilla" in data:
            per = []
            for a_img, b_img in zip(data[m]["imgs"], data["vanilla"]["imgs"]):
                a, b = to_float(a_img), to_float(b_img)
                mse = torch.mean((a - b) ** 2).item()
                e = {"psnr": 99.0 if mse == 0 else round(10 * math.log10(1.0 / mse), 3)}
                if lp is not None:
                    with torch.no_grad():
                        e["lpips"] = round(lp(a[None].cuda() * 2 - 1, b[None].cuda() * 2 - 1).item(), 4)
                per.append(e)
            summary[m]["vs_vanilla"] = per
    passed = all(s["aa_bitwise"] and s["consecutive_bitwise"] for s in summary.values()) and \
        summary.get("fp8", {}).get("recompiles_timed", 0) <= summary.get("vanilla", {}).get(
            "recompiles_timed", 0)
    print(json.dumps({"passed": passed,
                      "compared_against": "VANILLA (main default) in a separate fresh process, "
                                          "same prompts/seeds; bitwise within process",
                      "summary": summary}))


if __name__ == "__main__":
    main()
