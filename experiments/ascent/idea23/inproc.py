"""Idea 23: FLUX.3 i2i reference prep (pipeline_flux3.py), in-process.

Modes (python inproc.py <mode> <checkpoint>):
  unit  : no model. _load_image_pixels of THIS tree vs main's code (copied below as
          ref_load_image_pixels) on the two frozen 2048^2 references and on fallback images
          (EXIF-rotated JPEG and PNG, RGBA, palette, gray, odd size, >4096^2 resize):
          values and strides bitwise. Step-0 timings of main's parts, the tree's loader,
          sample_image_latents, and torchvision/cv2 PNG decoders (speed + bitwise vs PIL).
  bench : served pipeline (engine.json, CUDA graphs, compiled VAE encoder, 2048 warmup).
          Timed span = Flux3Pipeline.infer(i2i request with the reference PNG bytes) from
          entry to the first transformer forward call, after torch.cuda.synchronize() there
          (all pre-denoise GPU work done), then the request is stopped. N_WARM + N_ITERS
          alternating A/B. Also a t2i span (no-regression). Prints the step_ms JSON.
  check : served pipeline. main's _load_image_pixels / pack_i2i_image_condition /
          _prepare_i2i (copied below) are the in-process reference. Full requests
          ref/cand interleaved A, B, A, t2i: output image, reference pixels (+strides), VAE
          latents bitwise; A3 == A1; then an untimed truncated sequence shaped like bench
          (N_WARM + N_ITERS alternating A/B) comparing every transformer input at the first
          denoise call with the reference; dynamo counters and CUDA-graph counts/replays.
"""

import glob
import hashlib
import json
import math
import os
import statistics
import sys
import time
import types
from io import BytesIO

import numpy as np
import torch

MODE = sys.argv[1]
CKPT = sys.argv[2] if len(sys.argv) > 2 else None
STORE = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
         "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
REFS = {  # experiments/ascent-benchmark/inputs.json
    "A": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e",  # check00
    "B": "76bf07ca4e247699c100eaeff046d3335fea0019844ab3a2b60206e2d798a714",  # check10
}
PROMPTS = {"A": ("Make the sky a vivid sunset orange, keep everything else unchanged", 1234),
           "B": ("Turn the scene into a snowy winter day", 4321)}
T2I = ("A red fox standing in fresh snow at sunrise, photograph", 77)
N_WARM = int(os.environ.get("N_WARM", "5"))
N_ITERS = int(os.environ.get("N_ITERS", "40"))
HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(f"[idea23 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_refs():
    want = {v: k for k, v in REFS.items()}
    found = {}
    files = [f for f in sorted(glob.glob(os.path.join(STORE, "**", "*"), recursive=True))
             if os.path.isfile(f)]
    named = [f for f in files if any(s in os.path.basename(f) for s in want)]
    for f in named + [f for f in files if f not in named]:
        if len(found) == len(want):
            break
        data = open(f, "rb").read()
        h = hashlib.sha256(data).hexdigest()
        if h in want:
            found[want[h]] = data
    if len(found) != len(want):
        raise SystemExit(f"references not found in {STORE}: have {list(found)}")
    return found


# ----------------------------------------------------------------------------- main's code
def ref_load_image_pixels(source, *, device):
    from PIL import Image, ImageOps

    with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        scale = min(1.0, 4096 / math.sqrt(image.width * image.height))
        if scale < 1:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
            )
        width, height = image.width // 16 * 16, image.height // 16 * 16
        if min(width, height) == 0:
            raise ValueError("FLUX.3 reference images must be at least 16 pixels on each axis")
        left, top = (image.width - width) // 2, (image.height - height) // 2
        pixels = np.array(image.crop((left, top, left + width, top + height)))
    return (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 127.5
        - 1
    )


def ref_pack_i2i_image_condition(latents, *, device, l_start=-1, image_rope_scale=0.5, copies=2):
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import I2I_COND_T, prc_img

    x_cond_parts = []
    x_cond_id_parts = []
    for i_ref, latent in enumerate(latents):
        x_ref, x_ref_ids = prc_img(
            latent,
            t_coord=torch.tensor([I2I_COND_T], device=latent.device, dtype=torch.long),
            l_coord=torch.tensor([l_start - i_ref], device=latent.device, dtype=torch.long),
        )
        x_cond_parts.append(x_ref)
        x_cond_id_parts.append(x_ref_ids)
    x_cond = torch.cat(x_cond_parts, dim=0).unsqueeze(0)
    x_cond_ids = torch.cat(x_cond_id_parts, dim=0).unsqueeze(0)
    x_cond_ids = x_cond_ids.to(dtype=torch.float32)
    x_cond_ids[..., 1:3] = x_cond_ids[..., 1:3] * image_rope_scale
    x_cond_timesteps = torch.zeros(x_cond.shape[:2], device=x_cond.device, dtype=x_cond.dtype)
    x_cond = x_cond.repeat(copies, 1, 1).to(device=device, dtype=torch.bfloat16)
    x_cond_ids = x_cond_ids.repeat(copies, 1, 1).to(device=device)
    x_cond_timesteps = x_cond_timesteps.repeat(copies, 1).to(device=device, dtype=torch.bfloat16)
    return x_cond, x_cond_ids, x_cond_timesteps


def ref_prepare_i2i(self, ctx, image_cond, height=1024, width=1024, num_inference_steps=None,
                    cfg=None, seed=0, pag_single_layers=None, pag_strength=1.0, timer=None):
    x_image, x_image_ids, x_image_orig_ids = self._sample_image_latents(height, width, seed)
    device = x_image.device
    image_vae_dtype = next(self.vae.parameters()).dtype
    images = [image_cond] if isinstance(image_cond, torch.Tensor) else image_cond
    image_latents = [self.vae.encode(image.to(image_vae_dtype))[0] for image in images]
    x_image_cond, x_image_cond_ids, _ = ref_pack_i2i_image_condition(
        image_latents,
        device=device,
        image_rope_scale=self.pipeline_config.extra_attrs["flux3_checkpoint"].config["dit"][
            "image_rope_scale"
        ],
        copies=1 if self.transformer.flow_map else 2,
    )
    latent = self._run_denoise(
        x=x_image, x_ids=x_image_ids, ctx=ctx, num_inference_steps=num_inference_steps, cfg=cfg,
        pag_single_layers=pag_single_layers, pag_strength=pag_strength,
        cond_kwargs=dict(x_image_cond=x_image_cond, x_image_cond_ids=x_image_cond_ids),
        timer=timer,
    )
    return latent, x_image_orig_ids


# ----------------------------------------------------------------------------- helpers
def spread(v):
    q = statistics.quantiles(sorted(v), n=10)
    return (q[-1] - q[0]) / statistics.median(v)


def same(a, b):
    return bool(a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride()
                and torch.equal(a, b))


def fallback_images(ref_bytes):
    """Images that exercise the PIL fallbacks: name -> bytes."""
    from PIL import Image

    base = Image.open(BytesIO(ref_bytes)).convert("RGB")
    out = {}
    small = base.resize((1000, 776))
    for orient in (3, 6, 8):
        exif = Image.Exif()
        exif[0x0112] = orient
        b = BytesIO()
        small.save(b, format="JPEG", quality=90, exif=exif.tobytes())
        out[f"jpeg_orient{orient}"] = b.getvalue()
        b = BytesIO()
        small.save(b, format="PNG", exif=exif.tobytes())
        out[f"png_orient{orient}"] = b.getvalue()
    exif = Image.Exif()
    exif[0x0112] = 1
    b = BytesIO()
    small.save(b, format="JPEG", quality=90, exif=exif.tobytes())
    out["jpeg_orient1"] = b.getvalue()
    b = BytesIO()
    small.save(b, format="JPEG", quality=90)
    out["jpeg_noexif"] = b.getvalue()
    rgba = small.copy()
    rgba.putalpha(Image.linear_gradient("L").resize(small.size))
    b = BytesIO()
    rgba.save(b, format="PNG")
    out["png_rgba"] = b.getvalue()
    b = BytesIO()
    small.convert("P", palette=Image.Palette.ADAPTIVE).save(b, format="PNG")
    out["png_palette"] = b.getvalue()
    b = BytesIO()
    small.convert("L").save(b, format="PNG")
    out["png_gray"] = b.getvalue()
    b = BytesIO()
    base.crop((0, 0, 2040, 2033)).save(b, format="PNG")
    out["png_odd_2040x2033"] = b.getvalue()
    b = BytesIO()
    base.resize((4200, 4200)).save(b, format="PNG", compress_level=1)
    out["png_resize_4200"] = b.getvalue()
    return out


# ----------------------------------------------------------------------------- unit
def unit():
    from PIL import Image, ImageOps

    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as flux3

    dev = torch.device("cuda")
    refs = find_refs()
    res = {"loader": {}, "timing_ms": {}}
    ok = True
    cases = dict(refs)
    cases.update(fallback_images(refs["A"]))
    for name, data in cases.items():
        for device in (dev, torch.device("cpu")):
            r = ref_load_image_pixels(data, device=device)
            c = flux3._load_image_pixels(data, device=device)
            c2 = flux3._load_image_pixels(data, device=device)  # repeated call
            eq = same(r, c) and same(c, c2)
            ok &= eq
            res["loader"][f"{name}@{device.type}"] = {
                "bitwise": eq, "shape": list(c.shape), "stride_ref": list(r.stride()),
                "stride_cand": list(c.stride())}
        log(f"{name}: {res['loader'][name + '@cuda']}")
    # A, B, A repeated on cuda with no sync between (stale pinned block guard).
    outs = [flux3._load_image_pixels(refs[k], device=dev) for k in "ABA"]
    torch.cuda.synchronize()
    rA, rB = ref_load_image_pixels(refs["A"], device=dev), ref_load_image_pixels(refs["B"], device=dev)
    aba = same(outs[0], rA) and same(outs[1], rB) and same(outs[2], rA)
    ok &= aba
    res["ABA_no_sync_bitwise"] = aba

    def t(fn, n=8):
        v = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            v.append((time.perf_counter() - t0) * 1e3)
        return round(statistics.median(v), 2)

    data = refs["A"]
    tm = res["timing_ms"]
    tm["main_load_image_pixels"] = t(lambda: ref_load_image_pixels(data, device=dev))
    tm["cand_load_image_pixels"] = t(lambda: flux3._load_image_pixels(data, device=dev))
    tm["main_load_image_pixels"] = t(lambda: ref_load_image_pixels(data, device=dev))
    tm["cand_load_image_pixels"] = t(lambda: flux3._load_image_pixels(data, device=dev))

    def opened():
        im = Image.open(BytesIO(data))
        im.load()
        return im

    im = opened()
    tm["pil_open_load"] = t(opened)
    tm["exif_transpose"] = t(lambda: ImageOps.exif_transpose(im))
    tm["convert_rgb"] = t(lambda: im.convert("RGB"))
    tm["crop_full"] = t(lambda: im.crop((0, 0, im.width, im.height)))
    tm["np_array"] = t(lambda: np.array(im))
    tm["np_asarray"] = t(lambda: np.asarray(im))
    px = np.array(im)
    tm["to_cuda_fp32_main"] = t(
        lambda: torch.from_numpy(px).permute(2, 0, 1).unsqueeze(0).to(device=dev, dtype=torch.float32))
    tm["pinned_u8_h2d_gpu_cast"] = t(
        lambda: torch.from_numpy(px).pin_memory().to(dev, non_blocking=True)
        .permute(2, 0, 1).unsqueeze(0).to(torch.float32))
    tm["sample_image_latents_2048"] = t(lambda: flux3.sample_image_latents(
        height=2048, width=2048, seed=3, device=dev))
    rng = torch.Generator().manual_seed(3)
    tm["cpu_randn_16384x128"] = t(lambda: torch.randn((16384, 128), generator=rng))
    # Alternative PNG decoders (speed + bitwise vs PIL).
    alt = {}
    try:
        import torchvision.io as tvio

        buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        d = tvio.decode_png(buf)
        alt["torchvision_decode_png"] = {
            "ms": t(lambda: tvio.decode_png(buf)),
            "bitwise_vs_pil": bool(np.array_equal(d.permute(1, 2, 0).numpy(), px))}
    except Exception as exc:  # noqa: BLE001
        alt["torchvision_decode_png"] = repr(exc)
    try:
        import cv2

        arr = np.frombuffer(data, np.uint8)
        d = cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        alt["cv2_imdecode"] = {"ms": t(lambda: cv2.imdecode(arr, cv2.IMREAD_COLOR)),
                               "bitwise_vs_pil": bool(np.array_equal(d, px))}
    except Exception as exc:  # noqa: BLE001
        alt["cv2_imdecode"] = repr(exc)
    res["alt_decoders"] = alt
    res["pil_info"] = {"mode": im.mode, "info_keys": sorted(map(str, im.info.keys()))}
    log(json.dumps(res["timing_ms"]))
    log(json.dumps(alt))
    print(json.dumps({"passed": bool(ok), "compared": "tree _load_image_pixels vs main's code "
                      "(copied) on 2 frozen refs + fallback images, cuda and cpu, values+strides "
                      "bitwise, repeated calls, A/B/A without sync", **res}))


# ----------------------------------------------------------------------------- pipeline
class _Stop(Exception):
    pass


def load_pipe():
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    tcc = dict(engine.pop("torch_compile_config", {}))
    tcc.update({"enable": True, "enable_fullgraph": False, "enable_autotune": False})
    args = VisualGenArgs(
        model=CKPT,
        pipeline_config=pipeline_config,
        torch_compile_config=tcc,
        compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1],
                            "modes": ["t2i", "i2i"]},
        **engine,
    )
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    log(f"pipeline loaded + warmed up in {time.time() - t0:.0f} s; engine={json.dumps(engine)}")
    return pipe


def make_req(prompt, seed, ref=None):
    params = types.SimpleNamespace(
        height=2048, width=2048, num_inference_steps=16, guidance_scale=None, seed=seed,
        negative_prompt=None, extra_params={},
        image_reference=[types.SimpleNamespace(content=ref)] if ref is not None else [])
    return types.SimpleNamespace(prompt=prompt, params=params)


class Probe:
    """Wraps transformer.forward: optionally stop at the first denoise call."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.orig = pipe.transformer.forward
        self.stop = False
        self.capture = False
        self.t_host = self.t_gpu = None
        self.kwargs = None
        pipe.transformer.forward = self

    def __call__(self, *args, **kwargs):
        if self.stop:
            self.t_host = time.perf_counter()
            torch.cuda.synchronize()
            self.t_gpu = time.perf_counter()
            if self.capture:
                self.kwargs = {k: (v.clone(), v.stride()) for k, v in kwargs.items()
                               if isinstance(v, torch.Tensor)}
            raise _Stop()
        return self.orig(*args, **kwargs)


def span(pipe, probe, req):
    probe.stop = True
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        pipe.infer(req)
        raise RuntimeError("infer finished without reaching the denoiser")
    except _Stop:
        pass
    finally:
        probe.stop = False
    return (probe.t_gpu - t0) * 1e3, (probe.t_host - t0) * 1e3


def bench():
    pipe = load_pipe()
    probe = Probe(pipe)
    refs = find_refs()
    reqs = {k: make_req(PROMPTS[k][0], PROMPTS[k][1], refs[k]) for k in "AB"}
    t2i = make_req(*T2I)
    for i in range(N_WARM):
        span(pipe, probe, reqs["AB"[i % 2]])
        span(pipe, probe, t2i)
    gi, hi, gt = [], [], []
    for i in range(N_ITERS):
        g, h = span(pipe, probe, reqs["AB"[i % 2]])
        gi.append(g)
        hi.append(h)
        gt.append(span(pipe, probe, t2i)[0])
        if i % 8 == 0:
            log(f"iter {i}: i2i pre-denoise {g:.1f} ms (host reached denoise at {h:.1f}), "
                f"t2i {gt[-1]:.1f} ms")
    res = {
        "batch": {
            "i2i-2048": {"step_ms": statistics.median(gi), "noise_floor": spread(gi)},
            "t2i-2048": {"step_ms": statistics.median(gt), "noise_floor": spread(gt)},
        },
        "output": {"class": "exact"},
        "i2i_host_reached_denoise_ms_median": statistics.median(hi),
        "i2i_A_median": statistics.median(gi[0::2]), "i2i_B_median": statistics.median(gi[1::2]),
        "measure": "Flux3Pipeline.infer entry (ref PNG bytes in) -> first transformer forward, "
                   "after cuda sync there; served engine.json pipeline, 2048^2",
        "n_warm": N_WARM, "n_iters": N_ITERS,
    }
    print(json.dumps(res))


def loadavg1():
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except Exception:  # noqa: BLE001
        return -1.0


def wait_load(limit=2.0, max_wait=600.0, poll=15.0):
    t0 = time.time()
    while True:
        la = loadavg1()
        log(f"load average (1 min) = {la:.2f}")
        if la < limit:
            return la, True
        if time.time() - t0 > max_wait:
            log(f"LOAD_TOO_HIGH: 1-min load {la:.2f} never fell below {limit} in {max_wait:.0f} s")
            return la, False
        time.sleep(poll)


def stats(v):
    v = list(v)
    q = statistics.quantiles(sorted(v), n=10)
    med = statistics.median(v)
    mean = statistics.fmean(v)
    return {"n": len(v), "mean": mean, "median": med, "p10": q[0], "p90": q[-1],
            "rel_std": statistics.pstdev(v) / mean, "p10_p90_over_median": (q[-1] - q[0]) / med}


def ibench():
    """Both arms interleaved in ONE process on ONE pipeline instance.

    Arm A = main's _load_image_pixels / pack_i2i_image_condition / _prepare_i2i (copied
    verbatim above). Arm B = this tree's functions. In the base tree arm B is main's code
    too (an A/A), in the branch tree it is the change. Per case: N_ARM_WARM warmups per arm,
    then N_ROUNDS rounds per arm alternating A,B,A,B..., each round ROUND_ITERS timed
    iterations of that arm. i2i alternates the two frozen references within each round.
    Then one untimed interleaved sequence of the same shape: every transformer input at the
    first denoise call must be bitwise equal between arms and to iteration 0.
    """
    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as flux3

    n_arm_warm = int(os.environ.get("N_ARM_WARM", "10"))
    n_rounds = int(os.environ.get("N_ROUNDS", "20"))
    round_iters = int(os.environ.get("ROUND_ITERS", "10"))
    la_wait, load_ok = wait_load()
    if not load_ok:
        print(json.dumps({"LOAD_TOO_HIGH": True, "load1": la_wait}))
        sys.exit(3)
    pipe = load_pipe()
    probe = Probe(pipe)
    refs = find_refs()
    cand_load = flux3._load_image_pixels
    cand_prep = type(pipe)._prepare_i2i

    def use(arm):
        if arm == "A":
            flux3._load_image_pixels = ref_load_image_pixels
            pipe._prepare_i2i = types.MethodType(ref_prepare_i2i, pipe)
        else:
            flux3._load_image_pixels = cand_load
            pipe.__dict__.pop("_prepare_i2i", None)
        want_load = ref_load_image_pixels if arm == "A" else cand_load
        want_prep = ref_prepare_i2i if arm == "A" else cand_prep
        assert flux3._load_image_pixels is want_load, "loader swap failed"
        assert pipe._prepare_i2i.__func__ is want_prep, "_prepare_i2i swap failed"

    # Requests are rebuilt from the immutable reference bytes (pristine inputs) every call.
    def req(kind):
        if kind == "t2i":
            return make_req(*T2I)
        return make_req(PROMPTS[kind][0], PROMPTS[kind][1], refs[kind])

    cases = {"i2i-2048": ["A", "B"], "t2i-2048": ["t2i"]}
    load_at = {}
    times = {c: {"A": [], "B": []} for c in cases}
    per_ref = {c: {"A": {}, "B": {}} for c in cases}
    rounds = {c: {"A": [], "B": []} for c in cases}
    for case, kinds in cases.items():
        for i in range(n_arm_warm):
            for arm in "AB":
                use(arm)
                span(pipe, probe, req(kinds[i % len(kinds)]))
        load_at[f"{case}_start"] = loadavg1()
        log(f"{case}: timed section starts, load1={load_at[case + '_start']:.2f}")
        for r in range(2 * n_rounds):
            arm = "AB"[r % 2]
            use(arm)
            rt = []
            for j in range(round_iters):
                kind = kinds[j % len(kinds)]
                g, _ = span(pipe, probe, req(kind))
                rt.append(g)
                per_ref[case][arm].setdefault(kind, []).append(g)
            times[case][arm].extend(rt)
            rounds[case][arm].append(statistics.median(rt))
            if r % 8 == 0:
                log(f"{case} round {r} arm {arm}: median {statistics.median(rt):.2f} ms")
        load_at[f"{case}_end"] = loadavg1()
        log(f"{case}: timed section ends, load1={load_at[case + '_end']:.2f}")
    # Untimed interleaved sequence of the same shape, bitwise.
    probe.capture = True
    first = {}
    seq_bad = []
    n_seq = 0
    for case, kinds in cases.items():
        for r in range(2 * (n_rounds + 1)):
            arm = "AB"[r % 2]
            use(arm)
            for j in range(round_iters if r >= 2 else n_arm_warm):
                kind = kinds[j % len(kinds)]
                span(pipe, probe, req(kind))
                kw = probe.kwargs
                n_seq += 1
                if kind not in first:
                    first[kind] = kw
                    continue
                f = first[kind]
                good = kw.keys() == f.keys() and all(
                    same(kw[n][0], f[n][0]) and kw[n][1] == f[n][1] for n in kw)
                if not good:
                    seq_bad.append((case, r, arm, j))
    probe.capture = False
    use("B")
    seq_ok = not seq_bad
    log(f"untimed interleaved sequence: {n_seq} calls, ok={seq_ok}, bad={seq_bad[:10]}")
    out = {"batch": {}, "output": {"class": "exact"}, "arms": {}, "per_reference": {},
           "round_ratio_B_over_A": {}}
    for case in cases:
        sa, sb = stats(times[case]["A"]), stats(times[case]["B"])
        out["arms"][case] = {"A_main_copy": sa, "B_tree": sb}
        out["per_reference"][case] = {
            arm: {k: stats(v) for k, v in per_ref[case][arm].items()} for arm in "AB"}
        ratios = [b / a for a, b in zip(rounds[case]["A"], rounds[case]["B"])]
        rq = statistics.quantiles(sorted(ratios), n=10)
        out["round_ratio_B_over_A"][case] = {"median": statistics.median(ratios),
                                             "p10": rq[0], "p90": rq[-1], "n": len(ratios)}
        # Pair JSON: this tree's own code (arm B). Noise = (p90-p10)/median of one input:
        # the max over the references, since the two references are different inputs.
        nf = max(stats(v)["p10_p90_over_median"] for v in per_ref[case]["B"].values())
        out["batch"][case] = {"step_ms": sb["median"], "noise_floor": nf,
                              "noise_floor_pooled_refs": sb["p10_p90_over_median"]}
    out.update({
        "untimed_interleaved_bitwise_ok": seq_ok, "untimed_calls": n_seq,
        "untimed_bad": seq_bad[:20], "load1": {"before_load_wait": la_wait, **load_at},
        "load_ok": all(v < 4 for k, v in load_at.items() if k.endswith("_start")),
        "n_arm_warm": n_arm_warm, "n_rounds_per_arm": n_rounds, "round_iters": round_iters,
        "measure": "Flux3Pipeline.infer entry (ref PNG bytes in) -> first transformer forward, "
                   "after cuda sync there; served engine.json pipeline, 2048^2; arms A (main's "
                   "functions copied) and B (tree) interleaved by rounds in one process",
    })
    print(json.dumps(out))


def pbench():
    """Per-iteration paired interleave in ONE process on ONE pipeline (top-lead ruling 17:12).

    Arm A = main's _load_image_pixels / pack_i2i_image_condition / _prepare_i2i (copied
    verbatim above); arm B = this tree's functions (in the base tree: main's code again, an
    A/A). Per case: N_ARM_WARM warmup iterations, then N_PAIRED iterations; iteration i runs
    both arms on the same input (i2i alternates the two frozen references by i), order AB on
    even i and BA on odd i. Reports per-arm stats and the per-iteration paired difference
    B - A (median, IQR) and ratio B / A. load1 recorded at start/end of each timed section
    (no load gate). Then an untimed interleaved sequence of the same shape: every
    transformer input at the first denoise call bitwise equal between arms and to iter 0.
    """
    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as flux3

    n_warm = int(os.environ.get("N_ARM_WARM", "10"))
    n_paired = int(os.environ.get("N_PAIRED", "200"))
    pipe = load_pipe()
    probe = Probe(pipe)
    refs = find_refs()
    cand_load = flux3._load_image_pixels
    cand_prep = type(pipe)._prepare_i2i

    def use(arm):
        if arm == "A":
            flux3._load_image_pixels = ref_load_image_pixels
            pipe._prepare_i2i = types.MethodType(ref_prepare_i2i, pipe)
        else:
            flux3._load_image_pixels = cand_load
            pipe.__dict__.pop("_prepare_i2i", None)
        assert flux3._load_image_pixels is (ref_load_image_pixels if arm == "A" else cand_load)
        assert pipe._prepare_i2i.__func__ is (ref_prepare_i2i if arm == "A" else cand_prep)

    def req(kind):  # rebuilt from the immutable reference bytes every call (pristine inputs)
        if kind == "t2i":
            return make_req(*T2I)
        return make_req(PROMPTS[kind][0], PROMPTS[kind][1], refs[kind])

    cases = {"i2i-2048": ["A", "B"], "t2i-2048": ["t2i"]}

    def q(v, n):
        return statistics.quantiles(sorted(v), n=n)

    out = {"batch": {}, "output": {"class": "exact"}, "arms": {}, "paired": {}, "load1": {}}
    for case, kinds in cases.items():
        for i in range(n_warm):
            for arm in "AB":
                use(arm)
                span(pipe, probe, req(kinds[i % len(kinds)]))
        out["load1"][f"{case}_start"] = loadavg1()
        log(f"{case}: timed start, load1={out['load1'][case + '_start']:.2f}")
        t = {"A": [], "B": []}
        by_ref = {"A": {}, "B": {}}
        for i in range(n_paired):
            kind = kinds[i % len(kinds)]
            for arm in ("AB" if i % 2 == 0 else "BA"):
                use(arm)
                g, _ = span(pipe, probe, req(kind))
                t[arm].append(g)
                by_ref[arm].setdefault(kind, []).append(g)
            if i % 25 == 0:
                log(f"{case} iter {i}: A {t['A'][-1]:.2f} ms, B {t['B'][-1]:.2f} ms")
        out["load1"][f"{case}_end"] = loadavg1()
        log(f"{case}: timed end, load1={out['load1'][case + '_end']:.2f}")
        diff = [b - a for a, b in zip(t["A"], t["B"])]
        ratio = [b / a for a, b in zip(t["A"], t["B"])]
        dq, rq = q(diff, 4), q(ratio, 4)
        out["arms"][case] = {"A_main_copy": stats(t["A"]), "B_tree": stats(t["B"]),
                             "per_reference": {arm: {k: stats(v) for k, v in by_ref[arm].items()}
                                               for arm in "AB"}}
        out["paired"][case] = {
            "diff_B_minus_A_ms": {"median": statistics.median(diff), "p25": dq[0], "p75": dq[2],
                                  "iqr": dq[2] - dq[0]},
            "ratio_B_over_A": {"median": statistics.median(ratio), "p25": rq[0], "p75": rq[2]},
            "n": len(diff)}
        nf = max(stats(v)["p10_p90_over_median"] for v in by_ref["B"].values())
        out["batch"][case] = {"step_ms": statistics.median(t["B"]), "noise_floor": nf}
        log(f"{case}: {json.dumps(out['paired'][case])}")
    # Untimed interleaved sequence, same shape, bitwise.
    probe.capture = True
    first, bad, n_seq = {}, [], 0
    for case, kinds in cases.items():
        for i in range(n_warm + n_paired):
            kind = kinds[i % len(kinds)]
            for arm in ("AB" if i % 2 == 0 else "BA"):
                use(arm)
                span(pipe, probe, req(kind))
                kw = probe.kwargs
                n_seq += 1
                if kind not in first:
                    first[kind] = kw
                    continue
                f = first[kind]
                if not (kw.keys() == f.keys() and all(
                        same(kw[n][0], f[n][0]) and kw[n][1] == f[n][1] for n in kw)):
                    bad.append((case, i, arm))
    probe.capture = False
    use("B")
    log(f"untimed interleaved sequence: {n_seq} calls, ok={not bad}, bad={bad[:10]}")
    out.update({
        "untimed_interleaved_bitwise_ok": not bad, "untimed_calls": n_seq, "untimed_bad": bad[:20],
        "n_arm_warm": n_warm, "n_paired": n_paired,
        "measure": "Flux3Pipeline.infer entry (ref PNG bytes in) -> first transformer forward, "
                   "after cuda sync there; served engine.json pipeline, 2048^2; arms A (main's "
                   "functions copied) and B (tree) interleaved per iteration in one process",
    })
    print(json.dumps(out))


def check():
    import torch._dynamo.utils as dyn_utils

    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as flux3

    pipe = load_pipe()
    probe = Probe(pipe)
    runner = pipe._cuda_graph_runners.get("transformer")
    replays = {"n": 0}
    if runner is not None:
        orig_replay = runner.replay

        def counting_replay(*a, **k):
            replays["n"] += 1
            return orig_replay(*a, **k)

        runner.replay = counting_replay
    refs = find_refs()
    cand_load, cand_pack = flux3._load_image_pixels, flux3.pack_i2i_image_condition
    rec = {}

    def rec_load(fn):
        def w(source, *, device):
            out = fn(source, device=device)
            rec["pixels"] = (out.clone(), out.stride())
            return out
        return w

    orig_encode = pipe.vae.encode

    def rec_encode(x):
        out = orig_encode(x)
        rec.setdefault("latents", []).append(out.clone())
        return out

    pipe.vae.encode = rec_encode

    def use(arm):
        if arm == "ref":
            flux3._load_image_pixels = rec_load(ref_load_image_pixels)
            pipe._prepare_i2i = types.MethodType(ref_prepare_i2i, pipe)
        else:
            flux3._load_image_pixels = rec_load(cand_load)
            if "_prepare_i2i" in pipe.__dict__:
                del pipe.__dict__["_prepare_i2i"]

    def snap():
        return {k: {kk: vv for kk, vv in v.items()} for k, v in dyn_utils.counters.items()
                if k in ("frames", "stats", "recompiles", "inductor")}

    graphs0 = len(runner.graphs) if runner is not None else None
    dyn0 = snap()
    reqs = {k: make_req(PROMPTS[k][0], PROMPTS[k][1], refs[k]) for k in "AB"}
    reqs["t2i"] = make_req(*T2I)
    full = []
    ok = True
    first_cand = {}
    for name in ("A", "B", "A", "t2i"):
        row = {"input": name}
        outs = {}
        for arm in ("ref", "cand"):
            use(arm)
            rec.clear()
            replays["n"] = 0
            torch.cuda.synchronize()
            t0 = time.time()
            out = pipe.infer(reqs[name])
            torch.cuda.synchronize()
            outs[arm] = {"image": out.image.clone(), "pixels": rec.get("pixels"),
                         "latents": rec.get("latents")}
            row[f"{arm}_s"] = round(time.time() - t0, 2)
            row[f"{arm}_replays"] = replays["n"]
        r, c = outs["ref"], outs["cand"]
        row["image_bitwise"] = same(r["image"], c["image"])
        if name != "t2i":
            row["pixels_bitwise"] = same(r["pixels"][0], c["pixels"][0])
            row["pixel_strides"] = {"ref": list(r["pixels"][1]), "cand": list(c["pixels"][1])}
            row["latents_bitwise"] = all(same(a, b) for a, b in zip(r["latents"], c["latents"]))
            ok &= row["pixels_bitwise"] and row["latents_bitwise"] and r["pixels"][1] == c["pixels"][1]
        if name in first_cand:
            row["cand_equals_first"] = same(first_cand[name]["cand"], c["image"])
            row["ref_equals_first"] = same(first_cand[name]["ref"], r["image"])
            ok &= row["cand_equals_first"]
        else:
            first_cand[name] = {"cand": c["image"], "ref": r["image"]}
        ok &= row["image_bitwise"] and row["cand_replays"] == 16
        full.append(row)
        log(json.dumps(row))
    # Untimed truncated sequence shaped like bench: every transformer input at the first
    # denoise call vs the reference path's.
    probe.capture = True
    ref_kw = {}
    use("ref")
    for k in "AB":
        span(pipe, probe, reqs[k])
        ref_kw[k] = probe.kwargs
    use("cand")
    seq_ok = True
    seq_bad = []
    for i in range(N_WARM + N_ITERS):
        k = "AB"[i % 2]
        span(pipe, probe, reqs[k])
        kw = probe.kwargs
        good = kw.keys() == ref_kw[k].keys() and all(
            same(kw[n][0], ref_kw[k][n][0]) and kw[n][1] == ref_kw[k][n][1] for n in kw)
        if not good:
            seq_bad.append(i)
        seq_ok &= good
        span(pipe, probe, reqs["t2i"])
    probe.capture = False
    log(f"truncated sequence: {N_WARM + N_ITERS} iters, ok={seq_ok}, bad={seq_bad}, "
        f"inputs compared={sorted(ref_kw['A'].keys())}")
    dyn1 = snap()
    graphs1 = len(runner.graphs) if runner is not None else None
    no_recompile = dyn0 == dyn1
    ok &= seq_ok and no_recompile and graphs0 == graphs1
    print(json.dumps({
        "passed": bool(ok),
        "repeated_execution_passed": bool(all(r.get("cand_equals_first", True) for r in full) and seq_ok),
        "compared": "in-process reference = main's _load_image_pixels/pack_i2i_image_condition/"
                    "_prepare_i2i (copied verbatim) on the same served pipeline; full requests "
                    "A,B,A,t2i ref/cand interleaved: output uint8 image, reference pixels "
                    "(values+strides), VAE latents bitwise; A3==A1; untimed truncated sequence "
                    "(N_WARM+N_ITERS alternating A/B, t2i between) comparing every tensor "
                    "kwarg of the first transformer call bitwise with the reference",
        "full_requests": full,
        "truncated_seq_ok": seq_ok, "truncated_seq_bad": seq_bad,
        "truncated_inputs_compared": sorted(ref_kw["A"].keys()),
        "cuda_graphs_before": graphs0, "cuda_graphs_after": graphs1,
        "dynamo_no_change_after_warmup": no_recompile, "dynamo_before": dyn0, "dynamo_after": dyn1,
    }))


if __name__ == "__main__":
    {"unit": unit, "bench": bench, "ibench": ibench, "pbench": pbench, "check": check}[MODE]()
