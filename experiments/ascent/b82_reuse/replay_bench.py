"""Branch 82 microbench: CUDAGraphRunner.replay input staging at Flux3 Klein served shapes.

No model load. A CUDAGraphRunner (this tree's) wraps a tiny forward that reads every kwarg
(so the graph depends on all static inputs) and returns a small output. One "request" = 16
replays with the same kwargs the real _run_denoise passes: the 13 step-invariant tensors are the
SAME objects for all 16 steps (fresh objects per request), the per-step tensors (x_image,
*_timesteps, timesteps_ref, timestep, _flux3_text_step_*, _flux3_image_step_*) change per step.
Shapes reconstruct the served T=80, 1024x1024 call (x_text [80, 20480] bf16 = 3.28 MB; ~16 MB
step-invariant in 13 tensors). Baseline tree (main) has no register_reusable_input_fn, so it
copies every tensor kwarg on every replay; the candidate registers pipeline_flux3's allowlist
and flag exactly as _setup_cuda_graphs does.

Timing: CUDA events around each request's 16 replays (host runs ahead, as served), 40 warmup
requests, 300 timed requests, report median ms/request and (p90-p10)/median.
Correctness (untimed, same sequence): every replay's output is compared bitwise with the eager
forward on the same per-step inputs, and the source tensors are checked unchanged.
"""

import json
import statistics
import sys

import torch

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig

dev = torch.device("cuda")
STEPS = 16
WARM = 40
ITERS = 300
T, S_IMG, D, C_IMG, CTX = 80, 4096, 3072, 128, 20480
J = T + S_IMG
g = torch.Generator(device=dev).manual_seed(0)


def rnd(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device=dev, dtype=torch.float32, generator=g).to(dtype)


def invariant():
    return {
        "x_text": rnd(T, CTX),
        "x_text_ids": torch.randint(0, 100, (T, 4), device=dev, generator=g),
        "x_image_ids": torch.randint(0, 100, (S_IMG, 4), device=dev, generator=g),
        "_flux3_text_cos": rnd(1, T, 1, 128, dtype=torch.float32),
        "_flux3_text_sin": rnd(1, T, 1, 128, dtype=torch.float32),
        "_flux3_image_cos": rnd(1, S_IMG, 1, 128, dtype=torch.float32),
        "_flux3_image_sin": rnd(1, S_IMG, 1, 128, dtype=torch.float32),
        "_flux3_joint_cos": rnd(1, J, 1, 128, dtype=torch.float32),
        "_flux3_joint_sin": rnd(1, J, 1, 128, dtype=torch.float32),
        "_flux3_text_cos_sin": rnd(T, 128, dtype=torch.float32),
        "_flux3_image_cos_sin": rnd(S_IMG, 128, dtype=torch.float32),
        "_flux3_joint_cos_sin": rnd(J, 128, dtype=torch.float32),
        "_flux3_joint_select": (torch.arange(J, device=dev) >= T).view(1, J, 1),
    }


def per_step(text_steps, image_steps, step):
    return {
        "x_image": rnd(S_IMG, C_IMG),
        "x_image_timesteps": torch.full((S_IMG,), 0.5 + step, device=dev, dtype=torch.bfloat16),
        "x_text_timesteps": torch.zeros(T, device=dev, dtype=torch.bfloat16),
        "timesteps_ref": torch.full((1,), 0.1 * step, device=dev, dtype=torch.bfloat16),
        "_flux3_text_step_tokens": text_steps[:, step * T:(step + 1) * T],
        "_flux3_text_step_single_shift": image_steps[step, :, :, :D],
        "_flux3_text_step_single_scale": image_steps[step, :, :, D:2 * D],
        "_flux3_text_step_single_gate": image_steps[step, :, :, 2 * D:],
        "_flux3_image_step_early": image_steps[step],
        "_flux3_image_step_single": image_steps[(step + 1) % STEPS],
        "_flux3_image_step_final": image_steps[step, :, :, :2 * D],
        "timestep": torch.tensor(0.9 - 0.05 * step, device=dev),
    }


def forward(**kw):
    # Read every tensor input so the graph depends on all of them (one tiny reduction each).
    acc = torch.zeros(1, device=dev, dtype=torch.float32)
    for k in sorted(kw):
        v = kw[k]
        if isinstance(v, torch.Tensor):
            acc = acc + v.reshape(-1)[:64].float().sum()
    return {"x_image": kw["x_image"] * 2 + acc.to(kw["x_image"].dtype)}


runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
registered = False
if hasattr(runner, "register_reusable_input_fn"):
    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as pf

    runner.register_reusable_input_fn(
        pf._FLUX3_GRAPH_REUSABLE_INPUTS.__contains__, enabled=pf._flux3_graph_reuse_inputs_enabled
    )
    registered = pf.FLUX3_GRAPH_REUSE_INPUTS
wrapped = runner.wrap(forward)
print("registered copy-skip:", registered, flush=True)

# Two prebuilt request input sets, alternated, so each request's invariant tensors are new
# objects relative to the previous request (as served: fresh per request).
reqs = []
for r in range(2):
    inv = invariant()
    text_steps = rnd(1, STEPS * T, D)
    image_steps = rnd(STEPS, 1, 1, 3 * D)
    steps = [per_step(text_steps, image_steps, s) for s in range(STEPS)]
    reqs.append((inv, steps))
torch.cuda.synchronize()


def request(i, check=False):
    inv, steps = reqs[i % 2]
    outs = []
    for s in range(STEPS):
        out = wrapped(**inv, **steps[s], layout=(("image", (S_IMG,)), ("text", (T,))))
        if check:
            outs.append(out["x_image"].clone())
    return outs


with torch.inference_mode():
    request(0)  # capture
    torch.cuda.synchronize()
    for i in range(WARM):
        request(i)
    torch.cuda.synchronize()
    times = []
    for i in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        request(i)
        e.record()
        times.append((s, e))
        if i % 100 == 99:
            print("timed", i + 1, flush=True)
    torch.cuda.synchronize()
    ms = sorted(a.elapsed_time(b) for a, b in times)

    # Untimed correctness: same warmup/iteration pattern (alternating request sets), each replay
    # output vs eager forward on the same inputs, and sources unchanged.
    pristine = [({k: v.clone() for k, v in inv.items()}, [{k: v.clone() for k, v in st.items()}
                 for st in steps]) for inv, steps in reqs]
    ok = True
    n_cmp = 0
    for i in range(8):
        outs = request(i, check=True)
        inv, steps = reqs[i % 2]
        for s in range(STEPS):
            ref = forward(**inv, **steps[s])["x_image"]
            n_cmp += 1
            if not torch.equal(outs[s], ref):
                ok = False
                print("MISMATCH request", i, "step", s, flush=True)
        pinv, psteps = pristine[i % 2]
        for k, v in inv.items():
            if not torch.equal(v, pinv[k]):
                ok = False
                print("SOURCE CHANGED", k, flush=True)
    # The two request sets must give different outputs (a stale replay would make them equal).
    o0, o1 = request(0, check=True), request(1, check=True)
    distinct = not torch.equal(o0[3], o1[3])
    ok = ok and distinct


def pct(v, q):
    i = (len(v) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


med = statistics.median(ms)
print(json.dumps({
    "batch": {"1": {"step_ms": med, "noise_floor": (pct(ms, 0.9) - pct(ms, 0.1)) / med}},
    "output": {"class": "exact"},
    "unit": "ms per request = 16 CUDAGraphRunner replays (input staging + tiny graph)",
    "p10": pct(ms, 0.1), "p90": pct(ms, 0.9),
    "registered_copy_skip": registered,
    "correct": ok, "replays_compared": n_cmp, "requests_distinct": distinct,
}))
