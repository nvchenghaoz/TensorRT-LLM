"""Branch 49 (idea 100) microbenchmark: one FLUX.3 Klein denoise step's single-block adaLN work
when the text stream comes from the step-batched flow-map path (landing 21).

Per step (B=1, S_img=4096, T=80, D=3072, bf16, Inductor emulate_precision_casts, dynamic=False):
  text single_* rows = Flux3Model.flow_map_text_step(prepared, step)   <- THE TREE'S OWN CODE
  image single_* rows = [1, 1, D] broadcast rows (as _packed_stream gives on SM120)
  forward's joint/table decision (copied verbatim from Flux3Model.forward): [1,2,D] tables +
  select mask when every chunk's single_* has shape[1] == 1, else joint() = torch.cat of
  per-token [1,S,D] mods; then 28 single blocks: LN+modulate and gated tail (separate compiled
  fns, as GEMMs sit between them in the real block), via the tree's _flux3_select_mod.
GEMM/attention outputs are fixed synthetic tensors (unchanged by this branch).
Base tree (main): flow_map_text_step expands the text rows -> per-token path.
Candidate tree: [1,1,D] rows -> table path.

mode "time" (default): capture STEPS steps in one CUDA graph, replay ITERS times interleaved
order irrelevant (one arm per process), median ms/step.
mode "check" (candidate tree only): run the OFF arm (tf.FLUX3_TEXT_STEP_BROADCAST=False ==
main's expand) and the ON arm untimed with the same warmup/iteration counts and reset points;
compare every output of every iteration with OFF iteration 0 (declared tolerance below) and
the ON outputs bitwise with ON iteration 0; inputs checked unchanged before each reset; also
records the mod tuple arity/shapes each arm hands the single blocks.
Declared tolerance (reordered: Inductor picks another reduction config for the LN+modulate
kernel when the modulation is a table, ruling 565 / branch 32): <= 1 bf16 ulp of
max(|ref|, 2) per element.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf

D, S_IMG, T = 3072, 4096, int(os.environ.get("B49_T", "80"))
NSTEPS = 16
S = S_IMG + T
STEPS, ITERS, WARM = 2, int(os.environ.get("B49_ITERS", "60")), 5
NBLK = 28
dev = "cuda"
MODE = sys.argv[1] if len(sys.argv) > 1 else "time"
HAS_SWITCH = hasattr(tf, "FLUX3_TEXT_STEP_BROADCAST")
opts = {"emulate_precision_casts": True}


def adaln(x, shift, scale):
    return (1 + scale) * F.layer_norm(x, (D,), eps=1e-6) + shift


def head_part(mod, x):
    shift, scale, _ = tf._flux3_select_mod(mod)
    return adaln(x, shift, scale)


def tail_part(mod, x, out):
    _, _, gate = tf._flux3_select_mod(mod)
    return x + gate * out


fns = {}


def compiled(key):
    if key not in fns:
        head = torch.compile(head_part, dynamic=False, options=opts)
        tl_ = torch.compile(tail_part, dynamic=False, options=opts)
        fns[key] = (head, tl_)
    return fns[key]


g = torch.Generator(device=dev).manual_seed(0)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, device=dev, generator=g) * scale).to(torch.bfloat16)


prepared = {"_flux3_text_steps_tokens": rnd(1, NSTEPS * T, D)}
for f in ("shift", "scale", "gate"):
    prepared["_flux3_text_steps_single_" + f] = rnd(1, NSTEPS, D, scale=0.3)
img = {"tokens": rnd(1, S_IMG, D)}
for f in ("shift", "scale", "gate"):
    img["single_" + f] = rnd(1, 1, D, scale=0.3)
x_joint = rnd(1, S, D)
o_joint = [rnd(1, S, D) for _ in range(NBLK)]
mutable = [*prepared.values(), *img.values(), x_joint]
pristine = [t.clone() for t in mutable]
seen = {}


def step(i):
    """forward()'s single-block modulation build + 28 blocks' adaLN work for step i."""
    text = tf.Flux3Model.flow_map_text_step(prepared, i)
    streams = {
        "text": {
            "tokens": text["_flux3_text_step_tokens"],
            **{"single_" + f: text["_flux3_text_step_single_" + f] for f in ("shift", "scale", "gate")},
        },
        "image": img,
    }
    chunks = [("text", 0, T), ("image", 0, S_IMG)]

    def rows(name, field, start, end):
        value = streams[name][field]
        if value.shape[1] == 1:
            return value.expand(-1, end - start, -1)
        return value[:, start:end]

    def joint(field):
        return torch.cat([rows(name, field, start, end) for name, start, end in chunks], dim=1)

    combined = joint("tokens")
    mod_fields = tuple("single_" + field for field in ("shift", "scale", "gate"))
    if len(chunks) == 2 and all(
        streams[name][field].shape[1] == 1 for name, _, _ in chunks for field in mod_fields
    ):
        select = tf.Flux3Model._joint_select(chunks, combined.device)
        mod = tuple(
            torch.cat([streams[name][field] for name, _, _ in chunks], dim=1)
            for field in mod_fields
        ) + (select,)
    else:
        mod = tuple(joint(field) for field in mod_fields)
    seen.setdefault(len(mod), [list(t.shape) for t in mod])
    head, tl_ = compiled(len(mod))
    outs = [combined]
    for o in o_joint:
        outs.append(head(mod, x_joint))
        outs.append(tl_(mod, x_joint, o))
    return outs


def reset():
    for t, p in zip(mutable, pristine):
        t.copy_(p)


def inputs_ok():
    return all(torch.equal(t, p) for t, p in zip(mutable, pristine))


if MODE == "check":
    assert HAS_SWITCH, "check runs in the candidate tree"
    fails, maxabs, maxrel_vs_scale, ndiff = [], 0.0, 0.0, 0
    ref = None
    arms = {}
    for on in (False, True):
        tf.FLUX3_TEXT_STEP_BROADCAST = on
        seen.clear()
        reset()
        for w in range(WARM):
            step(w % STEPS)
        torch.cuda.synchronize()
        if not inputs_ok():
            fails.append(f"inputs changed during warmup on={on}")
        reset()
        first = None
        for it in range(ITERS):
            for s in range(STEPS):
                outs = step(s)
                if it == 0:
                    if not on:
                        ref = ref or {}
                        ref[s] = [o.clone() for o in outs]
                    first = first or {}
                    first[s] = [o.clone() for o in outs]
                for j, o in enumerate(outs):
                    r = ref[s][j]
                    if not torch.equal(o, r):
                        d = (o.float() - r.float()).abs()
                        maxabs = max(maxabs, d.max().item())
                        ulp = torch.exp2(torch.floor(torch.log2(r.float().abs().clamp_min(2.0))) - 7)
                        maxrel_vs_scale = max(maxrel_vs_scale, (d / ulp).max().item())
                        ndiff += int((d > 0).sum())
                        if not on or bool((d > ulp).any()):
                            fails.append(f"on={on} iter {it} step {s} out {j} beyond 1 ulp of OFF iter0")
                    if not torch.equal(o, first[s][j]):
                        fails.append(f"on={on} iter {it} step {s} out {j} != own iter0")
                del outs
        torch.cuda.synchronize()
        if not inputs_ok():
            fails.append(f"inputs changed on={on}")
        arms["on" if on else "off"] = {"mod_arity": list(seen.keys()), "mod_shapes": list(seen.values())}
    tf.FLUX3_TEXT_STEP_BROADCAST = True
    if arms["off"]["mod_arity"] != [3] or arms["on"]["mod_arity"] != [4]:
        fails.append(f"unexpected mod arity {arms}")
    print(json.dumps({
        "passed": not fails, "failures": fails[:20], "n_failures": len(fails),
        "output": {"class": "reordered", "tolerance": "<= 1 bf16 ulp of max(|ref|,2) per element; "
                   "each arm bitwise with its own iteration 0"},
        "arms": arms, "elements_differing_total": ndiff, "max_abs": maxabs,
        "max_err_in_ulps_of_max_ref_2": maxrel_vs_scale, "iters": ITERS, "steps_per_iter": STEPS,
        "compared_against": "OFF arm = main's flow_map_text_step (expand to [1,T,D]) -> per-token "
        "joint() path, iteration 0, same compiled ops"}))
    sys.exit(0)

arm = "cand" if HAS_SWITCH else "base"
print(f"arm={arm} T={T} S={S}", flush=True)
reset()
for w in range(WARM):
    step(w % STEPS)
torch.cuda.synchronize()
print("mod seen", seen, flush=True)
reset()
graph = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(2):
        step(i)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
with torch.cuda.graph(graph):
    for i in range(STEPS):
        step(i)
for _ in range(5):
    graph.replay()
torch.cuda.synchronize()
times = []
for i in range(ITERS):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    graph.replay()
    b.record()
    b.synchronize()
    times.append(a.elapsed_time(b) / STEPS)
    if i % 20 == 0:
        print(f"iter {i} {times[-1]:.4f} ms/step", flush=True)
times.sort()
med = times[len(times) // 2]
p10, p90 = times[len(times) // 10], times[(9 * len(times)) // 10]
print(json.dumps({"batch": {"1": {"kernel_ms": med, "noise_floor": (p90 - p10) / med}},
                  "output": {"class": "reordered",
                             "max_abs_vs_base": "<= 1 bf16 ulp of max(|base|,2) (check mode)"},
                  "detail": {"arm": arm, "p10": p10, "p90": p90, "mod_seen": {str(k): v for k, v in seen.items()},
                             "unit": "ms per denoise step: single-block modulation build + 28 x "
                             "(LN+modulate, gated tail) at M=4176"}}))
