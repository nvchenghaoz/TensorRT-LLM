"""Branch 32 microbenchmark: one FLUX.3 Klein denoise step's adaLN modulation work, real shapes.

Per step (B=1, S_img=4096, T=80, D=3072, bf16, Inductor with emulate_precision_casts):
  modulation build: early/single shift/scale/gate per stream, joint single_* tables,
  5 image mode blocks (M=4096) + 5 text mode blocks (M=80): LN+modulate and gated tail,
  28 single blocks (M=4176): LN+modulate and gated tail.
GEMM/attention outputs are fixed synthetic tensors (they are not changed by this branch).
"base" materializes per-token [1,S,D] modulation exactly as main's _packed_stream/joint();
"cand" uses the branch's broadcast rows / [1,2,D] tables + select mask through the branch's
own _flux3_select_mod. The arm is picked by whether the tree has FLUX3_BROADCAST_ADALN.

mode "time" (default): capture STEPS steps in one CUDA graph, replay ITERS times, report
median ms per step. mode "check": in the candidate tree, run the base and the candidate
sequences untimed (same warmup and iteration counts, same reset points) and compare every
output bitwise with the base reference and with iteration 0; inputs checked unchanged.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf

D, S_IMG, T = 3072, 4096, int(os.environ.get("B32_T", "80"))
S = S_IMG + T
STEPS, ITERS, WARM = 2, int(os.environ.get("B32_ITERS", "60")), 5
dev = "cuda"
HAS_CAND = hasattr(tf, "FLUX3_BROADCAST_ADALN")
MODE = sys.argv[1] if len(sys.argv) > 1 else "time"


def adaln(x, shift, scale):
    return (1 + scale) * F.layer_norm(x, (D,), eps=1e-6) + shift


def tail(x, gate, out):
    return x + gate * out


def block_parts(mod, x, out):
    shift, scale, gate = mod
    return adaln(x, shift, scale), tail(x, gate, out)


def single_parts(mod, x, out):
    shift, scale, gate = tf._flux3_select_mod(mod) if len(mod) == 4 else mod
    return adaln(x, shift, scale), tail(x, gate, out)


opts = {"emulate_precision_casts": True}
mode_fns = {}
single_fn = {}


def compiled(kind, key):
    table = mode_fns if kind == "mode" else single_fn
    if key not in table:
        fn = block_parts if kind == "mode" else single_parts
        table[key] = torch.compile(fn, dynamic=False, options=opts)
    return table[key]


g = torch.Generator(device=dev).manual_seed(0)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, device=dev, generator=g) * scale).to(torch.bfloat16)


# Modulation linear outputs per stream/stage: [1, 1, 3D] (chunked like Flux3Modulation).
mods_raw = {
    (s, st): rnd(1, 1, 3 * D, scale=0.3) for s in ("image", "text") for st in ("early", "single")
}
x_img, x_txt, x_joint = rnd(1, S_IMG, D), rnd(1, T, D), rnd(1, S, D)
o_img = [rnd(1, S_IMG, D) for _ in range(5)]
o_txt = [rnd(1, T, D) for _ in range(5)]
o_joint = [rnd(1, S, D) for _ in range(28)]
pristine = [t.clone() for t in [x_img, x_txt, x_joint, *mods_raw.values()]]
lengths = {"image": S_IMG, "text": T}


def stream_mods(name, broadcast):
    state = {}
    for st in ("early", "single"):
        for field, value in zip(("shift", "scale", "gate"), mods_raw[(name, st)].chunk(3, dim=-1)):
            if broadcast:
                state[st + "_" + field] = value[:, 0:1]
            else:
                state[st + "_" + field] = torch.cat(
                    [value[:, 0:1].expand(-1, lengths[name], -1)], dim=1
                )
    return state


def step(broadcast):
    outs = []
    streams = {n: stream_mods(n, broadcast) for n in ("image", "text")}
    for name, x, os_ in (("image", x_img, o_img), ("text", x_txt, o_txt)):
        mod = tuple(streams[name]["early_" + f] for f in ("shift", "scale", "gate"))
        fn = compiled("mode", (name, broadcast))
        for o in os_:
            outs.extend(fn(mod, x, o))
    fields = ("single_shift", "single_scale", "single_gate")
    chunks = (("image", 0, S_IMG), ("text", 0, T))
    if broadcast:
        select = tf.Flux3Model._joint_select(list(chunks), x_joint.device)
        mod = tuple(torch.cat([streams[n][f] for n, _, _ in chunks], dim=1) for f in fields)
        mod = mod + (select,)
    else:
        mod = tuple(torch.cat([streams[n][f][:, s:e] for n, s, e in chunks], dim=1) for f in fields)
    fn = compiled("single", broadcast)
    for o in o_joint:
        outs.extend(fn(mod, x_joint, o))
    return outs


def reset():
    for t, p in zip([x_img, x_txt, x_joint, *mods_raw.values()], pristine):
        t.copy_(p)


def inputs_ok():
    return all(torch.equal(t, p) for t, p in zip([x_img, x_txt, x_joint, *mods_raw.values()],
                                                 pristine))


if MODE == "check":
    assert HAS_CAND, "check runs in the candidate tree"
    fails = []
    ref = None
    cand0 = None
    maxabs = 0.0
    for arm in (False, True):
        reset()
        for _ in range(WARM):
            step(arm)
        torch.cuda.synchronize()
        if not inputs_ok():
            fails.append(f"inputs changed during warmup arm={arm}")
        reset()
        for i in range(ITERS):
            outs = step(arm)
            if ref is None:
                ref = [o.clone() for o in outs]  # base iteration 0 = reference
            if arm and cand0 is None:
                cand0 = [o.clone() for o in outs]
            for j, o in enumerate(outs):
                if not torch.equal(o, ref[j]):
                    maxabs = max(maxabs, (o.float() - ref[j].float()).abs().max().item())
                    fails.append(f"arm={arm} iter {i} out {j} != base iter0 reference")
                if arm and not torch.equal(o, cand0[j]):
                    fails.append(f"cand iter {i} out {j} != cand iter0")
            del outs
        torch.cuda.synchronize()
        if not inputs_ok():
            fails.append(f"inputs changed arm={arm}")
    print(json.dumps({"passed": not fails, "failures": fails[:20], "n_failures": len(fails),
                      "max_abs": maxabs, "iters": ITERS, "outputs_per_iter": len(ref),
                      "compared_against": "main's materialized per-token modulation path "
                      "(base arm, same compiled ops) iteration 0 and every iteration"}))
    sys.exit(0)

broadcast = HAS_CAND
print(f"arm={'cand' if broadcast else 'base'} T={T} S={S}", flush=True)
reset()
for _ in range(WARM):
    step(broadcast)
torch.cuda.synchronize()
reset()
graph = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        step(broadcast)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
with torch.cuda.graph(graph):
    for _ in range(STEPS):
        step(broadcast)
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
                  "output": {"class": "exact"},
                  "detail": {"arm": "cand" if broadcast else "base", "p10": p10, "p90": p90,
                             "unit": "ms per denoise step of adaLN modulation work"}}))
