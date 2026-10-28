"""Branch 44 (idea 92, adaLN Part C): per-request RoPE cos_sin table vs a cat in every fused QKV.

Runs the tree's own primsts.qkv_forward under torch.compile (dynamic=False,
emulate_precision_casts, like pipeline_flux3.torch_compile) with the fused QKV op and
qkv_planes replaced by an opaque no-op custom op, so the only GPU work left is what this
branch changes: main's Inductor cat kernel (triton_poi_fused_cat_*) building the FP32
[rows,128] cos_sin table on every call.

Timed unit = one request on the served shape (1024^2, T=80, 16 steps):
  16 steps x (24 calls at M=4176 + 4 calls at M=4096) = 448 qkv_forward calls,
  plus, when the tree has transformer_flux3._block_freqs (the branch), the 3 per-request
  table builds _packed_positions adds (text M=80, image M=4096, joint M=4176).
Captured in one CUDA graph, replayed REPS times, median ms per request (kernel time).

mode "check" (branch tree): untimed, same warmup/iteration counts as timing. Every call's
table as the op receives it is compared bitwise with the protected reference (eager torch.cat
of main's expression on the pristine cos/sin) and with iteration 0, for the branch path
(3-tuple freqs) and the fallback path (2-tuple freqs, main's inline cat); inputs checked
unchanged before each reset; CUDA graph replays compared too.
"""

import json
import sys

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.flux3 import primsts as P
from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf

MODE = sys.argv[1] if len(sys.argv) > 1 else "time"
HAS_CAND = hasattr(tf, "_block_freqs")
STEPS, WARM, ITERS, REPS = 16, 3, 5, 60
D = 3072
dev = "cuda"
SEEN = []


def log(*a):
    print(*a, flush=True)


@torch.library.custom_op("b44::sink", mutates_args=(), device_types="cuda")
def sink(data: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    if MODE == "check":
        SEEN.append(cos_sin.clone())
    return data.new_empty(data.shape[0], 0)


@sink.register_fake
def _(data, cos_sin):
    return data.new_empty(data.shape[0], 0)


def fake_fp8(data, weight, scales, scale, nq, nk, cos_sin, _):
    return torch.ops.b44.sink(data, cos_sin)


def fake_planes(output, data, weight, head_dim, qkv_scale=False):
    # Return the op's (empty) output so the op and its table input stay live in the graph.
    return data, data, output


P.fp8_qkv_qknorm_rope = fake_fp8
P.qkv_planes = fake_planes


class Adapter:
    weight = None
    scale = None

    def can_run(self, x):
        return True

    def activation(self, x):
        return x.reshape(-1, x.shape[-1]), None


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self._flux3_primsts_nvfp4_qkv = None
        self._primsts_fp8_qkv_rope = True
        self.qkv_proj = nn.Module()
        self.qkv_proj._flux3_primsts_fp8 = Adapter()
        self.norm_q = nn.Module()
        self.norm_q.weight = nn.Parameter(torch.ones(128, device=dev, dtype=torch.bfloat16))
        self.norm_k = nn.Module()
        self.norm_k.weight = nn.Parameter(torch.ones(128, device=dev, dtype=torch.bfloat16))


attn = Attn()
fwd = torch.compile(
    lambda x, freqs: P.qkv_forward(attn, x, freqs),
    dynamic=False,
    options={"emulate_precision_casts": True},
)

g = torch.Generator(device=dev).manual_seed(0)


def make_freqs(rows):
    # Real layout: _pe_to_block_freqs gives (1, S, 1, 128) fp32 repeat-interleaved cos/sin.
    ang = torch.rand(1, rows, 1, 64, device=dev, generator=g) * 100
    cos = ang.cos().repeat_interleave(2, dim=-1).contiguous()
    sin = ang.sin().repeat_interleave(2, dim=-1).contiguous()
    return cos, sin


streams = {"text": make_freqs(80), "image": make_freqs(4096), "joint": make_freqs(4176)}
pristine = {k: (c.clone(), s.clone()) for k, (c, s) in streams.items()}
xs = {
    "image": torch.randn(1, 4096, D, device=dev, generator=g).to(torch.bfloat16),
    "joint": torch.randn(1, 4176, D, device=dev, generator=g).to(torch.bfloat16),
}


def ref_table(name):
    c, s = pristine[name]
    return torch.cat((c.reshape(-1, 128)[:, ::2], s.reshape(-1, 128)[:, ::2]), dim=-1).float()


def request(cand):
    """One request's worth of the RoPE table work through the tree's code."""
    if cand:
        prepared = {}
        for name, (c, s) in streams.items():
            prepared[f"_flux3_{name}_cos"], prepared[f"_flux3_{name}_sin"] = c, s
            table = P.rope_cos_sin_table(c, s)
            prepared[f"_flux3_{name}_cos_sin"] = table.float().contiguous()
        fi = tf._block_freqs(prepared, "_flux3_image_")
        fj = tf._block_freqs(prepared, "_flux3_joint_")
        assert len(fi) == 3 and len(fj) == 3
    else:
        fi, fj = streams["image"], streams["joint"]
    for _ in range(STEPS):
        for _ in range(4):
            fwd(xs["image"], fi)
        for _ in range(24):
            fwd(xs["joint"], fj)


def inputs_unchanged():
    return all(
        torch.equal(streams[k][0], pristine[k][0]) and torch.equal(streams[k][1], pristine[k][1])
        for k in streams
    )


def time_mode():
    cand = HAS_CAND
    log(f"arm={'cand' if cand else 'base'}; warmup")
    for _ in range(WARM):
        request(cand)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        request(cand)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        request(cand)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for i in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    med = times[len(times) // 2]
    p10, p90 = times[int(0.1 * REPS)], times[int(0.9 * REPS)]
    # Eager (no graph) per-request wall for information only (host launch included).
    eager = []
    for _ in range(10):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        request(cand)
        b.record()
        b.synchronize()
        eager.append(a.elapsed_time(b))
    eager.sort()
    log(
        json.dumps(
            {
                "arm": "cand" if cand else "base",
                "graph_ms_median": med,
                "p10": p10,
                "p90": p90,
                "eager_ms_median": eager[len(eager) // 2],
                "inputs_unchanged": inputs_unchanged(),
            }
        )
    )
    print(
        json.dumps(
            {
                "batch": {"1": {"kernel_ms": med, "noise_floor": (p90 - p10) / med}},
                "output": {"class": "exact"},
                "unit": "one request: 448 fused-QKV RoPE table uses (+3 per-request builds on the branch)",
            }
        )
    )


def check_mode():
    assert HAS_CAND, "check runs in the branch tree"
    fails = []
    refs = {"image": ref_table("image"), "joint": ref_table("joint")}
    report = {}
    for cand in (True, False):
        label = "branch_3tuple" if cand else "fallback_2tuple"
        ok_ref, ok_it0, n = True, True, 0
        first = None
        for it in range(WARM + ITERS):
            assert inputs_unchanged(), "inputs changed"
            SEEN.clear()
            request(cand)
            torch.cuda.synchronize()
            seq = list(SEEN)
            n += len(seq)
            assert len(seq) == STEPS * 28, len(seq)
            for j, t in enumerate(seq):
                ref = refs["image"] if j % 28 < 4 else refs["joint"]
                if not torch.equal(t, ref):
                    ok_ref = False
            if first is None:
                first = seq
            elif not all(torch.equal(a, b) for a, b in zip(first, seq)):
                ok_it0 = False
        report[label] = {"calls": n, "bitwise_vs_reference": ok_ref, "bitwise_vs_iter0": ok_it0}
        if not (ok_ref and ok_it0):
            fails.append(label)
        log(label, json.dumps(report[label]))
    # Graph capture + replays on the branch path: the sink's clones are captured, so after
    # each replay they hold the tables that replay passed to the op.
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        request(True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    SEEN.clear()
    with torch.cuda.graph(graph):
        request(True)
    captured = list(SEEN)
    graph_ok = len(captured) == STEPS * 28
    for _ in range(3):
        for t in captured:
            t.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for j, t in enumerate(captured):
            ref = refs["image"] if j % 28 < 4 else refs["joint"]
            graph_ok = graph_ok and torch.equal(t, ref)
    graph_ok = graph_ok and inputs_unchanged()
    report["graph_replays_bitwise_vs_reference"] = graph_ok
    if not graph_ok:
        fails.append("graph")
    passed = not fails
    print(
        json.dumps(
            {
                "passed": passed,
                "failures": fails,
                "compared_against": "eager torch.cat((cos.reshape(-1,128)[:, ::2], "
                "sin.reshape(-1,128)[:, ::2]), -1).float() of the pristine cos/sin (main's "
                "qkv_forward expression), per call, bitwise; and iteration 0",
                "report": report,
            }
        )
    )


if MODE == "check":
    check_mode()
else:
    time_mode()
