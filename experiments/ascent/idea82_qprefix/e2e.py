"""Idea 82: in-process pipeline check of the block-27 Q-prefix FMHA (served config).

One process, the pipeline as served (engine.json overlay, torch.compile + autotune, CUDA graphs,
2048^2 t2i + i2i warmup), loaded with the branch default (TRTLLM_FLUX3_LAST_BLOCK_QPREFIX on):
  ON    graph-on : t2i A, i2i X, t2i B, i2i X(rep), t2i A(rep), i2i X(rep2)  -> 0 compile/capture
  ON    graph-off: t2i A, i2i X with the FMHA call hooks (census)
  flip primsts.FLUX3_LAST_BLOCK_QPREFIX = False (landing 18's route; Dynamo guards on it),
        runner.clear()
  QP0   graph-on : t2i A, i2i X ; graph-off: t2i A, i2i X with the hooks
  flip transformer_flux3.FLUX3_LAST_BLOCK_ROWS = False (pre-L18 full last block), runner.clear()
  ROWS0 graph-on : t2i A, i2i X
Hooks (graph-off requests): every fmha_v2_flux3_d128_nvfp4_out / _q_rows call records seq_len,
q_rows and a GPU fingerprint of fp4 rows [0, 16384) + the leading 128 SF row blocks.
Checks: ON == QP0 == ROWS0 (final latent + uint8 image sha256) per request in both graph modes;
repeats equal; graph == eager; ON's nvfp4_out calls == QP0's call by call (seq_len, prefix
fingerprint); ON has exactly 16 q_rows=16384 calls per request, each the last S_full call of its
step (block 27), QP0 has none.
Usage: python e2e.py <checkpoint>
"""

import hashlib
import json
import os
import subprocess
import sys
import time

import torch
from torch._dynamo.utils import counters

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CKPT = sys.argv[1]
fails = []
QR = 16384


def log(*a):
    print(*a, flush=True)


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


def store_png(sha):
    store = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
             "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
    for root, _, files in os.walk(store):
        for f in files:
            if sha in f:
                return os.path.join(root, f)
    for root, _, files in os.walk(store):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() == sha:
                    return p
    raise FileNotFoundError(sha)


log("git", subprocess.run(["git", "-C", WT, "log", "--oneline", "-3"], capture_output=True,
                          text=True).stdout)

from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as fx  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import primsts  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    _load_image_pixels,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

HOOK = {"on": False}
CALLS = []
_W = {}


def fingerprint(t):
    """Order-sensitive GPU fingerprint of a uint8 tensor (int64 weighted sum, wraps)."""
    x = t.contiguous().view(-1)
    pad = (-x.numel()) % 4
    if pad:
        x = torch.cat([x, x.new_zeros(pad)])
    x = x.view(torch.int32).to(torch.int64)
    n = x.numel()
    w = _W.get(n)
    if w is None:
        g = torch.Generator(device=x.device).manual_seed(n)
        w = torch.randint(1, 2**62, (n,), dtype=torch.int64, device=x.device, generator=g) | 1
        _W[n] = w
    return int((x * w).sum().item()) ^ int(((x + 0x9E37) * (w >> 7)).sum().item())


_nfo = fx.fmha_v2_flux3_d128_nvfp4_out
_nfo_q = fx.fmha_v2_flux3_d128_nvfp4_out_q_rows


def _rec(fp4, sf, seq_len, q_rows):
    if not HOOK["on"]:
        return
    cols = (fp4.shape[-1] * 2 // 16 + 3) // 4 * 4
    rows = min(QR, seq_len)
    CALLS.append({"S": int(seq_len), "q_rows": int(q_rows),
                  "fp4": fingerprint(fp4.reshape(-1, fp4.shape[-1])[:rows]),
                  "sf": fingerprint(sf[: (rows + 127) // 128 * 128 * cols])})


def h_nfo(q, k, v, fp4, sf, gs, heads, hd, seq_len, **kw):
    r = _nfo(q, k, v, fp4, sf, gs, heads, hd, seq_len, **kw)
    _rec(fp4, sf, seq_len, 0)
    return r


def h_nfo_q(q, k, v, fp4, sf, gs, heads, hd, seq_len, q_rows, **kw):
    r = _nfo_q(q, k, v, fp4, sf, gs, heads, hd, seq_len, q_rows, **kw)
    _rec(fp4, sf, seq_len, q_rows)
    return r


fx.fmha_v2_flux3_d128_nvfp4_out = h_nfo
fx.fmha_v2_flux3_d128_nvfp4_out_q_rows = h_nfo_q

inputs = {"check00": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e"}
engine = json.load(open(os.path.join(WT, "experiments", "ascent-config", "engine.json")))
engine.pop("env", None)
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
engine.setdefault("torch_compile_config",
                  {"enable": True, "enable_fullgraph": False, "enable_autotune": True})
engine.setdefault("parallel_config", {"tp_size": 1})
args = VisualGenArgs(
    model=CKPT, pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["t2i", "i2i"]},
    **engine,
)
log(f"QPREFIX={primsts.FLUX3_LAST_BLOCK_QPREFIX} ROWS={tf.FLUX3_LAST_BLOCK_ROWS} engine={engine}")
note(primsts.FLUX3_LAST_BLOCK_QPREFIX and tf.FLUX3_LAST_BLOCK_ROWS, "branch defaults are ON")
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None and runner.enabled} "
    f"graphs={len(runner.graphs) if runner else None}")
dev = torch.device("cuda:0")
refs = {}


def ref_image(key):
    if key not in refs:
        data = open(store_png(inputs[key]), "rb").read()
        assert hashlib.sha256(data).hexdigest() == inputs[key]
        refs[key] = _load_image_pixels(data, device=dev)
    return refs[key]


def sha(t):
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]


def stats():
    return {"frames": dict(counters["frames"]),
            "unique_graphs": counters["stats"].get("unique_graphs", 0),
            "graphs": len(runner.graphs) if runner else 0}


REQ = {
    "t2i_A": ("t2i", "A red fox standing in fresh snow at sunrise, photograph", 11, None),
    "t2i_B": ("t2i", "A cozy reading nook with a green armchair, a lamp and a stack of books", 22,
              None),
    "i2i_X": ("i2i", "Turn the scene into a watercolor painting", 5, "check00"),
}
records = []
outs = {}


def run(tag, name, graphs, census=False):
    mode, prompt, seed, ref = REQ[name]
    if runner is not None:
        runner.enabled = graphs
    CALLS.clear()
    HOOK["on"] = census
    before = stats()
    t1 = time.time()
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        if mode == "t2i":
            latent, ids = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048,
                                            num_inference_steps=16, seed=seed)
        else:
            latent, ids = pipe._prepare_i2i(ctx=ctx, image_cond=[ref_image(ref)], height=2048,
                                            width=2048, num_inference_steps=16, seed=seed)
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    HOOK["on"] = False
    after = stats()
    rec = {"tag": tag, "req": name, "graphs": graphs, "s": round(time.time() - t1, 2),
           "latent": sha(latent), "image": sha(img),
           "new_frames": {k: after["frames"].get(k, 0) - before["frames"].get(k, 0)
                          for k in after["frames"]},
           "new_unique_graphs": after["unique_graphs"] - before["unique_graphs"],
           "new_cuda_graphs": after["graphs"] - before["graphs"]}
    if census:
        rec["calls"] = list(CALLS)
        by = {}
        for c in CALLS:
            k = f"S{c['S']}_q{c['q_rows']}"
            by[k] = by.get(k, 0) + 1
        rec["census"] = by
    log(tag, json.dumps({k: v for k, v in rec.items() if k != "calls"}))
    records.append(rec)
    outs[tag] = rec
    return rec


def compiled_nothing(rec):
    return (not rec["new_unique_graphs"] and not rec["new_cuda_graphs"]
            and not any(rec["new_frames"].values()))


# ---- ON (branch default) ----
for tag, name in (("ON_t2i_A", "t2i_A"), ("ON_i2i_X", "i2i_X"), ("ON_t2i_B", "t2i_B"),
                  ("ON_i2i_X_rep", "i2i_X"), ("ON_t2i_A_rep", "t2i_A"),
                  ("ON_i2i_X_rep2", "i2i_X")):
    r = run(tag, name, True)
    note(compiled_nothing(r), f"{tag}: compiled or captured after warmup")
run("ON_t2i_A_eager", "t2i_A", False, census=True)
run("ON_i2i_X_eager", "i2i_X", False, census=True)
# ---- QP0 (landing 18's full-row FMHA) ----
primsts.FLUX3_LAST_BLOCK_QPREFIX = False
if runner is not None:
    runner.clear()
run("QP0_t2i_A", "t2i_A", True)
run("QP0_i2i_X", "i2i_X", True)
run("QP0_t2i_A_eager", "t2i_A", False, census=True)
run("QP0_i2i_X_eager", "i2i_X", False, census=True)
# ---- ROWS0 (pre-L18 full last block) ----
tf.FLUX3_LAST_BLOCK_ROWS = False
if runner is not None:
    runner.clear()
run("ROWS0_t2i_A", "t2i_A", True)
run("ROWS0_i2i_X", "i2i_X", True)
run("ROWS0_i2i_X_eager", "i2i_X", False, census=True)


def same(a, b, what):
    ok = outs[a]["latent"] == outs[b]["latent"] and outs[a]["image"] == outs[b]["image"]
    note(ok, what)
    return ok


comparisons = {}
for a, b_, what in (
    ("ON_t2i_A", "QP0_t2i_A", "ON==QP0 t2i graph"),
    ("ON_i2i_X", "QP0_i2i_X", "ON==QP0 i2i graph"),
    ("ON_t2i_A_eager", "QP0_t2i_A_eager", "ON==QP0 t2i eager"),
    ("ON_i2i_X_eager", "QP0_i2i_X_eager", "ON==QP0 i2i eager"),
    ("ON_t2i_A", "ROWS0_t2i_A", "ON==ROWS0 t2i graph"),
    ("ON_i2i_X", "ROWS0_i2i_X", "ON==ROWS0 i2i graph"),
    ("ON_i2i_X", "ROWS0_i2i_X_eager", "ON==ROWS0 i2i eager"),
    ("ON_t2i_A", "ON_t2i_A_rep", "ON t2i repeat"),
    ("ON_i2i_X", "ON_i2i_X_rep", "ON i2i repeat"),
    ("ON_i2i_X", "ON_i2i_X_rep2", "ON i2i repeat2"),
    ("ON_t2i_A", "ON_t2i_A_eager", "ON t2i graph==eager"),
    ("ON_i2i_X", "ON_i2i_X_eager", "ON i2i graph==eager"),
):
    comparisons[what] = same(a, b_, what)

calls = {}
for case, s_full in (("t2i_A", 16464), ("i2i_X", 32848)):
    on, off = outs[f"ON_{case}_eager"]["calls"], outs[f"QP0_{case}_eager"]["calls"]
    note(len(on) == len(off) and len(on) > 0, f"{case}: nvfp4_out calls ON {len(on)} QP0 {len(off)}")
    eq = sum(int(a["S"] == b["S"] and a["fp4"] == b["fp4"] and a["sf"] == b["sf"])
             for a, b in zip(on, off))
    note(eq == len(on), f"{case}: prefix fingerprints ON==QP0 {eq}/{len(on)}")
    qidx = [i for i, c in enumerate(on) if c["q_rows"]]
    note(len(qidx) == 16 and all(on[i]["q_rows"] == QR and on[i]["S"] == s_full for i in qidx),
         f"{case}: 16 q_rows={QR} calls at S={s_full} ({len(qidx)})")
    note(not any(c["q_rows"] for c in off), f"{case}: QP0 has q_rows calls")
    # each q_rows call is the last S_full call before the next step's first call (block 27)
    full_idx = [i for i, c in enumerate(on) if c["S"] == s_full]
    per_step = len(full_idx) // 16 if full_idx else 0
    last_of_step = [full_idx[(k + 1) * per_step - 1] for k in range(16)] if per_step else []
    note(qidx == last_of_step, f"{case}: q_rows calls are the last S_full call of each step")
    calls[case] = {"on": outs[f"ON_{case}_eager"]["census"],
                   "qp0": outs[f"QP0_{case}_eager"]["census"], "equal_calls": eq,
                   "S_full_calls_per_step": per_step,
                   "grid_x_q_rows_call": (((QR + 63) // 64) + 2) // 3,
                   "grid_x_full_call": (((s_full + 63) // 64) + 2) // 3}
calls["i2i_X_rows0"] = outs["ROWS0_i2i_X_eager"]["census"]
note(not any(c["q_rows"] for c in outs["ROWS0_i2i_X_eager"]["calls"]), "ROWS0 has q_rows calls")
print(json.dumps({
    "passed": not fails, "failures": fails, "comparisons": comparisons, "calls": calls,
    "compared_against": "TRTLLM_FLUX3_LAST_BLOCK_QPREFIX off (landing 18 route) and "
    "TRTLLM_FLUX3_LAST_BLOCK_ROWS off (pre-L18 full block) in the same process, same requests: "
    "final latent + uint8 image sha256; per nvfp4_out FMHA call prefix fingerprints",
    "requests": [{k: v for k, v in r.items() if k != "calls"} for r in records],
}))
