"""Idea 161 STEP 0: side-stream NVFP4 packed-SwiGLU gate_up beside attention (FLUX.3 klein, SM120).

Synthetic tensors at the served shapes (H=3072, 24 heads x 128, FF=9216; single blocks S=4176,
image mode blocks S=4096), no model load. One "block front" = the kernels from QKV to the tail
pair, as the NVFP4 tail-pair block runs them:
  QKV op (trtllm::flux3_primsts_nvfp4_qkv_rope: CUTLASS q GEMM || k/v GEMMs, qknorm_rope pass)
  -> attention (K1 amax, K2 quantize, routed fmha_v2_flux3)
  -> gate_up (CUTLASS packed SwiGLU -> NVFP4)  [OFF: sequential after attention]
  -> trtllm::flux3_sm120_nvfp4_out_pair (down || attn_out) -> residual add.
Arms: off (main's ops), m1/m2/m3 (trtllm::flux3_sm120_fp8_attention_side_gate_up mode 1/2/3).
IDEA161_PATH=direct (default, main after landing 54): QKV op = trtllm::flux3_primsts_nvfp4_qkv_rope_fp8
(QKV GEMMs + fused norm/RoPE/E4M3), attention = trtllm::visual_gen_fp8_attention_sm120_prequant;
the arms use trtllm::flux3_sm120_fp8_prequant_attention_side_gate_up (gate_up forked right before
FMHA; m1 == m2 there, m3 = dedicated stream).

  table : per S, graphs of K block fronts per arm, two graphs per arm (A and A' = its own A/A),
          ROUNDS interleaved rounds behind a spin, us/block median; weighted 24:4.
  pair <arm>: evidence unit = one graph of 24 x S4176 + 4 x S4096 block fronts (one denoise
          step's NVFP4 blocks), SAMPLES replays after a soak; prints the evidence JSON. arm 'off'
          uses only main's ops, so the baseline tree runs it.
  check : untimed bitwise check vs main's sequential ops (eager and graph replay), same warmup /
          iteration / reset points as timing, inputs unchanged, call-0 consistency.
"""

import json
import math
import os
import sys
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

H, HEADS, D, FF = 3072, 24, 128, 9216
K_BLOCKS = int(os.environ.get("IDEA161_K", "10"))
ROUNDS = 48
SPIN = 6_000_000
PAIR_SAMPLES = 200
PAIR_SOAK_S = 30.0
PAIR_REPLAYS = 3  # graph replays per timed sample (sample = per-replay mean)
# direct: main's default after landing 54 (QKV -> E4M3 op + prequant FMHA); bf16: the BF16 QKV
# planes + fp8_sm12x K1/K2 + FMHA path (TRTLLM_FLUX3_QKV_FP8_DIRECT=0).
PATH = os.environ.get("IDEA161_PATH", "direct")


def log(*a):
    print(*a, flush=True)


def _gs(t):
    return float(t.abs().amax().float() / (448.0 * 6.0))


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def make_case(S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x = torch.randn(S, H, device=dev, dtype=torch.bfloat16, generator=g)
    w_qkv = torch.randn(3 * H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_up = torch.randn(2 * FF, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    s_x, s_qkv, s_up, s_down, s_out = map(_gs, (x, w_qkv, w_up, w_down, w_out))
    s_attn, s_down_in = 0.05, 0.05
    a_x, sf_x = fp.nvfp4_quantize(x, _inv(s_x))
    q_qkv, sf_qkv = fp.nvfp4_quantize(w_qkv, _inv(s_qkv))
    q_up, sf_up = fp.nvfp4_quantize(w_up, _inv(s_up))
    q_down, sf_down = fp.nvfp4_quantize(w_down, _inv(s_down))
    q_out, sf_out = fp.nvfp4_quantize(w_out, _inv(s_out))
    qkv_ad = PreparedNVFP4(
        q_qkv, sf_qkv, s_x, s_qkv,
        qkv_scale=torch.tensor([s_x * s_qkv] * 3, device=dev, dtype=torch.float32),
    )
    up_ad = PreparedNVFP4(q_up, sf_up, s_x, s_up, swiglu=True)
    down_ad = PreparedNVFP4(q_down, sf_down, s_down_in, s_down)
    out_ad = PreparedNVFP4(q_out, sf_out, s_attn, s_out)
    theta = torch.rand(S, 64, device=dev, generator=g) * 6.28
    cos_sin = torch.cat((theta.cos(), theta.sin()), dim=-1).float().contiguous()
    q_norm = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    k_norm = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    residual = torch.randn(S, H, device=dev, dtype=torch.bfloat16, generator=g)
    return {
        "S": S, "x": x, "a_x": a_x, "sf_x": sf_x, "qkv": qkv_ad, "up": up_ad, "down": down_ad,
        "out": out_ad, "cos_sin": cos_sin, "q_norm": q_norm, "k_norm": k_norm,
        "residual": residual,
    }


def inputs(c):
    ts = [c[k] for k in ("x", "a_x", "sf_x", "cos_sin", "q_norm", "k_norm", "residual")]
    for k in ("qkv", "up", "down", "out"):
        ts += [c[k].weight, c[k].scale]
    return ts


def block_front(c, arm):
    if PATH == "direct":
        return block_front_direct(c, arm)
    return block_front_bf16(c, arm)


def block_front_direct(c, arm):
    """Main after landing 54: NVFP4 QKV straight to E4M3 (trtllm::flux3_primsts_nvfp4_qkv_rope_fp8)
    then the prequant fmha_v2 (trtllm::visual_gen_fp8_attention_sm120_prequant)."""
    S = c["S"]
    sm_scale = 1.0 / math.sqrt(D)
    qkv8, scales, _ = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope_fp8(
        c["a_x"], c["qkv"].weight, c["sf_x"], c["qkv"].scale, c["q_norm"], c["k_norm"],
        c["cos_sin"], c["qkv"].qkv_scale, c["x"], c["qkv"].weight, sm_scale,
    )
    up, down = c["up"], c["down"]
    if arm == "off":
        attn = torch.ops.trtllm.visual_gen_fp8_attention_sm120_prequant(qkv8, scales, 1, HEADS)
        packed, sf = torch.ops.trtllm.flux3_nvfp4_swiglu_prepared(
            c["a_x"], up.weight, c["sf_x"], up.scale, down.inverse_scale, c["x"],
            c["qkv"].weight,
        )
    else:
        mode = {"m1": 1, "m2": 2, "m3": 3}[arm]
        attn, packed, sf = torch.ops.trtllm.flux3_sm120_fp8_prequant_attention_side_gate_up(
            qkv8, scales, 1, HEADS, c["a_x"], up.weight, c["sf_x"], up.scale, down.inverse_scale,
            c["x"], c["qkv"].weight, mode,
        )
    attn_y, mlp_y = torch.ops.trtllm.flux3_sm120_nvfp4_out_pair(
        packed, down.weight, sf, down.scale, attn.reshape(S, H), c["out"].weight, c["out"].scale,
    )
    return c["residual"] + (attn_y + mlp_y), attn, packed, sf


def block_front_bf16(c, arm):
    S = c["S"]
    qkv = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope(
        c["a_x"], c["qkv"].weight, c["sf_x"], c["qkv"].scale,
        c["qkv"].input_global_scale, c["qkv"].weight_global_scale,
        c["q_norm"], c["k_norm"], c["cos_sin"], None, c["qkv"].qkv_scale,
    )
    planes = qkv.view(3, S, H)
    q, k, v = (planes[i].view(1, S, HEADS, D) for i in range(3))
    up, down = c["up"], c["down"]
    if arm == "off":
        attn = torch.ops.trtllm.visual_gen_fp8_attention_sm120(q, k, v, 1.0 / math.sqrt(D), False)
        packed, sf = torch.ops.trtllm.flux3_nvfp4_swiglu_prepared(
            c["a_x"], up.weight, c["sf_x"], up.scale, down.inverse_scale, c["x"],
            c["qkv"].weight,
        )
    else:
        mode = {"m1": 1, "m2": 2, "m3": 3}[arm]
        attn, packed, sf = torch.ops.trtllm.flux3_sm120_fp8_attention_side_gate_up(
            q, k, v, 1.0 / math.sqrt(D), False, c["a_x"], up.weight, c["sf_x"], up.scale,
            down.inverse_scale, c["x"], c["qkv"].weight, mode,
        )
    attn_y, mlp_y = torch.ops.trtllm.flux3_sm120_nvfp4_out_pair(
        packed, down.weight, sf, down.scale, attn.reshape(S, H), c["out"].weight, c["out"].scale,
    )
    return c["residual"] + (attn_y + mlp_y), attn, packed, sf


_CAPTURE_STREAM = []


def capture(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # One capture stream for every graph: torch.cuda.graph() otherwise takes a fresh stream from
    # PyTorch's 32-per-priority pool per capture, and after enough captures it aliases a side
    # stream already in use (pitfall 726); jobs 1318/1330 failed with capture invalidated on the
    # direct path's many-graph table while every single capture passed (job 1341).
    if not _CAPTURE_STREAM:
        _CAPTURE_STREAM.append(torch.cuda.Stream())
    with torch.cuda.graph(g, stream=_CAPTURE_STREAM[0], capture_error_mode="thread_local"):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def stat(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med


def time_graph(g):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(SPIN)
    a.record()
    g.replay()
    b.record()
    return a, b


def table(arms):
    log(f"current stream priority {torch.cuda.current_stream().priority}; tail stream priority "
        f"{fp._sm120_tail_pair_stream(torch.cuda.current_device()).priority}; "
        f"priority range {torch.cuda.Stream.priority_range() if hasattr(torch.cuda.Stream, 'priority_range') else 'n/a'}")
    res = {}
    for S, seed in ((4176, 1), (4096, 2)):
        c = make_case(S, seed)
        graphs = {}
        for arm in arms:
            for rep in ("A", "B"):
                graphs[f"{arm}{rep}"], _ = capture(
                    lambda: [block_front(c, arm)[0] for _ in range(K_BLOCKS)]
                )
        log(f"S={S}: captured {len(graphs)} graphs; soak")
        t0 = time.time()
        while time.time() - t0 < 5.0:
            for g in graphs.values():
                g.replay()
            torch.cuda.synchronize()
        times = {k: [] for k in graphs}
        keys = list(graphs)
        for r in range(ROUNDS):
            order = keys if r % 2 == 0 else keys[::-1]
            evs = []
            for k in order:
                torch.cuda.synchronize()
                evs.append((k, *time_graph(graphs[k])))
            torch.cuda.synchronize()
            for k, a, b in evs:
                times[k].append(a.elapsed_time(b) * 1e3 / K_BLOCKS)
            if r % 12 == 0:
                log(f"  S={S} round {r}: " + " ".join(f"{k}={times[k][-1]:.1f}" for k in keys))
        row = {}
        for k in keys:
            med, noise = stat(times[k])
            row[k] = {"us_per_block": round(med, 2), "noise": round(noise, 4)}
        for arm in arms:
            a, b = row[f"{arm}A"]["us_per_block"], row[f"{arm}B"]["us_per_block"]
            row[arm] = {"mean": round((a + b) / 2, 2), "aa_spread": round(abs(a - b), 2)}
        for arm in arms:
            row[arm]["delta_vs_off"] = round(row[arm]["mean"] - row["off"]["mean"], 2)
        res[S] = row
        log(f"S={S}: " + json.dumps({a: row[a] for a in arms}))
        del graphs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    weighted = {
        arm: round((24 * res[4176][arm]["delta_vs_off"] + 4 * res[4096][arm]["delta_vs_off"]) / 28, 2)
        for arm in arms
    }
    print(json.dumps({"table": {str(k): v for k, v in res.items()}, "weighted_delta_us_per_block":
                      weighted, "rounds": ROUNDS, "blocks_per_graph": K_BLOCKS,
                      "counters": fp.SIDE_GATE_UP_COUNTERS if hasattr(fp, "SIDE_GATE_UP_COUNTERS")
                      else None}))


def pair(arm):
    cases = [(make_case(4176, 1), 24), (make_case(4096, 2), 4)]
    pristine = [[t.clone() for t in inputs(c)] for c, _ in cases]

    def step():
        outs = []
        for c, n in cases:
            for _ in range(n):
                outs.append(block_front(c, arm)[0])
        return outs[-1]

    g, _ = capture(step)
    log(f"pair arm={arm}: captured; soak {PAIR_SOAK_S}s")
    t0 = time.time()
    while time.time() - t0 < PAIR_SOAK_S:
        g.replay()
        torch.cuda.synchronize()
    for (c, _), p in zip(cases, pristine):
        assert all(torch.equal(a, b) for a, b in zip(p, inputs(c)))
    times = []
    for i in range(PAIR_SAMPLES):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(SPIN)
        a.record()
        for _ in range(PAIR_REPLAYS):
            g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / PAIR_REPLAYS)
        if i % 40 == 0:
            log(f"  sample {i}: {times[-1]:.3f} ms")
    med, noise = stat(times)
    log(f"step block fronts: {med:.4f} ms noise {noise:.4f}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
        "unit": "one graph replay = 24 x S4176 + 4 x S4096 NVFP4 block fronts (QKV op, "
                "attention, gate_up, down||attn_out pair, residual add)",
        "arm": arm, "samples": PAIR_SAMPLES, "replays_per_sample": PAIR_REPLAYS,
        "soak_s": PAIR_SOAK_S, "output": {"class": "exact"},
    }))


def check(arms):
    results, passed = {}, True
    for S, seed in ((4176, 11), (4096, 12)):
        c = make_case(S, seed)
        pristine = [t.clone() for t in inputs(c)]
        ref = [t.clone() for t in block_front(c, "off")]
        torch.cuda.synchronize()
        for arm in arms:
            for p, t in zip(pristine, inputs(c)):
                t.copy_(p)
            for _ in range(2):
                block_front(c, arm)
            n_ref = n_first = 0
            first = None
            iters = 20
            for i in range(iters):
                out = block_front(c, arm)
                if first is None:
                    first = [t.clone() for t in out]
                n_ref += all(torch.equal(a, b) for a, b in zip(out, ref))
                n_first += all(torch.equal(a, b) for a, b in zip(out, first))
            inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
            # graph capture + replays
            g, gout = capture(lambda: block_front(c, arm))
            g_ok = 0
            for _ in range(5):
                for t in gout:
                    t.zero_()
                g.replay()
                torch.cuda.synchronize()
                g_ok += all(torch.equal(a, b) for a, b in zip(gout, ref))
            del g
            inputs_ok &= all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
            ok = n_ref == iters and n_first == iters and g_ok == 5 and inputs_ok
            passed &= ok
            results[f"S{S}_{arm}"] = {
                "eager_bitwise_vs_off": f"{n_ref}/{iters}", "eager_vs_call0": f"{n_first}/{iters}",
                "graph_replay_bitwise_vs_off": f"{g_ok}/5", "inputs_unchanged": inputs_ok, "ok": ok,
            }
            log(f"S={S} {arm}: {json.dumps(results[f'S{S}_{arm}'])}")
    print(json.dumps({
        "passed": bool(passed),
        "reference": "main's sequential ops (attention: visual_gen_fp8_attention_sm120_prequant "
                     "on the direct path / visual_gen_fp8_attention_sm120 on the bf16 path; then "
                     "flux3_nvfp4_swiglu_prepared) on pristine inputs, bitwise on block output, "
                     "attention output, packed gate_up bytes and scales",
        "path": PATH,
        "results": results, "counters": fp.SIDE_GATE_UP_COUNTERS,
    }))


def capone(arm, k, warm):
    """Capture probe: one graph of k block fronts of `arm` at S=4176 after `warm` eager calls."""
    c = make_case(4176, 1)
    for _ in range(warm):
        block_front(c, arm)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            [block_front(c, arm)[0] for _ in range(k)]
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        print(json.dumps({"arm": arm, "k": k, "warm": warm, "ok": True}), flush=True)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"arm": arm, "k": k, "warm": warm, "ok": False,
                          "err": f"{type(e).__name__}: {str(e)[:200]}"}), flush=True)


def capseq(arms, k):
    """Capture probe: graphs of k block fronts for each arm in `arms`, in order, one process."""
    c = make_case(4176, 1)
    graphs = []
    for i, arm in enumerate(arms):
        try:
            g, _ = capture(lambda: [block_front(c, arm)[0] for _ in range(k)])
            graphs.append(g)
            g.replay()
            torch.cuda.synchronize()
            print(json.dumps({"seq": arms, "i": i, "arm": arm, "ok": True}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"seq": arms, "i": i, "arm": arm, "ok": False,
                              "err": f"{type(e).__name__}: {str(e)[:160]}"}), flush=True)
            return


def capdbg():
    import subprocess

    res = []
    for seq in ("off,off", "off,m3", "m3,off", "m3,m3", "off,off,off,off"):
        p = subprocess.run([sys.executable, __file__, "capseq", seq, "2"],
                           capture_output=True, text=True)
        for x in p.stdout.splitlines():
            if x.startswith("{"):
                res.append(json.loads(x))
                log(x)
        if p.returncode:
            res.append({"seq": seq, "rc": p.returncode, "tail": p.stderr[-400:]})
    for arm, k, warm in (("off", 1, 2),):
        p = subprocess.run([sys.executable, __file__, "capone", arm, str(k), str(warm)],
                           capture_output=True, text=True)
        line = [x for x in p.stdout.splitlines() if x.startswith("{")]
        res.append(json.loads(line[-1]) if line else {"arm": arm, "k": k, "warm": warm,
                                                      "ok": None, "rc": p.returncode,
                                                      "tail": p.stderr[-600:]})
        log(json.dumps(res[-1]))
    print(json.dumps({"capdbg": res}))


if __name__ == "__main__":
    torch.manual_seed(0)
    cmd = sys.argv[1]
    if cmd == "capone":
        capone(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "capseq":
        capseq(sys.argv[2].split(","), int(sys.argv[3]))
    elif cmd == "capdbg":
        capdbg()
    elif cmd == "table":
        table(sys.argv[2].split(",") if len(sys.argv) > 2 else ["off", "m1", "m2", "m3"])
    elif cmd == "pair":
        pair(sys.argv[2])
    else:
        check(sys.argv[2].split(",") if len(sys.argv) > 2 else ["m1", "m2", "m3"])
