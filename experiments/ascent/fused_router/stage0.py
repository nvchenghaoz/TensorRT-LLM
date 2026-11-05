"""Stage 0 for idea 67 (decode fused router): graph-chain cost of the DSv4.1 decode router row.

One GPU, EP4 rank 0 shapes: x [M,5120] bf16, 40 distinct router weights [384,5120] bf16 + fp32 bias,
96 local experts (slot_start 0), the real Sm12xW4a8DecodeState.run (FlashInfer SM12x chain:
route_assign_decode -> q0 direct -> fc1_act_q1 -> fc2_finalize), synthetic MXFP4 expert weights
(one expert bank shared by all 40 layers; each layer routes different tokens).

40-layer CUDA graphs, replayed interleaved, read-only 512 MB flush before each replay:
  A   : router -> gate_forward -> run
  B   : run fed precomputed ids/weights (same routing as A)
  C   : router -> gate_forward -> fc1 -> fc2 fed a precomputed route workspace (no route/q0)
  A1/B1 : A/B with a K1 stand-in before every layer (PDL, triggers at start, spin c1)
  A4/B4 : same with the c4 spin
Row cost = (A - B)/40; route+q0 share = (A - C)/40; P1 row cost = (A1 - B1)/40.

Arm g (M=1,4): R = K1 -> router, RG = K1 -> router -> gate_forward (main body / legacy body via
TRTLLM_GATE_FORWARD_LEGACY_BODY, read per launch at capture), each with and without a concurrent
188-CTA aux kernel on a second stream. gate us/call = (RG - R)/40.
"""
import json
import os
import sys
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

HIDDEN, EXPERTS, LAYERS, TOPK = 5120, 384, 40, 6
E_LOCAL, INTER = 96, 2304
SEED = 4242
WARMUP, ITERS = 5, 30
SPIN_C1_NS, SPIN_C4_NS = 4900, 3300  # K1 post-trigger duration: router starts ~0.6 us after K1 trigger
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


@triton.jit
def _gtime():
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,~{memory}", [], dtype=tl.int64,
                                     is_pure=False, pack=1)


@triton.jit
def _gdc_wait():
    tl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r,~{memory}", [], dtype=tl.int32,
                              is_pure=False, pack=1)


@triton.jit
def _gdc_trigger():
    tl.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r,~{memory}", [],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _k1_standin_kernel(src, dst, SPIN_NS: tl.constexpr, HID: tl.constexpr, BLOCK: tl.constexpr):
    # Like _mhc_lagged_layer_input_kernel: wait, trigger at once, ~4 us of work, then write x.
    _gdc_wait()
    _gdc_trigger()
    pid = tl.program_id(0)
    t0 = _gtime()
    t = t0
    while t - t0 < SPIN_NS:
        t = _gtime()
    for k in tl.static_range(0, HID, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        v = tl.load(src + pid * HID + offs)
        tl.store(dst + pid * HID + offs, v)


def k1_standin(src, dst, spin_ns):
    m = src.shape[0]
    _k1_standin_kernel[(m,)](src, dst, SPIN_NS=spin_ns, HID=HIDDEN, BLOCK=1024, num_warps=4, launch_pdl=True)


@triton.jit
def _aux_kernel(buf, out, NBUF: tl.constexpr, DUR_NS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    t0 = _gtime()
    t = t0
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    i = 0
    while t - t0 < DUR_NS:
        offs = ((pid + i * 188) * BLOCK) % NBUF + tl.arange(0, BLOCK)
        acc += tl.load(buf + offs)
        i += 1
        t = _gtime()
    tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), acc)


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as dec

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w4a8_fi_chain"))
    from common import make_weights, trtllm_layout

    dev = "cuda"
    log("device", torch.cuda.get_device_name(), "pdl env", os.environ.get("TRTLLM_ENABLE_PDL"))
    ok, why = dec.fi_decode_available()
    assert ok, why
    w1, s1, w2, s2 = make_weights(seed=0)
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    del w1, w2
    state = dec.Sm12xW4a8DecodeState(w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v,
                                     w2_weight_scale=s2s, num_local_experts=E_LOCAL, slot_start=0,
                                     swiglu_limit=10.0, enable_pdl=True)
    assert dec._route_fuses_prologue(), "route without expert_offset/zero_buf"
    route, make_ws, fc1, out_sf_shape, fc2, fi_act = dec._load_ops()

    gen = torch.Generator(device=dev).manual_seed(SEED)
    rws = [(torch.randn((EXPERTS, HIDDEN), generator=gen, device=dev) * HIDDEN ** -0.5).to(torch.bfloat16)
           for _ in range(LAYERS)]
    biases = [(torch.randn((EXPERTS,), generator=gen, device=dev) * 0.05).float() for _ in range(LAYERS)]
    empty_i = torch.empty(0, dtype=torch.int32, device=dev)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev).fill_(1.0)
    aux_buf = torch.ones(64 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    aux_out = torch.empty(188 * 1024, dtype=torch.float32, device=dev)
    side = torch.cuda.Stream()

    def router(x, w):
        return torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)

    def gate(logits, bias):
        m = logits.shape[0]
        ow = torch.empty(m, TOPK, dtype=torch.float32, device=dev)
        oi = torch.empty(m, TOPK, dtype=torch.int32, device=dev)
        torch.ops.trtllm.gate_forward(logits, bias, empty_i, empty_i, ow, oi, TOPK, 1.5, False, None, None)
        return oi, ow

    def run_from_ws(x, pre):
        # Sm12xW4a8DecodeState.run minus route(): fc1 + fc2 fed a precomputed route workspace.
        offsets, token_map, route_weights, a_q, a_sf, q1, sf1, out = pre
        fc1(a_q, a_sf, state.w1, state.w1_sf, offsets, tune=False, enable_pdl=True, out_q=q1, out_sf=sf1,
            activation=fi_act.Swiglu, swiglu_limit=10.0)
        fc2(q1, sf1, state.w2, state.w2_sf, offsets, token_map, route_weights, x.shape[0], tune=False,
            enable_pdl=True, out=out)
        return out

    def capture(fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            fn()
            fn()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        return g

    def time_graphs(graphs, aux_ns=0, warmup=WARMUP, iters=ITERS):
        ts = {k: [] for k in graphs}
        for it in range(warmup + iters):
            for name, g in graphs.items():
                flush.sum()
                if aux_ns:
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        _aux_kernel[(188,)](aux_buf, aux_out, NBUF=aux_buf.numel(), DUR_NS=aux_ns, BLOCK=1024,
                                            num_warps=8)
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                torch.cuda.synchronize()
                if it >= warmup:
                    ts[name].append(a.elapsed_time(b) * 1000.0)
        res = {}
        for k, v in ts.items():
            v.sort()
            med = v[len(v) // 2]
            res[k] = (med, (v[int(0.9 * (len(v) - 1))] - v[int(0.1 * (len(v) - 1))]) / med)
        return res

    out_all = {"stage0": {}, "arm_g": {}}
    for m in (1, 2, 4, 8):
        srcs = [torch.randn((m, HIDDEN), generator=gen, device=dev).to(torch.bfloat16) for _ in range(LAYERS)]
        xs = [s.clone() for s in srcs]
        pre_ids, pre_w, pre_ws = [], [], []
        for li in range(LAYERS):
            oi, ow = gate(router(xs[li], rws[li]), biases[li])
            pre_ids.append(oi.clone())
            pre_w.append(ow.clone())
            # per-layer precomputed route workspace and buffers (what run() would hand fc1/fc2)
            ws = make_ws(xs[li], oi, E_LOCAL)
            pairs = oi.numel()
            sf1_shape = out_sf_shape(pairs, INTER, E_LOCAL)
            q1 = torch.zeros(pairs, INTER, dtype=torch.float8_e4m3fn, device=dev)
            sf1 = torch.zeros(sf1_shape, dtype=torch.int32, device=dev)
            outb = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=dev)
            offs, tmap, rw, a_q, a_sf = route(xs[li], oi, ow, E_LOCAL, workspace=ws, enable_pdl=True,
                                              expert_offset=0, zero_buf=None)
            pre_ws.append((offs, tmap, rw, a_q, a_sf, q1, sf1, outb))
        torch.cuda.synchronize()
        nloc = sum(int(((t >= 0) & (t < E_LOCAL)).sum()) for t in pre_ids) / LAYERS
        log(f"M={m}: mean local pairs/layer {nloc:.2f}")

        def mk(kind, spin=0):
            def fn():
                for li in range(LAYERS):
                    if spin:
                        k1_standin(srcs[li], xs[li], spin)
                    if kind == "A":
                        oi, ow = gate(router(xs[li], rws[li]), biases[li])
                        state.run(xs[li], oi, ow)
                    elif kind == "B":
                        state.run(xs[li], pre_ids[li], pre_w[li])
                    elif kind == "C":
                        gate(router(xs[li], rws[li]), biases[li])
                        run_from_ws(xs[li], pre_ws[li])
            return fn

        graphs = {"A": capture(mk("A")), "B": capture(mk("B")), "C": capture(mk("C")),
                  "A1": capture(mk("A", SPIN_C1_NS)), "B1": capture(mk("B", SPIN_C1_NS)),
                  "A4": capture(mk("A", SPIN_C4_NS)), "B4": capture(mk("B", SPIN_C4_NS))}
        r = time_graphs(graphs)
        del graphs
        L = LAYERS
        res = {k + "_us_layer": v[0] / L for k, v in r.items()}
        res["nf"] = max(v[1] for v in r.values())
        res["row_cost_P0_us"] = (r["A"][0] - r["B"][0]) / L
        res["route_q0_share_us"] = (r["A"][0] - r["C"][0]) / L
        res["row_cost_P1c1_us"] = (r["A1"][0] - r["B1"][0]) / L
        res["row_cost_P1c4_us"] = (r["A4"][0] - r["B4"][0]) / L
        log(f"STAGE0 M={m} kernels[router_gemm_mma_kernel,gate_forward_kernel,_route_assign_decode_kernel,"
            f"_mxfp8_q0_route_direct_kernel,fc1_act_q1,fc2_finalize] "
            + json.dumps({k: round(v, 3) for k, v in res.items()}))
        out_all["stage0"][str(m)] = res

        if m in (1, 4):
            def mkg(with_gate, spin=SPIN_C1_NS):
                def fn():
                    for li in range(LAYERS):
                        k1_standin(srcs[li], xs[li], spin)
                        lg = router(xs[li], rws[li])
                        if with_gate:
                            gate(lg, biases[li])
                return fn

            gg = {}
            gg["R"] = capture(mkg(False))
            os.environ.pop("TRTLLM_GATE_FORWARD_LEGACY_BODY", None)
            gg["RG_main"] = capture(mkg(True))
            os.environ["TRTLLM_GATE_FORWARD_LEGACY_BODY"] = "1"
            gg["RG_legacy"] = capture(mkg(True))
            os.environ.pop("TRTLLM_GATE_FORWARD_LEGACY_BODY", None)
            for aux_ns in (0, 1_500_000):
                rg = time_graphs(gg, aux_ns=aux_ns)
                gres = {
                    "R_us_layer": rg["R"][0] / L,
                    "gate_main_us_call": (rg["RG_main"][0] - rg["R"][0]) / L,
                    "gate_legacy_us_call": (rg["RG_legacy"][0] - rg["R"][0]) / L,
                    "nf": max(v[1] for v in rg.values()),
                }
                tag = "aux188" if aux_ns else "noaux"
                log(f"ARM_G M={m} {tag} kernels[_k1_standin_kernel,router_gemm_mma_kernel,gate_forward_kernel"
                    f"<384,6,false,false|true>] " + json.dumps({k: round(v, 3) for k, v in gres.items()}))
                out_all["arm_g"][f"{m}_{tag}"] = gres
            del gg
    print(json.dumps(out_all))


if __name__ == "__main__":
    with torch.no_grad():
        main()
