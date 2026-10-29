"""Idea 17 untimed correctness: large-M routes vs main's kernels, bitwise, on real block weights.

Weights: the nvfp4 checkpoint's own NVFP4 tensors (U8 weight, E4M3 [N, K/16] scales swizzled with
torch.ops.trtllm.block_scale_interleave as the loader does, weight_scale_2, input_scale) of
single_blocks.0, single_blocks.13 and content_mode_blocks.image.1 (q/k/v planes, attn_out,
mlp_out = down); image.0's BF16 q/k/v, mlp_in, mlp_out, attn_out quantized with
quantize_fp8r_weight as the FP8R prepare does. Activations: BF16 normals scaled to the layer's
input_scale range and quantized with the served quantizers (nvfp4_quantize / quantize_fp8r).

Per M in (16384, 16896, 33280) and 4176 (must stay on main's route):
  QKV3 : routed strided.gemm_qkv3 vs main's module gemm_qkv3_pingpong (main raster/swizzle).
  pair : routed lpt.run vs main's module lpt_pair (main sched mode); tile counter after calls.
  FP8R : routed flux3_fp8r_gemm.gemm (default arm) vs arm 1 and torch._scaled_mm (M 16384/33280).
Sequence (same as the timed unit's reset points): WARMUP + ITERS calls, output poisoned before
each, every call vs reference and vs call 0; inputs vs pristine copies; CUDA-graph capture + 3
replays (poisoned before each) vs reference.
Kill switch: a subprocess with TRTLLM_FLUX3_SM120_LARGE_M_TILES=0 checks the routed calls equal
main's outputs and that the large-M modules were never built or loaded.
Last stdout line: {"passed": ...}.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

WARMUP, ITERS = 2, 5
MS = [16384, 16896, 33280, 4176]
BLOCKS = ["single_blocks.0", "single_blocks.13", "content_mode_blocks.image.1"]
H = 3072


def log(*a):
    print(*a, flush=True)


def ckpt_file(ckpt):
    return Path(ckpt) / "variants" / "nvfp4" / "model.safetensors"


def load_nvfp4(h, name):
    w = h.get_tensor(name + ".weight").cuda()
    sf = h.get_tensor(name + ".weight_scale").cuda()
    try:
        swz = torch.ops.trtllm.block_scale_interleave(sf.view(torch.uint8))
    except Exception:  # noqa: BLE001
        swz = torch.ops.trtllm.block_scale_interleave(sf)
    ws2 = float(h.get_tensor(name + ".weight_scale_2").float().item())
    ins = float(h.get_tensor(name + ".input_scale").float().item())
    return w.contiguous(), swz.reshape(-1).view(torch.uint8).contiguous(), ws2, ins


def act(m, k, in_scale, g):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    x = x * (in_scale * 448.0 * 6.0 / 4.5)
    inv = torch.tensor([1.0 / in_scale], device="cuda", dtype=torch.float32)
    return fp.nvfp4_quantize(x, inv)


def block_inputs(h, blk, m, g):
    q, k, v = (load_nvfp4(h, f"{blk}.{p}_proj") for p in "qkv")
    wqkv = torch.cat([q[0], k[0], v[0]], 0).contiguous()
    sqkv = torch.cat([q[1], k[1], v[1]], 0).contiguous()
    a, a_sf = act(m, H, q[3], g)
    aqkv = torch.tensor([q[3] * q[2], k[3] * k[2], v[3] * v[2]], device="cuda", dtype=torch.float32)
    wo, swo, wo2, wo_in = load_nvfp4(h, f"{blk}.attn_out")
    wd, swd, wd2, wd_in = load_nvfp4(h, f"{blk}.mlp_out")
    ao, ao_sf = act(m, H, wo_in, g)
    d, d_sf = act(m, 3 * H, wd_in, g)
    return {
        "qkv": (a, wqkv, a_sf, sqkv, aqkv),
        "pair": (d, wd, d_sf, swd, torch.tensor([wd_in * wd2], device="cuda"),
                 ao, wo, ao_sf, swo, torch.tensor([wo_in * wo2], device="cuda")),
    }


def run_seq(name, call, out_tensors, ref, inputs, fails, info):
    """WARMUP + ITERS poisoned calls vs ref and call 0, then graph capture + 3 replays."""
    pristine = [t.clone() for t in inputs]
    first = None
    nref = 0
    for i in range(WARMUP + ITERS):
        for o in out_tensors:
            o.view(torch.int16).fill_(-1)
        call()
        torch.cuda.synchronize()
        cur = [o.clone() for o in out_tensors]
        if first is None:
            first = cur
        eq_ref = all(torch.equal(c.view(torch.int16), r.view(torch.int16)) for c, r in zip(cur, ref))
        eq_first = all(torch.equal(c.view(torch.int16), f.view(torch.int16)) for c, f in zip(cur, first))
        nref += eq_ref
        if not eq_ref:
            fails.append(f"{name} call {i} != main reference")
        if not eq_first:
            fails.append(f"{name} call {i} != call 0")
    if not all(torch.equal(a, b) for a, b in zip(pristine, inputs)):
        fails.append(f"{name} inputs mutated")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    gok = True
    for _ in range(3):
        for o in out_tensors:
            o.view(torch.int16).fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        gok &= all(torch.equal(o.view(torch.int16), r.view(torch.int16)) for o, r in zip(out_tensors, ref))
    if not gok:
        fails.append(f"{name} graph replay != main reference")
    del graph
    info[name] = {"bitwise_vs_main": f"{nref}/{WARMUP + ITERS}", "graph_x3": gok}


def fp8r_checks(h, g, fails, info):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import quantize_fp8r_weight

    blk = "content_mode_blocks.image.0"
    ws = {
        "qkv": torch.cat([h.get_tensor(f"{blk}.{p}_proj.weight") for p in "qkv"], 0),
        "gate_up": h.get_tensor(f"{blk}.mlp_in.weight"),
        "down": h.get_tensor(f"{blk}.mlp_out.weight"),
        "attn_out": h.get_tensor(f"{blk}.attn_out.weight"),
    }
    for m in (16384, 33280, 4176):
        for name, w in ws.items():
            w = w.cuda().to(torch.bfloat16)
            wd, wsc = quantize_fp8r_weight(w)
            x = (torch.randn((m, w.shape[1]), device="cuda", generator=g) * 1.5).to(torch.bfloat16)
            x[:, :8] *= 20
            a, sa = quantize_fp8r(x)
            a = a.contiguous()
            sa = sa.float().reshape(-1).contiguous()
            ref = torch._scaled_mm(a, wd.t(), scale_a=sa.reshape(-1, 1), scale_b=wsc.reshape(1, -1),
                                   out_dtype=torch.bfloat16)
            arm1 = fg.gemm(a, wd, sa, wsc, arm=1)
            if not torch.equal(arm1.view(torch.int16), ref.view(torch.int16)):
                fails.append(f"fp8r {name} M{m}: main arm1 != torch (reference broken)")
            out = torch.empty_like(ref)
            run_seq(f"fp8r/{name}/M{m}", lambda: fg.gemm(a, wd, sa, wsc, out=out), [out], [arm1],
                    [a, wd, sa, wsc], fails, info)
            from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_large_m as lm

            info[f"fp8r/{name}/M{m}"]["arm"] = lm.fp8r_arm(w.shape[0]) if lm.use("fp8r", m) else fg.ARM
        torch.cuda.empty_cache()


def main_check(ckpt):
    from safetensors import safe_open

    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_large_m as lm
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda", torch.cuda.current_device())
    fp.sm120_linear_preload(dev)
    strided.preload(dev)
    lpt.preload(dev)
    fg.preload(dev)
    info = {"routes": sorted(lm._enabled_routes()), "min_m": lm.LARGE_M_MIN,
            "qkv3_t256_info": int(lm.qkv3_module().t256_info(1)),
            "pair_t256_info": int(lm.pair_module().lpt_pair_info()),
            "main_pair_info": int(lpt.module().lpt_pair_info())}
    log(json.dumps(info))
    fails = []
    g = torch.Generator(device="cuda")
    g.manual_seed(99)
    main_qkv = strided.qkv_pingpong_module()
    main_pair = lpt.module()
    qws = strided.workspace(dev, 0)
    pws = torch.zeros(lpt._WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    with safe_open(str(ckpt_file(ckpt)), "pt", device="cpu") as h:
        for m in MS:
            for blk in BLOCKS:
                inp = block_inputs(h, blk, m, g)
                a, w, sa, sw, al = inp["qkv"]
                ref = torch.empty(3 * m * H, device="cuda", dtype=torch.bfloat16)
                main_qkv.gemm_qkv3_pingpong(a, w, sa, sw, al, ref, qws, strided.QKV3_RASTER,
                                            strided.QKV3_SWIZZLE)
                torch.cuda.synchronize()
                out = torch.empty_like(ref)
                run_seq(f"qkv3/{blk}/M{m}", lambda: strided.gemm_qkv3(a, w, sa, sw, al, out), [out],
                        [ref], [a, w, sa, sw, al], fails, info)
                info[f"qkv3/{blk}/M{m}"]["route"] = "t256" if lm.use("qkv3", m) else "main"
                d, wd, dsf, swd, ald, ao, wo, aosf, swo, alo = inp["pair"]
                rd = torch.empty(m, H, device="cuda", dtype=torch.bfloat16)
                ra = torch.empty(m, H, device="cuda", dtype=torch.bfloat16)
                main_pair.lpt_pair(d, wd, dsf, swd, ald, rd, ao, wo, aosf, swo, alo, ra, pws,
                                   lpt.sched_mode(), 0)
                torch.cuda.synchronize()
                od = torch.empty_like(rd)
                oa = torch.empty_like(ra)
                run_seq(f"pair/{blk}/M{m}",
                        lambda: lpt.run(d, wd, dsf, swd, ald, od, ao, wo, aosf, swo, alo, oa),
                        [od, oa], [rd, ra], [d, wd, dsf, swd, ald, ao, wo, aosf, swo, alo], fails, info)
                tile_m = 256 if lm.use("pair", m) else 128
                expect = 2 * ((m + tile_m - 1) // tile_m) * (H // 128)
                counter = int(lpt.workspace(dev)[:4].view(torch.int32).item())
                info[f"pair/{blk}/M{m}"].update({"route": f"t{tile_m}", "counter": counter,
                                                 "expected_tiles": expect})
                if lpt.sched_mode() >= 1 and counter != expect:
                    fails.append(f"pair/{blk}/M{m}: tile counter {counter} != {expect}")
                if m == 4176 and (lm.use("qkv3", m) or lm.use("pair", m)):
                    fails.append("M4176 took a large-M route")
                log(f"M{m} {blk}: qkv3 {info[f'qkv3/{blk}/M{m}']} pair {info[f'pair/{blk}/M{m}']} "
                    f"fails={len(fails)}")
                del inp, ref, out, rd, ra, od, oa
                torch.cuda.empty_cache()
        fp8r_checks(h, g, fails, info)
    return info, fails


def kill_switch_child(ckpt):
    """TRTLLM_FLUX3_SM120_LARGE_M_TILES=0: routed calls == main modules, large-M modules unused."""
    from safetensors import safe_open

    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_large_m as lm
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import quantize_fp8r_weight

    dev = torch.device("cuda", torch.cuda.current_device())
    fp.sm120_linear_preload(dev)
    strided.preload(dev)
    lpt.preload(dev)
    fg.preload(dev)
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    res = {"routes": sorted(lm._enabled_routes())}
    m = 16896
    with safe_open(str(ckpt_file(ckpt)), "pt", device="cpu") as h:
        inp = block_inputs(h, "single_blocks.0", m, g)
        a, w, sa, sw, al = inp["qkv"]
        ref = torch.empty(3 * m * H, device="cuda", dtype=torch.bfloat16)
        strided.qkv_pingpong_module().gemm_qkv3_pingpong(a, w, sa, sw, al, ref,
                                                         strided.workspace(dev, 0),
                                                         strided.QKV3_RASTER, strided.QKV3_SWIZZLE)
        out = torch.empty_like(ref)
        strided.gemm_qkv3(a, w, sa, sw, al, out)
        d, wd, dsf, swd, ald, ao, wo, aosf, swo, alo = inp["pair"]
        pws = torch.zeros(lpt._WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
        rd = torch.empty(m, H, device="cuda", dtype=torch.bfloat16)
        ra = torch.empty_like(rd)
        lpt.module().lpt_pair(d, wd, dsf, swd, ald, rd, ao, wo, aosf, swo, alo, ra, pws,
                              lpt.sched_mode(), 0)
        od = torch.empty_like(rd)
        oa = torch.empty_like(rd)
        lpt.run(d, wd, dsf, swd, ald, od, ao, wo, aosf, swo, alo, oa)
        wb = h.get_tensor("content_mode_blocks.image.0.mlp_in.weight").cuda().to(torch.bfloat16)
        wq, wsc = quantize_fp8r_weight(wb)
        x = (torch.randn((m, wb.shape[1]), device="cuda", generator=g) * 1.5).to(torch.bfloat16)
        xa, xs = quantize_fp8r(x)
        xa = xa.contiguous()
        xs = xs.float().reshape(-1).contiguous()
        f_ref = fg.gemm(xa, wq, xs, wsc, arm=1)
        f_out = fg.gemm(xa, wq, xs, wsc)
        torch.cuda.synchronize()
    res["qkv3_eq_main"] = bool(torch.equal(out.view(torch.int16), ref.view(torch.int16)))
    res["pair_eq_main"] = bool(torch.equal(od.view(torch.int16), rd.view(torch.int16))
                               and torch.equal(oa.view(torch.int16), ra.view(torch.int16)))
    res["fp8r_eq_arm1"] = bool(torch.equal(f_out.view(torch.int16), f_ref.view(torch.int16)))
    res["large_m_qkv3_module_built"] = lm.qkv3_module.cache_info().currsize
    res["large_m_pair_module_built"] = lm.pair_module.cache_info().currsize
    loaded = [mname for mname in sys.modules if "t256" in mname]
    res["t256_python_modules"] = loaded
    try:
        maps = Path("/proc/self/maps").read_text()
        res["t256_so_mapped"] = "t256" in maps
    except OSError:
        res["t256_so_mapped"] = None
    res["passed"] = (res["routes"] == [] and res["qkv3_eq_main"] and res["pair_eq_main"]
                     and res["fp8r_eq_arm1"] and res["large_m_qkv3_module_built"] == 0
                     and res["large_m_pair_module_built"] == 0 and not res["t256_so_mapped"])
    print(json.dumps(res), flush=True)


def main():
    ckpt = sys.argv[2] if sys.argv[1] == "child" else sys.argv[1]
    if sys.argv[1] == "child":
        kill_switch_child(ckpt)
        return
    info, fails = main_check(ckpt)
    env = dict(os.environ, TRTLLM_FLUX3_SM120_LARGE_M_TILES="0")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "child", ckpt], env=env,
                       capture_output=True, text=True)
    log(r.stdout[-3000:])
    if r.returncode != 0:
        log(r.stderr[-3000:])
    try:
        ks = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        ks = {"passed": False, "error": r.stderr[-500:]}
    if not ks.get("passed"):
        fails.append(f"kill switch: {ks}")
    print(json.dumps({"passed": not fails, "fails": fails[:30], "kill_switch": ks,
                      "compared": "routed gemm_qkv3 / lpt.run / fp8r gemm vs main's modules "
                                  "(and torch._scaled_mm), bitwise, real checkpoint weights",
                      "detail": info}), flush=True)


if __name__ == "__main__":
    main()
