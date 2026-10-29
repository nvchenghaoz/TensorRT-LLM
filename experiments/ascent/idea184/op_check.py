"""Idea 184 STEP 1 op-level correctness (untimed) for the nvfp4_out ("nfo") epilogue of the routed
fmha_v2 flux3 d128 kernel.

Per S in (4176, 4096, 4256) and per case (randn, peaky, zero blocks, tiny O -> subnormal / zero SF,
large global scale -> saturated SF), unit and real global scale:
  (a) nfo module run() (BF16 path, fp4 pointer null) == main's routed module BF16 O, bitwise;
  (b) run_nvfp4_out fp4 + sf (padding rows included; outputs poisoned before the call) ==
      flashinfer nvfp4_quantize(O_main [S,3072], gs, layout_128x4, do_shuffle=False, cute-dsl),
      i.e. exactly what flux3_primsts.nvfp4_quantize_prepared computes, bitwise;
  (c) attn_out from lpt.run fed (fp4, sf) == main's path (nvfp4_quantize_prepared + lpt.run),
      bitwise, same down operands.
Then: 100 repeated calls == call 0 (poisoned outputs each call), inputs unchanged; interleaved S
== fresh; CUDA-graph replay x5 == eager; register / stack / local-memory usage of the nfo kernel
(cuobjdump) and SASS STL/LDL count; SF histogram (zero, subnormal, 448) to show edge cases hit.
Last stdout line: {"passed": ...}.
"""

import json
import os
import re
import subprocess
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as fx
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_nvfp4 import _scalar
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

H, D, HID, FF = 24, 128, 3072, 9216


def log(*a):
    print(*a, flush=True)


def make_qkv(S, kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, S, H, D)
    q = torch.randn(shape, device="cuda", generator=g) * 32
    k = torch.randn(shape, device="cuda", generator=g) * 32
    v = torch.randn(shape, device="cuda", generator=g) * 32
    s1 = (1 / 32) ** 2 / (D ** 0.5)
    s2 = 1 / 32
    if kind == "peaky":
        s1 *= 6.0
        # a few very hot keys per head
        k[:, ::97] *= 4
    elif kind == "zero_blocks":
        # zero 16-column V blocks in some heads -> exactly zero O blocks
        v[:, :, ::3, 16:32] = 0
        v[:, :, 5, :] = 0
    elif kind == "tiny":
        s2 = 1 / 32 * 1e-3  # O ~ 1e-4..1e-3: subnormal / zero E4M3 scales at gs=1
    elif kind == "mixed_tiny":
        v[:, :, 1::2, :] *= 1e-3  # half the heads tiny
    q8, k8, v8 = (t.clamp(-448, 448).to(torch.float8_e4m3fn).contiguous() for t in (q, k, v))
    return q8, k8, v8, torch.tensor([s1], device="cuda"), torch.tensor([s2], device="cuda")


def bf16_o(mod, q8, k8, v8, s1, s2):
    S = q8.shape[1]
    out = torch.empty((1, S, H, D), device="cuda", dtype=torch.bfloat16)
    fx.fmha_v2_flux3_d128(q8, k8, v8, out, H, D, S, scale_bmm1_d=s1, scale_bmm2_d=s2,
                          causal=False, lse=None, module=mod)
    return out


def nfo_out(mod, q8, k8, v8, s1, s2, gs_t, poison=0xAB):
    S = q8.shape[1]
    fp4 = torch.full((S, HID // 2), poison, device="cuda", dtype=torch.uint8)
    sf = torch.full((fx.nvfp4_scale_bytes(S, HID),), poison, device="cuda", dtype=torch.uint8)
    fx.fmha_v2_flux3_d128_nvfp4_out(q8, k8, v8, fp4, sf, gs_t, H, D, S, scale_bmm1_d=s1,
                                    scale_bmm2_d=s2, module=mod)
    return fp4, sf


def ref_quant(o, gs):
    from flashinfer import SfLayout
    from flashinfer import nvfp4_quantize as quantize

    d, s = quantize(o.reshape(-1, HID), gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
                    backend="cute-dsl")
    return d.view(torch.uint8).contiguous(), s.view(torch.uint8).reshape(-1).contiguous()


def diff(a, b):
    return int((a != b).sum().item())


def sf_hist(sf):
    s = sf.to(torch.int32)
    return {"zero": int((s == 0).sum()), "subnormal": int(((s > 0) & (s < 8)).sum()),
            "max448": int((s == 0x7E).sum()), "n": int(s.numel())}


def resource_usage(levers):
    spec = fx.gen_flux3_d128_spec(levers)
    path = str(spec.get_library_path())
    out = {"lib": path}
    try:
        r = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", path],
                           capture_output=True, text=True, timeout=120)
        lines = [ln.strip() for ln in r.stdout.splitlines() if "REG" in ln or "Function" in ln]
        out["resource"] = lines[:12]
        regs = [int(m) for m in re.findall(r"REG:(\d+)", r.stdout)]
        stack = [int(m) for m in re.findall(r"STACK:(\d+)", r.stdout)]
        local = [int(m) for m in re.findall(r"LOCAL:(\d+)", r.stdout)]
        out.update({"REG": regs, "STACK": stack, "LOCAL": local})
        s = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", path], capture_output=True,
                           text=True, timeout=300)
        out["STL"] = len(re.findall(r"\bSTL\b", s.stdout))
        out["LDL"] = len(re.findall(r"\bLDL\b", s.stdout))
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    return out


def lpt_check(S, o_main, fp4, sf, s_attn, seed):
    """attn_out via lpt.run: main path (nvfp4_quantize_prepared(O) + lpt.run) vs (fp4, sf)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x_down = torch.randn(S, FF, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
    w_out = torch.randn(HID, HID, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(HID, FF, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02

    def gsf(t):
        return float(t.abs().amax().float() / (448.0 * 6.0))

    def inv(s):
        return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)

    s_down_in, s_wout, s_wdown = gsf(x_down), gsf(w_out), gsf(w_down)
    a_down, sf_down_in = fp.nvfp4_quantize(x_down, inv(s_down_in))
    q_out, sf_out = fp.nvfp4_quantize(w_out, inv(s_wout))
    q_down, sf_down = fp.nvfp4_quantize(w_down, inv(s_wdown))
    out_ad = PreparedNVFP4(q_out, sf_out, s_attn, s_wout)
    down_ad = PreparedNVFP4(q_down, sf_down, s_down_in, s_wdown)
    a_down_alpha = _scalar(o_main.device, s_down_in * s_wdown)
    a_attn_alpha = _scalar(o_main.device, s_attn * s_wout)

    def run(data, scales):
        down = torch.empty((S, HID), device="cuda", dtype=torch.bfloat16)
        attn = torch.empty((S, HID), device="cuda", dtype=torch.bfloat16)
        lpt.run(a_down, down_ad.weight, sf_down_in, down_ad.scale, a_down_alpha, down, data,
                out_ad.weight, scales, out_ad.scale, a_attn_alpha, attn)
        return attn, down

    d_main, s_main = fp.nvfp4_quantize_prepared(o_main.reshape(S, HID), out_ad.weight)
    attn_m, down_m = run(d_main, s_main)
    attn_n, down_n = run(fp4, sf)
    return {"prepared_fp4_eq_nfo": bool(torch.equal(d_main, fp4)),
            "prepared_sf_eq_nfo": bool(torch.equal(s_main, sf)),
            "attn_out_eq": bool(torch.equal(attn_m, attn_n)),
            "down_eq": bool(torch.equal(down_m, down_n))}


def main():
    torch.manual_seed(0)
    main_levers = tuple(fx.ROUTED_LEVERS)
    nfo_levers = main_levers + ("nvfp4_out",)
    log(f"[build] main levers {main_levers} -> {fx.flux3_d128_kernel_name(main_levers)}")
    log(f"[build] nfo levers {nfo_levers} -> {fx.flux3_d128_kernel_name(nfo_levers)}")
    mod_main = fx.get_flux3_d128_module(main_levers)
    log("[build] main module loaded")
    mod_nfo = fx.get_flux3_d128_module(nfo_levers)
    log("[build] nfo module loaded")
    res = {"kernel_main": fx.flux3_d128_kernel_name(main_levers),
           "kernel_nfo": fx.flux3_d128_kernel_name(nfo_levers), "cases": []}
    ok = True
    lpt.preload(torch.device("cuda"))
    seed = 100
    for S in (4176, 4096, 4256):
        for kind in ("randn", "peaky", "zero_blocks", "tiny", "mixed_tiny"):
            seed += 1
            q8, k8, v8, s1, s2 = make_qkv(S, kind, seed)
            pristine = [t.clone() for t in (q8, k8, v8, s1, s2)]
            o_main = bf16_o(mod_main, q8, k8, v8, s1, s2)
            o_nfo = bf16_o(mod_nfo, q8, k8, v8, s1, s2)
            amax = float(o_main.float().abs().amax())
            gss = {"unit": 1.0, "real": 448.0 * 6.0 / max(amax, 1e-30)}
            if kind == "randn":
                gss["sat"] = gss["real"] * 1e3  # SF saturates at 448
            for gname, gs in gss.items():
                gs_t = torch.tensor([gs], device="cuda", dtype=torch.float32)
                fp4, sf = nfo_out(mod_nfo, q8, k8, v8, s1, s2, gs_t)
                rd, rs = ref_quant(o_main, gs)
                fp4_b, sf_b = nfo_out(mod_nfo, q8, k8, v8, s1, s2, gs_t, poison=0x5C)
                row = {"S": S, "kind": kind, "gs": gname, "gs_value": gs,
                       "bf16_nfo_eq_main": bool(torch.equal(o_main, o_nfo)),
                       "fp4_mismatch": diff(fp4, rd), "sf_mismatch": diff(sf, rs),
                       "poison_independent": bool(torch.equal(fp4, fp4_b) and torch.equal(sf, sf_b)),
                       "sf_shape": [int(sf.numel()), int(rs.numel())],
                       "sf_hist": sf_hist(rs),
                       "pad_rows_sf_zero": None}
                padded = (S + 127) // 128 * 128
                if padded > S:
                    # padded rows' SF bytes must be 0 in both
                    rows = torch.arange(S, padded, device="cuda")
                    cols = torch.arange(HID // 16, device="cuda")
                    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
                    off = ((cc % 4) + (cc // 4) * 512 + (rr % 32) * 16 + ((rr % 128) // 32) * 4
                           + (rr // 128) * (128 * (HID // 16)))
                    row["pad_rows_sf_zero"] = bool((sf[off.reshape(-1)] == 0).all())
                if kind in ("randn", "tiny") and gname in ("unit", "real"):
                    s_attn = 1.0 if gname == "unit" else 1.0 / gs
                    row["lpt"] = lpt_check(S, o_main, fp4, sf, s_attn, seed)
                row["inputs_unchanged"] = all(torch.equal(a, b) for a, b in
                                              zip(pristine, (q8, k8, v8, s1, s2)))
                good = (row["bf16_nfo_eq_main"] and row["fp4_mismatch"] == 0
                        and row["sf_mismatch"] == 0 and row["poison_independent"]
                        and row["inputs_unchanged"] and row["pad_rows_sf_zero"] in (None, True)
                        and all(row.get("lpt", {"x": True}).values()))
                row["ok"] = bool(good)
                ok &= good
                res["cases"].append(row)
                log(f"[case] {json.dumps(row)}")

    # Repeats, interleaved S, CUDA graph.
    sets = {S: make_qkv(S, "randn", 900 + S) for S in (4176, 4096, 4256)}
    gs_t = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    fresh = {S: nfo_out(mod_nfo, *sets[S], gs_t) for S in sets}
    pristine = {S: [t.clone() for t in sets[S]] for S in sets}
    rep_bad = 0
    for i in range(100):
        fp4, sf = nfo_out(mod_nfo, *sets[4176], gs_t, poison=(i * 37) & 0xFF)
        rep_bad += int(not (torch.equal(fp4, fresh[4176][0]) and torch.equal(sf, fresh[4176][1])))
    inter_bad = 0
    for i in range(30):
        S = (4176, 4096, 4256)[i % 3]
        fp4, sf = nfo_out(mod_nfo, *sets[S], gs_t, poison=(i * 11) & 0xFF)
        inter_bad += int(not (torch.equal(fp4, fresh[S][0]) and torch.equal(sf, fresh[S][1])))
    # graph: 4176 then 4096 into fixed buffers poisoned before each replay
    bufs = {S: (torch.empty((S, HID // 2), device="cuda", dtype=torch.uint8),
                torch.empty((fx.nvfp4_scale_bytes(S, HID),), device="cuda", dtype=torch.uint8))
            for S in (4176, 4096)}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for S in (4176, 4096):  # warm on the capture stream
            fx.fmha_v2_flux3_d128_nvfp4_out(*sets[S][:3], *bufs[S], gs_t, H, D, S,
                                            scale_bmm1_d=sets[S][3], scale_bmm2_d=sets[S][4],
                                            module=mod_nfo)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for S in (4176, 4096):
            fx.fmha_v2_flux3_d128_nvfp4_out(*sets[S][:3], *bufs[S], gs_t, H, D, S,
                                            scale_bmm1_d=sets[S][3], scale_bmm2_d=sets[S][4],
                                            module=mod_nfo)
    graph_bad = 0
    for i in range(5):
        for S in bufs:
            bufs[S][0].fill_(0xC3 ^ i)
            bufs[S][1].fill_(0x3C ^ i)
        graph.replay()
        torch.cuda.synchronize()
        for S in bufs:
            graph_bad += int(not (torch.equal(bufs[S][0], fresh[S][0])
                                  and torch.equal(bufs[S][1], fresh[S][1])))
    inputs_ok = all(torch.equal(a, b) for S in sets for a, b in zip(pristine[S], sets[S]))
    res["repeat"] = {"repeat_100_mismatching": rep_bad, "interleaved_30_mismatching": inter_bad,
                     "graph_replay_5x2_mismatching": graph_bad, "inputs_unchanged": inputs_ok}
    log(f"[repeat] {json.dumps(res['repeat'])}")
    ok &= rep_bad == 0 and inter_bad == 0 and graph_bad == 0 and inputs_ok
    res["resource_nfo"] = resource_usage(nfo_levers)
    res["resource_main"] = resource_usage(main_levers)
    log(f"[resource] nfo {json.dumps(res['resource_nfo'])}")
    log(f"[resource] main {json.dumps(res['resource_main'])}")
    out_dir = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(out_dir, "idea184_op_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"passed": bool(ok), "n_cases": len(res["cases"]),
                      "failing_cases": [(c["S"], c["kind"], c["gs"]) for c in res["cases"]
                                        if not c["ok"]],
                      "repeat": res["repeat"],
                      "reg_nfo": res["resource_nfo"].get("REG"),
                      "stack_nfo": res["resource_nfo"].get("STACK"),
                      "stl_ldl_nfo": [res["resource_nfo"].get("STL"),
                                      res["resource_nfo"].get("LDL")],
                      "reg_main": res["resource_main"].get("REG"),
                      "kernel_nfo": res["kernel_nfo"]}))


if __name__ == "__main__":
    sys.exit(main())
