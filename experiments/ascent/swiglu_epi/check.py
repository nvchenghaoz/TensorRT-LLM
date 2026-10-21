"""Branch 128 correctness (untimed). Run the SAME script in both trees:

  base tree (main 66cff73e): MODE=save    -> writes REF_DIR files (stock path)
  candidate tree:            MODE=compare -> compares against REF_DIR files

Per case (M, input kind) on a Qwen3.8-27B-shaped NVFP4 GatedMLP:
  down_proj input (FP4 bytes + swizzled SF incl. padding) and the MLP output.
Prefill rows (M >= 2048) feed the MLP an Fp4QuantizedTensor from
fused_gemma_add_rmsnorm_nvfp4 as in serving; smaller M feed BF16.

Candidate extras:
  * kill switch TRTLLM_GATEUP_SWIGLU_EPI=0 module == base files.
  * layout invariance: every valid gate_up nvfp4_gemm tactic at M=1..16 on the
    permuted weights == its own output on the unpermuted weights, column-
    permuted; fused_silu_mul_nvfp4_permuted(h_perm) == fused_silu_mul_nvfp4(h).
  * repeated execution: 10 warmup + 50 runs (eager prefill with the allocator
    pool NaN-poisoned before every run; decode as CUDA-graph replays with the
    output NaN-filled before every replay), each equal to run 1 and to base.
  * tactic-list lengths per bucket (must equal the base arm's print).
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_common import (DownInputTap, H, I, build_mlp, dev, make_input,  # noqa: E402
                        sha, tree_info)

from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner  # noqa: E402

MODE = os.environ.get("MODE", "compare")
REF_DIR = os.environ.get(
    "REF_DIR", "/home/scratch.chengzhang/trtllm-dev-qwen38-27b-20261015-171031/branch128_ref/66cff73e")
GS_H = 1.0  # SwiGLU amax ~2.6e3 at these weights (job 1072) -> 448*6/2623
MS = [1, 2, 3, 4, 8, 16, 17, 2048, 4096, 8192]
KINDS = ("random", "outlier")
WARM, ITERS = 10, 50


def log(*a):
    print(*a, flush=True)


def case_input(m, kind):
    return make_input(m, 1000 + m + (7 if kind == "outlier" else 0), kind, quantized=m >= 2048)


def run_case(mlp, tap, m, kind):
    x = case_input(m, kind)
    out = mlp(x)
    torch.cuda.synchronize()
    fp4, sf = tap.last
    return {"fp4": fp4.cpu(), "sf": sf.cpu() if sf is not None else None, "out": out.view(torch.int16).cpu()}


def poison_pool():
    t = torch.empty(1536 * 1024 * 1024, dtype=torch.uint8, device=dev)
    t.fill_(0xFF)
    del t


def eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and torch.equal(a, b)


def tactic_lists(mlp):
    res = {}
    runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, ["cutlass", "cublaslt", "cuda_core"])
    for name, lin, k in (("gate_up", mlp.gate_up_proj, H), ("down", mlp.down_proj, I)):
        for b in (1, 2, 4, 8, 16):
            a = torch.zeros((b, k // 2), dtype=torch.uint8, device=dev)
            asf = torch.zeros(((b + 127) // 128 * 128 * ((k // 16 + 3) // 4 * 4), ), dtype=torch.uint8, device=dev)
            try:
                t = runner.get_valid_tactics([a, lin.weight, asf, lin.weight_scale, lin.alpha], None)
                res[f"{name}@{b}"] = {"n": len(t), "sha1": hashlib.sha1(repr(list(t)).encode()).hexdigest()[:12]}
            except Exception as e:  # noqa: BLE001
                res[f"{name}@{b}"] = {"error": str(e)[:120]}
    return res


def main():
    info = tree_info()
    log("TREE", json.dumps(info))
    os.makedirs(REF_DIR, exist_ok=True)
    gs_h = GS_H
    torch.manual_seed(0)
    mlp = build_mlp(11, gs_h)
    permuted = bool(getattr(mlp, "_swiglu_epi_permuted", False))
    log("permuted", permuted)
    tap = DownInputTap(mlp)
    res = {"tree": info, "mode": MODE, "permuted": permuted, "cases": {}}
    ok = True
    # No autotune: both trees run the AutoTuner's deterministic fallback
    # tactic, so base and candidate GEMM picks are identical. Every valid
    # tactic is covered by the layout-invariance section below.
    res["tactic_lists"] = tactic_lists(mlp)
    log("TACTICS", json.dumps(res["tactic_lists"]))
    outs = {}
    for m in MS:
        for kind in KINDS:
            r = run_case(mlp, tap, m, kind)
            outs[(m, kind)] = r
            key = f"{m}_{kind}"
            hs = {k: sha(v) for k, v in r.items()}
            entry = {"sha": hs, "nan_out": bool(torch.isnan(r["out"].view(torch.bfloat16).float()).any())}
            path = os.path.join(REF_DIR, f"{key}.pt")
            if MODE == "save":
                torch.save(r, path)
            else:
                ref = torch.load(path)
                entry["eq_base"] = {k: eq(r[k], ref[k]) for k in ("fp4", "sf", "out")}
                ok &= all(entry["eq_base"].values())
            res["cases"][key] = entry
            log("CASE", key, json.dumps(entry))
    # ---- C3: every gate_up tactic at decode M vs BASE-tree files -----------
    from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4 as fsm
    runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, ["cutlass", "cublaslt", "cuda_core"])
    gsd = mlp.down_proj.input_scale
    per_tac = {}
    pt_ok = True
    for m in (1, 2, 3, 4, 8, 16):
        x = case_input(m, "random")
        a, asf = torch.ops.trtllm.fp4_quantize(x, mlp.gate_up_proj.input_scale, 16, False)
        ins = [a, mlp.gate_up_proj.weight, asf, mlp.gate_up_proj.weight_scale, mlp.gate_up_proj.alpha]
        tacs = runner.get_valid_tactics(ins, None)
        got = {}
        for t in tacs:
            h = runner(ins, tactic=t)
            if permuted:
                q, s = torch.ops.trtllm.fused_silu_mul_nvfp4_permuted(h, gsd)
            else:
                q, s = torch.ops.trtllm.fused_silu_mul_nvfp4(h, gsd)
            o = torch.ops.trtllm.nvfp4_gemm(q, mlp.down_proj.weight, s, mlp.down_proj.weight_scale,
                                            mlp.down_proj.alpha, torch.bfloat16)
            torch.cuda.synchronize()
            got[str(t)] = {"fp4": q.cpu(), "sf": s.cpu(), "out": o.view(torch.int16).cpu()}
        path = os.path.join(REF_DIR, f"tactics_{m}.pt")
        if MODE == "save":
            torch.save(got, path)
            per_tac[str(m)] = {"tactics": list(got.keys())}
        else:
            ref = torch.load(path)
            bad = [k for k in ref if k not in got or not all(eq(got[k][f], ref[k][f]) for f in ("fp4", "sf", "out"))]
            extra = [k for k in got if k not in ref]
            per_tac[str(m)] = {"tactics": len(got), "base_tactics": len(ref), "bad": bad, "extra": extra}
            pt_ok &= not bad and not extra and len(ref) > 0
        log("PER_TACTIC", m, json.dumps(per_tac[str(m)]))
    del fsm
    res["per_tactic_vs_base"] = per_tac
    ok &= pt_ok
    # A2: derived tensors of gate_up after load.
    gu = mlp.gate_up_proj
    res["gate_up_derived"] = {
        "weight_shape": list(gu.weight.shape), "weight_scale_numel": gu.weight_scale.numel(),
        "sf_padding_rows": (gu.weight_scale.numel() // (H // 16)) - gu.out_features,
        "uses_marlin_nvfp4": bool(gu.uses_marlin_nvfp4),
        "nvfp4_decode_copy": getattr(gu, "_nvfp4_decode_weight", None) is not None,
        "w4a16_scale_linear": hasattr(gu, "_w4a16_weight_scale_linear"),
        "cute_dsl_swiglu": bool(gu.can_use_cute_dsl_nvfp4_swiglu_blackwell()),
    }
    log("DERIVED", json.dumps(res["gate_up_derived"]))
    if MODE == "save":
        print(json.dumps({"passed": True, "saved": REF_DIR, "result": res}))
        return

    # ---- kill switch ------------------------------------------------------
    os.environ["TRTLLM_GATEUP_SWIGLU_EPI"] = "0"
    mlp_off = build_mlp(11, gs_h)
    os.environ.pop("TRTLLM_GATEUP_SWIGLU_EPI")
    tap_off = DownInputTap(mlp_off)
    ks = {"permuted": bool(getattr(mlp_off, "_swiglu_epi_permuted", False))}
    ks_ok = not ks["permuted"]
    for m in MS:
        for kind in KINDS:
            r = run_case(mlp_off, tap_off, m, kind)
            ref = torch.load(os.path.join(REF_DIR, f"{m}_{kind}.pt"))
            e = all(eq(r[k], ref[k]) for k in ("fp4", "sf", "out"))
            ks[f"{m}_{kind}"] = e
            ks_ok &= e
    res["kill_switch"] = ks
    log("KILL_SWITCH", json.dumps(ks))
    ok &= ks_ok

    # ---- layout invariance per tactic (decode GEMMs) ----------------------
    from tensorrt_llm._torch.modules.fused_silu_mul_nvfp4 import gate_up_swiglu_row_permutation
    perm = gate_up_swiglu_row_permutation(2 * I).to(dev)
    runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, ["cutlass", "cublaslt", "cuda_core"])
    inv = {}
    inv_ok = True
    gs = torch.tensor([gs_h], dtype=torch.float32, device=dev)
    for m in (1, 2, 3, 4, 8, 16):
        x = case_input(m, "random")
        a, asf = torch.ops.trtllm.fp4_quantize(x, mlp.gate_up_proj.input_scale, 16, False)
        ins_p = [a, mlp.gate_up_proj.weight, asf, mlp.gate_up_proj.weight_scale, mlp.gate_up_proj.alpha]
        ins_o = [a, mlp_off.gate_up_proj.weight, asf, mlp_off.gate_up_proj.weight_scale, mlp_off.gate_up_proj.alpha]
        tacs = runner.get_valid_tactics(ins_p, None)
        bad = []
        for t in tacs:
            try:
                hp = runner(ins_p, tactic=t)
                ho = runner(ins_o, tactic=t)
            except Exception as e:  # noqa: BLE001
                bad.append([str(t), "error " + str(e)[:80]])
                continue
            if not torch.equal(hp, ho[:, perm]):
                bad.append([str(t), "mismatch"])
                continue
            qp, sp = torch.ops.trtllm.fused_silu_mul_nvfp4_permuted(hp, gs)
            qo, so = torch.ops.trtllm.fused_silu_mul_nvfp4(ho, gs)
            if not (torch.equal(qp, qo) and torch.equal(sp, so)):
                bad.append([str(t), "silu mismatch"])
        inv[str(m)] = {"tactics": len(tacs), "bad": bad}
        inv_ok &= not bad
    res["layout_invariance"] = inv
    log("INVARIANCE", json.dumps(inv))
    ok &= inv_ok

    # ---- repeated execution ------------------------------------------------
    rep = {}
    rep_ok = True
    for m in (1, 2, 4, 2048, 8192):
        ref = torch.load(os.path.join(REF_DIR, f"{m}_random.pt"))
        x = case_input(m, "random")
        if m <= 16:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    mlp(x)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = mlp(x)
            torch.cuda.synchronize()

            def step():
                gout.view(torch.int16).fill_(-1)  # BF16 NaN pattern
                g.replay()
                torch.cuda.synchronize()
                return gout.view(torch.int16).cpu(), tap.last
        else:

            def step():
                poison_pool()
                o = mlp(x)
                torch.cuda.synchronize()
                return o.view(torch.int16).cpu(), tap.last

        first = None
        nbad_first, nbad_base, maxdiff = 0, 0, 0
        for it in range(WARM + ITERS):
            o, (fq, fsf) = step()
            cur = (o, fq.cpu(), fsf.cpu() if fsf is not None else None)
            if first is None:
                first = cur
            if it >= WARM:
                if not (eq(cur[0], first[0]) and eq(cur[1], first[1]) and eq(cur[2], first[2])):
                    nbad_first += 1
                if not eq(cur[0], ref["out"]):
                    nbad_base += 1
                    d = (cur[0].view(torch.bfloat16).float() - ref["out"].view(torch.bfloat16).float()).abs().max()
                    maxdiff = max(maxdiff, float(d))
        rep[str(m)] = {"iters": ITERS, "neq_run1": nbad_first, "neq_base": nbad_base, "max_abs_diff_vs_base": maxdiff,
                       "mode": "graph_replay" if m <= 16 else "eager_poisoned_pool"}
        rep_ok &= nbad_first == 0 and nbad_base == 0
        log("REPEAT", m, json.dumps(rep[str(m)]))
        if m <= 16:
            del g
    res["repeated"] = rep
    ok &= rep_ok
    log("FULL_RESULT", json.dumps(res))
    summary = {
        "cases_eq_base": sum(all(v["eq_base"].values()) for v in res["cases"].values()),
        "cases_total": len(res["cases"]),
        "kill_switch_ok": all(v for k, v in res["kill_switch"].items() if k != "permuted")
        and not res["kill_switch"]["permuted"],
        "per_tactic_bad": {k: v["bad"] for k, v in res["per_tactic_vs_base"].items()},
        "invariance_bad": {k: v["bad"] for k, v in res["layout_invariance"].items()},
        "repeated": {k: [v["neq_run1"], v["neq_base"]] for k, v in res["repeated"].items()},
        "tactic_lists": res["tactic_lists"],
        "head": res["tree"]["head"], "tree": res["tree"]["tree"],
    }
    print(json.dumps({"passed": bool(ok),
                      "compared": "GatedMLP down_proj input FP4+SF bytes and MLP output vs BASE-tree (66cff73e) "
                                  "saved files; kill switch; per-tactic layout invariance; 10+50 repeats",
                      "summary": summary}))


if __name__ == "__main__":
    main()
