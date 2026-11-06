"""Idea 208 acceptance items 3 + 4 in ONE job (queue-bound rule: one job per decision).

(1) S1 audit (static, reads the tree as the job runs it): every consumer of the MoE output all-reduce and of the
    hc_attn K1, each with the file:line of its first-statement wait (or its plain, fully ordered launch); the
    hc_ffn K1 and the o-proj ARs untouched; the 128-row tail-MoE / prefill / ADP-zero-row calls keep END trigger.
    Every entry is located by pattern and printed as file:line: text; a missing pattern fails the job.
(2) The trigger rule _moe_ar_trigger_at_end(M) for M = 0..9, 16, 128, 512 with the default env and (in a
    subprocess) with TRTLLM_DSV4_MOE_AR_EARLY_TRIGGER=0.
(3) layer_check.py (real DeepseekV41DecoderLayer.forward, 40 layers, real mHC split boundaries, stand-in MoE AR with
    the tree's trigger rule, real attention-input A/B chain): graphs REF / SPLIT / SLEEP (K2-late) / NOPDL
    (TRTLLM_ENABLE_PDL=0) at M = 1/2/4/8, bitwise vs REF every replay.
argv: --ckpt <checkpoint>
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def log(*a):
    print(*a, flush=True)


def find(path, pattern, start=None, nth=1):
    """1-based line number of the nth line matching pattern (after the first line matching start)."""
    with open(os.path.join(ROOT, path)) as f:
        lines = f.read().splitlines()
    i0 = 0
    if start is not None:
        for i, ln in enumerate(lines):
            if re.search(start, ln):
                i0 = i
                break
        else:
            return None, None
    hits = 0
    for i in range(i0, len(lines)):
        if re.search(pattern, lines[i]):
            hits += 1
            if hits == nth:
                return i + 1, lines[i].strip()
    return None, None


V4 = "tensorrt_llm/_torch/models/modeling_deepseekv4.py"
V41 = "tensorrt_llm/_torch/models/modeling_deepseekv41.py"
SPLIT = "tensorrt_llm/_torch/modules/mhc/mhc_lagged_split.py"
FMA = "cpp/tensorrt_llm/kernels/mhcKernels/mhc_fused_fma.cuh"
FHC = "cpp/tensorrt_llm/kernels/mhcKernels/mhcFusedHcKernel.cu"
ARK = "cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu"
LOWM = "cpp/tensorrt_llm/kernels/minimaxM3/mxfp8LowMGemm.cu"
WO = "tensorrt_llm/_torch/attention/backends/sparse/csa2/b12x_wo.py"

# (role, path, pattern, start-anchor or None, nth)
AUDIT = [
    # --- the producer ---
    ("MoE AR call (DeepseekV4MoE.forward)", V4, r"final_hidden_states, all_reduce_params=final_all_reduce_params", None, 1),
    ("MoE AR params: per-call trigger rule", V4,
     r"trigger_completion_at_end=_moe_ar_trigger_at_end\(hidden_states.shape\[0\]\)", None, 1),
    ("AR kernel: wait first", ARK, r"cudaGridDependencySynchronize\(\);", r"TriggerCompletionAtEnd = true>", 1),
    ("AR kernel: early trigger (after its wait)", ARK, r"if constexpr \(!TriggerCompletionAtEnd\)", None, 1),
    # --- consumers of the MoE AR output x_ffn ---
    ("C1 V4.1 split boundary: next layer hc_attn (which=0)", V41, r"= self._mhc_boundary\($", None, 1),
    ("C1 fork: event0.record() before K1 (K2 on aux = full-completion edge)",
     "tensorrt_llm/_torch/modules/multi_stream_utils.py", r"event0.record\(\)",
     r"def maybe_execute_in_parallel_deferred_join", 1),
    ("C1 K1 trigger-first: launch_dependents", SPLIT, r"tl.extra.cuda.gdc_launch_dependents\(\)",
     r"if TRIGGER_FIRST == 1:", 1),
    ("C1 K1 trigger-first: wait before any global access", SPLIT, r"gdc_wait_with_memory_clobber\(\)",
     r"if TRIGGER_FIRST == 1:", 1),
    ("C1 K1 default (hc_ffn, which=1): wait first", SPLIT, r"gdc_wait_with_memory_clobber\(\)",
     r"elif TRIGGER_FIRST == 2:", 1),
    ("C1/C2 K2 + fused boundary: fused_pmap_gemm_fma_ksplit wait first", FMA, r"cudaGridDependencySynchronize\(\);",
     r"void fused_pmap_gemm_fma_ksplit", 1),
    ("C2 fused boundary: fused_pmap_gemm_fma_allinone wait first", FMA, r"cudaGridDependencySynchronize\(\);",
     r"void fused_pmap_gemm_fma_allinone", 1),
    ("C2 fused boundary: mtile variant wait first", FMA, r"cudaGridDependencySynchronize\(\);",
     r"__launch_bounds__\(256, \(TM", 1),
    ("C2 fused boundary: zero-workspaces plain launch (fully ordered)", FHC, r"fhcZeroWorkspacesKernel<<<", None, 1),
    ("C2 fused boundary: tcgen05 path plain launch (fully ordered)", FHC, r"fa<<<grid, block, fused_smem, stream>>>",
     None, 1),
    ("C3 post_mapping kernel: plain launch (fully ordered)", "cpp/tensorrt_llm/kernels/mhcKernels/mhcKernels.cu",
     r"mhcPostMappingKernel<<<", None, 1),
    ("C3 V4.1 resolved post_mapping call", V41, r"residual = self.hc_ffn.post_mapping\(", None, 1),
    ("C3 model end (_resolve_hc_state)", V41, r"residual = self.layers\[0\].hc_attn.post_mapping\(", None, 1),
    ("C4 V4 forward_MoE caller (next: fused_hc ksplit or post_mapping)", V4, r"x_ffn = self.forward_MoE\(", None, 1),
    ("C5 ADP zero-row forward_MoE (M=0 -> END)", V41, r"hidden_states=residual\[:, 0, :\],", None, 1),
    ("C6 128-row tail-MoE forward_MoE (M=128 -> END)", "tensorrt_llm/_torch/models/_dsv41_tail_moe_graph.py",
     r"x_ffn = layer.forward_MoE\(", None, 1),
    ("C7 landing-60 shared-expert eager overlap gate (M >= 1024 -> END; same AR call)", V4,
     r"and hidden_states.shape\[0\] >= SHARED_OVERLAP_MIN_TOKENS", None, 1),
    ("C7 landing-60 overlap default threshold", V4, r"SHARED_OVERLAP_MIN_TOKENS = int\(", None, 1),
    # --- consumers of K1 (hc_attn) when it triggers first ---
    ("K1 caller: trigger_first only at hc_attn", V41, r"k1_trigger_first=which == 0", None, 1),
    ("K1 successor A: wq_a||wkv low_m_mm trigger_at_start", "tensorrt_llm/_torch/attention/backends/sparse/csa2/module.py",
     # Landing 56 (branch 179) appended l2_prefetch=chain[:2] (a hint on B's constant weights) to this call.
     r"rows = low_m_mm\(hidden_states, \*self._wq_a_wkv, trigger_at_start=True(, l2_prefetch=chain\[:2\])?\)", None, 1),
    ("K1 successor A (no B chain): plain low_m_mm, wait-first", "tensorrt_llm/_torch/attention/backends/sparse/csa2/module.py",
     r"rows = low_m_mm\(hidden_states, \*self._wq_a_wkv\)$", None, 1),
    ("A kernel: launch_dependents (bit 1) after weight staging", LOWM, r"griddepcontrol.launch_dependents;", None, 2),
    ("A kernel: wait before reading activations", LOWM, r"griddepcontrol.wait;", None, 2),
    ("A fallback: mxfp8_quantize wait first", "cpp/tensorrt_llm/kernels/quantization.cuh",
     r"cudaGridDependencySynchronize\(\);", None, 1),
    ("K1 plain call (not the fold) carries trigger_first", "tensorrt_llm/_torch/modules/mhc/hyper_connection.py",
     r"trigger_first=k1_trigger_first,", None, 1),
    ("K1 fold returns before the plain K1 (175 path first)", "tensorrt_llm/_torch/modules/mhc/hyper_connection.py",
     r"return k1_fold.out", None, 1),
    ("K1 M-gate: trigger-first only at M >= K1_TRIGGER_FIRST_MIN_TOKENS", SPLIT,
     r"M >= K1_TRIGGER_FIRST_MIN_TOKENS", None, 1),
    # --- other x_ffn consumer (Engram/DSpark resolve at entry): plain post_mapping launch ---
    ("C3b V4.1 entry resolve (needs_resolved) post_mapping", V41, r"residual = self.hc_attn.post_mapping\(", None, 1),
    # --- landing 55 (branch 175) paths: consumers of the ATTENTION AR / x_attn, untouched by 177 ---
    ("175 attn AR early trigger M<=4 (main, untouched)", WO, r"_ATTN_AR_EARLY_TRIGGER_MAX_M = 4", None, 1),
    ("o-proj AR site 1 (main's 175 params, untouched)", WO, r"all_reduce_params=_attn_all_reduce_params\(m\)",
     None, 1),
    ("o-proj AR site 2 (main's 175 params, untouched)", WO,
     r"all_reduce_params=_attn_all_reduce_params\(output.shape\[0\]\)", None, 1),
    ("175 K1 fold only at hc_ffn (which=1)", V41, r"defer_k1 = self._k1_fold_eligible\(x_attn, image_mask\)", None, 1),
    ("175 fold consumer dsv4FusedRouterKernel: wait before reading activations",
     "cpp/tensorrt_llm/kernels/moe/routing/dsv4FusedRouter.cu", r"asm volatile\(\"griddepcontrol.wait;\"", None, 1),
    ("175 shared-expert K1 on aux (run_copy): default wait-first, no trigger_first", SPLIT,
     r"return mhc_lagged_layer_input\(\*self.args, self.norm_eps, self.reducer_threads\)$", r"def run_copy", 1),
]


def static_audit():
    rows, ok = [], True
    for role, path, pat, start, nth in AUDIT:
        ln, txt = find(path, pat, start, nth)
        rows.append({"role": role, "where": f"{path}:{ln}" if ln else f"{path}:MISSING", "text": txt})
        log(f"  {role:72s} {path}:{ln}: {txt}")
        ok &= ln is not None
    with open(os.path.join(ROOT, WO)) as f:
        # Main (landing 55) owns the only override: _attn_all_reduce_params' trigger_completion_at_end=False.
        # (the docstring also names it, so count the constructor call only)
        wo_clean = f.read().count("AllReduceParams(trigger_completion_at_end=False)") == 1
    log(f"  b12x_wo.py has only main's single (175) trigger override: {wo_clean}")
    return bool(ok and wo_clean), rows


RULE_SNIPPET = (
    "import json; from tensorrt_llm._torch.models import modeling_deepseekv4 as V4; "
    "print(json.dumps({m: V4._moe_ar_trigger_at_end(m) for m in [0,1,2,3,4,5,6,7,8,9,16,128,512,1024,4096]}))"
)


def rule(env_off):
    env = dict(os.environ)
    if env_off:
        env["TRTLLM_DSV4_MOE_AR_EARLY_TRIGGER"] = "0"
    out = subprocess.run([sys.executable, "-c", RULE_SNIPPET], env=env, cwd=ROOT, capture_output=True, text=True,
                         timeout=900)
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        log(out.stderr[-2000:])
        return {}
    return {int(k): v for k, v in json.loads(lines[-1]).items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sequence", type=int, default=0, help="1: also run the job-1683 chain.py check")
    p.add_argument("--ws", type=float, nargs="+", default=[1.84, 2.37, 2.9, 5])
    p.add_argument("--iters", type=int, default=6)
    args = p.parse_args()
    log("== S1 audit (static) ==")
    audit_ok, rows = static_audit()
    log(f"static audit ok={audit_ok}")

    log("== trigger rule ==")
    on, off = rule(False), rule(True)
    exp_on = {m: not (0 < m <= 8) for m in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 128, 512, 1024, 4096]}
    rule_ok = bool(on == exp_on and off and all(off.values()))
    log(f"default (True = END trigger): {on}\nenv=0: {off}\nrule ok={rule_ok}")

    log("== layer_check (real decoder layers: REF/SPLIT/SLEEP/NOPDL, M=1/2/4/8) ==")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "layer_check.py"), "--ckpt", args.ckpt,
                             "--batches", "1", "2", "4", "8"], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    last = ""
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.strip():
            last = line.strip()
    rc = proc.wait()
    try:
        lc = json.loads(last)
    except Exception:
        lc = {"passed": False, "error": f"no JSON (rc {rc})"}
    lc_ok = bool(rc == 0 and lc.get("passed"))
    k1 = lc.get("k1_calls") or {}
    hcffn_plain = k1.get("plain", 0) > 0  # hc_ffn K1 still launched wait-first

    seq = None
    seq_ok = True
    if args.sequence:
        # Job-1683 equivalent: chain.py --mode check (untimed sequence at the timed harness's W points, NaN-poisoned
        # outputs before every replay, bitwise every iteration, PDL0 arms, 1000-replay runs, both negative controls).
        log("== sequence check (chain.py --mode check, as job 1683) ==")
        cmd = [sys.executable, os.path.join(HERE, "chain.py"), "--mode", "check", "--ckpt", args.ckpt,
               "--batches", "1", "2", "4", "8", "--ws", *[str(w) for w in args.ws], "--iters", str(args.iters)]
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, text=True)
        last = ""
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.strip():
                last = line.strip()
        src = proc.wait()
        try:
            seq = json.loads(last)
        except Exception:
            seq = {"passed": False, "error": f"no JSON (rc {src})"}
        seq_ok = bool(src == 0 and seq.get("passed"))
        rep = seq.get("report") or {}
        seq = {"passed": seq_ok, "failures": seq.get("failures"),
               "per_M": {k: {"max_mismatch_bytes": max((v.get("max_mismatch_bytes_over_iters") or {0: 0}).values()),
                             "jit_in_capture": sum((v.get("jit_in_capture") or {}).values()),
                             "tree_k1_trigger_first": v.get("tree_k1_trigger_first"),
                             "iters": v.get("iters"),
                             **{x: v[x] for x in ("long_replays_ab", "long_replays_tree") if x in v}}
                         for k, v in rep.items() if k.startswith("M")},
               "negative_controls": rep.get("negative_controls")}

    passed = bool(audit_ok and rule_ok and lc_ok and hcffn_plain and seq_ok)
    print(json.dumps({
        "sequence_check": seq,
        "passed": passed,
        "compared": "S1 static audit of every MoE-AR-output and hc_attn-K1 consumer (file:line); trigger rule at "
                    "M=0..512 default and env=0; real-layer graphs SPLIT/SLEEP/NOPDL vs REF bitwise every replay",
        "audit_ok": audit_ok, "audit": rows, "rule_default": on, "rule_env0": off, "rule_ok": rule_ok,
        "layer_check": {k: lc.get(k) for k in ("passed", "summary", "moe_ar_trigger_calls", "k1_calls",
                                                "k1_order_per_m", "k1_order_ok",
                                                "k1_launches_m16", "k1_launches_eager", "sleep_cycles", "error")},
        "hc_ffn_k1_wait_first_launches": k1.get("plain", 0),
    }))


if __name__ == "__main__":
    main()
