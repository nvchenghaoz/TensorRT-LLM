"""Idea 211 correctness job (branch tree, untimed). Runs, each as a subprocess:

 1. k1fast_stage1.py (K1_SKIP_TIMING=1): 8192 tokens x 2 seeds, P in {480, 224, 96, 256}, num_warps {4, 8, 16}:
    fast K1 bitwise == main's K1 (fast=False) and == mhc_fused_hc's layer_input, with PDL+TRIGGER_FIRST and with PDL
    off; fast K1 sum-of-squares totals == k1_probe3.emulate() and == main K1's; rstd equal; PTX: no st.global
    before griddepcontrol.wait.
 2. k1fast_chain_check.py: mhc_split/chain.py --mode check through the real DeepseekV41DecoderLayer at M=1/2/4/8,
    graphs REF / SPLIT / SLEEP (aux sleep 800k cycles) / NOPDL, 8 replays with new inputs and NaN-poisoned
    outputs, bitwise vs REF; no K1 (fast or main) Triton compile inside capture.
 3. router_k1fold/check_module.py with TRTLLM_MHC_K1_FAST=1 and with TRTLLM_MHC_K1_FAST=0.
 4. moe_ar_trigger/chain.py --mode check (claim A: hc_attn K1 trigger-first through the fast K1): M=1/2/4/8, arms
    base/a/ab/tree x W {2.9, 10} us + PDL0, 8 iterations bitwise vs eager reference and vs base graph, 1000
    back-to-back replays at M=1/8, negative controls k1nowait (fast K1 trigger + skip wait) and lowmskip caught.
Usage: k1fast_check_all.py <ckpt>
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, timeout, env=None):
    print("=== " + " ".join(cmd) + (f"  env {env}" if env else ""), flush=True)
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        s = line.strip()
        if s.startswith("{"):
            try:
                last = json.loads(s)
            except json.JSONDecodeError:
                pass
    rc = p.wait(timeout=timeout)
    return rc, last


def brief(d, keys):
    return {k: (d or {}).get(k) for k in keys}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    py = sys.executable
    res = {}
    rc, o = run([py, os.path.join(HERE, "k1fast_stage1.py")], 1800, {"K1_SKIP_TIMING": "1"})
    res["stage1_bitwise"] = {"rc": rc, "passed": bool(o and o.get("passed")), "ptx_audit": (o or {}).get("ptx_audit"),
                             "bitwise": (o or {}).get("bitwise")}
    rc, o = run([py, os.path.join(HERE, "k1fast_chain_check.py"), ckpt], 2400)
    res["mhc_chain_check"] = {"rc": rc, "passed": bool(o and o.get("passed")),
                              **brief(o, ("k1_fast", "chain_check_passed", "audit", "summary"))}
    for fast in ("1", "0"):
        rc, o = run([py, os.path.join(HERE, "..", "router_k1fold", "check_module.py")], 2400,
                    {"TRTLLM_MHC_K1_FAST": fast})
        res[f"router_k1fold_check_module_FAST{fast}"] = {"rc": rc, "passed": bool(o and o.get("passed")),
                                                         **brief(o, ("failures", "skip_wait_bad_iterations"))}
    rc, o = run([py, os.path.join(HERE, "..", "moe_ar_trigger", "chain.py"), "--mode", "check", "--ckpt", ckpt,
                 "--batches", "1", "2", "4", "8", "--ws", "2.9", "10", "--P", "480"], 3000)
    res["moe_ar_trigger_check"] = {"rc": rc, "passed": bool(o and o.get("passed")),
                                   **brief(o, ("failures",)),
                                   "negative_controls": ((o or {}).get("report") or {}).get("negative_controls")}
    passed = all(v["passed"] for v in res.values())
    print(json.dumps({"passed": passed,
                      "compared": "fast K1 vs main K1 (TRTLLM_MHC_K1_FAST=0 kernel) and mhc_fused_hc layer_input, "
                                  "bitwise; real decoder-layer chain REF/SPLIT/SLEEP/NOPDL; router fold module "
                                  "FAST on/off; hc_attn trigger-first chain vs eager reference",
                      "results": res}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
