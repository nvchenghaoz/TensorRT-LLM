"""Idea 127, one 2-GPU job (acceptance 4 + Stage 2 gain evidence).

Part A (timing, needs a PIX x16 pair): block_fused_inter.py, main FFN chain vs fused, co-run
with the chunked TP all-reduces, ABBA + A/A, n = 40, spans per rank and min over ranks.
Part B (non-timing, any pair): two trtllm-serve servers back to back on the same 2 GPUs, both with
this worktree's engine.json overlay on the benchmark's base settings:
  arm "main":   TLLM_GATE_UP_SWIGLU_SM120=0 -> GatedMLP runs main's chain verbatim
                (pinned cublas_mm gate_up -> silu_and_mul);
  arm "branch": the fused SM120 gate_up + SwiGLU kernel.
Each arm: startup (warmup + autotune), the adapters' untimed 2-step warmup request, then
park_portrait, perfume_bottle, golden_robot, park_portrait (seed 42, 1344x768, 124 f, 50 steps).
Compares MP4 sha256 branch vs main per request and the park repeat vs the first park; quotes the
'[gate_up_swiglu_sm120] rank R fused calls N' lines (both ranks), counts recompile / graph-break
lines in the timed requests (TORCH_LOGS=recompiles,graph_breaks), OOM lines, and the per-rank
peak-memory lines.
usage: python experiments/ascent/fusion127/combined_check.py <checkpoint>
Last stdout line: {"passed": bool, ...}
"""

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
WT = HERE.parents[3]
sys.path.insert(0, str(WT / "experiments" / "ascent-benchmark"))

from ab_common import (Heartbeat, Server, effective_server, ensure_ffmpeg, load_json, log,  # noqa: E402
                       run_bench, sha256_file, warmup, workload_params)

BASE_VISUAL_GEN_ARGS = {
    "attention_config": {"backend": "VANILLA"},
    "compilation_config": {"num_frames": [124], "resolutions": [[768, 1344]], "skip_warmup": False},
    "cuda_graph_config": {"enable": False},
    "parallel_config": {"parallel_vae_size": 2, "text_encoder_tp_size": 2, "tp_size": 2},
    "torch_compile_config": {"enable": True, "enable_autotune": True},
}
KEYS = ["park_portrait", "perfume_bottle", "golden_robot", "park_portrait"]
RC_RE = re.compile(r"[Rr]ecompil|[Gg]raph break")
MEM_RE = re.compile(r"peak|max_memory|memory_allocated|reserved", re.I)


def file_size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def run_block(ckpt):
    cmd = [sys.executable, str(HERE.parent / "block_fused_inter.py"), ckpt]
    log(f"[combined] $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("{"):
            last = line
    proc.wait()
    try:
        return json.loads(last) if last else {"passed": False, "error": f"rc={proc.returncode}"}
    except json.JSONDecodeError:
        return {"passed": False, "error": "unparsable"}


def run_arm(name, harness, server_cfg, prompts, common, extra_env, hb):
    env = {"TORCH_LOGS": "recompiles,graph_breaks"}
    env.update(extra_env)
    res = {"arm": name, "extra_env": env}
    with Server(harness, server_cfg, f"gateup127-{name}", extra_env=env) as srv:
        hb.phase = f"{name}: server startup"
        srv.start()
        res["startup_s"] = srv.startup_s
        res["warmup"] = warmup(srv, common, hb=hb)
        offs = {p: file_size(p) for p in srv.rank_logs}
        seed = int(prompts["seed"])
        bench = run_bench(srv, [{"key": k, "prompt": prompts["prompts"][k], "seed": seed} for k in KEYS],
                          f"{name}-timed", common, hb=hb)
        texts_all = {p.name: p.read_text(errors="replace") for p in srv.rank_logs}
        texts_timed = {}
        for p in srv.rank_logs:
            with open(p, "rb") as fh:
                fh.seek(offs[p])
                texts_timed[p.name] = fh.read().decode("utf-8", "replace")
    outs = []
    for rec, key in zip(bench["records"], KEYS):
        paths = rec.get("output_paths") or []
        outs.append({"key": key, "success": rec.get("success"),
                     "sha256": sha256_file(Path(paths[0])) if paths else None,
                     "server_e2e_s": rec.get("server_e2e")})
    res["outputs"] = outs
    res["fused_lines"] = {n: [l.strip()[-160:] for l in t.splitlines() if "[gate_up_swiglu_sm120]" in l]
                          for n, t in texts_all.items()}
    res["timed_recompile_or_graph_break_lines"] = {
        n: [l[:300] for l in t.splitlines() if RC_RE.search(l)][:20] for n, t in texts_timed.items()}
    res["memory_lines"] = {n: [l.strip()[-220:] for l in t.splitlines() if MEM_RE.search(l)
                               and ("GiB" in l or "GB" in l or "MiB" in l)][-6:]
                           for n, t in texts_all.items()}
    res["oom_lines"] = {
        n: {"all": sum("failed with OOM" in l for l in texts_all[n].splitlines()),
            "timed": sum("failed with OOM" in l for l in texts_timed[n].splitlines())}
        for n in texts_all}
    log(f"arm {name}: {json.dumps(res, default=str)[:4000]}")
    return res


def main():
    ckpt = sys.argv[1]
    summary = {"block": run_block(ckpt)}
    manifest = load_json("reference_manifest.json")
    harness = {
        "checkpoint": ckpt,
        "model_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
        "reference_store": manifest["store"],
        "server": {"visual_gen_args": BASE_VISUAL_GEN_ARGS, "env": {}},
        "workload_identity": {
            "workload": {"width": 1344, "height": 768, "num_frames": 124, "num_inference_steps": 50,
                         "gpus_per_instance": 2, "kind": "visual-gen", "runtime": "tensorrt-llm",
                         "task": "text-to-video"},
            "resolved_model": {"id": "MiniMaxAI/MiniMax-H3"},
        },
    }
    server_cfg, overlay = effective_server(harness)
    prompts = load_json("prompts.json")
    common = workload_params(harness)
    summary["engine_overlay"] = overlay.get("content")
    with Heartbeat() as hb:
        summary["ffmpeg"] = ensure_ffmpeg(manifest["store"])
        main_arm = run_arm("main", harness, server_cfg, prompts, common,
                           {"TLLM_GATE_UP_SWIGLU_SM120": "0"}, hb)
        branch_arm = run_arm("branch", harness, server_cfg, prompts, common, {}, hb)
    summary["main"] = main_arm
    summary["branch"] = branch_arm
    eq = [m["sha256"] is not None and m["sha256"] == b["sha256"]
          for m, b in zip(main_arm["outputs"], branch_arm["outputs"])]
    summary["sha256_equal_per_request"] = dict(zip([f"{i}:{k}" for i, k in enumerate(KEYS)], eq))
    summary["park_repeat_equal"] = {
        a["arm"]: a["outputs"][0]["sha256"] == a["outputs"][3]["sha256"] for a in (main_arm, branch_arm)}
    fused_ranks = [n for n, v in branch_arm["fused_lines"].items() if v]
    summary["fused_lines_ranks"] = {"branch": fused_ranks,
                                    "main": [n for n, v in main_arm["fused_lines"].items() if v]}
    rc_branch = sum(len(v) for v in branch_arm["timed_recompile_or_graph_break_lines"].values())
    rc_main = sum(len(v) for v in main_arm["timed_recompile_or_graph_break_lines"].values())
    summary["timed_recompile_or_graph_break_count"] = {"main": rc_main, "branch": rc_branch}
    summary["e2e_passed"] = bool(all(eq) and len(eq) == 4 and len(fused_ranks) >= 2
                                 and not summary["fused_lines_ranks"]["main"]
                                 and rc_branch <= rc_main
                                 and all(o["success"] for o in branch_arm["outputs"] + main_arm["outputs"]))
    blk = summary["block"]
    summary["block_passed"] = blk.get("passed") if not blk.get("wrong_pair") else "wrong_pair"
    summary["passed"] = bool(summary["e2e_passed"] and blk.get("passed"))
    summary["compared"] = ("MP4 sha256 of park, perfume, robot, park-repeat: branch (fused SM120 gate_up+"
                           "SwiGLU) vs main's chain (TLLM_GATE_UP_SWIGLU_SM120=0) in the same job on the "
                           "same GPUs; block co-run span main vs fused")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
