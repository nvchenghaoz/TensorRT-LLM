"""Idea 68 non-timing 2-GPU serve check: SM120 cublas_mm tile pin vs main's F.linear, bitwise.

Two servers back to back in this job on the same 2 GPUs, both with this worktree's engine.json
overlay on the benchmark's base settings:
  arm "main":   TLLM_H3_SM120_CUBLAS_PIN=0 -> the 4 DiT Linears run F.linear (main's path verbatim;
                the LUT only affects trtllm::cublas_mm);
  arm "branch": the pin (use_custom_cublas_mm + sm120_bf16_algo_list).
Each arm: startup (warmup + autotune), the adapters' untimed 2-step warmup request, then
park_portrait, perfume_bottle, golden_robot, park_portrait (seed 42, 1344x768, 124 frames, 50 steps).
Compares MP4 sha256 branch vs main per request and the park repeat vs first pass, quotes the
"[MiniMax-H3 SM120 cublas pin]" line on both ranks, counts recompile / graph-break lines after warmup
(TORCH_LOGS=recompiles,graph_breaks) and 'failed with OOM' lines per rank.
usage: python experiments/ascent/gemm_pin/e2e_check.py <checkpoint> {worktree}/experiments/ascent-benchmark
Last stdout line: {"passed": bool, ...}
"""

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
WT = HERE.parents[3]
# The frozen benchmark rides the job under {worktree}; pass it as argv[2].
BENCH = Path(sys.argv[2]) if len(sys.argv) > 2 else WT / "experiments" / "ascent-benchmark"
sys.path.insert(0, str(BENCH))

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
PIN_TAG = "[MiniMax-H3 SM120 cublas pin]"


def file_size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def topo():
    out = {}
    for cmd in (["nvidia-smi", "--query-gpu=index,pci.bus_id,uuid,pcie.link.width.current", "--format=csv,noheader"],
                ["nvidia-smi", "topo", "-m"]):
        try:
            out[" ".join(cmd[:2])] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # noqa
            out[" ".join(cmd[:2])] = repr(e)
    return out


def run_arm(name, harness, server_cfg, prompts, common, extra_env, hb):
    env = {"TORCH_LOGS": "recompiles,graph_breaks"}
    env.update(extra_env)
    res = {"arm": name, "extra_env": env}
    with Server(harness, server_cfg, f"gemmpin-{name}", extra_env=env) as srv:
        hb.phase = f"{name}: server startup"
        srv.start()
        res["startup_s"] = srv.startup_s
        res["warmup"] = warmup(srv, common, hb=hb)
        offs = {p: file_size(p) for p in srv.rank_logs}
        seed = int(prompts["seed"])
        bench = run_bench(srv, [{"key": k, "prompt": prompts["prompts"][k], "seed": seed} for k in KEYS],
                          f"{name}-req", common, hb=hb)
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
                     "path": paths[0] if paths else None, "server_e2e_s": rec.get("server_e2e")})
    res["outputs"] = outs
    res["pin_lines"] = {n: [l[-300:] for l in t.splitlines() if PIN_TAG in l][:2] for n, t in texts_all.items()}
    res["timed_recompile_or_graph_break_lines"] = {
        n: [l[:300] for l in t.splitlines() if RC_RE.search(l)][:20] for n, t in texts_timed.items()}
    res["all_recompile_or_graph_break_count"] = {
        n: sum(bool(RC_RE.search(l)) for l in t.splitlines()) for n, t in texts_all.items()}
    res["oom_lines"] = {
        n: {"all": sum("failed with OOM" in l for l in texts_all[n].splitlines()),
            "timed": sum("failed with OOM" in l for l in texts_timed[n].splitlines())}
        for n in texts_all}
    log(f"arm {name}: {json.dumps(res, default=str)[:3000]}")
    return res


def main():
    ckpt = sys.argv[1]
    log(f"topology: {json.dumps(topo())}")
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
    summary = {"engine_overlay": overlay.get("content")}
    with Heartbeat() as hb:
        summary["ffmpeg"] = ensure_ffmpeg(manifest["store"])
        main_arm = run_arm("main", harness, server_cfg, prompts, common,
                           {"TLLM_H3_SM120_CUBLAS_PIN": "0"}, hb)
        branch_arm = run_arm("branch", harness, server_cfg, prompts, common, {}, hb)
    summary["main"] = main_arm
    summary["branch"] = branch_arm
    eq = [m["sha256"] is not None and m["sha256"] == b["sha256"]
          for m, b in zip(main_arm["outputs"], branch_arm["outputs"])]
    summary["sha256_equal_per_request"] = dict(zip([f"{i}:{k}" for i, k in enumerate(KEYS)], eq))
    summary["park_repeat_equal"] = {
        a["arm"]: a["outputs"][0]["sha256"] == a["outputs"][3]["sha256"] for a in (main_arm, branch_arm)}

    def pin_state(arm, want):
        lines = arm["pin_lines"]
        return len(lines) == 2 and all(any(want in l for l in v) for v in lines.values())

    # main arm: the pin is off, so no line at all; branch: all four True on both ranks
    summary["pin_line"] = {
        "main_absent": all(not v for v in main_arm["pin_lines"].values()),
        "branch_on_both_ranks": pin_state(branch_arm, "[True, True, True, True]"),
    }
    rc_branch = sum(len(v) for v in branch_arm["timed_recompile_or_graph_break_lines"].values())
    rc_main = sum(len(v) for v in main_arm["timed_recompile_or_graph_break_lines"].values())
    summary["timed_recompile_or_graph_break_count"] = {"main": rc_main, "branch": rc_branch}
    summary["all_recompile_or_graph_break_count"] = {
        "main": sum(main_arm["all_recompile_or_graph_break_count"].values()),
        "branch": sum(branch_arm["all_recompile_or_graph_break_count"].values())}
    summary["passed"] = bool(all(eq) and len(eq) == 4 and all(summary["park_repeat_equal"].values())
                             and all(summary["pin_line"].values())
                             and rc_branch <= rc_main
                             and summary["all_recompile_or_graph_break_count"]["branch"]
                             <= summary["all_recompile_or_graph_break_count"]["main"]
                             and all(o["success"] for o in branch_arm["outputs"] + main_arm["outputs"]))
    summary["compared"] = ("MP4 sha256 of park, perfume, robot, park-repeat: branch (SM120 cublas_mm tile pin) vs "
                           "main's F.linear (TLLM_H3_SM120_CUBLAS_PIN=0) in the same job on the same GPUs")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
