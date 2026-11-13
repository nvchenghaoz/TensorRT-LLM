"""Non-timing 2-GPU end-to-end check for idea 106 Stage 1 (run as a base/branch pair, any GPU pair).

Uses the frozen benchmark's own server and load generator (experiments/ascent-benchmark/ab_common:
Server, warmup, run_bench) with this tree's engine.json overlaid on the original-source base settings,
exactly as run_check.py does, and records what a sha256 comparison between main and the branch needs:

  cache_on  : engine.json as committed (mask F Cache-DiT, FP8 attention onset 14); untimed 2-step warmup,
              then park_portrait, perfume_bottle, golden_robot (seed 42, 1344x768, 124 frames, 50 steps),
              then park_portrait again in the same server (repeat).
  cache_off : engine.json without cache_config (every forward dense); warmup, then park_portrait.

Per output: MP4 sha256 and server_e2e_s. Per rank: every log line containing 'OOM', the
'[step_precision] ... peak_memory <phase>' lines (max_allocated / max_reserved after denoise and VAE),
the step gate line and the chunk-residual line (present only on the branch).
Last stdout line: {"passed": ..., "sha256": {...}, ...}; passed = every request succeeded and the
park repeat is sha256-equal to the first pass. Main-vs-branch equality is read across the pair's arms.

usage: python experiments/ascent/tp_overlap/stage1_e2e_check.py experiments/ascent-benchmark/run_check.py <checkpoint>
(the frozen benchmark files ride a job only when an adapter path is named in its command)
"""

import copy
import json
import sys
from pathlib import Path

BENCH = Path(sys.argv[1]).resolve().parent
sys.path.insert(0, str(BENCH))
if not (BENCH / "ab_common.py").is_file():
    _root = Path.cwd() / "experiments"
    print(json.dumps({"passed": False, "error": f"no ab_common.py in {BENCH}", "cwd": str(Path.cwd()),
                      "experiments": sorted(p.name for p in _root.iterdir()) if _root.is_dir() else None,
                      "bench_listing": sorted(p.name for p in BENCH.iterdir()) if BENCH.is_dir() else None}),
          flush=True)
    sys.exit(1)

from ab_common import (Heartbeat, Server, effective_server, ensure_ffmpeg, log, run_bench,  # noqa: E402
                       sha256_file, warmup, workload_params)

BASE_VISUAL_GEN_ARGS = {  # the original source's server settings (run_check_wrapper.py, job 226)
    "attention_config": {"backend": "VANILLA"},
    "compilation_config": {"num_frames": [124], "resolutions": [[768, 1344]], "skip_warmup": False},
    "cuda_graph_config": {"enable": False},
    "parallel_config": {"parallel_vae_size": 2, "text_encoder_tp_size": 2, "tp_size": 2},
    "torch_compile_config": {"enable": True, "enable_autotune": True},
}
KEYS = ["park_portrait", "perfume_bottle", "golden_robot"]


def scan_logs(paths):
    out = {}
    for p in paths:
        p = Path(p)
        lines = {"oom": [], "peak_memory": [], "step_gate": [], "chunk_residual": []}
        try:
            text = p.read_text(errors="replace")
        except OSError as exc:
            out[p.name] = {"error": str(exc)}
            continue
        for ln in text.splitlines():
            s = ln.strip()[-400:]
            if "OOM" in ln:
                lines["oom"].append(s)
            if "peak_memory" in ln:
                lines["peak_memory"].append(s)
            if "[step_precision]" in ln and "attention_bf16_forwards" in ln:
                lines["step_gate"].append(s)
            if "chunk_residual" in ln or "per-chunk final gated residual" in ln:
                lines["chunk_residual"].append(s)
        lines["oom"] = lines["oom"][:40]
        lines["step_gate"] = lines["step_gate"][:4]
        lines["chunk_residual"] = lines["chunk_residual"][:4]
        out[p.name] = lines
    return out


def main():
    ckpt = sys.argv[2]
    prompts = json.loads((BENCH / "prompts.json").read_text())
    manifest = json.loads((BENCH / "reference_manifest.json").read_text())
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
    common = workload_params(harness)
    seed = int(prompts["seed"])
    server_on, overlay = effective_server(harness)
    server_off = copy.deepcopy(server_on)
    server_off["visual_gen_args"].pop("cache_config", None)
    result = {"engine_overlay_sha256": overlay.get("sha256"), "arms": {}}
    ok = True
    with Heartbeat() as hb:
        result["ffmpeg"] = ensure_ffmpeg(manifest["store"])
        arms = [("main_cache_on", "0", server_on, KEYS, ["park_portrait"]),
                ("branch_cache_on", "1", server_on, KEYS, ["park_portrait"]),
                ("main_cache_off", "0", server_off, ["park_portrait"], []),
                ("branch_cache_off", "1", server_off, ["park_portrait"], [])]
        if len(sys.argv) > 3:  # optional subset, e.g. branch_cache_on,branch_cache_off
            sel = sys.argv[3].split(",")
            arms = [a for a in arms if a[0] in sel]
        for name, switch, cfg, keys, repeat in arms:
            # TLLM_H3_TP_CHUNK_RESIDUAL=0 runs main's code path unchanged (compiled cat + gated residual).
            arm = {"visual_gen_args": cfg["visual_gen_args"], "TLLM_H3_TP_CHUNK_RESIDUAL": switch}
            srv = Server(harness, cfg, name, extra_env={"TLLM_H3_TP_CHUNK_RESIDUAL": switch})
            try:
                with srv:
                    hb.phase = f"{name}: server startup"
                    srv.start()
                    arm["startup_s"] = round(srv.startup_s, 1)
                    arm["warmup"] = warmup(srv, common, hb=hb)
                    benches = [("pass1", run_bench(srv, [{"key": k, "prompt": prompts["prompts"][k],
                                                          "seed": seed} for k in keys],
                                                   f"{name}_pass1", common, hb=hb))]
                    if repeat:
                        benches.append(("repeat", run_bench(
                            srv, [{"key": k, "prompt": prompts["prompts"][k], "seed": seed} for k in repeat],
                            f"{name}_repeat", common, hb=hb)))
                reqs = []
                for tag, b in benches:
                    for rec in b["records"]:
                        paths = rec.get("output_paths") or []
                        r = {"pass": tag, "key": rec.get("key"), "success": bool(rec.get("success")),
                             "server_e2e_s": rec.get("server_e2e"),
                             "server_denoise_s": rec.get("server_denoise")}
                        if paths:
                            r["sha256"] = sha256_file(Path(paths[0]))
                        else:
                            r["error"] = str(rec.get("error"))[:300]
                        reqs.append(r)
                        ok = ok and r["success"] and "sha256" in r
                arm["requests"] = reqs
                first = {r["key"]: r.get("sha256") for r in reqs if r["pass"] == "pass1"}
                rep = [r for r in reqs if r["pass"] == "repeat"]
                arm["repeat_bitwise"] = [r.get("sha256") == first.get(r["key"]) for r in rep]
                ok = ok and all(arm["repeat_bitwise"])
            except Exception as exc:  # report and keep going to the next arm
                import traceback

                arm["error"] = traceback.format_exc()[-3000:]
                ok = False
            arm["logs"] = scan_logs(srv.rank_logs)
            result["arms"][name] = arm
            log(f"arm {name}: {json.dumps({k: v for k, v in arm.items() if k != 'logs'}, default=str)[:3000]}")
    result["sha256"] = {f"{a}/{r['pass']}/{r['key']}": r.get("sha256")
                        for a, arm in result["arms"].items() for r in arm.get("requests", [])}
    eq = {}
    for k, v in result["sha256"].items():
        if k.startswith("branch_"):
            mk = "main_" + k[len("branch_"):]
            if mk in result["sha256"]:
                eq[k[len("branch_"):]] = v is not None and v == result["sha256"][mk]
    result["branch_eq_main"] = eq
    ok = ok and all(eq.values())
    result["passed"] = ok
    result["compared"] = ("MP4 sha256 per output, branch (TLLM_H3_TP_CHUNK_RESIDUAL=1) vs main's code path "
                          "(=0) in the same job, cache on (3 prompts) and off (park); park repeat sha256 vs "
                          "first pass in the same server; per-rank OOM lines and peak_memory lines")
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
