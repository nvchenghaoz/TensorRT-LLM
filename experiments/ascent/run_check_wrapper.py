"""Run the frozen check adapter (experiments/ascent-benchmark/run_check.py) in a worker job.

The adapter takes the controller's harness config JSON as argv[1]. A worker job does not receive
it, so this wrapper rebuilds it from what the controller's own proxy arms recorded (job 226: the
base server's visual_gen_args.yaml, execution.parameters, the reference store). The base server
settings are the original source's (VANILLA attention); the adapter overlays this worktree's
experiments/ascent-config/engine.json on them, exactly as in every controller run, and copies the
effective visual_gen_args.yaml into artifacts/logs/ for confirmation. Nothing in the adapter, its
prompts, references or tolerance rule is changed.

usage: python experiments/ascent/run_check_wrapper.py experiments/ascent-benchmark/run_check.py <checkpoint>
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_VISUAL_GEN_ARGS = {
    "attention_config": {"backend": "VANILLA"},
    "compilation_config": {"num_frames": [124], "resolutions": [[768, 1344]], "skip_warmup": False},
    "cuda_graph_config": {"enable": False},
    "parallel_config": {"parallel_vae_size": 2, "text_encoder_tp_size": 2, "tp_size": 2},
    "torch_compile_config": {"enable": True, "enable_autotune": True},
}


def main() -> int:
    adapter = Path(sys.argv[1])
    checkpoint = sys.argv[2]
    manifest = json.loads((adapter.parent / "reference_manifest.json").read_text())
    harness = {
        "checkpoint": checkpoint,
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
    print(f"[check-wrapper] harness: {json.dumps(harness)}", flush=True)
    proc = subprocess.run([sys.executable, str(adapter), json.dumps(harness)])
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
