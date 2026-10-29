"""Stale-tree probe for branch 150 (top lead event 5353 / area lead guard).

Prints, for this job's tree: git log -3, git cat-file -e <main>, merge-base --is-ancestor rc, and the
commits between the old base b8afb8d1 (and main) and HEAD that touch the FLUX.3 VAE decode path.
Last stdout line: JSON {"passed": true, ...} (a report; passed only means the probe ran).
"""

import json
import subprocess

MAIN = "0f76026a2d3a75fe5d3c56234a6ed1a6ffb1f200"
OLD = "b8afb8d1"
PATHS = ["tensorrt_llm/_torch/visual_gen/models/flux3/flux3_core/image_vae/",
         "tensorrt_llm/_torch/visual_gen/modules/vae/",
         "tensorrt_llm/_torch/visual_gen/models/flux3/pipeline_flux3.py",
         "experiments/ascent-config/engine.json"]


def run(*a):
    r = subprocess.run(["git", *a], capture_output=True, text=True)
    print("$ git", " ".join(a), "-> rc", r.returncode, flush=True)
    print(r.stdout.strip()[:6000], r.stderr.strip()[:500], flush=True)
    return {"rc": r.returncode, "out": r.stdout.strip()[:6000]}


res = {
    "log3": run("log", "--oneline", "-3"),
    "main_present": run("cat-file", "-e", MAIN + "^{commit}"),
    "old_present": run("cat-file", "-e", OLD + "^{commit}"),
    "is_ancestor_main": run("merge-base", "--is-ancestor", MAIN, "HEAD"),
    "is_ancestor_old": run("merge-base", "--is-ancestor", OLD, "HEAD"),
    "old_to_head_vae_paths": run("log", "--oneline", f"{OLD}..HEAD", "--", *PATHS),
    "main_to_head_vae_paths": run("log", "--oneline", f"{MAIN}..HEAD", "--", *PATHS),
    "head_to_main_vae_paths": run("log", "--oneline", f"HEAD..{MAIN}", "--", *PATHS),
    "diffstat_main_head_vae_paths": run("diff", "--stat", MAIN, "HEAD", "--", *PATHS),
}
print(json.dumps({"passed": True, "probe": res}), flush=True)
