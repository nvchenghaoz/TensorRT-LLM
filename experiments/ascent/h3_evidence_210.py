"""Re-emit job 210's completed check-adapter results as JSON evidence (no model run).

Job 210 ran the frozen check adapter for w14_alt and w14_c2 to completion, then crashed in its
own post-check judge step (ffmpeg not on PATH in the parent process), so the job is marked failed.
This reads the adapter_result.json files and both ranks' server logs it left on shared scratch and
prints one JSON line: per-prompt LPIPS/PSNR/audio SNR vs the frozen BF16 references, floor and
margin to floor+0.03, the same-server park_portrait repeat, the server-side cache_config, and the
per-rank Cache-DiT decisions (context ids stripped) compared across ranks.

Usage: python h3_evidence_210.py <job-210 artifacts dir> [tag ...]
"""

import json
import re
import sys
from pathlib import Path


def decisions(path: Path) -> list:
    out = []
    if path.is_file():
        for ln in path.read_text(errors="replace").splitlines():
            m = re.search(r"Cache-DiT decisions rank=\d+: context=\S+ (steps=\d+ cached=\d+ cached_steps=\[[^\]]*\])", ln)
            if m:
                out.append(m.group(1))
    return out


def main() -> None:
    root = Path(sys.argv[1])
    tags = sys.argv[2:] or ["w14_c2", "w14_alt"]
    report = {}
    ok = True
    for tag in tags:
        art = root / tag
        res = json.loads((art / "adapter_result.json").read_text())
        floor = {}
        for p in res.get("variation_pairs", []):
            if p.get("lpips") is not None:
                floor[p["key"]] = max(floor.get(p["key"], 0.0), p["lpips"])
        rows = []
        for p in res["comparisons"]["1"]["per_prompt"]:
            f = floor[p["key"]]
            rows.append({"key": p["key"], "lpips": round(p["lpips"], 4), "psnr": round(p["psnr"], 2),
                         "audio_snr_db": round(p.get("audio_snr_db", 0.0), 2), "floor": round(f, 4),
                         "bar": round(f + 0.03, 4), "margin": round(f + 0.03 - p["lpips"], 4)})
        r0 = decisions(art / "logs" / "server-check.log")
        r1 = decisions(art / "logs" / "server-check-rank1.log")
        yaml_text = (art / "logs" / "server-check.visual_gen_args.yaml").read_text()
        rep = res.get("repeated_execution", [])
        entry = {
            "valid": res.get("valid"),
            "errors": res.get("errors"),
            "per_prompt": rows,
            "fidelity_pass": all(r["margin"] >= 0 for r in rows),
            "repeated_execution_passed": res.get("repeated_execution_passed"),
            "repeat_vs_first_pass_bitwise": [r.get("vs_first_pass", {}).get("bitwise_equal") for r in rep],
            "server_visual_gen_args_yaml_cache_lines": [ln for ln in yaml_text.splitlines()
                                                        if ln.startswith(" ") or "cache" in ln][:30],
            "decisions_rank0": r0,
            "decisions_rank1": r1,
            "ranks_identical": bool(r0) and r0 == r1,
            "request_client_e2e_s": [round(r.get("client_e2e_s") or 0, 1) for r in res.get("requests", [])],
        }
        ok = ok and bool(entry["valid"]) and bool(entry["repeated_execution_passed"]) and \
            entry["fidelity_pass"] and entry["ranks_identical"]
        report[tag] = entry
        print(f"[evidence] {tag}: {json.dumps(entry)}", flush=True)
    print(json.dumps({"passed": ok, "source": str(root), "configs": report}), flush=True)


if __name__ == "__main__":
    main()
