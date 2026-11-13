"""One 2-GPU correctness job for idea 105 (any GPU pair; nothing here is timed).

1. block_pair2.py --mode check: two real TP2 blocks, main path / op-miss / prefetch-hit arms,
   bitwise per iteration vs main's path, inputs read-only, hits/misses per pair, no recompiles
   after warmup, live peek semantics.
2. e2e_check.py: the frozen check adapter (3 brief prompts + same-server park_portrait repeat) on
   the committed engine.json, with per-rank [adaln_prefetch] hit/miss lines, OOM lines (warmup vs
   timed), peak memory, recompile/graph-break lines and output sha256s.
Last stdout line: {"passed": block and e2e passed, "block": {...}, "e2e": {...}}
Usage: python combined_check.py <checkpoint>
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd):
    print(f"[combined] $ {' '.join(cmd)}", flush=True)
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


def main():
    ckpt = sys.argv[1]
    block = run([sys.executable, str(HERE / "block_pair2.py"), ckpt, "--mode", "check"])
    b = {"passed": block.get("passed"),
         "ranks": [{k: r.get(k) for k in ("rank", "check", "passed", "chunks", "dedicated_pg")}
                   for r in block.get("ranks", [])]}
    e2e = run([sys.executable, str(HERE / "e2e_check.py"), ckpt])
    print(json.dumps({"passed": bool(block.get("passed")) and bool(e2e.get("passed")),
                      "block": b, "e2e": e2e}, default=str), flush=True)


if __name__ == "__main__":
    main()
