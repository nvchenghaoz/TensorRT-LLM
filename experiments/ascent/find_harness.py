"""Recover the frozen harness_config JSON that the controller embedded in an earlier serving job.

Reads runner/jobs/<id>/request.json, walks every string, decodes base64 / zlib payloads, and returns
the first JSON object that has "server" and "workload_identity". Used only to run the frozen check
adapter (experiments/ascent-benchmark/run_check.py) in a worker job with the same harness config.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import zlib
from pathlib import Path

LEASE_JOBS = Path("/home/scratch.chengzhang/.auto-dev/leases/112dd6c0ef8e4d3d9b4c985b5bedcf3c/runner/jobs")


def _try_json(s: str):
    s = s.strip()
    if not s.startswith("{"):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _is_harness(o) -> bool:
    return isinstance(o, dict) and "server" in o and "workload_identity" in o


def _search(obj, depth=0):
    if depth > 8:
        return None
    if _is_harness(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = _search(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _search(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, str):
        j = _try_json(obj)
        if j is not None:
            r = _search(j, depth + 1)
            if r is not None:
                return r
        for m in re.finditer(r"[A-Za-z0-9+/=]{200,}", obj):
            try:
                raw = base64.b64decode(m.group(0))
            except Exception:
                continue
            for dec in (lambda b: b, zlib.decompress):
                try:
                    txt = dec(raw).decode("utf-8")
                except Exception:
                    continue
                j = _try_json(txt)
                if j is not None:
                    r = _search(j, depth + 1)
                    if r is not None:
                        return r
                r = _search(txt, depth + 1) if txt != obj else None
                if r is not None:
                    return r
    return None


def find(job_dir_ids: list[str]) -> dict:
    for jid in job_dir_ids:
        p = LEASE_JOBS / jid / "request.json"
        if not p.is_file():
            continue
        h = _search(json.loads(p.read_text()))
        if h is not None:
            return h
    raise SystemExit("harness not found")


if __name__ == "__main__":
    h = find(sys.argv[1:] or ["4285f767d2ff4900a5b55c2c9abf8847"])
    print(json.dumps(h, indent=1)[:6000])
