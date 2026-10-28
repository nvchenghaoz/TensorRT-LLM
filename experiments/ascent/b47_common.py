"""Shared helpers for branch 47 scripts: find and load the saved served tensors."""
import glob
import os
import time

ROOT = "/home/scratch.chengzhang/trtllm-dev-flux3-klein-20261027-125409/runner/slots"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_dump():
    env = os.environ.get("B47_DUMP")
    if env and os.path.isdir(env):
        return env
    cands = [d for d in glob.glob(f"{ROOT}/*/jobs/*/artifacts/b47_dump")
             if os.path.exists(os.path.join(d, "requests.json"))]
    cands.sort(key=os.path.getmtime)
    return cands[-1] if cands else None


def load_images(dump):
    """[(seq, kind, cpu_src, cuda_tensor_in_served_layout, stride)]"""
    import json

    import torch
    reqs = json.load(open(os.path.join(dump, "requests.json")))
    kinds = {r["seq"]: r["kind"] for r in reqs}
    out = []
    for p in sorted(glob.glob(os.path.join(dump, "img_*.pt"))):
        d = torch.load(p)
        g = torch.empty_strided(d["shape"], d["stride"], dtype=torch.uint8, device="cuda")
        g.copy_(d["data"].cuda())
        assert list(g.stride()) == d["stride"]
        out.append((d["seq"], kinds.get(d["seq"]), d["data"], g, d["stride"]))
    return out
