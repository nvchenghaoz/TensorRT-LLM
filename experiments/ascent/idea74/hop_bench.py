"""Idea 74 timing: coordinator -> rank0 reference hop, A/A2/B arms in one job (method 544/646).

Producer = this process, consumer = a spawned child that receives the pickled request over
a Pipe (as over ZMQ). Arms:
  A, A2 : main's DiffusionRequest.refs_to_shm / refs_from_shm (cd622508), copied verbatim
          as two separate function objects;
  B     : this tree's DiffusionRequest.refs_to_shm / refs_from_shm (in the base tree's
          job, B is main's code again, so B/A must include 1 there).
Per iteration, per case (i2i ref A, i2i ref B, t2i = no reference), the three arms run in
one of the 6 permutations (iteration % 6), so each arm follows each other arm equally often.
Per request time = producer refs_to_shm (parent) + consumer refs_from_shm (child).
Reports medians, (p90-p10)/median, and B/A, A2/A ratios of medians with bootstrap CI95.
"""

import hashlib
import itertools
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "idea63"))

WARMUP = 10
ITERS = int(os.environ.get("IDEA74_ITERS", "240"))
ARMS = ("A", "A2", "B")
PERMS = list(itertools.permutations(ARMS))


def log(*a):
    print(f"[idea74-bench {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- main's code, verbatim (x2)
def _main_refs_to_shm(self) -> None:
    import torch
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer

    if self.params is None:
        return
    self.ref_handles = handles = []
    for slot in ("image_reference", "video_reference", "audio_reference"):
        for index, ref in enumerate(getattr(self.params, slot, None) or []):
            # A read-only view: the one copy happens inside from_tensor(),
            # which is what moves the payload into shared memory.
            buffer = torch.frombuffer(ref.content, dtype=torch.uint8)
            handles.append(
                {
                    "slot": slot,
                    "index": index,
                    "handle": SharedTensorContainer.from_tensor(buffer).dump_to_dict(),
                }
            )
            ref.content = b""
    if not handles:
        self.ref_handles = None


def _main_refs_from_shm(self) -> None:
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer

    failures = []
    for entry in self.ref_handles or []:
        try:
            ref = getattr(self.params, entry["slot"])[entry["index"]]
            container = SharedTensorContainer.from_dict(entry["handle"])
            ref.content = container.get_local_view().numpy().tobytes()
        except Exception as exc:
            failures.append(f"{entry['slot']}[{entry['index']}]: {exc}")
    self.ref_handles = None
    if failures:
        raise RuntimeError(
            "failed to restore reference payloads from shared memory: " + "; ".join(failures)
        )


def _copy_fn(f):
    return types.FunctionType(f.__code__, f.__globals__, f.__name__ + "_A2", f.__defaults__,
                              f.__closure__)


def arm_fns():
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
    return {"A": (_main_refs_to_shm, _main_refs_from_shm),
            "A2": (_copy_fn(_main_refs_to_shm), _copy_fn(_main_refs_from_shm)),
            "B": (DiffusionRequest.refs_to_shm, DiffusionRequest.refs_from_shm)}


def make_request(content):
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
    params = types.SimpleNamespace(
        seed=1, video_reference=None, audio_reference=None,
        image_reference=None if content is None else [types.SimpleNamespace(content=content,
                                                                             format="bytes")])
    return DiffusionRequest(request_id=0, prompt=["p"], params=params)


def child(conn):
    import torch  # noqa: F401
    fns = arm_fns()
    while True:
        msg = conn.recv()
        if msg is None:
            break
        arm, req = msg
        t0 = time.perf_counter()
        fns[arm][1](req)
        t1 = time.perf_counter()
        refs = req.params.image_reference or []
        conn.send(((t1 - t0) * 1e3, hashlib.sha256(refs[0].content).hexdigest() if refs else ""))


def boot_ratio(num, den, reps=2000, seed=0):
    rng = random.Random(seed)
    n = len(num)
    out = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(statistics.median(num[i] for i in idx) / statistics.median(den[i] for i in idx))
    out.sort()
    return [round(out[int(0.025 * reps)], 4), round(out[int(0.975 * reps) - 1], 4)]


def main():
    from pngtest import find_refs
    import torch  # noqa: F401
    refs = find_refs()
    shas = {k: hashlib.sha256(v).hexdigest() for k, v in refs.items()}
    fns = arm_fns()
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    p = ctx.Process(target=child, args=(b,), daemon=True)
    p.start()
    cases = {"refA": refs["A"], "refB": refs["B"], "t2i": None}
    t = {(c, arm): [] for c in cases for arm in ARMS}
    split = {(c, arm): ([], []) for c in cases for arm in ARMS}
    bad = 0
    for it in range(WARMUP + ITERS):
        perm = PERMS[it % len(PERMS)]
        for c, content in cases.items():
            for arm in perm:
                req = make_request(content)
                t0 = time.perf_counter()
                fns[arm][0](req)
                t1 = time.perf_counter()
                a.send((arm, req))
                cms, h = a.recv()
                bad += h != (shas[c[-1]] if content is not None else "")
                if it >= WARMUP:
                    pm = (t1 - t0) * 1e3
                    t[(c, arm)].append(pm + cms)
                    split[(c, arm)][0].append(pm)
                    split[(c, arm)][1].append(cms)
        if it % 40 == 0:
            log("iter", it, "load", open("/proc/loadavg").read().split()[:3])
    a.send(None)
    p.join()
    # i2i composite per iteration: mean of the ref A and ref B requests (fixed mix).
    for arm in ARMS:
        t[("i2i", arm)] = [(x + y) / 2 for x, y in zip(t[("refA", arm)], t[("refB", arm)])]
    out = {}
    for c in ("i2i", "refA", "refB", "t2i"):
        row = {}
        for arm in ARMS:
            v = sorted(t[(c, arm)])
            med = statistics.median(v)
            row[arm] = {"median_ms": round(med, 4),
                        "spread": round((v[len(v) * 9 // 10] - v[len(v) // 10]) / med, 4)}
            if (c, arm) in split:
                row[arm]["producer_ms"] = round(statistics.median(split[(c, arm)][0]), 4)
                row[arm]["consumer_ms"] = round(statistics.median(split[(c, arm)][1]), 4)
        A, A2, B = (t[(c, x)] for x in ARMS)
        row["B/A"] = round(statistics.median(B) / statistics.median(A), 4)
        row["B/A_ci95"] = boot_ratio(B, A)
        row["A2/A"] = round(statistics.median(A2) / statistics.median(A), 4)
        row["A2/A_ci95"] = boot_ratio(A2, A, seed=1)
        out[c] = row
        log(c, json.dumps(row))
    left = sorted(f for f in os.listdir("/dev/shm") if f.startswith(f"trtllm_vgref_{os.getpid()}_"))
    batch = {"i2i-2048": {"step_ms": out["i2i"]["B"]["median_ms"],
                          "noise_floor": max(out["i2i"][x]["spread"] for x in ARMS)},
             "t2i-2048": {"step_ms": out["t2i"]["B"]["median_ms"],
                          "noise_floor": max(out["t2i"][x]["spread"] for x in ARMS)}}
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "sha_mismatch": bad,
                      "leftover": left, "iters": ITERS, "results": out}))


if __name__ == "__main__":
    main()
