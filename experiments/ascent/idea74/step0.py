"""Idea 74 STEP 0: coordinator -> rank0 reference hop, two real processes.

Producer = this process, consumer = a spawned child that gets the handle dict over a Pipe.
Per iteration and per ref (A, B), every variant runs once in rotated order:
  torch      : main's path (SharedTensorContainer.from_tensor(frombuffer).dump_to_dict();
               consumer from_dict + get_local_view().numpy().tobytes())
  torch_new  : torch.UntypedStorage._new_shared(n) + copy_, then the same handle path
  mmap_pop   : /dev/shm file, ftruncate, mmap MAP_SHARED|MAP_POPULATE, slice assign;
               consumer open, unlink, mmap MAP_POPULATE, mm[:n]
  mmap_nopop : same without MAP_POPULATE
  write      : /dev/shm file, os.write loop; consumer open, unlink, os.read loop
Each consumer checks sha256 of the restored bytes against the input (untimed).
Prints medians (ms) of producer, consumer and total per variant and ref.
"""

import hashlib
import json
import mmap
import multiprocessing as mp
import os
import statistics
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "idea63"))

WARMUP = 10
ITERS = 200
VARIANTS = ["torch", "torch_new", "mmap_pop", "mmap_nopop", "write"]
POP = getattr(mmap, "MAP_POPULATE", 0)


def log(*a):
    print(f"[idea74 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def produce(variant, content):
    import torch
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
    n = len(content)
    if variant == "torch":
        buf = torch.frombuffer(content, dtype=torch.uint8)
        return {"torch": SharedTensorContainer.from_tensor(buf).dump_to_dict()}
    if variant == "torch_new":
        st = torch.UntypedStorage._new_shared(n)
        t = torch.empty(0, dtype=torch.uint8).set_(st)
        t.copy_(torch.frombuffer(content, dtype=torch.uint8))
        return {"torch": SharedTensorContainer.from_tensor(t).dump_to_dict()}
    name = f"/dev/shm/trtllm_vgref_{os.getpid()}_{uuid.uuid4().hex}"
    fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        if variant == "write":
            mv = memoryview(content)
            off = 0
            while off < n:
                off += os.write(fd, mv[off:])
        else:
            os.ftruncate(fd, n)
            flags = mmap.MAP_SHARED | (POP if variant == "mmap_pop" else 0)
            with mmap.mmap(fd, n, flags=flags) as mm:
                mm[:] = content
    finally:
        os.close(fd)
    return {"posix_shm": name, "size": n}


def consume(variant, h):
    if "torch" in h:
        from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
        c = SharedTensorContainer.from_dict(h["torch"])
        return c.get_local_view().numpy().tobytes()
    name, n = h["posix_shm"], h["size"]
    fd = os.open(name, os.O_RDONLY)
    try:
        os.unlink(name)
        if variant == "write":
            parts = []
            got = 0
            while got < n:
                b = os.read(fd, n - got)
                if not b:
                    raise EOFError(name)
                parts.append(b)
                got += len(b)
            return parts[0] if len(parts) == 1 else b"".join(parts)
        flags = mmap.MAP_SHARED | (POP if variant == "mmap_pop" else 0)
        with mmap.mmap(fd, n, flags=flags, prot=mmap.PROT_READ) as mm:
            return mm[:n]
    finally:
        os.close(fd)


def child(conn):
    import torch  # noqa: F401  (match rank0, which has torch loaded)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        variant, h, want = msg
        t0 = time.perf_counter()
        data = consume(variant, h)
        t1 = time.perf_counter()
        ok = type(data) is bytes and hashlib.sha256(data).hexdigest() == want
        conn.send(((t1 - t0) * 1e3, ok))


def main():
    from pngtest import find_refs
    import torch  # noqa: F401
    refs = find_refs()
    log("refs", {k: len(v) for k, v in refs.items()}, "MAP_POPULATE", POP)
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    p = ctx.Process(target=child, args=(b,))
    p.start()
    shas = {k: hashlib.sha256(v).hexdigest() for k, v in refs.items()}
    res = {(v, r): {"prod": [], "cons": []} for v in VARIANTS for r in refs}
    bad = 0
    for it in range(WARMUP + ITERS):
        order = VARIANTS[it % len(VARIANTS):] + VARIANTS[:it % len(VARIANTS)]
        for r, content in refs.items():
            for v in order:
                t0 = time.perf_counter()
                h = produce(v, content)
                t1 = time.perf_counter()
                a.send((v, h, shas[r]))
                cms, ok = a.recv()
                bad += not ok
                if it >= WARMUP:
                    res[(v, r)]["prod"].append((t1 - t0) * 1e3)
                    res[(v, r)]["cons"].append(cms)
        if it % 50 == 0:
            log("iter", it)
    a.send(None)
    p.join()
    left = [f for f in os.listdir("/dev/shm") if f.startswith("trtllm_vgref_")]
    out = {}
    for (v, r), d in res.items():
        pm, cm = statistics.median(d["prod"]), statistics.median(d["cons"])
        tot = sorted(x + y for x, y in zip(d["prod"], d["cons"]))
        out[f"{v}/{r}"] = {"prod_ms": round(pm, 3), "cons_ms": round(cm, 3),
                           "total_ms": round(statistics.median(tot), 3),
                           "p10": round(tot[len(tot) // 10], 3),
                           "p90": round(tot[len(tot) * 9 // 10], 3)}
        log(v, r, out[f"{v}/{r}"])
    print(json.dumps({"passed": bad == 0 and not left, "sha_mismatch": bad,
                      "leftover": left, "results": out}))


if __name__ == "__main__":
    main()
