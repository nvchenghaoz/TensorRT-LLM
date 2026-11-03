"""Shared synthetic setup for the FA2 hd512 K/V-alias benchmark and check (idea 34).

One TP4 rank of DeepSeek-V4.1-Flash CSA2: 16 query heads against 1 KV head, head_dim 512,
BF16, decode width 640 (128 SWA + 512 top-k), prefill 2048 queries x 640 indices into a
shared one-row-page bank. Uses the tree's own CSA2FlashInfer plans so plan arguments match.
"""

import hashlib
import math
import os
import sys
import threading
import time

import torch

HEADS = 16
DIM = 512
WIDTH = 640
SCALE = 1.0 / math.sqrt(DIM)
BANK_ROWS = 2800
PREFILL_Q = 2048


def log(*args):
    print(*args, flush=True)
    sys.stdout.flush()


def start_heartbeat(period=60.0):
    """Keep the job printing while FlashInfer JIT compiles (silence watchdog)."""
    t0 = time.time()

    def beat():
        while True:
            time.sleep(period)
            log(f"[heartbeat] {time.time() - t0:.0f}s")

    threading.Thread(target=beat, daemon=True).start()


def log_flashinfer():
    try:
        import flashinfer

        log("flashinfer", flashinfer.__version__, os.path.dirname(flashinfer.__file__))
    except Exception as e:  # noqa: BLE001
        log("flashinfer import failed", e)
        return
    try:
        from flashinfer.jit import env as jit_env

        for name in ("FLASHINFER_JIT_DIR", "FLASHINFER_WORKSPACE_DIR", "FLASHINFER_INCLUDE_DIR",
                     "FLASHINFER_CSRC_DIR"):
            if hasattr(jit_env, name):
                log(name, getattr(jit_env, name))
        inc = getattr(jit_env, "FLASHINFER_INCLUDE_DIR", None)
        if inc is not None:
            p = os.path.join(str(inc), "flashinfer", "attention", "prefill.cuh")
            if os.path.exists(p):
                data = open(p, "rb").read()
                log("prefill.cuh", p, "sha256", hashlib.sha256(data).hexdigest()[:16],
                    "has_alias_patch", b"kKVAliasCapable" in data)
    except Exception as e:  # noqa: BLE001
        log("jit env probe failed", e)


def log_jit_modules(tag=""):
    try:
        from flashinfer.jit import env as jit_env

        root = str(getattr(jit_env, "FLASHINFER_JIT_DIR"))
        found = []
        for d, _, files in os.walk(root):
            for f in files:
                if f.endswith(".so") and "prefill" in f:
                    p = os.path.join(d, f)
                    found.append((os.path.getmtime(p), p))
        found.sort(reverse=True)
        now = time.time()
        for mt, p in found[:6]:
            log(f"jit module {tag}: {p} age {now - mt:.0f}s")
    except Exception as e:  # noqa: BLE001
        log("jit module probe failed", e)


def new_adapter():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer

    return CSA2FlashInfer()


def make_lengths(count, width, step=0):
    choices = [1, width, 7, 129, 131, 255, 384, 511, 600, 639, 64, 8, 9, 300]
    picked = [choices[(i * 5 + step * 3) % len(choices)] for i in range(count)]
    picked = [min(max(v, 1), width) for v in picked]
    return torch.tensor(picked, dtype=torch.int32, device="cuda")


def make_decode(count, width, seed, dim=DIM, heads=HEADS):
    """q, rows [count*width,1,dim] (zeros past each length), bool valid [count,width]."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    q = (torch.randn(count, heads, dim, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    kv = torch.randn(count, width, dim, generator=gen, device="cuda").to(torch.bfloat16)
    lens = make_lengths(count, width)
    valid = torch.arange(width, device="cuda")[None, :] < lens[:, None]
    rows = torch.where(valid[..., None], kv, 0).reshape(-1, 1, dim).contiguous()
    sink = torch.randn(heads, generator=gen, device="cuda", dtype=torch.float32)
    return q, rows, valid, sink


def make_prefill(seed, count=PREFILL_Q, width=WIDTH, bank_rows=BANK_ROWS):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    q = (torch.randn(count, HEADS, DIM, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    bank = torch.randn(bank_rows, DIM, generator=gen, device="cuda").to(torch.bfloat16)
    indices = torch.randint(0, bank_rows, (count, width), generator=gen, device="cuda",
                            dtype=torch.int32)
    holes = torch.rand(count, width, generator=gen, device="cuda") < 0.1
    indices = torch.where(holes, torch.full_like(indices, -1), indices)
    indices[0, 1:] = -1  # a query with one valid row
    indices[0, 0] = 5
    # every query keeps >= 1 valid row (a fully masked row is NaN by design; run_shared zeroes it)
    indices[:, 0] = indices[:, 0].clamp_min(0)
    sink = torch.randn(HEADS, generator=gen, device="cuda", dtype=torch.float32)
    return q, bank, indices, sink


def decode_plan(adapter, count, width, valid, dim=DIM, heads=HEADS):
    plan = adapter._ragged_plan(count, heads, width, dim, torch.device("cuda"), SCALE)
    plan.packed_mask.copy_(adapter.pack_query_masks(valid))
    return plan


def prefill_plan(adapter, q, bank, indices, sink):
    """Create (via run_shared itself) and fill the paged plan; return it and pages."""
    adapter.run_shared(q, bank, indices, sink, SCALE, protect=True)
    key = (q.shape[0], q.shape[1], indices.shape[1], q.device, SCALE)
    plan = adapter._paged_plans[key]
    return plan, bank.view(-1, 1, 1, DIM)


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:20]


class L2Flush:
    def __init__(self, mb=256):
        self.buf = torch.empty(mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

    def __call__(self):
        self.buf.add_(1.0)
