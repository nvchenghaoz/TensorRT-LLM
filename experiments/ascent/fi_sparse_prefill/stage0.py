"""Stage 0 for idea 239: CSA2 shared-context prefill, main (stage_shared_rows + FA2 run_shared)
vs FlashInfer SM120 DSv4.1 FullBF16 direct sparse prefill over CSA2's own packed pools.

Synthetic tensors at one TP4 rank's shapes (16 heads x 512), no model load.
  python stage0.py [--mode all|check|time] [--iters 30] [--warmup 3]
Prints progress, then one JSON line.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import fi_sparse_prefill as fsp
from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer
from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import stage_shared_rows
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2SharedKVPlan
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import store_rows, unpack_rows

HEADS = 16
D = 512
W = 128
M = 512
CTX = 34816
SWA_PSZ = 64
RATIO = 4
MAIN_PSZ = SWA_PSZ // RATIO
SCALE = D**-0.5
DEV = torch.device("cuda")


def log(*a):
    print(*a, flush=True)


def gpu_memory():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        return out
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


class Scenario:
    def __init__(self, requests, per_request, has_main, seed):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.name = f"{requests * per_request}x{W + (M if has_main else 0)}"
        self.R, self.Tr, self.has_main = requests, per_request, has_main
        count = requests * per_request
        self.count = count
        pages = CTX // SWA_PSZ
        # Pools filled by CSA2's own writer (QuantizeScatter) from random BF16 rows.
        swa_rows = requests * pages * SWA_PSZ
        self.swa_pool = torch.empty(swa_rows, 528, dtype=torch.uint8, device=DEV)
        vals = torch.randn(swa_rows, D, device=DEV, generator=g, dtype=torch.float32)
        vals = vals * torch.exp2(torch.randint(-2, 3, (swa_rows, 1), device=DEV, generator=g).float())
        store_rows(self.swa_pool, torch.arange(swa_rows, device=DEV), vals.to(torch.bfloat16), "swa")
        del vals
        self.swa_table = (
            torch.randperm(requests * pages, device=DEV, generator=g).view(requests, pages).to(torch.int32)
        )
        positions = (CTX - per_request + torch.arange(per_request, device=DEV)).repeat(requests)
        req = torch.arange(requests, device=DEV).repeat_interleave(per_request)
        self.positions = positions.to(torch.int32)
        self.requests = req.to(torch.int64)
        logical = positions[:, None] - W + 1 + torch.arange(W, device=DEV)[None, :]
        phys = (
            self.swa_table[req[:, None], (logical.clamp_min(0) // SWA_PSZ)].long() * SWA_PSZ
            + logical.clamp_min(0) % SWA_PSZ
        )
        swa_slots = torch.where(logical >= 0, phys, -1)
        holes = torch.rand(count, W, device=DEV, generator=g) < 0.03
        swa_slots = torch.where(holes, -1, swa_slots)
        oob = torch.rand(count, W, device=DEV, generator=g) < 0.005
        swa_slots = torch.where(oob, swa_rows + 7, swa_slots)
        self.empty = torch.tensor([5, 6, count - 1], device=DEV)
        swa_slots[self.empty] = -1
        self.swa_slots = swa_slots.contiguous()
        first = positions.view(requests, per_request)[:, 0]
        self.swa_starts = (first - W + 1).clamp_min(0).to(torch.int64)
        swa_counts = [per_request + W - 1] * requests
        main_count = CTX // RATIO
        main_counts = [main_count if has_main else 0] * requests
        offs, total = [1], 1
        for c in swa_counts:
            total += c
            offs.append(total)
        moffs = [total]
        for c in main_counts:
            total += c
            moffs.append(total)
        self.swa_offsets = torch.tensor(offs, dtype=torch.int64, device=DEV)
        self.main_offsets = torch.tensor(moffs, dtype=torch.int64, device=DEV)
        self.bank_rows = total
        self.main_pool = self.main_table = self.main_logical = self.mapping = None
        if has_main:
            mrows = requests * pages * MAIN_PSZ
            self.main_pool = torch.empty(mrows, 288, dtype=torch.uint8, device=DEV)
            vals = torch.randn(mrows, D, device=DEV, generator=g, dtype=torch.float32)
            vals = vals * torch.exp2(torch.randint(-2, 3, (mrows, 1), device=DEV, generator=g).float())
            store_rows(self.main_pool, torch.arange(mrows, device=DEV), vals.to(torch.bfloat16), "main")
            del vals
            self.main_table = (
                torch.randperm(requests * pages, device=DEV, generator=g).view(requests, pages).to(torch.int32)
            )
            visible = ((positions + 1) // RATIO).to(torch.int64)
            self.visible = visible
            sel = []
            for start in range(0, count, 512):
                stop = min(count, start + 512)
                scores = torch.rand(stop - start, main_count, device=DEV, generator=g)
                cols = torch.arange(main_count, device=DEV)[None, :]
                scores = torch.where(cols < visible[start:stop, None], scores, -1.0)
                sel.append(scores.topk(M, dim=1).indices)
            logical_main = torch.cat(sel).to(torch.int64)
            holes = torch.rand(count, M, device=DEV, generator=g) < 0.03
            logical_main = torch.where(holes, -1, logical_main)
            late = torch.rand(count, M, device=DEV, generator=g) < 0.005
            logical_main = torch.where(late, visible[:, None] + 3, logical_main)
            logical_main[self.empty] = -1
            self.main_logical = logical_main.to(torch.int32).contiguous()
            self.mapping = (self.main_table, MAIN_PSZ, main_count, self.requests, self.visible)
        self.plan = CSA2SharedKVPlan(
            self.requests,
            self.positions,
            self.swa_starts,
            self.swa_offsets,
            self.main_offsets,
            self.swa_table,
            self.main_table,
            SWA_PSZ,
            MAIN_PSZ,
            requests,
            sum(swa_counts),
            sum(main_counts),
        )
        capacity = W + (M if has_main else 0)
        self.metadata = SimpleNamespace(
            num_tokens=count,
            num_sparse_topk=capacity,
            shared_pool=torch.empty(self.bank_rows, D, dtype=torch.bfloat16, device=DEV),
            _csa2_shared_selected=torch.zeros(self.bank_rows, dtype=torch.int32, device=DEV),
            prepared_indices=torch.empty(count, capacity, dtype=torch.int32, device=DEV),
            prepared_lens=torch.empty(count, dtype=torch.int32, device=DEV),
            prepared_counter=torch.zeros(4, dtype=torch.int32, device=DEV),
        )
        self.q = (torch.randn(count, HEADS, D, device=DEV, generator=g)).to(torch.bfloat16)
        self.sink = torch.randn(HEADS, device=DEV, generator=g, dtype=torch.float32)
        self.out_main = torch.empty_like(self.q)
        self.out_fi = torch.empty_like(self.q)

    # --- the two routes -------------------------------------------------------------
    def run_main(self, helper):
        stage_shared_rows(
            self.swa_pool,
            self.swa_slots,
            self.main_pool,
            None,
            self.main_logical,
            self.metadata,
            self.plan,
            main_mapping=self.mapping,
            softmax_scale=SCALE,
        )
        return helper.run_shared(
            self.q, self.metadata.shared_pool, self.metadata.prepared_indices, self.sink, SCALE, out=self.out_main
        )

    def run_fi(self, fi):
        return fi(
            self.q,
            self.swa_pool,
            self.swa_slots,
            self.main_pool,
            self.main_logical,
            self.mapping,
            self.plan,
            self.sink,
            SCALE,
            self.out_fi,
        )

    def run_fi_kernel_only(self, fi):
        # Attention launch alone, on the already-prepared index scratch.
        has_main = self.has_main
        dev = self.q.device  # the key FISharedPrefill uses (cuda:N, not bare cuda)
        bufs = fi._scratch(self.count, HEADS, W, M if has_main else 0, dev)
        fi._wrapper(has_main, dev).run(
            self.q,
            self.swa_pool.view(-1, 1, 528),
            bufs["swa"],
            self.out_fi,
            SCALE,
            attn_sink=self.sink,
            extra_kv_cache=self.main_pool.view(-1, 1, 288) if has_main else None,
            extra_indices=bufs["main"] if has_main else None,
            out_lse=bufs["lse"],
        )

    # --- independent FP64 reference ---------------------------------------------------
    def reference(self):
        count = self.count
        req = self.requests
        swa_deq = unpack_rows(self.swa_pool, D, "swa", torch.float32)
        logical = self.positions.long()[:, None] - W + 1 + torch.arange(W, device=DEV)[None, :]
        cnt = self.Tr + W - 1
        start = self.swa_starts[req][:, None]
        s = self.swa_slots
        valid = (s >= 0) & (s < self.swa_pool.shape[0]) & (logical >= start) & (logical < start + cnt)
        lg = logical.clamp_min(0)
        phys = self.swa_table[req[:, None], lg // SWA_PSZ].long() * SWA_PSZ + lg % SWA_PSZ
        parts = [(swa_deq, phys, valid)]
        if self.has_main:
            main_deq = unpack_rows(self.main_pool, D, "main", torch.float32)
            x = self.main_logical.long()
            mc = CTX // RATIO
            mvalid = (x >= 0) & (x < self.visible[:, None]) & (x < mc)
            lx = x.clamp(0, mc - 1)
            mphys = self.main_table[req[:, None], lx // MAIN_PSZ].long() * MAIN_PSZ + lx % MAIN_PSZ
            parts.append((main_deq, mphys, mvalid))
        out = torch.empty(count, HEADS, D, dtype=torch.float64, device=DEV)
        sink = self.sink.double()
        chunk = 64
        for a in range(0, count, chunk):
            b = min(count, a + chunk)
            ks, vs = [], []
            for deq, ph, va in parts:
                ks.append(deq[ph[a:b]].double())
                vs.append(va[a:b])
            K = torch.cat(ks, 1)  # [n, width, D]
            V = torch.cat(vs, 1)  # [n, width]
            q = self.q[a:b].double()
            logits = torch.einsum("nhd,nwd->nhw", q, K) * SCALE
            logits = torch.where(V[:, None, :], logits, -math.inf)
            m = torch.maximum(logits.amax(-1), sink[None, :])
            p = torch.exp(logits - m[..., None])
            den = p.sum(-1) + torch.exp(sink[None, :] - m)
            o = torch.einsum("nhw,nwd->nhd", p, K) / den[..., None]
            o = torch.where(V.any(-1)[:, None, None], o, 0.0)
            out[a:b] = o
        return out


def errors(out, ref):
    d = (out.double() - ref).abs()
    denom = ref.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel_rowmax": float((d / denom).max()),
        "finite": bool(torch.isfinite(out).all()),
    }


def time_fn(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    med = times[len(times) // 2]
    p10, p90 = times[int(0.1 * (len(times) - 1))], times[int(0.9 * (len(times) - 1))]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "noise": (p90 - p10) / med}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=("all", "check", "time"))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    log("gpu memory.used at start:", gpu_memory())
    log("device", torch.cuda.get_device_name(), "sm", torch.cuda.get_device_capability())
    shapes = [
        (1, 2048, True),
        (1, 2048, False),
        (1, 128, True),
        (1, 128, False),
        (3, 2048, True),
        (3, 2048, False),
        (3, 128, True),
        (3, 128, False),
    ]
    t0 = time.time()
    scen = {}
    for i, (r, t, hm) in enumerate(shapes):
        s = Scenario(r, t, hm, seed=1000 + i)
        scen[s.name] = s
        log(f"built {s.name} bank_rows={s.bank_rows} ({time.time() - t0:.1f}s)")
    helper = CSA2FlashInfer(attention=None)
    fi = fsp.FISharedPrefill()
    result = {"shapes": {}, "build": {}}
    # JIT build / first call of each route
    for name, s in scen.items():
        a = time.time()
        s.run_main(helper)
        torch.cuda.synchronize()
        b = time.time()
        s.run_fi(fi)
        torch.cuda.synchronize()
        c = time.time()
        result["build"][name] = {"main_first_s": b - a, "fi_first_s": c - b}
        log(f"first call {name}: main {b - a:.1f}s fi {c - b:.1f}s")
    ok_all = True
    if args.mode in ("all", "check"):
        for name, s in scen.items():
            ref = s.reference()
            first_fi = None
            worst_fi, worst_main = 0.0, 0.0
            bitwise_repeat = True
            for it in range(args.warmup + args.iters):
                s.out_main.fill_(float("nan"))
                s.out_fi.fill_(float("nan"))
                s.run_main(helper)
                s.run_fi(fi)
                torch.cuda.synchronize()
                em, ef = errors(s.out_main, ref), errors(s.out_fi, ref)
                worst_main = max(worst_main, em["max_abs"])
                worst_fi = max(worst_fi, ef["max_abs"])
                if first_fi is None:
                    first_fi = s.out_fi.clone()
                    first_main_err, first_fi_err = em, ef
                    empty_zero = bool((s.out_fi[s.empty] == 0).all()) and bool((s.out_main[s.empty] == 0).all())
                    fi_vs_main = float((s.out_fi.float() - s.out_main.float()).abs().max())
                elif not torch.equal(first_fi, s.out_fi):
                    bitwise_repeat = False
                if not (ef["finite"] and em["finite"]):
                    ok_all = False
            passed = (
                bitwise_repeat
                and empty_zero
                and worst_fi <= worst_main * 1.25 + 1e-6
                and first_fi_err["max_rel_rowmax"] < 1.6e-2
            )
            ok_all &= passed
            result["shapes"].setdefault(name, {})["check"] = {
                "main_vs_fp64": first_main_err,
                "fi_vs_fp64": first_fi_err,
                "worst_main_max_abs": worst_main,
                "worst_fi_max_abs": worst_fi,
                "fi_vs_main_max_abs": fi_vs_main,
                "empty_rows_zero": empty_zero,
                "fi_repeat_bitwise": bitwise_repeat,
                "passed": passed,
            }
            log(name, json.dumps(result["shapes"][name]["check"]))
            del ref
            torch.cuda.empty_cache()
    if args.mode in ("all", "time"):
        for name, s in scen.items():
            tm = time_fn(lambda: s.run_main(helper), args.iters, args.warmup)
            tf = time_fn(lambda: s.run_fi(fi), args.iters, args.warmup)
            tk = time_fn(lambda: s.run_fi_kernel_only(fi), args.iters, args.warmup)
            result["shapes"].setdefault(name, {})["time"] = {"main": tm, "fi_with_prep": tf, "fi_kernel": tk}
            log(
                f"P {name}: main {tm['median_ms']:.4f} ms (n {tm['noise']:.3f}) | fi+prep {tf['median_ms']:.4f} "
                f"(n {tf['noise']:.3f}) | fi kernel {tk['median_ms']:.4f}"
            )

        def seq(names, fn_name):
            def run():
                for nm in names:
                    if fn_name == "main":
                        scen[nm].run_main(helper)
                    elif fn_name == "fi":
                        scen[nm].run_fi(fi)
                    else:
                        scen[nm].run_fi_kernel_only(fi)
            return run

        c1 = ["2048x640", "2048x128"] * 10 + ["128x640", "128x128"] * 10
        c4 = ["6144x640", "6144x128"] * 10 + ["384x640", "384x128"] * 10
        result["sequences"] = {}
        for label, names in (("c1_request_40", c1), ("c4_follower_40", c4)):
            res = {k: time_fn(seq(names, k), max(10, args.iters // 3), 2) for k in ("main", "fi", "fi_kernel")}
            res["saved_ms"] = res["main"]["median_ms"] - res["fi"]["median_ms"]
            result["sequences"][label] = res
            log(
                f"R {label}: main {res['main']['median_ms']:.3f} ms | fi+prep {res['fi']['median_ms']:.3f} | "
                f"fi kernel {res['fi_kernel']['median_ms']:.3f} | saved {res['saved_ms']:.3f} ms"
            )
    result["passed"] = bool(ok_all)
    print(json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())
