"""Shared inputs / references / arms for the CSA2 Cake decode experiments (branch 129).

One TP4 rank: q [count, 16, 512] bf16, sink [16] fp32, sm_scale 1/sqrt(512).
Pools are filled with CSA2's OWN writer (quantization.store_rows -> QuantizeScatter):
SWA pool [SWA_ROWS, 528] (E4M3 + UE8M0 g32), main pool [MAIN_ROWS, 288] (E2M1 + E4M3 g16).
Selections: SWA slots int64 [count, 128] (physical, with -1 holes and out-of-pool slots),
main logical positions int32 [count, 512] mapped through an int32 page table
(page_size 16, visible lengths, -1 holes, positions past visible), like production.

Arms
  main : stage_selected_rows (StageCompact once + StageDequant) -> CSA2FlashInfer._forward_pools_decode
         (Triton _split_sink_main + _split_sink_merge) -- main's production decode chain.
  cake : cake_decode.CakeDecode (int32 index prep + Cake decode + PDL merge), when present.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

HEADS, DIM, SWA_W, MAIN_W = 16, 512, 128, 512
SCALE = 1.0 / math.sqrt(DIM)
SWA_ROWS = 8192
MAIN_ROWS = 65536
PAGE = 16
FLUSH_BYTES = 256 * 1024 * 1024


def log(*a):
    print(*a, flush=True)


def make_values(rows, gen, *, large=True):
    """BF16 rows with per-group magnitudes over many binades, +-0 and all-zero groups."""
    v = torch.randn((rows, DIM), generator=gen)
    if large:
        expo = torch.randint(-6, 7, (rows, DIM // 16), generator=gen).float()
        v = v * torch.exp2(expo).repeat_interleave(16, dim=1)
        big = torch.rand(rows, generator=gen) < 0.02
        v[big] *= 2.0 ** 10
    # +-0 entries
    z = torch.rand((rows, DIM), generator=gen) < 0.03
    sign = torch.rand((rows, DIM), generator=gen) < 0.5
    v = torch.where(z, torch.where(sign, torch.tensor(-0.0), torch.tensor(0.0)), v)
    # all-zero groups (32 wide covers both formats' groups)
    zg = torch.rand((rows, DIM // 32), generator=gen) < 0.03
    v = v * (~zg).float().repeat_interleave(32, dim=1)
    return v.to(torch.bfloat16)


def build_pools(seed, device="cuda", extremes=False):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import store_rows

    gen = torch.Generator().manual_seed(seed)
    swa_pool = torch.zeros((SWA_ROWS, 528), dtype=torch.uint8, device=device)
    main_pool = torch.zeros((MAIN_ROWS, 288), dtype=torch.uint8, device=device)
    sv = make_values(SWA_ROWS, gen).to(device)
    mv = make_values(MAIN_ROWS, gen).to(device)
    store_rows(swa_pool, torch.arange(SWA_ROWS, device=device), sv, "swa")
    store_rows(main_pool, torch.arange(MAIN_ROWS, device=device), mv, "main")
    if extremes:
        # UE8M0 extremes on a few SWA rows (byte 0 = 2^-127, 1 = 2^-126), large exponent 150.
        swa_pool[0:8, 512:] = 0
        swa_pool[8:16, 512:] = 1
        swa_pool[16:24, 512:] = 150
        # E4M3 payload extremes (+-448, +-min subnormal) on a few rows.
        swa_pool[24:28, :512] = 0x7E
        swa_pool[28:32, :512] = 0xFE
        swa_pool[32:36, :512] = 0x01
        # Main: E4M3 scale extremes (min subnormal, 448) and every E2M1 code.
        main_pool[0:8, 256:] = 0x01
        main_pool[8:16, 256:] = 0x7E
        codes = torch.arange(256, device=device, dtype=torch.int32).to(torch.uint8)
        main_pool[16:32, :256] = codes[None, :]
    torch.cuda.synchronize()
    return swa_pool, main_pool


def build_selection(count, seed, device="cuda", width=640, empty_rows=(), dense=False,
                    force_rows=None):
    """Selections + mapping. Returns dict with swa (int64), logical (int32), mapping tuple."""
    gen = torch.Generator().manual_seed(seed)
    swa = torch.stack([torch.randperm(SWA_ROWS, generator=gen)[:SWA_W] for _ in range(count)])
    if force_rows is not None:
        swa[:, : len(force_rows)] = torch.tensor(force_rows)[None, :]
    if not dense:
        holes = torch.rand((count, SWA_W), generator=gen) < 0.08
        holes[:, : 0 if force_rows is None else len(force_rows)] = False
        swa[holes] = -1
        oob = torch.rand((count, SWA_W), generator=gen) < 0.02
        oob[:, : 0 if force_rows is None else len(force_rows)] = False
        swa[oob] = SWA_ROWS + 5
    requests = torch.arange(count, dtype=torch.int32)
    pages_per_req = MAIN_ROWS // PAGE // max(count, 1)
    pages_per_req = min(pages_per_req, 600)
    max_positions = pages_per_req * PAGE
    perm = torch.randperm(MAIN_ROWS // PAGE, generator=gen)[: count * pages_per_req]
    table = perm.view(count, pages_per_req).to(torch.int32)
    if force_rows is not None:
        # identity pages 0,1 at logical pages 0,1 so physical rows 0..31 are selectable
        table[:, 0] = 0
        table[:, 1] = 1
    if not dense:
        # an unmapped page in every request
        table[:, 3] = -1
    visible = torch.full((count,), max_positions - 37, dtype=torch.int64)
    logical = torch.stack(
        [torch.randperm(max_positions, generator=gen)[:MAIN_W] for _ in range(count)]
    ).to(torch.int32)
    if force_rows is not None:
        logical[:, :32] = torch.arange(32, dtype=torch.int32)[None, :]
    if not dense:
        holes = torch.rand((count, MAIN_W), generator=gen) < 0.08
        if force_rows is not None:
            holes[:, :32] = False
        logical[holes] = -1
    for r in empty_rows:
        swa[r] = -1
        logical[r] = -1
    sel = dict(
        swa=swa.to(device),
        logical=logical.to(device),
        table=table.to(device),
        page_size=PAGE,
        max_positions=max_positions,
        requests=requests.to(device),
        visible=visible.to(device),
        width=width,
    )
    sel["mapping"] = (sel["table"], PAGE, max_positions, sel["requests"], sel["visible"])
    return sel


def map_reference(sel):
    """Torch reference of map_logical_slot + pool bounds -> physical int64 slots (-1 masked)."""
    logical = sel["logical"].long()
    table = sel["table"].long()
    page = logical.clamp_min(0) // PAGE
    req = sel["requests"].long()[:, None].expand_as(logical)
    safe_page = page.clamp(max=table.shape[1] - 1)
    physical = table[req.clamp(0, table.shape[0] - 1), safe_page]
    valid = (logical >= 0) & (logical < sel["max_positions"]) & (page < table.shape[1])
    valid &= physical >= 0
    valid &= logical < sel["visible"][:, None]
    slot = torch.where(valid, physical * PAGE + logical.clamp_min(0) % PAGE, -1)
    slot = torch.where((slot >= 0) & (slot < MAIN_ROWS), slot, -1)
    swa = sel["swa"]
    swa = torch.where((swa >= 0) & (swa < SWA_ROWS), swa, -1)
    return swa, slot


def reference(q, sink, swa_pool, main_pool, sel):
    """FP64 sink attention over CSA2's exactly dequantized rows (gather_rows)."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows

    swa, main = map_reference(sel)
    count = q.shape[0]
    rows = [gather_rows(swa_pool, swa, DIM, "swa").double()]
    valid = [swa >= 0]
    if sel["width"] > SWA_W:
        rows.append(gather_rows(main_pool, main, DIM, "main").double())
        valid.append(main >= 0)
    kv = torch.cat(rows, 1)
    ok = torch.cat(valid, 1)
    kv = torch.where(ok[..., None], kv, 0.0)
    s = torch.einsum("bhd,bkd->bhk", q.double(), kv) * SCALE
    s = s.masked_fill(~ok[:, None, :], -math.inf)
    lse = torch.logsumexp(s, -1)
    p = torch.nan_to_num(torch.softmax(s, -1), nan=0.0)
    o = torch.einsum("bhk,bkd->bhd", p, kv)
    out = o * torch.sigmoid(lse - sink.double()[None, :])[..., None]
    return torch.where(ok.any(-1)[:, None, None], out, 0.0).view(count, HEADS, DIM)


def make_q(count, seed, device="cuda"):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn((count, HEADS, DIM), generator=gen).to(torch.bfloat16).to(device)
    sink = (torch.randn((HEADS,), generator=gen) * 2.0).float().to(device)
    return q, sink


class MainArm:
    """main's production decode chain: StageCompact (optional) + StageDequant + split sink."""

    def __init__(self, count, width, device="cuda"):
        from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer

        extra_rows = width - SWA_W
        self.count, self.width = count, width
        self.meta = SimpleNamespace(
            num_tokens=count,
            num_sparse_topk=width,
            swa_pool=torch.zeros((count, SWA_W, DIM), dtype=torch.bfloat16, device=device),
            extra_pool=torch.zeros(
                (count, max(extra_rows, 1), DIM), dtype=torch.bfloat16, device=device
            ),
            prepared_counter=torch.zeros(1, dtype=torch.uint32, device=device),
            prepared_lens=torch.empty(count, dtype=torch.int32, device=device),
            prepared_indices=torch.full((count, width), -1, dtype=torch.int32, device=device),
            mla_bmm1_scale=None,
            mla_bmm2_scale=None,
        )
        self.scratch = {
            "tags": torch.empty((count, width), dtype=torch.int32, device=device),
            "counts": torch.empty(count, dtype=torch.int32, device=device),
            "main_slots": torch.empty((count, max(extra_rows, 1)), dtype=torch.int64, device=device),
        }
        self.fi = CSA2FlashInfer()

    def __call__(self, q, sink, swa_pool, main_pool, sel, out, compact=True):
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import stage_selected_rows

        has_main = self.width > SWA_W
        stage_selected_rows(
            swa_pool,
            sel["swa"],
            main_pool if has_main else None,
            self.scratch["main_slots"][: self.count, :MAIN_W] if has_main else None,
            self.meta,
            self.scratch,
            compact=compact,
            main_logical=sel["logical"] if has_main else None,
            main_mapping=sel["mapping"] if has_main else None,
            softmax_scale=SCALE,
        )
        extra = self.meta.extra_pool if has_main else None
        return self.fi._forward_pools_decode(
            q, self.meta.swa_pool, extra, self.meta.prepared_lens[: self.count], sink, SCALE,
            out=out,
        )


def cake_arm_available():
    try:
        from tensorrt_llm._torch.attention.backends.sparse.csa2 import cake_decode  # noqa: F401
    except ImportError:
        return False
    return True


class CakeArm:
    def __init__(self, count, width, device="cuda"):
        from tensorrt_llm._torch.attention.backends.sparse.csa2.cake_decode import CakeDecode

        self.width = width
        self.run = CakeDecode()

    def __call__(self, q, sink, swa_pool, main_pool, sel, out, compact=True):
        has_main = self.width > SWA_W
        return self.run(
            q,
            swa_pool,
            sel["swa"],
            main_pool if has_main else None,
            sel["logical"] if has_main else None,
            sink,
            SCALE,
            out,
            mapping=sel["mapping"] if has_main else None,
        )


class CakeNoPrepArm:
    """Cake decode on pre-built int32 indices (prep outside the timed region)."""

    def __init__(self, count, width, sel, device="cuda"):
        self.width = width
        swa, main = map_reference(sel)
        self.swa = swa.to(torch.int32).contiguous()
        self.main = main.to(torch.int32).contiguous()
        from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (
            cake_sparse_mla_sm120_dsv41_mixed_num_chunks,
        )

        splits = cake_sparse_mla_sm120_dsv41_mixed_num_chunks(
            SWA_W, MAIN_W if width > SWA_W else 0
        )
        self.lse = torch.empty((count, HEADS), dtype=torch.float32, device=device)
        self.mid_out = torch.empty((count, HEADS, splits, DIM), dtype=torch.bfloat16, device=device)
        self.mid_lse = torch.empty((count, HEADS, splits), dtype=torch.float32, device=device)
        self.plan = None

    def __call__(self, q, sink, swa_pool, main_pool, sel, out, compact=True):
        from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (
            cake_sparse_mla_sm120_dsv41_mixed_decode,
        )

        has_main = self.width > SWA_W
        self.plan = cake_sparse_mla_sm120_dsv41_mixed_decode(
            q, swa_pool, self.swa, out, self.lse, SCALE, attn_sink=sink,
            extra_kv_cache=main_pool if has_main else None,
            extra_indices=self.main if has_main else None,
            mid_out=self.mid_out, mid_lse=self.mid_lse,
        )
        return out


_FLUSH = None


def time_graph(fn, warmup=10, iters=100, flush=True):
    """CUDA-graph replay, cold L2 (256 MB memset before every replay). Returns stats in us."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup):
        if flush:
            _FLUSH.zero_()
        graph.replay()
    for i in range(iters):
        if flush:
            _FLUSH.zero_()
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) * 1e3 for a, b in zip(starts, ends))
    med = t[len(t) // 2]
    p10, p90 = t[len(t) // 10], t[(9 * len(t)) // 10]
    return dict(median_us=med, p10_us=p10, p90_us=p90, noise=(p90 - p10) / med), graph


def sm_clock():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return dict(
            sm_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            temp_c=pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            throttle=int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)),
        )
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e)[:100])
