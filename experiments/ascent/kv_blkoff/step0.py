"""STEP 0 for idea 30: copyBatchBlockOffsetsToDeviceKernel arms at numSeqs 8/16/32.

Arms (GPU span measured with CUDA events while the host is ahead of the GPU,
i.e. after a torch.cuda._sleep so every op of the arm is already enqueued):
  stock      : real binding copy_batch_block_offsets_to_device, all inputs pinned host
  stock_aa   : same again (A/A floor)
  A2         : host index_select snapshot (both planes, fresh pinned) -> H2D DMA
               -> the same stock kernel on device pointers (device copy_idx/scales/offsets)
  A1         : K-plane-only host snapshot -> H2D DMA -> ext kernel (src seq stride B)
  P{S}{l|o}  : split-row zero-copy kernel from pinned host, S splits per row,
               copy idx / scales / offsets by value (hoisted), l=loop, o=oneshot
Diagnostics: stock with contiguous copy_idx, stock with 288-block rows, DMA alone,
device-source kernel alone.
"""
import json
import os
import statistics
import time

import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

constexpr int MAXSEQ = 256;
constexpr int MAXPOOL = 64;
struct Params {
  int copyIdx[MAXSEQ];
  int scale[MAXPOOL];
  int off[MAXPOOL];
};

template <bool ONESHOT>
__global__ void pKernel(const int* __restrict__ src, int* __restrict__ dst, int srcMax, int dstMax,
                        int B, int srcSeqStride, const __grid_constant__ Params p) {
  const int pool = blockIdx.x, seq = blockIdx.y, split = blockIdx.z, S = gridDim.z;
  const int nVec = B / 4;
  const int per = (nVec + S - 1) / S;
  const int beg = split * per;
  const int end = min(nVec, beg + per);
  const int scale = p.scale[pool];
  const int off = p.off[pool];
  const int ci = p.copyIdx[seq];
  const int4* s = reinterpret_cast<const int4*>(src + (size_t)(pool * srcMax + ci) * srcSeqStride);
  int4* dK = reinterpret_cast<int4*>(dst + (size_t)(pool * dstMax + seq) * 2 * B);
  int4* dV = dK + nVec;
  if (ONESHOT) {
    int4 v[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
      int i = beg + threadIdx.x + r * blockDim.x;
      if (i < end) v[r] = s[i];
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
      int i = beg + threadIdx.x + r * blockDim.x;
      if (i < end) {
        int4 k, vv;
        k.x = v[r].x == -1 ? 0 : scale * v[r].x; vv.x = v[r].x == -1 ? 0 : scale * v[r].x + off;
        k.y = v[r].y == -1 ? 0 : scale * v[r].y; vv.y = v[r].y == -1 ? 0 : scale * v[r].y + off;
        k.z = v[r].z == -1 ? 0 : scale * v[r].z; vv.z = v[r].z == -1 ? 0 : scale * v[r].z + off;
        k.w = v[r].w == -1 ? 0 : scale * v[r].w; vv.w = v[r].w == -1 ? 0 : scale * v[r].w + off;
        dK[i] = k; dV[i] = vv;
      }
    }
  } else {
    for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
      int4 x = s[i];
      int4 k, vv;
      k.x = x.x == -1 ? 0 : scale * x.x; vv.x = x.x == -1 ? 0 : scale * x.x + off;
      k.y = x.y == -1 ? 0 : scale * x.y; vv.y = x.y == -1 ? 0 : scale * x.y + off;
      k.z = x.z == -1 ? 0 : scale * x.z; vv.z = x.z == -1 ? 0 : scale * x.z + off;
      k.w = x.w == -1 ? 0 : scale * x.w; vv.w = x.w == -1 ? 0 : scale * x.w + off;
      dK[i] = k; dV[i] = vv;
    }
  }
}

void p_copy(torch::Tensor src, torch::Tensor dst, std::vector<int64_t> copyIdx, std::vector<int64_t> scale,
            std::vector<int64_t> off, int64_t S, bool oneshot, int64_t srcMax, int64_t srcSeqStride) {
  Params p;
  int nSeq = copyIdx.size();
  int nPool = scale.size();
  TORCH_CHECK(nSeq <= MAXSEQ && nPool <= MAXPOOL);
  for (int i = 0; i < nSeq; i++) p.copyIdx[i] = (int)copyIdx[i];
  for (int i = 0; i < nPool; i++) { p.scale[i] = (int)scale[i]; p.off[i] = (int)off[i]; }
  int dstMax = dst.size(1);
  int B = dst.size(3);
  dim3 grid(nPool, nSeq, S);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* sp = reinterpret_cast<const int*>(src.data_ptr());
  if (oneshot) {
    TORCH_CHECK((B / 4 + S - 1) / S <= 4 * 128, "oneshot needs <= 512 vec per CTA");
    pKernel<true><<<grid, 128, 0, stream>>>(sp, dst.data_ptr<int>(), srcMax, dstMax, B, srcSeqStride, p);
  } else {
    pKernel<false><<<grid, 128, 0, stream>>>(sp, dst.data_ptr<int>(), srcMax, dstMax, B, srcSeqStride, p);
  }
}
"""
CPP_SRC = r"""
#include <torch/extension.h>
void p_copy(torch::Tensor src, torch::Tensor dst, std::vector<int64_t> copyIdx, std::vector<int64_t> scale,
            std::vector<int64_t> off, int64_t S, bool oneshot, int64_t srcMax, int64_t srcSeqStride);
"""


def log(*a):
    print(*a, flush=True)


def _bdir(n):
    d = os.path.join("/tmp", n)
    os.makedirs(d, exist_ok=True)
    return d


def load_ext():
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    mod = load_inline(name="kvblkoff_step0", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                      functions=["p_copy"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
                      build_directory=_bdir("kvblkoff_step0"), verbose=False)
    log(f"ext built in {time.time() - t0:.1f}s")
    return mod


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        IndexMapper, copy_batch_block_offsets_to_device)
    ext = load_ext()
    dev = torch.device("cuda")
    stream = torch.cuda.current_stream()
    WARM, TIMED = 10, 50
    SRC_MAX = 34
    results = {}
    pools_list = [int(x) for x in os.environ.get("POOLS", "1,2").split(",")]
    thrash = os.environ.get("THRASH", "none")
    big = None
    if thrash != "none":
        big = torch.empty(int(os.environ.get("THRASH_GB", "12")) << 30, dtype=torch.uint8, device=dev)
        big.fill_(1)
        log(f"thrash mode {thrash} over {big.numel() >> 30} GB")

    def do_thrash():
        if thrash == "touch":
            # one 8-byte read per 64 KB: evicts GPU TLB entries like the decode graph's weight stream
            big.view(torch.int64)[::8192].sum()
        elif thrash == "stream":
            big.view(torch.int64).sum()

    blist = [int(x) for x in os.environ.get("BLOCKS", "1152,288").split(",")]
    for P in pools_list:
        for B in blist:
            g = torch.Generator().manual_seed(1234 + P + B)
            host = torch.randint(0, 200000, (P, SRC_MAX, 2, B), dtype=torch.int32, generator=g)
            host[torch.rand(host.shape, generator=g) < 0.05] = -1
            host = host.pin_memory()
            scales_l = [3 + 2 * p for p in range(P)]
            offs_l = [7 + 11 * p for p in range(P)]
            scales_h = torch.tensor(scales_l, dtype=torch.int32).pin_memory()
            offs_h = torch.tensor(offs_l, dtype=torch.int32).pin_memory()
            scales_d = scales_h.to(dev)
            offs_d = offs_h.to(dev)
            mapper = IndexMapper(SRC_MAX, 1, 0)
            rids = list(range(1000, 1000 + SRC_MAX))
            slot_of = {}
            for r in rids:
                slot_of[r] = mapper.add_new_sequence(r)
            stage_d = torch.empty(P * SRC_MAX * 2 * B, dtype=torch.int32, device=dev)
            ident_d = torch.arange(SRC_MAX, dtype=torch.int32, device=dev)
            for N in (8, 16, 32):
                if B == 288 and N != 32:
                    continue
                perm = torch.randperm(SRC_MAX, generator=g)[:N].tolist()
                req = [rids[i] for i in perm]
                cidx = mapper.get_copy_index(req, 0, 1)  # pinned, persistent
                cidx_list = cidx.tolist()
                assert cidx_list == [slot_of[r] for r in req]
                contig_idx = torch.arange(N, dtype=torch.int32).pin_memory()
                dst_ref = torch.full((P, SRC_MAX, 2, B), 0x7f7f7f7f, dtype=torch.int32, device=dev)
                copy_batch_block_offsets_to_device(host, dst_ref, cidx, scales_h, offs_h, stream.cuda_stream)
                torch.cuda.synchronize()
                src_k = host[:, cidx_list, 0, :].long()
                sc = torch.tensor(scales_l).view(P, 1, 1).long()
                of = torch.tensor(offs_l).view(P, 1, 1).long()
                refk = torch.where(src_k == -1, 0, src_k * sc).int()
                refv = torch.where(src_k == -1, 0, src_k * sc + of).int()
                dr = dst_ref.cpu()
                assert torch.equal(dr[:, :N, 0], refk) and torch.equal(dr[:, :N, 1], refv), "stock vs torch ref"

                host_times = {}
                dst = torch.full((P, SRC_MAX, 2, B), 0x7f7f7f7f, dtype=torch.int32, device=dev)

                def arm_stock():
                    copy_batch_block_offsets_to_device(host, dst, cidx, scales_h, offs_h, stream.cuda_stream)

                def arm_stock_contig():
                    copy_batch_block_offsets_to_device(host, dst, contig_idx, scales_h, offs_h, stream.cuda_stream)

                def arm_A2():
                    t0 = time.perf_counter()
                    snap = torch.empty((P, N, 2, B), dtype=torch.int32, pin_memory=True)
                    torch.index_select(host, 1, cidx, out=snap)
                    sd = stage_d[:P * N * 2 * B].view(P, N, 2, B)
                    sd.copy_(snap, non_blocking=True)
                    copy_batch_block_offsets_to_device(sd, dst, ident_d[:N], scales_d, offs_d, stream.cuda_stream)
                    host_times.setdefault("A2", []).append(time.perf_counter() - t0)

                def arm_A1():
                    t0 = time.perf_counter()
                    snap = torch.empty((P, N, 1, B), dtype=torch.int32, pin_memory=True)
                    mapper.gather_k_block_offsets(host, snap, req, B)
                    sd = stage_d[:P * N * B].view(P, N, 1, B)
                    sd.copy_(snap, non_blocking=True)
                    ext.p_copy(sd, dst, list(range(N)), scales_l, offs_l, 1, True, N, B)
                    host_times.setdefault("A1", []).append(time.perf_counter() - t0)

                snapA2 = torch.empty((P, N, 2, B), dtype=torch.int32, pin_memory=True)
                torch.index_select(host, 1, cidx, out=snapA2)
                sdA2 = stage_d[:P * N * 2 * B].view(P, N, 2, B)

                def arm_dma_only():
                    sdA2.copy_(snapA2, non_blocking=True)

                def arm_devkernel_only():
                    copy_batch_block_offsets_to_device(sdA2, dst, ident_d[:N], scales_d, offs_d, stream.cuda_stream)

                def mk_p(S, oneshot):
                    def f():
                        ext.p_copy(host, dst, cidx_list, scales_l, offs_l, S, oneshot, SRC_MAX, 2 * B)
                    return f

                arms = [("stock", arm_stock), ("stock_aa", arm_stock), ("A2", arm_A2), ("A1", arm_A1)]
                for S in (1, 2, 4, 8, 16):
                    arms.append((f"P{S}l", mk_p(S, False)))
                    arms.append((f"P{S}o", mk_p(S, True)))
                arms += [("diag_stock_contig_idx", arm_stock_contig), ("diag_dma_only", arm_dma_only),
                         ("diag_devkernel_only", arm_devkernel_only)]
                if B == 288:
                    arms = [a for a in arms if a[0] in ("stock", "stock_aa", "P1o", "P16o", "A2")]
                times = {a: [] for a, _ in arms}
                exact = {}
                for name, fn in arms:
                    if name.startswith("diag_dma") or name.startswith("diag_stock_contig"):
                        continue
                    dst.fill_(0x7f7f7f7f)
                    fn()
                    torch.cuda.synchronize()
                    exact[name] = bool(torch.equal(dst, dst_ref))
                for k in host_times:
                    host_times[k] = []
                for it in range(WARM + TIMED):
                    for name, fn in arms:
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        do_thrash()
                        torch.cuda._sleep(200000)
                        e0.record()
                        fn()
                        e1.record()
                        e1.synchronize()
                        if it >= WARM:
                            times[name].append(e0.elapsed_time(e1) * 1000.0)
                key = f"P{P}_B{B}_N{N}"
                res = {}
                for name, _ in arms:
                    xs = times[name]
                    res[name] = {"med_us": round(statistics.median(xs), 2), "p10_us": round(pct(xs, 0.1), 2),
                                 "p90_us": round(pct(xs, 0.9), 2), "exact": exact.get(name)}
                for k, v in host_times.items():
                    xs = [x * 1e6 for x in v[WARM:]]
                    res[k]["host_med_us"] = round(statistics.median(xs), 1)
                results[key] = res
                log(key, json.dumps(res))
    print(json.dumps({"step0": results}))


if __name__ == "__main__":
    main()
