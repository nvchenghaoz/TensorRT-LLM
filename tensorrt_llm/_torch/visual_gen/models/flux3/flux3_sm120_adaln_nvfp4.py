# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All
# rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""SM120 warp-per-row LayerNorm + adaLN modulate + NVFP4 quantize (FLUX.3, D=3072).

Drop-in for flux3_adaln_quant._adaln_nvfp4_kernel (same inputs, same outputs): one warp owns
one row, the row stays in registers (one x read), all reductions are shuffles.

Numerics (class: reordered). The LN statistics replay the Triton kernel's Welford tree below;
the best-matching tree leaves 1-bf16-ulp differences on ~1e-6 of x_mod elements (a residual
reduction-order difference), while fp4 data and the 128x4 scale factors were bitwise to the
Triton kernel in every tested case. The Triton kernel's LN statistics are Inductor's looped Welford over a
1024-wide block held by 8 warps x 32 lanes x 4 contiguous slots, followed by tl.reduce:
in-thread combine of the 4 slots, a 32-lane xor butterfly (each lane combines (own, partner)),
then the 8 warp partials (lane 0 of each warp) through an 8-lane xor butterfly. This kernel
replays that tree: real lane L of the row's warp holds the virtual threads (w, l) with
w = 2g + (L >> 4), l = 2 (L & 15) + l0 for g = 0..3, l0 = 0..1, i.e. the 8 contiguous columns
256 g + 8 L (+1024 c for chunk c), so every x load is a coalesced 16-byte load. Virtual
butterfly offsets 16..2 are real-lane xor 8..1, offset 1 is in-register. fp32 division is
div.full.f32 (what Triton emits for '/'), the build uses --fmad=false (Triton:
enable_fp_fusion=False).
Pointwise: the bf16 roundings of (1 + scale), (1 + scale) * norm and + shift take bf16 inputs,
so fp32-op-then-round equals the single-rounding bf16x2 op (double rounding is innocuous for
+ and x when the wide format has >= 2p + 2 bits: 24 >= 18); the NVFP4 quantize is FlashInfer's
fast-math PTX, as in the Triton kernel.

Built with NVRTC once per (device, variant) on first use (model warmup), launched through the
driver API on torch's current stream (CUDA-graph capturable).
"""

import ctypes
import os
import time

import torch

D_SUPPORTED = 3072
_SF_COLS = 192

_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef long long i64;

#ifndef WARPS
#define WARPS 4
#endif
#ifndef USE_PDL
#define USE_PDL 0
#endif
#ifndef IT_MODE
#define IT_MODE 1
#endif
#ifndef BF_ASC
#define BF_ASC 0
#endif
#ifndef INTER_SEQ
#define INTER_SEQ 0
#endif
#ifndef PW_X2
#define PW_X2 1
#endif
// LN_ID=1: the LayerNorm affine is the identity (weight == 1.0 bf16, bias == +0.0 bf16,
// verified on the host by register_identity_affine). fmul(n, 1.0f) == n for every n
// (--ftz=false), so only the +0.0f add stays (it still maps -0 to +0): bitwise to LN_ID=0.
#ifndef LN_ID
#define LN_ID 0
#endif
#define DIM 3072
#define SFC 192

__device__ __forceinline__ float bf_lo(u32 w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(u32 w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ u32 f2bf(float f) {
  unsigned short r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
  return (u32)r;
}
__device__ __forceinline__ float rbf(float f) { return __uint_as_float(f2bf(f) << 16); }
__device__ __forceinline__ u32 word(const uint4 &v, int k) {
  return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w;
}
__device__ __forceinline__ float elem(const uint4 &v, int e) {
  u32 w = word(v, e >> 1);
  return (e & 1) ? bf_hi(w) : bf_lo(w);
}
__device__ __forceinline__ u32 cvt_bf16x2(float hi, float lo) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ u32 add_bf16x2(u32 a, u32 b) {
  u32 r;
  asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ u32 mul_bf16x2(u32 a, u32 b) {
  u32 r;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ u32 max_u16x2(u32 a, u32 b) {
  u32 r;
  asm("max.u16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// Triton lowers an fp32 '/' to div.full.f32 (not IEEE div.rn).
__device__ __forceinline__ float fdiv(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rcp_approx3() {
  float r;
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(3.0f));
  return r;
}

#ifndef FAST_DIV
#define FAST_DIV 1
#endif
#ifndef DGUARD
#define DGUARD 0
#endif
// Welford-reduce divisions by the constant weights 2 and 3. On SM120, div.full.f32(a, 2) ==
// a * 0.5 and div.full.f32(a, 3) == a * rcp.approx(3) bit for bit for EVERY finite fp32 a
// (exhaustive 2^32 check, branch 125 job 1511), so the multiply forms are exact replacements.
__device__ __forceinline__ float div2(float a) {
#if FAST_DIV
  return __fmul_rn(a, 0.5f);
#else
  return fdiv(a, 2.0f);
#endif
}
__device__ __forceinline__ float div3(float a) {
#if FAST_DIV
  return __fmul_rn(a, rcp_approx3());
#else
  return fdiv(a, 3.0f);
#endif
}

// triton_helpers.welford_combine (torch 2.14), no contraction (--fmad=false).
__device__ __forceinline__ void wcomb(float &m1, float &q1, float &w1, float m2, float q2,
                                      float w2) {
#if DGUARD
  float delta = (m1 == m2) ? 0.0f : __fsub_rn(m2, m1);
#else
  // m2 - m1 is +0 when m1 == m2 (finite), i.e. the guard only differs for inf - inf.
  float delta = __fsub_rn(m2, m1);
#endif
  float nw = __fadd_rn(w1, w2);
  float r = (nw == 0.0f) ? 0.0f : fdiv(w2, nw);
  float nm = __fadd_rn(m1, __fmul_rn(delta, r));
  float nq = __fadd_rn(__fadd_rn(q1, q2), __fmul_rn(__fmul_rn(__fmul_rn(delta, delta), w1), r));
  m1 = nm;
  q1 = nq;
  w1 = nw;
}

__device__ __forceinline__ float rcp_approx(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ u32 cvt_e4m3(float x) {
  u32 out;
  asm("{\n .reg .b16 fp8_pair;\n .reg .f32 zero;\n mov.f32 zero, 0f00000000;\n"
      " cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, %1;\n cvt.u32.u16 %0, fp8_pair;\n}"
      : "=r"(out) : "f"(x));
  return out;
}
__device__ __forceinline__ float out_scale(u32 sf_bits, float gs) {
  float out;
  asm("{\n .reg .pred p_zero;\n .reg .b16 fp8_pair;\n .reg .b32 h2_32;\n .reg .b16 h_lo, h_hi;\n"
      " .reg .f32 scale_f32, rcp_gs, product, result;\n cvt.u16.u32 fp8_pair, %1;\n"
      " cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;\n mov.b32 {h_lo, h_hi}, h2_32;\n"
      " cvt.f32.f16 scale_f32, h_lo;\n rcp.approx.ftz.f32 rcp_gs, %2;\n"
      " mul.f32 product, scale_f32, rcp_gs;\n rcp.approx.ftz.f32 result, product;\n"
      " setp.eq.f32 p_zero, scale_f32, 0f00000000;\n selp.f32 %0, 0f00000000, result, p_zero;\n}"
      : "=f"(out) : "r"(sf_bits), "f"(gs));
  return out;
}
// 8 bf16 values (4 bf16x2 words) * out_scale -> 8 e2m1 nibbles (even element in the low nibble).
__device__ __forceinline__ u32 e2m1x8(u32 w0, u32 w1, u32 w2, u32 w3, float os) {
  u32 out;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n .reg .f32 p0, p1, p2, p3, p4, p5, p6, p7;\n"
      " mul.f32 p0, %1, %9;\n mul.f32 p1, %2, %9;\n mul.f32 p2, %3, %9;\n mul.f32 p3, %4, %9;\n"
      " mul.f32 p4, %5, %9;\n mul.f32 p5, %6, %9;\n mul.f32 p6, %7, %9;\n mul.f32 p7, %8, %9;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b0, p1, p0;\n cvt.rn.satfinite.e2m1x2.f32 b1, p3, p2;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b2, p5, p4;\n cvt.rn.satfinite.e2m1x2.f32 b3, p7, p6;\n"
      " mov.b32 %0, {b0, b1, b2, b3};\n}"
      : "=r"(out)
      : "f"(bf_lo(w0)), "f"(bf_hi(w0)), "f"(bf_lo(w1)), "f"(bf_hi(w1)), "f"(bf_lo(w2)),
        "f"(bf_hi(w2)), "f"(bf_lo(w3)), "f"(bf_hi(w3)), "f"(os));
  return out;
}

#ifndef MINB
#define MINB 1
#endif
extern "C" __global__ void __launch_bounds__(WARPS * 32, MINB)
flux3_adaln_nvfp4_warp(const uint4 *__restrict__ x, const u8 *__restrict__ sel,
                       const u16 *__restrict__ scale, const u16 *__restrict__ shift,
                       const uint4 *__restrict__ lnw, const uint4 *__restrict__ lnb,
                       const float *__restrict__ gs_ptr, uint4 *__restrict__ xmod,
                       u32 *__restrict__ fp4, u8 *__restrict__ sf, int M, int padded,
                       i64 scale_rs, i64 shift_rs, float eps, int has_select, int store_xmod) {
#if USE_PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * WARPS + (threadIdx.x >> 5);
  if (row >= padded) return;
  const i64 rowpart = (i64)(row & 31) * 16 + (i64)((row & 127) >> 5) * 4 + (i64)(row >> 7) * (128 * SFC);
  if (row >= M) {
    // Padding rows of the 128x4 layout: zero scale factors (48 u32 per row).
    for (int k = lane; k < SFC / 4; k += 32) *(u32 *)(sf + (i64)k * 512 + rowpart) = 0u;
    return;
  }
  const uint4 *xr = x + (i64)row * (DIM / 8);
  uint4 v[12];
#pragma unroll
  for (int i = 0; i < 12; ++i) v[i] = xr[i * 32 + lane];

  // ---- Inductor's looped Welford per slot (chunks c = 0, 1, 2), then the in-thread combine
  // of the 4 slots of each virtual thread k = 2 g + l0 (slot j = column 4 l0 + j of v[.]) ----
  float vm[8], vq[8];
  float wt = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int g = k >> 1;
    float mean[4], m2[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int e = (k & 1) * 4 + j;
      const int s = j;
      float mu = elem(v[g], e);
      float q = 0.0f;
      float v1 = elem(v[4 + g], e);
      float d1 = __fsub_rn(v1, mu);
      mu = __fadd_rn(mu, div2(d1));
      q = __fadd_rn(q, __fmul_rn(d1, __fsub_rn(v1, mu)));
      float v2 = elem(v[8 + g], e);
      float d2 = __fsub_rn(v2, mu);
      mu = __fadd_rn(mu, div3(d2));
      q = __fadd_rn(q, __fmul_rn(d2, __fsub_rn(v2, mu)));
      mean[s] = mu;
      m2[s] = q;
    }
    const int s = 0;
    float am = mean[s], aq = m2[s], aw = 3.0f;
#if IT_MODE == 0
    wcomb(am, aq, aw, mean[s + 1], m2[s + 1], 3.0f);
    wcomb(am, aq, aw, mean[s + 2], m2[s + 2], 3.0f);
    wcomb(am, aq, aw, mean[s + 3], m2[s + 3], 3.0f);
#elif IT_MODE == 1
    wcomb(am, aq, aw, mean[s + 1], m2[s + 1], 3.0f);
    float bm = mean[s + 2], bq = m2[s + 2], bw = 3.0f;
    wcomb(bm, bq, bw, mean[s + 3], m2[s + 3], 3.0f);
    wcomb(am, aq, aw, bm, bq, bw);
#else
    wcomb(am, aq, aw, mean[s + 2], m2[s + 2], 3.0f);
    float bm = mean[s + 1], bq = m2[s + 1], bw = 3.0f;
    wcomb(bm, bq, bw, mean[s + 3], m2[s + 3], 3.0f);
    wcomb(am, aq, aw, bm, bq, bw);
#endif
    vm[k] = am;
    vq[k] = aq;
    wt = aw;
  }
  // ---- 32-lane butterfly per virtual warp ----
#if BF_ASC
  {
    // virtual xor 1 (in-register) first, both virtual lanes
    float nw = wt;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      float am = vm[2 * g], aq = vq[2 * g], aw = wt;
      float bm = vm[2 * g + 1], bq = vq[2 * g + 1], bw = wt;
      wcomb(am, aq, aw, vm[2 * g + 1], vq[2 * g + 1], wt);
      wcomb(bm, bq, bw, vm[2 * g], vq[2 * g], wt);
      vm[2 * g] = am; vq[2 * g] = aq; vm[2 * g + 1] = bm; vq[2 * g + 1] = bq;
      nw = aw;
    }
    wt = nw;
  }
#pragma unroll
  for (int off = 1; off <= 8; off <<= 1) {
#else
#pragma unroll
  for (int off = 8; off >= 1; off >>= 1) {
#endif
    float nw = wt;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float om = __shfl_xor_sync(0xffffffffu, vm[k], off);
      float oq = __shfl_xor_sync(0xffffffffu, vq[k], off);
      float w1 = wt;
      wcomb(vm[k], vq[k], w1, om, oq, wt);
      nw = w1;
    }
    wt = nw;
  }
  float pm[4], pq[4];
  float pw = wt;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    float m = vm[2 * g], q = vq[2 * g], w = wt;
#if !BF_ASC
    // virtual xor 1 last: only virtual lane 0 (l0 = 0) is read afterwards.
    wcomb(m, q, w, vm[2 * g + 1], vq[2 * g + 1], wt);
#endif
    pm[g] = m;
    pq[g] = q;
    pw = w;
  }
  // ---- 8 warp partials P_w, w = 2 g + (lane >> 4): P_2g at lane 0, P_2g+1 at lane 16 ----
#if INTER_SEQ
  float fm = __shfl_sync(0xffffffffu, pm[0], 0), fq = __shfl_sync(0xffffffffu, pq[0], 0), fw = pw;
#pragma unroll
  for (int w = 1; w < 8; ++w) {
    float om = __shfl_sync(0xffffffffu, pm[w >> 1], (w & 1) * 16);
    float oq = __shfl_sync(0xffffffffu, pq[w >> 1], (w & 1) * 16);
    wcomb(fm, fq, fw, om, oq, pw);
  }
#else
  // xor 4, 2, 1 at lane 0: comb(comb(comb(P0,P4), comb(P2,P6)), comb(comb(P1,P5), comb(P3,P7)))
  float am = pm[0], aq = pq[0], aw = pw;
  wcomb(am, aq, aw, pm[2], pq[2], pw);
  float bm = pm[1], bq = pq[1], bw = pw;
  wcomb(bm, bq, bw, pm[3], pq[3], pw);
  wcomb(am, aq, aw, bm, bq, bw);
  float fm = __shfl_sync(0xffffffffu, am, 0);
  float fq = __shfl_sync(0xffffffffu, aq, 0);
  float cm = __shfl_sync(0xffffffffu, am, 16);
  float cq = __shfl_sync(0xffffffffu, aq, 16);
  float fw = aw;
  wcomb(fm, fq, fw, cm, cq, aw);
#endif

  const float mu = fm;
  const float rstd = rsqrtf(__fadd_rn(fdiv(fq, 3072.0f), eps));

  // ---- pointwise modulate + NVFP4 quantize ----
  const float gs = *gs_ptr;
  const float fp4_max_rcp = rcp_approx(6.0f);
  bool s = false;
  if (has_select) s = sel[row] != 0;
  const uint4 *scr = (const uint4 *)(scale + (s ? scale_rs : 0));
  const uint4 *shr = (const uint4 *)(shift + (s ? shift_rs : 0));
  uint4 *xo = xmod + (i64)row * (DIM / 8);
  u32 *fo = fp4 + (i64)row * (DIM / 8);
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    const int vi = i * 32 + lane;  // 16-byte column index: columns 256 i + 8 lane .. +7
    const uint4 sc = scr[vi];
    const uint4 sh = shr[vi];
#if !LN_ID
    const uint4 wv = lnw[vi];
    const uint4 bv = lnb[vi];
#endif
    u32 xmw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#if LN_ID
      const u32 xw = word(v[i], k);
      const float t0 = __fadd_rn(__fmul_rn(__fsub_rn(bf_lo(xw), mu), rstd), 0.0f);
      const float t1 = __fadd_rn(__fmul_rn(__fsub_rn(bf_hi(xw), mu), rstd), 0.0f);
#else
      const u32 xw = word(v[i], k), ww = word(wv, k), bw = word(bv, k);
      const float t0 = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(bf_lo(xw), mu), rstd), bf_lo(ww)), bf_lo(bw));
      const float t1 = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(bf_hi(xw), mu), rstd), bf_hi(ww)), bf_hi(bw));
#endif
#if PW_X2
      const u32 t50 = cvt_bf16x2(t1, t0);
      const u32 s1p = add_bf16x2(word(sc, k), 0x3F803F80u);
      xmw[k] = add_bf16x2(mul_bf16x2(s1p, t50), word(sh, k));
#else
      const u32 scw = word(sc, k), shw = word(sh, k);
      float r0 = rbf(__fadd_rn(rbf(__fmul_rn(rbf(__fadd_rn(bf_lo(scw), 1.0f)), rbf(t0))), bf_lo(shw)));
      float r1 = rbf(__fadd_rn(rbf(__fmul_rn(rbf(__fadd_rn(bf_hi(scw), 1.0f)), rbf(t1))), bf_hi(shw)));
      xmw[k] = f2bf(r0) | (f2bf(r1) << 16);
#endif
    }
    if (store_xmod) xo[vi] = make_uint4(xmw[0], xmw[1], xmw[2], xmw[3]);
    // amax of |bf16| as unsigned 16-bit compare (non-negative floats order like their bits)
    u32 a2 = max_u16x2(max_u16x2(xmw[0] & 0x7FFF7FFFu, xmw[1] & 0x7FFF7FFFu),
                       max_u16x2(xmw[2] & 0x7FFF7FFFu, xmw[3] & 0x7FFF7FFFu));
    u32 ab = max(a2 & 0xFFFFu, a2 >> 16);
    ab = max(ab, __shfl_xor_sync(0xffffffffu, ab, 1));
    const float amax = __uint_as_float(ab << 16);
    const float scale_float = __fmul_rn(gs, __fmul_rn(amax, fp4_max_rcp));
    const u32 sfb = cvt_e4m3(scale_float);
    const float os = out_scale(sfb, gs);
    fo[vi] = e2m1x8(xmw[0], xmw[1], xmw[2], xmw[3], os);
    // SF column 64 c + 16 g + lane/2 = 16 i + lane/2; 4 columns (lanes 8m..8m+7) -> one u32.
    const u32 s1 = __shfl_down_sync(0xffffffffu, sfb, 2);
    const u32 s2 = __shfl_down_sync(0xffffffffu, sfb, 4);
    const u32 s3 = __shfl_down_sync(0xffffffffu, sfb, 6);
    if ((lane & 7) == 0) {
      const u32 wd = (sfb & 0xFFu) | ((s1 & 0xFFu) << 8) | ((s2 & 0xFFu) << 16) | ((s3 & 0xFFu) << 24);
      *(u32 *)(sf + (i64)(4 * i + (lane >> 3)) * 512 + rowpart) = wd;
    }
  }
}
"""

# Launch bounds: 1 warp (row) per CTA, MINB=12 -> 167 registers, 0 B local (12 warps/SM). The
# previous default (4 warps, MINB=4: 128 regs + a 24 B spill frame, 16 warps/SM) ran 15.1-15.2
# us/call vs 12.4-12.6 here at M=4176 select with a gated-residual stand-in writing x before each
# call (branch 183 jobs 2038/2044, bitwise at every tested M/form/gs). Launch bounds change only
# register allocation, never the instruction math, so every config is bitwise to every other.
_DEFAULT_WARPS = 1
_DEFAULT_DEFINES = ("MINB=12",)
_MODULES = {}
COMPILE_SECONDS = {}
NUM_REGS = {}

# LN_ID (exact): launch() builds the LN_ID=1 variant (no lnw/lnb loads, no weight multiply) only
# for a (weight, bias) pair registered by register_identity_affine, i.e. verified bitwise 1.0 /
# +0.0 by the caller outside capture, and unchanged since (same data_ptr and _version).
# TRTLLM_FLUX3_ADALN_LN_ID=0 restores the affine launch byte for byte.
_LN_ID = os.environ.get("TRTLLM_FLUX3_ADALN_LN_ID", "1").strip().lower() not in (
    "0",
    "off",
    "false",
    "",
)
_IDENTITY_AFFINE = {}


def _version_of(t: torch.Tensor):
    try:
        return t._version
    except RuntimeError:  # inference tensors carry no version counter
        return None


def register_identity_affine(weight: torch.Tensor, bias: torch.Tensor) -> None:
    """Record (weight, bias) as an identity LayerNorm affine (caller verified the bits)."""
    _IDENTITY_AFFINE[(weight.data_ptr(), bias.data_ptr())] = (_version_of(weight), _version_of(bias))


def unregister_identity_affine(weight: torch.Tensor, bias: torch.Tensor) -> None:
    _IDENTITY_AFFINE.pop((weight.data_ptr(), bias.data_ptr()), None)


def is_identity_affine(weight: torch.Tensor, bias: torch.Tensor) -> bool:
    ver = _IDENTITY_AFFINE.get((weight.data_ptr(), bias.data_ptr()))
    return ver is not None and ver == (_version_of(weight), _version_of(bias))


def _check(result):
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"flux3 sm120 adaln nvfp4: CUDA/NVRTC error {err}")
    return result[1] if len(result) == 2 else result[1:]


def _compile(device: torch.device, warps: int, use_pdl: bool, extra_defines=()):
    key = (device.index, warps, use_pdl, tuple(extra_defines))
    fn = _MODULES.get(key)
    if fn is not None:
        return fn
    from cuda.bindings import driver, nvrtc

    t0 = time.perf_counter()
    major, minor = torch.cuda.get_device_capability(device)
    prog = _check(nvrtc.nvrtcCreateProgram(_SRC.encode(), b"flux3_adaln_nvfp4.cu", 0, [], []))
    opts = [
        f"--gpu-architecture=sm_{major}{minor}a".encode(),
        b"--fmad=false",
        b"--prec-div=true",
        b"--prec-sqrt=true",
        b"--ftz=false",
        b"-std=c++17",
        f"-DWARPS={warps}".encode(),
        f"-DUSE_PDL={1 if use_pdl else 0}".encode(),
    ] + [f"-D{d}".encode() for d in extra_defines]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(err) != 0:
        log_size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"flux3 sm120 adaln nvfp4: NVRTC failed:\n{log.decode(errors='replace')}")
    size = _check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * size
    _check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    nvrtc.nvrtcDestroyProgram(prog)
    with torch.cuda.device(device):
        torch.cuda.current_stream(device)  # make torch's primary context current
        module = _check(driver.cuModuleLoadData(cubin))
        fn = _check(driver.cuModuleGetFunction(module, b"flux3_adaln_nvfp4_warp"))
        NUM_REGS[key] = _check(
            driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)
        )
    _MODULES[key] = fn
    COMPILE_SECONDS[key] = time.perf_counter() - t0
    from tensorrt_llm.logger import logger

    logger.info(
        f"flux3 sm120 adaln nvfp4 warp kernel built for {key} in {COMPILE_SECONDS[key]:.3f} s"
        f" ({NUM_REGS[key]} regs)"
    )
    return fn


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def supported(x, select, scale, shift, weight, bias) -> bool:
    """Shapes/layouts the warp kernel handles (else the caller uses the Triton kernel)."""
    d = x.shape[-1]
    if not (
        d == D_SUPPORTED
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and torch.cuda.get_device_capability(x.device) == (12, 0)
        and _aligned(x)
        and weight.dtype == torch.bfloat16
        and bias.dtype == torch.bfloat16
        and weight.is_contiguous()
        and bias.is_contiguous()
        and weight.numel() == d
        and bias.numel() == d
        and _aligned(weight)
        and _aligned(bias)
        and scale.dtype == torch.bfloat16
        and shift.dtype == torch.bfloat16
        and scale.stride(-1) == 1
        and shift.stride(-1) == 1
        and _aligned(scale)
        and _aligned(shift)
    ):
        return False
    if select is not None:
        if not (
            select.dtype == torch.bool
            and select.is_contiguous()
            and select.numel() == x.numel() // d
            and scale.dim() == 3
            and shift.dim() == 3
            and (scale.stride(1) * 2) % 16 == 0
            and (shift.stride(1) * 2) % 16 == 0
        ):
            return False
    return True


def launch(
    x,
    select,
    scale,
    shift,
    weight,
    bias,
    global_scale,
    eps,
    store_x_mod=True,
    outs=None,
    warps=_DEFAULT_WARPS,
    use_pdl=False,
    extra_defines=_DEFAULT_DEFINES,
):
    """Same contract as flux3_adaln_quant._launch: returns (x_mod, fp4 [M, D/2] u8, sf 128x4)."""
    from cuda.bindings import driver

    d = x.shape[-1]
    m = x.numel() // d
    padded = ((m + 127) // 128) * 128
    has_select = select is not None
    if outs is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        fp4 = torch.empty((m, d // 2), dtype=torch.uint8, device=x.device)
        sf = torch.empty((padded * _SF_COLS,), dtype=torch.uint8, device=x.device)
    else:
        out, fp4, sf = outs
    if _LN_ID and is_identity_affine(weight, bias):
        extra_defines = tuple(extra_defines) + ("LN_ID=1",)
    fn = _compile(x.device, warps, use_pdl, extra_defines)
    grid = (padded + warps - 1) // warps
    stream = torch.cuda.current_stream(x.device).cuda_stream
    vp = ctypes.c_void_p
    args = (
        x.data_ptr(),
        select.data_ptr() if has_select else x.data_ptr(),
        scale.data_ptr(),
        shift.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        global_scale.data_ptr(),
        out.data_ptr(),
        fp4.data_ptr(),
        sf.data_ptr(),
        m,
        padded,
        scale.stride(1) if has_select else 0,
        shift.stride(1) if has_select else 0,
        float(eps),
        int(has_select),
        int(bool(store_x_mod)),
    )
    types = (vp,) * 10 + (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    )
    _check(
        driver.cuLaunchKernel(
            fn, grid, 1, 1, warps * 32, 1, 1, 0, driver.CUstream(stream), (args, types), 0
        )
    )
    return out, fp4, sf
