# Idea 292 S2.0: standalone fused_pmap_gemm_fma_mtile_v3<TM, NG, MIN_BLOCKS> sweep

Job 1732, node smc521ge-0043, austin:2 (clean start: 0% util, 33.8 W, 405 MHz mem clock), commit e63098b9.
`sweep.py` builds `proto.cu` with nvcc sm_120 `-Xptxas -v`. Synthetic DeepSeek-V4.1-Flash shapes:
hc=4, hidden=5120, N=24, K=20480. Every call (3 warmup + 20 timed) has NaN-poisoned outputs and is compared
bitwise against `fused_pmap_gemm_fma_ksplit<24,1,0,true>` on Yp, Rp and residual_out. Cold L2: a 256 MB
memset before each call. Times are medians in us. Noise is (p90-p10)/median and is at most 6.6%.

| kernel | regs | STACK | spill st/ld B | M=33 | 128 | 512 | 2047 | 2048 | 4096 | 8192 | bitwise bad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1<24,4> (main) | 255 | 408 | 832/1300 | 105.5 | 115.7 | 150.5 | 359.4 | 361.5 | 607.2 | 1063.9 | 0 |
| v3<8,8,1>  | 168 | 0 | 0/0 | **88.1** | **91.1** | **117.7** | 246.8 | 246.8 | **345.1** | **617.5** | 0 |
| v3<8,4,1>  | 227 | 0 | 0/0 | 97.3 | 99.3 | 125.9 | 263.2 | 262.1 | 365.6 | 635.9 | 0 |
| v3<8,2,1>  | 255 | 0 | 0/0 | 111.6 | 112.6 | 137.2 | 298.0 | 295.9 | 414.7 | 736.3 | 0 |
| v3<16,8,1> | 254 | 0 | 0/0 | 122.9 | 124.9 | 140.3 | **216.1** | **216.1** | 384.0 | 646.1 | 0 |
| v3<16,4,1> | 255 | 24 | 24/24 | 136.2 | 139.3 | 154.6 | 226.3 | 226.3 | 406.5 | 656.4 | 0 |
| v3<16,2,1> | 255 | 176 | 340/176 | 175.1 | 186.4 | 193.5 | 265.2 | 265.2 | 500.7 | 850.9 | 0 |

TM=24 and TM=32 cannot launch. They need 96/128 KB of dynamic smem for new_r plus 18/24 KB of static smem,
which is above the SM120 per-block opt-in limit (~99 KB).

## Idea 348 S0 (branch 198, GPU job 1807, smc521ge-0043 austin:4, main c900305b build, commit fd66298c)

### S0.P probe (probe348.py, no source edit)
- TRTLLM_CACHE_DIR and TLLM_AUTOTUNER_CACHE_PATH unset before and after `import tensorrt_llm`; no cache file.
- libtensorrt_llm.so of this job (slot 4, mtime 01:25 11-06, after landing 61 = 146d3e13 at 22:30 11-05):
  6 `fused_pmap_gemm_fma_mtile_v3` strings. libth_common.so holds none (only mhcOp.cpp is there).
  Other slots' builds at 01:3x: slots 0/4/6 have mtile_v3 (6), slots 1/3/5/7 have 0 (v2 present),
  slot 2 has neither v2 nor v3. These are per-branch candidate builds, so a 0 count shows that slot last built a
  pre-61 branch. It does not show that a main build is stale.
- Direct launch of ('fused_half_fma',24,1,256,72) at M=2048 (lagged, norm_weight): ok, bitwise vs ks=1 on all 5 outputs.
  Same for id 20.
- Fresh tuner, per M (clear_cache, autotune(cache_path=None, skip_dynamic_tuning_buckets=True)), times in ms:

| M | bucket | pick | id 20 | id 72 |
|---|---|---|---|---|
| 128 | 128 | (12,1,512,2) | 0.050 | 0.072 |
| 512 | 512 | id 72 | 0.078 | 0.077 |
| 1024 | 1024 | id 72 | 0.147 | 0.095 |
| 2048 | 2048 | id 72 | 0.307 | 0.223 |
| 2176 | 2176 | id 72 | 0.319 | 0.210 |
| 4096 | 4096 | id 72 | 0.618 | 0.439 |
| 8192 | 8192 | id 72 | 1.117 | 0.869 |

  Production-like run (one autotune over every bucket at M=8192, then the pick per M) gives the same picks,
  except that M=512 picks id 20 (a 0.077/0.078 tie). Verdict: (i), (ii) and (iii) hold, so not (b1)/(b2)/(b3).
  Profile 1783's id 20 points to (a), a pre-61 build.

### S0.S standalone sweep (sweep.py + proto.cu + v3x.cuh), kernel-only us, cold L2, bitwise vs ksplit ks=1 every call
v3x = v3 with (A): TM need not be a multiple of 8. Pmap slots t = warp + 8i >= TM are skipped at compile time, so
TM % 8 == 0 is unchanged code. (B) sets the dynamic-smem opt-in when dynamic + static smem > 48 KB. Without it,
TM 9-12 cannot launch: dynamic smem is <= 48 KB but the total is above. All builds had 0 bad iterations of 23,
at every M, including the 129 and 2047 tails.

| kernel | regs | STACK | static B | dyn KB | 129 | 512 | 1024 | 2047 | 2048 (grid) | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2<24,4> id 20 | 255 | 0 | 608 | 80 | 89.1 | 115.7 | 187.4 | 285.7 | 285.7 (512) | 492.5 | 844.8 |
| v3<8,8,1> id 72 | 168 | 0 | 7040 | 32 | 91.1 | 117.8 | 149.5 | 248.8 | 248.8 (256) | 349.2 | 619.5 |
| v3<8,8,2> | 128 | **40** (68/92 B spill) | 7040 | 32 | 100.4 | 125.9 | 158.7 | 212.0 | 212.0 (256) | 369.7 | 594.9 |
| v3<16,8,1> | 254 | 0 | 14080 | 64 | 128.0 | 142.3 | 166.9 | 218.1 | 218.1 (128) | 388.1 | 644.1 |
| v3x<4,8,1> | 164 | 0 | 3520 | 16 | 79.9 | 113.6 | 182.3 | 265.2 | 265.2 (512) | 457.7 | 810.0 |
| v3x<8,8,1> | 168 | 0 | 7040 | 32 | 91.1 | 117.8 | 148.5 | 248.8 | 248.8 (256) | 349.2 | 621.6 |
| v3x<9,8,1> | 219 | 0 | 7920 | 36 | 91.1 | 118.8 | 150.5 | 261.1 | 259.1 (228) | 365.6 | 592.9 |
| v3x<10,8,1> | 220 | 0 | 8800 | 40 | 94.2 | 119.8 | 152.5 | 275.5 | 275.5 (205) | 390.1 | 627.7 |
| **v3x<11,8,1>** | 229 | 0 | 9680 | 44 | 98.3 | 121.9 | 149.5 | **203.8** | **207.9** (187) | **338.9** | 597.0 |
| v3x<12,8,1> | 228 | 0 | 10560 | 48 | 99.3 | 126.0 | 154.6 | 214.0 | 214.0 (171) | 341.0 | 605.2 |
| v3x<13,8,1> | 236 | 0 | 11440 | 52 | 106.5 | 126.0 | 158.7 | 212.0 | 212.0 (158) | 351.2 | 638.0 |
| v3x<14,8,1> | 242 | 0 | 12320 | 56 | 111.6 | 132.1 | 154.6 | 205.8 | 205.8 (147) | 361.5 | 699.4 |
| v3x<15,8,1> | 247 | 0 | 13200 | 60 | 119.8 | 130.0 | 156.7 | 212.0 | 214.0 (137) | 373.8 | 629.8 |
| v3x<16,8,1> | 254 | 0 | 14080 | 64 | 128.0 | 142.3 | 166.9 | 218.1 | 218.1 (128) | 390.1 | 648.2 |

Noise ((p90-p10)/median) is at most 5.8%, and 2-4% on the M=2048 rows. Pick for 348 S1: **v3x<11,8,1> = tactic id 75**
(tile_m 64+11). At M=2048 it runs as one wave of 187 CTAs on 188 SMs: 4.7% under TM=16 (noise 3.9%) and 16% under id 72.
It is also the best build at 4096 and within 0.7% of the best at 8192, where TM=16 loses to id 72. v3x<14> ties it at 2048
but is slower at 4096/8192.

Pick for submission 1: **v3<8,8,1>** as tactic id tile_m = 64 + 8 = 72 (tile_n 24, ks 1). It is the only build
that is spill-free and faster than v1<24,4> at every M, M=33 and M=128 included. v3<16,8,1> is 13% faster at
M~2048 but slower below M=512. It is a candidate for a later, separate tactic id.
