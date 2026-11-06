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

Pick for submission 1: **v3<8,8,1>** as tactic id tile_m = 64 + 8 = 72 (tile_n 24, ks 1). It is the only build
that is spill-free and faster than v1<24,4> at every M, M=33 and M=128 included. v3<16,8,1> is 13% faster at
M~2048 but slower below M=512. It is a candidate for a later, separate tactic id.
