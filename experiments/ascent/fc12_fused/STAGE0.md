# Idea 201 Stage 0: census of the FI SM12x decode-MoE chain (main 2fa8e034)

Jobs 1064 (EP4, 96 x 2304) and 1067 (EP2xTP2, 192 x 1152), 1 GPU each (austin:0 / austin:1),
script `experiments/ascent/fc12_fused/stage0.py`. Production leaf call sequence
(route_assign_decode -> q0_route_direct -> fc1_act_q1 -> fc2_finalize, PDL on) in a CUDA graph,
512 MB READ-ONLY flush before each replay (pitfall 1356), Kineto per node, p50 of 30 replays.
Span = route_assign start -> last GEMM end (us). Primary layout = EP4 (engine.json on main has
no MoE layout override).

Arms: A chain (BASE); B no fc2; C fc2 forked onto a 2nd stream right after the route (no
fc1->fc2 dependency, timing only: hard UPPER BOUND on what any fc1/fc2 overlap can recover);
R A with the active experts' fc2 weights pre-read into L2.
exposed_fc2 = A - B. UB_overlap = A - C. UB_l2warm = A - R.

## EP4 (primary), M=4 rows (M=1 rows equal within noise for D<=6)

| D | A chain | exposed fc2 | fc2 byte floor | UB_overlap | UB_l2warm |
|---|---|---|---|---|---|
| 0 | 8.10 | 2.62 | 0 | 0.45 | 0.0 |
| 1 | 28.35 | 10.56 | 4.16 | 5.95 | 0.32 |
| 2 | 38.14 | 12.93 | 8.31 | 3.20 | 1.66 |
| 3 | 50.98 | 16.42 | 12.47 | 4.51 | 5.18 |
| 4 | 63.01 | 20.99 | 16.62 | 3.46 | 7.46 |
| 5 | 79.71 | 30.30 | 20.78 | 6.88 | 9.98 |
| 6 | 89.57 | 31.71 | 24.94 | 5.73 | 8.70 |
| 7 | 101.31 | 34.27 | 29.09 | 4.22 | 6.82 |
| 8 | 113.22 | 38.53 | 33.25 | 5.15 | -3.58 |
| 9 | 124.70 | 43.10 | 37.40 | 4.99 | -6.94 |
| 10 | 141.25 | 49.76 | 41.56 | 4.70 | -10.05 |
| 11 | 166.31 | 52.90 | 45.71 | 9.79 | -9.38 |
| 12 | 172.26 | 56.45 | 49.87 | 7.62 | -9.12 |

M=1: D0 8.19, D1 29.06 (UB 6.62), D2 37.95 (3.39), D3 52.06 (4.86), D4 63.39 (3.65), D5 80.64, D6 90.15.

Busiest-rank D (uniform top-6 Monte Carlo): M1 {2: .356, 3: .501, 4: .127, 5: .016};
M2 {4: .40, 5: .37, 6: .15}; M4 {7: .24, 8: .33, 9: .23, 10: .11}.
D-weighted at that mix: UB_overlap M1 4.22 / M2 5.15 / M4 5.07 us/layer; chain 49.0 / 74.8 / 118.2.

## EP2xTP2 (secondary), M=4 rows

| D | A | exposed fc2 | floor | UB_overlap |
|---|---|---|---|---|
| 0 | 10.21 | 3.65 | 0 | 0.70 |
| 1 | 25.44 | 8.16 | 2.08 | 5.70 |
| 2 | 28.64 | 9.38 | 4.16 | 5.89 |
| 3 | 33.57 | 11.07 | 6.23 | 2.53 |
| 4 | 39.74 | 13.09 | 8.31 | 2.98 |
| 5 | 48.99 | 18.40 | 10.39 | 4.90 |
| 6 | 55.14 | 19.84 | 12.47 | 5.95 |
| 7 | 60.29 | 21.54 | 14.55 | 3.36 |
| 8 | 66.59 | 23.62 | 16.62 | 3.55 |
| 9 | 72.77 | 26.14 | 18.70 | 3.71 |
| 10 | 80.19 | 29.92 | 20.78 | 5.92 |
| 11 | 86.02 | 31.30 | 22.86 | 6.02 |
| 12 | 92.00 | 33.22 | 24.94 | 5.22 |

Busiest-rank D: M1 {3: .31, 4: .47, 5: .18, 6: .03}. Mix UB_overlap: M1 4.45 / M2 4.17 / M4 5.28.

## Reading

- An EMPTY fc2_finalize (D=0) costs 2.62 us exposed at EP4 (3.65 at EP2xTP2): launch, prologue,
  griddepcontrol wait for fc1's full completion, teardown. A fused kernel pays none of that.
- fc2's boundary is already PDL-overlapped (fc2 starts 0.45-0.55 us before fc1 ends), and its
  weight loader (load_warp_1) already issues TMA before griddepcontrol_wait.
- UB_overlap (no dependency at all) is 4.2 us/layer at the c1 EP4 mix. The recoverable part for
  a dependency-preserving fused kernel lies between the empty-fc2 cost (~2.6) and that bound.
- fc2 weights warm in L2 save <= 1.7 us at D <= 2: fc2 is bound by its per-CTA rate (40 tiles
  per expert, 32x128 tile), not DRAM latency (agrees with dead end 1770).
- GATE (recoverable >= 2 us/layer at the primary mix): PASSES (2.6 fixed .. 4.2 bound).
