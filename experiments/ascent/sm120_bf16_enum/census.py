"""Stage A step 1: cuBLASLt BF16 (FP32 compute) algo census on this GPU. No timing.

Per algo id from cublasLtMatmulAlgoGetIds: every CAP attribute (tile ids, stages ids, custom option max,
split-K support, reduction mask, swizzle support, cluster ids if the header has that cap).
Then counts AlgoCheck-valid configs over tile x stages x custom (cga 0/auto, splitK 1, swizzle 0) per algo at
sl_qkv M=300 and times AlgoCheck, and lists the heuristic top-32 at M=300 for each shape (sweep3's pool source).

usage: python census.py <checkpoint>
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ltx import LtX, read_shapes  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def main():
    ckpt = sys.argv[1]
    shapes, info = read_shapes(ckpt)
    log("config", json.dumps(info), "shapes", json.dumps(shapes))
    lt = LtX()
    log("cublasLt", lt.path, lt.version, "header", lt.hdr, torch.cuda.get_device_name())
    log("cap enum", json.dumps(lt.caps))
    log("cluster enum", json.dumps(lt.clusters))
    ids = lt.algo_ids()
    log("algo ids", ids)
    caps = {}
    for a in ids:
        c = lt.caps_of(a)
        caps[a] = c
        if c is None:
            log(f"algo {a}: init failed")
            continue
        brief = {k: (v if not isinstance(v, list) else f"{len(v)} ids") for k, v in c.items()}
        log(f"algo {a}: {json.dumps(brief)}")
        log(f"algo {a} tiles: {[(t, lt.tile_name(t)) for t in (c.get('TILE_IDS') or [])]}")
        log(f"algo {a} stages: {[(s, lt.stages.get(s, s)) for s in (c.get('STAGES_IDS') or [])]}")
        if c.get("CLUSTER_SHAPE_IDS"):
            log(f"algo {a} clusters: {c['CLUSTER_SHAPE_IDS']}")
    # heuristic lists at M=300
    for sname, (N, K) in shapes.items():
        for M in (300, 1000, 2000):
            h = lt.heuristics(M, N, K, 32)
            log(f"heur {sname} M={M}: " + " | ".join(lt.name(c8) for (c8, inn), ws, wv, _ in h))
    # AlgoCheck enumeration count at sl_qkv M=300 and sl_o M=300
    for sname in ("sl_qkv", "sl_o"):
        N, K = shapes[sname]
        M = 300
        for a in ids:
            c = caps.get(a)
            if not c:
                continue
            tiles = c.get("TILE_IDS") or [0]
            stages = c.get("STAGES_IDS") or [0]
            cmax = c.get("CUSTOM_OPTION_MAX") or 0
            t1 = time.time()
            n_try = n_ok = 0
            valid_tiles = set()
            customs = set()
            for t in tiles:
                for s in stages:
                    for cu in range(cmax + 1):
                        algo = lt.make_algo_raw((a, t, s, 1, 0, 0, cu, 0))
                        n_try += 1
                        if algo is None:
                            continue
                        ok, ws, wv = lt.check_ws(M, N, K, algo)
                        if ok:
                            n_ok += 1
                            valid_tiles.add(t)
                            customs.add(cu)
                if time.time() - t1 > 120:
                    log(f"  algo {a}: stopping enumeration after 120 s")
                    break
            dt = time.time() - t1
            log(f"enum {sname} M={M} algo {a}: tried {n_try}, valid {n_ok}, {dt:.1f}s "
                f"({dt / max(n_try, 1) * 1e6:.1f} us/try); valid tiles {len(valid_tiles)}: "
                f"{sorted(lt.tile_name(t) for t in valid_tiles)}; customs {sorted(customs)}")
    print(json.dumps({"done": True, "ids": ids}))


if __name__ == "__main__":
    main()
