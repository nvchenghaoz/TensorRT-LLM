"""Kernel census of nsys timeline.sqlite exports: launches/total time of the generic-route kernels
(silu_and_mul, quantize_with_block_size, _qkv_amax_partials, _qkv_quantize, cuBLASLt NVFP4 nvjet
ootst, TRT-LLM DeviceGemmFp4GemmSm120) and the prepared-route ones, split by grid size."""
import json
import sqlite3
import sys

PATTERNS = ["silu_and_mul", "quantize_with_block_size", "_qkv_amax_partials", "_qkv_quantize",
            "nvjet_sm120_ootst", "DeviceGemmFp4GemmSm120", "flux3_adaln_nvfp4", "qknorm_rope_fp8",
            "nl_tiled_g3", "_nfo_", "NVFP4QuantizeSwizzled", "tunable_fp4_quantize",
            "GroupProblemShape"]
out = {}
for label, path in zip(sys.argv[1::2], sys.argv[2::2]):
    con = sqlite3.connect(path)
    rows = con.execute(
        "SELECT s.value, k.gridX*k.gridY*k.gridZ, COUNT(*), SUM(k.end-k.start) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id "
        "GROUP BY s.value, k.gridX*k.gridY*k.gridZ").fetchall()
    res = {}
    for pat in PATTERNS:
        hits = [(name, grid, n, t) for name, grid, n, t in rows if pat in name]
        res[pat] = {"launches": sum(h[2] for h in hits),
                    "total_ms": round(sum(h[3] for h in hits) / 1e6, 3),
                    "by_grid": {str(h[1]): [h[2], round(h[3] / 1e6, 3)] for h in hits}}
    res["all_kernels"] = sum(r[2] for r in rows)
    out[label] = res
    print(label, json.dumps(res, indent=1), flush=True)
print(json.dumps({"passed": True, "census": out}))
