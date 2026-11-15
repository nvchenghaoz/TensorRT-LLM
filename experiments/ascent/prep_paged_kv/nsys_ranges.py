"""Count kernels per NVTX range in an nsys sqlite export (llm_ab.py nsys mode).

Usage: python nsys_ranges.py <timeline.sqlite>
For each range (prep_direct_on / prep_direct_off), prints launch counts and total
GPU time of the prep kernel, AppendPagedKVCacheKernel (per head dim) and the FP8
cast (ConvertToFloat8 / float8_copy) kernels whose GPU start lies inside the
range's [start, end]. Last stdout line: JSON.
"""
import json
import sqlite3
import sys


def main(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    nv = "NVTX_EVENTS"
    strings = "StringIds"
    rng = {}
    for start, end, text, tid in cur.execute(
            f"select start, end, text, textId from {nv} where end is not null"):
        name = text
        if name is None and tid is not None:
            r = con.execute(f"select value from {strings} where id=?", (tid,)).fetchone()
            name = r[0] if r else None
        if name in ("prep_direct_on", "prep_direct_off"):
            rng[name] = (start, end)
    kt = "CUPTI_ACTIVITY_KIND_KERNEL"
    out = {}
    for name, (s, e) in rng.items():
        rows = cur.execute(
            f"select k.start, k.end, coalesce(sd.value, sm.value) from {kt} k "
            f"left join {strings} sd on k.demangledName = sd.id "
            f"left join {strings} sm on k.shortName = sm.id "
            f"where k.start >= ? and k.start <= ?", (s, e)).fetchall()
        res = {"kernels_total": len(rows), "range_ms": (e - s) / 1e6}
        for tag, pat in (("prep", "_gemma4_qkv_norm_rope_quant_kernel"),
                         ("append_d256", "AppendPagedKVCacheKernel<(unsigned int)256"),
                         ("append_d512", "AppendPagedKVCacheKernel<(unsigned int)512"),
                         ("append_any", "AppendPagedKVCacheKernel"),
                         ("fp8_cast", "Float8"),
                         ("fwd_steps_qkv_gemm_proxy", "BatchPrefill")):
            sel = [(a, b) for a, b, n in rows if n and pat in n]
            res[tag] = {"calls": len(sel), "total_us": round(sum(b - a for a, b in sel) / 1e3, 1)}
        out[name] = res
        print(name, json.dumps(res), flush=True)
    print(json.dumps({"ranges": out, "tables_ok": kt in tables and nv in tables}))


if __name__ == "__main__":
    main(sys.argv[1])
