#!/usr/bin/env bash
# Build the harness against main's fmha_v2 headers (git archive of BASE) and this tree's, run both.
#   diag.sh [BASE_COMMIT] [CASES] [WARMUP] [ITERS]
set -euo pipefail
BASE="${1:-87bcb21958a58c5bf22327e50aa0fdfee07db213}"
CASES="${2:-c1,c4,ragged,empty}"
WARMUP="${3:-10}"
ITERS="${4:-50}"
TREE="$(pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W=/tmp/fmha_ctx_diag
rm -rf "$W" && mkdir -p "$W/base"
git archive "$BASE" cpp/kernels/fmha_v2/src | tar -x -C "$W/base"
echo "[diag] base headers from $BASE; candidate headers from $TREE ($(git rev-parse --short HEAD))"
for f in fmha/kernel_traits.h fmha/gmem_tile_o_packed.h fused_multihead_flash_attention_kernel_noloop_tiled.h; do
  if cmp -s "$W/base/cpp/kernels/fmha_v2/src/$f" "$TREE/cpp/kernels/fmha_v2/src/$f"; then echo "  same: $f"; else echo "  differs: $f"; fi
done
bash "$HERE/build.sh" "$W/base/cpp/kernels/fmha_v2/src" "$W/bench_base" &
bash "$HERE/build.sh" "$TREE/cpp/kernels/fmha_v2/src" "$W/bench_cand" &
wait
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader || true
for arm in base cand; do
  echo "===== $arm ====="
  "$W/bench_$arm" --cases "$CASES" --warmup "$WARMUP" --iters "$ITERS" | tee "$W/$arm.log"
done
python3 - "$W" <<'EOF'
import json, sys
w = sys.argv[1]
res = {}
for arm in ("base", "cand"):
    line = [l for l in open(f"{w}/{arm}.log") if l.startswith("CASES_JSON ")][-1]
    res[arm] = json.loads(line[len("CASES_JSON "):])
out = {"kernel": {a: res[a]["kernel"] for a in res}, "cases": {}}
for c, b in res["base"]["cases"].items():
    k = res["cand"]["cases"][c]
    out["cases"][c] = {"base_ms": b["kernel_ms"], "cand_ms": k["kernel_ms"], "speedup": b["kernel_ms"] / k["kernel_ms"],
                       "base_p90": b["p90_ms"], "cand_p90": k["p90_ms"], "same_out_hash": b["out_hash"] == k["out_hash"],
                       "base_noise": b["noise_floor"], "cand_noise": k["noise_floor"]}
out["ok"] = res["base"]["ok"] and res["cand"]["ok"] and all(v["same_out_hash"] for v in out["cases"].values())
print(json.dumps(out))
EOF
