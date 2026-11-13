#!/bin/bash
# Stage job for step_cache idea 56 (analysis idea 70): Part A toy check, Part B self-test on synthetic
# logs (with real cache-dit stats payloads from Part A), and a search for branch 2's job artifacts.
# Usage: bash experiments/ascent/run_cache_dit_stage.sh {python}
PY="$1"
A="${AUTO_DEV_ARTIFACTS:-/tmp}"
D="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$A"
echo "== Part A"
"$PY" "$D/cache_dit_static_mask_check.py" 2>&1 | tee "$A/partA.log"
RA=${PIPESTATUS[0]}
echo "== Part B self-test"
"$PY" "$D/h3_cache_arms_table.py" --self-test --outdir "$A/arms_table" \
  --stats-line-file "w14_c2=$A/cache_dit_toy/stats_line_w14_c2.txt" \
  --stats-line-file "E=$A/cache_dit_toy/stats_line_E.txt" \
  --json-out "$A/partB_selftest.json" 2>&1 | tee "$A/partB.log"
RB=${PIPESTATUS[0]}
echo "== Part B: look for branch 2 jobs 260/261/302/292"
# $A = <runner>/slots/<n>/jobs/<uuid>/artifacts
RUNNER="$(cd "$A/../../../../.." 2>/dev/null && pwd)"
timeout 240 "$PY" "$D/h3_cache_arms_table.py" \
  --find-artifacts "$RUNNER,/home/scratch.chengzhang/.auto-dev/leases/*/runner" 2>&1 | head -c 40000 | tee "$A/partB_find.log"
if [ "$RA" -eq 0 ] && [ "$RB" -eq 0 ]; then P=true; else P=false; fi
echo "{\"passed\": $P, \"partA_rc\": $RA, \"partB_rc\": $RB}"
