#!/bin/sh
# Runs every step and the checks. The first run trains the model (about ten minutes on a laptop CPU).
PYTHON=${PYTHON:-python3}
for f in step1_slow step2_same step3_cache step4_race step5_read step6_price step7_share exercise tests; do
  echo
  echo "=== $f.py ==="
  $PYTHON -W ignore $f.py || exit 1
done
