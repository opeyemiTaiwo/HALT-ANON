#!/usr/bin/env bash
# Run the whole pipeline. Set HALLUBENCH_RAW first (see README).
set -euo pipefail
cd "$(dirname "$0")/scripts"
for s in 00_build_benchmark.py 01_baselines.py 02_logo_grounded.py \
         03_artifact_test.py 04_gt_audit.py 05_judge_confound.py; do
  echo; echo ">>> $s"
  python3 "$s"
done
echo; echo "All done. Outputs in ${HALLUBENCH_OUT:-./out}"
