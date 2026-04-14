#!/usr/bin/env bash
# Train ODE-tripod + PD policy matched to Teensy deployment (IMU obs, sim time, exportable actor).
# See docs/TEENSY_DEPLOY.md and scripts/export_ode_tripod_teensy.py
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/teensy_walk_$(date +%Y%m%d_%H%M%S)}"
NE="${TEENSY_TRAIN_NUM_ENVS:-2}"
python3 train.py \
  --config configs/config_new_body_flat_ode_imu.yaml \
  --output-dir "$OUT" \
  --num-envs "$NE"
echo ""
echo "Artifacts: $OUT"
echo "Export actor headers for Teensy (use best checkpoint if present):"
echo "  python3 scripts/export_ode_tripod_teensy.py --checkpoint $OUT/checkpoints/best/best_model.zip --output-dir teensy_export"
echo "  (fallback: $OUT/checkpoints/final_model.zip)"
