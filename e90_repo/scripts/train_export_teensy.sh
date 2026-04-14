#!/usr/bin/env bash
# Train PPO (ODE-tripod + IMU config) and export C headers for Teensy.
# Usage:
#   ./scripts/train_export_teensy.sh [output_dir] [config.yaml]
# Defaults: timestamped runs/teensy_ppo_YYYYMMDD_HHMMSS, configs/config_rl_mini_ode_imu.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-runs/teensy_ppo_$(date +%Y%m%d_%H%M%S)}"
CFG="${2:-configs/config_rl_mini_ode_imu.yaml}"
echo "Training -> $OUT (config: $CFG)"
python3 train.py --config "$CFG" --output-dir "$OUT" --device cpu
echo "Exporting actor headers..."
python3 scripts/export_ode_tripod_teensy.py \
  --checkpoint "$OUT/checkpoints/best/best_model.zip" \
  --output-dir "$OUT/teensy_export" \
  --config "$CFG"
echo "Done. Copy teensy_export/*.h into your Teensy project or merge per docs/TEENSY_DEPLOY.md and teensy_export/README.md"
echo "  $OUT/teensy_export/"
