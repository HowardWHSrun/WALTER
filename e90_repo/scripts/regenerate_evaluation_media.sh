#!/usr/bin/env bash
# Regenerate evaluation_summary.yaml + videos for every released checkpoint.
# Run from repository root after: pip install -r requirements.txt

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_eval() {
  local name="$1" config="$2" episodes="$3"
  echo "=== $name ($episodes ep, max 300 steps) ==="
  python3 evaluate.py "released_checkpoints/${name}.zip" \
    --config "$config" \
    --no-plot \
    --save-video \
    -n "$episodes" \
    --max-steps 300 \
    -o "released_checkpoints/evaluations/${name}"
}

run_eval scpg_anti_lazy_1M_best configs/config_v7_anti_lazy.yaml 2
run_eval scpg_anti_lazy_1M_final configs/config_v7_anti_lazy.yaml 2
run_eval mlp_clean_run005_best configs/config_v7_mlp_clean.yaml 2
run_eval imu6_run001_best configs/config_imu6.yaml 2
run_eval terrain_low_shake_run001_best configs/config_terrain_low_shake.yaml 3
run_eval pd_velocity_window_run018_best configs/config_v6_train_short_pd_ppo_velocity_window.yaml 2

echo "Done. Updated evaluation_summary.yaml + videos under released_checkpoints/evaluations/*/"
echo "REPORT.md files are maintained manually; re-run numbers in them if you change --max-steps or -n."
