#!/bin/bash
# Start PPO training for hexapod sCPG with tripod gait
# Usage: ./scripts/start_training.sh [config_name] [run_name]
# Example: ./scripts/start_training.sh config_v2_train_short my_first_run

CONFIG=${1:-config_v2_train_short.yaml}
RUN_NAME=${2:-run_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR="runs/${RUN_NAME}"

echo "Starting training..."
echo "  Config: configs/${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python train.py \
    --config "configs/${CONFIG}" \
    --version 2 \
    --output-dir "${OUTPUT_DIR}"
