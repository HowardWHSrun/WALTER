#!/bin/bash
# Resume training from a checkpoint
# Usage: ./scripts/resume_training.sh run_name [config] [new_run_name]
# Example: ./scripts/resume_training.sh run_20260203_170424 config_v2_continue run_continued

if [ -z "$1" ]; then
    echo "Usage: ./scripts/resume_training.sh <run_name> [config] [new_run_name]"
    echo ""
    echo "Available runs:"
    ls -1d runs/*/ 2>/dev/null | sed 's|runs/||' | sed 's|/||'
    exit 1
fi

RUN_NAME=$1
CONFIG=${2:-config_v2_continue.yaml}
NEW_RUN_NAME=${3:-${RUN_NAME}_continued}

CHECKPOINT="runs/${RUN_NAME}/checkpoints/best/best_model.zip"
OUTPUT_DIR="runs/${NEW_RUN_NAME}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Try one of these:"
    ls -1 "runs/${RUN_NAME}/checkpoints/"*.zip 2>/dev/null
    exit 1
fi

echo "Resuming training..."
echo "  From: ${CHECKPOINT}"
echo "  Config: configs/${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python train.py \
    --config "configs/${CONFIG}" \
    --version 2 \
    --resume "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}"
