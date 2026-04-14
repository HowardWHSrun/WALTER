#!/bin/bash
# Check progress of the most recent or specified run
# Usage: ./scripts/check_progress.sh [run_name]

if [ -z "$1" ]; then
    # Find most recent run
    RUN_DIR=$(ls -td runs/*/ 2>/dev/null | head -1)
else
    RUN_DIR="runs/$1"
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "No runs found. Start training first with ./scripts/start_training.sh"
    exit 1
fi

echo "Run: $RUN_DIR"
echo "=================================================="
echo ""

# Show run info
if [ -f "$RUN_DIR/run_info.yaml" ]; then
    echo "=== Run Info ==="
    cat "$RUN_DIR/run_info.yaml"
    echo ""
fi

# Show progress (checkpoints with episode counts)
if [ -f "$RUN_DIR/checkpoints/progress.yaml" ]; then
    echo "=== Progress (timesteps → episodes) ==="
    cat "$RUN_DIR/checkpoints/progress.yaml"
    echo ""
fi

# Show training summary if complete
if [ -f "$RUN_DIR/training_summary.yaml" ]; then
    echo "=== Training Complete ==="
    cat "$RUN_DIR/training_summary.yaml"
    echo ""
else
    echo "=== Training still running or incomplete ==="
fi

# List checkpoints
if [ -d "$RUN_DIR/checkpoints" ]; then
    echo "=== Checkpoints ==="
    ls -lh "$RUN_DIR/checkpoints/"*.zip 2>/dev/null | awk '{print $9, "("$5")"}'
    if [ -d "$RUN_DIR/checkpoints/best" ]; then
        echo "Best model: checkpoints/best/best_model.zip"
    fi
fi
