#!/bin/bash
# Launch train_v2.py with full stdout+stderr captured to log.
set -euo pipefail

VENV="/mnt/zardos/charm-hal-env/bin/python"
SCRIPT="/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/train_v2.py"
LOG="/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/train_v2.log"

echo "=== Starting train_v2 at $(date) ===" | tee -a "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 "$VENV" -u "$SCRIPT" 2>&1 | tee -a "$LOG"
echo "=== Finished at $(date) ===" | tee -a "$LOG"
