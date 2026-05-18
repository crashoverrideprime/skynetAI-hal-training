#!/bin/bash
LOG="/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1/train.log"
while true; do
  sleep 300
  if grep -q "Final training stats" "$LOG" 2>/dev/null; then
    echo "=== TRAINING COMPLETE! ==="
    tail -5 "$LOG"
    echo "=== GPU ==="
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
    break
  fi
  # Check if process is still running
  if ! pgrep -f "train.py" > /dev/null 2>&1; then
    echo "=== TRAINING PROCESS ENDED ==="
    tail -20 "$LOG"
    break
  fi
  echo "$(date): $(tail -1 "$LOG" | tr -d '\r')"
done
