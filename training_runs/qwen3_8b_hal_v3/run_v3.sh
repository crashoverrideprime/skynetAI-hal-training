#!/bin/bash
# Launch train_v3.py — clears GPU 0 first, then trains.
set -euo pipefail

VENV="/mnt/zardos/charm-hal-env/bin/python"
SCRIPT="/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v3/train_v3.py"
LOG="/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v3/train_v3.log"
GPU_ID=0
VRAM_LIMIT_MB=512   # abort if more than this remains after cleanup

# ── Step 1: stop Ollama ──────────────────────────────────────────────────────
echo "[preflight] Stopping Ollama..." | tee -a "$LOG"
sudo systemctl stop ollama 2>/dev/null \
  || { pkill -SIGTERM -f "/usr/local/bin/ollama" 2>/dev/null; sleep 2; \
       pkill -9 -f "/usr/local/bin/ollama" 2>/dev/null || true; }
sleep 2

# ── Step 2: kill any remaining processes on GPU 0 ───────────────────────────
echo "[preflight] Checking GPU ${GPU_ID} for active compute processes..." | tee -a "$LOG"
GPU0_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
            --id=${GPU_ID} 2>/dev/null | tr -d ' \r' | grep -v '^$' || true)

if [ -n "$GPU0_PIDS" ]; then
    echo "[preflight] Killing PIDs on GPU ${GPU_ID}: $GPU0_PIDS" | tee -a "$LOG"
    echo "$GPU0_PIDS" | xargs -r kill -SIGTERM 2>/dev/null || true
    sleep 5
    # Force-kill anything still alive
    GPU0_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
                --id=${GPU_ID} 2>/dev/null | tr -d ' \r' | grep -v '^$' || true)
    [ -n "$GPU0_PIDS" ] && echo "$GPU0_PIDS" | xargs -r kill -9 2>/dev/null || true
    sleep 3
else
    echo "[preflight] GPU ${GPU_ID} is already clear." | tee -a "$LOG"
fi

# ── Step 3: verify VRAM is sufficiently free ─────────────────────────────────
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            --id=${GPU_ID} | tr -d ' ')
echo "[preflight] GPU ${GPU_ID} VRAM used: ${VRAM_USED} MiB" | tee -a "$LOG"
nvidia-smi --id=${GPU_ID} --query-gpu=name,memory.used,memory.free,memory.total \
           --format=csv | tee -a "$LOG"

if [ "${VRAM_USED}" -gt "${VRAM_LIMIT_MB}" ]; then
    echo "[preflight] ERROR: ${VRAM_USED} MiB still in use on GPU ${GPU_ID} — aborting." | tee -a "$LOG"
    exit 1
fi

echo "[preflight] GPU ${GPU_ID} ready. Starting training..." | tee -a "$LOG"

# ── Step 4: train ─────────────────────────────────────────────────────────────
echo "=== Starting train_v3 at $(date) ===" | tee -a "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=${GPU_ID} "$VENV" -u "$SCRIPT" 2>&1 | tee -a "$LOG"
echo "=== Finished at $(date) ===" | tee -a "$LOG"
