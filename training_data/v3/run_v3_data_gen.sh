#!/bin/bash
# Generate all v3 training data, then combine.
# Run from: /mnt/zardos/charm-hal-env/training_data/v3/
# Prerequisites: ha_states.json must exist; Ollama qwen2.5:14b must be running.
set -euo pipefail

VENV="/mnt/zardos/charm-hal-env/bin/python"
DIR="/mnt/zardos/charm-hal-env/training_data/v3"
LOG="$DIR/data_gen.log"

echo "=== v3 data generation started $(date) ===" | tee -a "$LOG"

echo "[1/4] Corrections (96 failure fixes)..." | tee -a "$LOG"
"$VENV" -u "$DIR/generate_v3_corrections.py" 2>&1 | tee -a "$LOG"

echo "[2/4] NL→sensor mapping (~150 examples)..." | tee -a "$LOG"
"$VENV" -u "$DIR/generate_v3_nl_mapping.py" 2>&1 | tee -a "$LOG"

echo "[3/4] Disambiguation (~25 examples, no LLM calls)..." | tee -a "$LOG"
"$VENV" -u "$DIR/generate_v3_disambiguation.py" 2>&1 | tee -a "$LOG"

echo "[4/4] HAL persona examples (~30 examples + expansions)..." | tee -a "$LOG"
"$VENV" -u "$DIR/generate_v3_persona.py" 2>&1 | tee -a "$LOG"

echo "[5/5] Combining into hal_training_v3.jsonl..." | tee -a "$LOG"
"$VENV" -u "$DIR/combine_v3_dataset.py" 2>&1 | tee -a "$LOG"

echo "=== Done $(date) ===" | tee -a "$LOG"
echo "Output: $DIR/hal_training_v3.jsonl"
