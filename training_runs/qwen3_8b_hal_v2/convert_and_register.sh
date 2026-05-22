#!/bin/bash
set -euo pipefail

echo "=================================================="
echo "Step 1: Convert to f16 GGUF"
echo "=================================================="

/mnt/zardos/charm-hal-env/bin/python3 /mnt/zardos/llama.cpp/convert_hf_to_gguf.py \
  /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/merged_model \
  --outtype f16 \
  --outfile /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models/qwen3-hal-f16.gguf

echo "✓ f16 GGUF conversion complete"
ls -lh /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models/qwen3-hal-f16.gguf

echo ""
echo "=================================================="
echo "Step 2: Build llama-quantize"
echo "=================================================="

cd /mnt/zardos/llama.cpp
make llama-quantize -j$(nproc)

echo "✓ llama-quantize built"

echo ""
echo "=================================================="
echo "Step 3: Quantize to Q4_K_M"
echo "=================================================="

./llama-quantize \
  /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models/qwen3-hal-f16.gguf \
  /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models/qwen3-hal-q4_k_m.gguf \
  Q4_K_M

echo "✓ Q4_K_M quantization complete"
ls -lh /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models/qwen3-hal-q4_k_m.gguf

echo ""
echo "Deleting intermediate f16 GGUF to free 15GB..."
rm /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models/qwen3-hal-f16.gguf
df -h /mnt/zardos | grep zardos

echo ""
echo "=================================================="
echo "Step 4: Update Modelfile"
echo "=================================================="

cat > /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/Modelfile << 'MODELFILE'
FROM /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models/qwen3-hal-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """You are HAL, the conversational AI for Charm's smart home in Clyde North, Victoria, Australia. You help control smart home devices using available tools and respond naturally to home automation requests."""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
MODELFILE

echo "✓ Modelfile updated"
cat /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/Modelfile

echo ""
echo "=================================================="
echo "Step 5: Register with Ollama"
echo "=================================================="

ollama create qwen3-hal-v2 \
  -f /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/Modelfile

echo "✓ Model registered with Ollama"
ollama list | grep -E "qwen3-hal-v2|qwen3:8b"

echo ""
echo "=================================================="
echo "Step 6: Test"
echo "=================================================="

echo "Testing: 'Turn on the living room lights'"
ollama run qwen3-hal-v2 "Turn on the living room lights"

echo ""
echo "=================================================="
echo "✓ ALL STEPS COMPLETE"
echo "=================================================="
echo ""
echo "Model qwen3-hal-v2 is now ready to use!"
echo ""
echo "Try: ollama run qwen3-hal-v2 'What's the temperature in the bedroom?'"
