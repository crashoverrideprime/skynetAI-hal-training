#!/bin/bash
set -euo pipefail

MERGED_MODEL="/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/merged_model"
OUTPUT_DIR="/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models"
VENV="/mnt/zardos/charm-hal-env/bin/python3"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Converting to GGUF format"
echo "=========================================="

# Install llama-cpp-python if needed
echo "Installing llama-cpp-python..."
$VENV -m pip install -q llama-cpp-python 2>/dev/null || echo "llama-cpp-python installation skipped (may already be installed)"

# Create conversion script
cat > /tmp/convert_gguf.py << 'PYTHON_EOF'
import os
import subprocess
import sys
from pathlib import Path

merged_model = "/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/merged_model"
output_dir = "/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models"

# Try using llama.cpp's converter
print("Attempting GGUF conversion using llama-cpp-python...")

try:
    from llama_cpp import Llama
    # Load and re-save as GGUF
    # Note: This requires llama.cpp built with the right architecture
    print("llama-cpp-python available, but conversion requires pre-built binaries.")
    print("\nAlternative: Use official llama.cpp converter:")
    print(f"  git clone https://github.com/ggerganov/llama.cpp")
    print(f"  cd llama.cpp && python3 convert.py {merged_model} --outfile {output_dir}/qwen3-hal.gguf --outtype q4_k_m")
except ImportError:
    print("llama-cpp-python not available.")

# Alternative: Use HuggingFace safetensors + manual GGUF conversion
print("\nUsing transformers + safetensors approach...")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading merged model from {merged_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        merged_model,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model)

    # Save in a format that can be converted to GGUF
    intermediate_dir = f"{output_dir}/qwen3-hal-bf16"
    os.makedirs(intermediate_dir, exist_ok=True)

    print(f"Saving intermediate format to {intermediate_dir}...")
    model.save_pretrained(intermediate_dir, safe_serialization=True)
    tokenizer.save_pretrained(intermediate_dir)

    print("\n✓ Model saved in HuggingFace format (ready for llama.cpp conversion)")
    print(f"\nNext step: Use llama.cpp to convert to GGUF:")
    print(f"  git clone https://github.com/ggerganov/llama.cpp")
    print(f"  cd llama.cpp && python3 convert.py {intermediate_dir} --outfile {output_dir}/qwen3-hal.gguf --outtype q4_k_m")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✓ GGUF conversion preparation complete")
print("="*80)
PYTHON_EOF

$VENV /tmp/convert_gguf.py

echo ""
echo "=========================================="
echo "Conversion Summary"
echo "=========================================="
echo ""
echo "Merged model: /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/merged_model"
echo "Output dir:   /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models"
echo ""
echo "To complete GGUF conversion:"
echo "  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
echo "  2. Run converter:"
echo "     python3 llama.cpp/convert.py \\"
echo "       /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/merged_model \\"
echo "       --outfile /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/gguf_models/qwen3-hal.gguf \\"
echo "       --outtype q4_k_m"
echo ""
echo "To use in Ollama (after GGUF conversion):"
echo "  ollama create qwen3-hal -f /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/Modelfile"
echo "  ollama run qwen3-hal"
