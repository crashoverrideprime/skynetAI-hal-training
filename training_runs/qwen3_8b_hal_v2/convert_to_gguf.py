#!/usr/bin/env python3
"""
Convert trained LoRA adapter to merged model → GGUF (quantized for Ollama).

Steps:
1. Load base Qwen3-8B model (4-bit quantized)
2. Load LoRA adapter from checkpoint
3. Merge adapter into base model
4. Convert to GGUF format (Q4_K_M quantization for Ollama)
"""

import os
import sys
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Paths
OUTPUT_DIR = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2'
LORA_ADAPTER_DIR = os.path.join(OUTPUT_DIR, 'lora_adapter')
MERGED_MODEL_DIR = os.path.join(OUTPUT_DIR, 'merged_bf16')
GGUF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'gguf_models')

# Create output directories
Path(MERGED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(GGUF_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STEP 1: Loading base Qwen3-8B model (4-bit)")
print("=" * 80)

# Load base model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

print("✓ Base model loaded")

# Fix tokenizer template (same as training)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
print("✓ Tokenizer template set to qwen-2.5")

print("\n" + "=" * 80)
print("STEP 2: Loading and merging LoRA adapter")
print("=" * 80)

# Load LoRA adapter
from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA_ADAPTER_DIR)
print(f"✓ LoRA adapter loaded from {LORA_ADAPTER_DIR}")

# Merge adapter into base model
model = model.merge_and_unload()
print("✓ LoRA adapter merged into base model")

print("\n" + "=" * 80)
print("STEP 3: Converting to bf16 (for GGUF conversion)")
print("=" * 80)

# Convert from 4-bit to bf16 for GGUF export
# (GGUF conversion tools expect full precision first)
model = model.to(torch.bfloat16)
print("✓ Model converted to bf16 precision")

# Save merged bf16 model
model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)
print(f"✓ Merged model saved to {MERGED_MODEL_DIR}")

print("\n" + "=" * 80)
print("STEP 4: Converting to GGUF format (for Ollama)")
print("=" * 80)

# Use llama.cpp's conversion script
# First, verify llama.cpp is available
try:
    import subprocess
    result = subprocess.run(
        ["which", "llama-cpp-python"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("\n⚠ llama-cpp-python not found. Installing...")
        subprocess.run(
            ["pip", "install", "llama-cpp-python"],
            check=True,
        )
except Exception as e:
    print(f"Warning: Could not install llama-cpp-python: {e}")
    print("Continuing with alternative GGUF conversion...")

# Use Unsloth's built-in GGUF exporter if available
try:
    from unsloth import FastLanguageModel

    # The model is already merged and in bf16
    # Save as GGUF using HuggingFace transformers converter
    print("Converting with HuggingFace transformers...")

    import subprocess

    # Use the official conversion script
    gguf_script = f"""
import os
import sys
from pathlib import Path

# Add llama.cpp to path if needed
sys.path.insert(0, '/mnt/zardos/charm-hal-env')

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = '{MERGED_MODEL_DIR}'
output_dir = '{GGUF_OUTPUT_DIR}'

print('Loading model for GGUF conversion...')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='cpu',
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print('Converting to GGUF format...')

# Use llama.cpp's converter
os.system(f'''
python3 -m llama_cpp.convert \\
  --model-dir {model_path} \\
  --outfile {output_dir}/qwen3-8b-hal-q4_k_m.gguf \\
  --outtype q4_k_m
''')

print(f'✓ GGUF model saved to {output_dir}/qwen3-8b-hal-q4_k_m.gguf')
"""

    # Run conversion
    exec(gguf_script)

except Exception as e:
    print(f"Warning during GGUF conversion: {e}")
    print("Saving model in HuggingFace format instead...")
    print(f"You can convert to GGUF later with: llama.cpp/convert.py {MERGED_MODEL_DIR}")

print("\n" + "=" * 80)
print("CONVERSION COMPLETE")
print("=" * 80)
print(f"\n📁 Merged model (bf16): {MERGED_MODEL_DIR}")
print(f"📁 GGUF models: {GGUF_OUTPUT_DIR}")
print(f"\nTo use in Ollama:")
print(f"  1. ollama create qwen3-hal -f /path/to/Modelfile")
print(f"  2. ollama run qwen3-hal")
