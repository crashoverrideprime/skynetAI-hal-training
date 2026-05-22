#!/usr/bin/env python3
"""Merge LoRA adapter and save in formats suitable for Ollama/GGUF."""

import os
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2')
LORA_ADAPTER_DIR = OUTPUT_DIR / 'lora_adapter'
MERGED_MODEL_DIR = OUTPUT_DIR / 'merged_model'

MERGED_MODEL_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Merging LoRA adapter into base model...")
print("=" * 80)

# Load base model in bf16
print("\n1. Loading base Qwen3-8B model...")
model = AutoModelForCausalLM.from_pretrained(
    'unsloth/Qwen3-8B',
    torch_dtype=torch.bfloat16,
    device_map='cpu',
)
tokenizer = AutoTokenizer.from_pretrained('unsloth/Qwen3-8B')
print("✓ Base model loaded")

# Load and merge LoRA
print("\n2. Loading and merging LoRA adapter...")
model = PeftModel.from_pretrained(model, str(LORA_ADAPTER_DIR))
model = model.merge_and_unload()
print("✓ LoRA merged")

# Save merged model
print("\n3. Saving merged model...")
model.save_pretrained(str(MERGED_MODEL_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(MERGED_MODEL_DIR))
print(f"✓ Merged model saved to {MERGED_MODEL_DIR}")

# Create Modelfile for Ollama
print("\n4. Creating Ollama Modelfile...")
modelfile_content = f"""FROM {MERGED_MODEL_DIR}/model.safetensors

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

modelfile_path = OUTPUT_DIR / 'Modelfile'
modelfile_path.write_text(modelfile_content)
print(f"✓ Modelfile created at {modelfile_path}")

print("\n" + "=" * 80)
print("✓ CONVERSION COMPLETE")
print("=" * 80)
print(f"\nMerged model location: {MERGED_MODEL_DIR}")
print(f"\nNext steps for Ollama:")
print(f"  1. Install llama.cpp/ollama if not already installed")
print(f"  2. Create Ollama model: ollama create qwen3-hal -f {modelfile_path}")
print(f"  3. Run: ollama run qwen3-hal")
