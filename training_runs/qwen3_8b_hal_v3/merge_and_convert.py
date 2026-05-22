#!/usr/bin/env python3
"""
Merge v3 LoRA adapter into base model (or v2 merged base) and save for GGUF conversion.
Then convert to GGUF Q4_K_M and register with Ollama as charm-hal-v3.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

V3_DIR       = Path('/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v3')
LORA_DIR     = V3_DIR / 'lora_adapter'
MERGED_DIR   = V3_DIR / 'merged_model'
V2_MERGED    = Path('/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/merged_model')
BASE_MODEL   = 'unsloth/Qwen3-8B'
LLAMA_CPP    = Path('/mnt/zardos/llama.cpp')
GGUF_F16     = V3_DIR / 'charm-hal-v3-f16.gguf'
GGUF_Q4      = V3_DIR / 'charm-hal-v3-Q4_K_M.gguf'
MODELFILE    = V3_DIR / 'Modelfile'

MERGED_DIR.mkdir(exist_ok=True)

# Use v2 merged model as base if available (best transfer of v2 competence)
if V2_MERGED.is_dir():
    base = str(V2_MERGED)
    print(f"Using v2 merged model as base: {base}")
else:
    base = BASE_MODEL
    print(f"Using base Qwen3-8B (v2 merged not found)")

print("\n1. Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cpu')
tokenizer = AutoTokenizer.from_pretrained(base)
print("   Base loaded.")

print("\n2. Merging v3 LoRA...")
model = PeftModel.from_pretrained(model, str(LORA_DIR))
model = model.merge_and_unload()
print("   LoRA merged.")

print(f"\n3. Saving merged model to {MERGED_DIR}...")
model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(MERGED_DIR))
print("   Saved.")
del model

print("\n4. Converting to GGUF f16...")
convert_script = LLAMA_CPP / 'convert_hf_to_gguf.py'
subprocess.run([
    sys.executable, str(convert_script),
    str(MERGED_DIR),
    '--outtype', 'f16',
    '--outfile', str(GGUF_F16),
], check=True)
print(f"   f16 GGUF: {GGUF_F16}")

print("\n5. Quantizing to Q4_K_M...")
quantize_bin = LLAMA_CPP / 'build' / 'bin' / 'llama-quantize'
subprocess.run([str(quantize_bin), str(GGUF_F16), str(GGUF_Q4), 'Q4_K_M'], check=True)
print(f"   Q4_K_M GGUF: {GGUF_Q4}")

print("\n6. Writing Modelfile...")
# Reuse the exact Qwen3 template from the v2 Modelfile
V2_MODELFILE = Path('/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/Modelfile')
v2_modelfile_text = V2_MODELFILE.read_text()
# Swap the FROM line to point to v3 GGUF
import re
modelfile_text = re.sub(
    r'^FROM .*$',
    f'FROM {GGUF_Q4}',
    v2_modelfile_text,
    flags=re.MULTILINE,
)
MODELFILE.write_text(modelfile_text)
print(f"   Modelfile: {MODELFILE}")

print("\n7. Registering charm-hal-v3 with Ollama...")
subprocess.run(['ollama', 'create', 'charm-hal-v3', '-f', str(MODELFILE)], check=True)
print("   ollama create charm-hal-v3 done.")

print("\n8. Verifying capabilities...")
result = subprocess.run(['ollama', 'show', 'charm-hal-v3'], capture_output=True, text=True)
print(result.stdout[:600])
if 'tools' not in result.stdout.lower():
    print("WARNING: 'tools' capability not visible in ollama show output — check Modelfile TEMPLATE.")

print("\nDone. charm-hal-v3 is registered alongside charm-hal (v2).")
print("To A/B test: switch HA Voice Assistant agent from charm-hal to charm-hal-v3.")
