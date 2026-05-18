#!/usr/bin/env python3
"""
Phase 6: Export merged bf16 HF model from LoRA adapter.
Usage: python export_merged.py
"""
import os, torch
os.environ['HF_HOME'] = '/mnt/zardos/charm-hal-env/hf_cache'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from unsloth import FastLanguageModel

MODEL_NAME  = 'unsloth/Qwen3-8B'
LORA_DIR    = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1/lora_adapter'
MERGE_DIR   = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1/merged_bf16'

print('Loading base model + LoRA...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = 4096,
    load_in_4bit    = False,   # load in bf16 for merge
    dtype           = torch.bfloat16,
)

print('Loading LoRA weights...')
from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA_DIR)

print('Merging and unloading...')
model = model.merge_and_unload()

print(f'Saving merged model to {MERGE_DIR}...')
model.save_pretrained(MERGE_DIR, safe_serialization=True, max_shard_size='4GB')
tokenizer.save_pretrained(MERGE_DIR)
print('Done! Merged bf16 model saved.')
