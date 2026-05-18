#!/usr/bin/env python3
"""
Phase 7: Export GGUF q4_k_m from merged bf16 model for Ollama.
Usage: python export_gguf.py
"""
import os, sys
os.environ['HF_HOME'] = '/mnt/zardos/charm-hal-env/hf_cache'

MERGE_DIR  = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1/merged_bf16'
GGUF_DIR   = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1/gguf'
GGUF_FILE  = os.path.join(GGUF_DIR, 'qwen3_8b_hal_v1_q4_k_m.gguf')

os.makedirs(GGUF_DIR, exist_ok=True)

print('Converting merged model to GGUF q4_k_m using Unsloth...')

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MERGE_DIR,
    max_seq_length = 4096,
    load_in_4bit   = False,
    dtype          = torch.bfloat16,
)

model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method = 'q4_k_m',
)

print(f'GGUF saved to {GGUF_FILE}')
