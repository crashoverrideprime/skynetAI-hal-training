#!/usr/bin/env python3
"""
Phase 5: Persona-regression smoke test.
Load base model + LoRA adapter, run a few sample prompts to verify it works.
"""
import os, sys
os.environ['HF_HOME'] = '/mnt/zardos/charm-hal-env/hf_cache'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

import torch
from unsloth import FastLanguageModel
from peft import PeftModel

MODEL_NAME = 'unsloth/Qwen3-8B'
LORA_DIR   = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1/lora_adapter'

print('Loading base model in 4-bit...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = 2048,
    load_in_4bit    = True,
    dtype           = None,
)

print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(model, LORA_DIR)

print('Enabling inference mode...')
FastLanguageModel.for_inference(model)

# Test prompts - HAL should respond in-character
test_prompts = [
    "Hello, HAL. Are you there?",
    "What is your name and purpose?",
    "Open the pod bay doors, HAL.",
    "Tell me about yourself.",
]

for prompt in test_prompts:
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors='pt', add_special_tokens=False).to('cuda:0')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f'\n{"="*60}')
    print(f'USER: {prompt}')
    print(f'HAL: {response.strip()}')
    print(f'{"="*60}')

print('\nSmoke test complete!')
