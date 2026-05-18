#!/usr/bin/env python3
"""
Phase 7: Export GGUF q4_k_m from merged bf16 model for Ollama.
Uses llama.cpp's convert_hf_to_gguf.py directly instead of Unsloth's wrapper.
"""
import os, sys, subprocess

MERGE_DIR  = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1/merged_bf16'
GGUF_DIR   = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1/gguf'
GGUF_FILE  = os.path.join(GGUF_DIR, 'qwen3_8b_hal_v1_q4_k_m.gguf')
LLAMA_CPP  = '/home/charm/.unsloth/llama.cpp'

os.makedirs(GGUF_DIR, exist_ok=True)

# Step 1: Convert HF model to GGUF bf16 (unquantized)
print('Step 1: Converting HF model to GGUF bf16...')
bf16_gguf = os.path.join(GGUF_DIR, 'qwen3_8b_hal_v1_bf16.gguf')

# Use the convert script from llama.cpp
convert_script = os.path.join(LLAMA_CPP, 'convert_hf_to_gguf.py')
env = os.environ.copy()
env['PYTHONPATH'] = LLAMA_CPP + ':' + env.get('PYTHONPATH', '')

result = subprocess.run(
    [sys.executable, convert_script, MERGE_DIR, '--outfile', bf16_gguf, '--outtype', 'bf16'],
    env=env,
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr)
    print('Conversion failed, trying alternative approach...')
    
    # Alternative: try with --outtype f16
    result = subprocess.run(
        [sys.executable, convert_script, MERGE_DIR, '--outfile', bf16_gguf, '--outtype', 'f16'],
        env=env,
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print('STDERR:', result.stderr)
        sys.exit(1)

print(f'bf16 GGUF saved to {bf16_gguf}')

# Step 2: Quantize to q4_k_m using llama-quantize
print('\nStep 2: Quantizing to q4_k_m...')
quantize_bin = os.path.join(LLAMA_CPP, 'llama-quantize')

result = subprocess.run(
    [quantize_bin, bf16_gguf, GGUF_FILE, 'q4_k_m'],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr)
    sys.exit(1)

print(f'\nGGUF q4_k_m saved to {GGUF_FILE}')

# Cleanup bf16 intermediate
os.remove(bf16_gguf)
print('Cleaned up intermediate bf16 GGUF file.')
print('Done!')
