# /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1/train.py
import os, json, torch
os.environ['HF_HOME'] = '/mnt/zardos/charm-hal-env/hf_cache'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['UNSLOTH_RETURN_LOGITS'] = '0'

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME       = 'unsloth/Qwen3-8B'
MAX_SEQ_LENGTH   = 2048
DATA_PATH        = '/mnt/zardos/charm-hal-env/training_data/hal_training_all.jsonl'
OUTPUT_DIR       = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v1'
SEED             = 20260517

# --- Load 4-bit base + tokenizer ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = True,
    dtype           = None,         # let Unsloth pick bf16 on Blackwell
    full_finetuning = False,
)

# --- Attach LoRA ---
model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = ['q_proj','k_proj','v_proj','o_proj',
                                  'gate_proj','up_proj','down_proj'],
    lora_alpha                 = 16,
    lora_dropout               = 0,
    bias                       = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state               = SEED,
    use_rslora                 = False,
    loftq_config               = None,
)

# --- Load dataset, format with Qwen3 template (thinking disabled) ---
raw = load_dataset('json', data_files=DATA_PATH, split='train')

def format_example(example):
    # Fix: Qwen3 template crashes on None content in assistant messages with tool_calls
    messages = []
    for m in example['messages']:
        msg = dict(m)
        if msg.get('content') is None:
            msg['content'] = ''
        messages.append(msg)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize        = False,
        add_generation_prompt = False,
        enable_thinking = False,
    )
    return {'text': text}

ds = raw.map(format_example, remove_columns=raw.column_names)
split = ds.train_test_split(test_size=0.05, seed=SEED)
train_ds, eval_ds = split['train'], split['test']
print(f'Train: {len(train_ds)}  Eval: {len(eval_ds)}')
print('--- Sample formatted text (first 800 chars) ---')
print(train_ds[0]['text'][:800])
print('--- end sample ---')

# --- Trainer ---
trainer = SFTTrainer(
    model               = model,
    tokenizer           = tokenizer,
    train_dataset       = train_ds,
    args = SFTConfig(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        num_train_epochs            = 2,
        learning_rate               = 2e-4,
        warmup_ratio                = 0.03,
        lr_scheduler_type           = 'linear',
        optim                       = 'adamw_8bit',
        weight_decay                = 0.01,
        bf16                        = True,
        fp16                        = False,
        logging_steps               = 5,
        save_strategy               = 'no',
        eval_strategy               = 'no',
        report_to                   = 'none',
        seed                        = SEED,
        max_seq_length              = MAX_SEQ_LENGTH,
        dataset_text_field          = 'text',
        packing                     = False,
    ),
)

# Mask loss so we only train on assistant tokens
trainer = train_on_responses_only(
    trainer,
    instruction_part = '<|im_start|>user\n',
    response_part    = '<|im_start|>assistant\n',
)

# --- Train ---
stats = trainer.train()
print('Final training stats:', stats)

# --- Save LoRA adapter ---
lora_dir = os.path.join(OUTPUT_DIR, 'lora_adapter')
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print('LoRA saved to', lora_dir)
