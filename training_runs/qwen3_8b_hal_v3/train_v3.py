# /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v3/train_v3.py
"""
v3 training: LoRA on top of v2 merged base (or fresh from Qwen3-8B if merge unavailable).

v3 changes vs v2:
  - Dataset: hal_training_v3.jsonl (~2600 examples, includes failure corrections + NL mapping + disambiguation + persona)
  - GetLiveContext updated to use domain parameter (not name-only)
  - All other hyperparams identical to v2 to maintain comparability
"""

import os, json, sys, logging, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/mnt/zardos/charm-hal-env/hf_cache'
os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
os.environ['UNSLOTH_RETURN_LOGITS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

OUTPUT_DIR   = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v3'
V2_MERGED    = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/merged_model'
BASE_MODEL   = 'unsloth/Qwen3-8B'

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s %(levelname)s %(message)s',
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'train_v3.log')),
    ],
)

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

MAX_SEQ_LENGTH = 2048
DATA_PATH      = '/mnt/zardos/charm-hal-env/training_data/v3/hal_training_v3.jsonl'
SEED           = 20260520

# v3 GetLiveContext schema: domain + optional name + optional area (matches real HA AssistAPI)
HA_TOOLS = [
    {"type": "function", "function": {
        "name": "HassTurnOn",
        "description": "Turn on a device or entity (lights, switches, fans, etc.)",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "HassTurnOff",
        "description": "Turn off a device or entity (lights, switches, fans, etc.)",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "HassLightSet",
        "description": "Set brightness and/or color of a light",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "brightness": {"type": "integer"},
            "color": {"type": "string"},
        }, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "HassClimateSetTemperature",
        "description": "Set target temperature for a climate device",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "temperature": {"type": "number"},
        }, "required": ["name", "temperature"]},
    }},
    {"type": "function", "function": {
        "name": "HassListAddItem",
        "description": "Add an item to a shopping list or todo list",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "item": {"type": "string"},
        }, "required": ["name", "item"]},
    }},
    {"type": "function", "function": {
        "name": "HassMediaUnpause",
        "description": "Resume/unpause media playback",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "HassMediaPause",
        "description": "Pause media playback",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "HassMediaNext",
        "description": "Skip to next track",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "HassMediaPrevious",
        "description": "Go to previous track",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "HassSetVolume",
        "description": "Set volume level of a media player",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "volume_level": {"type": "number"},
        }, "required": ["name", "volume_level"]},
    }},
    {"type": "function", "function": {
        "name": "GetLiveContext",
        "description": (
            "Get the current state of entities in the home. "
            "Prefer calling with only the `domain` argument to get all entities in a domain, "
            "then select the relevant entity from the returned list. "
            "Only add `name` (friendly name) when the user has specified a particular entity. "
            "Do not invent entity names."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "HA entity domain e.g. sensor, climate, light, weather"},
                "name":   {"type": "string", "description": "Optional: entity friendly name filter"},
                "area":   {"type": "string", "description": "Optional: area/room name filter"},
            },
        },
    }},
]

# ── Prefer v2 merged model as base (LoRA-on-merged approach) ───────────────
import os.path
if os.path.isdir(V2_MERGED):
    MODEL_NAME = V2_MERGED
    print(f"Using v2 merged model as base: {V2_MERGED}", flush=True)
else:
    MODEL_NAME = BASE_MODEL
    print(f"v2 merged model not found — starting from base Qwen3-8B: {BASE_MODEL}", flush=True)
    print("Note: run merge_and_convert.py in v2 dir first for best results.", flush=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = True,
    dtype           = None,
    full_finetuning = False,
)

# Use qwen-2.5 template (same as v2 — avoids Qwen3 template crash on plain dict messages)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

model = FastLanguageModel.get_peft_model(
    model,
    r                          = 32,
    target_modules             = ['q_proj','k_proj','v_proj','o_proj',
                                  'gate_proj','up_proj','down_proj'],
    lora_alpha                 = 32,
    lora_dropout               = 0,
    bias                       = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state               = SEED,
    use_rslora                 = False,
    loftq_config               = None,
)

# ── Load + format dataset ──────────────────────────────────────────────────
formatted_texts = []
with open(DATA_PATH) as f:
    for line_num, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            example = json.loads(line)
            messages = []
            for m in example.get('messages', []):
                msg = dict(m)
                if msg.get('content') is None:
                    msg['content'] = ''
                if msg.get('tool_calls'):
                    for tc in msg['tool_calls']:
                        if 'function' in tc:
                            fn = tc['function']
                            if isinstance(fn.get('arguments'), str):
                                try:
                                    fn['arguments'] = json.loads(fn['arguments'])
                                except json.JSONDecodeError:
                                    pass
                messages.append(msg)
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tools                 = HA_TOOLS,
                    tokenize              = False,
                    add_generation_prompt = False,
                )
                formatted_texts.append({'text': text})
            except Exception as exc:
                print(f'[format ERROR] Line {line_num}: {exc}', file=sys.stderr, flush=True)
        except json.JSONDecodeError as exc:
            print(f'[JSON ERROR] Line {line_num}: {exc}', file=sys.stderr, flush=True)

ds = Dataset.from_list(formatted_texts)
split = ds.train_test_split(test_size=0.05, seed=SEED)
train_ds, eval_ds = split['train'], split['test']
print(f'Train: {len(train_ds)}  Eval: {len(eval_ds)}', flush=True)
print('--- Sample formatted text (first 800 chars) ---')
print(train_ds[0]['text'][:800])
print('--- end sample ---')

# ── Trainer (same hyperparams as v2) ──────────────────────────────────────
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_ds,
    eval_dataset  = eval_ds,
    args = SFTConfig(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size  = 1,
        gradient_accumulation_steps = 16,
        eval_accumulation_steps     = 4,
        num_train_epochs            = 3,
        learning_rate               = 5e-5,
        warmup_ratio                = 0.03,
        lr_scheduler_type           = 'linear',
        optim                       = 'adamw_8bit',
        weight_decay                = 0.01,
        bf16                        = True,
        fp16                        = False,
        logging_steps               = 5,
        save_strategy               = 'steps',
        save_steps                  = 50,
        save_total_limit            = 4,
        eval_strategy               = 'steps',
        eval_steps                  = 50,
        load_best_model_at_end      = True,
        metric_for_best_model       = 'eval_loss',
        report_to                   = 'none',
        seed                        = SEED,
        max_seq_length              = MAX_SEQ_LENGTH,
        dataset_text_field          = 'text',
        packing                     = True,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = '<|im_start|>user\n',
    response_part    = '<|im_start|>assistant\n',
)

import glob
_ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'checkpoint-*')), key=lambda p: int(p.rsplit('-', 1)[-1]))
print(f'[DEBUG] All checkpoints found: {_ckpts}', flush=True)
_resume = _ckpts[-1] if _ckpts else None
if _resume:
    print(f'Resuming from checkpoint: {_resume}', flush=True)
stats = trainer.train(resume_from_checkpoint=_resume)
print('Final training stats:', stats)

lora_dir = os.path.join(OUTPUT_DIR, 'lora_adapter')
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print(f'LoRA saved to {lora_dir}', flush=True)
