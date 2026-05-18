# /mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2/train_v2.py
"""
Phase D — Train v2 model with HA-native tools, mixed no-tool examples, and error handling.

Hyperparameters (v2):
  - lora_r=32, lora_alpha=32, lora_dropout=0
  - learning_rate=5e-5, num_train_epochs=3
  - packing=True, group_by_length=True
  - gradient_accumulation_steps=16
  - eval_strategy=steps, eval_steps=25
  - save_strategy=steps, save_steps=50, save_total_limit=4
  - load_best_model_at_end=True
"""

import os, json, sys, logging, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/mnt/zardos/charm-hal-env/hf_cache'
os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
os.environ['UNSLOTH_RETURN_LOGITS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

OUTPUT_DIR = '/mnt/zardos/charm-hal-env/training_runs/qwen3_8b_hal_v2'
logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s %(levelname)s %(message)s',
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'train_v2_debug.log')),
    ],
)

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME       = 'unsloth/Qwen3-8B'
MAX_SEQ_LENGTH   = 2048
DATA_PATH        = '/mnt/zardos/charm-hal-env/training_data/v2/hal_training_v2.jsonl'
SEED             = 20260517

# ── HA-native tool definitions (for template rendering) ────────────────────
HA_TOOLS = [
    {"type": "function", "function": {
        "name": "HassTurnOn",
        "description": "Turn on a device or entity (lights, switches, fans, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the device or area to turn on"}
            },
            "required": ["name"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassTurnOff",
        "description": "Turn off a device or entity (lights, switches, fans, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the device or area to turn off"}
            },
            "required": ["name"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassLightSet",
        "description": "Set brightness and/or color of a light",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the light or area"},
                "brightness": {"type": "integer", "description": "Brightness level 0-100"},
                "color": {"type": "string", "description": "Color name or hex"}
            },
            "required": ["name"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassClimateSetTemperature",
        "description": "Set target temperature for a climate device (thermostat, AC, heater)",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the climate device or area"},
                "temperature": {"type": "number", "description": "Target temperature in Celsius"}
            },
            "required": ["name", "temperature"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassClimateGetTemperature",
        "description": "Get current temperature from a climate sensor",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the climate device or area"}
            },
            "required": ["name"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassListAddItem",
        "description": "Add an item to a shopping list or todo list",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the list"},
                "item": {"type": "string", "description": "Item to add"}
            },
            "required": ["name", "item"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassMediaUnpause",
        "description": "Resume/unpause media playback",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"}
            },
            "required": ["name"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassMediaPause",
        "description": "Pause media playback",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"}
            },
            "required": ["name"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassMediaNext",
        "description": "Skip to next track",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"}
            },
            "required": ["name"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassMediaPrevious",
        "description": "Go to previous track",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"}
            },
            "required": ["name"]
        }
    }},
    {"type": "function", "function": {
        "name": "HassSetVolume",
        "description": "Set volume level of a media player",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"},
                "volume_level": {"type": "number", "description": "Volume level 0.0 to 1.0"}
            },
            "required": ["name", "volume_level"]
        }
    }},
    {"type": "function", "function": {
        "name": "GetLiveContext",
        "description": "Get the current state of one or more entities in the home. Use this to query sensor values, device states, and any live data.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name or area to query, or 'all' for everything"}
            },
            "required": ["name"]
        }
    }},
]

# --- Load 4-bit base + tokenizer ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = True,
    dtype           = None,         # let Unsloth pick bf16 on Blackwell
    full_finetuning = False,
)

# Fix: Qwen3's built-in chat template uses attribute access (.content) instead of
# dict access (["content"]), which crashes on plain dict messages. Use the
# Qwen2.5 template from Unsloth which handles dicts correctly.
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# --- Attach LoRA (v2: r=32, alpha=32, dropout=0 for full Unsloth kernel fusion) ---
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

# --- Load dataset, format with Qwen3 template + tools ---
# Workaround: load_dataset('json') segfaults (PyArrow/Triton incompatibility).
# Load, format, and filter examples manually to avoid PyArrow schema issues.
formatted_texts = []

with open(DATA_PATH) as f:
    for line_num, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            example = json.loads(line)
            messages = []
            for m in example.get('messages', []):
                msg = m
                if msg.get('content') is None:
                    msg['content'] = ''
                # Arguments arrive as JSON strings from OpenAI format; parse to dicts for Transformers
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
                print(f'[format_example ERROR] Line {line_num}: {exc}', file=sys.stderr, flush=True)
        except json.JSONDecodeError as exc:
            print(f'[JSON ERROR] Line {line_num}: {exc}', file=sys.stderr, flush=True)

# Create dataset from formatted texts
ds = Dataset.from_list(formatted_texts)
split = ds.train_test_split(test_size=0.05, seed=SEED)
train_ds, eval_ds = split['train'], split['test']
print(f'Train: {len(train_ds)}  Eval: {len(eval_ds)}')
print('--- Sample formatted text (first 800 chars) ---')
print(train_ds[0]['text'][:800])
print('--- end sample ---')

# --- Trainer (v2 hyperparameters) ---
trainer = SFTTrainer(
    model               = model,
    tokenizer           = tokenizer,
    train_dataset       = train_ds,
    eval_dataset        = eval_ds,
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
