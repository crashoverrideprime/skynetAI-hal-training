# HAL Training Dataset — Run & Validate

## Quick run

```bash
cd /mnt/zardos/charm-hal-env/training_data
source ../bin/activate     # or skip if running with system python3
python3 generate_dataset.py
```

This produces:
- `hal_training_all.jsonl` — combined 1,460-line shuffled dataset
- `batch_01_energy_status.jsonl` … `batch_22_multi_domain.jsonl` — per-batch files (1,460 total across all)

Reproducible: SEED = 20260517 inside the script. Change it to generate fresh randomisation.

## Validate

```bash
cd /mnt/zardos/charm-hal-env/training_data

# Line count
wc -l hal_training_all.jsonl    # must print 1460

# Per-batch counts
wc -l batch_*.jsonl

# Schema + persona validation
python3 - <<'PY'
import json
errs = 0
persona_bad = ['great!', 'sure!', 'of course', 'certainly!', 'happy to help']
with open('hal_training_all.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            assert 'messages' in obj and isinstance(obj['messages'], list)
            roles = [m['role'] for m in obj['messages']]
            assert roles[0] == 'system'
            assert roles[-1] == 'assistant'
            final = (obj['messages'][-1].get('content') or '').lower()
            for bad in persona_bad:
                assert bad not in final, f'Line {i}: persona violation "{bad}"'
        except Exception as e:
            print(f'Line {i}: {e}')
            errs += 1
print(f'Errors: {errs}')
PY

# HVAC direction-aware verification (must be >= 40/80)
python3 - <<'PY'
import json
hits = 0
for line in open('batch_09_hvac_control.jsonl'):
    final = (json.loads(line)['messages'][-1].get('content') or '').lower()
    if any(w in final for w in ['so it will heat', 'so it will cool',
                                'will heat', 'will cool', 'in deadband']):
        hits += 1
print(f'HVAC direction-aware hits: {hits}/80')
PY

# Spot-check 5 random examples
python3 - <<'PY'
import json, random
lines = open('hal_training_all.jsonl').readlines()
for _ in range(5):
    ex = json.loads(random.choice(lines))
    print('=' * 60)
    for m in ex['messages']:
        c = m.get('content') or json.dumps(m.get('tool_calls', ''))
        print(f"[{m['role']:9s}] {c[:140]}")
PY
```

## Training with unsloth (next step)

The `messages` format is consumable directly via:

```python
from datasets import load_dataset
ds = load_dataset('json', data_files='hal_training_all.jsonl', split='train')
# unsloth's get_chat_template() / standardize_sharegpt() / SFTTrainer pipeline
```

Apply Llama 3.2's chat template via `tokenizer.apply_chat_template(messages, tools=tools)` if you want to inject the tool schema at train time.

## Notes

- All entity IDs and attribute shapes match the live HA snapshot at `ha_snapshot.json` (1,333 entities pulled 2026-05-17).
- iZone zone climates (`climate.downstairs`, `climate.upstairs`) intentionally use `heat_cool` mode only — the dataset teaches HAL to read ambient first and choose setpoint direction (above ambient = heat, below = cool).
- Persona rules enforced: no banned openers, "Charm" as address, prices in cents/kWh, Tesla-sleeping fallback line, spike acknowledgement in load queries.
- `web_search` examples in batch 21 use plausible synthetic snippets (no fabricated URLs) so the model learns the *shape* of incorporating search results without memorising made-up sources.
