# Changelog — HAL Training (charm-hal-env)

All changes follow [Conventional Commits](https://www.conventionalcommits.org/).

---

## [v2.1.0] — 2026-05-18

### fix: parse tool_calls arguments from JSON string to dict before template rendering

**File:** `training_runs/qwen3_8b_hal_v2/train_v2.py`

**Root cause:** `apply_chat_template` in Transformers 5.5.0 requires
`tool_calls[].function.arguments` to be a Python **dict**. The v2 JSONL dataset
stores arguments as JSON **strings** (OpenAI API wire format, e.g.
`"{\"name\": \"living_room\", \"temperature\": 21}"`). This caused a `TypeError`
inside `datasets.map()` worker processes, which went to stderr and was invisible
because the script was launched without stderr capture.

**Fix:** `format_example` now calls `json.loads()` on any string-valued
`arguments` field before passing messages to `apply_chat_template`.

---

### fix: add try/except in format_example with flushed stderr output

**File:** `training_runs/qwen3_8b_hal_v2/train_v2.py`

**Root cause:** Exceptions inside `datasets.map()` workers are swallowed when
stderr is not captured, making silent crashes impossible to diagnose.

**Fix:** Wrapped `apply_chat_template` call in try/except; on failure, the full
exception and offending message are printed to stderr with `flush=True` so they
appear even in buffered log captures.

---

### perf: set lora_dropout=0 to restore full Unsloth kernel fusion

**File:** `training_runs/qwen3_8b_hal_v2/train_v2.py`

**Root cause:** `lora_dropout=0.05` caused Unsloth to patch **0/36** QKV, O, and
MLP layers (vs 36/36/36 with dropout=0). Log warned: *"Unsloth will patch all
other layers, except LoRA matrices, causing a performance hit."* Training would
have been ~2–3× slower with no regularisation benefit meaningful at this scale.

**Fix:** `lora_dropout=0` restores all 36 fused layers.

---

### chore: replace deprecated HF_HUB_ENABLE_HF_TRANSFER env var

**File:** `training_runs/qwen3_8b_hal_v2/train_v2.py`

`HF_HUB_ENABLE_HF_TRANSFER` is deprecated as of huggingface-hub current release.
Replaced with `HF_XET_HIGH_PERFORMANCE=1` per the deprecation warning.

---

### feat: add Python file-based logging for full debug capture

**File:** `training_runs/qwen3_8b_hal_v2/train_v2.py`

Added `logging.basicConfig` with both a `StreamHandler` (stdout) and a
`FileHandler` writing to `train_v2_debug.log`. Ensures Python-level log output
(INFO/WARNING/ERROR from Unsloth, Transformers, TRL) is captured regardless of
how the script is launched.

**Note:** `OUTPUT_DIR` constant moved above the logging setup so the file handler
path resolves correctly before any imports trigger log output.

---

### feat: add run_v2.sh launcher with merged stdout+stderr

**File:** `training_runs/qwen3_8b_hal_v2/run_v2.sh` *(new)*

`bash run_v2.sh` replaces ad-hoc `python train_v2.py > train_v2.log`. Uses
`2>&1 | tee -a train_v2.log` so the full traceback of any crash is captured.
Passes `CUDA_VISIBLE_DEVICES=0` explicitly and uses the venv Python.

---

### feat: add monitor_v2.sh completion poller

**File:** `training_runs/qwen3_8b_hal_v2/monitor_v2.sh` *(new)*

Polls `train_v2.log` every 5 minutes, prints GPU stats on completion, and
exits if the process dies. Mirrors the pattern of `qwen3_8b_hal_v1/monitor.sh`.

---

## [v2.0.0] — 2026-05-17

### feat: initial v2 training setup with HA-native tool schema

**Phase D** — introduced `train_v2.py` with:
- HA-native tool definitions (12 tools: `HassTurnOn`, `HassTurnOff`,
  `HassLightSet`, `HassClimateSetTemperature`, `HassClimateGetTemperature`,
  `HassListAddItem`, `HassMediaUnpause`, `HassMediaPause`, `HassMediaNext`,
  `HassMediaPrevious`, `HassSetVolume`, `GetLiveContext`)
- Qwen2.5 chat template (replaces Qwen3 native template which crashes on plain
  dict messages)
- Stronger LoRA: r=32, alpha=32 (vs v1's r=16, alpha=16)
- Lower LR: 5e-5 (vs v1's 2e-4) for finer convergence on tool-call format
- 3 epochs with step-level eval/save and `load_best_model_at_end`
- Mixed dataset: 70% tool-call, 20% no-tool, 10% error/ambiguous examples

**v2 dataset** generated via Ollama (qwen2.5:14b) with Gemini 2.5 Pro fallback:
1,907 examples in `training_data/v2/hal_training_v2.jsonl`.

---

## [v1.0.0] — 2026-05-17

### feat: initial v1 training — Qwen3-8B LoRA on HAL smart-home dataset

**Phase A–C** — `train.py` with:
- 1,490 examples across 22 domains (energy, HVAC, Tesla, Amber pricing,
  irrigation, lighting, security, appliances, web search, multi-domain)
- LoRA r=16, alpha=16, lr=2e-4, 2 epochs
- Result: train_loss=0.1815, 354 steps, ~51 min on RTX 5060 Ti

Model deprecated after v1 in favour of v2 with HA-native tool calling format.
Artifacts retained: `lora_adapter/`, `merged_bf16/`, `gguf/`.
