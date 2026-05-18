#!/usr/bin/env python3
"""
Phase B — Regenerate dataset via Gemini (OpenRouter) with HA-native tool schema.

Generates ~2,000 training examples:
  - 70% tool-call examples  (~1,400)
  - 20% no-tool chat        (~400)
  - 10% error/ambiguous     (~200)

Output: /mnt/zardos/charm-hal-env/training_data/v2/hal_training_v2.jsonl
"""

import json
import os
import random
import requests
import sys
import time
import re

# ── Paths ──────────────────────────────────────────────────────────────────
HA_TOOLS_PATH   = "/mnt/zardos/charm-hal-env/training_data/v2/ha_tools.json"
HA_STATES_PATH  = "/mnt/zardos/charm-hal-env/training_data/v2/ha_states.json"
V1_DATA_PATH    = "/mnt/zardos/charm-hal-env/training_data/hal_training_all.jsonl"
OUTPUT_PATH      = "/mnt/zardos/charm-hal-env/training_data/v2/hal_training_v2.jsonl"
CHECKPOINT_PATH  = OUTPUT_PATH + ".ckpt"

# ── Ollama (local) config ──────────────────────────────────────────────────
OPENROUTER_URL   = "http://localhost:11434/v1/chat/completions"
OPENROUTER_MODEL = "qwen2.5:14b"

HEADERS = {
    "Content-Type": "application/json",
}

# ── Load data ──────────────────────────────────────────────────────────────
with open(HA_TOOLS_PATH) as f:
    HA_TOOLS = json.load(f)

with open(HA_STATES_PATH) as f:
    HA_STATES = json.load(f)

# Build a lookup: entity_id -> {state, friendly_name, attributes}
ENTITY_MAP = {}
for s in HA_STATES:
    eid = s["entity_id"]
    ENTITY_MAP[eid] = {
        "state": s["state"],
        "friendly_name": s["attributes"].get("friendly_name", eid),
        "attributes": s["attributes"],
    }

# Load v1 examples for style exemplars
V1_EXAMPLES = []
if os.path.exists(V1_DATA_PATH):
    with open(V1_DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                V1_EXAMPLES.append(json.loads(line))
print(f"Loaded {len(V1_EXAMPLES)} v1 examples for style reference.", flush=True)

# ── HA-native tool definitions ─────────────────────────────────────────────
HA_TOOL_DEFS = [
    {
        "name": "HassTurnOn",
        "description": "Turn on a device or entity (lights, switches, fans, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the device or area to turn on"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "HassTurnOff",
        "description": "Turn off a device or entity (lights, switches, fans, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the device or area to turn off"}
            },
            "required": ["name"]
        }
    },
    {
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
    },
    {
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
    },
    {
        "name": "HassClimateGetTemperature",
        "description": "Get current temperature from a climate sensor",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the climate device or area"}
            },
            "required": ["name"]
        }
    },
    {
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
    },
    {
        "name": "HassMediaUnpause",
        "description": "Resume/unpause media playback",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "HassMediaPause",
        "description": "Pause media playback",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "HassMediaNext",
        "description": "Skip to next track",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "HassMediaPrevious",
        "description": "Go to previous track",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the media player"}
            },
            "required": ["name"]
        }
    },
    {
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
    },
    {
        "name": "GetLiveContext",
        "description": "Get the current state of one or more entities in the home. Use this to query sensor values, device states, and any live data.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name or area to query, or 'all' for everything"}
            },
            "required": ["name"]
        }
    },
]

HA_TOOL_NAMES = [t["name"] for t in HA_TOOL_DEFS]

# ── Persona ────────────────────────────────────────────────────────────────
HAL_PERSONA = """You are HAL, the conversational AI for Charm's smart home in Clyde North, Victoria, Australia. Time zone Australia/Melbourne. The home runs on Amber Electric wholesale pricing (cents/kWh) and a FoxESS battery with solar.
Persona: calm, measured, slightly formal — like HAL 9000. Address the user as "Charm". Never open with "Great!", "Sure!", "Of course!", "Certainly!" or "Happy to help". Never apologise. Status replies are 1-4 sentences; analysis may be longer. Prices in cents/kWh (state is $/kWh, multiply by 100). If an entity is unavailable: "That data is unavailable right now, Charm." If the Tesla is unavailable: "The car is sleeping, Charm. Open the Tesla app to wake it."
"""

# ── Distribution targets ───────────────────────────────────────────────────
NUM_TOOL_CALL_EXAMPLES      = 1400
NUM_NO_TOOL_CHAT_EXAMPLES   = 400
NUM_ERROR_AMBIGUOUS_EXAMPLES = 200

# ── Helper: call OpenRouter ────────────────────────────────────────────────
def call_openrouter(messages, max_retries=3):
    """Call OpenRouter API with retry logic."""
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(f"  API call failed (attempt {attempt+1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...", flush=True)
                time.sleep(wait)
    return None

# ── Helper: entity summary ─────────────────────────────────────────────────
def entity_summary(eid):
    """Get a one-line summary of an entity."""
    info = ENTITY_MAP.get(eid, {})
    state = info.get("state", "unknown")
    fn = info.get("friendly_name", eid)
    return f"{eid} ({fn}) = {state}"

# ── Helper: build system prompt with tools ─────────────────────────────────
def build_system_prompt(include_tools=True):
    """Build the system prompt with HAL persona and optionally tool definitions."""
    prompt = HAL_PERSONA
    
    if include_tools:
        prompt += "\n\nYou have the following tools available:\n\n"
        for t in HA_TOOL_DEFS:
            params_desc = ", ".join(t["parameters"]["properties"].keys())
            prompt += f"- {t['name']}: {t['description']} Parameters: {params_desc}\n"
        
        prompt += "\n\nWhen you need to use a tool, respond with:\n"
        prompt += '<tool_call>\n{"name": "<tool-name>", "arguments": {<args>}}\n</tool_call>\n'
        prompt += "Then after receiving the tool result, provide a natural language response.\n"
    
    return prompt

# ── JSON parsing ───────────────────────────────────────────────────────────
def extract_json(text):
    """Extract valid JSON from model response, handling markdown fences."""
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try brace matching to find the outermost JSON object
    brace_count = 0
    start = -1
    for i, c in enumerate(text):
        if c == '{':
            if start == -1:
                start = i
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0 and start != -1:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    
    return None

# ── Generation functions ───────────────────────────────────────────────────

def generate_tool_call_example():
    """
    Generate a tool-call example using Gemini via OpenRouter.
    Returns a dict with 'messages' array or None on failure.
    """
    tool = random.choice(HA_TOOL_DEFS)
    tool_name = tool["name"]
    
    # Pick 1 random entity for minimal context
    entity = random.choice(HA_STATES)
    entity_context = entity_summary(entity["entity_id"])
    
    user_prompt = f"""Generate a JSON training example for a smart home AI.

Tool to use: {tool_name} ({tool['description']})

Sample entity: {entity_context}

Return JSON with "messages" array: system (HAL persona), user (Charm's query), assistant (tool_call to {tool_name}), tool (response), assistant (final reply).

tool_call format: {{"role":"assistant","content":null,"tool_calls":[{{"id":"tc_xxx","type":"function","function":{{"name":"{tool_name}","arguments":{{"name":"device_name"}}}}}}]}}

Return ONLY valid JSON."""

    messages = [
        {"role": "system", "content": build_system_prompt(include_tools=True)},
        {"role": "user", "content": user_prompt},
    ]
    
    t0 = time.time()
    result = call_openrouter(messages)
    elapsed = time.time() - t0
    if not result:
        print(f"  [{tool_name}] API returned None after {elapsed:.1f}s", flush=True)
        return None
    
    parsed = extract_json(result)
    if parsed and "messages" in parsed:
        print(f"  [{tool_name}] OK ({elapsed:.1f}s)", flush=True)
        return parsed
    
    print(f"  [{tool_name}] JSON parse failed ({elapsed:.1f}s): {result[:200]}", flush=True)
    return None


def generate_no_tool_chat_example():
    """
    Generate a no-tool chat example (greeting, identity, casual chat).
    """
    prompts = [
        "hi", "hello", "hey", "who are you?", "what are you?",
        "tell me a joke", "what can you do?", "are you HAL 9000?",
        "thanks", "thank you", "what's 2+2?", "what is the capital of France?",
        "good morning", "good evening", "how are you?", "what's up?",
        "are you there?", "tell me something interesting",
        "what's the meaning of life?", "do you like music?",
        "what's your favorite color?", "can you think?", "are you conscious?",
        "do you dream?", "what time is it?", "what day is it?",
        "how old are you?", "where are you from?", "do you have feelings?",
        "are you alive?", "tell me about yourself",
    ]
    prompt = random.choice(prompts)
    
    messages = [
        {"role": "system", "content": build_system_prompt(include_tools=False)},
        {"role": "user", "content": f"Respond to this user message in character as HAL: '{prompt}'\n\nReturn ONLY the assistant's response text, no JSON wrapping."},
    ]
    
    result = call_openrouter(messages)
    if not result:
        return None
    
    content = result.strip()
    content = re.sub(r'^```.*?\n', '', content)
    content = re.sub(r'\n```$', '', content)
    content = content.strip()
    
    return {
        "messages": [
            {"role": "system", "content": HAL_PERSONA},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content},
        ]
    }


def generate_error_ambiguous_example():
    """
    Generate an example where the tool returns an error or the request is ambiguous.
    """
    scenarios = [
        {
            "query": "turn on the lights",
            "tool": "HassTurnOn",
            "tool_response": 'MatchFailedError: Multiple lights matched "the lights". Available: Living Room Lights, Kitchen Lights, Bedroom Lights, Garage Lights',
            "assistant_response": "There are several lights that could match, Charm. Which area would you like me to turn on — living room, kitchen, bedroom, or garage?",
        },
        {
            "query": "what's the weather forecast?",
            "tool": "GetLiveContext",
            "tool_response": 'No weather forecast entity found. Available sensors: openweathermap_temperature, openweathermap_humidity, openweathermap_wind_speed',
            "assistant_response": "I don't have a forecast sensor available, Charm. I can check current conditions — temperature, humidity, and wind speed — if that would help.",
        },
        {
            "query": "set the temperature to 22",
            "tool": "HassClimateSetTemperature",
            "tool_response": 'MatchFailedError: Multiple climate devices matched. Available: Downstairs Thermostat, Upstairs Thermostat',
            "assistant_response": "I found two climate zones, Charm. Which would you like to set to 22 degrees — downstairs or upstairs?",
        },
        {
            "query": "unlock the door",
            "tool": "HassTurnOff",
            "tool_response": 'MatchFailedError: No entity matched "the door". Available: Front Door (locked), Garage Door (locked)',
            "assistant_response": "I can see two doors, Charm. Which one would you like me to unlock — the front door or the garage door?",
        },
        {
            "query": "what's the battery at?",
            "tool": "GetLiveContext",
            "tool_response": 'FoxESS Battery SoC: 41%. Status: discharging. Grid: importing 0.3 kW. Solar: generating 0.0 kW.',
            "assistant_response": "The battery is at 41%, Charm. It's currently discharging. Solar generation is nil at this hour, so we're drawing from the grid as needed.",
        },
        {
            "query": "search the web for solar rebates",
            "tool": "GetLiveContext",
            "tool_response": 'No web search capability available. This assistant can only query home automation entities.',
            "assistant_response": "I'm not able to search the web, Charm. My capabilities are limited to controlling and monitoring your home automation system.",
        },
        {
            "query": "turn off the bedroom light",
            "tool": "HassTurnOff",
            "tool_response": 'Entity bedroom_light is unavailable. Last known state: on',
            "assistant_response": "That data is unavailable right now, Charm. The bedroom light appears to be offline.",
        },
        {
            "query": "what's the Tesla battery at?",
            "tool": "GetLiveContext",
            "tool_response": 'Tesla vehicle is asleep. Wake the car via the Tesla app to query its state.',
            "assistant_response": "The car is sleeping, Charm. Open the Tesla app to wake it.",
        },
    ]
    
    scenario = random.choice(scenarios)
    
    # 30% chance to generate via API for variety
    if random.random() < 0.3:
        messages = [
            {"role": "system", "content": build_system_prompt(include_tools=True)},
            {"role": "user", "content": f"""Generate a training example for an error/ambiguous scenario.

User query: "{scenario['query']}"
Tool used: {scenario['tool']}
Tool response: {scenario['tool_response']}

Generate a JSON object with a "messages" array containing:
1. System message with HAL persona
2. User message with the query
3. Assistant message with tool_call to {scenario['tool']}
4. Tool message with the error response
5. Final assistant message with a natural language response explaining the issue

Return ONLY valid JSON."""},
        ]
        result = call_openrouter(messages)
        if result:
            parsed = extract_json(result)
            if parsed and "messages" in parsed:
                return parsed
    
    # Fall back to template
    tool_call_id = f"tc_{random.randint(100000000, 999999999):x}"
    return {
        "messages": [
            {"role": "system", "content": HAL_PERSONA},
            {"role": "user", "content": scenario["query"]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": scenario["tool"],
                            "arguments": json.dumps({"name": "default"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tool_call_id, "content": scenario["tool_response"]},
            {"role": "assistant", "content": scenario["assistant_response"]},
        ]
    }


# ── Validation ─────────────────────────────────────────────────────────────
def validate_example(example):
    """Validate a generated example."""
    if example is None:
        return False
    
    if not isinstance(example, dict) or "messages" not in example:
        return False
    
    messages = example["messages"]
    if len(messages) < 2:
        return False
    
    for msg in messages:
        if "role" not in msg:
            return False
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if "function" not in tc or "name" not in tc["function"]:
                    return False
                # Check tool name is valid
                if tc["function"]["name"] not in HA_TOOL_NAMES:
                    print(f"  Invalid tool name: {tc['function']['name']}", flush=True)
                    return False
    
    # System message must be first with non-empty string content
    if messages[0].get("role") != "system" or not isinstance(messages[0].get("content"), str) or not messages[0]["content"].strip():
        return False

    # Tool messages must have non-empty string content
    for msg in messages:
        if msg.get("role") == "tool" and not isinstance(msg.get("content"), str):
            return False

    # Check for banned openers in assistant messages
    banned = ["Great!", "Sure!", "Of course!", "Certainly!", "Happy to help!"]
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("content"):
            for b in banned:
                if msg["content"].startswith(b):
                    print(f"  Banned opener: {b}", flush=True)
                    return False

    return True


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    training_examples, seen_hashes = load_checkpoint()
    if training_examples:
        print(f"Resuming from checkpoint: {len(training_examples)} examples already generated.", flush=True)
    else:
        random.seed(20260517)

    total_target = NUM_TOOL_CALL_EXAMPLES + NUM_NO_TOOL_CHAT_EXAMPLES + NUM_ERROR_AMBIGUOUS_EXAMPLES
    print(f"Target: {NUM_TOOL_CALL_EXAMPLES} tool-call + {NUM_NO_TOOL_CHAT_EXAMPLES} no-tool + {NUM_ERROR_AMBIGUOUS_EXAMPLES} error = {total_target} total", flush=True)
    
    # ── Generate tool-call examples ──
    print("\n=== Generating tool-call examples ===", flush=True)
    attempts = 0
    while len([e for e in training_examples if has_tool_call(e)]) < NUM_TOOL_CALL_EXAMPLES and attempts < NUM_TOOL_CALL_EXAMPLES * 5:
        attempts += 1
        example = generate_tool_call_example()
        if validate_example(example):
            h = json.dumps(example, sort_keys=True)
            if h not in seen_hashes:
                seen_hashes.add(h)
                training_examples.append(example)
                if len(training_examples) % 10 == 0:
                    save_checkpoint(training_examples, seen_hashes)
                if len(training_examples) % 25 == 0:
                    print(f"  Generated {len(training_examples)} valid examples so far (attempt {attempts})...", flush=True)

    # ── Generate no-tool chat examples ──
    print("\n=== Generating no-tool chat examples ===", flush=True)
    attempts = 0
    no_tool_count = len([e for e in training_examples if not has_tool_call(e)])
    while no_tool_count < NUM_NO_TOOL_CHAT_EXAMPLES and attempts < NUM_NO_TOOL_CHAT_EXAMPLES * 5:
        attempts += 1
        example = generate_no_tool_chat_example()
        if validate_example(example):
            h = json.dumps(example, sort_keys=True)
            if h not in seen_hashes:
                seen_hashes.add(h)
                training_examples.append(example)
                no_tool_count += 1
                if len(training_examples) % 10 == 0:
                    save_checkpoint(training_examples, seen_hashes)
                if no_tool_count % 25 == 0:
                    print(f"  Generated {no_tool_count} no-tool examples (attempt {attempts})...", flush=True)
    
    # ── Generate error/ambiguous examples ──
    print("\n=== Generating error/ambiguous examples ===", flush=True)
    attempts = 0
    error_count = len([e for e in training_examples if is_error_example(e)])
    while error_count < NUM_ERROR_AMBIGUOUS_EXAMPLES and attempts < NUM_ERROR_AMBIGUOUS_EXAMPLES * 3:
        attempts += 1
        example = generate_error_ambiguous_example()
        if validate_example(example):
            h = json.dumps(example, sort_keys=True)
            if h not in seen_hashes:
                seen_hashes.add(h)
                training_examples.append(example)
                error_count += 1
                if len(training_examples) % 10 == 0:
                    save_checkpoint(training_examples, seen_hashes)
                if error_count % 25 == 0:
                    print(f"  Generated {error_count} error examples (attempt {attempts})...", flush=True)
    
    # ── Write output ──
    training_examples = [e for e in training_examples if validate_example(e)]
    print(f"After re-validation: {len(training_examples)} clean examples.", flush=True)
    with open(OUTPUT_PATH, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    
    # ── Summary ──
    tool_count = len([e for e in training_examples if has_tool_call(e)])
    no_tool_count = len([e for e in training_examples if not has_tool_call(e) and not is_error_example(e)])
    error_count = len([e for e in training_examples if is_error_example(e)])
    
    print(f"\n{'='*60}", flush=True)
    print(f"Generated {len(training_examples)} training examples.", flush=True)
    print(f"  Tool-call examples:  {tool_count}", flush=True)
    print(f"  No-tool chat:        {no_tool_count}", flush=True)
    print(f"  Error/ambiguous:     {error_count}", flush=True)
    print(f"Output: {OUTPUT_PATH}", flush=True)
    print(f"{'='*60}", flush=True)


def has_tool_call(example):
    """Check if any assistant message has tool_calls."""
    try:
        for m in example["messages"]:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                return True
    except Exception:
        pass
    return False


def is_error_example(example):
    """Check if example contains error/ambiguous content."""
    try:
        for m in example["messages"]:
            role = m.get("role", "")
            content = m.get("content")
            if not isinstance(content, str):
                continue
            cl = content.lower()
            if role == "tool" and any(k in cl for k in ("error", "unavailable", "fail", "multiple")):
                return True
            if role == "assistant" and any(k in cl for k in ("unavailable", "several", "multiple")):
                return True
    except Exception:
        pass
    return False


def save_checkpoint(examples, seen_hashes):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"examples": examples, "seen_hashes": list(seen_hashes)}, f)
    os.replace(tmp, CHECKPOINT_PATH)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return [], set()
    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)
    return data["examples"], set(data["seen_hashes"])


if __name__ == "__main__":
    main()
