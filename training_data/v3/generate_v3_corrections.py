#!/usr/bin/env python3
"""
Generate corrected training examples for v3 from probe failures.

Reads probe_responses_2026-05-19.jsonl, filters failures, classifies each
into a bucket, then for each tractable failure calls qwen2.5:14b to write
the correct assistant response (given the right GetLiveContext arguments).

Output: corrections_2026-05-19.jsonl
"""

import json
import re
import sys
import time
import requests

# ── Paths ──────────────────────────────────────────────────────────────────
PROBE_PATH   = "/mnt/zardos/charm-hal-env/training_data/v3/probe_responses_2026-05-19.jsonl"
STATES_PATH  = "/mnt/zardos/charm-hal-env/training_data/v3/ha_states.json"
OUTPUT_PATH  = "/mnt/zardos/charm-hal-env/training_data/v3/corrections_2026-05-19.jsonl"

OLLAMA_URL   = "http://localhost:11434/v1/chat/completions"
GEN_MODEL    = "qwen2.5:14b"

# ── Load data ──────────────────────────────────────────────────────────────
probe_records = [json.loads(l) for l in open(PROBE_PATH)]
ha_states     = json.load(open(STATES_PATH))

# Build entity lookup: entity_id → {friendly_name, state, device_class, unit}
ENTITY_MAP = {}
for s in ha_states:
    eid = s["entity_id"]
    ENTITY_MAP[eid] = {
        "friendly_name": s["attributes"].get("friendly_name", eid),
        "state":         s["state"],
        "device_class":  s["attributes"].get("device_class", ""),
        "unit":          s["attributes"].get("unit_of_measurement", ""),
        "attributes":    s["attributes"],
    }

# ── Failure detection (same heuristics as probe_v3_analyze.py) ─────────────
FAIL_PATTERNS = [
    r"i (?:do not|don't) have",
    r"no (?:exposed )?entit(?:y|ies)",
    r"no such (?:entity|sensor)",
    r"not (?:available|found) in the system",
    r"i can'?t (?:find|access|provide|determine)",
    r"could (?:you|i) (?:please|kindly)?\s*(?:provide|clarify)",
    r"please (?:provide|clarify|check|specify)",
    r"unable to (?:find|access|process|determine|provide)",
    r"i'?m (?:unable|sorry|not (?:able|sure))",
    r"i apologi[zs]e",
    r"is currently unknown",
    r"unavailable",
    r"(?:invalid|incorrect) (?:entity|name|query)",
    r"there (?:is|was|seems) (?:no|an issue|a problem)",
    r"sorry,? i",
    r"cannot (?:find|provide|access|determine)",
    r"i'?m (?:not aware|afraid)",
    r"no information (?:on|about|available)",
]

def is_failure(speech):
    if not speech:
        return True
    sl = speech.lower()
    return any(re.search(p, sl) for p in FAIL_PATTERNS)

failures = [r for r in probe_records if is_failure(r.get("response", ""))]
print(f"Total failures: {len(failures)}", flush=True)

# ── HAL system prompt (same as v2, used as training system message) ─────────
HAL_SYSTEM = (
    "You are HAL, the conversational AI for Charm's smart home in Clyde North, Victoria, "
    "Australia. Time zone Australia/Melbourne. The home runs on Amber Electric wholesale "
    "pricing (cents/kWh) and a FoxESS battery with solar.\n"
    "Persona: calm, measured, slightly formal — like HAL 9000. Address the user as \"Charm\". "
    "Never open with \"Great!\", \"Sure!\", \"Of course!\", \"Certainly!\" or \"Happy to help\". "
    "Never apologise. Status replies are 1-4 sentences; analysis may be longer. "
    "Prices in cents/kWh (state is $/kWh, multiply by 100). "
    "If an entity is unavailable: \"That data is unavailable right now, Charm.\" "
    "If the Tesla is unavailable: \"The car is sleeping, Charm. Open the Tesla app to wake it.\" "
    "When reading state with GetLiveContext, prefer calling it with only the `domain` argument "
    "(e.g. {\"domain\":\"sensor\"}) and select the relevant entity from the returned list. "
    "Only add a `name` filter when the user has named a specific exposed entity. "
    "Do not invent entity names — if uncertain, call GetLiveContext with domain only."
)

# ── Category → GetLiveContext call + entity hints ──────────────────────────
# Maps each category to: (tool_args dict, list of entity_ids to show as tool result, notes)
CATEGORY_CONFIG = {
    "weather_direct": {
        "domain": "sensor",
        "key_entities": [
            "sensor.openweathermap_temperature",
            "sensor.openweathermap_apparent_temperature",
            "sensor.openweathermap_humidity",
            "sensor.openweathermap_cloud_coverage",
            "sensor.openweathermap_rain_intensity",
            "sensor.openweathermap_wind_speed",
            "sensor.openweathermap_wind_gust_speed",
            "sensor.openweathermap_pressure",
            "sensor.openweathermap_uv_index",
            "sensor.openweathermap_condition",
            "sensor.clyde_north_rain_chance_0",
            "sensor.clyde_north_rain_amount_min_0",
            "sensor.clyde_north_rain_amount_max_0",
            "sensor.clyde_north_extended_text_0",
            "sensor.clyde_north_short_text_0",
            "sensor.clyde_north_temp_max_0",
            "sensor.clyde_north_temp_min_0",
            "sensor.clyde_north_uv_max_index_0",
            "sensor.clyde_north_uv_forecast_0",
            "sensor.clyde_north_astronomical_sunrise_time_0",
            "sensor.clyde_north_astronomical_sunset_time_0",
            "sensor.clyde_north_warnings",
            "sensor.home_charm_rain_since_9am",
            "weather.openweathermap",
            "weather.clyde_north",
        ],
        "domain_call": "sensor",
        "notes": (
            "Clyde North sensors (_0 suffix = today): rain_chance_0=% chance rain, "
            "rain_amount_min/max_0=mm expected, extended_text_0=full forecast text, "
            "temp_max/min_0=today's high/low, astronomical_sunrise/sunset_time_0=times. "
            "OpenWeatherMap sensors: current conditions (rain_intensity=mm/h, uv_index, cloud_coverage%). "
            "For tomorrow forecasts: cite weather.clyde_north or weather.openweathermap entity "
            "(has daily forecast in attributes) and note exact sensor only covers today."
        ),
    },
    "weather_indirect": {
        "domain": "sensor",
        "key_entities": [
            "sensor.openweathermap_uv_index",
            "sensor.openweathermap_rain_intensity",
            "sensor.openweathermap_humidity",
            "sensor.openweathermap_cloud_coverage",
            "sensor.openweathermap_temperature",
            "sensor.openweathermap_wind_speed",
            "sensor.clyde_north_uv_max_index_0",
        ],
        "domain_call": "sensor",
        "notes": "For indirect weather inference (umbrella, sunglasses, jacket), read UV + cloud + rain + temperature.",
    },
    "solar_direct": {
        "domain": "sensor",
        "key_entities": [
            "sensor.solcast_pv_forecast_forecast_remaining_today",
            "sensor.solcast_pv_forecast_forecast_tomorrow",
            "sensor.solcast_pv_forecast_forecast_this_hour",
            "sensor.solcast_pv_forecast_forecast_next_hour",
            "sensor.solcast_pv_forecast_peak_forecast_today",
            "sensor.solcast_pv_forecast_peak_forecast_tomorrow",
            "sensor.solcast_pv_forecast_peak_time_today",
            "sensor.solcast_pv_forecast_peak_time_tomorrow",
            "sensor.solcast_pv_forecast_api_used",
            "sensor.solcast_pv_forecast_api_limit",
            "sensor.foxess_modbus_skynet_sun_harvester_pv_power_skynet_sun_harvester",
            "sensor.skynet_sun_harvester_total_yield_today",
            "sensor.skynet_sun_harvester_total_yield_total",
            "sensor.60hd15305b3m173_daily_generation",
            "input_number.solar_forecast_yesterday_actual",
            "sensor.charm_house_feed_in_price",
            "sensor.solar_yield_expected_remaining",
        ],
        "domain_call": "sensor",
        "notes": (
            "Solcast PV Forecast: 'Forecast Remaining Today'=remaining kWh today, "
            "'Forecast Tomorrow'=tomorrow total kWh, 'Peak Forecast Today/Tomorrow'=peak W, "
            "'Peak Time Today/Tomorrow'=when peak occurs. "
            "FoxESS 'Yield Today' and 'Daily generated energy' track today's actual generation. "
            "'Yesterday Actual' (input_number) tracks prior day. "
            "Feed-in price is $/kWh — multiply by 100 for c/kWh."
        ),
    },
    "solar_indirect": {
        "domain": "sensor",
        "key_entities": [
            "sensor.solcast_pv_forecast_forecast_remaining_today",
            "sensor.solcast_pv_forecast_forecast_tomorrow",
            "sensor.foxess_modbus_skynet_sun_harvester_pv_power_skynet_sun_harvester",
            "sensor.skynet_sun_harvester_battery_soc_1",
            "sensor.solcast_pv_forecast_peak_forecast_today",
            "sensor.charm_house_feed_in_price",
            "input_number.solar_forecast_yesterday_actual",
        ],
        "domain_call": "sensor",
        "notes": "Infer solar advice from Solcast forecast + current PV power + battery SoC.",
    },
    "battery_direct": {
        "domain": "sensor",
        "key_entities": [
            "sensor.skynet_sun_harvester_battery_soc_1",
            "sensor.skynet_sun_harvester_battery_soc_2",
            "sensor.skynet_sun_harvester_battery_soh_1",
            "sensor.skynet_sun_harvester_battery_charge_today",
            "sensor.skynet_sun_harvester_battery_discharge_today",
            "sensor.skynet_sun_harvester_min_soc",
            "sensor.skynet_sun_harvester_max_soc",
            "sensor.white_python_battery_level",
            "sensor.white_python_battery_range",
        ],
        "domain_call": "sensor",
        "notes": "Home battery = FoxESS/Skynet Sun Harvester. Car battery = White Python (Tesla). Clarify which if ambiguous.",
    },
    "battery_indirect": {
        "domain": "sensor",
        "key_entities": [
            "sensor.skynet_sun_harvester_battery_soc_1",
            "sensor.skynet_sun_harvester_battery_soc_2",
            "sensor.solcast_pv_forecast_forecast_remaining_today",
        ],
        "domain_call": "sensor",
        "notes": "Infer whether home battery will last the night based on SoC + remaining solar forecast.",
    },
    "device_batteries": {
        "domain": "sensor",
        "key_entities": [
            "sensor.charm_s_phone_battery_level",
            "sensor.ep3x_bd7670488_battery",
            "sensor.front_door_battery_3",
            "sensor.front_door_battery_4",
            "sensor.motion_sensor_door_side_battery",
            "sensor.kunurobo_battery",
        ],
        "domain_call": "sensor",
        "notes": "Device batteries have device_class=battery. Call GetLiveContext(domain=sensor) and filter for battery device_class.",
    },
    "energy_direct": {
        "domain": "sensor",
        "key_entities": [
            "sensor.daily_grid_import_energy",
            "sensor.yesterday_grid_import_energy",
            "sensor.yesterday_grid_import_cost_cents",
            "sensor.skynet_sun_harvester_battery_discharge_today",
            "sensor.skynet_sun_harvester_battery_charge_today",
            "sensor.60hd15305b3m173_generation_power",
            "sensor.foxess_modbus_skynet_sun_harvester_pv_power_skynet_sun_harvester",
            "sensor.60hd15305b3m173_eps_power",
            "sensor.60hd15305b3m173_eps_r_power",
            "sensor.60hd15305b3m173_eps_s_power",
        ],
        "domain_call": "sensor",
        "notes": (
            "Grid import today = sensor.daily_grid_import_energy (kWh). "
            "Yesterday = sensor.yesterday_grid_import_energy + sensor.yesterday_grid_import_cost_cents. "
            "EPS Power = Emergency Power Supply output (sensor.60hd15305b3m173_eps_power). "
            "Battery charge/discharge today = skynet_sun_harvester_battery_charge/discharge_today."
        ),
    },
    "energy_indirect": {
        "domain": "sensor",
        "key_entities": [
            "sensor.daily_grid_import_energy",
            "sensor.skynet_sun_harvester_battery_discharge_today",
            "sensor.solcast_pv_forecast_forecast_remaining_today",
            "sensor.charm_house_feed_in_price",
            "sensor.60hd15305b3m173_generation_power",
        ],
        "domain_call": "sensor",
        "notes": "Infer cost/savings advice from today's import + solar forecast + battery discharge.",
    },
    "tesla_direct": {
        "domain": "sensor",
        "key_entities": [
            "sensor.white_python_battery_level",
            "sensor.white_python_battery_range",
            "sensor.white_python_charging",
            "sensor.white_python_charger_power",
            "sensor.white_python_time_to_full_charge",
            "sensor.white_python_inside_temperature",
            "sensor.white_python_outside_temperature",
            "sensor.white_python_charge_energy_added",
            "sensor.charm_s_phone_car_range_remaining",
        ],
        "domain_call": "sensor",
        "notes": "Tesla is White Python. If unavailable (sleeping), say 'The car is sleeping, Charm. Open the Tesla app to wake it.'",
    },
    "tesla_indirect": {
        "domain": "sensor",
        "key_entities": [
            "sensor.white_python_battery_level",
            "sensor.white_python_battery_range",
            "sensor.white_python_time_to_full_charge",
        ],
        "domain_call": "sensor",
        "notes": "Infer Tesla advice (enough charge for a trip) from battery_range vs typical city distance ~40km.",
    },
    "doors_locks_security": {
        "domain": "binary_sensor",
        "key_entities": [],
        "domain_call": "binary_sensor",
        "notes": "Door/lock status from binary_sensor domain. Alarm from alarm_control_panel domain.",
    },
    "presence": {
        "domain": "device_tracker",
        "key_entities": [],
        "domain_call": "device_tracker",
        "notes": "Presence = device_tracker domain. 'home' state = home, 'not_home' = away.",
    },
    "lights_status": {
        "domain": "light",
        "key_entities": [],
        "domain_call": "light",
        "notes": "Light states from light domain. 'on'/'off' state. Include brightness if available.",
    },
    "general_status": {
        "domain": "sensor",
        "key_entities": [],
        "domain_call": "sensor",
        "notes": "General status queries — call domain-only GetLiveContext and report relevant entities.",
    },
    "hvac_indirect": {
        "domain": "climate",
        "key_entities": [],
        "domain_call": "climate",
        "notes": "HVAC inference from climate domain. Report current mode + temperature vs comfort range.",
    },
}

# ── Build synthetic tool result strings ────────────────────────────────────
def build_tool_result(category):
    """Build a plausible tool result string for the given category."""
    config = CATEGORY_CONFIG.get(category, {})
    key_entities = config.get("key_entities", [])

    if not key_entities:
        # For domains without specific entities listed, show domain entities count
        domain = config.get("domain_call", "sensor")
        domain_entities = [s for s in ha_states if s["entity_id"].startswith(domain + ".")]
        lines = [f"Found {len(domain_entities)} entities in domain '{domain}':"]
        for s in domain_entities[:8]:
            fn = s["attributes"].get("friendly_name", s["entity_id"])
            unit = s["attributes"].get("unit_of_measurement", "")
            lines.append(f"  {fn}: {s['state']}{' ' + unit if unit else ''}")
        if len(domain_entities) > 8:
            lines.append(f"  ... and {len(domain_entities) - 8} more")
        return "\n".join(lines)

    lines = [f"Found {len(key_entities)} matching entities:"]
    for eid in key_entities:
        info = ENTITY_MAP.get(eid, {})
        fn   = info.get("friendly_name", eid)
        state = info.get("state", "unknown")
        unit  = info.get("unit", "")
        lines.append(f"  {fn}: {state}{' ' + unit if unit else ''}")
    return "\n".join(lines)

# ── Ollama call ─────────────────────────────────────────────────────────────
def call_ollama(messages, max_retries=3):
    payload = {
        "model": GEN_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  Ollama error (attempt {attempt+1}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None

# ── Generate response using qwen2.5:14b ────────────────────────────────────
def generate_hal_response(prompt, category, tool_result):
    config = CATEGORY_CONFIG.get(category, {})
    notes  = config.get("notes", "")

    gen_prompt = f"""You are helping generate training data for HAL, a smart home AI.
Write a short, correct HAL response (1-4 sentences) to the user's question, based on the sensor data below.
HAL persona: calm, slightly formal, like HAL 9000. Address user as "Charm".
No "Great!", "Sure!", "Of course!". No apologies. Be direct and informative.

User question: {prompt}
Category: {category}
Notes: {notes}

Sensor/entity data available:
{tool_result}

Write ONLY the HAL response text (no JSON, no metadata):"""

    messages = [
        {"role": "system", "content": "You generate short training responses for a smart home AI named HAL."},
        {"role": "user", "content": gen_prompt},
    ]
    return call_ollama(messages)

# ── Bucket classification ──────────────────────────────────────────────────
BUCKET_B_PHRASES = [
    "camera", "frigate", "notification", "voice", "music", "spotify", "netflix",
    "play", "media", "photo", "video",
]

def classify_bucket(record):
    """Rough bucket classification:
    B = entity genuinely unavailable (cameras, phone GPS exact location, etc.)
    C = ambiguous term → should ask clarifying question
    D = temporal/historical query
    A/F/G = entity exists but model gave up or hallucinated
    """
    prompt = record["prompt"].lower()
    resp   = (record.get("response") or "").lower()

    # Bucket B: genuinely not exposed
    if any(p in prompt for p in BUCKET_B_PHRASES):
        return "B"

    # Bucket C: ambiguous — "the battery", "the temperature", "the lights"
    ambiguous_terms = ["which battery", "which temperature", "which light", "what battery"]
    if any(t in prompt for t in ["battery charge", "the battery", "my battery"]):
        # If category is battery_direct and prompt doesn't specify home/car → C
        if record["category"] in ("battery_direct", "battery_indirect") and \
           "car" not in prompt and "home" not in prompt and "foxess" not in prompt and "tesla" not in prompt:
            return "C"

    # Bucket D: temporal/historical
    if any(t in prompt for t in ["yesterday", "last week", "last month", "this week", "how much did"]):
        return "D"

    # Default: A (entity exists, model gave up)
    return "A"

# ── Main loop ─────────────────────────────────────────────────────────────
written = 0
skipped_b = 0
skip_cats = set()

# Load existing output to resume
existing_prompts = set()
try:
    for line in open(OUTPUT_PATH):
        ex = json.loads(line)
        user_msg = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
        existing_prompts.add(user_msg)
    print(f"Resuming: {len(existing_prompts)} already written", flush=True)
except FileNotFoundError:
    pass

out_f = open(OUTPUT_PATH, "a")

for i, record in enumerate(failures):
    prompt   = record["prompt"]
    category = record["category"]

    if prompt in existing_prompts:
        print(f"[{i+1}/{len(failures)}] SKIP (already done): {prompt[:60]}", flush=True)
        continue

    bucket = classify_bucket(record)
    print(f"[{i+1}/{len(failures)}] [{category}] bucket={bucket} | {prompt[:70]}", flush=True)

    if bucket == "B":
        skipped_b += 1
        print(f"  → Skip (entity not exposed)", flush=True)
        continue

    config = CATEGORY_CONFIG.get(category, {})
    domain_call = config.get("domain_call", "sensor")
    tool_result  = build_tool_result(category)

    if bucket == "C":
        # Generate a clarifying question
        cat = category
        if "battery" in cat:
            hal_resp = (
                "Charm, there are multiple batteries reporting state. "
                "The home FoxESS battery is currently at "
                f"{ENTITY_MAP.get('sensor.skynet_sun_harvester_battery_soc_1', {}).get('state', '—')}% SoC, "
                "and the vehicle (White Python) is at "
                f"{ENTITY_MAP.get('sensor.white_python_battery_level', {}).get('state', '—')}%. "
                "Which would you like a fuller report on?"
            )
        elif "temperature" in prompt.lower():
            hal_resp = (
                "There are several temperature readings available, Charm. "
                "Outside is currently "
                f"{ENTITY_MAP.get('sensor.openweathermap_temperature', {}).get('state', '—')} °C, "
                "with indoor climate zones also reporting. "
                "Which temperature are you interested in — outside, upstairs, or downstairs?"
            )
        else:
            hal_resp = generate_hal_response(prompt, category, tool_result)
            if not hal_resp:
                print(f"  → generation failed, skipping", flush=True)
                continue
    else:
        # Bucket A, D, F, G: generate the correct HAL response
        hal_resp = generate_hal_response(prompt, category, tool_result)
        if not hal_resp:
            print(f"  → generation failed, skipping", flush=True)
            continue

    # Build the training example
    example = {
        "messages": [
            {"role": "system", "content": HAL_SYSTEM},
            {"role": "user",   "content": prompt},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "tc_001",
                    "type": "function",
                    "function": {
                        "name": "GetLiveContext",
                        "arguments": {"domain": domain_call},
                    },
                }],
            },
            {"role": "tool", "content": tool_result},
            {"role": "assistant", "content": hal_resp.strip()},
        ]
    }

    out_f.write(json.dumps(example) + "\n")
    out_f.flush()
    written += 1
    print(f"  → wrote example ({written} total)", flush=True)

out_f.close()

print(f"\nDone. Written: {written}, Skipped (B): {skipped_b}", flush=True)
print(f"Output: {OUTPUT_PATH}", flush=True)
