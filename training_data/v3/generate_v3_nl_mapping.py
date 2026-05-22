#!/usr/bin/env python3
"""
Generate ~400 natural-language → sensor mapping training examples.

For each target entity, generates 3-5 colloquial user phrasings of "what is this
entity's value", paired with a correct GetLiveContext call and a HAL-styled response.

Output: nl_sensor_mapping.jsonl
"""

import json
import re
import sys
import time
import random
import requests

STATES_PATH = "/mnt/zardos/charm-hal-env/training_data/v3/ha_states.json"
OUTPUT_PATH = "/mnt/zardos/charm-hal-env/training_data/v3/nl_sensor_mapping.jsonl"
OLLAMA_URL  = "http://localhost:11434/v1/chat/completions"
GEN_MODEL   = "qwen2.5:14b"

random.seed(42)

ha_states = json.load(open(STATES_PATH))
ENTITY_MAP = {s["entity_id"]: s for s in ha_states}

HAL_SYSTEM = (
    "You are HAL, the conversational AI for Charm's smart home in Clyde North, Victoria, "
    "Australia. Time zone Australia/Melbourne. The home runs on Amber Electric wholesale "
    "pricing (cents/kWh) and a FoxESS battery with solar.\n"
    "Persona: calm, measured, slightly formal — like HAL 9000. Address the user as \"Charm\". "
    "Never open with \"Great!\", \"Sure!\", \"Of course!\", \"Certainly!\" or \"Happy to help\". "
    "Never apologise. Status replies are 1-4 sentences.\n"
    "When reading state with GetLiveContext, prefer calling it with only the `domain` argument "
    "and select the relevant entity from the returned list."
)

# ── Target entities: the ones the model consistently gets wrong ─────────────
TARGET_ENTITIES = [
    # Weather / UV
    ("sensor.openweathermap_uv_index",           "sensor",  "UV Index",                  ["is it sunny outside", "do I need sunscreen today", "whats the UV today", "how strong is the UV right now", "UV level outside"]),
    ("sensor.openweathermap_rain_intensity",     "sensor",  "Rain Intensity",             ["is it raining right now", "whats the rain intensity", "how heavy is the rain", "any rain right now"]),
    ("sensor.openweathermap_cloud_coverage",     "sensor",  "Cloud Coverage",             ["how cloudy is it today", "whats the cloud cover", "is it overcast", "how much cloud cover"]),
    ("sensor.openweathermap_humidity",           "sensor",  "Humidity",                   ["how humid is it outside", "whats the humidity", "how muggy is it today"]),
    ("sensor.openweathermap_temperature",        "sensor",  "Outside Temperature",        ["how hot is it outside", "whats the temperature outside", "how cold is it", "current outdoor temp"]),
    ("sensor.openweathermap_dew_point_temperature", "sensor", "Dew Point",               ["whats the dew point", "how is the dew point today"]),
    ("sensor.openweathermap_wind_speed",         "sensor",  "Wind Speed",                 ["how windy is it", "whats the wind speed", "is it breezy today"]),
    ("sensor.clyde_north_uv_max_index_0",        "sensor",  "UV Max Index Today",         ["whats the max UV today", "how bad will UV get today", "peak UV forecast"]),

    # Solcast / Solar forecast
    ("sensor.solcast_pv_forecast_forecast_today",    "sensor", "Solcast Forecast Today",    ["how much solar will we get today", "whats the solar forecast for today", "predicted solar generation today", "solcast today"]),
    ("sensor.solcast_pv_forecast_forecast_tomorrow", "sensor", "Solcast Forecast Tomorrow", ["how much solar tomorrow", "whats the solar forecast tomorrow", "predicted solar generation tomorrow"]),
    ("sensor.solcast_pv_forecast_power_now",         "sensor", "Solcast Power Now",         ["whats the solar power right now", "how much power from solar currently", "current solar output"]),
    ("sensor.solcast_pv_forecast_peak_forecast_today", "sensor", "Solcast Peak Today",    ["whats peak solar today", "when is solar peak", "maximum solar forecast today"]),
    ("sensor.solcast_pv_forecast_forecast_remaining_today", "sensor", "Solar Remaining Today", ["how much solar left today", "remaining solar generation today"]),

    # FoxESS / Home battery
    ("sensor.skynet_sun_harvester_battery_soc_1",  "sensor", "Home Battery SoC",          ["whats the home battery level", "how charged is the house battery", "how full is the foxess battery", "home battery percentage", "whats the battery soc"]),
    ("sensor.skynet_sun_harvester_battery_soc_2",  "sensor", "Battery SoC 2",             ["whats battery 2 charge", "second battery level"]),
    ("sensor.skynet_sun_harvester_battery_charge_today", "sensor", "Battery Charged Today", ["how much has the battery charged today", "battery charge amount today"]),
    ("sensor.skynet_sun_harvester_battery_discharge_today", "sensor", "Battery Discharged Today", ["how much has the battery discharged today", "battery discharge today"]),
    ("sensor.skynet_sun_harvester_min_soc",        "sensor", "Battery Min SoC",           ["whats the battery minimum charge set to", "what is the min SoC"]),

    # Tesla / White Python
    ("sensor.white_python_battery_level",        "sensor", "Tesla Battery Level",          ["whats my car battery level", "how charged is the tesla", "how charged is white python", "car battery percentage", "ev battery level"]),
    ("sensor.white_python_battery_range",        "sensor", "Tesla Battery Range",          ["how far can I drive on current charge", "whats my car range", "how many km left in the tesla", "remaining driving range", "car range in km"]),
    ("sensor.white_python_charging",             "sensor", "Tesla Charging Status",        ["is the tesla charging", "is my car plugged in and charging", "car charging status"]),
    ("sensor.white_python_time_to_full_charge",  "sensor", "Time to Full Charge",          ["how long to charge the car", "when will the tesla be full", "time to full charge"]),
    ("sensor.white_python_inside_temperature",   "sensor", "Tesla Cabin Temperature",      ["how hot is it inside the car", "whats the cabin temperature in the tesla", "car interior temperature"]),

    # Device batteries
    ("sensor.charm_s_phone_battery_level",       "sensor", "Phone Battery Level",          ["how charged is charms phone", "whats my phone battery", "charm phone battery level"]),
    ("sensor.ep3x_bd7670488_battery",            "sensor", "EP3x Battery",                 ["whats the ep3x battery", "how charged is the ep3x device"]),
    ("sensor.kunurobo_battery",                  "sensor", "Kunurobo Battery",              ["how charged is the robot vacuum", "robot vacuum battery", "kunurobo battery level"]),
    ("sensor.front_door_battery_3",              "sensor", "Front Door Sensor Battery",    ["front door sensor battery", "how charged is the front door sensor"]),

    # Energy / Grid
    ("sensor.daily_grid_import_energy",          "sensor", "Daily Grid Import",            ["how much energy have we imported today", "todays grid energy use", "how much electricity from grid today", "grid usage today"]),
    ("sensor.yesterday_grid_import_energy",      "sensor", "Yesterday Grid Import",        ["how much energy did we import yesterday", "yesterdays grid usage", "grid import yesterday"]),
    ("sensor.daily_grid_import_cost_cents",      "sensor", "Daily Grid Cost",              ["how much has electricity cost today", "todays electricity bill so far", "daily grid cost"]),
    ("sensor.charm_house_feed_in_price",         "sensor", "Feed-in Price",                ["whats the current feed-in price", "how much are we getting for solar export", "feed in tariff now"]),
]

def get_entity_state_line(eid):
    s = ENTITY_MAP.get(eid)
    if not s:
        return f"{eid}: unknown"
    fn   = s["attributes"].get("friendly_name", eid)
    state = s["state"]
    unit  = s["attributes"].get("unit_of_measurement", "")
    return f"{fn}: {state}{' ' + unit if unit else ''}"

def call_ollama(messages, max_retries=3):
    payload = {"model": GEN_MODEL, "messages": messages, "temperature": 0.7, "max_tokens": 256}
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=90)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  Ollama error (attempt {attempt+1}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None

def build_tool_result(eid, domain):
    """Return a synthetic tool result showing the entity value."""
    state_line = get_entity_state_line(eid)
    return f"GetLiveContext result for domain '{domain}':\n  {state_line}"

def generate_hal_response(prompt, eid, tool_result):
    s = ENTITY_MAP.get(eid, {})
    fn    = s.get("attributes", {}).get("friendly_name", eid) if s else eid
    state = s.get("state", "unknown") if s else "unknown"
    unit  = s.get("attributes", {}).get("unit_of_measurement", "") if s else ""

    gen_prompt = f"""Write a short HAL (HAL 9000 style home AI) response to this smart-home query.
HAL is calm, formal, addresses user as "Charm". No greetings. 1-3 sentences max.
Use the sensor data below to answer factually.

User: {prompt}
Sensor: {fn} = {state}{' ' + unit if unit else ''}
Full tool result: {tool_result}

Write ONLY the HAL response (no metadata, no JSON):"""

    return call_ollama([
        {"role": "system", "content": "Generate a short, factual HAL 9000 style smart home AI response."},
        {"role": "user", "content": gen_prompt},
    ])

# ── Load existing to resume ─────────────────────────────────────────────────
existing = set()
try:
    for line in open(OUTPUT_PATH):
        ex = json.loads(line)
        user_msg = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
        existing.add(user_msg)
    print(f"Resuming: {len(existing)} already written", flush=True)
except FileNotFoundError:
    pass

out_f = open(OUTPUT_PATH, "a")
written = 0

for eid, domain, label, phrasings in TARGET_ENTITIES:
    print(f"\n[{eid}] ({label})", flush=True)
    for phrase in phrasings:
        if phrase in existing:
            print(f"  SKIP: {phrase}", flush=True)
            continue

        tool_result  = build_tool_result(eid, domain)
        hal_resp     = generate_hal_response(phrase, eid, tool_result)

        if not hal_resp:
            print(f"  FAIL: {phrase}", flush=True)
            continue

        # 50% chance of domain-only call, 50% chance of domain+name call
        if random.random() < 0.5:
            tool_args = {"domain": domain}
        else:
            fn = ENTITY_MAP.get(eid, {}).get("attributes", {}).get("friendly_name", label) if ENTITY_MAP.get(eid) else label
            tool_args = {"domain": domain, "name": fn}

        example = {
            "messages": [
                {"role": "system", "content": HAL_SYSTEM},
                {"role": "user",   "content": phrase},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "tc_001",
                        "type": "function",
                        "function": {
                            "name": "GetLiveContext",
                            "arguments": tool_args,
                        },
                    }],
                },
                {"role": "tool", "content": tool_result},
                {"role": "assistant", "content": hal_resp},
            ]
        }
        out_f.write(json.dumps(example) + "\n")
        out_f.flush()
        written += 1
        existing.add(phrase)
        print(f"  OK: {phrase[:60]} → {hal_resp[:60]}...", flush=True)

out_f.close()
print(f"\nDone. Written: {written}", flush=True)
print(f"Output: {OUTPUT_PATH}", flush=True)
