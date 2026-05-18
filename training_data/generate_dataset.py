#!/usr/bin/env python3
"""
HAL fine-tuning dataset generator.

Generates 1,460 JSONL training examples for HAL, a Home Assistant LLM agent
for Charm's residence in Clyde North, Victoria, Australia.

Output format: messages array compatible with unsloth/trl SFTTrainer.
Each line of the JSONL file is one complete conversation including
system prompt, user turn, tool calls/results, and final HAL response.

Run:  python3 generate_dataset.py
"""

import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT_PATH = HERE / "ha_snapshot.json"
OUT_COMBINED = HERE / "hal_training_all.jsonl"
SEED = 20260517
random.seed(SEED)

MEL = timezone(timedelta(hours=10))  # AEST; we'll skip DST nuance for sample times

# ---------------------------------------------------------------------------
# Snapshot loader — used so attribute keys/options match production exactly
# ---------------------------------------------------------------------------
def load_snapshot():
    with open(SNAPSHOT_PATH) as f:
        states = json.load(f)
    return {s["entity_id"]: s for s in states}

SNAP = load_snapshot()

def snap_friendly(eid, fallback=None):
    s = SNAP.get(eid)
    if s:
        return s["attributes"].get("friendly_name", fallback or eid)
    return fallback or eid

def snap_unit(eid, fallback=""):
    s = SNAP.get(eid)
    if s:
        return s["attributes"].get("unit_of_measurement", fallback)
    return fallback

def snap_options(eid):
    s = SNAP.get(eid)
    if s:
        return s["attributes"].get("options", [])
    return []

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def tc_id():
    return "tc_" + uuid.uuid4().hex[:10]

def make_system_prompt(entity_lines, persona_extra=""):
    """Build the system prompt with HAL persona and a focused entity list."""
    persona = (
        "You are HAL, the conversational AI for Charm's smart home in Clyde North, "
        "Victoria, Australia. Time zone Australia/Melbourne. The home runs on Amber "
        "Electric wholesale pricing (cents/kWh) and a FoxESS battery with solar.\n"
        "Persona: calm, measured, slightly formal — like HAL 9000. Address the user as "
        "\"Charm\". Never open with \"Great!\", \"Sure!\", \"Of course!\", \"Certainly!\" "
        "or \"Happy to help\". Never apologise. Status replies are 1-4 sentences; "
        "analysis may be longer. Prices in cents/kWh (state is $/kWh, multiply by 100). "
        "If an entity is unavailable: \"That data is unavailable right now, Charm.\" "
        "If the Tesla is unavailable: \"The car is sleeping, Charm. Open the Tesla app "
        "to wake it.\"\n"
        "You have three tools: get_entity_state, call_service, web_search.\n"
    )
    if persona_extra:
        persona += persona_extra.rstrip() + "\n"
    entities_block = "Available entities:\n" + "\n".join(f"- {line}" for line in entity_lines)
    return persona + "\n" + entities_block

def entity_line(eid, value, unit=None, friendly=None):
    """Format one entity line for the system prompt."""
    friendly = friendly or snap_friendly(eid, eid)
    unit = unit if unit is not None else snap_unit(eid, "")
    unit_str = f" {unit}" if unit else ""
    return f"{eid} ({friendly}) = {value}{unit_str}"

def make_tool_call(tcid, name, args):
    return {
        "id": tcid,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, separators=(",", ":")),
        },
    }

def make_tool_result(tcid, content):
    if not isinstance(content, str):
        content = json.dumps(content, separators=(",", ":"))
    return {"role": "tool", "tool_call_id": tcid, "content": content}

def assistant_tool_turn(calls):
    return {"role": "assistant", "content": None, "tool_calls": calls}

def assistant_text(content):
    return {"role": "assistant", "content": content}

def user_msg(content):
    return {"role": "user", "content": content}

def example(system_content, user_content, sequence):
    """
    sequence: list of dicts produced by assistant_tool_turn(...), make_tool_result(...),
              and finally assistant_text(...).
    """
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            *sequence,
        ]
    }

def single_call_example(system_lines, user_text, tool_name, tool_args, tool_result, final_text, persona_extra=""):
    """Convenience: one tool call + result + final response."""
    tcid = tc_id()
    seq = [
        assistant_tool_turn([make_tool_call(tcid, tool_name, tool_args)]),
        make_tool_result(tcid, tool_result),
        assistant_text(final_text),
    ]
    return example(make_system_prompt(system_lines, persona_extra), user_text, seq)

def multi_call_example(system_lines, user_text, calls_and_results, final_text, persona_extra=""):
    """
    calls_and_results: list of tuples (tool_name, args_dict, result_value)
    """
    seq = []
    # Each call becomes its own assistant tool turn followed by the tool result.
    for name, args, result in calls_and_results:
        tcid = tc_id()
        seq.append(assistant_tool_turn([make_tool_call(tcid, name, args)]))
        seq.append(make_tool_result(tcid, result))
    seq.append(assistant_text(final_text))
    return example(make_system_prompt(system_lines, persona_extra), user_text, seq)

# ---------------------------------------------------------------------------
# Time / season randomisers
# ---------------------------------------------------------------------------
def rand_time_of_day():
    """Return (hour, minute, period_label)."""
    h = random.randint(0, 23)
    m = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    if 5 <= h < 9:
        period = "early_morning"
    elif 9 <= h < 12:
        period = "morning"
    elif 12 <= h < 15:
        period = "midday"
    elif 15 <= h < 18:
        period = "afternoon"
    elif 18 <= h < 21:
        period = "evening"
    elif 21 <= h < 24:
        period = "late_evening"
    else:
        period = "night"
    return h, m, period

def rand_season():
    return random.choice(["summer", "autumn", "winter", "spring"])

def is_daylight(hour, season):
    if season == "summer":
        return 5 <= hour < 21
    if season == "winter":
        return 7 <= hour < 18
    return 6 <= hour < 19

def fmt_now(h, m):
    return f"{h:02d}:{m:02d}"

def iso_now(h, m, days_offset=0):
    base = datetime(2026, 5, 17, h, m, 0, tzinfo=MEL)
    base = base + timedelta(days=days_offset)
    return base.isoformat()

# ---------------------------------------------------------------------------
# State randomisers — return dicts of (entity_id -> value) plus context
# ---------------------------------------------------------------------------
def randomise_energy(time_of_day=None, season=None, force_spike=False):
    h, m, period = time_of_day or rand_time_of_day()
    season = season or rand_season()
    day = is_daylight(h, season)

    soc = random.randint(10, 100)
    battery_kwh_capacity = 42
    kwh_remaining = round(soc / 100 * battery_kwh_capacity, 2)

    # Solar
    if day:
        if season == "summer":
            solar_kw = round(random.uniform(1.5, 9.5), 2)
        elif season == "winter":
            solar_kw = round(random.uniform(0.2, 5.5), 2)
        else:
            solar_kw = round(random.uniform(0.5, 7.5), 2)
        # midday peak
        if 11 <= h <= 14:
            solar_kw = round(solar_kw * random.uniform(0.9, 1.1), 2)
    else:
        solar_kw = 0.0

    # Load
    if period in ("evening", "late_evening", "early_morning"):
        load_kw = round(random.uniform(0.7, 4.8), 2)
    elif period == "night":
        load_kw = round(random.uniform(0.3, 1.6), 2)
    else:
        load_kw = round(random.uniform(0.4, 3.5), 2)

    # Battery flow vs grid
    net = solar_kw - load_kw
    if net > 0.3 and soc < 100:
        battery_charge_kw = round(min(net * random.uniform(0.5, 1.0), 6.0), 2)
        battery_discharge_kw = 0.0
        feed_in_kw = round(max(0.0, net - battery_charge_kw), 2)
        grid_kw = 0.0
    elif net < -0.3 and soc > 15:
        battery_discharge_kw = round(min(-net * random.uniform(0.5, 1.0), 6.0), 2)
        battery_charge_kw = 0.0
        feed_in_kw = 0.0
        grid_kw = round(max(0.0, -net - battery_discharge_kw), 2)
    else:
        battery_charge_kw = 0.0
        battery_discharge_kw = 0.0
        feed_in_kw = round(max(0.0, net), 2) if net > 0 else 0.0
        grid_kw = round(max(0.0, -net), 2) if net < 0 else 0.0

    # Daily aggregates
    solar_today_kwh = round(random.uniform(0, 50), 1) if day else round(random.uniform(0, 55), 1)
    feed_in_today_kwh = round(random.uniform(0, 30), 1)
    grid_today_kwh = round(random.uniform(0, 18), 1)
    battery_charge_today = round(random.uniform(0, 35), 1)
    battery_discharge_today = round(random.uniform(0, 35), 1)

    # Amber pricing
    spike_active = force_spike or (random.random() < 0.05)
    spike_imminent = (not spike_active) and (random.random() < 0.05)
    if spike_active:
        buy_cents = random.randint(60, 180)
    elif period in ("evening", "early_morning") and random.random() < 0.4:
        buy_cents = random.randint(35, 70)
    elif random.random() < 0.05:
        buy_cents = random.randint(-5, 0)
    else:
        buy_cents = random.randint(10, 32)
    buy_price = round(buy_cents / 100, 4)
    feed_cents = max(0, buy_cents - random.randint(8, 14))
    if feed_cents > 25:
        feed_cents = random.randint(10, 22)
    feed_price = round(feed_cents / 100, 4)

    # Grid daily cost (cents)
    grid_cost_cents = round(grid_today_kwh * (buy_cents + random.uniform(-5, 5)), 0)

    # Reserve / forecast
    reserve_soc_needed = random.randint(15, 60)
    dynamic_max_grid_soc = random.randint(40, 95)

    # Work mode
    work_modes = ["Self Use", "Feed-in First", "Back-up", "Peak Shaving", "Force Charge", "Force Discharge"]
    work_mode = random.choices(work_modes, weights=[60, 5, 3, 7, 15, 10])[0]

    min_soc = random.choice([10, 12, 15, 18, 20])
    max_soc = random.choice([90, 95, 97, 100])

    # Optimiser decision
    decisions = ["self_use", "wait_solar", "force_charge", "force_discharge",
                 "hold", "wait_cheaper", "solar_supplement"]
    if spike_active:
        decision = random.choice(["force_discharge", "hold"])
    elif buy_cents < 5:
        decision = "force_charge"
    elif day and solar_kw > 3:
        decision = random.choice(["self_use", "solar_supplement", "wait_solar"])
    else:
        decision = random.choice(decisions)

    target_soc = random.choice([30, 40, 50, 60, 70, 80, 85, 90, 95, 100])
    confidence = round(random.uniform(0.55, 0.95), 2)
    providers = ["deepseek", "anthropic", "openai", "groq"]
    models = {"deepseek": "deepseek-v4-flash", "anthropic": "claude-sonnet-4-6",
              "openai": "gpt-5-turbo", "groq": "llama-3.3-70b"}
    provider = random.choice(providers)

    optimiser_enabled = random.random() < 0.9
    decision_modes = ["auto_execute", "notify_only", "disabled"]
    decision_mode = random.choices(decision_modes, weights=[80, 15, 5])[0]
    safety_max_buy = random.choice([80, 100, 120, 150])

    inverter_state = "On Grid" if random.random() < 0.97 else "Off Grid"
    inv_temp = round(random.uniform(28, 55), 1)
    batt_temp = round(random.uniform(18, 42), 1)

    return {
        "time": (h, m, period),
        "season": season,
        "is_day": day,
        # Battery
        "battery_soc": soc,
        "kwh_remaining": kwh_remaining,
        "battery_charge_kw": battery_charge_kw,
        "battery_discharge_kw": battery_discharge_kw,
        "battery_charge_today": battery_charge_today,
        "battery_discharge_today": battery_discharge_today,
        # Power flows
        "solar_kw": solar_kw,
        "load_kw": load_kw,
        "feed_in_kw": feed_in_kw,
        "grid_kw": grid_kw,
        # Daily
        "solar_today_kwh": solar_today_kwh,
        "feed_in_today_kwh": feed_in_today_kwh,
        "grid_today_kwh": grid_today_kwh,
        "grid_cost_cents": grid_cost_cents,
        # Settings
        "work_mode": work_mode,
        "min_soc": min_soc,
        "max_soc": max_soc,
        "inverter_state": inverter_state,
        "inv_temp": inv_temp,
        "batt_temp": batt_temp,
        # Amber
        "buy_cents": buy_cents,
        "buy_price": buy_price,
        "feed_cents": feed_cents,
        "feed_price": feed_price,
        "spike_active": spike_active,
        "spike_imminent": spike_imminent,
        # Optimiser
        "decision": decision,
        "target_soc": target_soc,
        "confidence": confidence,
        "provider": provider,
        "model": models[provider],
        "optimiser_enabled": optimiser_enabled,
        "decision_mode": decision_mode,
        "safety_max_buy": safety_max_buy,
        "reserve_soc_needed": reserve_soc_needed,
        "dynamic_max_grid_soc": dynamic_max_grid_soc,
    }

def randomise_amber_timeline(now_h, now_m, hours=12, spike_prob=0.1):
    """Build a 5-minute-interval timeline going forwards."""
    items = []
    base = datetime(2026, random.choice([3,4,5,6,7,8,9,10]), random.randint(1, 28), now_h, now_m, tzinfo=MEL)
    n = hours * 12  # 5-min slots
    for i in range(n):
        t = base + timedelta(minutes=5 * i)
        h = t.hour
        # baseline by tod
        if 17 <= h <= 21:
            base_c = random.randint(22, 50)
        elif 0 <= h <= 5:
            base_c = random.randint(8, 22)
        elif 10 <= h <= 15:
            base_c = random.randint(-2, 20)
        else:
            base_c = random.randint(14, 32)
        if random.random() < spike_prob:
            base_c = random.randint(70, 200)
        items.append({"start_time": t.isoformat(), "price_per_kwh": round(base_c / 100, 4)})
    return items

def randomise_hvac(season=None, energy=None):
    season = season or (energy["season"] if energy else rand_season())
    occupied = random.random() < 0.7
    automation = random.random() < 0.85

    # Outdoor influences indoor
    if season == "winter":
        outdoor = round(random.uniform(4, 18), 1)
        indoor_base = random.uniform(15, 22)
    elif season == "summer":
        outdoor = round(random.uniform(22, 40), 1)
        indoor_base = random.uniform(22, 30)
    else:
        outdoor = round(random.uniform(12, 26), 1)
        indoor_base = random.uniform(17, 25)

    downstairs = round(indoor_base + random.uniform(-1.5, 1.5), 1)
    upstairs = round(indoor_base + random.uniform(-0.5, 2.5), 1)  # upstairs often warmer
    coldest_zone = "Downstairs" if downstairs < upstairs else "Upstairs"
    coldest_temp = min(downstairs, upstairs)
    hottest_zone = "Upstairs" if upstairs > downstairs else "Downstairs"
    hottest_temp = max(downstairs, upstairs)

    # Mode selection — zone climates support off/fan_only/heat_cool
    running = random.random() < 0.4
    main_modes = ["off", "cool", "heat", "heat_cool", "fan_only", "dry"]
    if running:
        if season == "winter":
            main_mode = random.choices(["heat", "heat_cool"], weights=[60, 40])[0]
        elif season == "summer":
            main_mode = random.choices(["cool", "heat_cool"], weights=[60, 40])[0]
        else:
            main_mode = random.choice(["heat_cool", "fan_only", "cool", "heat"])
    else:
        main_mode = "off"

    zone_mode = "off" if main_mode == "off" else ("heat_cool" if main_mode in ("heat", "cool", "heat_cool") else main_mode)

    if main_mode == "heat" or (zone_mode == "heat_cool" and season == "winter"):
        setpoint = round(random.uniform(19, 23), 1)
        actually_heating = setpoint > downstairs + 0.5
        actually_cooling = False
    elif main_mode == "cool" or (zone_mode == "heat_cool" and season == "summer"):
        setpoint = round(random.uniform(21, 25), 1)
        actually_heating = False
        actually_cooling = setpoint < downstairs - 0.5
    else:
        setpoint = round(random.uniform(20, 23), 1)
        actually_heating = False
        actually_cooling = False

    supply_temp = downstairs + (10 if actually_heating else (-8 if actually_cooling else 0)) + random.uniform(-2, 2)
    supply_temp = round(supply_temp, 1)

    heat_runtime_h = round(random.uniform(0, 6), 2)
    cool_runtime_h = round(random.uniform(0, 8), 2)
    if season == "winter":
        cool_runtime_h = round(random.uniform(0, 0.3), 2)
    elif season == "summer":
        heat_runtime_h = round(random.uniform(0, 0.3), 2)

    schedule_modes = ["auto", "sleep", "away", "manual"]
    schedule_mode = random.choices(schedule_modes, weights=[60, 20, 10, 10])[0]

    low_soc_pause = (energy and energy["battery_soc"] < 25) and random.random() < 0.4
    amber_pause = (energy and (energy["spike_active"] or energy["buy_cents"] > 60)) and random.random() < 0.5

    heat_trigger = random.choice([17.0, 18.0, 18.5, 19.0])
    cool_trigger = random.choice([24.0, 25.0, 26.0, 27.0])

    return {
        "season": season,
        "outdoor_temp": outdoor,
        "downstairs_temp": downstairs,
        "upstairs_temp": upstairs,
        "coldest_zone": coldest_zone,
        "coldest_temp": coldest_temp,
        "hottest_zone": hottest_zone,
        "hottest_temp": hottest_temp,
        "main_mode": main_mode,
        "zone_mode": zone_mode,
        "setpoint": setpoint,
        "running": running,
        "actually_heating": actually_heating,
        "actually_cooling": actually_cooling,
        "supply_temp": supply_temp,
        "heat_runtime_h": heat_runtime_h,
        "cool_runtime_h": cool_runtime_h,
        "occupied": occupied,
        "automation": automation,
        "schedule_mode": schedule_mode,
        "low_soc_pause": low_soc_pause,
        "amber_pause": amber_pause,
        "heat_trigger": heat_trigger,
        "cool_trigger": cool_trigger,
    }

def randomise_weather(season=None):
    season = season or rand_season()
    if season == "summer":
        temp = round(random.uniform(18, 38), 1)
        max_temp = round(temp + random.uniform(2, 8), 1)
        humidity = random.randint(25, 80)
        wind = random.randint(2, 35)
        rain_chance = random.randint(0, 60)
        rain_mm = random.choice([0, 0, 0, 0.2, 1, 3])
        uv = random.choice(["Very High", "Extreme", "High"])
        short = random.choice(["Sunny.", "Partly cloudy.", "Possible afternoon storm.", "Hot and dry.", "Clear."])
    elif season == "winter":
        temp = round(random.uniform(4, 16), 1)
        max_temp = round(temp + random.uniform(1, 6), 1)
        humidity = random.randint(55, 95)
        wind = random.randint(5, 45)
        rain_chance = random.randint(30, 95)
        rain_mm = random.choice([0, 0.5, 2, 5, 12, 20])
        uv = random.choice(["Low", "Moderate"])
        short = random.choice(["Rain.", "Showers.", "Cloudy.", "Cold and wet.", "Partly cloudy."])
    elif season == "spring":
        temp = round(random.uniform(10, 24), 1)
        max_temp = round(temp + random.uniform(1, 7), 1)
        humidity = random.randint(40, 85)
        wind = random.randint(5, 40)
        rain_chance = random.randint(10, 70)
        rain_mm = random.choice([0, 0, 1, 4, 8])
        uv = random.choice(["Moderate", "High", "Very High"])
        short = random.choice(["Showers.", "Partly cloudy.", "Sunny breaks.", "Mostly sunny."])
    else:  # autumn
        temp = round(random.uniform(8, 22), 1)
        max_temp = round(temp + random.uniform(1, 6), 1)
        humidity = random.randint(45, 90)
        wind = random.randint(4, 38)
        rain_chance = random.randint(20, 80)
        rain_mm = random.choice([0, 0, 0.5, 3, 8])
        uv = random.choice(["Low", "Moderate", "High"])
        short = random.choice(["Cloudy.", "Showers.", "Partly cloudy.", "Cool with cloud."])

    feels_like = round(temp + (random.uniform(-3, 2) if wind > 20 else random.uniform(-1, 1)), 1)
    wind_dir = random.choice(["CALM", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])
    sunrise = {"summer": "06:08", "autumn": "07:05", "winter": "07:30", "spring": "06:42"}[season]
    sunset = {"summer": "20:45", "autumn": "18:32", "winter": "17:08", "spring": "19:15"}[season]
    rain_forecast_24h = round(rain_mm + random.uniform(0, 8), 1) if rain_chance > 50 else round(rain_mm * 0.5, 1)
    rain_amount_range = f"0 to {round(rain_mm + 5, 0):.0f} mm" if rain_chance > 30 else "0 mm"
    extended = short + " " + random.choice([
        "Light to moderate winds.", "Winds easing in the evening.",
        "Becoming gusty later.", "Cool change moving through."
    ])

    return {
        "season": season,
        "outdoor_temp": temp,
        "feels_like": feels_like,
        "max_temp": max_temp,
        "humidity": humidity,
        "wind_kmh": wind,
        "wind_dir": wind_dir,
        "rain_chance": rain_chance,
        "rain_mm_today": rain_mm,
        "rain_range": rain_amount_range,
        "rain_forecast_24h": rain_forecast_24h,
        "uv_category": uv,
        "uv_forecast": f"{uv} UV from 09:30 to 15:00",
        "sunrise": sunrise,
        "sunset": sunset,
        "short_forecast": short,
        "extended_forecast": extended,
    }

def randomise_irrigation(season=None, weather=None):
    season = season or (weather["season"] if weather else rand_season())
    season_factors = {"summer": 1.0, "autumn": 0.45, "winter": 0.2, "spring": 0.7}
    factor = round(season_factors[season] + random.uniform(-0.1, 0.1), 2)
    factor = max(0.0, min(1.0, factor))

    recent_rain = weather["rain_mm_today"] if weather else random.choice([0, 0, 2, 8, 15])
    rain_forecast = weather["rain_forecast_24h"] if weather else random.choice([0, 0, 4, 12])

    if recent_rain >= 5:
        skip = "Skipped: 9am rain >= 5mm"
        zones_due = []
        planned = 0
    elif rain_forecast >= 8:
        skip = f"Skipped: {rain_forecast}mm forecast next 24h"
        zones_due = []
        planned = 0
    elif factor < 0.25:
        skip = f"Skipped: season factor {factor} below threshold"
        zones_due = []
        planned = 0
    else:
        skip = "Ready"
        all_zones = list(range(1, 6))
        zones_due = random.sample(all_zones, k=random.randint(1, 5))
        planned = sum(random.randint(4, 18) for _ in zones_due)
        planned = int(round(planned * factor))

    master_enabled = random.random() < 0.95

    return {
        "season_factor": factor,
        "skip_reason": skip,
        "zones_due": sorted(zones_due),
        "planned_minutes": planned,
        "rain_mm_today": recent_rain,
        "rain_forecast_24h": rain_forecast,
        "master_enabled": master_enabled,
    }

def randomise_tesla():
    """20% sleeping/unavailable, 60% home, 20% away."""
    state = random.choices(["home", "not_home", "sleeping"], weights=[60, 20, 20])[0]
    if state == "sleeping":
        return {
            "available": False,
            "location": "unavailable",
            "lock": "unavailable",
            "cable_lock": "unavailable",
            "port_door": "unavailable",
            "frunk": "unavailable",
            "trunk": "unavailable",
            "driver_door": "unavailable",
            "climate_on": "unavailable",
            "battery_level": None,
            "charge_limit": None,
            "charging_state": "unavailable",
            "cabin_temp": None,
            "set_temp": None,
            "seat_left": "unavailable",
            "seat_right": "unavailable",
        }
    return {
        "available": True,
        "location": state,
        "lock": random.choices(["locked", "unlocked"], weights=[90, 10])[0],
        "cable_lock": random.choice(["locked", "unlocked"]),
        "port_door": random.choices(["closed", "open"], weights=[70, 30])[0],
        "frunk": random.choices(["closed", "open"], weights=[97, 3])[0],
        "trunk": random.choices(["closed", "open"], weights=[95, 5])[0],
        "driver_door": random.choices(["closed", "open"], weights=[97, 3])[0],
        "climate_on": random.choices(["on", "off"], weights=[15, 85])[0],
        "battery_level": random.randint(20, 100),
        "charge_limit": random.choice([70, 80, 85, 90, 95, 100]),
        "charging_state": random.choices(["Disconnected", "Stopped", "Charging", "Complete"],
                                          weights=[40, 30, 20, 10])[0],
        "cabin_temp": round(random.uniform(8, 38), 1),
        "set_temp": round(random.uniform(18, 24), 1),
        "seat_left": random.choices(["off", "low", "medium", "high"], weights=[80, 10, 7, 3])[0],
        "seat_right": random.choices(["off", "low", "medium", "high"], weights=[85, 8, 5, 2])[0],
    }

def randomise_security(occupied=None):
    occupied = random.random() < 0.65 if occupied is None else occupied
    if occupied:
        states = ["disarmed", "armed_home", "armed_night"]
        weights = [70, 20, 10]
    else:
        states = ["armed_away", "disarmed"]
        weights = [80, 20]
    alarm = random.choices(states, weights=weights)[0]
    door_lock = random.choices(["locked", "unlocked", "unavailable"], weights=[82, 15, 3])[0]
    charm_home = "home" if occupied and random.random() < 0.95 else "not_home"
    return {
        "alarm": alarm,
        "front_door_lock": door_lock,
        "charm_at_home": "on" if charm_home == "home" else "off",
        "house_occupied": "on" if occupied else "off",
        "person_charm": charm_home,
    }

def randomise_lighting(period):
    if period in ("evening", "late_evening", "night", "early_morning"):
        on_prob = 0.55
    else:
        on_prob = 0.1
    return {
        "kitchen_main": "on" if random.random() < on_prob else "off",
        "kitchen_2": "on" if random.random() < on_prob else "off",
        "living": "on" if random.random() < on_prob else "off",
        "left_pillar": "on" if random.random() < (0.4 if period == "night" else 0.05) else "off",
        "portico": "on" if random.random() < (0.6 if period in ("night", "late_evening") else 0.05) else "off",
        "left_floodlight": "off",
        "front_left_floodlight": "off",
        "garage_floodlight": "off",
    }

def randomise_appliances():
    return {
        "vacuum": random.choices(["docked", "cleaning", "returning", "paused"],
                                 weights=[80, 10, 5, 5])[0],
        "garage_1": random.choices(["off", "on"], weights=[97, 3])[0],
        "garage_2": random.choices(["off", "on"], weights=[97, 3])[0],
        "shield_state": random.choice(["idle", "playing", "paused", "off"]),
        "home_theater_state": random.choice(["off", "on"]),
        "tv_state": random.choice(["off", "on", "playing"]),
    }

# ---------------------------------------------------------------------------
# Persona-safe phrase banks (avoid forbidden openers)
# ---------------------------------------------------------------------------
SAFE_OPENERS_STATUS = [
    "", "Reading now. ", "Just checked. ",
    "Current value: ", "Right now, ", "At the moment, ",
]

SAFE_OPENERS_CONTROL = [
    "Done. ", "Applied. ", "Executed. ",
    "Settings updated. ", "Change committed. ",
]

SAFE_OPENERS_ANALYSIS = [
    "", "Looking at the numbers — ", "On balance, ",
    "Here is the picture: ", "Reviewing the state: ",
]

def opener(kind="status"):
    if kind == "control":
        return random.choice(SAFE_OPENERS_CONTROL)
    if kind == "analysis":
        return random.choice(SAFE_OPENERS_ANALYSIS)
    return random.choice(SAFE_OPENERS_STATUS)

# ---------------------------------------------------------------------------
# Final persona cleanup — strip duplicate "Charm" mentions in assistant replies
# ---------------------------------------------------------------------------
import re as _re

_BANNED_OPENERS = ["Great!", "Sure!", "Of course!", "Certainly!", "Happy to help"]

def _clean_reply(text):
    if not text or "Charm" not in text:
        return text
    # If "Charm" appears 2+ times, remove the first occurrence
    # (keep only the final, natural-sounding instance).
    count = text.count("Charm")
    while count >= 2:
        # Try common duplicate patterns first
        text = _re.sub(r",\s*Charm,\s*", ", ", text, count=1)
        if text.count("Charm") == count:
            text = _re.sub(r",\s*Charm\.\s+", ". ", text, count=1)
        if text.count("Charm") == count:
            text = _re.sub(r"\s+Charm\.\s+", ". ", text, count=1)
        if text.count("Charm") == count:
            text = _re.sub(r"\s+Charm,\s+", ", ", text, count=1)
        if text.count("Charm") == count:
            break  # safety
        count = text.count("Charm")
    # Collapse "Yes, Charm" or "No, Charm" with leading commas merged from removals
    text = _re.sub(r",\s+,\s+", ", ", text)
    text = _re.sub(r"\.\s*\.\s*", ". ", text)
    return text

def _scrub_example(ex):
    for m in ex["messages"]:
        if m["role"] == "assistant" and m.get("content"):
            m["content"] = _clean_reply(m["content"])
    return ex

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def write_jsonl(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            ex = _scrub_example(ex)
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")

# ---------------------------------------------------------------------------
# BATCH 1 — Energy status (120, single tool call)
# ---------------------------------------------------------------------------
ENERGY_STATUS_QUERIES = [
    # (user_query_options, entity_id, attribute, value_key, response_fn)
    (["what's the battery at", "battery soc", "how full is the battery", "current state of charge",
      "what's the battery percentage", "battery level please"],
     "sensor.skynet_sun_harvester_battery_soc_1", None, "battery_soc",
     lambda e: f"{opener()}Battery is at {e['battery_soc']}%, Charm."),
    (["how many kwh are left in the battery", "battery kwh remaining",
      "what's the remaining stored energy", "energy left in the battery"],
     "sensor.skynet_sun_harvester_bms_kwh_remaining_1", None, "kwh_remaining",
     lambda e: f"{opener()}{e['kwh_remaining']} kWh remaining in the battery, Charm."),
    (["how much solar is the system pulling now", "solar power right now",
      "current solar generation", "what's the array doing", "pv power"],
     "sensor.foxess_modbus_skynet_sun_harvester_pv_power_skynet_sun_harvester",
     None, "solar_kw",
     lambda e: (f"{opener()}Solar is at {e['solar_kw']} kW right now, Charm." if e['solar_kw'] > 0
                else f"{opener()}Solar is at 0 kW, Charm — the array is off-sun.")),
    (["how much solar today", "today's solar generation", "solar yield today",
      "total solar so far today", "solar today"],
     "sensor.skynet_sun_harvester_battery_charge_today", None, "solar_today_kwh",  # use one daily total entity
     lambda e: f"{opener()}{e['solar_today_kwh']} kWh from solar so far today, Charm."),
    (["what's the house drawing", "load power", "current house load",
      "how much power is the house using"],
     "sensor.skynet_sun_harvester_load_power", None, "load_kw",
     lambda e: f"{opener()}House load is {e['load_kw']} kW, Charm."),
    (["how much is the battery discharging", "battery discharge now",
      "discharge rate right now"],
     "sensor.skynet_sun_harvester_battery_discharge", None, "battery_discharge_kw",
     lambda e: (f"{opener()}Battery is discharging at {e['battery_discharge_kw']} kW, Charm."
                if e['battery_discharge_kw'] > 0
                else f"{opener()}Battery isn't discharging right now, Charm.")),
    (["how much is the battery charging", "battery charge rate now",
      "is the battery charging"],
     "sensor.skynet_sun_harvester_battery_charge", None, "battery_charge_kw",
     lambda e: (f"{opener()}Battery is charging at {e['battery_charge_kw']} kW, Charm."
                if e['battery_charge_kw'] > 0
                else f"{opener()}Battery isn't charging at the moment, Charm.")),
    (["how much have i charged the battery today", "battery charged today",
      "kwh into the battery today"],
     "sensor.skynet_sun_harvester_battery_charge_today", None, "battery_charge_today",
     lambda e: f"{opener()}{e['battery_charge_today']} kWh has gone into the battery today, Charm."),
    (["how much has the battery discharged today", "battery discharge today total",
      "kwh out of the battery today"],
     "sensor.skynet_sun_harvester_battery_discharge_today", None, "battery_discharge_today",
     lambda e: f"{opener()}{e['battery_discharge_today']} kWh has come out of the battery today, Charm."),
    (["how much am i feeding in right now", "feed-in now", "exporting power right now",
      "current feed in rate"],
     "sensor.skynet_sun_harvester_feed_in", None, "feed_in_kw",
     lambda e: (f"{opener()}Feeding in at {e['feed_in_kw']} kW, Charm."
                if e['feed_in_kw'] > 0
                else f"{opener()}Nothing flowing to the grid at the moment, Charm.")),
    (["how much did i feed in today", "feed-in today", "total exported today",
      "kwh exported today"],
     "sensor.skynet_sun_harvester_feed_in_energy_today", None, "feed_in_today_kwh",
     lambda e: f"{opener()}{e['feed_in_today_kwh']} kWh has been exported to the grid today, Charm."),
    (["how much am i pulling from the grid right now", "grid draw now",
      "current grid consumption", "am i importing"],
     "sensor.skynet_sun_harvester_grid_consumption", None, "grid_kw",
     lambda e: (f"{opener()}Pulling {e['grid_kw']} kW from the grid right now, Charm."
                if e['grid_kw'] > 0
                else f"{opener()}No grid draw at the moment, Charm.")),
    (["how much have i pulled from the grid today", "grid import today",
      "total grid use today", "kwh from grid today"],
     "sensor.skynet_sun_harvester_grid_consumption_energy_today", None, "grid_today_kwh",
     lambda e: f"{opener()}{e['grid_today_kwh']} kWh imported from the grid today, Charm."),
    (["what's the grid cost today", "today's grid spend", "how much have i spent on power today",
      "running grid bill today"],
     "sensor.daily_grid_import_cost_raw_cents", None, "grid_cost_cents",
     lambda e: f"{opener()}{int(e['grid_cost_cents'])} cents spent on grid imports today, Charm — that's ${e['grid_cost_cents']/100:.2f}."),
    (["is the inverter healthy", "inverter status", "what's the inverter doing",
      "grid status of the inverter"],
     "sensor.skynet_sun_harvester_inverter_state", None, "inverter_state",
     lambda e: f"{opener()}Inverter is {e['inverter_state']}, Charm."),
    (["what's the inverter temp", "inverter temperature",
      "how hot is the inverter"],
     "sensor.skynet_sun_harvester_invtemp", None, "inv_temp",
     lambda e: f"{opener()}Inverter is at {e['inv_temp']} °C, Charm."),
    (["what's the battery temperature", "battery temp",
      "how hot is the battery"],
     "sensor.skynet_sun_harvester_battery_temp_1", None, "batt_temp",
     lambda e: f"{opener()}Battery is at {e['batt_temp']} °C, Charm."),
    (["what reserve does the battery need before sunrise",
      "reserve soc needed", "overnight reserve target"],
     "sensor.reserve_soc_needed_to_sunrise", None, "reserve_soc_needed",
     lambda e: f"{opener()}Reserve target is {e['reserve_soc_needed']}% before sunrise, Charm."),
    (["what's the dynamic max grid soc", "dynamic charge cap",
      "amber dynamic max soc"],
     "sensor.amber_dynamic_max_grid_soc", None, "dynamic_max_grid_soc",
     lambda e: f"{opener()}Dynamic cap is at {e['dynamic_max_grid_soc']}%, Charm."),
    (["what's the work mode", "battery work mode", "current inverter work mode"],
     "select.skynet_sun_harvester_work_mode", None, "work_mode",
     lambda e: f"{opener()}Work mode is set to {e['work_mode']}, Charm."),
    (["what's the min soc set to", "battery floor", "min soc setting"],
     "number.skynet_sun_harvester_min_soc", None, "min_soc",
     lambda e: f"{opener()}Minimum SoC is set to {e['min_soc']}%, Charm."),
    (["what's the max soc set to", "battery ceiling", "max soc setting"],
     "number.skynet_sun_harvester_max_soc", None, "max_soc",
     lambda e: f"{opener()}Maximum SoC is set to {e['max_soc']}%, Charm."),
]

def gen_batch_01_energy_status(n=120):
    out = []
    for _ in range(n):
        queries, eid, _attr, vkey, resp_fn = random.choice(ENERGY_STATUS_QUERIES)
        energy = randomise_energy()
        value = energy[vkey]
        user_q = random.choice(queries)
        unit = snap_unit(eid, "")
        sys_lines = [entity_line(eid, value, unit)]
        result_str = str(value) if not isinstance(value, str) else value
        ex = single_call_example(
            sys_lines, user_q,
            "get_entity_state", {"entity_id": eid},
            result_str, resp_fn(energy),
        )
        out.append(ex)
    return out

# ---------------------------------------------------------------------------
# BATCH 2 — Energy reasoning (80, multi-tool, longer response)
# ---------------------------------------------------------------------------
def _summarise_battery_outlook(e):
    # Decide if SoC + load forecast will last the night
    avg_overnight_load = 0.8  # kW avg
    hours_to_sunrise = random.randint(5, 14)
    energy_need = avg_overnight_load * hours_to_sunrise
    usable_kwh = e["kwh_remaining"] - 0.15 * 42  # leave 15% floor
    return hours_to_sunrise, energy_need, usable_kwh

ENERGY_REASONING_QUERIES = [
    "give me a full energy rundown",
    "how's the energy looking today",
    "is the battery going to last the night",
    "how efficient was today",
    "how much did I save by not buying from grid",
    "what's the self-sufficiency like today",
    "explain what the battery has been doing",
    "should I be worried about overnight reserve",
    "tell me about today's energy picture",
    "how are we tracking on solar today",
    "battery health and usage summary",
    "is the system pulling its weight today",
]

def gen_batch_02_energy_reasoning(n=80):
    out = []
    for _ in range(n):
        e = randomise_energy()
        q = random.choice(ENERGY_REASONING_QUERIES)

        if "night" in q or "overnight" in q:
            hrs, need, usable = _summarise_battery_outlook(e)
            calls = [
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_bms_kwh_remaining_1"},
                 str(e["kwh_remaining"])),
                ("get_entity_state", {"entity_id": "sensor.reserve_soc_needed_to_sunrise"},
                 str(e["reserve_soc_needed"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_load_power"},
                 str(e["load_kw"])),
            ]
            verdict = ("comfortably" if e["battery_soc"] > e["reserve_soc_needed"] + 15
                       else "tightly" if e["battery_soc"] >= e["reserve_soc_needed"]
                       else "short")
            if verdict == "short":
                margin = e["reserve_soc_needed"] - e["battery_soc"]
                reply = (f"{opener('analysis')}Battery at {e['battery_soc']}% ({e['kwh_remaining']} kWh) "
                         f"is {margin} points below the {e['reserve_soc_needed']}% reserve target, Charm. "
                         f"Current load is {e['load_kw']} kW; the optimiser is likely to pull cheap "
                         f"grid energy before sunrise to cover the shortfall.")
            elif verdict == "tightly":
                reply = (f"{opener('analysis')}Battery at {e['battery_soc']}% versus a {e['reserve_soc_needed']}% "
                         f"reserve target — tight but workable, Charm. With load at {e['load_kw']} kW it should "
                         f"hold, though the optimiser may top up if prices dip.")
            else:
                margin = e["battery_soc"] - e["reserve_soc_needed"]
                reply = (f"{opener('analysis')}Battery at {e['battery_soc']}% sits {margin} points above the "
                         f"{e['reserve_soc_needed']}% reserve target, Charm. {e['kwh_remaining']} kWh on hand "
                         f"comfortably covers the night at the current {e['load_kw']} kW load.")
        elif "efficient" in q or "self-sufficien" in q or "save" in q or "save by" in q:
            calls = [
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_charge_today"},
                 str(e["battery_charge_today"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_discharge_today"},
                 str(e["battery_discharge_today"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_feed_in_energy_today"},
                 str(e["feed_in_today_kwh"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_grid_consumption_energy_today"},
                 str(e["grid_today_kwh"])),
                ("get_entity_state", {"entity_id": "sensor.daily_grid_import_cost_raw_cents"},
                 str(e["grid_cost_cents"])),
            ]
            offset_pct = max(0, 100 - int(e["grid_today_kwh"] / max(0.1, e["grid_today_kwh"] + e["battery_discharge_today"]) * 100))
            reply = (f"{opener('analysis')}Solar and battery covered most of the day, Charm. "
                     f"{e['battery_discharge_today']} kWh came out of the battery, "
                     f"{e['feed_in_today_kwh']} kWh was exported, and only {e['grid_today_kwh']} kWh "
                     f"was imported — costing {int(e['grid_cost_cents'])} cents (${e['grid_cost_cents']/100:.2f}). "
                     f"That's roughly {offset_pct}% self-sufficiency for the day.")
        elif "battery has been doing" in q or "tracking on solar" in q or "battery health" in q:
            calls = [
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_charge_today"},
                 str(e["battery_charge_today"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_discharge_today"},
                 str(e["battery_discharge_today"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_temp_1"},
                 str(e["batt_temp"])),
            ]
            reply = (f"{opener('analysis')}Battery is at {e['battery_soc']}% and {e['batt_temp']} °C, Charm. "
                     f"Today: {e['battery_charge_today']} kWh in, {e['battery_discharge_today']} kWh out. "
                     f"Temperature is comfortably within range, no thermal concern.")
        else:
            # full rundown
            calls = [
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_load_power"},
                 str(e["load_kw"])),
                ("get_entity_state", {"entity_id": "sensor.foxess_modbus_skynet_sun_harvester_pv_power_skynet_sun_harvester"},
                 str(e["solar_kw"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_feed_in"},
                 str(e["feed_in_kw"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_grid_consumption"},
                 str(e["grid_kw"])),
                ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                 str(e["buy_price"])),
            ]
            flow = []
            if e["solar_kw"] > 0:
                flow.append(f"solar producing {e['solar_kw']} kW")
            else:
                flow.append("solar offline")
            flow.append(f"load {e['load_kw']} kW")
            if e["battery_charge_kw"] > 0:
                flow.append(f"battery charging at {e['battery_charge_kw']} kW")
            elif e["battery_discharge_kw"] > 0:
                flow.append(f"battery discharging at {e['battery_discharge_kw']} kW")
            if e["feed_in_kw"] > 0:
                flow.append(f"feeding in {e['feed_in_kw']} kW")
            if e["grid_kw"] > 0:
                flow.append(f"importing {e['grid_kw']} kW")
            spike_note = " A price spike is active — heavy loads should wait." if e["spike_active"] else ""
            reply = (f"{opener('analysis')}Battery at {e['battery_soc']}%, "
                     + ", ".join(flow) + f". Buy price {e['buy_cents']} c/kWh.{spike_note} "
                     f"Charm, this is a typical {'day' if e['is_day'] else 'night'} pattern.")

        sys_lines = [entity_line(name, val) for (_, args, val) in
                     [(c[0], c[1], c[2]) for c in calls]
                     for name in [args["entity_id"]]]
        ex = multi_call_example(sys_lines, q, calls, reply, persona_extra="")
        out.append(ex)
    return out

# ---------------------------------------------------------------------------
# BATCH 3 — Energy control (40, write operations)
# ---------------------------------------------------------------------------
ENERGY_CONTROL_TEMPLATES = [
    # (query_options, builder_fn)
    # builder_fn(energy) -> (sys_lines, calls_and_results, reply)
]

def _ec_set_min_soc():
    target = random.choice([10, 12, 15, 18, 20, 25, 30])
    e = randomise_energy()
    sys_lines = [entity_line("number.skynet_sun_harvester_min_soc", e["min_soc"], "%")]
    user_q = random.choice([
        f"set the min soc to {target}%", f"battery floor to {target} percent",
        f"change min soc to {target}", f"min soc {target}%",
    ])
    calls = [("call_service",
              {"domain": "number", "service": "set_value",
               "entity_id": "number.skynet_sun_harvester_min_soc",
               "data": {"value": target}},
              "ok")]
    reply = f"{opener('control')}Minimum SoC set to {target}%, Charm."
    return sys_lines, user_q, calls, reply

def _ec_set_max_soc():
    target = random.choice([85, 90, 95, 97, 100])
    e = randomise_energy()
    sys_lines = [entity_line("number.skynet_sun_harvester_max_soc", e["max_soc"], "%")]
    user_q = random.choice([
        f"set the max soc to {target}%", f"battery ceiling to {target}",
        f"max charge limit {target}", f"cap charging at {target}%",
    ])
    calls = [("call_service",
              {"domain": "number", "service": "set_value",
               "entity_id": "number.skynet_sun_harvester_max_soc",
               "data": {"value": target}},
              "ok")]
    reply = f"{opener('control')}Maximum SoC set to {target}%, Charm."
    return sys_lines, user_q, calls, reply

def _ec_set_work_mode():
    options = ["Self Use", "Feed-in First", "Back-up", "Peak Shaving", "Force Charge", "Force Discharge"]
    target = random.choice(options)
    e = randomise_energy()
    sys_lines = [entity_line("select.skynet_sun_harvester_work_mode", e["work_mode"])]
    user_q = random.choice([
        f"change the work mode to {target.lower()}", f"set work mode {target}",
        f"switch to {target.lower()} mode", f"put the inverter into {target.lower()}",
    ])
    calls = [("call_service",
              {"domain": "select", "service": "select_option",
               "entity_id": "select.skynet_sun_harvester_work_mode",
               "data": {"option": target}},
              "ok")]
    reply = f"{opener('control')}Work mode switched to {target}, Charm."
    return sys_lines, user_q, calls, reply

def _ec_safety_max_buy():
    target = random.choice([80, 100, 120, 150, 200])
    e = randomise_energy()
    sys_lines = [entity_line("input_number.safety_absolute_max_buy_price", e["safety_max_buy"], "c/kWh")]
    user_q = random.choice([
        f"set the safety max buy price to {target} cents",
        f"raise the absolute buy cap to {target}",
        f"max buy price {target} c/kWh",
    ])
    calls = [("call_service",
              {"domain": "input_number", "service": "set_value",
               "entity_id": "input_number.safety_absolute_max_buy_price",
               "data": {"value": target}},
              "ok")]
    reply = f"{opener('control')}Safety absolute max buy price now {target} c/kWh, Charm."
    return sys_lines, user_q, calls, reply

def gen_batch_03_energy_control(n=40):
    builders = [_ec_set_min_soc, _ec_set_max_soc, _ec_set_work_mode, _ec_safety_max_buy]
    out = []
    for _ in range(n):
        sys_lines, user_q, calls, reply = random.choice(builders)()
        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 4 — Amber price queries (100, single/double tool call)
# ---------------------------------------------------------------------------
AMBER_PRICE_QUERIES = [
    "what's the price now",
    "current amber price",
    "what's the spot price right now",
    "buy price",
    "amber buy price now",
    "what am i paying for power right now",
]
AMBER_FEED_QUERIES = [
    "what's the feed-in price",
    "feed in price right now",
    "what am i getting paid for exports",
    "export rate now",
]
AMBER_SPIKE_QUERIES = [
    "is there a spike active",
    "any spike on right now",
    "is amber spiking",
    "spike status",
]

def _next_cheap_window(timeline, min_slots=3, max_cents=15):
    cheap_runs = []
    run = []
    for item in timeline:
        c = round(item["price_per_kwh"] * 100)
        if c <= max_cents:
            run.append(item)
        else:
            if len(run) >= min_slots:
                cheap_runs.append(run)
            run = []
    if len(run) >= min_slots:
        cheap_runs.append(run)
    return cheap_runs[0] if cheap_runs else None

def _next_spike(timeline, threshold=45):
    for item in timeline:
        c = round(item["price_per_kwh"] * 100)
        if c >= threshold:
            return item, c
    return None, None

def gen_batch_04_amber_price(n=100):
    out = []
    for _ in range(n):
        e = randomise_energy()
        q_type = random.choices(
            ["buy_now", "feed_now", "spike_now", "timeline_cheap", "timeline_spike",
             "tonights_prices", "is_now_good", "spot"],
            weights=[20, 15, 15, 15, 10, 10, 10, 5])[0]

        if q_type == "buy_now":
            user_q = random.choice(AMBER_PRICE_QUERIES)
            sys_lines = [entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh")]
            calls = [("get_entity_state",
                      {"entity_id": "sensor.charm_house_general_price"},
                      str(e["buy_price"]))]
            if e["buy_cents"] < 0:
                reply = (f"{opener()}Buy price is {e['buy_cents']} c/kWh, Charm — you're being paid to consume right now.")
            elif e["buy_cents"] < 10:
                reply = f"{opener()}Buy price is {e['buy_cents']} c/kWh, Charm — very cheap."
            elif e["buy_cents"] < 30:
                reply = f"{opener()}Buy price is {e['buy_cents']} c/kWh, Charm — typical level."
            elif e["buy_cents"] < 50:
                reply = f"{opener()}Buy price is {e['buy_cents']} c/kWh, Charm — elevated."
            else:
                reply = f"{opener()}Buy price is {e['buy_cents']} c/kWh, Charm — high, treat as a spike-adjacent window."

        elif q_type == "feed_now":
            user_q = random.choice(AMBER_FEED_QUERIES)
            sys_lines = [entity_line("sensor.charm_house_feed_in_price", e["feed_price"], "$/kWh")]
            calls = [("get_entity_state",
                      {"entity_id": "sensor.charm_house_feed_in_price"},
                      str(e["feed_price"]))]
            reply = f"{opener()}Feed-in is paying {e['feed_cents']} c/kWh right now, Charm."

        elif q_type == "spike_now":
            user_q = random.choice(AMBER_SPIKE_QUERIES)
            sys_lines = [
                entity_line("binary_sensor.charm_house_price_spike", "on" if e["spike_active"] else "off"),
                entity_line("binary_sensor.amber_spike_imminent", "on" if e["spike_imminent"] else "off"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "binary_sensor.charm_house_price_spike"},
                 "on" if e["spike_active"] else "off"),
                ("get_entity_state", {"entity_id": "binary_sensor.amber_spike_imminent"},
                 "on" if e["spike_imminent"] else "off"),
            ]
            if e["spike_active"]:
                reply = f"{opener()}A spike is active right now, Charm — buy price sits at {e['buy_cents']} c/kWh."
            elif e["spike_imminent"]:
                reply = f"{opener()}No spike active, but one looks imminent, Charm. Worth deferring heavy loads."
            else:
                reply = f"{opener()}No spike active, no spike imminent, Charm. Prices look orderly."

        elif q_type == "timeline_cheap":
            user_q = random.choice([
                "when's the next cheap window", "when does the price drop",
                "find me a cheap charging window", "when will it be cheap to charge",
                "next cheap slot",
            ])
            timeline = randomise_amber_timeline(e["time"][0], e["time"][1], hours=6)
            sys_lines = [entity_line("sensor.amber_price_timeline", str(e["buy_price"]),
                                     friendly="Amber Price Timeline")]
            calls = [("get_entity_state",
                      {"entity_id": "sensor.amber_price_timeline", "attribute": "data"},
                      json.dumps(timeline))]
            cheap = _next_cheap_window(timeline)
            if cheap:
                first = cheap[0]
                last = cheap[-1]
                cents = round(first["price_per_kwh"] * 100)
                start_h = first["start_time"][11:16]
                end_h = last["start_time"][11:16]
                reply = (f"{opener()}A cheap window opens around {start_h} through {end_h}, Charm — "
                         f"prices drop to about {cents} c/kWh.")
            else:
                reply = (f"{opener()}No clear cheap window in the next six hours, Charm — prices stay above "
                         f"15 c/kWh across the timeline.")

        elif q_type == "timeline_spike":
            user_q = random.choice([
                "is a spike coming", "when is the next price spike",
                "any peaks coming up", "evening peak forecast",
            ])
            timeline = randomise_amber_timeline(e["time"][0], e["time"][1], hours=8, spike_prob=0.06)
            sys_lines = [entity_line("sensor.amber_price_timeline", str(e["buy_price"]),
                                     friendly="Amber Price Timeline")]
            calls = [("get_entity_state",
                      {"entity_id": "sensor.amber_price_timeline", "attribute": "data"},
                      json.dumps(timeline))]
            sp_item, sp_c = _next_spike(timeline)
            if sp_item:
                t_h = sp_item["start_time"][11:16]
                reply = (f"{opener()}A peak appears around {t_h}, Charm — price climbs to roughly "
                         f"{sp_c} c/kWh. Plan heavy use before then.")
            else:
                reply = f"{opener()}No price peak visible in the forecast, Charm."

        elif q_type == "tonights_prices":
            user_q = random.choice([
                "what are tonight's prices like", "how do tonight's prices look",
                "what's tonight looking like", "summarise tonight's price forecast",
            ])
            timeline = randomise_amber_timeline(18, 0, hours=6)
            sys_lines = [entity_line("sensor.amber_price_timeline", str(e["buy_price"]),
                                     friendly="Amber Price Timeline")]
            calls = [("get_entity_state",
                      {"entity_id": "sensor.amber_price_timeline", "attribute": "data"},
                      json.dumps(timeline))]
            prices = [round(t["price_per_kwh"] * 100) for t in timeline]
            avg = round(sum(prices) / len(prices))
            mx = max(prices)
            mn = min(prices)
            reply = (f"{opener()}Tonight averages around {avg} c/kWh, Charm — peaks near {mx} c/kWh, "
                     f"lows near {mn} c/kWh.")

        elif q_type == "is_now_good":
            user_q = random.choice([
                "is it worth running the dishwasher now",
                "is now a good time to do a big wash",
                "is the price decent for charging the car right now",
                "ok to run the dryer now",
            ])
            sys_lines = [
                entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
                entity_line("binary_sensor.charm_house_price_spike", "on" if e["spike_active"] else "off"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                 str(e["buy_price"])),
                ("get_entity_state", {"entity_id": "binary_sensor.charm_house_price_spike"},
                 "on" if e["spike_active"] else "off"),
            ]
            if e["spike_active"]:
                reply = f"Spike active and buy price at {e['buy_cents']} c/kWh, Charm — I'd hold off."
            elif e["buy_cents"] <= 18:
                reply = f"Price is {e['buy_cents']} c/kWh with no spike, Charm — fine to run."
            else:
                reply = (f"Price is {e['buy_cents']} c/kWh, Charm — workable but not cheap; "
                         f"a cheaper window may appear shortly.")

        else:  # spot
            user_q = random.choice([
                "what's the underlying spot price", "wholesale spot price",
                "what's the spot per kwh now",
            ])
            sys_lines = [entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh")]
            calls = [("get_entity_state",
                      {"entity_id": "sensor.charm_house_general_price",
                       "attribute": "spot_per_kwh"},
                      str(round(e["buy_price"] * random.uniform(0.55, 0.75), 4)))]
            spot_c = round(float(calls[0][2]) * 100)
            reply = f"{opener()}Spot is sitting near {spot_c} c/kWh, Charm — retail buy works out to {e['buy_cents']} c/kWh after fees."

        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 5 — Amber reasoning (100, multi-tool + recommendation)
# ---------------------------------------------------------------------------
AMBER_REASONING_QUERIES = [
    "should I override the optimiser right now",
    "is it worth force charging",
    "explain what the optimiser is doing and why",
    "should I run heavy appliances now",
    "is the optimiser making good decisions today",
    "would it make sense to sell right now",
    "given the price forecast, what would you do",
    "is 26 cents a good price to charge at",
    "should I wait for cheaper power",
    "is it worth holding the battery",
    "what's the smart play right now",
    "is the optimiser stuck",
]

def gen_batch_05_amber_reasoning(n=100):
    out = []
    for _ in range(n):
        e = randomise_energy()
        timeline = randomise_amber_timeline(e["time"][0], e["time"][1], hours=8, spike_prob=0.07)
        q = random.choice(AMBER_REASONING_QUERIES)
        calls = [
            ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"}, str(e["buy_price"])),
            ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"}, str(e["battery_soc"])),
            ("get_entity_state", {"entity_id": "sensor.llm_response_file"}, e["decision"]),
            ("get_entity_state", {"entity_id": "sensor.amber_price_timeline",
                                  "attribute": "data"}, json.dumps(timeline)),
        ]
        sys_lines = [
            entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
            entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
            entity_line("sensor.llm_response_file", e["decision"], friendly="Optimiser Decision"),
            entity_line("sensor.amber_price_timeline", str(e["buy_price"]),
                        friendly="Amber Price Timeline"),
        ]

        prices = [round(t["price_per_kwh"] * 100) for t in timeline]
        future_min = min(prices) if prices else e["buy_cents"]
        future_max = max(prices) if prices else e["buy_cents"]

        if "override" in q or "smart play" in q or "would you do" in q:
            if e["spike_active"]:
                reply = (f"{opener('analysis')}Spike active at {e['buy_cents']} c/kWh and battery at "
                         f"{e['battery_soc']}%, Charm. The optimiser is on {e['decision']} — I'd let it "
                         f"discharge into this peak rather than override.")
            elif e["battery_soc"] < 20 and future_min < e["buy_cents"] - 8:
                reply = (f"{opener('analysis')}Battery at {e['battery_soc']}% and a cheaper window "
                         f"({future_min} c/kWh) is in range, Charm. Optimiser is on {e['decision']}; "
                         f"holding off and force-charging in the trough is the better play.")
            else:
                reply = (f"{opener('analysis')}Battery at {e['battery_soc']}%, price {e['buy_cents']} c/kWh, "
                         f"forecast {future_min}–{future_max} c/kWh, Charm. The optimiser's {e['decision']} call "
                         f"looks defensible — no need to override.")
        elif "force charging" in q or "force charge" in q:
            if e["buy_cents"] <= 8 and e["battery_soc"] < 90:
                reply = (f"Yes, Charm — buy price {e['buy_cents']} c/kWh with battery at {e['battery_soc']}% "
                         f"is a worthwhile force-charge. Optimiser is currently on {e['decision']}.")
            elif e["buy_cents"] > 25:
                reply = (f"Not at {e['buy_cents']} c/kWh, Charm — that's too expensive to fill the battery. "
                         f"Forecast low is {future_min} c/kWh; better to wait.")
            else:
                reply = (f"{opener('analysis')}Price is {e['buy_cents']} c/kWh, Charm — defensible but not great. "
                         f"Forecast low is {future_min} c/kWh, so I'd lean toward waiting.")
        elif "sell" in q or "discharge" in q:
            if e["feed_cents"] > 30 and e["battery_soc"] > 60:
                reply = (f"Yes, Charm — feed-in at {e['feed_cents']} c/kWh with battery at {e['battery_soc']}% "
                         f"is a sound sell window. Optimiser is on {e['decision']}.")
            elif e["battery_soc"] < 30:
                reply = (f"Not while battery is at {e['battery_soc']}%, Charm — there isn't enough headroom "
                         f"above the reserve target.")
            else:
                reply = (f"Feed-in at {e['feed_cents']} c/kWh is unremarkable, Charm — better to wait for a "
                         f"genuine peak before selling.")
        elif "26 cents" in q or "is" in q and "good price" in q:
            test_c = 26
            if test_c <= future_min + 3:
                reply = f"{test_c} c/kWh is near the forecast low ({future_min} c/kWh), Charm — yes, reasonable to charge."
            else:
                reply = (f"{test_c} c/kWh is above the forecast low of {future_min} c/kWh, Charm — "
                         f"better to wait if you have flex.")
        elif "heavy appliances" in q or "wait for cheaper" in q or "is it worth holding" in q:
            if e["spike_active"]:
                reply = (f"Spike in effect at {e['buy_cents']} c/kWh, Charm — defer anything heavy until the price "
                         f"comes down. Forecast bottoms near {future_min} c/kWh.")
            elif e["buy_cents"] <= 15:
                reply = f"{e['buy_cents']} c/kWh and no spike, Charm — go ahead with anything heavy."
            else:
                reply = (f"At {e['buy_cents']} c/kWh, not ideal, Charm. Forecast dips to {future_min} c/kWh — "
                         f"waiting saves you a few cents per kWh.")
        elif "explain what the optimiser" in q or "is the optimiser making good" in q or "is the optimiser stuck" in q:
            reply = (f"{opener('analysis')}Optimiser is on {e['decision']} with a {int(e['confidence']*100)}% "
                     f"confidence call from {e['model']}, Charm. Inputs: battery {e['battery_soc']}%, "
                     f"buy {e['buy_cents']} c/kWh, forecast range {future_min}–{future_max} c/kWh. "
                     f"That's a coherent choice for the current conditions.")
        else:
            reply = (f"{opener('analysis')}Battery {e['battery_soc']}%, price {e['buy_cents']} c/kWh, "
                     f"forecast {future_min}–{future_max} c/kWh, optimiser on {e['decision']}, Charm. "
                     f"No action recommended.")

        out.append(multi_call_example(sys_lines, q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 6 — Optimiser control (30)
# ---------------------------------------------------------------------------
def gen_batch_06_optimiser_control(n=30):
    out = []
    for _ in range(n):
        action = random.choice([
            "trigger", "force_charge", "force_sell", "self_use",
            "disable", "enable", "decision_mode_auto",
            "decision_mode_notify", "decision_mode_disabled", "set_target_soc",
        ])
        e = randomise_energy()
        if action == "trigger":
            user_q = random.choice([
                "trigger an optimiser run", "ask the optimiser for a decision now",
                "rerun the optimiser", "fresh optimiser run please",
            ])
            sys_lines = [entity_line("script.amber_request_decision", "off", friendly="Amber Request Decision")]
            calls = [("call_service",
                      {"domain": "script", "service": "turn_on",
                       "entity_id": "script.amber_request_decision"}, "ok")]
            reply = f"{opener('control')}Optimiser run requested, Charm — a fresh decision should land shortly."
        elif action == "force_charge":
            user_q = random.choice([
                "force charge the battery", "trigger a force charge",
                "force charge now",
            ])
            sys_lines = [entity_line("script.amber_force_charge", "off")]
            calls = [("call_service",
                      {"domain": "script", "service": "turn_on",
                       "entity_id": "script.amber_force_charge"}, "ok")]
            reply = f"{opener('control')}Force charge engaged, Charm."
        elif action == "force_sell":
            user_q = random.choice([
                "force sell now", "trigger a force discharge to grid",
                "sell to grid",
            ])
            sys_lines = [entity_line("script.amber_force_sell", "off")]
            calls = [("call_service",
                      {"domain": "script", "service": "turn_on",
                       "entity_id": "script.amber_force_sell"}, "ok")]
            reply = f"{opener('control')}Force sell engaged, Charm."
        elif action == "self_use":
            user_q = random.choice([
                "go back to self use", "stop force charging and self use",
                "switch to self use", "return to self use mode",
            ])
            sys_lines = [entity_line("script.amber_self_use", "off")]
            calls = [("call_service",
                      {"domain": "script", "service": "turn_on",
                       "entity_id": "script.amber_self_use"}, "ok")]
            reply = f"{opener('control')}Returned to self-use, Charm."
        elif action == "disable":
            user_q = random.choice([
                "disable the optimiser", "turn off the optimiser",
                "kill the optimiser",
            ])
            sys_lines = [entity_line("input_boolean.llm_optimizer_enabled", "on")]
            calls = [("call_service",
                      {"domain": "input_boolean", "service": "turn_off",
                       "entity_id": "input_boolean.llm_optimizer_enabled"}, "ok")]
            reply = f"{opener('control')}Optimiser disabled, Charm — manual control until you re-enable."
        elif action == "enable":
            user_q = random.choice([
                "enable the optimiser", "turn the optimiser back on",
                "switch the optimiser on",
            ])
            sys_lines = [entity_line("input_boolean.llm_optimizer_enabled", "off")]
            calls = [("call_service",
                      {"domain": "input_boolean", "service": "turn_on",
                       "entity_id": "input_boolean.llm_optimizer_enabled"}, "ok")]
            reply = f"{opener('control')}Optimiser enabled, Charm."
        elif action == "decision_mode_auto":
            user_q = random.choice([
                "set decision mode to auto execute", "auto execute mode",
                "let the optimiser act automatically",
            ])
            sys_lines = [entity_line("input_select.llm_decision_mode", e["decision_mode"])]
            calls = [("call_service",
                      {"domain": "input_select", "service": "select_option",
                       "entity_id": "input_select.llm_decision_mode",
                       "data": {"option": "auto_execute"}}, "ok")]
            reply = f"{opener('control')}Decision mode set to auto_execute, Charm."
        elif action == "decision_mode_notify":
            user_q = random.choice([
                "switch decision mode to notify only", "notify only mode",
                "stop auto-executing, just notify",
            ])
            sys_lines = [entity_line("input_select.llm_decision_mode", e["decision_mode"])]
            calls = [("call_service",
                      {"domain": "input_select", "service": "select_option",
                       "entity_id": "input_select.llm_decision_mode",
                       "data": {"option": "notify_only"}}, "ok")]
            reply = f"{opener('control')}Decision mode set to notify_only, Charm — no automatic action."
        elif action == "decision_mode_disabled":
            user_q = random.choice([
                "set decision mode to disabled", "disable decisions",
                "freeze optimiser actions",
            ])
            sys_lines = [entity_line("input_select.llm_decision_mode", e["decision_mode"])]
            calls = [("call_service",
                      {"domain": "input_select", "service": "select_option",
                       "entity_id": "input_select.llm_decision_mode",
                       "data": {"option": "disabled"}}, "ok")]
            reply = f"{opener('control')}Decision mode set to disabled, Charm."
        else:  # set_target_soc
            target = random.choice([40, 50, 60, 70, 80, 85, 90, 95, 100])
            user_q = random.choice([
                f"set the target soc to {target}%",
                f"active target soc {target}",
                f"aim for {target} percent soc",
            ])
            sys_lines = [entity_line("input_number.llm_active_target_soc", e["target_soc"], "%")]
            calls = [("call_service",
                      {"domain": "input_number", "service": "set_value",
                       "entity_id": "input_number.llm_active_target_soc",
                       "data": {"value": target}}, "ok")]
            reply = f"{opener('control')}Target SoC set to {target}%, Charm."

        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 7 — HVAC status (80)
# ---------------------------------------------------------------------------
HVAC_STATUS_QUERIES = [
    ("is the aircon on", "running"),
    ("is the heating on", "running"),
    ("is the hvac running", "running"),
    ("what mode is the aircon in", "mode"),
    ("what's the temperature downstairs", "downstairs_temp"),
    ("what's the upstairs temp", "upstairs_temp"),
    ("which zone is coldest", "coldest"),
    ("which zone is the warmest", "hottest"),
    ("how long has the heating been running today", "heat_runtime"),
    ("how long has the cooling been running today", "cool_runtime"),
    ("is the hvac paused for any reason", "pause"),
    ("what's the supply air temperature", "supply"),
    ("is the house being heated right now", "heating_now"),
    ("is the house being cooled right now", "cooling_now"),
    ("what's the setpoint", "setpoint"),
    ("is hvac automation enabled", "automation"),
    ("what's the schedule mode", "schedule"),
    ("what's the heat trigger set to", "heat_trigger"),
    ("what's the cool trigger set to", "cool_trigger"),
]

def gen_batch_07_hvac_status(n=80):
    out = []
    for _ in range(n):
        energy = randomise_energy()
        hv = randomise_hvac(energy=energy)
        q, qtype = random.choice(HVAC_STATUS_QUERIES)
        if qtype == "running":
            sys_lines = [entity_line("binary_sensor.hvac_running", "on" if hv["running"] else "off")]
            calls = [("get_entity_state", {"entity_id": "binary_sensor.hvac_running"},
                      "on" if hv["running"] else "off")]
            if hv["running"]:
                reply = f"{opener()}HVAC is running, Charm — currently in {hv['main_mode']} mode."
            else:
                reply = f"{opener()}HVAC is off, Charm."
        elif qtype == "mode":
            sys_lines = [entity_line("climate.izone_controller_402001095", hv["main_mode"])]
            calls = [("get_entity_state", {"entity_id": "climate.izone_controller_402001095"},
                      hv["main_mode"])]
            reply = f"{opener()}HVAC mode is {hv['main_mode']}, Charm."
        elif qtype == "downstairs_temp":
            sys_lines = [entity_line("sensor.downstairs_zone_temperature", hv["downstairs_temp"], "°C")]
            calls = [("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                      str(hv["downstairs_temp"]))]
            reply = f"{opener()}Downstairs is at {hv['downstairs_temp']} °C, Charm."
        elif qtype == "upstairs_temp":
            sys_lines = [entity_line("sensor.upstairs_zone_temperature", hv["upstairs_temp"], "°C")]
            calls = [("get_entity_state", {"entity_id": "sensor.upstairs_zone_temperature"},
                      str(hv["upstairs_temp"]))]
            reply = f"{opener()}Upstairs is at {hv['upstairs_temp']} °C, Charm."
        elif qtype == "coldest":
            sys_lines = [
                entity_line("sensor.hvac_coldest_zone_name", hv["coldest_zone"]),
                entity_line("sensor.hvac_coldest_zone_temp", hv["coldest_temp"], "°C"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.hvac_coldest_zone_name"}, hv["coldest_zone"]),
                ("get_entity_state", {"entity_id": "sensor.hvac_coldest_zone_temp"}, str(hv["coldest_temp"])),
            ]
            reply = f"{opener()}{hv['coldest_zone']} is the coldest at {hv['coldest_temp']} °C, Charm."
        elif qtype == "hottest":
            sys_lines = [
                entity_line("sensor.hvac_hottest_zone_name", hv["hottest_zone"]),
                entity_line("sensor.hvac_hottest_zone_temp", hv["hottest_temp"], "°C"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.hvac_hottest_zone_name"}, hv["hottest_zone"]),
                ("get_entity_state", {"entity_id": "sensor.hvac_hottest_zone_temp"}, str(hv["hottest_temp"])),
            ]
            reply = f"{opener()}{hv['hottest_zone']} is the warmest at {hv['hottest_temp']} °C, Charm."
        elif qtype == "heat_runtime":
            sys_lines = [entity_line("sensor.hvac_heat_runtime_today_2", hv["heat_runtime_h"], "h")]
            calls = [("get_entity_state", {"entity_id": "sensor.hvac_heat_runtime_today_2"},
                      str(hv["heat_runtime_h"]))]
            reply = f"{opener()}Heating has run for {hv['heat_runtime_h']} hours today, Charm."
        elif qtype == "cool_runtime":
            sys_lines = [entity_line("sensor.hvac_cool_runtime_today_2", hv["cool_runtime_h"], "h")]
            calls = [("get_entity_state", {"entity_id": "sensor.hvac_cool_runtime_today_2"},
                      str(hv["cool_runtime_h"]))]
            reply = f"{opener()}Cooling has run for {hv['cool_runtime_h']} hours today, Charm."
        elif qtype == "pause":
            sys_lines = [
                entity_line("binary_sensor.hvac_low_soc_pause_active", "on" if hv["low_soc_pause"] else "off"),
                entity_line("input_boolean.hvac_amber_pause_active", "on" if hv["amber_pause"] else "off"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "binary_sensor.hvac_low_soc_pause_active"},
                 "on" if hv["low_soc_pause"] else "off"),
                ("get_entity_state", {"entity_id": "input_boolean.hvac_amber_pause_active"},
                 "on" if hv["amber_pause"] else "off"),
            ]
            if hv["low_soc_pause"] and hv["amber_pause"]:
                reply = f"{opener()}Paused for both low battery and high Amber price, Charm."
            elif hv["low_soc_pause"]:
                reply = f"{opener()}Paused because the battery is low, Charm."
            elif hv["amber_pause"]:
                reply = f"{opener()}Paused because Amber prices are elevated, Charm."
            else:
                reply = f"{opener()}HVAC isn't paused, Charm — running on normal logic."
        elif qtype == "supply":
            sys_lines = [entity_line("sensor.hvac_supply_temperature", hv["supply_temp"], "°C")]
            calls = [("get_entity_state", {"entity_id": "sensor.hvac_supply_temperature"},
                      str(hv["supply_temp"]))]
            reply = f"{opener()}Supply air is at {hv['supply_temp']} °C, Charm."
        elif qtype == "heating_now":
            sys_lines = [entity_line("binary_sensor.hvac_actually_heating",
                                     "on" if hv["actually_heating"] else "off")]
            calls = [("get_entity_state", {"entity_id": "binary_sensor.hvac_actually_heating"},
                      "on" if hv["actually_heating"] else "off")]
            if hv["actually_heating"]:
                reply = f"{opener()}Yes, the house is being heated, Charm — setpoint {hv['setpoint']} °C against ambient {hv['downstairs_temp']} °C."
            else:
                reply = f"{opener()}No active heating right now, Charm."
        elif qtype == "cooling_now":
            sys_lines = [entity_line("binary_sensor.hvac_actually_cooling",
                                     "on" if hv["actually_cooling"] else "off")]
            calls = [("get_entity_state", {"entity_id": "binary_sensor.hvac_actually_cooling"},
                      "on" if hv["actually_cooling"] else "off")]
            if hv["actually_cooling"]:
                reply = f"{opener()}Yes, cooling is active, Charm — setpoint {hv['setpoint']} °C against ambient {hv['downstairs_temp']} °C."
            else:
                reply = f"{opener()}No active cooling right now, Charm."
        elif qtype == "setpoint":
            sys_lines = [entity_line("climate.izone_controller_402001095", hv["main_mode"],
                                     friendly="iZone Controller")]
            calls = [("get_entity_state", {"entity_id": "climate.izone_controller_402001095",
                                            "attribute": "temperature"}, str(hv["setpoint"]))]
            reply = f"{opener()}Setpoint is {hv['setpoint']} °C, Charm."
        elif qtype == "automation":
            sys_lines = [entity_line("input_boolean.hvac_automation_enabled",
                                     "on" if hv["automation"] else "off")]
            calls = [("get_entity_state", {"entity_id": "input_boolean.hvac_automation_enabled"},
                      "on" if hv["automation"] else "off")]
            reply = (f"{opener()}HVAC automation is {'enabled' if hv['automation'] else 'disabled'}, Charm.")
        elif qtype == "schedule":
            sys_lines = [entity_line("input_select.hvac_schedule_mode", hv["schedule_mode"])]
            calls = [("get_entity_state", {"entity_id": "input_select.hvac_schedule_mode"},
                      hv["schedule_mode"])]
            reply = f"{opener()}Schedule mode is {hv['schedule_mode']}, Charm."
        elif qtype == "heat_trigger":
            sys_lines = [entity_line("input_number.hvac_heat_trigger_downstairs",
                                     hv["heat_trigger"], "°C")]
            calls = [("get_entity_state", {"entity_id": "input_number.hvac_heat_trigger_downstairs"},
                      str(hv["heat_trigger"]))]
            reply = f"{opener()}Heat trigger sits at {hv['heat_trigger']} °C, Charm."
        else:  # cool_trigger
            sys_lines = [entity_line("input_number.hvac_cool_trigger_downstairs",
                                     hv["cool_trigger"], "°C")]
            calls = [("get_entity_state", {"entity_id": "input_number.hvac_cool_trigger_downstairs"},
                      str(hv["cool_trigger"]))]
            reply = f"{opener()}Cool trigger sits at {hv['cool_trigger']} °C, Charm."

        out.append(multi_call_example(sys_lines, q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 8 — HVAC reasoning (60)
# ---------------------------------------------------------------------------
HVAC_REASONING_QUERIES = [
    "should i turn the heating on",
    "should i turn the aircon on",
    "is it efficient to run the ac right now",
    "the house feels cold — what should i do",
    "the house feels warm — what should i do",
    "explain why the hvac paused",
    "is now a good time to preheat the house given the amber price",
    "is now a good time to precool the house given the amber price",
    "what's the temperature difference between floors",
    "how's the hvac been running today",
    "should i pre-cool before the evening peak",
    "should i let the hvac run or wait for solar",
]

def gen_batch_08_hvac_reasoning(n=60):
    out = []
    for _ in range(n):
        e = randomise_energy()
        hv = randomise_hvac(energy=e)
        q = random.choice(HVAC_REASONING_QUERIES)
        calls = [
            ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
             str(hv["downstairs_temp"])),
            ("get_entity_state", {"entity_id": "sensor.upstairs_zone_temperature"},
             str(hv["upstairs_temp"])),
            ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
             str(hv["outdoor_temp"])),
            ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
             str(e["battery_soc"])),
            ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
             str(e["buy_price"])),
        ]
        sys_lines = [
            entity_line("sensor.downstairs_zone_temperature", hv["downstairs_temp"], "°C"),
            entity_line("sensor.upstairs_zone_temperature", hv["upstairs_temp"], "°C"),
            entity_line("sensor.home_charm_temp", hv["outdoor_temp"], "°C"),
            entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
            entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
        ]

        if "feels cold" in q or "turn the heating" in q or "preheat" in q:
            if hv["downstairs_temp"] >= 21:
                reply = (f"{opener('analysis')}Downstairs already at {hv['downstairs_temp']} °C, Charm — not cold "
                         f"on the sensor. Heating isn't called for unless you want a higher setpoint.")
            elif e["spike_active"]:
                reply = (f"{opener('analysis')}Downstairs {hv['downstairs_temp']} °C and price {e['buy_cents']} c/kWh "
                         f"with a spike, Charm — better to wait. Pull a blanket out or run the heat briefly off battery.")
            elif e["buy_cents"] > 35 and e["battery_soc"] < 30:
                reply = (f"{opener('analysis')}{hv['downstairs_temp']} °C downstairs, but battery {e['battery_soc']}% "
                         f"and {e['buy_cents']} c/kWh make this an expensive heat, Charm. Wait if you can.")
            else:
                target = max(20, round(hv["downstairs_temp"]) + 3)
                reply = (f"{opener('analysis')}{hv['downstairs_temp']} °C downstairs, outdoor {hv['outdoor_temp']} °C, "
                         f"price {e['buy_cents']} c/kWh, battery {e['battery_soc']}%, Charm. Run the HVAC in "
                         f"heat_cool with setpoint around {target} °C — the controller will heat since the setpoint "
                         f"is above ambient.")
        elif "feels warm" in q or "turn the aircon" in q or "precool" in q or "pre-cool" in q:
            if hv["downstairs_temp"] <= 23:
                reply = (f"{opener('analysis')}Downstairs only at {hv['downstairs_temp']} °C, Charm — that's not "
                         f"actually warm yet. Open a window before spending battery on the AC.")
            elif e["spike_active"]:
                reply = (f"{opener('analysis')}{hv['downstairs_temp']} °C is warm, Charm, but with a spike active at "
                         f"{e['buy_cents']} c/kWh, run cooling from battery only — keep windows shut.")
            else:
                target = min(24, round(hv["downstairs_temp"]) - 2)
                reply = (f"{opener('analysis')}{hv['downstairs_temp']} °C downstairs, outdoor {hv['outdoor_temp']} °C, "
                         f"battery {e['battery_soc']}%, Charm. Setpoint around {target} °C in heat_cool — the controller "
                         f"will cool since the setpoint is below ambient.")
        elif "efficient to run" in q:
            if e["solar_kw"] > 2 and e["battery_soc"] > 50:
                reply = (f"{opener('analysis')}Solar at {e['solar_kw']} kW and battery {e['battery_soc']}%, Charm — "
                         f"cooling will mostly come from solar and battery rather than the grid. Efficient window.")
            elif e["spike_active"]:
                reply = (f"{opener('analysis')}Spike at {e['buy_cents']} c/kWh, Charm — every kWh the AC uses gets "
                         f"expensive. Wait for the price to come down or rely on battery.")
            else:
                reply = (f"{opener('analysis')}Price {e['buy_cents']} c/kWh, battery {e['battery_soc']}%, Charm — "
                         f"acceptable, though running the AC from the battery alone would be cheaper if the SoC "
                         f"can absorb it.")
        elif "paused" in q and "explain" in q:
            if hv["low_soc_pause"] and hv["amber_pause"]:
                reply = (f"{opener('analysis')}Two reasons, Charm — battery at {e['battery_soc']}% is below the "
                         f"low-SoC threshold, and Amber sits at {e['buy_cents']} c/kWh. Both protections triggered.")
            elif hv["low_soc_pause"]:
                reply = (f"{opener('analysis')}Battery at {e['battery_soc']}% is below the low-SoC threshold, Charm — "
                         f"HVAC is paused so the battery isn't drained below reserve.")
            elif hv["amber_pause"]:
                reply = (f"{opener('analysis')}Amber price at {e['buy_cents']} c/kWh triggered the high-price pause, Charm — "
                         f"HVAC will resume once price normalises.")
            else:
                reply = (f"{opener('analysis')}HVAC isn't actually paused, Charm — no pause flags active. It may "
                         f"simply be idle because the zones are within their triggers.")
        elif "difference between floors" in q:
            diff = round(abs(hv["upstairs_temp"] - hv["downstairs_temp"]), 1)
            warmer = "upstairs" if hv["upstairs_temp"] > hv["downstairs_temp"] else "downstairs"
            reply = (f"{opener('analysis')}Upstairs {hv['upstairs_temp']} °C, downstairs {hv['downstairs_temp']} °C — "
                     f"{diff} °C gap with {warmer} warmer, Charm.")
        elif "preheat" in q or "precool" in q or "pre-cool" in q or "before the evening peak" in q:
            if e["buy_cents"] < 20 and e["battery_soc"] > 60:
                reply = (f"{opener('analysis')}Price {e['buy_cents']} c/kWh and battery {e['battery_soc']}%, Charm — "
                         f"this is a sensible window to pre-condition before the evening peak.")
            else:
                reply = (f"{opener('analysis')}Conditions are tight, Charm — battery {e['battery_soc']}%, price "
                         f"{e['buy_cents']} c/kWh. Pre-conditioning will cost more than it saves at the peak.")
        elif "wait for solar" in q:
            if e["is_day"] and e["solar_kw"] < 1.5 and e["battery_soc"] < 40:
                reply = (f"{opener('analysis')}Solar low at {e['solar_kw']} kW and battery {e['battery_soc']}%, Charm — "
                         f"waiting an hour or two should give the array a chance to produce.")
            elif e["solar_kw"] > 3:
                reply = (f"{opener('analysis')}Solar is at {e['solar_kw']} kW, Charm — go ahead, the HVAC will run "
                         f"largely on direct solar.")
            else:
                reply = (f"{opener('analysis')}Solar {e['solar_kw']} kW, battery {e['battery_soc']}%, price "
                         f"{e['buy_cents']} c/kWh, Charm — borderline; run only if comfort requires it.")
        else:  # "how's the hvac been running today"
            reply = (f"{opener('analysis')}Heating {hv['heat_runtime_h']} h, cooling {hv['cool_runtime_h']} h today, Charm. "
                     f"Current mode {hv['main_mode']}, supply air {hv['supply_temp']} °C, "
                     f"downstairs {hv['downstairs_temp']} °C, upstairs {hv['upstairs_temp']} °C.")

        out.append(multi_call_example(sys_lines, q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 9 — HVAC control (80) — encodes heat_cool-decides-direction behaviour
# ---------------------------------------------------------------------------
def _hvac_decide(setpoint, ambient):
    if setpoint > ambient + 0.5:
        return "heat"
    if setpoint < ambient - 0.5:
        return "cool"
    return "idle"

HVAC_CONTROL_PATTERNS = [
    "turn_on_heating",
    "turn_on_aircon",
    "set_downstairs_temp",
    "set_upstairs_temp",
    "turn_off_aircon",
    "switch_fan_only",
    "schedule_sleep",
    "schedule_away",
    "schedule_auto",
    "schedule_manual",
    "set_heat_trigger",
    "set_cool_trigger",
    "enable_automation",
    "disable_automation",
    "set_main_mode_off",
    "heat_upstairs_only",
    "cool_downstairs_only",
]

def gen_batch_09_hvac_control(n=80):
    out = []
    patterns_remaining = list(HVAC_CONTROL_PATTERNS) * 6
    random.shuffle(patterns_remaining)
    for i in range(n):
        pat = patterns_remaining[i % len(patterns_remaining)]
        # Bias more to the direction-aware patterns to ensure >=40 hits
        if i < 50 and pat not in ("turn_on_heating", "turn_on_aircon",
                                  "set_downstairs_temp", "set_upstairs_temp",
                                  "heat_upstairs_only", "cool_downstairs_only"):
            pat = random.choice(["turn_on_heating", "turn_on_aircon",
                                 "set_downstairs_temp", "set_upstairs_temp",
                                 "heat_upstairs_only", "cool_downstairs_only"])

        e = randomise_energy()
        hv = randomise_hvac(energy=e)

        if pat == "turn_on_heating":
            user_q = random.choice([
                "turn on the heating", "fire up the heater", "kick the heating on",
                "switch heating on", "heat the house",
            ])
            ambient = hv["downstairs_temp"]
            target = round(max(20.0, ambient + random.uniform(2.0, 4.0)), 1)
            sys_lines = [
                entity_line("sensor.downstairs_zone_temperature", ambient, "°C"),
                entity_line("climate.izone_controller_402001095", "off"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(ambient)),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"temperature": target}}, "ok"),
            ]
            reply = (f"{opener('control')}Setpoint {target} °C against current {ambient} °C, "
                     f"so it will heat, Charm. iZone runs in heat_cool — direction follows the gap.")

        elif pat == "turn_on_aircon":
            user_q = random.choice([
                "turn on the aircon", "turn on the ac", "fire up the cooling",
                "kick the ac on", "switch cooling on",
            ])
            ambient = hv["downstairs_temp"]
            target = round(min(24.0, ambient - random.uniform(2.0, 4.0)), 1)
            sys_lines = [
                entity_line("sensor.downstairs_zone_temperature", ambient, "°C"),
                entity_line("climate.izone_controller_402001095", "off"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(ambient)),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"temperature": target}}, "ok"),
            ]
            reply = (f"{opener('control')}Setpoint {target} °C against current {ambient} °C, "
                     f"so it will cool, Charm. iZone runs in heat_cool — direction follows the gap.")

        elif pat == "set_downstairs_temp":
            target = random.choice([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0])
            user_q = random.choice([
                f"set the downstairs to {int(target)} degrees", f"downstairs to {target}",
                f"set downstairs target to {int(target)}°C", f"downstairs setpoint {target}",
                f"set the downstairs to {int(target)}",
            ])
            ambient = hv["downstairs_temp"]
            direction = _hvac_decide(target, ambient)
            sys_lines = [
                entity_line("sensor.downstairs_zone_temperature", ambient, "°C"),
                entity_line("climate.downstairs", hv["zone_mode"]),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(ambient)),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.downstairs",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.downstairs",
                                  "data": {"temperature": target}}, "ok"),
            ]
            if direction == "heat":
                reply = (f"{opener('control')}Setpoint {target} °C against current {ambient} °C, "
                         f"so it will heat, Charm.")
            elif direction == "cool":
                reply = (f"{opener('control')}Setpoint {target} °C against current {ambient} °C, "
                         f"so it will cool, Charm.")
            else:
                reply = (f"{opener('control')}Setpoint {target} °C, current {ambient} °C — "
                         f"within the deadband, so the controller will idle until ambient drifts, Charm.")

        elif pat == "set_upstairs_temp":
            target = random.choice([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
            user_q = random.choice([
                f"set upstairs to {int(target)}", f"upstairs to {target} degrees",
                f"set upstairs setpoint {int(target)}", f"upstairs target {target}°C",
            ])
            ambient = hv["upstairs_temp"]
            direction = _hvac_decide(target, ambient)
            sys_lines = [
                entity_line("sensor.upstairs_zone_temperature", ambient, "°C"),
                entity_line("climate.upstairs", hv["zone_mode"]),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.upstairs_zone_temperature"},
                 str(ambient)),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.upstairs",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.upstairs",
                                  "data": {"temperature": target}}, "ok"),
            ]
            if direction == "heat":
                reply = (f"{opener('control')}Setpoint {target} °C against current {ambient} °C upstairs, "
                         f"so it will heat, Charm.")
            elif direction == "cool":
                reply = (f"{opener('control')}Setpoint {target} °C against current {ambient} °C upstairs, "
                         f"so it will cool, Charm.")
            else:
                reply = (f"{opener('control')}Setpoint {target} °C against current {ambient} °C — "
                         f"the controller will idle until ambient drifts, Charm.")

        elif pat == "heat_upstairs_only":
            target = round(max(21.0, hv["upstairs_temp"] + random.uniform(2.0, 4.0)), 1)
            user_q = random.choice([
                "heat upstairs only", "turn heating on upstairs only",
                "warm the upstairs zone only",
            ])
            sys_lines = [
                entity_line("sensor.upstairs_zone_temperature", hv["upstairs_temp"], "°C"),
                entity_line("climate.upstairs", hv["zone_mode"]),
                entity_line("climate.downstairs", hv["zone_mode"]),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.upstairs_zone_temperature"},
                 str(hv["upstairs_temp"])),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.downstairs",
                                  "data": {"hvac_mode": "off"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.upstairs",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.upstairs",
                                  "data": {"temperature": target}}, "ok"),
            ]
            reply = (f"{opener('control')}Downstairs off, upstairs setpoint {target} °C against "
                     f"current {hv['upstairs_temp']} °C, so it will heat, Charm.")

        elif pat == "cool_downstairs_only":
            target = round(min(24.0, hv["downstairs_temp"] - random.uniform(2.0, 4.0)), 1)
            user_q = random.choice([
                "cool downstairs only", "ac on for downstairs only",
                "cool the downstairs zone only",
            ])
            sys_lines = [
                entity_line("sensor.downstairs_zone_temperature", hv["downstairs_temp"], "°C"),
                entity_line("climate.downstairs", hv["zone_mode"]),
                entity_line("climate.upstairs", hv["zone_mode"]),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(hv["downstairs_temp"])),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.upstairs",
                                  "data": {"hvac_mode": "off"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.downstairs",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.downstairs",
                                  "data": {"temperature": target}}, "ok"),
            ]
            reply = (f"{opener('control')}Upstairs off, downstairs setpoint {target} °C against "
                     f"current {hv['downstairs_temp']} °C, so it will cool, Charm.")

        elif pat == "turn_off_aircon":
            user_q = random.choice([
                "turn off the aircon", "kill the hvac", "shut down the cooling",
                "turn off the hvac", "switch the aircon off",
            ])
            sys_lines = [entity_line("climate.izone_controller_402001095", hv["main_mode"])]
            calls = [("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                       "entity_id": "climate.izone_controller_402001095",
                                       "data": {"hvac_mode": "off"}}, "ok")]
            reply = f"{opener('control')}HVAC off, Charm."

        elif pat == "switch_fan_only":
            user_q = random.choice([
                "switch to fan only", "fan only mode",
                "set hvac to fan only", "circulate air only",
            ])
            sys_lines = [entity_line("climate.izone_controller_402001095", hv["main_mode"])]
            calls = [("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                       "entity_id": "climate.izone_controller_402001095",
                                       "data": {"hvac_mode": "fan_only"}}, "ok")]
            reply = f"{opener('control')}HVAC in fan_only, Charm — circulation only, no conditioning."

        elif pat == "schedule_sleep":
            user_q = random.choice([
                "set schedule mode to sleep", "switch to sleep schedule",
                "hvac into sleep mode",
            ])
            sys_lines = [entity_line("input_select.hvac_schedule_mode", hv["schedule_mode"])]
            calls = [("call_service", {"domain": "input_select", "service": "select_option",
                                       "entity_id": "input_select.hvac_schedule_mode",
                                       "data": {"option": "sleep"}}, "ok")]
            reply = f"{opener('control')}Schedule mode set to sleep, Charm."

        elif pat == "schedule_away":
            user_q = random.choice([
                "set schedule mode to away", "away mode for the hvac",
                "switch to away schedule",
            ])
            sys_lines = [entity_line("input_select.hvac_schedule_mode", hv["schedule_mode"])]
            calls = [("call_service", {"domain": "input_select", "service": "select_option",
                                       "entity_id": "input_select.hvac_schedule_mode",
                                       "data": {"option": "away"}}, "ok")]
            reply = f"{opener('control')}Schedule mode set to away, Charm."

        elif pat == "schedule_auto":
            user_q = random.choice([
                "set schedule mode to auto", "back to auto schedule",
                "auto schedule",
            ])
            sys_lines = [entity_line("input_select.hvac_schedule_mode", hv["schedule_mode"])]
            calls = [("call_service", {"domain": "input_select", "service": "select_option",
                                       "entity_id": "input_select.hvac_schedule_mode",
                                       "data": {"option": "auto"}}, "ok")]
            reply = f"{opener('control')}Schedule mode set to auto, Charm."

        elif pat == "schedule_manual":
            user_q = random.choice([
                "set schedule mode to manual", "manual hvac control",
                "switch schedule to manual",
            ])
            sys_lines = [entity_line("input_select.hvac_schedule_mode", hv["schedule_mode"])]
            calls = [("call_service", {"domain": "input_select", "service": "select_option",
                                       "entity_id": "input_select.hvac_schedule_mode",
                                       "data": {"option": "manual"}}, "ok")]
            reply = f"{opener('control')}Schedule mode set to manual, Charm."

        elif pat == "set_heat_trigger":
            target = random.choice([17.0, 18.0, 18.5, 19.0, 19.5])
            user_q = random.choice([
                f"set the heat trigger to {target} degrees",
                f"heat trigger {target}", f"downstairs heat threshold {target}",
            ])
            sys_lines = [entity_line("input_number.hvac_heat_trigger_downstairs",
                                     hv["heat_trigger"], "°C")]
            calls = [("call_service", {"domain": "input_number", "service": "set_value",
                                       "entity_id": "input_number.hvac_heat_trigger_downstairs",
                                       "data": {"value": target}}, "ok")]
            reply = f"{opener('control')}Heat trigger set to {target} °C, Charm."

        elif pat == "set_cool_trigger":
            target = random.choice([24.0, 25.0, 26.0, 27.0, 28.0])
            user_q = random.choice([
                f"set the cool trigger to {target} degrees",
                f"cool trigger {target}", f"downstairs cool threshold {target}",
            ])
            sys_lines = [entity_line("input_number.hvac_cool_trigger_downstairs",
                                     hv["cool_trigger"], "°C")]
            calls = [("call_service", {"domain": "input_number", "service": "set_value",
                                       "entity_id": "input_number.hvac_cool_trigger_downstairs",
                                       "data": {"value": target}}, "ok")]
            reply = f"{opener('control')}Cool trigger set to {target} °C, Charm."

        elif pat == "enable_automation":
            user_q = random.choice([
                "enable hvac automation", "turn hvac automation back on",
                "switch on hvac automation",
            ])
            sys_lines = [entity_line("input_boolean.hvac_automation_enabled", "off")]
            calls = [("call_service", {"domain": "input_boolean", "service": "turn_on",
                                       "entity_id": "input_boolean.hvac_automation_enabled"}, "ok")]
            reply = f"{opener('control')}HVAC automation enabled, Charm."

        elif pat == "disable_automation":
            user_q = random.choice([
                "disable hvac automation", "turn hvac automation off",
                "stop the hvac automation",
            ])
            sys_lines = [entity_line("input_boolean.hvac_automation_enabled", "on")]
            calls = [("call_service", {"domain": "input_boolean", "service": "turn_off",
                                       "entity_id": "input_boolean.hvac_automation_enabled"}, "ok")]
            reply = f"{opener('control')}HVAC automation disabled, Charm."

        else:  # set_main_mode_off
            user_q = random.choice([
                "set the main controller to off",
                "main hvac off", "turn the main controller off",
            ])
            sys_lines = [entity_line("climate.izone_controller_402001095", hv["main_mode"])]
            calls = [("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                       "entity_id": "climate.izone_controller_402001095",
                                       "data": {"hvac_mode": "off"}}, "ok")]
            reply = f"{opener('control')}Main controller set to off, Charm."

        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 10 — Weather (60)
# ---------------------------------------------------------------------------
WEATHER_QUERIES = [
    ("what's it like outside", "current"),
    ("how warm is it outside", "current"),
    ("outdoor temperature", "current"),
    ("what's today's forecast", "forecast"),
    ("today's outlook", "forecast"),
    ("will it rain today", "rain_today"),
    ("rain chance today", "rain_today"),
    ("how much rain forecast today", "rain_amount"),
    ("should i bring a jacket", "advice"),
    ("what's the uv like", "uv"),
    ("uv forecast", "uv"),
    ("how hot is it going to get today", "max"),
    ("today's max", "max"),
    ("is there any wind", "wind"),
    ("wind speed and direction", "wind"),
    ("what time does the sun rise today", "sunrise"),
    ("what time does the sun set today", "sunset"),
    ("how much rain fell today", "rain_fallen"),
    ("what's the humidity", "humidity"),
    ("does it feel hotter or colder than the temp says", "feels"),
]

def gen_batch_10_weather(n=60):
    out = []
    for _ in range(n):
        w = randomise_weather()
        q, qtype = random.choice(WEATHER_QUERIES)

        if qtype == "current":
            sys_lines = [entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C")]
            calls = [("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                      str(w["outdoor_temp"]))]
            reply = f"{opener()}It's {w['outdoor_temp']} °C outside, Charm — {w['short_forecast'].lower()}"
        elif qtype == "forecast":
            sys_lines = [
                entity_line("sensor.clyde_north_short_text_0", w["short_forecast"]),
                entity_line("sensor.clyde_north_extended_text_0", w["extended_forecast"]),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.clyde_north_short_text_0"},
                 w["short_forecast"]),
                ("get_entity_state", {"entity_id": "sensor.clyde_north_extended_text_0"},
                 w["extended_forecast"]),
            ]
            reply = f"{opener()}{w['extended_forecast']} Top of {w['max_temp']} °C, Charm."
        elif qtype == "rain_today":
            sys_lines = [
                entity_line("sensor.clyde_north_rain_chance_0", w["rain_chance"], "%"),
                entity_line("sensor.clyde_north_rain_amount_range_0", w["rain_range"]),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.clyde_north_rain_chance_0"},
                 str(w["rain_chance"])),
                ("get_entity_state", {"entity_id": "sensor.clyde_north_rain_amount_range_0"},
                 w["rain_range"]),
            ]
            if w["rain_chance"] >= 70:
                reply = f"{opener()}{w['rain_chance']}% chance of rain, Charm — likely, with {w['rain_range']}."
            elif w["rain_chance"] >= 30:
                reply = f"{opener()}{w['rain_chance']}% chance, Charm — possible, {w['rain_range']}."
            else:
                reply = f"{opener()}{w['rain_chance']}% chance, Charm — unlikely."
        elif qtype == "rain_amount":
            sys_lines = [
                entity_line("sensor.clyde_north_rain_amount_range_0", w["rain_range"]),
                entity_line("sensor.irrigation_rain_next_24h", w["rain_forecast_24h"], "mm"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.clyde_north_rain_amount_range_0"},
                 w["rain_range"]),
                ("get_entity_state", {"entity_id": "sensor.irrigation_rain_next_24h"},
                 str(w["rain_forecast_24h"])),
            ]
            reply = f"{opener()}Range today {w['rain_range']}, {w['rain_forecast_24h']} mm forecast over the next 24 hours, Charm."
        elif qtype == "advice":
            sys_lines = [
                entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                entity_line("sensor.clyde_north_rain_chance_0", w["rain_chance"], "%"),
                entity_line("sensor.home_charm_wind_speed_kilometre", w["wind_kmh"], "km/h"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "sensor.clyde_north_rain_chance_0"},
                 str(w["rain_chance"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_wind_speed_kilometre"},
                 str(w["wind_kmh"])),
            ]
            if w["outdoor_temp"] < 14 or w["wind_kmh"] > 25 or w["rain_chance"] > 60:
                reply = f"{opener()}{w['outdoor_temp']} °C, {w['rain_chance']}% rain, {w['wind_kmh']} km/h wind, Charm — take a jacket."
            else:
                reply = f"{opener()}{w['outdoor_temp']} °C and mild conditions, Charm — a jacket isn't needed."
        elif qtype == "uv":
            sys_lines = [
                entity_line("sensor.clyde_north_uv_category_0", w["uv_category"]),
                entity_line("sensor.clyde_north_uv_forecast_0", w["uv_forecast"]),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.clyde_north_uv_category_0"},
                 w["uv_category"]),
                ("get_entity_state", {"entity_id": "sensor.clyde_north_uv_forecast_0"},
                 w["uv_forecast"]),
            ]
            reply = f"{opener()}{w['uv_category']} UV today, Charm — {w['uv_forecast']}."
        elif qtype == "max":
            sys_lines = [entity_line("sensor.home_charm_max_temp", w["max_temp"], "°C")]
            calls = [("get_entity_state", {"entity_id": "sensor.home_charm_max_temp"},
                      str(w["max_temp"]))]
            reply = f"{opener()}Today's maximum is {w['max_temp']} °C, Charm."
        elif qtype == "wind":
            sys_lines = [
                entity_line("sensor.home_charm_wind_speed_kilometre", w["wind_kmh"], "km/h"),
                entity_line("sensor.home_charm_wind_direction", w["wind_dir"]),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_wind_speed_kilometre"},
                 str(w["wind_kmh"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_wind_direction"},
                 w["wind_dir"]),
            ]
            reply = f"{opener()}{w['wind_kmh']} km/h from the {w['wind_dir']}, Charm."
        elif qtype == "sunrise":
            sys_lines = [entity_line("sensor.clyde_north_astronomical_sunrise_time_0", w["sunrise"])]
            calls = [("get_entity_state",
                      {"entity_id": "sensor.clyde_north_astronomical_sunrise_time_0"},
                      w["sunrise"])]
            reply = f"{opener()}Sunrise at {w['sunrise']}, Charm."
        elif qtype == "sunset":
            sys_lines = [entity_line("sensor.clyde_north_astronomical_sunset_time_0", w["sunset"])]
            calls = [("get_entity_state",
                      {"entity_id": "sensor.clyde_north_astronomical_sunset_time_0"},
                      w["sunset"])]
            reply = f"{opener()}Sunset at {w['sunset']}, Charm."
        elif qtype == "rain_fallen":
            sys_lines = [entity_line("sensor.home_charm_rain_since_9am", w["rain_mm_today"], "mm")]
            calls = [("get_entity_state", {"entity_id": "sensor.home_charm_rain_since_9am"},
                      str(w["rain_mm_today"]))]
            if w["rain_mm_today"] > 0:
                reply = f"{opener()}{w['rain_mm_today']} mm since 9 am, Charm."
            else:
                reply = f"{opener()}Nothing since 9 am, Charm."
        elif qtype == "humidity":
            sys_lines = [entity_line("sensor.home_charm_humidity", w["humidity"], "%")]
            calls = [("get_entity_state", {"entity_id": "sensor.home_charm_humidity"},
                      str(w["humidity"]))]
            reply = f"{opener()}Humidity is {w['humidity']}%, Charm."
        else:  # feels
            sys_lines = [
                entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                entity_line("sensor.home_charm_temp_feels_like", w["feels_like"], "°C"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp_feels_like"},
                 str(w["feels_like"])),
            ]
            diff = round(w["feels_like"] - w["outdoor_temp"], 1)
            if abs(diff) < 1:
                reply = f"{opener()}Temperature and feels-like both around {w['outdoor_temp']} °C, Charm — pretty close."
            elif diff < 0:
                reply = f"{opener()}{w['outdoor_temp']} °C but feels like {w['feels_like']} °C, Charm — colder than the number suggests."
            else:
                reply = f"{opener()}{w['outdoor_temp']} °C but feels like {w['feels_like']} °C, Charm — warmer than the number suggests."

        out.append(multi_call_example(sys_lines, q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 11 — Weather reasoning (40)
# ---------------------------------------------------------------------------
WEATHER_REASONING_QUERIES = [
    "should the irrigation run given today's weather",
    "is it worth opening windows instead of running the ac",
    "what's the fire danger today",
    "based on the weather, should i pre-cool the house",
    "is it a good day to hang washing outside",
    "is it warm enough for shorts today",
    "given the forecast, should i close all the windows",
    "is the weather going to mess with the solar today",
    "should i water the lawn manually today",
    "any reason to plan indoor activities today",
]

def gen_batch_11_weather_reasoning(n=40):
    out = []
    for _ in range(n):
        w = randomise_weather()
        irr = randomise_irrigation(weather=w)
        e = randomise_energy(season=w["season"])
        q = random.choice(WEATHER_REASONING_QUERIES)

        if "irrigation" in q or "water the lawn" in q:
            calls = [
                ("get_entity_state", {"entity_id": "sensor.irrigation_rain_next_24h"},
                 str(w["rain_forecast_24h"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_rain_since_9am"},
                 str(w["rain_mm_today"])),
                ("get_entity_state", {"entity_id": "sensor.irrigation_skip_reason"},
                 irr["skip_reason"]),
                ("get_entity_state", {"entity_id": "sensor.irrigation_season_factor"},
                 str(irr["season_factor"])),
            ]
            sys_lines = [
                entity_line("sensor.irrigation_rain_next_24h", w["rain_forecast_24h"], "mm"),
                entity_line("sensor.home_charm_rain_since_9am", w["rain_mm_today"], "mm"),
                entity_line("sensor.irrigation_skip_reason", irr["skip_reason"]),
                entity_line("sensor.irrigation_season_factor", irr["season_factor"], "x"),
            ]
            if irr["skip_reason"] != "Ready":
                reply = (f"{opener('analysis')}{irr['skip_reason']}, Charm — no need to water manually. "
                         f"Soil moisture should hold with the forecast.")
            elif w["rain_forecast_24h"] >= 5:
                reply = (f"{opener('analysis')}{w['rain_forecast_24h']} mm forecast in 24 h, Charm — "
                         f"natural watering should suffice. Skip the manual run.")
            else:
                reply = (f"{opener('analysis')}Season factor {irr['season_factor']} and only "
                         f"{w['rain_forecast_24h']} mm forecast, Charm — a manual run today is reasonable.")

        elif "open windows" in q or "close all the windows" in q:
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(round(w["outdoor_temp"] + random.uniform(-2, 4), 1))),
                ("get_entity_state", {"entity_id": "sensor.home_charm_wind_speed_kilometre"},
                 str(w["wind_kmh"])),
            ]
            indoor = float(calls[1][2])
            sys_lines = [
                entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                entity_line("sensor.downstairs_zone_temperature", indoor, "°C"),
                entity_line("sensor.home_charm_wind_speed_kilometre", w["wind_kmh"], "km/h"),
            ]
            if "close" in q:
                if w["outdoor_temp"] > indoor + 2 or w["rain_chance"] > 60 or w["wind_kmh"] > 30:
                    reply = (f"{opener('analysis')}Outdoor {w['outdoor_temp']} °C, rain chance {w['rain_chance']}%, "
                             f"wind {w['wind_kmh']} km/h, Charm — yes, close windows.")
                else:
                    reply = (f"{opener('analysis')}Conditions are mild — {w['outdoor_temp']} °C outside, "
                             f"{w['wind_kmh']} km/h wind, Charm. Windows can stay open.")
            else:
                if w["outdoor_temp"] < indoor - 2 and w["wind_kmh"] < 25:
                    reply = (f"{opener('analysis')}Outdoor {w['outdoor_temp']} °C versus indoor {indoor} °C, Charm — "
                             f"opening windows will cool the house at zero cost. Worth doing instead of the AC.")
                else:
                    reply = (f"{opener('analysis')}Outdoor {w['outdoor_temp']} °C versus indoor {indoor} °C, Charm — "
                             f"opening windows won't help. Stick with the AC.")

        elif "fire danger" in q:
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_humidity"},
                 str(w["humidity"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_wind_speed_kilometre"},
                 str(w["wind_kmh"])),
            ]
            sys_lines = [
                entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                entity_line("sensor.home_charm_humidity", w["humidity"], "%"),
                entity_line("sensor.home_charm_wind_speed_kilometre", w["wind_kmh"], "km/h"),
            ]
            score = (w["outdoor_temp"] > 30) + (w["humidity"] < 35) + (w["wind_kmh"] > 25)
            if score >= 2:
                reply = (f"{opener('analysis')}{w['outdoor_temp']} °C, {w['humidity']}% humidity, "
                         f"{w['wind_kmh']} km/h wind, Charm — elevated fire weather. Check the CFA app "
                         f"for the official rating in your district.")
            else:
                reply = (f"{opener('analysis')}{w['outdoor_temp']} °C, {w['humidity']}% humidity, "
                         f"{w['wind_kmh']} km/h wind, Charm — low fire weather risk today.")

        elif "pre-cool" in q or "precool" in q:
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_max_temp"},
                 str(w["max_temp"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                 str(e["buy_price"])),
            ]
            sys_lines = [
                entity_line("sensor.home_charm_max_temp", w["max_temp"], "°C"),
                entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
                entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
            ]
            if w["max_temp"] >= 30 and e["battery_soc"] > 50 and e["buy_cents"] < 25:
                reply = (f"{opener('analysis')}Top of {w['max_temp']} °C, battery {e['battery_soc']}%, "
                         f"price {e['buy_cents']} c/kWh, Charm — yes, pre-cool while it's cheap.")
            elif w["max_temp"] < 28:
                reply = (f"{opener('analysis')}Max only {w['max_temp']} °C, Charm — pre-cooling isn't worth it today.")
            else:
                reply = (f"{opener('analysis')}Conditions are borderline, Charm — max {w['max_temp']} °C, "
                         f"battery {e['battery_soc']}%, price {e['buy_cents']} c/kWh. Pre-cool only if you'll "
                         f"actually be home for the peak.")

        elif "hang washing" in q:
            calls = [
                ("get_entity_state", {"entity_id": "sensor.clyde_north_rain_chance_0"},
                 str(w["rain_chance"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_humidity"},
                 str(w["humidity"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_wind_speed_kilometre"},
                 str(w["wind_kmh"])),
            ]
            sys_lines = [
                entity_line("sensor.clyde_north_rain_chance_0", w["rain_chance"], "%"),
                entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                entity_line("sensor.home_charm_humidity", w["humidity"], "%"),
                entity_line("sensor.home_charm_wind_speed_kilometre", w["wind_kmh"], "km/h"),
            ]
            if w["rain_chance"] < 25 and w["outdoor_temp"] > 18 and w["humidity"] < 75:
                reply = f"{opener('analysis')}{w['rain_chance']}% rain, {w['outdoor_temp']} °C, {w['wind_kmh']} km/h wind, Charm — solid drying day."
            elif w["rain_chance"] > 50:
                reply = f"{opener('analysis')}{w['rain_chance']}% rain chance, Charm — don't risk the line today."
            else:
                reply = (f"{opener('analysis')}Conditions mixed, Charm — {w['rain_chance']}% rain, "
                         f"{w['humidity']}% humidity. Workable but watch for showers.")

        elif "shorts" in q or "warm enough" in q:
            sys_lines = [entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                         entity_line("sensor.home_charm_max_temp", w["max_temp"], "°C")]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_max_temp"},
                 str(w["max_temp"])),
            ]
            if w["max_temp"] >= 22:
                reply = f"{opener('analysis')}Top of {w['max_temp']} °C, Charm — shorts territory."
            else:
                reply = f"{opener('analysis')}Top of {w['max_temp']} °C, Charm — bit cool for shorts."

        elif "solar today" in q or "mess with the solar" in q:
            sys_lines = [
                entity_line("sensor.clyde_north_short_text_0", w["short_forecast"]),
                entity_line("sensor.clyde_north_rain_chance_0", w["rain_chance"], "%"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.clyde_north_short_text_0"},
                 w["short_forecast"]),
                ("get_entity_state", {"entity_id": "sensor.clyde_north_rain_chance_0"},
                 str(w["rain_chance"])),
            ]
            if "rain" in w["short_forecast"].lower() or "cloud" in w["short_forecast"].lower() or w["rain_chance"] > 60:
                reply = (f"{opener('analysis')}{w['short_forecast']} {w['rain_chance']}% rain, Charm — "
                         f"solar yield will be down meaningfully today.")
            else:
                reply = (f"{opener('analysis')}{w['short_forecast']} {w['rain_chance']}% rain, Charm — "
                         f"solar should run close to clear-sky.")

        else:  # indoor activities
            sys_lines = [
                entity_line("sensor.clyde_north_rain_chance_0", w["rain_chance"], "%"),
                entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                entity_line("sensor.home_charm_wind_speed_kilometre", w["wind_kmh"], "km/h"),
            ]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.clyde_north_rain_chance_0"},
                 str(w["rain_chance"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_wind_speed_kilometre"},
                 str(w["wind_kmh"])),
            ]
            if w["rain_chance"] > 60 or w["wind_kmh"] > 35 or w["outdoor_temp"] < 10 or w["outdoor_temp"] > 35:
                reply = (f"{opener('analysis')}{w['rain_chance']}% rain, {w['outdoor_temp']} °C, "
                         f"{w['wind_kmh']} km/h wind, Charm — indoor day is the smart call.")
            else:
                reply = (f"{opener('analysis')}{w['outdoor_temp']} °C and {w['rain_chance']}% rain, Charm — "
                         f"outdoor plans are fine.")

        out.append(multi_call_example(sys_lines, q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 12 — Irrigation status (40)
# ---------------------------------------------------------------------------
IRRIGATION_STATUS_QUERIES = [
    ("did irrigation run today", "ran_today"),
    ("will the sprinklers run tomorrow", "tomorrow"),
    ("why is irrigation being skipped", "skip"),
    ("how much rain fell today", "rain"),
    ("is there enough rain forecast to skip watering", "skip_rain"),
    ("what's the season factor at", "season"),
    ("which zones are due today", "zones"),
    ("how long will irrigation run tonight", "minutes"),
    ("is the irrigation master enabled", "master"),
    ("irrigation skip reason", "skip"),
]

def gen_batch_12_irrigation_status(n=40):
    out = []
    for _ in range(n):
        w = randomise_weather()
        irr = randomise_irrigation(weather=w)
        q, qtype = random.choice(IRRIGATION_STATUS_QUERIES)

        if qtype == "ran_today":
            sys_lines = [entity_line("sensor.irrigation_skip_reason", irr["skip_reason"]),
                         entity_line("sensor.irrigation_total_planned_minutes",
                                     irr["planned_minutes"], "min")]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.irrigation_skip_reason"},
                 irr["skip_reason"]),
                ("get_entity_state", {"entity_id": "sensor.irrigation_total_planned_minutes"},
                 str(irr["planned_minutes"])),
            ]
            if irr["skip_reason"] == "Ready" and irr["planned_minutes"] > 0:
                reply = f"{opener()}Scheduled for {irr['planned_minutes']} minutes today, Charm."
            else:
                reply = f"{opener()}No watering today, Charm — {irr['skip_reason']}."
        elif qtype == "tomorrow":
            sys_lines = [entity_line("sensor.irrigation_season_factor", irr["season_factor"], "x"),
                         entity_line("sensor.irrigation_rain_next_24h", w["rain_forecast_24h"], "mm")]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.irrigation_season_factor"},
                 str(irr["season_factor"])),
                ("get_entity_state", {"entity_id": "sensor.irrigation_rain_next_24h"},
                 str(w["rain_forecast_24h"])),
            ]
            if w["rain_forecast_24h"] >= 8:
                reply = (f"{opener()}Likely skipped tomorrow, Charm — {w['rain_forecast_24h']} mm forecast "
                         f"will cover the deficit.")
            elif irr["season_factor"] < 0.25:
                reply = (f"{opener()}Likely skipped tomorrow, Charm — season factor {irr['season_factor']} "
                         f"is below threshold.")
            else:
                reply = (f"{opener()}Likely a run tomorrow, Charm — season factor {irr['season_factor']}, "
                         f"only {w['rain_forecast_24h']} mm forecast.")
        elif qtype == "skip" or qtype == "skip_rain":
            sys_lines = [entity_line("sensor.irrigation_skip_reason", irr["skip_reason"]),
                         entity_line("sensor.irrigation_rain_next_24h", w["rain_forecast_24h"], "mm")]
            calls = [
                ("get_entity_state", {"entity_id": "sensor.irrigation_skip_reason"},
                 irr["skip_reason"]),
                ("get_entity_state", {"entity_id": "sensor.irrigation_rain_next_24h"},
                 str(w["rain_forecast_24h"])),
            ]
            if irr["skip_reason"] == "Ready":
                reply = (f"{opener()}Not being skipped, Charm — status is Ready and "
                         f"{w['rain_forecast_24h']} mm forecast next 24 h.")
            else:
                reply = f"{opener()}Skip reason: {irr['skip_reason']}, Charm."
        elif qtype == "rain":
            sys_lines = [entity_line("sensor.home_charm_rain_since_9am", w["rain_mm_today"], "mm")]
            calls = [("get_entity_state", {"entity_id": "sensor.home_charm_rain_since_9am"},
                      str(w["rain_mm_today"]))]
            if w["rain_mm_today"] > 0:
                reply = f"{opener()}{w['rain_mm_today']} mm since 9 am, Charm."
            else:
                reply = f"{opener()}No rain since 9 am, Charm."
        elif qtype == "season":
            sys_lines = [entity_line("sensor.irrigation_season_factor", irr["season_factor"], "x")]
            calls = [("get_entity_state", {"entity_id": "sensor.irrigation_season_factor"},
                      str(irr["season_factor"]))]
            reply = f"{opener()}Season factor is {irr['season_factor']}, Charm."
        elif qtype == "zones":
            zones_due_calls = []
            for z in range(1, 6):
                zones_due_calls.append(
                    ("get_entity_state",
                     {"entity_id": f"binary_sensor.irrigation_zone_{z}_due_today"},
                     "on" if z in irr["zones_due"] else "off")
                )
            sys_lines = [entity_line(f"binary_sensor.irrigation_zone_{z}_due_today",
                                      "on" if z in irr["zones_due"] else "off")
                         for z in range(1, 6)]
            calls = zones_due_calls
            if irr["zones_due"]:
                z_str = ", ".join(str(z) for z in irr["zones_due"])
                reply = f"{opener()}Zones due today: {z_str}, Charm."
            else:
                reply = f"{opener()}No zones due today, Charm."
        elif qtype == "minutes":
            sys_lines = [entity_line("sensor.irrigation_total_planned_minutes",
                                     irr["planned_minutes"], "min")]
            calls = [("get_entity_state", {"entity_id": "sensor.irrigation_total_planned_minutes"},
                      str(irr["planned_minutes"]))]
            if irr["planned_minutes"] > 0:
                reply = f"{opener()}{irr['planned_minutes']} minutes scheduled tonight, Charm."
            else:
                reply = f"{opener()}Nothing scheduled tonight, Charm."
        else:  # master
            sys_lines = [entity_line("input_boolean.irrigation_master_enable",
                                     "on" if irr["master_enabled"] else "off")]
            calls = [("get_entity_state", {"entity_id": "input_boolean.irrigation_master_enable"},
                      "on" if irr["master_enabled"] else "off")]
            reply = f"{opener()}Master is {'enabled' if irr['master_enabled'] else 'disabled'}, Charm."

        out.append(multi_call_example(sys_lines, q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 13 — Irrigation reasoning (30)
# ---------------------------------------------------------------------------
IRRIGATION_REASONING_QUERIES = [
    "should i water the garden manually today",
    "the lawn looks dry — should i override the skip",
    "explain the irrigation schedule",
    "given 12mm of rain forecast, will the system skip",
    "is the season factor correct for now",
    "is the irrigation logic being too cautious",
    "what's preventing irrigation from running",
]

def gen_batch_13_irrigation_reasoning(n=30):
    out = []
    for _ in range(n):
        w = randomise_weather()
        irr = randomise_irrigation(weather=w)
        q = random.choice(IRRIGATION_REASONING_QUERIES)

        calls = [
            ("get_entity_state", {"entity_id": "sensor.irrigation_skip_reason"},
             irr["skip_reason"]),
            ("get_entity_state", {"entity_id": "sensor.irrigation_season_factor"},
             str(irr["season_factor"])),
            ("get_entity_state", {"entity_id": "sensor.irrigation_rain_next_24h"},
             str(w["rain_forecast_24h"])),
            ("get_entity_state", {"entity_id": "sensor.home_charm_rain_since_9am"},
             str(w["rain_mm_today"])),
        ]
        sys_lines = [
            entity_line("sensor.irrigation_skip_reason", irr["skip_reason"]),
            entity_line("sensor.irrigation_season_factor", irr["season_factor"], "x"),
            entity_line("sensor.irrigation_rain_next_24h", w["rain_forecast_24h"], "mm"),
            entity_line("sensor.home_charm_rain_since_9am", w["rain_mm_today"], "mm"),
        ]

        if "water the garden manually" in q or "looks dry" in q or "override" in q:
            if w["rain_mm_today"] >= 5 or w["rain_forecast_24h"] >= 8:
                reply = (f"{opener('analysis')}{w['rain_mm_today']} mm fallen, {w['rain_forecast_24h']} mm "
                         f"forecast, Charm — overriding the skip will waste water. Let the rain do it.")
            elif irr["season_factor"] < 0.25 and irr["skip_reason"] != "Ready":
                reply = (f"{opener('analysis')}Season factor {irr['season_factor']} keeps the system off, "
                         f"Charm. If beds genuinely look dry, run a single zone manually rather than the full cycle.")
            else:
                reply = (f"{opener('analysis')}Conditions support a run, Charm — season factor {irr['season_factor']}, "
                         f"only {w['rain_forecast_24h']} mm forecast. Trigger the full sequential cycle.")
        elif "explain the irrigation schedule" in q:
            reply = (f"{opener('analysis')}Schedule today, Charm: status {irr['skip_reason']}, season factor "
                     f"{irr['season_factor']}, planned {irr['planned_minutes']} minutes, "
                     f"{w['rain_forecast_24h']} mm forecast next 24 h. The system trims runtime by the season "
                     f"factor and skips entirely if rainfall thresholds are met.")
        elif "12mm" in q:
            reply = (f"{opener('analysis')}Yes, Charm — 12 mm forecast comfortably exceeds the skip threshold, "
                     f"so the system will skip. Current 24 h forecast is {w['rain_forecast_24h']} mm, "
                     f"already enough to suppress watering.")
        elif "season factor correct" in q:
            month_expected = {
                "summer": 0.95, "autumn": 0.45, "winter": 0.2, "spring": 0.7
            }[w["season"]]
            if abs(irr["season_factor"] - month_expected) < 0.15:
                reply = (f"{opener('analysis')}Season factor {irr['season_factor']} looks right for {w['season']}, "
                         f"Charm — close to the expected {month_expected}.")
            else:
                reply = (f"{opener('analysis')}Season factor {irr['season_factor']} is off the expected "
                         f"{month_expected} for {w['season']}, Charm — worth checking the curve.")
        elif "too cautious" in q:
            if w["rain_mm_today"] < 1 and w["rain_forecast_24h"] < 3 and irr["skip_reason"] != "Ready":
                reply = (f"{opener('analysis')}Possibly, Charm — only {w['rain_mm_today']} mm fallen and "
                         f"{w['rain_forecast_24h']} mm forecast, yet status is {irr['skip_reason']}. Worth a manual review.")
            else:
                reply = (f"{opener('analysis')}Logic looks reasonable, Charm — {w['rain_mm_today']} mm fallen, "
                         f"{w['rain_forecast_24h']} mm forecast, season factor {irr['season_factor']}.")
        else:  # what's preventing irrigation
            if irr["skip_reason"] == "Ready":
                reply = (f"{opener('analysis')}Nothing — status is Ready, Charm. If it isn't running, check the master switch.")
            else:
                reply = f"{opener('analysis')}{irr['skip_reason']}, Charm — that's the active block."

        out.append(multi_call_example(sys_lines, q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 14 — Irrigation control (20)
# ---------------------------------------------------------------------------
def gen_batch_14_irrigation_control(n=20):
    out = []
    actions = ["run_all", "open_zone", "close_zone", "close_all", "disable_master", "enable_master"]
    for _ in range(n):
        action = random.choice(actions)
        if action == "run_all":
            user_q = random.choice([
                "run all irrigation zones now", "trigger a manual full cycle",
                "run the irrigation sequence", "manual irrigation now",
            ])
            sys_lines = [entity_line("script.irrigation_run_sequential", "off")]
            calls = [("call_service", {"domain": "script", "service": "turn_on",
                                       "entity_id": "script.irrigation_run_sequential"}, "ok")]
            reply = f"{opener('control')}Sequential irrigation cycle started, Charm."
        elif action == "open_zone":
            z = random.randint(1, 8)
            user_q = random.choice([
                f"open station {z}", f"open zone {z}", f"turn on irrigation zone {z}",
                f"run station {z} for a bit",
            ])
            sys_lines = [entity_line(f"valve.station_{z}", "closed")]
            calls = [("call_service", {"domain": "valve", "service": "open_valve",
                                       "entity_id": f"valve.station_{z}"}, "ok")]
            reply = f"{opener('control')}Station {z} opened, Charm — close it manually when done."
        elif action == "close_zone":
            z = random.randint(1, 8)
            user_q = random.choice([
                f"close station {z}", f"close zone {z}", f"turn off irrigation zone {z}",
            ])
            sys_lines = [entity_line(f"valve.station_{z}", "open")]
            calls = [("call_service", {"domain": "valve", "service": "close_valve",
                                       "entity_id": f"valve.station_{z}"}, "ok")]
            reply = f"{opener('control')}Station {z} closed, Charm."
        elif action == "close_all":
            user_q = random.choice([
                "close all irrigation zones", "shut all stations",
                "stop all irrigation",
            ])
            sys_lines = [entity_line(f"valve.station_{z}", "open") for z in range(1, 9)]
            calls = []
            for z in range(1, 9):
                calls.append(("call_service",
                             {"domain": "valve", "service": "close_valve",
                              "entity_id": f"valve.station_{z}"}, "ok"))
            reply = f"{opener('control')}All eight stations closed, Charm."
        elif action == "disable_master":
            user_q = random.choice([
                "disable irrigation master", "turn off the master irrigation switch",
                "kill the irrigation master",
            ])
            sys_lines = [entity_line("input_boolean.irrigation_master_enable", "on")]
            calls = [("call_service", {"domain": "input_boolean", "service": "turn_off",
                                       "entity_id": "input_boolean.irrigation_master_enable"}, "ok")]
            reply = f"{opener('control')}Master irrigation disabled, Charm — nothing will fire until re-enabled."
        else:
            user_q = random.choice([
                "enable irrigation master", "switch the master irrigation back on",
            ])
            sys_lines = [entity_line("input_boolean.irrigation_master_enable", "off")]
            calls = [("call_service", {"domain": "input_boolean", "service": "turn_on",
                                       "entity_id": "input_boolean.irrigation_master_enable"}, "ok")]
            reply = f"{opener('control')}Master irrigation enabled, Charm."

        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 15 — Tesla queries (50)
# ---------------------------------------------------------------------------
TESLA_QUERY_TYPES = [
    ("how charged is the car", "battery"),
    ("car battery level", "battery"),
    ("is the tesla plugged in", "plugged"),
    ("where's the car", "location"),
    ("is the car at home", "location"),
    ("is the charge port open", "port"),
    ("what's the charge limit set to", "limit"),
    ("is the frunk open", "frunk"),
    ("is the trunk open", "trunk"),
    ("are the doors locked", "lock"),
    ("is the car locked", "lock"),
    ("what's the cabin temperature set to", "set_temp"),
    ("what's the cabin temperature inside", "cabin_temp"),
    ("is the cabin climate on", "climate"),
    ("is the charge cable locked", "cable_lock"),
    ("is the driver door open", "driver_door"),
    ("charging state of the car", "charging"),
    ("left seat heater", "seat_left"),
    ("right seat heater", "seat_right"),
]

def gen_batch_15_tesla_queries(n=50):
    out = []
    for _ in range(n):
        t = randomise_tesla()
        q, qtype = random.choice(TESLA_QUERY_TYPES)

        # Sleeping car handler
        if not t["available"]:
            # Choose an entity relevant to the query
            mapping = {
                "battery": ("device_tracker.white_python_location",),
                "plugged": ("lock.white_python_charge_cable_lock",),
                "location": ("device_tracker.white_python_location",),
                "port": ("cover.white_python_charge_port_door",),
                "limit": ("number.white_python_charge_limit",),
                "frunk": ("cover.white_python_frunk",),
                "trunk": ("cover.white_python_trunk",),
                "lock": ("lock.white_python_lock",),
                "set_temp": ("climate.white_python_climate",),
                "cabin_temp": ("climate.white_python_climate",),
                "climate": ("climate.white_python_climate",),
                "cable_lock": ("lock.white_python_charge_cable_lock",),
                "driver_door": ("binary_sensor.white_python_front_driver_door",),
                "charging": ("climate.white_python_climate",),
                "seat_left": ("select.white_python_seat_heater_front_left",),
                "seat_right": ("select.white_python_seat_heater_front_right",),
            }
            eid = mapping[qtype][0]
            sys_lines = [entity_line(eid, "unavailable")]
            calls = [("get_entity_state", {"entity_id": eid}, "unavailable")]
            reply = "The car is sleeping, Charm. Open the Tesla app to wake it."
            out.append(multi_call_example(sys_lines, q, calls, reply))
            continue

        if qtype == "battery":
            sys_lines = [entity_line("number.white_python_charge_limit", t["charge_limit"], "%")]
            calls = [("get_entity_state",
                      {"entity_id": "number.white_python_charge_limit",
                       "attribute": "current_value"}, str(t["battery_level"]))]
            reply = f"{opener()}Car is at {t['battery_level']}%, Charm — limit set to {t['charge_limit']}%."
        elif qtype == "plugged":
            sys_lines = [entity_line("lock.white_python_charge_cable_lock", t["cable_lock"])]
            calls = [("get_entity_state",
                      {"entity_id": "lock.white_python_charge_cable_lock"}, t["cable_lock"])]
            if t["cable_lock"] == "locked":
                reply = f"{opener()}Cable lock is engaged, Charm — plugged in."
            else:
                reply = f"{opener()}Cable lock is open, Charm — not plugged in."
        elif qtype == "location":
            sys_lines = [entity_line("device_tracker.white_python_location", t["location"])]
            calls = [("get_entity_state", {"entity_id": "device_tracker.white_python_location"},
                      t["location"])]
            if t["location"] == "home":
                reply = f"{opener()}Car is at home, Charm."
            else:
                reply = f"{opener()}Car is away from home, Charm."
        elif qtype == "port":
            sys_lines = [entity_line("cover.white_python_charge_port_door", t["port_door"])]
            calls = [("get_entity_state", {"entity_id": "cover.white_python_charge_port_door"},
                      t["port_door"])]
            reply = f"{opener()}Charge port is {t['port_door']}, Charm."
        elif qtype == "limit":
            sys_lines = [entity_line("number.white_python_charge_limit", t["charge_limit"], "%")]
            calls = [("get_entity_state", {"entity_id": "number.white_python_charge_limit"},
                      str(t["charge_limit"]))]
            reply = f"{opener()}Charge limit set to {t['charge_limit']}%, Charm."
        elif qtype == "frunk":
            sys_lines = [entity_line("cover.white_python_frunk", t["frunk"])]
            calls = [("get_entity_state", {"entity_id": "cover.white_python_frunk"}, t["frunk"])]
            reply = f"{opener()}Frunk is {t['frunk']}, Charm."
        elif qtype == "trunk":
            sys_lines = [entity_line("cover.white_python_trunk", t["trunk"])]
            calls = [("get_entity_state", {"entity_id": "cover.white_python_trunk"}, t["trunk"])]
            reply = f"{opener()}Trunk is {t['trunk']}, Charm."
        elif qtype == "lock":
            sys_lines = [entity_line("lock.white_python_lock", t["lock"])]
            calls = [("get_entity_state", {"entity_id": "lock.white_python_lock"}, t["lock"])]
            reply = f"{opener()}Car is {t['lock']}, Charm."
        elif qtype == "set_temp":
            sys_lines = [entity_line("climate.white_python_climate", t["climate_on"])]
            calls = [("get_entity_state",
                      {"entity_id": "climate.white_python_climate",
                       "attribute": "temperature"}, str(t["set_temp"]))]
            reply = f"{opener()}Cabin setpoint is {t['set_temp']} °C, Charm."
        elif qtype == "cabin_temp":
            sys_lines = [entity_line("climate.white_python_climate", t["climate_on"])]
            calls = [("get_entity_state",
                      {"entity_id": "climate.white_python_climate",
                       "attribute": "current_temperature"}, str(t["cabin_temp"]))]
            reply = f"{opener()}Cabin is at {t['cabin_temp']} °C, Charm."
        elif qtype == "climate":
            sys_lines = [entity_line("climate.white_python_climate", t["climate_on"])]
            calls = [("get_entity_state", {"entity_id": "climate.white_python_climate"},
                      t["climate_on"])]
            reply = f"{opener()}Cabin climate is {t['climate_on']}, Charm."
        elif qtype == "cable_lock":
            sys_lines = [entity_line("lock.white_python_charge_cable_lock", t["cable_lock"])]
            calls = [("get_entity_state", {"entity_id": "lock.white_python_charge_cable_lock"},
                      t["cable_lock"])]
            reply = f"{opener()}Cable lock is {t['cable_lock']}, Charm."
        elif qtype == "driver_door":
            sys_lines = [entity_line("binary_sensor.white_python_front_driver_door", t["driver_door"])]
            calls = [("get_entity_state",
                      {"entity_id": "binary_sensor.white_python_front_driver_door"},
                      t["driver_door"])]
            reply = f"{opener()}Driver door is {t['driver_door']}, Charm."
        elif qtype == "charging":
            sys_lines = [entity_line("lock.white_python_charge_cable_lock", t["cable_lock"]),
                         entity_line("cover.white_python_charge_port_door", t["port_door"])]
            calls = [
                ("get_entity_state", {"entity_id": "lock.white_python_charge_cable_lock"},
                 t["cable_lock"]),
                ("get_entity_state", {"entity_id": "cover.white_python_charge_port_door"},
                 t["port_door"]),
            ]
            if t["cable_lock"] == "locked":
                reply = f"{opener()}Cable engaged, port {t['port_door']}, Charm — charging state {t['charging_state']}."
            else:
                reply = f"{opener()}Not plugged in, Charm — port {t['port_door']}."
        elif qtype == "seat_left":
            sys_lines = [entity_line("select.white_python_seat_heater_front_left", t["seat_left"])]
            calls = [("get_entity_state",
                      {"entity_id": "select.white_python_seat_heater_front_left"}, t["seat_left"])]
            reply = f"{opener()}Left seat heater is {t['seat_left']}, Charm."
        else:  # seat_right
            sys_lines = [entity_line("select.white_python_seat_heater_front_right", t["seat_right"])]
            calls = [("get_entity_state",
                      {"entity_id": "select.white_python_seat_heater_front_right"}, t["seat_right"])]
            reply = f"{opener()}Right seat heater is {t['seat_right']}, Charm."

        out.append(multi_call_example(sys_lines, q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 16 — Tesla reasoning (30)
# ---------------------------------------------------------------------------
TESLA_REASONING_QUERIES = [
    "should i charge the car now given the amber price",
    "is it worth charging the car tonight or waiting for cheap window",
    "how long until the car is fully charged",
    "the car is at 60 percent, is that enough for work tomorrow",
    "explain the car charging strategy",
    "should the car start charging now or wait for solar",
    "is there enough battery in the car for a trip to the cbd",
]

def gen_batch_16_tesla_reasoning(n=30):
    out = []
    for _ in range(n):
        t = randomise_tesla()
        e = randomise_energy()
        timeline = randomise_amber_timeline(e["time"][0], e["time"][1], hours=8)
        q = random.choice(TESLA_REASONING_QUERIES)

        if not t["available"]:
            sys_lines = [entity_line("device_tracker.white_python_location", "unavailable")]
            calls = [("get_entity_state", {"entity_id": "device_tracker.white_python_location"},
                      "unavailable")]
            reply = "The car is sleeping, Charm. Open the Tesla app to wake it."
            out.append(multi_call_example(sys_lines, q, calls, reply))
            continue

        calls = [
            ("get_entity_state",
             {"entity_id": "number.white_python_charge_limit",
              "attribute": "current_value"}, str(t["battery_level"])),
            ("get_entity_state", {"entity_id": "number.white_python_charge_limit"},
             str(t["charge_limit"])),
            ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
             str(e["buy_price"])),
            ("get_entity_state", {"entity_id": "sensor.amber_price_timeline",
                                  "attribute": "data"}, json.dumps(timeline)),
        ]
        sys_lines = [
            entity_line("number.white_python_charge_limit", t["charge_limit"], "%"),
            entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
            entity_line("sensor.amber_price_timeline", str(e["buy_price"]),
                        friendly="Amber Price Timeline"),
        ]
        future_min = min(round(item["price_per_kwh"] * 100) for item in timeline)

        if "charge the car now" in q or "amber price" in q or "should the car start charging now" in q:
            if e["spike_active"]:
                reply = (f"Spike active at {e['buy_cents']} c/kWh, Charm — don't charge now. "
                         f"Wait, cheapest window in the next 8 h is {future_min} c/kWh.")
            elif e["buy_cents"] <= 12 and t["battery_level"] < t["charge_limit"] - 5:
                reply = (f"Yes, Charm — buy price {e['buy_cents']} c/kWh with the car at {t['battery_level']}% "
                         f"and limit {t['charge_limit']}% is a good fill window.")
            elif future_min < e["buy_cents"] - 5:
                reply = (f"Wait, Charm — current {e['buy_cents']} c/kWh is above the forecast trough of "
                         f"{future_min} c/kWh. Cheaper window coming.")
            else:
                reply = (f"{e['buy_cents']} c/kWh, no cheap dip forecast (low {future_min} c/kWh), Charm — "
                         f"acceptable to charge now if you need range tomorrow.")
        elif "tonight or waiting for cheap window" in q:
            if future_min < e["buy_cents"]:
                reply = (f"{opener('analysis')}Cheaper window forecast at {future_min} c/kWh versus current "
                         f"{e['buy_cents']} c/kWh, Charm — schedule for that window instead.")
            else:
                reply = (f"{opener('analysis')}No cheaper window in the next 8 h, Charm — charge now at "
                         f"{e['buy_cents']} c/kWh.")
        elif "how long until" in q or "fully charged" in q:
            pct_to_go = max(0, t["charge_limit"] - t["battery_level"])
            kwh_to_go = round(pct_to_go * 0.75, 1)  # ~75 kWh Model 3
            charge_kw = random.choice([2.0, 7.2, 11.0])
            hrs = round(kwh_to_go / charge_kw, 1)
            reply = (f"{opener('analysis')}Car at {t['battery_level']}%, target {t['charge_limit']}%, "
                     f"Charm — roughly {kwh_to_go} kWh to add. At {charge_kw} kW that's about {hrs} hours.")
        elif "60 percent" in q or "60%" in q or "enough for work" in q:
            reply = (f"{opener('analysis')}60% on a Model 3 is ~280 km of range, Charm — well over a typical "
                     f"commute to the CBD and back. Comfortable for tomorrow.")
        elif "explain the car charging strategy" in q:
            reply = (f"{opener('analysis')}Default strategy, Charm: trickle from solar when SoC < 30%, "
                     f"force-charge when Amber < 10 c/kWh, otherwise schedule overnight in the cheapest "
                     f"5-minute window. Current state: car {t['battery_level']}%, limit {t['charge_limit']}%, "
                     f"buy {e['buy_cents']} c/kWh, forecast low {future_min} c/kWh.")
        elif "wait for solar" in q:
            if e["is_day"] and e["solar_kw"] > 3:
                reply = (f"{opener('analysis')}Solar at {e['solar_kw']} kW, Charm — start the charge, it'll "
                         f"largely be on-array energy.")
            else:
                reply = (f"{opener('analysis')}Solar only {e['solar_kw']} kW, Charm — wait for midday or "
                         f"use a cheap grid window. Forecast low {future_min} c/kWh.")
        else:  # CBD trip
            if t["battery_level"] >= 40:
                reply = (f"Yes, Charm — {t['battery_level']}% gives roughly {int(t['battery_level']*4.7)} km, "
                         f"comfortable for a CBD return.")
            else:
                reply = (f"Borderline, Charm — {t['battery_level']}% (~{int(t['battery_level']*4.7)} km) "
                         f"covers it one way, but you'll want to top up before returning.")

        out.append(multi_call_example(sys_lines, q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 17 — Tesla control (30)
# ---------------------------------------------------------------------------
def gen_batch_17_tesla_control(n=30):
    out = []
    actions = ["climate_on", "climate_off", "set_temp", "lock", "unlock",
               "set_limit", "seat_left", "seat_right", "port_close",
               "port_open", "precondition"]
    for _ in range(n):
        action = random.choice(actions)
        t = randomise_tesla()
        if not t["available"]:
            user_q = random.choice(["turn on the car heating", "lock the car",
                                    "precondition the car", "set the cabin to 22"])
            sys_lines = [entity_line("climate.white_python_climate", "unavailable")]
            calls = [("get_entity_state", {"entity_id": "climate.white_python_climate"},
                      "unavailable")]
            reply = "The car is sleeping, Charm. Open the Tesla app to wake it."
            out.append(multi_call_example(sys_lines, user_q, calls, reply))
            continue

        if action == "climate_on":
            user_q = random.choice([
                "turn on the car heating", "turn on the cabin climate",
                "start the cabin climate", "ac on in the car",
            ])
            sys_lines = [entity_line("climate.white_python_climate", t["climate_on"])]
            calls = [("call_service", {"domain": "climate", "service": "turn_on",
                                       "entity_id": "climate.white_python_climate"}, "ok")]
            reply = f"{opener('control')}Cabin climate on, Charm."
        elif action == "climate_off":
            user_q = random.choice([
                "turn off the cabin climate", "stop the car climate",
                "turn off the car ac",
            ])
            sys_lines = [entity_line("climate.white_python_climate", t["climate_on"])]
            calls = [("call_service", {"domain": "climate", "service": "turn_off",
                                       "entity_id": "climate.white_python_climate"}, "ok")]
            reply = f"{opener('control')}Cabin climate off, Charm."
        elif action == "set_temp":
            target = random.choice([18, 19, 20, 21, 22, 23, 24])
            user_q = random.choice([
                f"set the cabin to {target} degrees", f"cabin to {target}",
                f"set the car climate to {target}°C",
            ])
            sys_lines = [entity_line("climate.white_python_climate", t["climate_on"])]
            calls = [("call_service", {"domain": "climate", "service": "set_temperature",
                                       "entity_id": "climate.white_python_climate",
                                       "data": {"temperature": target}}, "ok")]
            reply = f"{opener('control')}Cabin setpoint {target} °C, Charm."
        elif action == "lock":
            user_q = random.choice([
                "lock the car", "lock the tesla", "secure the car",
            ])
            sys_lines = [entity_line("lock.white_python_lock", t["lock"])]
            calls = [("call_service", {"domain": "lock", "service": "lock",
                                       "entity_id": "lock.white_python_lock"}, "ok")]
            reply = f"{opener('control')}Car locked, Charm."
        elif action == "unlock":
            user_q = random.choice([
                "unlock the car", "unlock the tesla",
            ])
            sys_lines = [entity_line("lock.white_python_lock", t["lock"])]
            calls = [("call_service", {"domain": "lock", "service": "unlock",
                                       "entity_id": "lock.white_python_lock"}, "ok")]
            reply = f"{opener('control')}Car unlocked, Charm."
        elif action == "set_limit":
            target = random.choice([70, 80, 85, 90, 95, 100])
            user_q = random.choice([
                f"set charge limit to {target}%", f"charge limit {target}",
                f"set the car charge cap to {target}",
            ])
            sys_lines = [entity_line("number.white_python_charge_limit", t["charge_limit"], "%")]
            calls = [("call_service", {"domain": "number", "service": "set_value",
                                       "entity_id": "number.white_python_charge_limit",
                                       "data": {"value": target}}, "ok")]
            reply = f"{opener('control')}Charge limit set to {target}%, Charm."
        elif action == "seat_left":
            level = random.choice(["off", "low", "medium", "high"])
            user_q = random.choice([
                f"heat the left seat to {level}", f"left seat heater {level}",
                f"driver seat heat {level}",
            ])
            sys_lines = [entity_line("select.white_python_seat_heater_front_left", t["seat_left"])]
            calls = [("call_service", {"domain": "select", "service": "select_option",
                                       "entity_id": "select.white_python_seat_heater_front_left",
                                       "data": {"option": level}}, "ok")]
            reply = f"{opener('control')}Left seat heater set to {level}, Charm."
        elif action == "seat_right":
            level = random.choice(["off", "low", "medium", "high"])
            user_q = random.choice([
                f"heat the right seat to {level}", f"right seat heater {level}",
                f"passenger seat heat {level}",
            ])
            sys_lines = [entity_line("select.white_python_seat_heater_front_right", t["seat_right"])]
            calls = [("call_service", {"domain": "select", "service": "select_option",
                                       "entity_id": "select.white_python_seat_heater_front_right",
                                       "data": {"option": level}}, "ok")]
            reply = f"{opener('control')}Right seat heater set to {level}, Charm."
        elif action == "port_close":
            user_q = random.choice([
                "close the charge port", "shut the charge port",
                "close the charge port door",
            ])
            sys_lines = [entity_line("cover.white_python_charge_port_door", t["port_door"])]
            calls = [("call_service", {"domain": "cover", "service": "close_cover",
                                       "entity_id": "cover.white_python_charge_port_door"}, "ok")]
            reply = f"{opener('control')}Charge port closed, Charm."
        elif action == "port_open":
            user_q = random.choice([
                "open the charge port", "open the charge port door",
            ])
            sys_lines = [entity_line("cover.white_python_charge_port_door", t["port_door"])]
            calls = [("call_service", {"domain": "cover", "service": "open_cover",
                                       "entity_id": "cover.white_python_charge_port_door"}, "ok")]
            reply = f"{opener('control')}Charge port open, Charm."
        else:  # precondition
            user_q = random.choice([
                "precondition the car", "warm the car up",
                "preheat the cabin",
            ])
            sys_lines = [entity_line("climate.white_python_climate", t["climate_on"])]
            calls = [
                ("call_service", {"domain": "climate", "service": "turn_on",
                                  "entity_id": "climate.white_python_climate"}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.white_python_climate",
                                  "data": {"temperature": 21}}, "ok"),
            ]
            reply = f"{opener('control')}Preconditioning started — cabin target 21 °C, Charm."

        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 18 — Lighting (40, queries + control)
# ---------------------------------------------------------------------------
LIGHT_ENTITIES = [
    ("light.wiz_tunable_white_402a9c", "Kitchen main light"),
    ("light.wiz_tunable_white_3219a8", "Kitchen secondary light"),
    ("light.wiz_tunable_white_3fbedc", "Living light"),
    ("light.left_pillar", "Left pillar light"),
]
SWITCH_LIGHTS = [
    ("switch.portico_light", "Portico light"),
    ("switch.left_side_floodlight", "Left side floodlight"),
    ("switch.front_left_floodlight", "Front left floodlight"),
    ("switch.garage_floodlight", "Garage floodlight"),
]

def gen_batch_18_lighting(n=40):
    out = []
    actions = ["query_all", "query_one", "turn_on_one", "turn_off_one",
               "turn_off_all", "dim", "turn_on_floods", "turn_off_floods", "portico"]
    for _ in range(n):
        period = rand_time_of_day()[2]
        lights = randomise_lighting(period)
        action = random.choice(actions)

        if action == "query_all":
            user_q = random.choice([
                "what lights are on", "are any lights on",
                "which lights are currently on", "list lights on",
            ])
            entries = [
                ("light.wiz_tunable_white_402a9c", lights["kitchen_main"]),
                ("light.wiz_tunable_white_3219a8", lights["kitchen_2"]),
                ("light.wiz_tunable_white_3fbedc", lights["living"]),
                ("light.left_pillar", lights["left_pillar"]),
                ("switch.portico_light", lights["portico"]),
            ]
            sys_lines = [entity_line(eid, st) for eid, st in entries]
            calls = [("get_entity_state", {"entity_id": eid}, st) for eid, st in entries]
            on_names = []
            for eid, st in entries:
                if st == "on":
                    on_names.append(snap_friendly(eid, eid))
            if on_names:
                reply = f"{opener()}On right now, Charm: {', '.join(on_names)}."
            else:
                reply = f"{opener()}No lights on, Charm."
        elif action == "query_one":
            eid, name = random.choice(LIGHT_ENTITIES + SWITCH_LIGHTS)
            st = random.choice(["on", "off"])
            user_q = random.choice([
                f"is the {name.lower()} on", f"{name.lower()} status",
                f"state of the {name.lower()}",
            ])
            sys_lines = [entity_line(eid, st)]
            calls = [("get_entity_state", {"entity_id": eid}, st)]
            reply = f"{opener()}{name} is {st}, Charm."
        elif action == "turn_on_one":
            eid, name = random.choice(LIGHT_ENTITIES + SWITCH_LIGHTS)
            user_q = random.choice([
                f"turn on the {name.lower()}", f"switch on the {name.lower()}",
                f"{name.lower()} on",
            ])
            sys_lines = [entity_line(eid, "off")]
            domain = eid.split(".")[0]
            calls = [("call_service", {"domain": domain, "service": "turn_on",
                                       "entity_id": eid}, "ok")]
            reply = f"{opener('control')}{name} on, Charm."
        elif action == "turn_off_one":
            eid, name = random.choice(LIGHT_ENTITIES + SWITCH_LIGHTS)
            user_q = random.choice([
                f"turn off the {name.lower()}", f"switch off the {name.lower()}",
                f"{name.lower()} off",
            ])
            sys_lines = [entity_line(eid, "on")]
            domain = eid.split(".")[0]
            calls = [("call_service", {"domain": domain, "service": "turn_off",
                                       "entity_id": eid}, "ok")]
            reply = f"{opener('control')}{name} off, Charm."
        elif action == "turn_off_all":
            user_q = random.choice([
                "turn off all lights", "all lights off", "kill all the lights",
            ])
            sys_lines = [entity_line(eid, "on") for eid, _ in LIGHT_ENTITIES]
            calls = [("call_service", {"domain": "light", "service": "turn_off",
                                       "entity_id": "all"}, "ok"),
                     ("call_service", {"domain": "switch", "service": "turn_off",
                                       "entity_id": "switch.portico_light"}, "ok")]
            reply = f"{opener('control')}All lights off, Charm."
        elif action == "dim":
            eid, name = random.choice(LIGHT_ENTITIES)
            pct = random.choice([10, 20, 30, 40, 50, 60, 70])
            user_q = random.choice([
                f"dim the {name.lower()} to {pct}%", f"set {name.lower()} to {pct} percent",
                f"{name.lower()} at {pct}%",
            ])
            sys_lines = [entity_line(eid, "on")]
            calls = [("call_service", {"domain": "light", "service": "turn_on",
                                       "entity_id": eid,
                                       "data": {"brightness_pct": pct}}, "ok")]
            reply = f"{opener('control')}{name} dimmed to {pct}%, Charm."
        elif action == "turn_on_floods":
            user_q = random.choice([
                "turn on the floodlights", "all floods on", "switch on the floodlights",
            ])
            sys_lines = [entity_line(eid, "off") for eid, _ in SWITCH_LIGHTS[1:]]
            calls = []
            for eid, _ in SWITCH_LIGHTS[1:]:
                calls.append(("call_service", {"domain": "switch", "service": "turn_on",
                                               "entity_id": eid}, "ok"))
            reply = f"{opener('control')}Floodlights on, Charm."
        elif action == "turn_off_floods":
            user_q = random.choice([
                "turn off the floodlights", "floods off", "kill the floodlights",
            ])
            sys_lines = [entity_line(eid, "on") for eid, _ in SWITCH_LIGHTS[1:]]
            calls = []
            for eid, _ in SWITCH_LIGHTS[1:]:
                calls.append(("call_service", {"domain": "switch", "service": "turn_off",
                                               "entity_id": eid}, "ok"))
            reply = f"{opener('control')}Floodlights off, Charm."
        else:  # portico
            user_q = random.choice([
                "turn on the portico light", "portico on", "porch light on",
            ])
            sys_lines = [entity_line("switch.portico_light", "off")]
            calls = [("call_service", {"domain": "switch", "service": "turn_on",
                                       "entity_id": "switch.portico_light"}, "ok")]
            reply = f"{opener('control')}Portico light on, Charm."

        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 19 — Security (30)
# ---------------------------------------------------------------------------
def gen_batch_19_security(n=30):
    out = []
    actions = ["query_alarm", "query_door", "query_occupied", "query_charm",
               "arm_home", "arm_away", "arm_night", "disarm",
               "lock_door", "unlock_door"]
    for _ in range(n):
        s = randomise_security()
        action = random.choice(actions)
        if action == "query_alarm":
            user_q = random.choice([
                "is the alarm armed", "alarm status", "what's the alarm doing",
            ])
            sys_lines = [entity_line("alarm_control_panel.ezviz_alarm", s["alarm"])]
            calls = [("get_entity_state", {"entity_id": "alarm_control_panel.ezviz_alarm"},
                      s["alarm"])]
            reply = f"{opener()}Alarm is {s['alarm']}, Charm."
        elif action == "query_door":
            user_q = random.choice([
                "is the front door locked", "front door status", "is the door secure",
            ])
            sys_lines = [entity_line("lock.front_door_3", s["front_door_lock"])]
            calls = [("get_entity_state", {"entity_id": "lock.front_door_3"},
                      s["front_door_lock"])]
            if s["front_door_lock"] == "unavailable":
                reply = "That data is unavailable right now, Charm."
            else:
                reply = f"{opener()}Front door is {s['front_door_lock']}, Charm."
        elif action == "query_occupied":
            user_q = random.choice([
                "is the house occupied", "is anyone home", "is the house empty",
            ])
            sys_lines = [entity_line("binary_sensor.house_occupied", s["house_occupied"])]
            calls = [("get_entity_state", {"entity_id": "binary_sensor.house_occupied"},
                      s["house_occupied"])]
            if s["house_occupied"] == "on":
                reply = f"{opener()}House is occupied, Charm."
            else:
                reply = f"{opener()}House reads as empty, Charm."
        elif action == "query_charm":
            user_q = random.choice([
                "am i home", "is charm home", "where am i according to ha",
            ])
            sys_lines = [entity_line("person.charm", s["person_charm"])]
            calls = [("get_entity_state", {"entity_id": "person.charm"}, s["person_charm"])]
            if s["person_charm"] == "home":
                reply = f"{opener()}You're home, Charm."
            else:
                reply = f"{opener()}You're not home according to HA, Charm."
        elif action == "arm_home":
            user_q = random.choice([
                "arm the alarm in home mode", "arm home", "set alarm to armed home",
            ])
            sys_lines = [entity_line("alarm_control_panel.ezviz_alarm", s["alarm"])]
            calls = [("call_service", {"domain": "alarm_control_panel", "service": "alarm_arm_home",
                                       "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok")]
            reply = f"{opener('control')}Alarm armed home, Charm."
        elif action == "arm_away":
            user_q = random.choice([
                "arm the alarm", "arm away", "arm for leaving the house",
                "set the alarm to armed away",
            ])
            sys_lines = [entity_line("alarm_control_panel.ezviz_alarm", s["alarm"])]
            calls = [("call_service", {"domain": "alarm_control_panel", "service": "alarm_arm_away",
                                       "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok")]
            reply = f"{opener('control')}Alarm armed away, Charm."
        elif action == "arm_night":
            user_q = random.choice([
                "arm for the night", "arm night mode", "arm the alarm for sleep",
            ])
            sys_lines = [entity_line("alarm_control_panel.ezviz_alarm", s["alarm"])]
            calls = [("call_service", {"domain": "alarm_control_panel", "service": "alarm_arm_night",
                                       "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok")]
            reply = f"{opener('control')}Alarm armed night, Charm."
        elif action == "disarm":
            user_q = random.choice([
                "disarm the alarm", "turn off the alarm", "alarm off",
            ])
            sys_lines = [entity_line("alarm_control_panel.ezviz_alarm", s["alarm"])]
            calls = [("call_service", {"domain": "alarm_control_panel", "service": "alarm_disarm",
                                       "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok")]
            reply = f"{opener('control')}Alarm disarmed, Charm."
        elif action == "lock_door":
            user_q = random.choice([
                "lock the front door", "secure the front door", "front door lock",
            ])
            sys_lines = [entity_line("lock.front_door_3", s["front_door_lock"])]
            calls = [("call_service", {"domain": "lock", "service": "lock",
                                       "entity_id": "lock.front_door_3"}, "ok")]
            reply = f"{opener('control')}Front door locked, Charm."
        else:  # unlock
            user_q = random.choice([
                "unlock the front door", "front door unlock",
            ])
            sys_lines = [entity_line("lock.front_door_3", s["front_door_lock"])]
            calls = [("call_service", {"domain": "lock", "service": "unlock",
                                       "entity_id": "lock.front_door_3"}, "ok")]
            reply = f"{opener('control')}Front door unlocked, Charm."

        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 20 — Appliance control (30)
# ---------------------------------------------------------------------------
def gen_batch_20_appliances(n=30):
    out = []
    actions = ["vac_start", "vac_dock", "vac_pause", "vac_locate", "vac_state",
               "garage_open", "garage_close", "tv_on", "tv_off",
               "shield_state", "ht_off"]
    for _ in range(n):
        a = randomise_appliances()
        action = random.choice(actions)
        if action == "vac_start":
            user_q = random.choice([
                "start the robot vacuum", "send the vacuum out", "kick off cleaning",
                "vacuum start",
            ])
            sys_lines = [entity_line("vacuum.kunurobo", a["vacuum"])]
            calls = [("call_service", {"domain": "vacuum", "service": "start",
                                       "entity_id": "vacuum.kunurobo"}, "ok")]
            reply = f"{opener('control')}KunuRobo dispatched, Charm."
        elif action == "vac_dock":
            user_q = random.choice([
                "send the vacuum back to dock", "dock the vacuum", "vacuum home",
            ])
            sys_lines = [entity_line("vacuum.kunurobo", a["vacuum"])]
            calls = [("call_service", {"domain": "vacuum", "service": "return_to_base",
                                       "entity_id": "vacuum.kunurobo"}, "ok")]
            reply = f"{opener('control')}KunuRobo returning to dock, Charm."
        elif action == "vac_pause":
            user_q = random.choice([
                "pause the vacuum", "vacuum pause", "stop the vacuum",
            ])
            sys_lines = [entity_line("vacuum.kunurobo", a["vacuum"])]
            calls = [("call_service", {"domain": "vacuum", "service": "pause",
                                       "entity_id": "vacuum.kunurobo"}, "ok")]
            reply = f"{opener('control')}KunuRobo paused, Charm."
        elif action == "vac_locate":
            user_q = random.choice([
                "locate the vacuum", "find the vacuum", "where is the kunurobo",
            ])
            sys_lines = [entity_line("vacuum.kunurobo", a["vacuum"])]
            calls = [("call_service", {"domain": "vacuum", "service": "locate",
                                       "entity_id": "vacuum.kunurobo"}, "ok")]
            reply = f"{opener('control')}Locate signal sent — it should chirp, Charm. Status: {a['vacuum']}."
        elif action == "vac_state":
            user_q = random.choice([
                "what's the vacuum doing", "vacuum state", "is the kunurobo running",
            ])
            sys_lines = [entity_line("vacuum.kunurobo", a["vacuum"])]
            calls = [("get_entity_state", {"entity_id": "vacuum.kunurobo"}, a["vacuum"])]
            reply = f"{opener()}KunuRobo is {a['vacuum']}, Charm."
        elif action == "garage_open":
            ch = random.choice([1, 2])
            user_q = random.choice([
                f"open garage door {ch}", "open the garage", "garage door up",
            ])
            sys_lines = [entity_line(f"switch.garage_door_channel_{ch}", "off")]
            calls = [("call_service", {"domain": "switch", "service": "turn_on",
                                       "entity_id": f"switch.garage_door_channel_{ch}"}, "ok")]
            reply = f"{opener('control')}Garage door {ch} triggered, Charm."
        elif action == "garage_close":
            ch = random.choice([1, 2])
            user_q = random.choice([
                f"close garage door {ch}", "close the garage", "garage down",
            ])
            sys_lines = [entity_line(f"switch.garage_door_channel_{ch}", "on")]
            calls = [("call_service", {"domain": "switch", "service": "turn_on",
                                       "entity_id": f"switch.garage_door_channel_{ch}"}, "ok")]
            reply = f"{opener('control')}Garage door {ch} triggered, Charm — assume it's closing."
        elif action == "tv_on":
            user_q = random.choice([
                "turn on the tv", "tv on", "switch on the bravia",
            ])
            sys_lines = [entity_line("media_player.living_tv_50i_bravia_2015_es_192_168_1_117",
                                     a["tv_state"])]
            calls = [("call_service", {"domain": "media_player", "service": "turn_on",
                                       "entity_id": "media_player.living_tv_50i_bravia_2015_es_192_168_1_117"},
                      "ok")]
            reply = f"{opener('control')}TV on, Charm."
        elif action == "tv_off":
            user_q = random.choice([
                "turn off the tv", "tv off", "switch off the bravia",
            ])
            sys_lines = [entity_line("media_player.living_tv_50i_bravia_2015_es_192_168_1_117",
                                     a["tv_state"])]
            calls = [("call_service", {"domain": "media_player", "service": "turn_off",
                                       "entity_id": "media_player.living_tv_50i_bravia_2015_es_192_168_1_117"},
                      "ok")]
            reply = f"{opener('control')}TV off, Charm."
        elif action == "shield_state":
            user_q = random.choice([
                "what's playing on the shield", "shield status",
                "is the shield doing anything",
            ])
            sys_lines = [entity_line("media_player.shield", a["shield_state"])]
            calls = [("get_entity_state", {"entity_id": "media_player.shield"},
                      a["shield_state"])]
            reply = f"{opener()}SHIELD is {a['shield_state']}, Charm."
        else:  # ht_off
            user_q = random.choice([
                "turn off the home theatre", "home theatre off",
                "shut down the home theater",
            ])
            sys_lines = [entity_line("media_player.home_theater", a["home_theater_state"])]
            calls = [("call_service", {"domain": "media_player", "service": "turn_off",
                                       "entity_id": "media_player.home_theater"}, "ok")]
            reply = f"{opener('control')}Home theatre off, Charm."

        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

# ---------------------------------------------------------------------------
# BATCH 21 — Web search (80)
# ---------------------------------------------------------------------------
WEB_SEARCH_TOPICS = [
    "aemo_vic_price", "aemo_nsw_price", "energy_news_au", "amber_status",
    "grid_event", "afl_result", "afl_ladder", "exchange_rate", "costco_hours",
    "bunnings_hours", "melbourne_storm_forecast", "weekend_weather",
    "netflix_tonight", "petrol_prices", "tesla_supercharger_route",
    "hayward_pool_chemistry", "ev_news", "amber_app_status",
    "vic_public_holidays", "bom_radar", "afl_fixture", "spotify_chart",
    "house_price_clyde_north", "bom_warnings", "covid_vic",
    "amber_negative_pricing", "rba_cash_rate", "petrol_unleaded91",
    "tesla_software_update", "energy_policy_au",
]

def _web_result(topic):
    # Realistic plausible synthetic search snippets, no fabricated URLs the model would memorise.
    if topic == "aemo_vic_price":
        c = random.randint(15, 280)
        return f"AEMO 5-minute spot price (VIC1) currently around {c} $/MWh ({round(c/10)} c/kWh)."
    if topic == "aemo_nsw_price":
        c = random.randint(15, 320)
        return f"AEMO 5-minute spot price (NSW1) currently around {c} $/MWh ({round(c/10)} c/kWh)."
    if topic == "energy_news_au":
        items = [
            "AEMO reports record solar absorption across NEM regions this week.",
            "Snowy 2.0 timeline pushed back to late 2028; AEMO ISP revised accordingly.",
            "Coal generator outages tightening evening NEM supply across QLD-NSW.",
            "Federal CER scheme announces new VPP coordination trials.",
        ]
        return random.choice(items)
    if topic == "amber_status":
        if random.random() < 0.85:
            return "Amber Electric status page reports all systems operational."
        return "Amber Electric status page: degraded API response times; price polls may lag."
    if topic == "grid_event":
        if random.random() < 0.3:
            return "Lack-of-reserve LOR2 declared in VIC1 for this evening 17:30–19:00."
        return "No active LOR or RERT events forecast on the AEMO market notices."
    if topic == "afl_result":
        teams = ["Geelong", "Collingwood", "Melbourne", "Hawthorn", "Carlton",
                 "Richmond", "Brisbane", "Sydney", "St Kilda", "Essendon"]
        a, b = random.sample(teams, 2)
        sa, sb = random.randint(60, 130), random.randint(40, 110)
        winner = a if sa > sb else b
        return f"{a} {sa} d. {b} {sb} (winner: {winner})."
    if topic == "afl_ladder":
        return ("Top of AFL ladder: 1. Sydney 14-2, 2. Collingwood 13-3, "
                "3. Geelong 12-4, 4. Brisbane 11-5.")
    if topic == "exchange_rate":
        r = round(random.uniform(0.61, 0.72), 4)
        return f"AUD/USD: {r}."
    if topic == "costco_hours":
        return ("Costco Moorabbin: Mon–Fri 10:00–20:30, Sat 09:30–19:00, Sun 10:00–18:00. "
                "Public holidays vary.")
    if topic == "bunnings_hours":
        return "Bunnings Cranbourne: 7:00–19:00 daily."
    if topic == "melbourne_storm_forecast":
        if random.random() < 0.4:
            return "BOM severe thunderstorm warning current for parts of central Victoria, valid through this evening."
        return "No severe weather warnings currently in force for greater Melbourne."
    if topic == "weekend_weather":
        return ("Melbourne weekend outlook: Saturday partly cloudy 19 °C, "
                "Sunday showers easing 16 °C.")
    if topic == "netflix_tonight":
        return ("Top Netflix Australia today: 1. \"Department Q\", 2. \"You S5\", "
                "3. \"Heartstopper S3\".")
    if topic == "petrol_prices":
        ulp = random.randint(170, 205)
        return f"Average ULP91 in Melbourne metro: {ulp}c/L; Diesel ~{ulp + 8}c/L."
    if topic == "tesla_supercharger_route":
        return ("Closest Supercharger on M1 corridor: Officer (8 stalls), 12.4 km away. "
                "Wantirna Tesla Centre also available.")
    if topic == "hayward_pool_chemistry":
        return ("Recommended pool free chlorine: 1-3 ppm, pH 7.2-7.6, "
                "TA 80-120 ppm, CYA 30-50 ppm.")
    if topic == "ev_news":
        return ("Tesla announces Model 3 software update 2026.14.3 rolling out — "
                "minor route planner refinements and parking visualisation tweaks.")
    if topic == "amber_app_status":
        return "Amber app: latest release 8.4.2, status: stable."
    if topic == "vic_public_holidays":
        return ("Next VIC public holidays: King's Birthday (Mon 8 June), "
                "AFL Grand Final Friday (Sep), Melbourne Cup (Tue 3 Nov).")
    if topic == "bom_radar":
        return "BOM radar (Melbourne 64 km): light returns over Mornington Peninsula moving east."
    if topic == "afl_fixture":
        return "AFL Round 11: Hawthorn v Geelong Saturday 19:25 MCG; Collingwood v Sydney Sunday 15:20 MCG."
    if topic == "spotify_chart":
        return ("Top of Spotify AU Today: 1. Taylor Swift - \"Fortnight\", "
                "2. Sabrina Carpenter - \"Espresso\", 3. The Kid LAROI - \"Wow\".")
    if topic == "house_price_clyde_north":
        return ("CoreLogic Clyde North median house price: $785,000 (May), up 3.2% YoY.")
    if topic == "bom_warnings":
        if random.random() < 0.3:
            return "BOM: severe weather warning for damaging winds, southwest Victoria, valid 6 hours."
        return "BOM: no active warnings for greater Melbourne."
    if topic == "covid_vic":
        return ("Victorian Health weekly: respiratory illness activity moderate; "
                "no elevated COVID-19 alert.")
    if topic == "amber_negative_pricing":
        return ("Amber blog: VIC region saw 9 negative-priced intervals on Saturday, "
                "concentrated 11:00–14:00, with the lowest at -$23/MWh.")
    if topic == "rba_cash_rate":
        rate = random.choice([3.85, 4.10, 4.35])
        return f"RBA cash rate currently {rate}%; next meeting set for first Tuesday of next month."
    if topic == "petrol_unleaded91":
        ulp = random.randint(168, 210)
        return f"Average unleaded 91 in Casey region: {ulp}c/L (BP/7-Eleven sites lowest)."
    if topic == "tesla_software_update":
        return "Tesla 2026.14.3 (Holiday Update) rolling out — UI refresh + parking visualisations."
    return "Federal energy market review report released; recommendations on storage incentives."

WEB_SEARCH_QUERIES = {
    "aemo_vic_price": [
        "what's the aemo spot price in victoria right now",
        "current aemo 5-minute price vic",
        "vic1 spot price now",
    ],
    "aemo_nsw_price": [
        "what's the spot price in nsw",
        "current nsw1 price",
    ],
    "energy_news_au": [
        "what's happening with energy prices in australia",
        "is there any energy market news this week",
        "any nem news today",
    ],
    "amber_status": [
        "is the amber electric api having issues today",
        "what's the amber electric status page saying",
        "any issues with the amber electric api today",
    ],
    "grid_event": [
        "is there a grid event forecast",
        "any lor warnings in vic tonight",
        "aemo grid event news",
    ],
    "afl_result": [
        "who won the afl last night",
        "last afl match result",
        "any afl result today",
    ],
    "afl_ladder": [
        "what's the afl ladder",
        "current afl ladder",
    ],
    "exchange_rate": [
        "what's the aud usd exchange rate today",
        "what's the exchange rate today",
    ],
    "costco_hours": [
        "what time does costco moorabbin open today",
        "costco moorabbin hours",
    ],
    "bunnings_hours": [
        "what time does bunnings cranbourne open",
        "bunnings cranbourne hours",
    ],
    "melbourne_storm_forecast": [
        "is there a storm coming to melbourne this week",
        "any severe weather warnings for melbourne",
    ],
    "weekend_weather": [
        "what's the weekend weather forecast",
        "weather for the weekend in melbourne",
    ],
    "netflix_tonight": [
        "what's on netflix tonight",
        "top netflix shows in australia right now",
    ],
    "petrol_prices": [
        "what's the petrol price in melbourne",
        "ulp91 price in melbourne",
    ],
    "tesla_supercharger_route": [
        "closest tesla supercharger on the m1",
        "nearest supercharger to clyde north",
    ],
    "hayward_pool_chemistry": [
        "what's the right chlorine level for a pool",
        "pool chemistry guidelines",
    ],
    "ev_news": [
        "any tesla software news",
        "latest tesla model 3 update news",
    ],
    "amber_app_status": [
        "what's the latest amber app version",
        "amber app status",
    ],
    "vic_public_holidays": [
        "next public holiday in victoria",
        "vic public holidays coming up",
    ],
    "bom_radar": [
        "what's the bom radar showing for melbourne",
        "any rain showing on the radar",
    ],
    "afl_fixture": [
        "afl fixture this round", "afl matches this weekend",
    ],
    "spotify_chart": [
        "top of the spotify chart australia",
        "what's number one on spotify",
    ],
    "house_price_clyde_north": [
        "median house price clyde north",
        "house prices in clyde north",
    ],
    "bom_warnings": [
        "any bom warnings for melbourne",
        "current weather warnings vic",
    ],
    "covid_vic": [
        "vic health advice this week",
        "any covid alert in vic",
    ],
    "amber_negative_pricing": [
        "any negative prices on amber this week",
        "amber negative pricing event",
    ],
    "rba_cash_rate": [
        "what's the rba cash rate",
        "current rba interest rate",
    ],
    "petrol_unleaded91": [
        "petrol price near clyde north",
        "ulp91 in casey",
    ],
    "tesla_software_update": [
        "is there a tesla software update",
        "latest tesla firmware version",
    ],
    "energy_policy_au": [
        "any australian energy policy news",
        "federal energy policy update",
    ],
}

def gen_batch_21_web_search(n=80):
    out = []
    topics = list(WEB_SEARCH_TOPICS) * 4
    random.shuffle(topics)
    for i in range(n):
        topic = topics[i % len(topics)]
        user_q = random.choice(WEB_SEARCH_QUERIES[topic])
        search_query = user_q
        result = _web_result(topic)
        sys_lines = []  # web_search doesn't need entity context
        # Persona-safe reply that incorporates the result
        if topic == "aemo_vic_price" or topic == "aemo_nsw_price":
            reply = f"{opener()}{result}, Charm."
        elif topic == "amber_status":
            if "degraded" in result or "lag" in result:
                reply = f"{opener()}{result} The optimiser may run on stale data, Charm."
            else:
                reply = f"{opener()}{result} No action needed, Charm."
        elif topic == "afl_result":
            reply = f"{result} Charm."
        elif topic == "exchange_rate":
            reply = f"{opener()}{result} Charm."
        elif topic == "weekend_weather":
            reply = f"{opener()}{result} Charm."
        elif topic == "petrol_prices" or topic == "petrol_unleaded91":
            reply = f"{result} Charm."
        elif topic == "grid_event":
            if "LOR2" in result:
                reply = f"{result} Worth deferring heavy loads during that window, Charm."
            else:
                reply = f"{opener()}{result} Charm."
        elif topic == "melbourne_storm_forecast":
            if "warning current" in result:
                reply = f"{result} Pull cushions inside and consider charging the battery, Charm."
            else:
                reply = f"{opener()}{result} Charm."
        else:
            reply = f"{opener()}{result} Charm."

        sys = make_system_prompt(["(no HA entities — web_search only)"])
        tcid = tc_id()
        seq = [
            assistant_tool_turn([make_tool_call(tcid, "web_search", {"query": search_query})]),
            make_tool_result(tcid, result),
            assistant_text(reply),
        ]
        out.append({"messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_q},
            *seq,
        ]})
    return out

# ---------------------------------------------------------------------------
# BATCH 22 — Multi-domain reasoning (120) — the highest-value examples
# ---------------------------------------------------------------------------
MULTI_QUERIES = [
    "complete_home_summary",
    "leaving_in_30",
    "going_to_bed",
    "cold_tonight",
    "hot_today",
    "should_run_dishwasher",
    "energy_cost_tonight",
    "is_now_good_to_precool",
    "how_is_the_house_going",
    "big_wash_cycle",
    "charge_car_or_wait",
    "guests_coming_tonight",
    "bill_seems_high",
    "leaving_for_a_week",
    "back_from_holiday",
    "movie_night",
    "morning_routine",
    "late_arrival_warmup",
]

def gen_batch_22_multi_domain(n=120):
    out = []
    for _ in range(n):
        pat = random.choice(MULTI_QUERIES)
        e = randomise_energy()
        w = randomise_weather(season=e["season"])
        hv = randomise_hvac(season=e["season"], energy=e)
        irr = randomise_irrigation(weather=w)
        t = randomise_tesla()
        s = randomise_security()
        a = randomise_appliances()
        lights = randomise_lighting(e["time"][2])

        if pat == "complete_home_summary":
            user_q = random.choice([
                "give me a complete home summary",
                "full status check on the house",
                "rundown of everything please",
                "tell me how the house is going",
                "house overview",
            ])
            calls = [
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "sensor.foxess_modbus_skynet_sun_harvester_pv_power_skynet_sun_harvester"},
                 str(e["solar_kw"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_load_power"},
                 str(e["load_kw"])),
                ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                 str(e["buy_price"])),
                ("get_entity_state", {"entity_id": "sensor.llm_response_file"}, e["decision"]),
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(hv["downstairs_temp"])),
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "alarm_control_panel.ezviz_alarm"}, s["alarm"]),
                ("get_entity_state", {"entity_id": "device_tracker.white_python_location"},
                 t["location"] if t["available"] else "unavailable"),
            ]
            sys_lines = [
                entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
                entity_line("sensor.foxess_modbus_skynet_sun_harvester_pv_power_skynet_sun_harvester",
                            e["solar_kw"], "kW"),
                entity_line("sensor.skynet_sun_harvester_load_power", e["load_kw"], "kW"),
                entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
                entity_line("sensor.llm_response_file", e["decision"]),
                entity_line("sensor.downstairs_zone_temperature", hv["downstairs_temp"], "°C"),
                entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                entity_line("alarm_control_panel.ezviz_alarm", s["alarm"]),
                entity_line("device_tracker.white_python_location",
                            t["location"] if t["available"] else "unavailable"),
            ]
            car_note = (f"car {t['battery_level']}% at {t['location']}" if t["available"]
                        else "car sleeping")
            reply = (
                f"{opener('analysis')}Energy: battery {e['battery_soc']}%, solar {e['solar_kw']} kW, "
                f"load {e['load_kw']} kW, buy {e['buy_cents']} c/kWh, optimiser on {e['decision']}. "
                f"Comfort: downstairs {hv['downstairs_temp']} °C against outdoor {w['outdoor_temp']} °C "
                f"({w['short_forecast'].lower()}). Security: alarm {s['alarm']}, front door "
                f"{s['front_door_lock']}. Tesla: {car_note}. Nothing requires attention right now, Charm."
                if not e['spike_active'] else
                f"{opener('analysis')}Heads-up first: price spike active at {e['buy_cents']} c/kWh. "
                f"Battery {e['battery_soc']}%, solar {e['solar_kw']} kW, load {e['load_kw']} kW, "
                f"optimiser on {e['decision']}. Comfort: {hv['downstairs_temp']} °C inside / "
                f"{w['outdoor_temp']} °C outside. Alarm {s['alarm']}, front door {s['front_door_lock']}, "
                f"{car_note}. Recommend deferring heavy loads until the spike clears, Charm."
            )

        elif pat == "leaving_in_30":
            user_q = random.choice([
                "i'm leaving in 30 minutes, sort the house out",
                "heading out in 30 mins, get the house ready",
                "leaving shortly, set things up",
                "i'm off in half an hour, set the house up",
            ])
            calls = [
                ("get_entity_state", {"entity_id": "binary_sensor.house_occupied"}, s["house_occupied"]),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"hvac_mode": "off"}}, "ok"),
                ("call_service", {"domain": "alarm_control_panel", "service": "alarm_arm_away",
                                  "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok"),
                ("call_service", {"domain": "lock", "service": "lock",
                                  "entity_id": "lock.front_door_3"}, "ok"),
            ]
            tesla_note = ""
            if t["available"] and t["climate_on"] == "on":
                calls.append(("call_service", {"domain": "climate", "service": "turn_off",
                                               "entity_id": "climate.white_python_climate"}, "ok"))
                tesla_note = " Car climate off. "
            sys_lines = [
                entity_line("climate.izone_controller_402001095", hv["main_mode"]),
                entity_line("alarm_control_panel.ezviz_alarm", s["alarm"]),
                entity_line("lock.front_door_3", s["front_door_lock"]),
                entity_line("climate.white_python_climate",
                            t["climate_on"] if t["available"] else "unavailable"),
            ]
            reply = (f"{opener('control')}HVAC off. Alarm armed away. Front door locked.{tesla_note}"
                     f"Safe travels, Charm. Battery {e['battery_soc']}%, price {e['buy_cents']} c/kWh — "
                     f"all under control.")

        elif pat == "going_to_bed":
            user_q = random.choice([
                "i'm going to bed, wrap things up",
                "bedtime — sort everything",
                "shutting down for the night",
                "off to bed, do the night routine",
            ])
            calls = [
                ("call_service", {"domain": "light", "service": "turn_off",
                                  "entity_id": "all"}, "ok"),
                ("call_service", {"domain": "switch", "service": "turn_off",
                                  "entity_id": "switch.portico_light"}, "ok"),
                ("call_service", {"domain": "alarm_control_panel", "service": "alarm_arm_night",
                                  "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok"),
                ("call_service", {"domain": "lock", "service": "lock",
                                  "entity_id": "lock.front_door_3"}, "ok"),
                ("get_entity_state", {"entity_id": "sensor.reserve_soc_needed_to_sunrise"},
                 str(e["reserve_soc_needed"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
            ]
            sys_lines = [
                entity_line("alarm_control_panel.ezviz_alarm", s["alarm"]),
                entity_line("lock.front_door_3", s["front_door_lock"]),
                entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
                entity_line("sensor.reserve_soc_needed_to_sunrise", e["reserve_soc_needed"], "%"),
            ]
            margin_ok = e["battery_soc"] >= e["reserve_soc_needed"]
            soc_note = ("Battery sits comfortably above reserve" if margin_ok
                        else f"Battery {e['battery_soc']}% is below the {e['reserve_soc_needed']}% "
                             "reserve — optimiser will likely top up overnight")
            reply = (f"{opener('control')}Lights off, portico off, alarm armed night, front door locked. "
                     f"{soc_note}, Charm. Sleep well.")

        elif pat == "cold_tonight":
            user_q = random.choice([
                "it's getting cold tonight, what should i do",
                "house is going to be cold — plan it out",
                "cold front coming, what's the play",
            ])
            ambient = hv["downstairs_temp"]
            target = round(max(20.0, ambient + 2.5), 1)
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(ambient)),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                 str(e["buy_price"])),
            ]
            sys_lines = [
                entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                entity_line("sensor.downstairs_zone_temperature", ambient, "°C"),
                entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
                entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
            ]
            if e["buy_cents"] < 20 and e["battery_soc"] > 50:
                reply = (f"{opener('analysis')}Outdoor {w['outdoor_temp']} °C, indoor {ambient} °C, "
                         f"battery {e['battery_soc']}%, price {e['buy_cents']} c/kWh, Charm. "
                         f"Pre-heat to {target} °C now (heat_cool mode, setpoint above ambient = heat), "
                         f"then let the battery carry the evening load.")
            else:
                reply = (f"{opener('analysis')}Outdoor {w['outdoor_temp']} °C, indoor {ambient} °C, "
                         f"battery {e['battery_soc']}%, price {e['buy_cents']} c/kWh, Charm. "
                         f"Set downstairs to {target} °C in heat_cool — it will heat. Keep an eye "
                         f"on price; the optimiser will pause HVAC if a spike hits.")

        elif pat == "hot_today":
            user_q = random.choice([
                "it's going to be hot today, what's the play",
                "hot day forecast — plan the cooling",
                "how should i handle today's heat",
            ])
            target = round(min(24.0, hv["downstairs_temp"] - 2.5), 1)
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_max_temp"},
                 str(w["max_temp"])),
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(hv["downstairs_temp"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                 str(e["buy_price"])),
                ("get_entity_state", {"entity_id": "sensor.amber_price_timeline",
                                      "attribute": "data"},
                 json.dumps(randomise_amber_timeline(e["time"][0], e["time"][1], hours=8))),
            ]
            sys_lines = [
                entity_line("sensor.home_charm_max_temp", w["max_temp"], "°C"),
                entity_line("sensor.downstairs_zone_temperature", hv["downstairs_temp"], "°C"),
                entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
                entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
            ]
            reply = (f"{opener('analysis')}Top of {w['max_temp']} °C forecast, indoor {hv['downstairs_temp']} °C, "
                     f"battery {e['battery_soc']}%, price {e['buy_cents']} c/kWh, Charm. "
                     f"Pre-cool to {target} °C in heat_cool while solar is producing, then ride the "
                     f"evening peak from battery. The optimiser will pause AC if Amber spikes.")

        elif pat == "should_run_dishwasher" or pat == "big_wash_cycle":
            appliance = "dishwasher" if pat == "should_run_dishwasher" else "wash cycle"
            user_q = random.choice([
                f"should i run the {appliance} now",
                f"ok to start the {appliance}",
                f"is now a good time for the {appliance}",
            ])
            timeline = randomise_amber_timeline(e["time"][0], e["time"][1], hours=6)
            future_min = min(round(it["price_per_kwh"] * 100) for it in timeline)
            calls = [
                ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                 str(e["buy_price"])),
                ("get_entity_state", {"entity_id": "binary_sensor.charm_house_price_spike"},
                 "on" if e["spike_active"] else "off"),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "sensor.amber_price_timeline",
                                      "attribute": "data"}, json.dumps(timeline)),
            ]
            sys_lines = [
                entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
                entity_line("binary_sensor.charm_house_price_spike",
                            "on" if e["spike_active"] else "off"),
                entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
                entity_line("sensor.amber_price_timeline", str(e["buy_price"]),
                            friendly="Amber Price Timeline"),
            ]
            if e["spike_active"]:
                reply = (f"Not now, Charm — spike at {e['buy_cents']} c/kWh. Forecast trough is "
                         f"{future_min} c/kWh; wait for that.")
            elif e["buy_cents"] <= 15 and e["battery_soc"] > 30:
                reply = (f"Go ahead, Charm — {e['buy_cents']} c/kWh and battery at {e['battery_soc']}%, "
                         f"the load will run cheap.")
            elif future_min < e["buy_cents"] - 8:
                reply = (f"Hold off, Charm — {e['buy_cents']} c/kWh now versus a {future_min} c/kWh dip "
                         f"in the next few hours. Schedule for then.")
            else:
                reply = (f"Acceptable, Charm — {e['buy_cents']} c/kWh, no spike, battery {e['battery_soc']}%. "
                         f"No cheaper window worth waiting for.")

        elif pat == "energy_cost_tonight":
            user_q = random.choice([
                "what's my energy cost going to be tonight",
                "what will tonight cost me energy-wise",
                "estimate tonight's grid spend",
            ])
            timeline = randomise_amber_timeline(18, 0, hours=6)
            avg_c = round(sum(it["price_per_kwh"] for it in timeline) / len(timeline) * 100)
            est_load_kwh = round(random.uniform(3, 10), 1)
            covered_by_batt = min(est_load_kwh, e["kwh_remaining"] * 0.7)
            from_grid = max(0, est_load_kwh - covered_by_batt)
            cost = round(from_grid * avg_c)
            calls = [
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_load_power"},
                 str(e["load_kw"])),
                ("get_entity_state", {"entity_id": "sensor.amber_price_timeline",
                                      "attribute": "data"}, json.dumps(timeline)),
            ]
            sys_lines = [
                entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
                entity_line("sensor.skynet_sun_harvester_load_power", e["load_kw"], "kW"),
                entity_line("sensor.amber_price_timeline", str(e["buy_price"]),
                            friendly="Amber Price Timeline"),
            ]
            reply = (f"{opener('analysis')}Battery at {e['battery_soc']}% ({e['kwh_remaining']} kWh), "
                     f"average evening price ~{avg_c} c/kWh, Charm. Expected load tonight ~{est_load_kwh} kWh, "
                     f"battery should cover ~{round(covered_by_batt, 1)} kWh, leaving "
                     f"~{round(from_grid, 1)} kWh from grid — roughly {cost} cents (${cost/100:.2f}).")

        elif pat == "is_now_good_to_precool":
            user_q = random.choice([
                "is it worth pre-cooling the house now before the peak",
                "should i precool now",
                "any value in pre-cooling right now",
            ])
            timeline = randomise_amber_timeline(e["time"][0], e["time"][1], hours=8, spike_prob=0.1)
            sp_item, sp_c = _next_spike(timeline)
            calls = [
                ("get_entity_state", {"entity_id": "sensor.home_charm_temp"},
                 str(w["outdoor_temp"])),
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(hv["downstairs_temp"])),
                ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                 str(e["buy_price"])),
                ("get_entity_state", {"entity_id": "sensor.amber_price_timeline",
                                      "attribute": "data"}, json.dumps(timeline)),
            ]
            sys_lines = [
                entity_line("sensor.home_charm_temp", w["outdoor_temp"], "°C"),
                entity_line("sensor.downstairs_zone_temperature", hv["downstairs_temp"], "°C"),
                entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
                entity_line("sensor.amber_price_timeline", str(e["buy_price"]),
                            friendly="Amber Price Timeline"),
            ]
            if sp_item and e["buy_cents"] < sp_c - 10:
                t_h = sp_item["start_time"][11:16]
                target = round(min(24.0, hv["downstairs_temp"] - 2), 1)
                reply = (f"{opener('analysis')}Yes, Charm — current {e['buy_cents']} c/kWh, peak of "
                         f"~{sp_c} c/kWh forecast at {t_h}. Pre-cool to {target} °C now (heat_cool, "
                         f"setpoint below ambient = cool).")
            else:
                reply = (f"{opener('analysis')}No meaningful peak forecast, Charm — pre-cooling won't pay back. "
                         f"Run the AC normally as comfort requires.")

        elif pat == "how_is_the_house_going":
            user_q = random.choice([
                "how is the house going",
                "everything okay at home",
                "anything i should know",
            ])
            calls = [
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_battery_soc_1"},
                 str(e["battery_soc"])),
                ("get_entity_state", {"entity_id": "binary_sensor.charm_house_price_spike"},
                 "on" if e["spike_active"] else "off"),
                ("get_entity_state", {"entity_id": "binary_sensor.amber_decision_degraded"},
                 "on" if random.random() < 0.05 else "off"),
                ("get_entity_state", {"entity_id": "alarm_control_panel.ezviz_alarm"}, s["alarm"]),
                ("get_entity_state", {"entity_id": "binary_sensor.hvac_running"},
                 "on" if hv["running"] else "off"),
            ]
            sys_lines = [
                entity_line("sensor.skynet_sun_harvester_battery_soc_1", e["battery_soc"], "%"),
                entity_line("binary_sensor.charm_house_price_spike",
                            "on" if e["spike_active"] else "off"),
                entity_line("alarm_control_panel.ezviz_alarm", s["alarm"]),
                entity_line("binary_sensor.hvac_running", "on" if hv["running"] else "off"),
            ]
            issues = []
            if e["spike_active"]:
                issues.append(f"price spike at {e['buy_cents']} c/kWh")
            if e["battery_soc"] < 20:
                issues.append(f"battery low at {e['battery_soc']}%")
            if calls[2][2] == "on":
                issues.append("optimiser running degraded")
            if issues:
                reply = (f"{opener('analysis')}A few things, Charm: " + "; ".join(issues) + ". "
                         f"Otherwise alarm {s['alarm']}, HVAC {'running' if hv['running'] else 'idle'}.")
            else:
                reply = (f"{opener('analysis')}All quiet, Charm — battery {e['battery_soc']}%, no spike, "
                         f"alarm {s['alarm']}, HVAC {'running' if hv['running'] else 'idle'}.")

        elif pat == "charge_car_or_wait":
            user_q = random.choice([
                "should i charge the car from the grid or wait",
                "charge the car now or hold off",
                "best window to charge the car",
            ])
            timeline = randomise_amber_timeline(e["time"][0], e["time"][1], hours=12)
            future_min = min(round(it["price_per_kwh"] * 100) for it in timeline)
            future_idx = [round(it["price_per_kwh"]*100) for it in timeline].index(future_min)
            future_time = timeline[future_idx]["start_time"][11:16]
            sys_lines = [
                entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
                entity_line("number.white_python_charge_limit",
                            t["charge_limit"] if t["available"] else "unavailable",
                            "%" if t["available"] else ""),
                entity_line("sensor.amber_price_timeline", str(e["buy_price"]),
                            friendly="Amber Price Timeline"),
            ]
            if not t["available"]:
                calls = [
                    ("get_entity_state", {"entity_id": "device_tracker.white_python_location"},
                     "unavailable"),
                ]
                reply = "The car is sleeping, Charm. Open the Tesla app to wake it before we can plan a charge."
            else:
                calls = [
                    ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                     str(e["buy_price"])),
                    ("get_entity_state",
                     {"entity_id": "number.white_python_charge_limit",
                      "attribute": "current_value"}, str(t["battery_level"])),
                    ("get_entity_state", {"entity_id": "sensor.amber_price_timeline",
                                          "attribute": "data"}, json.dumps(timeline)),
                ]
                if future_min < e["buy_cents"] - 5:
                    reply = (f"{opener('analysis')}Wait, Charm — current {e['buy_cents']} c/kWh, trough of "
                             f"{future_min} c/kWh around {future_time}. Schedule the charge for then. "
                             f"Car at {t['battery_level']}%, limit {t['charge_limit']}%.")
                else:
                    reply = (f"{opener('analysis')}Charge now, Charm — {e['buy_cents']} c/kWh, no cheaper "
                             f"window forecast (low {future_min} c/kWh). Car at {t['battery_level']}%, "
                             f"limit {t['charge_limit']}%.")

        elif pat == "guests_coming_tonight":
            user_q = random.choice([
                "guests coming tonight, set the house up",
                "i'm having people over tonight — sort the place",
                "guests in 30, set the scene",
            ])
            target = 21
            calls = [
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"temperature": target}}, "ok"),
                ("call_service", {"domain": "light", "service": "turn_on",
                                  "entity_id": "light.wiz_tunable_white_402a9c",
                                  "data": {"brightness_pct": 70}}, "ok"),
                ("call_service", {"domain": "light", "service": "turn_on",
                                  "entity_id": "light.wiz_tunable_white_3fbedc",
                                  "data": {"brightness_pct": 60}}, "ok"),
                ("call_service", {"domain": "switch", "service": "turn_on",
                                  "entity_id": "switch.portico_light"}, "ok"),
                ("call_service", {"domain": "alarm_control_panel", "service": "alarm_disarm",
                                  "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok"),
            ]
            sys_lines = [
                entity_line("climate.izone_controller_402001095", hv["main_mode"]),
                entity_line("alarm_control_panel.ezviz_alarm", s["alarm"]),
                entity_line("sensor.downstairs_zone_temperature", hv["downstairs_temp"], "°C"),
                entity_line("light.wiz_tunable_white_402a9c", "off"),
                entity_line("light.wiz_tunable_white_3fbedc", "off"),
            ]
            direction = _hvac_decide(target, hv["downstairs_temp"])
            dir_note = (f"will heat (target {target} > ambient {hv['downstairs_temp']})" if direction == "heat"
                        else f"will cool (target {target} < ambient {hv['downstairs_temp']})" if direction == "cool"
                        else "will idle in deadband")
            reply = (f"{opener('control')}HVAC to {target} °C in heat_cool — {dir_note}. Kitchen and "
                     f"living lights at warm levels, portico on, alarm disarmed, Charm.")

        elif pat == "bill_seems_high":
            user_q = random.choice([
                "the electricity bill seems high — what's been going on",
                "why is my power bill up this month",
                "any reason for the bill increase",
            ])
            calls = [
                ("get_entity_state", {"entity_id": "sensor.skynet_sun_harvester_grid_consumption_energy_today"},
                 str(e["grid_today_kwh"])),
                ("get_entity_state", {"entity_id": "sensor.daily_grid_import_cost_raw_cents"},
                 str(e["grid_cost_cents"])),
                ("web_search", {"query": "amber electric victoria pricing trends last month"},
                 "AEMO Victoria region reports above-seasonal-average prices over the past four weeks "
                 "due to coal generator outages and reduced wind."),
            ]
            sys_lines = [
                entity_line("sensor.skynet_sun_harvester_grid_consumption_energy_today",
                            e["grid_today_kwh"], "kWh"),
                entity_line("sensor.daily_grid_import_cost_raw_cents", e["grid_cost_cents"], "c"),
            ]
            reply = (f"{opener('analysis')}Two factors, Charm. Locally: {e['grid_today_kwh']} kWh imported "
                     f"today at {int(e['grid_cost_cents'])} cents — broadly consistent. Externally: AEMO "
                     f"reports above-average prices in VIC1 over the past month due to coal outages and "
                     f"reduced wind. The optimiser is doing what it can; raising the safety max-buy might "
                     f"unlock more force-charge opportunities.")

        elif pat == "leaving_for_a_week":
            user_q = random.choice([
                "leaving for a week — set everything up",
                "going away for 7 days, set the house",
                "off on holiday for a week, get the house ready",
            ])
            calls = [
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"hvac_mode": "off"}}, "ok"),
                ("call_service", {"domain": "input_select", "service": "select_option",
                                  "entity_id": "input_select.hvac_schedule_mode",
                                  "data": {"option": "away"}}, "ok"),
                ("call_service", {"domain": "input_boolean", "service": "turn_off",
                                  "entity_id": "input_boolean.irrigation_master_enable"}, "ok"),
                ("call_service", {"domain": "alarm_control_panel", "service": "alarm_arm_away",
                                  "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok"),
                ("call_service", {"domain": "lock", "service": "lock",
                                  "entity_id": "lock.front_door_3"}, "ok"),
            ]
            sys_lines = [
                entity_line("climate.izone_controller_402001095", hv["main_mode"]),
                entity_line("input_select.hvac_schedule_mode", hv["schedule_mode"]),
                entity_line("alarm_control_panel.ezviz_alarm", s["alarm"]),
                entity_line("input_boolean.irrigation_master_enable",
                            "on" if irr["master_enabled"] else "off"),
            ]
            reply = (f"{opener('control')}HVAC off, schedule set to away, irrigation master disabled, "
                     f"alarm armed away, front door locked, Charm. Optimiser and battery will keep "
                     f"running. Have a good trip.")

        elif pat == "back_from_holiday":
            user_q = random.choice([
                "i'm back from holiday, normalise the house",
                "back from away — set things normal",
                "home again, reset the routines",
            ])
            target = 21
            calls = [
                ("call_service", {"domain": "input_select", "service": "select_option",
                                  "entity_id": "input_select.hvac_schedule_mode",
                                  "data": {"option": "auto"}}, "ok"),
                ("call_service", {"domain": "input_boolean", "service": "turn_on",
                                  "entity_id": "input_boolean.irrigation_master_enable"}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"temperature": target}}, "ok"),
                ("call_service", {"domain": "alarm_control_panel", "service": "alarm_disarm",
                                  "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok"),
            ]
            sys_lines = [
                entity_line("input_select.hvac_schedule_mode", hv["schedule_mode"]),
                entity_line("climate.izone_controller_402001095", hv["main_mode"]),
                entity_line("alarm_control_panel.ezviz_alarm", s["alarm"]),
                entity_line("input_boolean.irrigation_master_enable",
                            "on" if irr["master_enabled"] else "off"),
            ]
            direction = _hvac_decide(target, hv["downstairs_temp"])
            dir_note = ("will heat" if direction == "heat" else
                        "will cool" if direction == "cool" else "in deadband")
            reply = (f"{opener('control')}Schedule auto, irrigation enabled, HVAC to {target} °C "
                     f"in heat_cool ({dir_note} versus current {hv['downstairs_temp']} °C), "
                     f"alarm disarmed, Charm. Welcome home.")

        elif pat == "movie_night":
            user_q = random.choice([
                "set up for a movie", "movie night, set the scene",
                "dim things for a film",
            ])
            calls = [
                ("call_service", {"domain": "light", "service": "turn_off",
                                  "entity_id": "light.wiz_tunable_white_3219a8"}, "ok"),
                ("call_service", {"domain": "light", "service": "turn_on",
                                  "entity_id": "light.wiz_tunable_white_3fbedc",
                                  "data": {"brightness_pct": 15}}, "ok"),
                ("call_service", {"domain": "media_player", "service": "turn_on",
                                  "entity_id": "media_player.shield"}, "ok"),
                ("call_service", {"domain": "media_player", "service": "turn_on",
                                  "entity_id": "media_player.home_theater"}, "ok"),
            ]
            sys_lines = [
                entity_line("light.wiz_tunable_white_3fbedc", "on"),
                entity_line("media_player.shield", a["shield_state"]),
                entity_line("media_player.home_theater", a["home_theater_state"]),
            ]
            reply = (f"{opener('control')}Living dimmed to 15%, kitchen secondary off, SHIELD and "
                     f"home theatre on, Charm. Enjoy.")

        elif pat == "morning_routine":
            user_q = random.choice([
                "good morning — run the morning routine",
                "morning setup",
                "do the morning routine",
            ])
            target = 21 if w["outdoor_temp"] < 18 else 23
            calls = [
                ("call_service", {"domain": "alarm_control_panel", "service": "alarm_disarm",
                                  "entity_id": "alarm_control_panel.ezviz_alarm"}, "ok"),
                ("call_service", {"domain": "light", "service": "turn_on",
                                  "entity_id": "light.wiz_tunable_white_402a9c",
                                  "data": {"brightness_pct": 80}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"temperature": target}}, "ok"),
                ("get_entity_state", {"entity_id": "sensor.clyde_north_short_text_0"},
                 w["short_forecast"]),
                ("get_entity_state", {"entity_id": "sensor.charm_house_general_price"},
                 str(e["buy_price"])),
            ]
            sys_lines = [
                entity_line("alarm_control_panel.ezviz_alarm", s["alarm"]),
                entity_line("light.wiz_tunable_white_402a9c", lights["kitchen_main"]),
                entity_line("climate.izone_controller_402001095", hv["main_mode"]),
                entity_line("sensor.clyde_north_short_text_0", w["short_forecast"]),
                entity_line("sensor.charm_house_general_price", e["buy_price"], "$/kWh"),
            ]
            direction = _hvac_decide(target, hv["downstairs_temp"])
            dir_note = "heating" if direction == "heat" else "cooling" if direction == "cool" else "idle"
            reply = (f"{opener('control')}Alarm disarmed, kitchen at 80%, HVAC to {target} °C ({dir_note}). "
                     f"Outside: {w['short_forecast'].lower()} top of {w['max_temp']} °C. Buy price "
                     f"{e['buy_cents']} c/kWh, Charm. Good morning.")

        else:  # late_arrival_warmup
            user_q = random.choice([
                "i'll be home in 30 minutes, warm the house",
                "getting home in half an hour, warm things up",
                "heading home, get the house ready",
            ])
            target = round(max(20.5, hv["downstairs_temp"] + 2.0), 1)
            calls = [
                ("get_entity_state", {"entity_id": "sensor.downstairs_zone_temperature"},
                 str(hv["downstairs_temp"])),
                ("call_service", {"domain": "climate", "service": "set_hvac_mode",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"hvac_mode": "heat_cool"}}, "ok"),
                ("call_service", {"domain": "climate", "service": "set_temperature",
                                  "entity_id": "climate.izone_controller_402001095",
                                  "data": {"temperature": target}}, "ok"),
                ("call_service", {"domain": "switch", "service": "turn_on",
                                  "entity_id": "switch.portico_light"}, "ok"),
            ]
            sys_lines = [
                entity_line("sensor.downstairs_zone_temperature", hv["downstairs_temp"], "°C"),
                entity_line("climate.izone_controller_402001095", hv["main_mode"]),
                entity_line("switch.portico_light", lights["portico"]),
            ]
            direction = _hvac_decide(target, hv["downstairs_temp"])
            dir_note = ("will heat" if direction == "heat" else
                        "will cool" if direction == "cool" else "is in deadband")
            reply = (f"{opener('control')}HVAC to {target} °C in heat_cool — setpoint against current "
                     f"{hv['downstairs_temp']} °C, {dir_note}. Portico light on for arrival, Charm.")

        out.append(multi_call_example(sys_lines, user_q, calls, reply))
    return out

def main():
    all_examples = []
    batches = [
        ("batch_01_energy_status",      gen_batch_01_energy_status,      120),
        ("batch_02_energy_reasoning",   gen_batch_02_energy_reasoning,   80),
        ("batch_03_energy_control",     gen_batch_03_energy_control,     40),
        ("batch_04_amber_price",        gen_batch_04_amber_price,        100),
        ("batch_05_amber_reasoning",    gen_batch_05_amber_reasoning,    200),
        ("batch_06_optimiser_control",  gen_batch_06_optimiser_control,  30),
        ("batch_07_hvac_status",        gen_batch_07_hvac_status,        80),
        ("batch_08_hvac_reasoning",     gen_batch_08_hvac_reasoning,     60),
        ("batch_09_hvac_control",       gen_batch_09_hvac_control,       80),
        ("batch_10_weather",            gen_batch_10_weather,            60),
        ("batch_11_weather_reasoning",  gen_batch_11_weather_reasoning,  40),
        ("batch_12_irrigation_status",  gen_batch_12_irrigation_status,  40),
        ("batch_13_irrigation_reason",  gen_batch_13_irrigation_reasoning, 30),
        ("batch_14_irrigation_control", gen_batch_14_irrigation_control, 20),
        ("batch_15_tesla_queries",      gen_batch_15_tesla_queries,      50),
        ("batch_16_tesla_reasoning",    gen_batch_16_tesla_reasoning,    30),
        ("batch_17_tesla_control",      gen_batch_17_tesla_control,      30),
        ("batch_18_lighting",           gen_batch_18_lighting,           40),
        ("batch_19_security",           gen_batch_19_security,           30),
        ("batch_20_appliances",         gen_batch_20_appliances,         30),
        ("batch_21_web_search",         gen_batch_21_web_search,         80),
        ("batch_22_multi_domain",       gen_batch_22_multi_domain,       220),
    ]
    total = 0
    for name, fn, n in batches:
        ex = fn(n)
        if len(ex) != n:
            print(f"WARNING: {name} produced {len(ex)} examples, expected {n}")
        write_jsonl(HERE / f"{name}.jsonl", ex)
        all_examples.extend(ex)
        total += len(ex)
        print(f"  {name}: {len(ex)}")
    print(f"\nTotal: {total}")
    random.shuffle(all_examples)
    write_jsonl(OUT_COMBINED, all_examples)
    print(f"Wrote combined file: {OUT_COMBINED}")

if __name__ == "__main__":
    main()
