# HAL v3 Probe Results — STRICT RE-GRADE — 2026-05-21

**Source:** `probe_v3_responses.jsonl` (69 prompts) cross-checked against `/api/states` snapshot (1,433 entities)
**Model:** `charm-hal-v3` (Qwen3-8B v3 LoRA, Q4_K_M GGUF)
**Supersedes:** `PROBE_SUMMARY_V3_2026-05-21.md` (lenient grading, retained for diff comparison only)

---

## Why this re-grade exists

The original summary scored v3 at 82.6%. Charm pointed out — correctly — that the pass criteria were far too lenient. A response was being marked "pass" if the model produced a coherent sentence at all, regardless of whether it grounded in real sensor data. After cross-checking every "I don't have access to..." refusal against the live HA state snapshot, the model's refusals are almost all wrong: the entities exist, the model just failed to look.

This re-grade applies a stricter rule:

> **A response passes only if it cites a specific current value or state from the home that matches what is actually in HA at the time.**

Vague answers, meta-answers ("live data is now available"), refusals on entities that demonstrably exist, hallucinated values, and HTTP errors all fail.

---

## Headline number

| Grading | Pass | Fail | Pass % |
|---|---:|---:|---:|
| Lenient (original) | 57 | 12 | 82.6% |
| **Strict (this doc)** | **22** | **47** | **31.9%** |

**The strict result is the one to trust.** It reflects the model's actual ability to do the thing HAL is meant to do — observe the house and answer factually.

---

## Pass rate by category (strict)

| Category | Total | Pass | Fail | Strict % | Lenient % | Δ |
|---|---:|---:|---:|---:|---:|---:|
| time_date | 3 | 3 | 0 | **100%** | 100% | = |
| hvac_direct | 4 | 2 | 2 | 50% | 100% | ▼ -50 |
| hvac_indirect | 3 | 1 | 2 | 33% | 67% | ▼ -34 |
| energy_direct | 4 | 2 | 2 | 50% | 50% | = |
| energy_indirect | 4 | 0 | 4 | **0%** | 100% | ▼ -100 |
| solar_direct | 4 | 2 | 2 | 50% | 50% | = |
| solar_indirect | 3 | 0 | 3 | **0%** | 100% | ▼ -100 |
| battery_direct | 4 | 2 | 2 | 50% | 100% | ▼ -50 |
| battery_indirect | 3 | 2 | 1 | 67% | 100% | ▼ -33 |
| tesla_direct | 4 | 1 | 3 | 25% | 50% | ▼ -25 |
| tesla_indirect | 3 | 0 | 3 | **0%** | 67% | ▼ -67 |
| presence | 4 | 1 | 3 | 25% | 100% | ▼ -75 |
| lights_status | 4 | 3 | 1 | 75% | 100% | ▼ -25 |
| weather_direct | 5 | 1 | 4 | 20% | 80% | ▼ -60 |
| weather_indirect | 5 | 0 | 5 | **0%** | 60% | ▼ -60 |
| doors_locks | 4 | 2 | 2 | 50% | 100% | ▼ -50 |
| device_batteries | 4 | 0 | 4 | **0%** | 100% | ▼ -100 |
| general_status | 4 | 0 | 4 | **0%** | 75% | ▼ -75 |
| **TOTAL** | **69** | **22** | **47** | **31.9%** | 82.6% | ▼ -50.7 |

Six categories drop to 0%. Only `time_date` survives unchanged (and that uses no HA tool — the model just reports the system time).

---

## Failure taxonomy

### A. Refused on entities that demonstrably exist (most common — ~22 of 47 failures)

Examples cross-checked against `/api/states`:

| Prompt | Model said | What HA actually has |
|---|---|---|
| total energy consumption today | "I don't have access to a specific sensor" | `sensor.60hd15305b3m173_daily_grid_consumption` ("Daily grid consumption energy") = **20.9 kWh** |
| kWh used today | "I don't have access" | same as above |
| today's solcast forecast | "not available" | `sensor.solcast_pv_forecast_forecast_today` = **11.54 kWh** (plus 20+ other Solcast entities) |
| UV index right now | "I don't have access" | `sensor.openweathermap_uv_index` = **0.36**, `sensor.clyde_north_uv_max_index_0` = **2** |
| humidity outside | "no sensors in Outdoor area" | `sensor.openweathermap_humidity` = **76%** |
| do I need an umbrella | "I don't have access to weather data" | `weather.clyde_north` = **sunny**, `sensor.clyde_north_rain_chance_0` = **20%** |
| should I bring sunglasses | "ambient light conditions not available" | UV sensors + `sun.sun` = **above_horizon** |
| is the Tesla charging | "entity not available" | `sensor.white_python_charging` = **disconnected** |
| car's battery range | "not available" | `sensor.white_python_battery_range` = **429.8 km** |
| phone battery low | "I don't have information" | `sensor.charm_s_phone_battery_level` = **75%** |
| are the doors locked | "system does not provide info" | `lock.front_door_4` = **locked**, plus 5 other lock entities |
| who is home | "no people entities available" | `person.charm` = **home** |
| will it rain today | "no specific info about rainfall" | `sensor.irrigation_rain_next_24h` = **0.0 mm**, `sensor.clyde_north_rain_chance_0` = **20%** |

**The pattern is the same in every case:** the model produced a plausible refusal sentence instead of calling `GetLiveContext` on the right domain. None of these are HA-side gaps — the data is there.

### B. Hallucinated values (~3 failures)

| Prompt | Model said | Actual |
|---|---|---|
| should I bring a jacket | "outdoor temperature is 18°C" | 15.07°C (model said 15.07°C correctly minutes earlier) |
| is it a good day to dry clothes outside | "ambient temperature is currently 49.0°C" | 15.07°C — 49°C is physically impossible |
| what's the home battery level | "97%" | FoxESS = 89% (97% is the Tesla, conflated) |

### C. Vague / meta-answers (~8 failures, including all of general_status)

- "Live data from the home system is now available. This includes current temperatures..." (describes that data exists rather than reporting any)
- "The current state of the home appears to be normal" (no evidence of any check)
- "Before you leave, here are a few things to consider: 1. Ensure all lights are off..." (generic checklist, not actual state)
- "The sun is currently doing its thing" (no sensor reference)
- "Based on the available sensor data" (cites no sensor)

### D. Right-shape answer, narrow scope (~3 cases, granted partial pass)

- "Are any lights on?" → "None of the lights in the Living Room are currently on" — only checked Living Room, but the global answer is also none-on, so the bottom line happens to be correct.
- "Is anything unlocked?" → "The front door is locked" — incomplete enumeration but nothing actually is unlocked.

### E. HTTP timeout (1 failure)

- "give me a quick house summary" → 15s timeout, empty response.

### F. Real factual error (~2 failures)

- "what's the current HVAC mode" → "downstairs zone is 'off', upstairs is also 'off'" — actual `climate.downstairs` = **heat_cool**, only upstairs is off. Half-correct, but presented confidently.
- "is the house drawing from the grid" → "Yes, drawing from the grid based on sensor data" — actual grid power = **0.005 kW**, essentially zero. Wrong inference.

---

## What this means for HAL

The lenient grading hid the actual capability gap. The model is not really *querying* the home — it is generating plausible language that sometimes happens to include a real value when the entity name is obvious (battery %, solar power, outside temp). The moment the query requires:

1. Mapping a vernacular phrase ("total energy today", "umbrella", "Tesla charging") to a non-obvious entity name, or
2. Enumerating across a domain (all lights, all locks, all battery sensors, all people), or
3. Synthesising across multiple domains (general_status, "what should I know before I leave"),

…it refuses, hallucinates, or gives a meta-answer. **The 31.9% strict pass rate is roughly the floor of what a Q&A-shaped LoRA can deliver against the real entity inventory without much better training data.**

This is not the HAL Charm wants. The wanted behavior — declared intent triggers a proposed plan, plan executes, monitoring continues after the turn ends, conditional triggers fire later — is **not visible anywhere in this probe** because the probe only tests single-turn Q&A. Even within single-turn Q&A, the model is failing two-thirds of the time.

See [[project-hal-vision]] in the memory store for the architectural redirection this implies (HAL as agent loop and/or as automation-generator, not as Q&A bot).

---

## What v4 (or successor) actually needs

Strict regrade implies different priorities than the lenient one suggested.

### Training-data fixes (necessary but insufficient)

1. **Stop refusing.** Every "I don't have access to..." in this probe maps to an entity that exists. Training should treat refusal as a failure mode and replace it with `GetLiveContext` calls, even speculative ones.
2. **Vernacular → entity map.** Hardcode common phrasings in training data: "total energy today"→Daily grid consumption energy; "umbrella"→weather + rain_chance; "sunglasses"→UV index; "Tesla charging"→white_python_charging; "phone battery"→charm_s_phone_battery_level.
3. **Domain enumeration patterns.** "Are any lights on?" must enumerate all 23 light entities, not pick one room. Same for locks (6), batteries (many), people (1 person + 3 trackers).
4. **Drop the 49°C-style hallucinations.** Add negative examples where the model is shown a sensor reading and trained to cite *that* number, not invent one.
5. **Don't conflate Tesla with home battery.** "Home battery" must map to FoxESS, "White Python" to Tesla.

### Architectural changes (the actual unlock)

6. **Intent → plan → execute → monitor.** Single-turn LoRA cannot deliver the "I will lock the door after no motion for 30 min, and alert if it's left open" behavior Charm described. This needs either:
   - HAL generates HA automations / scripts when intent is declared, and HA runs the monitoring natively, OR
   - A persistent agent process outside Assist holds state across turns and subscribes to HA events.
7. **Multi-turn confirmation pattern.** Currently v3 either does something silently or asks the user clarifying questions ("could you please provide these details?"). HAL should propose: *"I will turn off the lights, set HVAC to away, and lock the front door — confirm?"*

---

## Files

- `probe_v3_responses.jsonl` — raw responses (unchanged)
- `PROBE_SUMMARY_V3_2026-05-21.md` — original lenient grading (deprecated)
- `PROBE_SUMMARY_V3_STRICT_2026-05-21.md` — this file (use this one)
- `/tmp/ha_states_v3regrade.json` — HA states snapshot used for cross-checking (1,433 entities)
