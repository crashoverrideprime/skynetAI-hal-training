# HAL v2 Probe Results — 2026-05-19

**Source:** `probe_responses_2026-05-19.jsonl` (521 prompts, 22.3 min runtime, 0 HTTP errors)
**Model:** `charm-hal` (Qwen3-8B v2 LoRA, Q4_K_M GGUF)
**Agent:** `conversation.ollama_conversation` (HA Assist via HA Conversation API)

## Pass rate by category

A "fail" = response contained any of: *"I do not have access"*, *"no exposed entity"*, *"unable to"*, *"cannot find/provide"*, *"please provide/clarify"*, *"sorry"*, *"apologi[zs]e"*, *"is currently unknown"*, or empty/null response. The rest are treated as passes (may still contain wrong sensor selection — see qualitative review).

| Category | Total | Pass | Fail | Pass % |
|---|---:|---:|---:|---:|
| **time_date** | 10 | 10 | 0 | **100.0%** |
| **hvac_direct** | 40 | 40 | 0 | **100.0%** |
| general_status | 20 | 19 | 1 | 95.0% |
| battery_indirect | 20 | 19 | 1 | 95.0% |
| tesla_indirect | 20 | 19 | 1 | 95.0% |
| hvac_indirect | 20 | 19 | 1 | 95.0% |
| presence | 30 | 28 | 2 | 93.3% |
| tesla_direct | 30 | 26 | 4 | 86.7% |
| lights_status | 30 | 26 | 4 | 86.7% |
| energy_direct | 40 | 35 | 5 | 87.5% |
| solar_indirect | 20 | 17 | 3 | 85.0% |
| battery_direct | 40 | 32 | 8 | 80.0% |
| doors_locks_security | 30 | 23 | 7 | 76.7% |
| energy_indirect | 20 | 15 | 5 | 75.0% |
| weather_indirect | 40 | 29 | 11 | 72.5% |
| **weather_direct** | 51 | 33 | 18 | **64.7%** |
| **solar_direct** | 40 | 25 | 15 | **62.5%** |
| **device_batteries** | 20 | 10 | 10 | **50.0%** |
| **TOTAL** | **521** | **425** | **96** | **81.6%** |

## Strongest performers (≥95%)

- **time_date** and **hvac_direct** at 100% — these have direct, well-named entities and unambiguous mapping.
- **All indirect categories except weather/energy** scored ≥95% — meaning the model handles inferential reasoning well *once it has any relevant entity to ground on*. The model is willing to commit to advice ("not necessary to take an umbrella", "should last through the night") given the available state.

## Weakest performers (≤65%) and root causes

### 1. device_batteries — 50% fail

The model refuses or invents entity names for individual sensor batteries:
- *"The current state of Charm's phone battery is not available in the live context"*
- *"I do not have information on which battery to replace first"*
- *"I do not have a sensor named ..."*

**Root cause:** the Instructions block doesn't list specific battery sensors by name (it only mentions device groups). And the model doesn't iterate the sensor list looking for `device_class: battery`. Treats "battery" queries as referring to the home/EV battery, not peripheral devices.

**v3 training direction:** add examples that filter `domain=sensor` and post-filter the result text for `[battery]` device class entries; examples that handle "which battery is lowest" by enumerating.

### 2. solar_direct — 62.5% fail

Failures cluster around two patterns:
- **Yesterday/historical queries** — model refuses: *"I do not have access to historical solar generation data for yesterday"*. The entity `Cumulative power generation` exists but the model doesn't connect "yesterday" to it.
- **Solcast forecast queries** — model says forecast isn't available, but `Solcast PV Forecast Forecast Today`, `Solcast PV Forecast Peak Forecast Today` etc. exist in state and are exposed.

**v3 training direction:** examples that map "yesterday's solar"/"solar yield" → daily energy counters; examples that map "Solcast" / "forecast" terms to the Solcast PV Forecast family of entities by name.

### 3. weather_direct — 64.7% fail

Failures concentrated on **forecast** prompts (tomorrow, weekly, next 3 days, UV, sunrise/sunset, dew point):
- *"I do not have access to the weather forecast for tomorrow"* — but the dataset shows `pirateweather`, `home_charm`, and `OpenWeatherMap` family entities with min/max temperature, dew point, apparent temp, etc.
- *"The UV index is not available"* — there *is* a UV Index entity.

The model treats anything beyond "current temperature" as out of scope.

**v3 training direction:** examples that explicitly map "tomorrow's weather" → forecast min/max entities; "UV"/"dew point"/"sunrise" → their specific sensor names; treat OpenWeatherMap + Solcast as authoritative.

## Qualitative observations on indirect/inferential prompts

The indirect categories are the **most valuable signal** in this run because they reveal HAL's ability to chain readings into advice — which is exactly the differentiator vs. plain HA Assist.

### Excellent (training-data-quality) responses to keep

> Q: do I need an umbrella today
> A: The current temperature is 11.2 °C, with a humidity level of 83%. The dew point is at 8.42 °C. Based on these conditions, it does not appear necessary to take an umbrella today.

> Q: do I have enough charge to drive to the city
> A: The vehicle's battery level is currently at 70%, which should be sufficient for a drive to the city. However, it is always advisable to consider factors such as the distance to the city, driving conditions, and any additional energy requirements during the journey.

> Q: will the battery last the night
> A: The home battery currently has an SoC of 86%. Based on current conditions and energy usage patterns, it should last through the night without requiring additional charging from the grid.

> Q: should we eat outside tonight
> A: (HVAC state breakdown + comfort recommendation)

### Weak indirect responses

> Q: should I bring sunglasses
> A: I am unable to determine whether you should bring sunglasses at this moment...

The model gives up rather than reading the UV sensor and the current cloud cover. **v3 should teach explicit chains:** sunglasses → check `UV Index` + cloud cover; jacket → check current temp + forecast min; umbrella → check rain probability + current humidity.

## Failure-pattern frequency (top 10 phrases)

1. "I do not have access to ..." — 31 instances
2. "I am unable to find ..." / "unable to determine ..." — 18
3. "I cannot find a sensor named ..." — 11
4. "Please provide more details / clarify" — 9
5. "There is no entity named ..." — 7
6. "The data is not currently available" — 6
7. "is currently unknown" — 5
8. "I apologise / Sorry" — 4
9. Truncated mid-sentence `"...named \\` — 3 (model hallucinated entity name and the response was cut at the JSON-escaped quote)
10. *(misc)* — ≈2 each

## Files

- `probe_responses_2026-05-19.jsonl` — full raw data, one record per prompt
- `/mnt/zardos/probe_v3.py` — the probe script (prompt list + runner)
- `/mnt/zardos/probe_failures.txt` — extracted failure responses
- `/mnt/zardos/probe_cat_resp.txt` — extracted category+response pairs (one per line)

## Recommended next steps

1. **Curate this dataset for v3 training.** For each of the 96 failures, write the *correct* response by hand (or by querying the right entity manually) and add it as a v3 example. Re-use the 425 successful interactions to maintain bias.
2. **Re-run after each Instructions-field iteration.** The largest gains will come from extending the Instructions block with explicit mappings for weather forecast / solar forecast / device-battery enumeration — those alone could push pass rate above 90% without retraining.
3. **Add a system prompt rule:** "Before saying 'I do not have access', call GetLiveContext with the relevant domain and scan the entity list." Most failures are the model giving up before searching.
4. **Expand probe set for v3.5:** add 100 multi-turn prompts (follow-ups, ambiguous references like "what about tomorrow", clarification handling).
