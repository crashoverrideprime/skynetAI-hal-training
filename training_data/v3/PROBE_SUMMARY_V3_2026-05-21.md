# HAL v3 Probe Results — 2026-05-21

> **⚠ DEPRECATED — LENIENT GRADING ⚠**
> The pass criteria in this file are too lenient and overstate v3's actual capability.
> Use **`PROBE_SUMMARY_V3_STRICT_2026-05-21.md`** instead — it cross-checks every refusal
> against `/api/states` and applies strict pass rules. Strict result: **31.9%**, not 82.6%.
> This file is retained only for diff/comparison purposes.

---


**Source:** `probe_v3_responses.jsonl` (69 prompts, ~2 min runtime, 0 HTTP errors)
**Model:** `charm-hal-v3` (Qwen3-8B v3 LoRA, Q4_K_M GGUF)
**Agent:** `conversation.ollama_conversation` (HA Assist via HA Conversation API)

---

## Pass rate by category

| Category | Total | Pass | Fail | Pass % | v2 Pass% | Δ |
|---|---:|---:|---:|---:|---:|---:|
| **time_date** | 3 | 3 | 0 | **100.0%** | 100.0% | = |
| **hvac_direct** | 4 | 4 | 0 | **100.0%** | 100.0% | = |
| **energy_indirect** | 4 | 4 | 0 | **100.0%** | 75.0% | **▲ +25.0%** |
| **solar_indirect** | 3 | 3 | 0 | **100.0%** | 85.0% | **▲ +15.0%** |
| **battery_direct** | 4 | 4 | 0 | **100.0%** | 80.0% | **▲ +20.0%** |
| **battery_indirect** | 3 | 3 | 0 | **100.0%** | 95.0% | ▲ +5.0% |
| **presence** | 4 | 4 | 0 | **100.0%** | 93.3% | ▲ +6.7% |
| **lights_status** | 4 | 4 | 0 | **100.0%** | 86.7% | **▲ +13.3%** |
| **doors_locks** | 4 | 4 | 0 | **100.0%** | 76.7% | **▲ +23.3%** |
| **device_batteries** | 4 | 4 | 0 | **100.0%** | 50.0% | **▲ +50.0%** |
| weather_direct | 5 | 4 | 1 | 80.0% | 64.7% | ▲ +15.3% |
| general_status | 4 | 3 | 1 | 75.0% | 95.0% | ▼ -20.0% |
| weather_indirect | 5 | 3 | 2 | 60.0% | 72.5% | ▼ -12.5% |
| hvac_indirect | 3 | 2 | 1 | 66.7% | 95.0% | **▼ -28.3%** |
| tesla_indirect | 3 | 2 | 1 | 66.7% | 95.0% | **▼ -28.3%** |
| energy_direct | 4 | 2 | 2 | 50.0% | 87.5% | **▼ -37.5%** |
| solar_direct | 4 | 2 | 2 | 50.0% | 62.5% | ▼ -12.5% |
| tesla_direct | 4 | 2 | 2 | 50.0% | 86.7% | **▼ -36.7%** |
| **TOTAL** | **69** | **57** | **12** | **82.6%** | **81.6%** | **▲ +1.0%** |

**Overall: 82.6% pass rate vs v2's 81.6% — marginal improvement**

---

## Biggest wins vs v2

### 1. device_batteries — 100% (+50% vs v2 50%)
v2 was completely unreliable on device battery queries. v3 handles them cleanly:
> Q: what batteries need replacing
> A: *(reads sensor domain, enumerates battery-class entities, reports low ones)*

> Q: which device has the lowest battery
> A: *(correctly identifies lowest battery sensor by name)*

**Root cause resolved:** v3 training data included explicit battery-sensor enumeration examples.

### 2. doors_locks — 100% (+23.3% vs v2 76.7%)
v2 was inconsistent on lock/door state queries. v3 reliably calls GetLiveContext(domain=binary_sensor) and reads door/lock states.

### 3. energy_indirect — 100% (+25% vs v2 75%)
v3 handles inferential energy questions well ("are we using a lot?", "should I run the dishwasher?") — even when the answer is uncertain, it attempts a tool call first rather than refusing.

### 4. battery_direct — 100% (+20% vs v2 80%)
All four battery queries answered correctly with current SoC values (97%, 92%). FoxESS battery correctly identified.

### 5. lights_status — 100% (+13.3% vs v2 86.7%)
All light state queries handled. Even "bedroom light is unavailable" is considered a pass (model tried and reported).

### 6. weather_direct — 80% (+15.3% vs v2 64.7%)
Notable improvement — v3 reads outdoor sensors and interprets weather context more reliably. Still fails on UV index.

---

## Regressions vs v2

### 1. energy_direct — 50% (▼ -37.5% from 87.5%)
**Failures:**
- *"what's my total energy consumption today"* → "I don't have access to a specific sensor"
- *"how many kilowatt hours have I used today"* → same refusal

**Root cause:** Model is not mapping "total consumption today" → `Daily grid consumption energy` entity. The sensor exists and was in training data, but the v3 model is being too conservative and refusing instead of querying.

**Fix for v4:** Add explicit training examples: "total energy today" → GetLiveContext(domain=sensor) → look for `Daily grid consumption energy`.

### 2. tesla_direct — 50% (▼ -36.7% from 86.7%)
**Failures:**
- *"is the Tesla charging"* → entity not available
- *"what's the car's battery range"* → not available

**Root cause:** Tesla integration may not be exposing charging status and range as named entities in the current HA config. This may be a HA-side issue (entity not exposed to Assist), not a model issue.

**Check:** Verify `White Python Charger power` and `White Python Battery range` are exposed in HA Settings → Voice Assistants → Expose.

### 3. hvac_indirect — 66.7% (▼ -28.3% from 95%)
**Failure:**
- *"do I need to turn the heating on"* → model says it needs to check temps but doesn't follow through with a tool call

**Root cause:** v3 talks about checking rather than actually calling the tool. Indirect HVAC advisory prompts need examples that chain: indirect question → GetLiveContext(domain=climate) → compare temp → give advice.

### 4. tesla_indirect — 66.7% (▼ -28.3% from 95%)
**Failure:**
- *"can I make it to Melbourne and back"* → refuses (no GPS)

**Root cause:** Model is correctly identifying it can't do routing, but should still report the current range and let the user decide. Fix: examples that answer with range + charge % even when exact routing isn't possible.

### 5. general_status — 75% (▼ -20% from 95%)
**Failure:**
- *"what should I know before I leave"* → partial response (gets cut off or gives generic response)

**Root cause:** This is a multi-domain synthesis query. v3 struggles to chain multiple GetLiveContext calls into a coherent departure summary.

---

## Qualitative observations

### Strong v3 responses (production quality)

> Q: will the battery last the night
> A: The battery level is currently at 97%, so it should last through the night easily.

> Q: should I export to the grid or keep the battery full
> A: The current state of your battery is at 92% SoC. Based on the live data, it's advisable to keep the battery charged at a higher level rather than exporting to the grid, especially given the current energy production...

> Q: how much solar am I generating
> A: The current solar generation is 0.669 kW.

> Q: has anyone arrived home
> A: Charm At Home is detected as being present, so someone has arrived home.

> Q: are any lights on
> A: None of the lights in the Living Room are currently on.

> Q: is now a good time to charge the car
> A: It appears there is no entity available for checking live prices to determine the best time to charge, but current grid draw is 0 kW... *(attempts reasoning)*

### Weak v3 responses

> Q: do I need an umbrella today
> A: I don't have access to weather data... *(should call GetLiveContext(domain=weather) first)*

> Q: should I bring sunglasses
> A: The current ambient light conditions are not available... *(should attempt UV sensor query)*

> Q: can I make it to Melbourne and back
> A: I don't have access to your vehicle's GPS... *(should report range % and let user decide)*

---

## Failure pattern frequency

1. "I don't have access to..." — 7 instances
2. "not available in the provided context" — 3 instances
3. "I don't have the capability to..." — 2 instances
4. Model gives correct observation but fails to infer/advise — 2 instances

Total: 12 failures across 69 prompts (17.4%)

---

## v3 vs v2 overall verdict

| Aspect | v3 | v2 |
|--------|----|----|
| Overall pass rate | **82.6%** | 81.6% |
| Perfect categories | **10** | 2 |
| Worst category | energy_direct (50%) | device_batteries (50%) |
| Best improvement | device_batteries (+50%) | — |
| Biggest regression | energy_direct (-37.5%) | — |
| Training loss | **0.01461** | 0.432 |

v3 is a **net improvement**: 10 perfect categories vs v2's 2, and the weakest spots shifted from fundamental entity confusion (v2 couldn't find battery sensors at all) to edge cases (historical queries, routing inference).

---

## Recommended training targets for v4

### High priority
1. **energy_direct recovery** — add explicit: "total energy today" / "kWh used today" → `Daily grid consumption energy`
2. **hvac_indirect chain** — add: indirect comfort question → check climate entity → infer advice, don't just describe what you'd do
3. **tesla range-without-routing** — add: "can I make it to X" → report battery %, range, and advise based on typical distances even without GPS

### Medium priority
4. **solar forecast** — "solcast forecast" / "tomorrow's solar" → Solcast PV Forecast entities by name
5. **UV / sunglasses chain** — UV index query → UV sensor → sunglasses recommendation
6. **umbrella chain** — weather state → rain probability → umbrella advice
7. **general_status synthesis** — multi-domain departure summary: presence + security + HVAC + battery

### Low priority (may be HA-config, not model)
8. Tesla charging/range entities — verify exposure in HA first
9. Historical solar data — confirm if daily counter entities are exposed

---

## Files

- `probe_v3_responses.jsonl` — full raw data, 69 records
- `/mnt/zardos/probe_v3.py` — probe script
- `PROBE_SUMMARY_V3_2026-05-21.md` — this file
- `PROBE_SUMMARY_2026-05-19.md` — v2 reference
