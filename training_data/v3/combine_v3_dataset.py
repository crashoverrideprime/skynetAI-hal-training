#!/usr/bin/env python3
"""
Combine all v3 training data sources into hal_training_v3.jsonl.

v3 = (v2 base, minus prompts that overlap the failure corrections)
   + corrections_2026-05-19.jsonl
   + nl_sensor_mapping.jsonl
   + disambiguation.jsonl
   + persona_hal9000.jsonl

Output: hal_training_v3.jsonl
"""

import json
import re
import sys
from pathlib import Path

BASE_DIR = Path("/mnt/zardos/charm-hal-env/training_data")
V2_JSONL = BASE_DIR / "v2" / "hal_training_v2.jsonl"
V3_DIR   = BASE_DIR / "v3"

SOURCES = [
    V3_DIR / "corrections_2026-05-19.jsonl",
    V3_DIR / "nl_sensor_mapping.jsonl",
    V3_DIR / "disambiguation.jsonl",
    V3_DIR / "persona_hal9000.jsonl",
]

OUTPUT   = V3_DIR / "hal_training_v3.jsonl"

def get_user_prompt(ex):
    for m in ex.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "").strip().lower()
    return ""

def validate_example(ex):
    """Check role ordering and required fields."""
    msgs = ex.get("messages", [])
    if not msgs:
        return False
    if msgs[0].get("role") != "system":
        return False
    roles = [m["role"] for m in msgs]
    # All valid roles
    for r in roles:
        if r not in ("system", "user", "assistant", "tool"):
            return False
    # No consecutive duplicate roles (except tool after assistant with tool_calls)
    for i in range(1, len(roles)):
        if roles[i] == roles[i-1] and roles[i] != "tool":
            return False
    return True

# ── Step 1: collect all new v3 prompts ─────────────────────────────────────
new_examples = []
new_prompts  = set()
source_counts = {}

for src in SOURCES:
    if not src.exists():
        print(f"MISSING: {src}", flush=True)
        continue
    count = 0
    for line in open(src):
        line = line.strip()
        if not line:
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  JSON error in {src.name}: {e}", flush=True)
            continue
        if not validate_example(ex):
            print(f"  Invalid example in {src.name}, skipping", flush=True)
            continue
        p = get_user_prompt(ex)
        if p not in new_prompts:
            new_examples.append(ex)
            new_prompts.add(p)
            count += 1
    source_counts[src.name] = count
    print(f"  Loaded {count} from {src.name}", flush=True)

print(f"\nNew v3 examples: {len(new_examples)}", flush=True)

# ── Step 2: load v2 base, excluding overlapping prompts ────────────────────
v2_examples = []
v2_skipped  = 0
for line in open(V2_JSONL):
    line = line.strip()
    if not line:
        continue
    try:
        ex = json.loads(line)
    except json.JSONDecodeError:
        continue
    p = get_user_prompt(ex)
    if p in new_prompts:
        v2_skipped += 1
        continue
    if not validate_example(ex):
        v2_skipped += 1
        continue
    v2_examples.append(ex)

print(f"V2 base examples: {len(v2_examples)} (skipped {v2_skipped} overlapping/invalid)", flush=True)

# ── Step 3: combine and write ──────────────────────────────────────────────
all_examples = v2_examples + new_examples
print(f"Total combined: {len(all_examples)}", flush=True)

import random
random.seed(42)
random.shuffle(all_examples)

with open(OUTPUT, "w") as f:
    for ex in all_examples:
        f.write(json.dumps(ex) + "\n")

print(f"\nOutput: {OUTPUT}", flush=True)
print(f"Lines: {len(all_examples)}", flush=True)
print(f"\nBreakdown:", flush=True)
print(f"  V2 base (minus overlaps): {len(v2_examples)}", flush=True)
for name, count in source_counts.items():
    print(f"  {name}: {count}", flush=True)
