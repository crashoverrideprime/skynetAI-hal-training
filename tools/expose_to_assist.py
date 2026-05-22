#!/usr/bin/env python3
"""
Bulk-expose meaningful Home Assistant entities to the `conversation` assistant.

Why this exists: HAL has ~1,977 entities in the registry but only a small subset
is exposed to Assist by default. Exposure is per-entity-per-assistant and lives in
the entity registry's `options.conversation.should_expose` field. The strict v3
probe failed on several entities (UV index, daily grid consumption, phone battery,
person.charm, Tesla charging/range) that exist but were not exposed — the model
could not see them regardless of training quality.

This script:
  1. Connects to HA's WebSocket API and authenticates.
  2. Fetches entity registry + current exposure + states.
  3. Classifies each entity as meaningful or not via domain whitelist + filters.
  4. Prints a dry-run summary (per-domain counts, sample names).
  5. With --apply, exposes the new entities in batches; saves before/after JSON.

Default is dry-run. Default is additive (never unexposes). Idempotent.

Usage:
  ./expose_to_assist.py                       # dry-run, print plan only
  ./expose_to_assist.py --apply               # actually expose
  ./expose_to_assist.py --apply --yes         # skip y/N confirmation
  ./expose_to_assist.py --apply --unexpose-everything-else
                                              # cleanup: unexpose anything not meaningful
  ./expose_to_assist.py --only sensor light   # restrict to specific domains
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import websockets

HA_WS_URL    = "ws://homeassistant.local:8123/api/websocket"
SECRETS_FILE = Path("/mnt/zardos/!secrets")
OUT_DIR      = Path("/mnt/zardos/charm-hal-env/tools")
ASSISTANT    = "conversation"
BATCH_SIZE   = 100


# ─── Inclusion rules ────────────────────────────────────────────────────────────

# Domains where every (enabled, non-hidden) entity should be exposed by default.
ALWAYS_INCLUDE_DOMAINS = {
    "light", "switch", "cover", "lock", "climate", "fan",
    "vacuum", "media_player", "scene", "valve",
    "weather", "person", "sun", "alarm_control_panel",
}

# Domains never exposed (infrastructure, control surfaces, config storage, etc.)
NEVER_INCLUDE_DOMAINS = {
    "update", "button", "event", "image",
    "stt", "tts", "assist_satellite", "conversation",
    "automation", "pyscript", "script",
    "input_text", "input_select", "input_datetime", "input_number", "input_boolean",
    "camera", "number", "select", "time", "timer", "todo",
    "calendar",
}

# Sensor device_classes that are user-meaningful for voice queries.
USEFUL_SENSOR_DEVICE_CLASSES = {
    "battery", "temperature", "humidity", "power", "energy",
    "current", "voltage", "illuminance", "atmospheric_pressure", "pressure",
    "pm25", "pm10", "aqi", "uv_index",
    "moisture", "gas", "water", "monetary", "duration",
    "wind_speed", "precipitation", "precipitation_intensity",
    "speed", "distance", "data_rate", "frequency",
}

# Friendly-name substrings (case-insensitive) that promote a sensor even if its
# device_class isn't in the useful set above. Targets known high-value entities.
SENSOR_NAME_PROMOTE = [
    r"\bdaily[_\s]grid\b",
    r"\bsolcast\b",
    r"\bsolar\b",
    r"\bpv[_\s]power\b",
    r"\bpv[_\s]forecast\b",
    r"\bgrid[_\s]consumption\b",
    r"\bgrid[_\s]export\b",
    r"\bbattery[_\s]soc\b",
    r"\bsoc\b",
    r"\bcharge[_\s]rate\b",
    r"\bcharger[_\s]power\b",
    r"\bcharging\b",
    r"\bbattery[_\s]range\b",
    r"\brange[_\s]remaining\b",
    r"\boutdoor\b",
    r"\bambient\b",
    r"\brain\b",
    r"\bwind\b",
    r"\buv\b",
    r"\bdew[_\s]point\b",
    r"\bforecast\b",
    r"\btoday\b",
    r"\byesterday\b",
    r"\bphone\b",
    r"\bclyde[_\s]north\b",
    r"\bopenweathermap\b",
    r"\bwhite[_\s]python\b",
    r"\bfoxess\b",
    r"\bskynet\b",
    r"\bamber\b",
]
SENSOR_NAME_PROMOTE_RX = [re.compile(p, re.I) for p in SENSOR_NAME_PROMOTE]

# Binary sensor device_classes worth exposing.
USEFUL_BINARY_SENSOR_DEVICE_CLASSES = {
    "motion", "door", "window", "garage_door", "lock",
    "presence", "occupancy", "moisture", "smoke", "gas",
    "safety", "tamper", "sound", "vibration",
    "opening", "battery",
}

# device_tracker: only the genuinely useful trackers (phones, vehicles), not stale.
DEVICE_TRACKER_NAME_PROMOTE_RX = [
    re.compile(r"_phone$", re.I),
    re.compile(r"_location$", re.I),
    re.compile(r"_route$", re.I),
    re.compile(r"\bcharm\b", re.I),
    re.compile(r"\bwhite[_\s]python\b", re.I),
]

# Entity-name suffixes/substrings to *always* skip (diagnostic noise).
ENTITY_NAME_BLOCK_RX = [
    re.compile(p, re.I) for p in [
        r"_signal_strength$",
        r"_last_seen$",
        r"_last_update$",
        r"_uptime$",
        r"_software_version$",
        r"_firmware",
        r"_ip_address$",
        r"_mac_address$",
        r"_ssid$",
        r"_rssi$",
        r"_link_quality$",
        r"_lqi$",
        r"_update_available$",
        r"_diagnostic",
        r"_identify$",
        r"_factory_reset$",
        r"_reboot$",
        r"_restart$",
    ]
]


# ─── Auth + WS client ───────────────────────────────────────────────────────────


def load_bearer_token() -> str:
    for line in SECRETS_FILE.read_text().splitlines():
        if "Bearer " in line:
            return line.split("Bearer ", 1)[1].split()[0].strip()
    raise SystemExit(f"Could not find a 'Bearer <token>' line in {SECRETS_FILE}")


class HAClient:
    def __init__(self, ws):
        self.ws = ws
        self._next_id = 1

    async def cmd(self, payload: dict) -> dict:
        msg_id = self._next_id
        self._next_id += 1
        await self.ws.send(json.dumps({**payload, "id": msg_id}))
        while True:
            raw = await self.ws.recv()
            data = json.loads(raw)
            if data.get("id") == msg_id:
                return data

    async def get_entity_registry(self) -> list[dict]:
        r = await self.cmd({"type": "config/entity_registry/list"})
        if not r.get("success"):
            raise RuntimeError(f"entity_registry/list failed: {r}")
        return r["result"]

    async def get_exposed(self) -> dict[str, dict[str, bool]]:
        r = await self.cmd({"type": "homeassistant/expose_entity/list"})
        if not r.get("success"):
            raise RuntimeError(f"expose_entity/list failed: {r}")
        return r["result"]["exposed_entities"]

    async def get_states(self) -> list[dict]:
        r = await self.cmd({"type": "get_states"})
        if not r.get("success"):
            raise RuntimeError(f"get_states failed: {r}")
        return r["result"]

    async def expose(self, entity_ids: list[str], should_expose: bool) -> dict:
        return await self.cmd({
            "type": "homeassistant/expose_entity",
            "assistants": [ASSISTANT],
            "entity_ids": entity_ids,
            "should_expose": should_expose,
        })


# ─── Classification ─────────────────────────────────────────────────────────────


def domain_of(entity_id: str) -> str:
    return entity_id.split(".", 1)[0]


def is_blocked_by_name(entity_id: str, friendly_name: str | None) -> bool:
    target = entity_id
    name_target = (friendly_name or "").lower()
    for rx in ENTITY_NAME_BLOCK_RX:
        if rx.search(target) or rx.search(name_target):
            return True
    return False


def classify_entity(reg: dict, state: dict | None) -> tuple[bool, str]:
    """
    Returns (should_be_exposed, reason).

    `reg` is a registry entry; `state` is the corresponding state dict if available.

    Note on `entity_category=diagnostic`: HA marks many user-valuable entities
    diagnostic (phone battery, Tesla location, Tesla range, time-to-full-charge).
    Diagnostic alone is NOT a reason to reject — instead rely on the name
    blocklist for noise patterns and domain/device_class rules for relevance.
    """
    eid    = reg["entity_id"]
    domain = domain_of(eid)
    fn     = reg.get("name") or (state.get("attributes", {}).get("friendly_name") if state else None)
    dc     = (state.get("attributes", {}).get("device_class") if state else None) or reg.get("device_class")

    # hard registry skips
    if reg.get("disabled_by") is not None:
        return False, f"disabled_by={reg['disabled_by']}"
    if reg.get("hidden_by") is not None:
        return False, f"hidden_by={reg['hidden_by']}"
    if reg.get("entity_category") == "config":
        return False, "entity_category=config"
    if is_blocked_by_name(eid, fn):
        return False, "name_blocklist"

    if domain in NEVER_INCLUDE_DOMAINS:
        return False, f"domain_blocked={domain}"
    if domain in ALWAYS_INCLUDE_DOMAINS:
        return True, f"domain_allowlist={domain}"

    if domain == "sensor":
        if dc in USEFUL_SENSOR_DEVICE_CLASSES:
            return True, f"sensor.device_class={dc}"
        haystack = f"{eid} {fn or ''}"
        for rx in SENSOR_NAME_PROMOTE_RX:
            if rx.search(haystack):
                return True, f"sensor.name_match={rx.pattern}"
        return False, "sensor.no_useful_class_or_name"

    if domain == "binary_sensor":
        if dc in USEFUL_BINARY_SENSOR_DEVICE_CLASSES:
            return True, f"binary_sensor.device_class={dc}"
        return False, "binary_sensor.no_useful_class"

    if domain == "device_tracker":
        haystack = f"{eid} {fn or ''}"
        for rx in DEVICE_TRACKER_NAME_PROMOTE_RX:
            if rx.search(haystack):
                return True, f"device_tracker.name_match={rx.pattern}"
        return False, "device_tracker.no_match"

    return False, f"domain_not_in_rules={domain}"


# ─── Main ───────────────────────────────────────────────────────────────────────


async def amain(args: argparse.Namespace) -> int:
    token = load_bearer_token()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    async with websockets.connect(HA_WS_URL, max_size=20 * 1024 * 1024) as ws:
        # auth
        await ws.recv()  # auth_required
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        auth_ok = json.loads(await ws.recv())
        if auth_ok.get("type") != "auth_ok":
            print(f"auth failed: {auth_ok}", file=sys.stderr)
            return 2
        print(f"Connected to HA {auth_ok.get('ha_version')}")

        ha = HAClient(ws)
        registry = await ha.get_entity_registry()
        exposed  = await ha.get_exposed()
        states   = await ha.get_states()

        states_by_id = {s["entity_id"]: s for s in states}
        registry_by_id = {r["entity_id"]: r for r in registry}

        # save before-snapshot for rollback / diffing
        before_path = OUT_DIR / f"exposure_before_{timestamp}.json"
        before_path.write_text(json.dumps(exposed, indent=2))

        # ─ classify ────────────────────────────────────────────────────────────
        domain_filter = set(args.only) if args.only else None
        plan_expose:   list[str] = []  # not currently exposed but should be
        plan_unexpose: list[str] = []  # currently exposed but should not be (only used with --unexpose-everything-else)
        already_ok:    list[str] = []  # exposed and meaningful
        rejected:      dict[str, list[str]] = defaultdict(list)  # reason → entity_ids

        for reg in registry:
            eid    = reg["entity_id"]
            domain = domain_of(eid)
            if domain_filter and domain not in domain_filter:
                continue

            currently_exposed = exposed.get(eid, {}).get(ASSISTANT, False) is True
            keep, reason = classify_entity(reg, states_by_id.get(eid))

            if keep:
                if currently_exposed:
                    already_ok.append(eid)
                else:
                    plan_expose.append(eid)
            else:
                if currently_exposed:
                    plan_unexpose.append(eid)
                rejected[reason].append(eid)

        # ─ report ──────────────────────────────────────────────────────────────
        def by_domain(eids: list[str]) -> dict[str, int]:
            return dict(Counter(domain_of(e) for e in eids).most_common())

        print()
        print("=" * 78)
        print(f"  Assist exposure plan for assistant={ASSISTANT!r}")
        print("=" * 78)
        print(f"  Entity registry:               {len(registry):>4}")
        print(f"  Currently exposed (meaningful):{len(already_ok):>4}")
        print(f"  TO EXPOSE (new):               {len(plan_expose):>4}")
        print(f"  Already exposed but rejected:  {len(plan_unexpose):>4}  "
              f"(would be un-exposed only with --unexpose-everything-else)")
        print(f"  Total rejected:                {sum(len(v) for v in rejected.values()):>4}")
        print()

        print("  TO EXPOSE per domain:")
        for d, n in by_domain(plan_expose).items():
            print(f"    {d:24s} {n}")
        print()

        print("  Sample of TO EXPOSE (first 25, alphabetised):")
        for eid in sorted(plan_expose)[:25]:
            st = states_by_id.get(eid, {})
            fn = st.get("attributes", {}).get("friendly_name", "")
            val = str(st.get("state", ""))[:30]
            print(f"    {eid:60s}  {fn:35s}  = {val}")
        if len(plan_expose) > 25:
            print(f"    ... and {len(plan_expose) - 25} more")
        print()

        print("  Rejection reasons (top 15):")
        rejection_summary = sorted(rejected.items(), key=lambda kv: -len(kv[1]))[:15]
        for reason, eids in rejection_summary:
            ex = ", ".join(eids[:3])
            print(f"    {len(eids):>4}  {reason:40s}  e.g. {ex}")
        print()

        if args.unexpose_everything_else and plan_unexpose:
            print("  UNEXPOSE (currently exposed, will be removed):")
            for eid in plan_unexpose[:20]:
                print(f"    - {eid}")
            if len(plan_unexpose) > 20:
                print(f"    ... and {len(plan_unexpose) - 20} more")
            print()

        # save plan
        plan_path = OUT_DIR / f"exposure_plan_{timestamp}.json"
        plan_payload = {
            "timestamp": timestamp,
            "assistant": ASSISTANT,
            "domain_filter": sorted(domain_filter) if domain_filter else None,
            "already_ok": already_ok,
            "to_expose": plan_expose,
            "to_unexpose_if_flagged": plan_unexpose,
            "rejected_by_reason": {r: eids for r, eids in rejected.items()},
        }
        plan_path.write_text(json.dumps(plan_payload, indent=2))
        print(f"  Plan saved to {plan_path}")
        print(f"  Pre-change exposure snapshot saved to {before_path}")
        print()

        # ─ apply ───────────────────────────────────────────────────────────────
        if not args.apply:
            print("DRY-RUN. Pass --apply to actually expose these entities.")
            return 0

        if not args.yes:
            confirm = input(f"Apply changes ({len(plan_expose)} expose, "
                            f"{len(plan_unexpose) if args.unexpose_everything_else else 0} unexpose)? [y/N] ")
            if confirm.strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return 1

        # expose new
        for i in range(0, len(plan_expose), BATCH_SIZE):
            batch = plan_expose[i:i + BATCH_SIZE]
            r = await ha.expose(batch, should_expose=True)
            if not r.get("success"):
                print(f"  FAILED batch {i // BATCH_SIZE} ({len(batch)} entities): {r.get('error')}")
                return 3
            print(f"  exposed batch {i // BATCH_SIZE + 1} ({len(batch)} entities) ✓")

        # optionally unexpose
        if args.unexpose_everything_else and plan_unexpose:
            for i in range(0, len(plan_unexpose), BATCH_SIZE):
                batch = plan_unexpose[i:i + BATCH_SIZE]
                r = await ha.expose(batch, should_expose=False)
                if not r.get("success"):
                    print(f"  FAILED unexpose batch {i // BATCH_SIZE}: {r.get('error')}")
                    return 3
                print(f"  unexposed batch {i // BATCH_SIZE + 1} ({len(batch)} entities) ✓")

        # save after-snapshot
        exposed_after = await ha.get_exposed()
        after_path = OUT_DIR / f"exposure_after_{timestamp}.json"
        after_path.write_text(json.dumps(exposed_after, indent=2))
        print()
        print(f"  Post-change snapshot saved to {after_path}")

        # quick verification: count exposed for our assistant
        count_after = sum(1 for v in exposed_after.values() if v.get(ASSISTANT, False) is True)
        count_before = sum(1 for v in exposed.values() if v.get(ASSISTANT, False) is True)
        print(f"  Exposure to '{ASSISTANT}': {count_before} → {count_after} "
              f"(Δ {count_after - count_before:+d})")

        return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true",
                   help="Actually apply changes. Without this, dry-run only.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the y/N confirmation prompt when --apply is set.")
    p.add_argument("--unexpose-everything-else", action="store_true",
                   help="Also unexpose entities that are currently exposed but classified as not meaningful. "
                        "Default is additive only (never unexpose).")
    p.add_argument("--only", nargs="*", metavar="DOMAIN",
                   help="Restrict to these domains (e.g. --only sensor light). Default: all domains.")
    args = p.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
