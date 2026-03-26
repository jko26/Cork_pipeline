from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


NOVALUE = "NOVALUE"


REQUIRED_TOP_KEYS = {
    "type",
    "title",
    "address",
    "venue",
    "description",
    "links",
    "event",
}

REQUIRED_LINK_KEYS = {"website", "tickets", "reservation", "other"}

REQUIRED_EVENT_KEYS = {
    "recurring_weekly",
    "recurring_weekday",
    "start_date",
    "start_time",
    "end_date",
    "end_time",
}


def coerce_novalue_str(v: Any) -> str:
    if v is None:
        return NOVALUE
    if isinstance(v, str):
        s = v.strip()
        return s if s else NOVALUE
    return str(v)


def ensure_schema(item: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure item has all required keys and NOVALUE conventions."""

    out: Dict[str, Any] = dict(item or {})

    for k in REQUIRED_TOP_KEYS:
        out.setdefault(k, NOVALUE)

    # type
    t = out.get("type")
    if t not in ("event", "attraction"):
        # default to event if unknown
        out["type"] = "event"

    # scalar strings
    for k in ["title", "address", "venue", "description"]:
        out[k] = coerce_novalue_str(out.get(k))

    # links
    links = out.get("links")
    if not isinstance(links, dict):
        links = {}
    for k in REQUIRED_LINK_KEYS:
        links.setdefault(k, NOVALUE)

    links["website"] = coerce_novalue_str(links.get("website"))
    links["tickets"] = coerce_novalue_str(links.get("tickets"))
    links["reservation"] = coerce_novalue_str(links.get("reservation"))

    other = links.get("other")
    if not isinstance(other, list):
        other = []
    # ensure strings and cap
    other2: List[str] = []
    for x in other:
        if isinstance(x, str) and x.strip():
            if x.strip() not in other2:
                other2.append(x.strip())
        if len(other2) >= 3:
            break
    links["other"] = other2
    out["links"] = links

    # event
    ev = out.get("event")
    if not isinstance(ev, dict):
        ev = {}
    for k in REQUIRED_EVENT_KEYS:
        ev.setdefault(k, NOVALUE)

    for k in REQUIRED_EVENT_KEYS:
        ev[k] = coerce_novalue_str(ev.get(k))

    # attraction rule
    if out["type"] == "attraction":
        for k in REQUIRED_EVENT_KEYS:
            ev[k] = NOVALUE

    out["event"] = ev

    return out


def validate_array(data: Any) -> Tuple[bool, str]:
    if not isinstance(data, list):
        return False, "Top-level must be a JSON array"
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Item {i} is not an object"
        missing = REQUIRED_TOP_KEYS - set(item.keys())
        if missing:
            return False, f"Item {i} missing keys: {sorted(missing)}"
        links = item.get("links")
        if not isinstance(links, dict):
            return False, f"Item {i} links is not an object"
        if (REQUIRED_LINK_KEYS - set(links.keys())):
            return False, f"Item {i} links missing keys"
        ev = item.get("event")
        if not isinstance(ev, dict):
            return False, f"Item {i} event is not an object"
        if (REQUIRED_EVENT_KEYS - set(ev.keys())):
            return False, f"Item {i} event missing keys"
        other = links.get("other")
        if not isinstance(other, list):
            return False, f"Item {i} links.other must be an array"
    return True, "ok"


def json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
