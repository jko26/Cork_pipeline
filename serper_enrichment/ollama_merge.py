from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from .schema import NOVALUE, ensure_schema, validate_array


OLLAMA_DEFAULT_HOST = "http://localhost:11434"


def _extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try substring extraction
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    return None


def _ollama_chat(
    *,
    host: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    timeout_s: int = 120,
) -> str:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def merge_events_to_gpt_schema(
    *,
    qwen_events: List[Dict[str, Any]],
    evidence_by_event: List[Dict[str, Any]],
    normalized_by_event: List[Dict[str, str]],
    current_date: Optional[str] = None,
    host: str = OLLAMA_DEFAULT_HOST,
    model: str = "qwen2.5:7b-instruct",
    temperature: float = 0.1,
) -> List[Dict[str, Any]]:
    """One-shot merge of all flyer events into the old gpt_call.py schema.

    Returns: list of items (already coerced to schema/NOVALUE conventions).
    """

    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")

    system = f"""You are an expert information-extraction and research engine.

You will receive:
- qwen_events: raw extracted flyer events (may be missing fields)
- normalized_by_event: deterministic normalized date/time hints (may be NOVALUE)
- evidence_by_event: compact web search evidence (title/snippet/url lists) per event

YOUR TASK
For each event/attraction, output one object matching EXACTLY this schema:

{{
  \"type\": \"event\" | \"attraction\",
  \"title\": \"string\",
  \"address\": \"string\",
  \"venue\": \"string\",
  \"description\": \"string\",
  \"links\": {{
    \"website\": \"string\",
    \"tickets\": \"string\",
    \"reservation\": \"string\",
    \"other\": [\"string\"]
  }},
  \"event\": {{
    \"recurring_weekly\": \"yes\" | \"no\" | \"NOVALUE\",
    \"recurring_weekday\": \"string\",
    \"start_date\": \"string\",
    \"start_time\": \"string\",
    \"end_date\": \"string\",
    \"end_time\": \"string\"
  }}
}}

RULES
- Return ONLY a JSON ARRAY. No markdown, no explanation.
- Every object must include all fields shown above.
- Missing values MUST be the literal string \"NOVALUE\" (not null).
- links.other MUST be an array (use [] if none). Max 3.
- Only include links you can justify from evidence_by_event URLs. Do not fabricate URLs.

DATE/TIME NORMALIZATION
- Dates must be YYYY-MM-DD. Times must be HH:MM (24-hour).
- Resolve relative or missing-year dates using CURRENT DATE: {current_date}
- When month/day given without year, use the nearest upcoming date.
- Use normalized_by_event as a strong hint (but you may override if evidence is clearly better).

CONFIDENCE GATING FOR DATE/TIME
- If you are not confident about date/time, set date/time fields to NOVALUE.
- Only set start_date/end_date when date evidence is explicit and event-specific in high-priority sources (tickets or official website).
- Only set start_time/end_time when time evidence is explicit and event-specific in high-priority sources (tickets or official website).
- Do NOT infer date/time from weak clues, generic recurring series pages, or low-priority other links.
- When sources conflict or are ambiguous for date/time, prefer NOVALUE instead of guessing.
- If a ticket URL contains an explicit date token, you may use it as date evidence only when it does not conflict with qwen_events or higher-priority website evidence for the same event.

ADDRESS / GEOCODING RULE
- If you can find a full street address in evidence, use it.
- Otherwise output \"Venue, City\" (or \"Venue, City, State\") if supported by evidence.
- Never invent street numbers.

SOURCE PRIORITY
- Prioritize facts from official website and ticketing links over all other links.
- Treat social posts, blogs, and aggregator pages as lower-priority fallback evidence.

CONFLICT POLICY
- For date/time/venue/address conflicts, use this order: tickets links > official website links > other links > qwen_events.
- If tickets and official website disagree with each other, prefer qwen_events unless one source is clearly event-specific and dated for the same event.
- If only lower-priority (other) links disagree with qwen_events, prefer qwen_events.

ATTRACTION RULE
- If type is \"attraction\", set all event.* fields to NOVALUE.
"""

    payload = {
        "qwen_events": qwen_events,
        "normalized_by_event": normalized_by_event,
        "evidence_by_event": evidence_by_event,
    }

    user = "Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    text = _ollama_chat(host=host, model=model, messages=messages, temperature=temperature)
    data = _extract_json(text)

    if data is None:
        # One repair attempt
        repair_system = "Return ONLY valid JSON for the user's requested array schema."
        repair_user = "Fix this into valid JSON (no markdown):\n" + text
        text2 = _ollama_chat(
            host=host,
            model=model,
            messages=[{"role": "system", "content": repair_system}, {"role": "user", "content": repair_user}],
            temperature=0.0,
        )
        data = _extract_json(text2)

    if not isinstance(data, list):
        raise RuntimeError("Ollama did not return a JSON array")

    ok, msg = validate_array(data)
    if not ok:
        # Coerce what we can, but keep it strict
        coerced = [ensure_schema(x if isinstance(x, dict) else {}) for x in data]
        ok2, msg2 = validate_array(coerced)
        if not ok2:
            raise RuntimeError(f"LLM output failed schema validation: {msg} / {msg2}")
        return coerced

    return [ensure_schema(x) for x in data]
