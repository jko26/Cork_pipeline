from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


SERPER_ENDPOINT_DEFAULT = "https://google.serper.dev/search"


@dataclass(frozen=True)
class SerperResult:
    title: str
    snippet: str
    link: str


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def serper_search(
    query: str,
    *,
    api_key: Optional[str] = None,
    endpoint: str = SERPER_ENDPOINT_DEFAULT,
    gl: Optional[str] = None,
    hl: Optional[str] = None,
    num: int = 10,
    timeout_s: int = 20,
    cache_dir: Path = Path(".serper_cache"),
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Call Serper and return the raw JSON response (cached by query+params)."""

    key = api_key or os.getenv("SERPER_API_KEY")
    if not key:
        raise ValueError("SERPER_API_KEY not set (or api_key not provided)")

    payload: Dict[str, Any] = {"q": query}
    if gl:
        payload["gl"] = gl
    if hl:
        payload["hl"] = hl
    if num:
        payload["num"] = int(num)

    cache_key = _sha256(json.dumps({"endpoint": endpoint, "payload": payload}, sort_keys=True))
    cache_path = cache_dir / f"{cache_key}.json"

    if use_cache and cache_path.exists():
        return _load_json(cache_path)

    headers = {
        "X-API-KEY": key,
        "Content-Type": "application/json",
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    if use_cache:
        _save_json(cache_path, data)

    return data


def extract_evidence(raw: Dict[str, Any], *, top_k: int = 10) -> List[SerperResult]:
    """Extract compact (title/snippet/link) evidence from Serper response."""

    out: List[SerperResult] = []
    organic = raw.get("organic")
    if isinstance(organic, list):
        for item in organic:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            link = (item.get("link") or "").strip()
            if not link:
                continue
            out.append(SerperResult(title=title, snippet=snippet, link=link))
            if len(out) >= top_k:
                break

    return out


def build_queries(
    *,
    event_name: Optional[str],
    venue: Optional[str],
    date_text: Optional[str],
    time_text: Optional[str],
) -> List[str]:
    """Generate multiple targeted queries per event (similar spirit to old GPT prompt)."""

    name = (event_name or "").strip()
    v = (venue or "").strip()
    d = (date_text or "").strip()
    t = (time_text or "").strip()

    if not name and not v:
        return []

    base = " ".join([x for x in [name, v] if x]).strip()

    queries: List[str] = []
    if base:
        queries.append(base)
    if name and v:
        queries.append(f"{name} tickets {v}")
        queries.append(f"{name} reservation {v}")
        queries.append(f"{name} hours admission {v}")
    if base and d:
        queries.append(f"{base} {d}")
    if base and t:
        queries.append(f"{base} {t}")

    seen = set()
    deduped: List[str] = []
    for q in queries:
        q2 = " ".join(q.split())
        if q2 and q2.lower() not in seen:
            seen.add(q2.lower())
            deduped.append(q2)

    return deduped
