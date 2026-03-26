from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple

from dateutil import parser as dateparser


@dataclass(frozen=True)
class NormalizedDateTime:
    start_date: str  # YYYY-MM-DD or NOVALUE
    end_date: str    # YYYY-MM-DD or NOVALUE
    start_time: str  # HH:MM or NOVALUE
    end_time: str    # HH:MM or NOVALUE


NOVALUE = "NOVALUE"

# If month/day is slightly in the past, keep current year (likely recent event).
PAST_DATE_GRACE_DAYS = 30

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def infer_year_hint_from_text(text: str | None) -> Optional[int]:
    if not text:
        return None
    m = _YEAR_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _nearest_upcoming_month_day(mm: int, dd: int, *, today: date) -> date:
    candidate = date(today.year, mm, dd)
    if candidate >= today:
        return candidate

    # Keep recent past dates in the current year (e.g., event happened a few days ago).
    days_past = (today - candidate).days
    if days_past <= PAST_DATE_GRACE_DAYS:
        return candidate

    return date(today.year + 1, mm, dd)


def normalize_date_text(date_text: str | None, *, today: date, year_hint: Optional[int] = None) -> str:
    if not date_text or not str(date_text).strip():
        return NOVALUE

    raw = str(date_text).strip()
    y = infer_year_hint_from_text(raw) or year_hint

    # If the flyer included a year, dateutil can usually parse it directly.
    if y is not None:
        try:
            dt = dateparser.parse(raw, default=datetime(y, 1, 1))
            if dt is not None:
                return dt.date().isoformat()
        except Exception:
            pass

    # Common Qwen flyer format: "MAR 01" / "Mar 1" (no year)
    try:
        dt = dateparser.parse(raw, default=datetime(today.year, 1, 1))
        if dt is not None:
            # If raw did not include a year, dateutil will use default year.
            # Force "nearest upcoming" behavior based on month/day.
            mm = dt.month
            dd = dt.day
            return _nearest_upcoming_month_day(mm, dd, today=today).isoformat()
    except Exception:
        pass

    return NOVALUE


_TIME_TOKEN_RE = re.compile(
    r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>am|pm)?",
    re.IGNORECASE,
)


def _to_24h(h: int, m: int, ampm: Optional[str]) -> Optional[time]:
    if h < 0 or h > 23 or m < 0 or m > 59:
        return None

    if ampm:
        a = ampm.lower()
        # With am/pm present, hours must be 1-12.
        if h < 1 or h > 12:
            return None
        if a == "am":
            if h == 12:
                h = 0
        elif a == "pm":
            if h != 12:
                h += 12
        else:
            return None

    if h == 24:
        return None

    return time(hour=h, minute=m)


def _fmt_time(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"


def normalize_time_text(time_text: str | None) -> Tuple[str, str]:
    """Return (start_time, end_time) as HH:MM or NOVALUE.

    Supports:
    - "7 pm"        -> (19:00, NOVALUE)
    - "7:30 pm"     -> (19:30, NOVALUE)
    - "5 - 8:30 pm" -> (17:00, 20:30)
    """

    if not time_text or not str(time_text).strip():
        return NOVALUE, NOVALUE

    raw = str(time_text).strip().lower()

    # Normalize separators
    raw = raw.replace("–", "-").replace("—", "-")

    # Range pattern (keep it permissive)
    if "-" in raw:
        parts = [p.strip() for p in raw.split("-") if p.strip()]
        if len(parts) >= 2:
            left = parts[0]
            right = parts[1]

            ml = _TIME_TOKEN_RE.search(left)
            mr = _TIME_TOKEN_RE.search(right)
            if ml and mr:
                hl = int(ml.group("h"))
                mlm = int(ml.group("m") or 0)
                al = ml.group("ampm")

                hr = int(mr.group("h"))
                mrm = int(mr.group("m") or 0)
                ar = mr.group("ampm")

                # If one side is missing am/pm, inherit from the other.
                if not al and ar:
                    al = ar
                if not ar and al:
                    ar = al

                tl = _to_24h(hl, mlm, al)
                tr = _to_24h(hr, mrm, ar)
                if tl and tr:
                    return _fmt_time(tl), _fmt_time(tr)

    # Single time
    m = _TIME_TOKEN_RE.search(raw)
    if m:
        h = int(m.group("h"))
        mm = int(m.group("m") or 0)
        ampm = m.group("ampm")
        t = _to_24h(h, mm, ampm)
        if t:
            return _fmt_time(t), NOVALUE

    return NOVALUE, NOVALUE


def normalize_event_datetime(
    *,
    date_text: str | None,
    time_text: str | None,
    today: date,
    year_hint: Optional[int] = None,
) -> NormalizedDateTime:
    d = normalize_date_text(date_text, today=today, year_hint=year_hint)
    st, et = normalize_time_text(time_text)

    if d == NOVALUE:
        return NormalizedDateTime(start_date=NOVALUE, end_date=NOVALUE, start_time=st, end_time=et)

    return NormalizedDateTime(start_date=d, end_date=d, start_time=st, end_time=et)
