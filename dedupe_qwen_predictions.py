from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_text(v: Any) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _coerce(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _extract_events(pred_json: Dict[str, Any], source_name: str) -> List[Dict[str, Any]]:
    pred = pred_json.get("prediction")
    events = pred.get("events") if isinstance(pred, dict) else None
    out: List[Dict[str, Any]] = []
    if not isinstance(events, list):
        return out

    for e in events:
        if not isinstance(e, dict):
            continue
        item = {
            "event_name": _coerce(e.get("event_name")),
            "venue": _coerce(e.get("venue")),
            "date": _coerce(e.get("date")),
            "time": _coerce(e.get("time")),
            "recurring": _coerce(e.get("recurring")),
            "_source": source_name,
        }
        out.append(item)
    return out


def _name_similarity(a: Optional[str], b: Optional[str]) -> float:
    na = _norm_text(a)
    nb = _norm_text(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _events_match(a: Dict[str, Any], b: Dict[str, Any], strong_name_threshold: float, weak_name_threshold: float) -> bool:
    sim = _name_similarity(a.get("event_name"), b.get("event_name"))
    da = _coerce(a.get("date"))
    db = _coerce(b.get("date"))
    date_match = bool(da and db and da == db)
    return sim >= strong_name_threshold or (date_match and sim >= weak_name_threshold)


def _pick_best(values: List[Optional[str]]) -> Optional[str]:
    vals = [v for v in values if v]
    if not vals:
        return None

    counts: Dict[str, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1

    best = sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[0][0]
    return best


def _merge_cluster(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "event_name": _pick_best([c.get("event_name") for c in cluster]),
        "venue": _pick_best([c.get("venue") for c in cluster]),
        "date": _pick_best([c.get("date") for c in cluster]),
        "time": _pick_best([c.get("time") for c in cluster]),
        "recurring": _pick_best([c.get("recurring") for c in cluster]),
    }


def dedupe_events(
    events: List[Dict[str, Any]],
    strong_name_threshold: float = 0.84,
    weak_name_threshold: float = 0.72,
) -> List[Dict[str, Any]]:
    clusters: List[List[Dict[str, Any]]] = []

    for e in events:
        placed = False
        for cluster in clusters:
            if any(_events_match(e, c, strong_name_threshold, weak_name_threshold) for c in cluster):
                cluster.append(e)
                placed = True
                break
        if not placed:
            clusters.append([e])

    merged = [_merge_cluster(c) for c in clusters]
    merged = [m for m in merged if _coerce(m.get("event_name"))]
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicate Qwen prediction files into one merged prediction JSON")
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--glob", type=str, default="prediction*.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name-threshold", type=float, default=0.84)
    ap.add_argument("--date-name-threshold", type=float, default=0.72)
    args = ap.parse_args()

    pred_paths = sorted(args.pred_dir.glob(args.glob))
    if not pred_paths:
        raise FileNotFoundError(f"No {args.glob} found in {args.pred_dir}")

    all_events: List[Dict[str, Any]] = []
    for pp in pred_paths:
        pred_json = _load_json(pp)
        all_events.extend(_extract_events(pred_json, pp.name))

    merged = dedupe_events(
        all_events,
        strong_name_threshold=args.name_threshold,
        weak_name_threshold=args.date_name_threshold,
    )

    out_json = {
        "prediction": {"events": merged},
        "meta": {
            "source_predictions": [p.name for p in pred_paths],
            "num_input_events": len(all_events),
            "num_output_events": len(merged),
        },
    }
    _save_json(args.out, out_json)
    print(f"[dedupe] {len(all_events)} -> {len(merged)} events")
    print(f"[dedupe] wrote {args.out}")


if __name__ == "__main__":
    main()
