from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from serper_enrichment.normalize_datetime import (
    NormalizedDateTime,
    infer_year_hint_from_text,
    normalize_event_datetime,
)
from serper_enrichment.ollama_merge import merge_events_to_gpt_schema
from serper_enrichment.schema import ensure_schema
from serper_enrichment.serper_client import build_queries, extract_evidence, serper_search


def _load_local_env(path: Path = Path('.env')) -> None:
    """Load KEY=VALUE entries from .env into process env (without overwriting existing vars)."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _coerce_str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return str(v)


def _extract_qwen_events(pred_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    pred = pred_json.get("prediction")
    if isinstance(pred, dict):
        events = pred.get("events")
        if isinstance(events, list):
            out: List[Dict[str, Any]] = []
            for e in events:
                if isinstance(e, dict):
                    out.append(
                        {
                            "event_name": _coerce_str_or_none(e.get("event_name")),
                            "venue": _coerce_str_or_none(e.get("venue")),
                            "date": _coerce_str_or_none(e.get("date")),
                            "time": _coerce_str_or_none(e.get("time")),
                            "recurring": _coerce_str_or_none(e.get("recurring")),
                        }
                    )
            return out
    return []


def _year_hint_from_flyer(qwen_events: List[Dict[str, Any]]) -> Optional[int]:
    # Use any explicit year already present in extracted fields.
    for e in qwen_events:
        for k in ("date", "event_name", "venue"):
            y = infer_year_hint_from_text(e.get(k))
            if y:
                return y
    return None


def _build_evidence_for_event(
    *,
    event: Dict[str, Any],
    top_k: int,
    serper_num: int,
    cache_dir: Path,
    gl: Optional[str],
    hl: Optional[str],
) -> Dict[str, Any]:
    queries = build_queries(
        event_name=event.get("event_name"),
        venue=event.get("venue"),
        date_text=event.get("date"),
        time_text=event.get("time"),
    )

    evidence_items: List[Dict[str, str]] = []
    for q in queries:
        raw = serper_search(
            q,
            gl=gl,
            hl=hl,
            num=serper_num,
            cache_dir=cache_dir,
            use_cache=True,
        )
        ev = extract_evidence(raw, top_k=top_k)
        for r in ev:
            evidence_items.append({"title": r.title, "snippet": r.snippet, "url": r.link})

    seen = set()
    deduped: List[Dict[str, str]] = []
    for it in evidence_items:
        url = (it.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(it)
        if len(deduped) >= top_k:
            break

    return {
        "queries": queries,
        "results": deduped,
    }


def _merge_one_event(
    *,
    event: Dict[str, Any],
    evidence: Dict[str, Any],
    normalized: Dict[str, str],
    current_date: str,
    host: str,
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    merged = merge_events_to_gpt_schema(
        qwen_events=[event],
        evidence_by_event=[evidence],
        normalized_by_event=[normalized],
        current_date=current_date,
        host=host,
        model=model,
        temperature=temperature,
    )
    if not merged:
        return ensure_schema({})
    return ensure_schema(merged[0] if isinstance(merged[0], dict) else {})


def main() -> None:
    _load_local_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--serper-cache-dir", type=Path, default=Path(".serper_cache"))
    ap.add_argument("--serper-num", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=7)
    ap.add_argument("--gl", type=str, default=None)
    ap.add_argument("--hl", type=str, default=None)

    ap.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    ap.add_argument("--ollama-model", type=str, default="qwen2.5:7b-instruct")
    ap.add_argument("--temperature", type=float, default=0.1)

    ap.add_argument("--dry-run", action="store_true", help="Do not call Serper/Ollama; just show parsing.")
    ap.add_argument("--predictions-glob", type=str, default="prediction*.json", help="Glob pattern under predictions/ to process")

    args = ap.parse_args()

    preds_dir = args.data_root / "predictions"
    if not preds_dir.exists():
        raise FileNotFoundError(f"Predictions dir not found: {preds_dir}")

    pred_paths = sorted(preds_dir.glob(args.predictions_glob))
    if not pred_paths:
        raise FileNotFoundError(f"No {args.predictions_glob} found in {preds_dir}")

    current_date = datetime.now().strftime("%Y-%m-%d")
    today = date.today()

    for pred_path in pred_paths:
        pred_json = _load_json(pred_path)
        qwen_events = _extract_qwen_events(pred_json)
        flyer_year_hint = _year_hint_from_flyer(qwen_events)

        normalized_by_event: List[Dict[str, str]] = []
        for e in qwen_events:
            ndt: NormalizedDateTime = normalize_event_datetime(
                date_text=e.get("date"),
                time_text=e.get("time"),
                today=today,
                year_hint=flyer_year_hint,
            )
            normalized_by_event.append(asdict(ndt))

        if args.dry_run:
            out_items = []
            for e, n in zip(qwen_events, normalized_by_event):
                out_items.append({"qwen": e, "normalized": n})
            out_path = args.out_dir / pred_path.name.replace("prediction", "dryrun_")
            _save_json(out_path, out_items)
            print(f"[dry-run] wrote {out_path}")
            continue

        merged_items: List[Dict[str, Any]] = []
        for i, (event, normalized) in enumerate(zip(qwen_events, normalized_by_event), start=1):
            evidence = _build_evidence_for_event(
                event=event,
                top_k=args.top_k,
                serper_num=args.serper_num,
                cache_dir=args.serper_cache_dir,
                gl=args.gl,
                hl=args.hl,
            )

            merged_item = _merge_one_event(
                event=event,
                evidence=evidence,
                normalized=normalized,
                current_date=current_date,
                host=args.ollama_host,
                model=args.ollama_model,
                temperature=args.temperature,
            )
            merged_items.append(merged_item)
            print(f"[event {i}/{len(qwen_events)}] merged")

        out_path = args.out_dir / pred_path.name.replace("prediction", "enriched_")
        _save_json(out_path, merged_items)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
