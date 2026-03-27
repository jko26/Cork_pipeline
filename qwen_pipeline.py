#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import unsloth
from PIL import Image
from unsloth import FastVisionModel

BASE_MODEL = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
LORA_MODEL = "./qwen3vl_events_lora"


def extract_id(path: Path):
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else None


def resolve_paths(data_root: Path, idx: int):
    return {
        "image": data_root / "images" / f"out{idx}.png",
        "canvas": data_root / "canvas" / f"canvas{idx}.png",
        "annotation": data_root / "annotations" / f"annotation{idx}.json",
        "target": data_root / "targets" / f"target{idx}.json",
        "prediction": data_root / "predictions" / f"prediction{idx}.json",
    }


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00a0", " ").replace("\r", "\n")
    return "\n".join([ln.strip() for ln in text.split("\n") if ln.strip()])


def normalize_bbox_1000(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(1000, int(round(1000 * x1 / width)))),
        max(0, min(1000, int(round(1000 * y1 / height)))),
        max(0, min(1000, int(round(1000 * x2 / width)))),
        max(0, min(1000, int(round(1000 * y2 / height)))),
    ]


def _bbox_from_poly(poly):
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def build_blocks(annotation, image_path: Path):
    width = annotation.get("width")
    height = annotation.get("height")

    if not width or not height:
        with Image.open(image_path) as im:
            width, height = im.size

    blocks = []

    if isinstance(annotation.get("parsing_res_list"), list):
        for i, item in enumerate(annotation["parsing_res_list"]):
            if not isinstance(item, dict):
                continue

            text = clean_text(item.get("block_content", ""))
            if not text:
                continue

            bbox = item.get("block_bbox")
            if bbox is None and isinstance(item.get("block_poly"), list):
                try:
                    bbox = _bbox_from_poly(item["block_poly"])
                except Exception:
                    bbox = None

            if bbox is None:
                continue

            try:
                bbox = [float(v) for v in bbox]
            except Exception:
                continue

            if len(bbox) != 4:
                continue

            blocks.append({
                "id": f"B{i}",
                "text": text,
                "bbox_1000": normalize_bbox_1000(bbox, width, height),
                "order": item.get("block_order", i),
            })

    elif isinstance(annotation.get("rec_texts"), list) and isinstance(annotation.get("rec_boxes"), list):
        for i, (text, bbox) in enumerate(zip(annotation["rec_texts"], annotation["rec_boxes"])):
            text = clean_text(text)
            if not text:
                continue

            try:
                bbox = [float(v) for v in bbox]
            except Exception:
                continue

            if len(bbox) != 4:
                continue

            blocks.append({
                "id": f"B{i}",
                "text": text,
                "bbox_1000": normalize_bbox_1000(bbox, width, height),
                "order": i,
            })

    blocks.sort(key=lambda b: b.get("order", 10**9))
    return blocks


def build_prompt(blocks_json: str):
    output_schema_example = {
        "events": [
            {
                "event_name": "Jazz Night",
                "venue": "Blue Note Club, New York, NY",
                "date": "2026-04-04",
                "time": "8:00 PM",
                "recurring": "Every Friday",
            },
            {
                "event_name": "Art Exhibition Opening",
                "venue": None,
                "date": "2025-04-12",
                "time": "6:30 PM",
                "recurring": None,
            },
        ]
    }

    return (
        "Role: Event Extraction Specialist.\n"
        "Task: Extract structured event records from the original flyer, the annotated Canvas, and BLOCKS_JSON.\n\n"

        "Source-of-truth rule:\n"
        "- The ORIGINAL FLYER (Image 1) is always the ground truth.\n"
        "- The annotated Canvas (Image 2) and BLOCKS_JSON are only grounding aids.\n"
        "- If the Canvas or BLOCKS_JSON has OCR mistakes, missing words, missing boxes, wrong grouping, or incomplete text, trust the ORIGINAL FLYER.\n"
        "- Use the Canvas and BLOCKS_JSON to locate regions, but resolve final content from the ORIGINAL FLYER.\n\n"

        "For each distinct event found, output one object with exactly these fields:\n"
        "  - event_name: string|null\n"
        "  - venue: string|null\n"
        "  - date: string|null\n"
        "  - time: string|null\n"
        "  - recurring: string|null\n\n"

        "Inputs:\n"
        "- Image 1: original flyer (source of truth)\n"
        "- Image 2: annotated Canvas with visible B-ID badges\n"
        "- BLOCKS_JSON: OCR/layout blocks with text and normalized bounding boxes\n\n"

        "Rules:\n"
        "1. Do not hallucinate. If a field is not present, set it to null.\n"
        "2. Prefer faithful extraction over normalization, but format the date field as YYYY-MM-DD whenever possible.\n"
        "3. If event month/day is visible but year is missing, use the flyer-level year visible in the flyer header/title (for example, 'MARCH 2026').\n"
        "4. Include city in the venue field when visible in the flyer (for example, 'Venue Name, Washington, DC').\n"
        "5. If you cannot determine a valid YYYY-MM-DD date, set date to null.\n"
        "6. Keep time text faithful to the flyer; do not force 24-hour conversion here.\n"
        "7. Output one object per distinct event.\n"
        "8. If the same event appears multiple times, output it once.\n"
        "9. Ignore decorative text, watermarks, branding, and headers/footers unless they are clearly part of a specific event.\n"
        "10. Some event text may be stylized or embedded visually; use the ORIGINAL FLYER to recover it when Canvas/BLOCKS_JSON are imperfect.\n"
        "11. Use both visual layout and block text to decide event boundaries, but resolve disagreements in favor of the ORIGINAL FLYER.\n"
        "12. List events in reading order: top-to-bottom, then left-to-right.\n"
        "13. If Canvas/BLOCKS_JSON are missing a word that is visible in the ORIGINAL FLYER, include the visible word.\n"
        "14. If Canvas/BLOCKS_JSON include an OCR error, correct it using the ORIGINAL FLYER.\n"
        "15. If you are 50-50 on whether something is an event, prefer to include it as an event rather than omit it. You can set fields to null if uncertain about specific details, but try not to omit entire events if they are reasonably identifiable.\n\n"

        f"BLOCKS_JSON:\n{blocks_json}\n\n"

        "Output format:\n"
        "Return ONLY a valid JSON object with exactly one top-level key: \"events\".\n"
        "No explanation. No markdown fences.\n\n"

        f"Example:\n{json.dumps(output_schema_example, ensure_ascii=False)}"
    )


def extract_json(text: str):
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {"raw_output": text}


def load_inference_model(lora_path: str, base_model: str | None = None):
    """Load base Qwen3-VL + LoRA for inference (call once per process)."""
    base = base_model if base_model is not None else BASE_MODEL
    print(f"Loading model: {base}")
    model, tokenizer = FastVisionModel.from_pretrained(
        base,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model.load_adapter(lora_path)
    FastVisionModel.for_inference(model)
    return model, tokenizer


def predict_one(
    data_root: Path,
    idx: int,
    model,
    tokenizer,
    *,
    max_new_tokens: int = 1500,
    temperature: float = 0.0,
    write_prediction: bool = False,
) -> dict:
    """
    Run vision+LLM on one index under data_root (expects images/out{idx}.png,
    annotations/annotation{idx}.json, canvas/canvas{idx}.png).
    """
    paths = resolve_paths(data_root, idx)
    missing = [k for k in ("image", "canvas", "annotation") if not paths[k].exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for idx={idx}: {missing}")

    annotation = load_json(paths["annotation"])
    blocks = build_blocks(annotation, paths["image"])
    blocks_json = json.dumps(blocks, indent=2, ensure_ascii=False)
    prompt = build_prompt(blocks_json)

    image = Image.open(paths["image"]).convert("RGB")
    canvas = Image.open(paths["canvas"]).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "image", "image": canvas},
            {"type": "text", "text": prompt},
        ],
    }]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text=[chat_text],
        images=[image, canvas],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        use_cache=True,
    )

    generated = outputs[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    parsed = extract_json(response)

    result = {
        "index": idx,
        "image": str(paths["image"]),
        "canvas": str(paths["canvas"]),
        "annotation": str(paths["annotation"]),
        "num_blocks": len(blocks),
        "prediction": parsed,
        "raw_output": response,
    }

    if paths["target"].exists():
        try:
            result["target"] = load_json(paths["target"])
        except Exception:
            pass

    if write_prediction:
        pred_path = paths["prediction"]
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lora", type=str, default=LORA_MODEL)
    parser.add_argument("--base", type=str, default=BASE_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-index", type=int, default=None, help="If set, only process out{index}.png")
    args = parser.parse_args()

    predictions_dir = args.data_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_inference_model(args.lora, args.base)

    image_paths = sorted(
        (args.data_root / "images").glob("out*.png"),
        key=lambda p: extract_id(p) if extract_id(p) is not None else 10**9,
    )

    if args.only_index is not None:
        image_paths = [p for p in image_paths if extract_id(p) == args.only_index]

    print(f"Found {len(image_paths)} image(s)\n")

    for img_path in image_paths:
        idx = extract_id(img_path)
        if idx is None:
            print(f"[skip] could not parse id from {img_path.name}")
            continue

        paths = resolve_paths(args.data_root, idx)
        pred_path = paths["prediction"]

        if pred_path.exists() and not args.overwrite:
            print(f"[skip {idx}] prediction exists: {pred_path}")
            continue

        missing = [k for k in ("image", "canvas", "annotation") if not paths[k].exists()]
        if missing:
            print(f"[skip {idx}] missing {missing}")
            continue

        try:
            predict_one(
                args.data_root,
                idx,
                model,
                tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                write_prediction=True,
            )

            print(f"[saved {idx}] {pred_path}")

        except Exception as e:
            error_path = predictions_dir / f"prediction{idx}.json"
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "index": idx,
                        "image": str(paths["image"]),
                        "canvas": str(paths["canvas"]),
                        "annotation": str(paths["annotation"]),
                        "error": str(e),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[error {idx}] {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()