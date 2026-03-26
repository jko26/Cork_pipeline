#!/usr/bin/env python3
"""
build_dataset.py

Builds a multimodal event-extraction dataset for Qwen fine-tuning.

For each image out{N}.png:
1. Run PaddleOCR-VL with your fine-tuned layout model and save annotation{N}.json
2. Render an annotated canvas canvas{N}.png with B-ID badges
3. Ask OpenAI to extract structured event records
4. Save target{N}.json as:

{
  "events": [
    {
      "event_name": ...,
      "venue": ...,
      "date": ...,
      "time": ...,
      "recurring": ...
    }
  ]
}

Directory layout:
dataset/train/
  images/
    out1.png
    out2.png
    ...
  annotations/
    annotation1.json
    ...
  canvas/
    canvas1.png
    ...
  targets/
    target1.json
    ...

Requirements:
    pip install openai pillow paddleocr
    export OPENAI_API_KEY=...

Usage:
    python build_dataset.py
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT       = Path("./dataset/train")
IMAGES_DIR      = DATA_ROOT / "images"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
CANVAS_DIR      = DATA_ROOT / "canvas"
TARGETS_DIR     = DATA_ROOT / "targets"

OPENAI_MODEL = "gpt-4.1"
MAX_TOKENS   = 2000

SKIP_EXISTING = True

# Your exported fine-tuned layout detector inference folder.
# Example:
#   "./paddlex_output/best_model/inference"
# or:
#   "./paddlex_output/inference"
FINETUNED_LAYOUT_MODEL_DIR = "./paddlex_output/best_model/inference"

# PaddleOCR-VL settings
USE_GPU = True
DEVICE = "gpu:0" if USE_GPU else "cpu"
LAYOUT_MERGE_BBOXES_MODE = "large"   # try "small" if it over-merges
USE_DOC_ORIENTATION_CLASSIFY = True
USE_DOC_UNWARPING = True
USE_LAYOUT_DETECTION = True

OUTPUT_SCHEMA_EXAMPLE = {
    "events": [
        {
            "event_name": "Jazz Night",
            "venue": "Blue Note Club",
            "date": "Fri 4 Apr",
            "time": "8:00 PM",
            "recurring": "Every Friday",
        },
        {
            "event_name": "Art Exhibition Opening",
            "venue": None,
            "date": "12 April 2025",
            "time": "6:30 PM",
            "recurring": None,
        },
    ]
}

# ============================================================
# FILE NAMING
# ============================================================

def extract_numeric_suffix(path: Path):
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else None

def annotation_path_for(image_path: Path) -> Path:
    idx = extract_numeric_suffix(image_path)
    if idx is None:
        raise ValueError(f"No numeric suffix in {image_path.name}")
    return ANNOTATIONS_DIR / f"annotation{idx}.json"

def canvas_path_for(image_path: Path) -> Path:
    idx = extract_numeric_suffix(image_path)
    if idx is None:
        raise ValueError(f"No numeric suffix in {image_path.name}")
    return CANVAS_DIR / f"canvas{idx}.png"

def target_path_for(image_path: Path) -> Path:
    idx = extract_numeric_suffix(image_path)
    if idx is None:
        raise ValueError(f"No numeric suffix in {image_path.name}")
    return TARGETS_DIR / f"target{idx}.json"

# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_events_object(text: str) -> dict:
    """
    Robustly extract {"events": [...]} from the model response.
    Also tolerates a bare list and wraps it.
    """
    def normalize(parsed):
        if isinstance(parsed, dict) and isinstance(parsed.get("events"), list):
            return {"events": parsed["events"]}
        if isinstance(parsed, list):
            return {"events": parsed}
        return None

    try:
        parsed = json.loads(text)
        out = normalize(parsed)
        if out is not None:
            return out
    except json.JSONDecodeError:
        pass

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1:
            snippet = text[start:end + 1]
            try:
                parsed = json.loads(snippet)
                out = normalize(parsed)
                if out is not None:
                    return out
            except json.JSONDecodeError:
                continue

    raise ValueError(f"No valid event JSON found in response:\n{text}")

# ============================================================
# OCR BLOCK PROCESSING
# ============================================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00a0", " ").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines).strip()

def normalize_bbox_1000(bbox, width: int, height: int):
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(1000, round(1000 * x1 / width))),
        max(0, min(1000, round(1000 * y1 / height))),
        max(0, min(1000, round(1000 * x2 / width))),
        max(0, min(1000, round(1000 * y2 / height))),
    ]

def _bbox_from_poly(poly):
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

def build_llm_ready_blocks(raw_data: dict, image_path: Path) -> dict:
    """
    Prefer PaddleOCR-VL layout parsing output. Fall back to rec_texts/rec_boxes if needed.
    """
    width = raw_data.get("width")
    height = raw_data.get("height")

    if not width or not height:
        with Image.open(image_path) as im:
            width, height = im.size

    blocks = []
    next_id = 0

    # Case 1: PaddleOCR-VL / structured parsing output
    # Typical fields include parsing_res_list with block_bbox/block_content/block_label/block_order.
    if isinstance(raw_data.get("parsing_res_list"), list):
        for item in raw_data["parsing_res_list"]:
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

            try:
                bbox = [float(v) for v in bbox]
            except Exception:
                continue

            if len(bbox) != 4:
                continue

            blocks.append({
                "id": next_id,
                "text": text,
                "bbox": bbox,
                "bbox_1000": normalize_bbox_1000(bbox, width, height),
                "block_order": item.get("block_order", next_id),
            })
            next_id += 1

    # Case 2: OCR fallback
    elif isinstance(raw_data.get("rec_texts"), list) and isinstance(raw_data.get("rec_boxes"), list):
        for order, (text, bbox) in enumerate(zip(raw_data["rec_texts"], raw_data["rec_boxes"])):
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
                "id": next_id,
                "text": text,
                "bbox": bbox,
                "bbox_1000": normalize_bbox_1000(bbox, width, height),
                "block_order": order,
            })
            next_id += 1
    else:
        print("[warn] Unknown annotation format; no usable blocks found")

    blocks.sort(
        key=lambda b: (
            b["block_order"] is None,
            b["block_order"] if b["block_order"] is not None else 10**9,
        )
    )

    return {
        "width": width,
        "height": height,
        "blocks": blocks,
    }

def build_blocks_json_for_prompt(llm_ready: dict) -> str:
    compact = []
    for b in llm_ready["blocks"]:
        compact.append({
            "id":        f"B{b['id']}",
            "text":      b["text"],
            "order":     b["block_order"],
            "bbox_1000": b["bbox_1000"],
        })
    return json.dumps(compact, ensure_ascii=False, indent=2)

# ============================================================
# PROMPT
# ============================================================

def build_prompt(blocks_json: str) -> str:
    return (
        "Role: Event Extraction Specialist.\n"
        "Task: Extract structured event records from the original flyer, the annotated Canvas, and BLOCKS_JSON.\n\n"
        "For each distinct event found, output one object with exactly these fields:\n"
        "  - event_name: string|null\n"
        "  - venue: string|null\n"
        "  - date: string|null\n"
        "  - time: string|null\n"
        "  - recurring: string|null\n\n"
        "Inputs:\n"
        "- Image 1: original flyer\n"
        "- Image 2: annotated Canvas with visible B-ID badges\n"
        "- BLOCKS_JSON: OCR/layout blocks with text and normalized bounding boxes\n\n"
        "Rules:\n"
        "1. Do not hallucinate. If a field is not present, set it to null.\n"
        "2. Copy values verbatim from the source. Do not normalize or reformat dates/times.\n"
        "3. Output one object per distinct event.\n"
        "4. If the same event appears multiple times, output it once.\n"
        "5. Ignore decorative text, watermarks, branding, and headers/footers unless they are clearly part of a specific event.\n"
        "6. Some blocks may contain stylized text rendered as graphics. Use them if they clearly represent event information.\n"
        "7. Use both visual layout and block text to decide event boundaries.\n"
        "8. List events in reading order: top-to-bottom, then left-to-right.\n\n"
        f"BLOCKS_JSON:\n{blocks_json}\n\n"
        "Output format:\n"
        "Return ONLY a valid JSON object with exactly one top-level key: \"events\".\n"
        "No explanation. No markdown fences.\n\n"
        f"Example:\n{json.dumps(OUTPUT_SCHEMA_EXAMPLE, ensure_ascii=False)}"
    )

# ============================================================
# CANVAS RENDERING
# ============================================================

def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()
def wrap_text(draw, text, font, max_width):
    """
    Wrap text to fit inside max_width.
    Preserves paragraph breaks from existing newlines.
    """
    paragraphs = text.split("\n")
    wrapped_lines = []

    for para in paragraphs:
        words = para.split()
        if not words:
            wrapped_lines.append("")
            continue

        current = words[0]
        for w in words[1:]:
            test = current + " " + w
            tb = draw.textbbox((0, 0), test, font=font)
            if (tb[2] - tb[0]) <= max_width:
                current = test
            else:
                wrapped_lines.append(current)
                current = w
        wrapped_lines.append(current)

    return wrapped_lines


def fit_text_to_box(draw, text, box_w, box_h, min_font_size=8, max_font_size=16, pad=6, line_gap=3):
    """
    Find the largest font size that lets wrapped text fit inside the box.
    Returns (font, wrapped_lines).
    """
    best_font = load_font(min_font_size)
    best_lines = []

    usable_w = max(8, box_w - 2 * pad)
    usable_h = max(8, box_h - 2 * pad)

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = load_font(font_size)
        lines = wrap_text(draw, text, font, usable_w)

        line_heights = []
        for ln in lines:
            tb = draw.textbbox((0, 0), ln if ln else "Ag", font=font)
            line_heights.append(tb[3] - tb[1])

        total_h = sum(line_heights)
        if line_heights:
            total_h += line_gap * (len(line_heights) - 1)

        if total_h <= usable_h:
            return font, lines

        best_font = font
        best_lines = lines

    return best_font, best_lines


def get_box_color(idx):
    """
    Semi-transparent RGBA colors for overlapping boxes.
    """
    palette = [
        (255, 0, 0, 55),
        (0, 128, 255, 55),
        (0, 180, 0, 55),
        (255, 165, 0, 55),
        (160, 32, 240, 55),
        (0, 200, 200, 55),
        (220, 20, 60, 55),
        (120, 120, 0, 55),
    ]
    return palette[idx % len(palette)]


def get_outline_color(fill_rgba):
    r, g, b, _ = fill_rgba
    return (r, g, b, 255)


def place_badge(width, height, x1, y1, x2, y2, badge_w, badge_h, used_badges):
    """
    Try several positions so badges stay visible and avoid overlapping each other.
    """
    candidates = [
        (x1, y1 - badge_h - 2),      # above left
        (x2 - badge_w, y1 - badge_h - 2),  # above right
        (x1 + 2, y1 + 2),            # inside top-left
        (x2 - badge_w - 2, y1 + 2),  # inside top-right
        (x1, y2 + 2),                # below left
        (x2 - badge_w, y2 + 2),      # below right
    ]

    def intersects(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    for bx1, by1 in candidates:
        bx2 = bx1 + badge_w
        by2 = by1 + badge_h

        if bx1 < 0 or by1 < 0 or bx2 > width or by2 > height:
            continue

        rect = (bx1, by1, bx2, by2)
        if not any(intersects(rect, other) for other in used_badges):
            used_badges.append(rect)
            return bx1, by1, bx2, by2

    # fallback
    bx1 = max(0, min(width - badge_w, x1 + 2))
    by1 = max(0, min(height - badge_h, y1 + 2))
    bx2 = bx1 + badge_w
    by2 = by1 + badge_h
    used_badges.append((bx1, by1, bx2, by2))
    return bx1, by1, bx2, by2


def render_layout_canvas(llm_ready: dict, save_path: Path):
    width = int(llm_ready["width"])
    height = int(llm_ready["height"])

    # White background, but RGBA so fills can be transparent
    base = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    used_badges = []
    badges_to_draw = []

    for b in llm_ready["blocks"]:
        x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"]]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        box_w = x2 - x1
        box_h = y2 - y1

        #fill = get_box_color(b["id"])
        # outline = get_outline_color(fill)

        # transparent fill + opaque outline
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0, 255), width=3)

        font, lines = fit_text_to_box(draw, b["text"], box_w, box_h, min_font_size=8, max_font_size=16, pad=6, line_gap=3)

        y_cursor = y1 + 6
        for ln in lines:
            tb = draw.textbbox((0, 0), ln if ln else "Ag", font=font)
            line_h = tb[3] - tb[1]

            if y_cursor + line_h > y2 - 4:
                break

            draw.text((x1 + 6, y_cursor), ln, fill=(0, 0, 0, 255), font=font)
            y_cursor += line_h + 3

        badge_text = f"B{b['id']}"
        badge_font = load_font(14)
        tb = draw.textbbox((0, 0), badge_text, font=badge_font)
        badge_w = (tb[2] - tb[0]) + 8
        badge_h = (tb[3] - tb[1]) + 4

        bx1, by1, bx2, by2 = place_badge(width, height, x1, y1, x2, y2, badge_w, badge_h, used_badges)
        badges_to_draw.append((bx1, by1, bx2, by2, badge_text, badge_font))

    # draw badges last so they stay visible
    for bx1, by1, bx2, by2, badge_text, badge_font in badges_to_draw:
        draw.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0, 255))
        draw.text((bx1 + 4, by1 + 1), badge_text, fill=(255, 255, 255, 255), font=badge_font)

    canvas = Image.alpha_composite(base, overlay).convert("RGB")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    return canvas
    
# ============================================================
# PADDLEOCR-VL PIPELINE
# ============================================================

def create_vl_pipeline():
    """
    Creates a PaddleOCR-VL pipeline using fine-tuned PP-DocBlockLayout.
    """
    from paddleocr import PaddleOCRVL

    # Ensure the path is absolute and exists
    model_dir = str(Path(FINETUNED_LAYOUT_MODEL_DIR).resolve())
    
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"PP-DocBlockLayout directory not found: {model_dir}")

    pipeline = PaddleOCRVL(
        device=DEVICE,
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_layout_detection=USE_LAYOUT_DETECTION,
        # Force the name to pass the assertion check
        layout_detection_model_name="PP-DocLayoutV2", 
        layout_detection_model_dir=model_dir,
    )
    return pipeline

def run_vl_and_get_raw(pipeline, image_path: Path) -> dict:
    """
    Runs PaddleOCR-VL and returns a saved JSON dict.
    """
    results = pipeline.predict(
        str(image_path),
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_layout_detection=USE_LAYOUT_DETECTION,
        layout_merge_bboxes_mode=LAYOUT_MERGE_BBOXES_MODE,
        format_block_content=False,
    )

    results = list(results)
    if not results:
        raise RuntimeError(f"No PaddleOCR-VL result for {image_path}")

    res = results[0]

    # Try common save API used by PaddleOCR/PaddleX result objects
    tmp_dir = ANNOTATIONS_DIR / "_tmp_vl_json"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        res.save_to_json(str(tmp_dir))
        written = sorted(tmp_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not written:
            raise RuntimeError("PaddleOCR-VL save_to_json wrote no files")
        return load_json(written[-1])
    finally:
        # Keep temp dir; easiest and safest. Could clean it later if you want.
        pass

# ============================================================
# OPENAI CALL
# ============================================================

def image_to_data_url(path: Path) -> str:
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def call_openai(client: OpenAI, image_path: Path, canvas_path: Path, blocks_json: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path),
                            "detail": "high",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(canvas_path),
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": build_prompt(blocks_json),
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one_image(pipeline, openai_client: OpenAI, image_path: Path):
    annotation_path = annotation_path_for(image_path)
    canvas_path = canvas_path_for(image_path)
    target_path = target_path_for(image_path)

    if SKIP_EXISTING and annotation_path.exists() and canvas_path.exists():
        print(f"[skip ocr] {image_path.name}")
        raw_data = load_json(annotation_path)
        llm_ready = build_llm_ready_blocks(raw_data, image_path)

    elif SKIP_EXISTING and annotation_path.exists() and not canvas_path.exists():
        print(f"[re-render canvas] {image_path.name}")
        raw_data = load_json(annotation_path)
        llm_ready = build_llm_ready_blocks(raw_data, image_path)
        render_layout_canvas(llm_ready, canvas_path)
        print(f"  -> canvas: {canvas_path}")

    else:
        print(f"[ocr-vl] {image_path.name}")
        raw_data = run_vl_and_get_raw(pipeline, image_path)
        save_json(annotation_path, raw_data)

        llm_ready = build_llm_ready_blocks(raw_data, image_path)
        render_layout_canvas(llm_ready, canvas_path)

        print(f"  -> annotation: {annotation_path}")
        print(f"  -> canvas:     {canvas_path}")

    if SKIP_EXISTING and target_path.exists():
        print(f"[skip target] {image_path.name}")
        return

    blocks_json_str = build_blocks_json_for_prompt(llm_ready)

    if not llm_ready["blocks"]:
        print("  [warn] no usable blocks — writing empty target")
        save_json(target_path, {"events": []})
        return

    print(f"  [openai] {len(llm_ready['blocks'])} blocks -> {OPENAI_MODEL}")
    raw_response = call_openai(openai_client, image_path, canvas_path, blocks_json_str)
    print(f"  raw: {raw_response[:120]}{'...' if len(raw_response) > 120 else ''}")

    target_obj = extract_events_object(raw_response)
    save_json(target_path, target_obj)
    print(f"  -> target: {target_path}  ({len(target_obj['events'])} events)")

# ============================================================
# MAIN
# ============================================================

def _load_local_env(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE entries from .env into environment without overriding existing vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main():
    _load_local_env()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=...")

    for d in [IMAGES_DIR, ANNOTATIONS_DIR, CANVAS_DIR, TARGETS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in IMAGES_DIR.glob("out*.png")],
        key=lambda p: extract_numeric_suffix(p) if extract_numeric_suffix(p) is not None else -1,
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    openai_client = OpenAI(api_key=api_key)
    pipeline = create_vl_pipeline()

    print(f"Found {len(image_paths)} image(s)")
    for image_path in image_paths:
        try:
            process_one_image(pipeline, openai_client, image_path)
        except Exception as e:
            print(f"[error] {image_path.name}: {e}")

if __name__ == "__main__":
    main()