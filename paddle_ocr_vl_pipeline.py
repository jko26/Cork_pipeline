#!/usr/bin/env python3
"""
build_canvas_and_annotations.py

Runs PaddleOCR-VL with a fine-tuned layout detection model and saves:

1. Raw annotation JSON from PaddleOCR-VL
2. White-background canvas PNG with:
   - rectangle boxes
   - OCR text rendered inside boxes
   - B{ID} badges

No GPT / OpenAI usage.

Directory layout expected:
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

Usage:
    python build_canvas_and_annotations.py

Requirements:
    pip install paddleocr pillow

Notes:
- Update FINETUNED_LAYOUT_MODEL_DIR to your exported inference directory.
- This script prefers parsing_res_list from PaddleOCR-VL output.
- If parsing_res_list is unavailable, it falls back to rec_texts/rec_boxes.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT       = Path("./dataset/valid")
IMAGES_DIR      = DATA_ROOT / "images"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
CANVAS_DIR      = DATA_ROOT / "canvas"

SKIP_EXISTING = True

# Your fine-tuned layout detector export directory
FINETUNED_LAYOUT_MODEL_DIR = "./paddlex_output/best_model/inference"

# PaddleOCR-VL settings
USE_GPU = True
DEVICE = "gpu:0" if USE_GPU else "cpu"

USE_DOC_ORIENTATION_CLASSIFY = True
USE_DOC_UNWARPING = True
USE_LAYOUT_DETECTION = True

# try "small" if "large" merges too aggressively
LAYOUT_MERGE_BBOXES_MODE = "large"

# Render settings
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 16
TEXT_PAD = 6
LINE_GAP = 3
BOX_OUTLINE_WIDTH = 3
BADGE_FONT_SIZE = 14

# ============================================================
# FILE NAMING
# ============================================================

def extract_numeric_suffix(path: Path):
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else None

def _data_root(data_root: Path | None) -> Path:
    return data_root if data_root is not None else DATA_ROOT


def annotation_path_for(image_path: Path, data_root: Path | None = None) -> Path:
    idx = extract_numeric_suffix(image_path)
    if idx is None:
        raise ValueError(f"No numeric suffix in {image_path.name}")
    return _data_root(data_root) / "annotations" / f"annotation{idx}.json"

def canvas_path_for(image_path: Path, data_root: Path | None = None) -> Path:
    idx = extract_numeric_suffix(image_path)
    if idx is None:
        raise ValueError(f"No numeric suffix in {image_path.name}")
    return _data_root(data_root) / "canvas" / f"canvas{idx}.png"

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
    Build normalized block list from PaddleOCR-VL output.
    Despite the function name, this is used only for rendering/annotation here.
    """
    width = raw_data.get("width")
    height = raw_data.get("height")

    if not width or not height:
        with Image.open(image_path) as im:
            width, height = im.size

    blocks = []
    next_id = 0

    # Preferred: structured parsing output
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
                "label": item.get("block_label"),
            })
            next_id += 1

    # Fallback: OCR-only output
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
                "label": None,
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

# ============================================================
# CANVAS RENDERING
# ============================================================

def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def wrap_text(draw, text, font, max_width):
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

def fit_text_to_box(draw, text, box_w, box_h,
                    min_font_size=MIN_FONT_SIZE,
                    max_font_size=MAX_FONT_SIZE,
                    pad=TEXT_PAD,
                    line_gap=LINE_GAP):
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

def place_badge(width, height, x1, y1, x2, y2, badge_w, badge_h, used_badges):
    candidates = [
        (x1, y1 - badge_h - 2),
        (x2 - badge_w, y1 - badge_h - 2),
        (x1 + 2, y1 + 2),
        (x2 - badge_w - 2, y1 + 2),
        (x1, y2 + 2),
        (x2 - badge_w, y2 + 2),
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

    bx1 = max(0, min(width - badge_w, x1 + 2))
    by1 = max(0, min(height - badge_h, y1 + 2))
    bx2 = bx1 + badge_w
    by2 = by1 + badge_h
    used_badges.append((bx1, by1, bx2, by2))
    return bx1, by1, bx2, by2

def render_layout_canvas(blocks_data: dict, save_path: Path):
    width = int(blocks_data["width"])
    height = int(blocks_data["height"])

    base = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    used_badges = []
    badges_to_draw = []

    for b in blocks_data["blocks"]:
        x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"]]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        box_w = x2 - x1
        box_h = y2 - y1

        # White background canvas, black unfilled boxes
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0, 255), width=BOX_OUTLINE_WIDTH)

        font, lines = fit_text_to_box(draw, b["text"], box_w, box_h)

        y_cursor = y1 + TEXT_PAD
        for ln in lines:
            tb = draw.textbbox((0, 0), ln if ln else "Ag", font=font)
            line_h = tb[3] - tb[1]

            if y_cursor + line_h > y2 - 4:
                break

            draw.text((x1 + TEXT_PAD, y_cursor), ln, fill=(0, 0, 0, 255), font=font)
            y_cursor += line_h + LINE_GAP

        badge_text = f"B{b['id']}"
        badge_font = load_font(BADGE_FONT_SIZE)
        tb = draw.textbbox((0, 0), badge_text, font=badge_font)
        badge_w = (tb[2] - tb[0]) + 8
        badge_h = (tb[3] - tb[1]) + 4

        bx1, by1, bx2, by2 = place_badge(
            width, height, x1, y1, x2, y2, badge_w, badge_h, used_badges
        )
        badges_to_draw.append((bx1, by1, bx2, by2, badge_text, badge_font))

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

def create_vl_pipeline(layout_model_dir: str | None = None, device: str | None = None):
    """
    Creates a PaddleOCR-VL pipeline using your fine-tuned layout model.
    """
    from paddleocr import PaddleOCRVL

    resolved = layout_model_dir if layout_model_dir is not None else FINETUNED_LAYOUT_MODEL_DIR
    dev = device if device is not None else DEVICE

    model_dir = str(Path(resolved).resolve())
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Layout model directory not found: {model_dir}")

    pipeline = PaddleOCRVL(
        device=dev,
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_layout_detection=USE_LAYOUT_DETECTION,
        # This name may need adjusting depending on your installed version.
        layout_detection_model_name="PP-DocLayoutV2",
        layout_detection_model_dir=model_dir,
    )
    return pipeline

def run_vl_and_get_raw(pipeline, image_path: Path, data_root: Path | None = None) -> dict:
    """
    Runs PaddleOCR-VL and returns the raw result as a JSON-serializable dict.
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

    tmp_dir = _data_root(data_root) / "annotations" / "_tmp_vl_json"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Use PaddleOCR-VL's save API, then read the written JSON back in
    res.save_to_json(str(tmp_dir))
    written = sorted(tmp_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not written:
        raise RuntimeError("PaddleOCR-VL save_to_json wrote no files")

    return load_json(written[-1])

# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one_image(
    pipeline,
    image_path: Path,
    *,
    data_root: Path | None = None,
    skip_existing: bool | None = None,
):
    skip = SKIP_EXISTING if skip_existing is None else skip_existing
    annotation_path = annotation_path_for(image_path, data_root=data_root)
    canvas_path = canvas_path_for(image_path, data_root=data_root)

    if skip and annotation_path.exists() and canvas_path.exists():
        print(f"[skip] {image_path.name}")
        return

    if skip and annotation_path.exists() and not canvas_path.exists():
        print(f"[re-render canvas] {image_path.name}")
        raw_data = load_json(annotation_path)
        blocks_data = build_llm_ready_blocks(raw_data, image_path)
        render_layout_canvas(blocks_data, canvas_path)
        print(f"  -> canvas: {canvas_path}")
        return

    print(f"[ocr-vl] {image_path.name}")
    raw_data = run_vl_and_get_raw(pipeline, image_path, data_root=data_root)
    save_json(annotation_path, raw_data)

    blocks_data = build_llm_ready_blocks(raw_data, image_path)
    render_layout_canvas(blocks_data, canvas_path)

    print(f"  -> annotation: {annotation_path}")
    print(f"  -> canvas:     {canvas_path}")
    print(f"  -> blocks:     {len(blocks_data['blocks'])}")

# ============================================================
# MAIN
# ============================================================

def main():
    for d in [IMAGES_DIR, ANNOTATIONS_DIR, CANVAS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in IMAGES_DIR.glob("out*.png")],
        key=lambda p: extract_numeric_suffix(p) if extract_numeric_suffix(p) is not None else -1,
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    pipeline = create_vl_pipeline()

    print(f"Found {len(image_paths)} image(s)")
    for image_path in image_paths:
        try:
            process_one_image(pipeline, image_path)
        except Exception as e:
            print(f"[error] {image_path.name}: {e}")

if __name__ == "__main__":
    main()