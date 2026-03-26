#!/usr/bin/env python3
"""
train.py

Fine-tunes Qwen3-VL-4B on the event extraction dataset built by build_dataset.py.

Each training sample:
  - Image 1: original flyer (out{N}.png)   <-- source of truth
  - Image 2: annotated canvas (canvas{N}.png)
  - Text:    shared prompt + shared BLOCKS_JSON
  - Target:  {"events": [...]} from target{N}.json

Requirements:
    pip install unsloth trl datasets torch pillow

Usage:
    python train.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import Dataset
from PIL import Image
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from trl import SFTConfig, SFTTrainer

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT       = Path("./dataset/train")
IMAGES_DIR      = DATA_ROOT / "images"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
CANVAS_DIR      = DATA_ROOT / "canvas"
TARGETS_DIR     = DATA_ROOT / "targets"

MODEL_NAME = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
OUTPUT_DIR = "./outputs_qwen3vl_events"
LORA_SAVE_DIR = "./qwen3vl_events_lora"

MAX_STEPS              = 200
PER_DEVICE_BATCH_SIZE  = 1
GRAD_ACCUM_STEPS       = 4
LEARNING_RATE          = 2e-4
MAX_LENGTH             = 4096

FINETUNE_VISION_LAYERS   = True
FINETUNE_LANGUAGE_LAYERS = True
FINETUNE_ATTENTION       = True
FINETUNE_MLP             = True

LORA_R        = 8
LORA_ALPHA    = 8
LORA_DROPOUT  = 0.0
RANDOM_STATE  = 3407

IGNORE_BLOCK_LABELS          = set()
OPTIONAL_IGNORE_BLOCK_LABELS = set()

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
# FILE HELPERS
# ============================================================

def extract_numeric_suffix(path: Path):
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else None

def get_paths_for_index(idx: int):
    return {
        "image":      IMAGES_DIR      / f"out{idx}.png",
        "annotation": ANNOTATIONS_DIR / f"annotation{idx}.json",
        "canvas":     CANVAS_DIR      / f"canvas{idx}.png",
        "target":     TARGETS_DIR     / f"target{idx}.json",
    }

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# BLOCK PROCESSING
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
    Shared block format for training.
    Supports:
      1) parsing_res_list style outputs
      2) rec_texts + rec_boxes fallback
    """
    width = raw_data.get("width")
    height = raw_data.get("height")

    if not width or not height:
        with Image.open(image_path) as im:
            width, height = im.size

    blocks = []
    next_id = 0

    # Case 1: structured parsing output
    if isinstance(raw_data.get("parsing_res_list"), list):
        for item in raw_data["parsing_res_list"]:
            if not isinstance(item, dict):
                continue

            if item.get("block_label") in IGNORE_BLOCK_LABELS:
                continue
            if item.get("block_label") in OPTIONAL_IGNORE_BLOCK_LABELS:
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
                "id":              next_id,
                "text":            text,
                "bbox":            bbox,
                "bbox_1000":       normalize_bbox_1000(bbox, width, height),
                "source_block_id": item.get("block_id"),
                "group_id":        item.get("group_id"),
                "block_order":     item.get("block_order", next_id),
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
                "id":              next_id,
                "text":            text,
                "bbox":            bbox,
                "bbox_1000":       normalize_bbox_1000(bbox, width, height),
                "source_block_id": None,
                "group_id":        None,
                "block_order":     order,
            })
            next_id += 1

    blocks = sorted(
        blocks,
        key=lambda b: (
            b["block_order"] is None,
            b["block_order"] if b["block_order"] is not None else 10**9,
        ),
    )

    return {"width": width, "height": height, "blocks": blocks}

def build_blocks_json_for_prompt(llm_ready: dict) -> str:
    """
    Shared schema for BOTH build_dataset.py and train.py.
    """
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
        "2. Prefer faithful extraction over normalization. Copy values as they appear in the ORIGINAL FLYER.\n"
        "3. Do not normalize or reformat dates/times.\n"
        "4. Output one object per distinct event.\n"
        "5. If the same event appears multiple times, output it once.\n"
        "6. Ignore decorative text, watermarks, branding, and headers/footers unless they are clearly part of a specific event.\n"
        "7. Some event text may be stylized or embedded visually; use the ORIGINAL FLYER to recover it when Canvas/BLOCKS_JSON are imperfect.\n"
        "8. Use both visual layout and block text to decide event boundaries, but resolve disagreements in favor of the ORIGINAL FLYER.\n"
        "9. List events in reading order: top-to-bottom, then left-to-right.\n"
        "10. If Canvas/BLOCKS_JSON are missing a word that is visible in the ORIGINAL FLYER, include the visible word.\n"
        "11. If Canvas/BLOCKS_JSON include an OCR error, correct it using the ORIGINAL FLYER.\n\n"

        f"BLOCKS_JSON:\n{blocks_json}\n\n"

        "Output format:\n"
        "Return ONLY a valid JSON object with exactly one top-level key: \"events\".\n"
        "No explanation. No markdown fences.\n\n"

        f"Example:\n{json.dumps(OUTPUT_SCHEMA_EXAMPLE, ensure_ascii=False)}"
    )

# ============================================================
# DATASET BUILDING
# ============================================================

def normalize_target_obj(target_data):
    """
    Ensure target is always {"events": [...]}
    """
    if isinstance(target_data, dict) and isinstance(target_data.get("events"), list):
        return {"events": target_data["events"]}
    if isinstance(target_data, list):
        return {"events": target_data}
    raise ValueError(f"Invalid target format: {type(target_data)}")

def build_training_samples() -> list[dict]:
    samples = []

    image_paths = sorted(
        [p for p in IMAGES_DIR.glob("out*.png")],
        key=lambda p: extract_numeric_suffix(p) if extract_numeric_suffix(p) is not None else -1,
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    for image_path in image_paths:
        idx = extract_numeric_suffix(image_path)
        if idx is None:
            print(f"[skip] {image_path.name}: no numeric suffix")
            continue

        paths = get_paths_for_index(idx)
        missing = [k for k, v in paths.items() if not v.exists()]
        if missing:
            print(f"[skip] index {idx}: missing {missing}")
            continue

        raw_annotation = load_json(paths["annotation"])
        llm_ready = build_llm_ready_blocks(raw_annotation, paths["image"])
        blocks_json_str = build_blocks_json_for_prompt(llm_ready)
        prompt_text = build_prompt(blocks_json_str)

        target_obj = normalize_target_obj(load_json(paths["target"]))
        target_text = json.dumps(target_obj, ensure_ascii=False)

        samples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(paths["image"].resolve())},
                        {"type": "image", "image": str(paths["canvas"].resolve())},
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_text}],
                },
            ]
        })

        print(
            f"[sample] index {idx} — "
            f"{len(llm_ready['blocks'])} blocks, "
            f"{len(target_obj['events'])} events"
        )

    if not samples:
        raise RuntimeError("No training samples were built.")

    return samples

# ============================================================
# MAIN
# ============================================================

def main():
    print("Building training samples.")
    samples = build_training_samples()
    train_dataset = Dataset.from_list(samples)
    print(f"\nBuilt {len(train_dataset)} training sample(s)\n")

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=FINETUNE_VISION_LAYERS,
        finetune_language_layers=FINETUNE_LANGUAGE_LAYERS,
        finetune_attention_modules=FINETUNE_ATTENTION,
        finetune_mlp_modules=FINETUNE_MLP,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        random_state=RANDOM_STATE,
        use_rslora=False,
        loftq_config=None,
    )

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            warmup_steps=5,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=RANDOM_STATE,
            output_dir=OUTPUT_DIR,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_LENGTH,
        ),
    )

    trainer.train()

    print(f"\nSaving LoRA adapter to: {LORA_SAVE_DIR}")
    model.save_pretrained(LORA_SAVE_DIR)
    tokenizer.save_pretrained(LORA_SAVE_DIR)

    print("Done.")

if __name__ == "__main__":
    main()