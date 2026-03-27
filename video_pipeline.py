from __future__ import annotations

# NOTE: Default mode runs Paddle on all sampled frames, then Qwen on all sampled frames in one process.
# Use --qwen-per-frame when you hit GPU OOM; it runs one frame per subprocess to hard-reset VRAM between frames.
# For cloud/high-VRAM inference, prefer batched Qwen (default) for better throughput.

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent

@dataclass
class Sample:
    sec: int
    frame_index: int
    text: str
    triggered_full: bool
    reason: str
    image_name: str | None


def _default_python_for_env(env_name: str) -> str:
    root = REPO_ROOT
    # Support both Windows venv layout (`Scripts/python.exe`) and POSIX venv layout
    # (`bin/python`). This matters when running under WSL where `os.name != "nt"`
    # but your venv might still be the Windows one.
    candidates = [
        root / env_name / "Scripts" / "python.exe",
        root / env_name / "bin" / "python",
        root / env_name / "bin" / "python3",
    ]
    for p in candidates:
        try:
            if p.exists():
                return str(p)
        except Exception:
            # If Path.exists() fails for any reason, just keep trying candidates.
            pass

    # Fall back to the platform-typical path so the error is at least actionable.
    if os.name == "nt":
        return str(root / env_name / "Scripts" / "python.exe")
    return str(root / env_name / "bin" / "python")


def _run_sampling_only_in_current_process(
    *,
    video_path: Path,
    data_root: Path,
    sample_seconds: float,
    change_threshold: float,
    max_seconds: int,
) -> None:
    import cv2

    images_dir = data_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Avoid carrying sampled frames from a previous run.
    for p in images_dir.glob("out*.png"):
        try:
            p.unlink()
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = int(total_frames / fps) if fps > 0 else 0
    if max_seconds > 0:
        duration_s = min(duration_s, max_seconds)

    print(f"[video] fps={fps:.2f} total_frames={total_frames} duration_s={duration_s}")
    print(f"[sampling] fixed interval={sample_seconds}s (change_threshold ignored)")

    sampled_secs: List[int] = []
    samples: List[Sample] = []
    next_idx = 1

    sample_step_s = max(1, int(round(sample_seconds)))
    for sec in range(0, duration_s + 1, sample_step_s):
        frame_idx = min(total_frames - 1, int(round(sec * fps))) if total_frames > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        image_name = f"out{next_idx}.png"
        out_path = images_dir / image_name
        cv2.imwrite(str(out_path), frame)
        sampled_secs.append(sec)
        next_idx += 1

        samples.append(
            Sample(
                sec=sec,
                frame_index=frame_idx,
                text="",
                triggered_full=True,
                reason="fixed_interval",
                image_name=image_name,
            )
        )

    cap.release()

    if not sampled_secs:
        raise RuntimeError("No sampled frames selected")

    print(f"[video] sampled {len(sampled_secs)} frame(s): {sampled_secs}")

    log = {
        "video_path": str(video_path),
        "sample_seconds": sample_seconds,
        "change_threshold": change_threshold,
        "sampled_secs": sampled_secs,
        "samples": [s.__dict__ for s in samples],
    }
    (data_root / "sampling_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
def _run_paddle_stage(*, paddle_python: str, data_root: Path, layout_dir: str | None) -> None:
    cmd = [
        paddle_python,
        str(REPO_ROOT / "paddle_ocr_vl_pipeline.py"),
        "--data-root",
        str(data_root),
    ]
    if layout_dir:
        cmd += ["--layout-model-dir", layout_dir]

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _run_qwen_stage(
    *,
    unsloth_python: str,
    data_root: Path,
    qwen_lora: str,
    qwen_base: str,
    max_new_tokens: int,
    temperature: float,
    per_frame: bool,
) -> None:
    if not per_frame:
        cmd = [
            unsloth_python,
            str(REPO_ROOT / "qwen_pipeline.py"),
            "--data-root",
            str(data_root),
            "--lora",
            qwen_lora,
            "--base",
            qwen_base,
            "--max-new-tokens",
            str(max_new_tokens),
            "--temperature",
            str(temperature),
            "--overwrite",
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        return

    image_paths = sorted(
        (data_root / "images").glob("out*.png"),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or "0"),
    )

    if not image_paths:
        raise FileNotFoundError(f"No sampled images found in {data_root / 'images'}")

    for img_path in image_paths:
        stem_digits = "".join(ch for ch in img_path.stem if ch.isdigit())
        if not stem_digits:
            continue
        idx = int(stem_digits)
        cmd = [
            unsloth_python,
            str(REPO_ROOT / "qwen_pipeline.py"),
            "--data-root",
            str(data_root),
            "--lora",
            qwen_lora,
            "--base",
            qwen_base,
            "--max-new-tokens",
            str(max_new_tokens),
            "--temperature",
            str(temperature),
            "--overwrite",
            "--only-index",
            str(idx),
        ]
        print(f"[qwen] processing frame index {idx}")
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _run_enrichment(
    *,
    enrich_python: str,
    data_root: Path,
    out_dir: Path,
    top_k: int,
    serper_num: int,
    predictions_glob: str = "prediction*.json",
) -> None:
    cmd = [
        enrich_python,
        str(REPO_ROOT / "enrich_qwen_predictions.py"),
        "--data-root",
        str(data_root),
        "--out-dir",
        str(out_dir),
        "--top-k",
        str(top_k),
        "--serper-num",
        str(serper_num),
        "--predictions-glob",
        predictions_glob,
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _run_qwen_dedupe_stage(*, enrich_python: str, data_root: Path) -> Path:
    out_path = data_root / "predictions" / "prediction_dedup.json"
    cmd = [
        enrich_python,
        str(REPO_ROOT / "dedupe_qwen_predictions.py"),
        "--pred-dir",
        str(data_root / "predictions"),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Video -> adaptive lightweight OCR -> full flyer pipeline")
    ap.add_argument("--video-path", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True, help="Output root containing images/annotations/canvas/predictions")
    ap.add_argument("--sample-seconds", type=float, default=2.0, help="Lightweight OCR interval in seconds")
    ap.add_argument("--change-threshold", type=float, default=0.72, help="Trigger full pipeline when Jaccard similarity drops below threshold")
    ap.add_argument("--max-seconds", type=int, default=0, help="Optional cap; 0 means full video")
    ap.add_argument("--sampling-only", action="store_true", help="Only sample trigger frames (writes out*.png + sampling_log.json)")

    ap.add_argument("--qwen-lora", type=str, default="./qwen3vl_events_lora")
    ap.add_argument("--qwen-base", type=str, default="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit")
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--qwen-per-frame", action="store_true", help="Run Qwen as one subprocess per frame to reset VRAM between frames (slower, but more stable on low VRAM)")

    ap.add_argument("--enrich-top-k", type=int, default=3)
    ap.add_argument("--enrich-serper-num", type=int, default=10)

    ap.add_argument("--paddle-python", type=str, default=os.getenv("PADDLE_PYTHON", _default_python_for_env("paddle_vl_env")))
    ap.add_argument("--unsloth-python", type=str, default=os.getenv("UNSLOTH_PYTHON", _default_python_for_env("unsloth_env")))
    ap.add_argument("--enrich-python", type=str, default=os.getenv("ENRICH_PYTHON", sys.executable))
    ap.add_argument("--layout-model-dir", type=str, default=os.getenv("PADDLE_LAYOUT_DIR", ""))

    args = ap.parse_args()
    # Interpreter paths can come from env vars; strip to avoid accidental whitespace/newlines.
    args.paddle_python = str(args.paddle_python).strip()
    args.unsloth_python = str(args.unsloth_python).strip()
    args.enrich_python = str(args.enrich_python).strip()
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    if args.sampling_only:
        _run_sampling_only_in_current_process(
            video_path=args.video_path,
            data_root=args.data_root,
            sample_seconds=args.sample_seconds,
            change_threshold=args.change_threshold,
            max_seconds=args.max_seconds,
        )
        return

    # 1) Sample trigger frames in Paddle env (keeps PaddleOCR deps out of launcher env).
    paddle_py = Path(args.paddle_python)
    if not paddle_py.exists():
        raise FileNotFoundError(
            f"Paddle python executable not found: {args.paddle_python}\n"
            f"Run with --paddle-python \"<path-to-paddle-venv-python>\".\n"
            f"This script defaults to `paddle_vl_env` under the repo root; "
            f"if that venv exists only in Windows layout, pass its `Scripts/python.exe` explicitly."
        )

    sampling_cmd = [
        args.paddle_python,
        str(REPO_ROOT / "video_pipeline.py"),
        "--sampling-only",
        "--video-path",
        str(args.video_path),
        "--data-root",
        str(args.data_root),
        "--sample-seconds",
        str(args.sample_seconds),
        "--change-threshold",
        str(args.change_threshold),
        "--max-seconds",
        str(args.max_seconds),
    ]
    subprocess.run(sampling_cmd, cwd=str(REPO_ROOT), check=True)

    sampling_log = {}
    sampling_log_path = args.data_root / "sampling_log.json"
    if sampling_log_path.exists():
        sampling_log = json.loads(sampling_log_path.read_text(encoding="utf-8"))

    sampled_secs = sampling_log.get("sampled_secs", sampling_log.get("trigger_secs", []))

    print(f"[video] sampled {len(sampled_secs)} frame(s)")
    if not sampled_secs:
        raise RuntimeError("No sampled frames selected (unexpected; check sampling_log.json)")

    data_root = args.data_root
    print(f"[env] paddle_python={args.paddle_python}")
    print(f"[env] unsloth_python={args.unsloth_python}")
    print(f"[env] enrich_python={args.enrich_python}")

    # Stage 1: Paddle in paddle env
    _run_paddle_stage(
        paddle_python=args.paddle_python,
        data_root=data_root,
        layout_dir=(args.layout_model_dir or None),
    )

    # Stage 2: Qwen in unsloth env
    _run_qwen_stage(
        unsloth_python=args.unsloth_python,
        data_root=data_root,
        qwen_lora=args.qwen_lora,
        qwen_base=args.qwen_base,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        per_frame=args.qwen_per_frame,
    )

    # Stage 3: merge Qwen predictions across sampled frames
    dedup_pred_path = _run_qwen_dedupe_stage(enrich_python=args.enrich_python, data_root=data_root)

    # Stage 4: Enrichment once on deduped predictions
    out_dir = data_root / "enriched_results"
    _run_enrichment(
        enrich_python=args.enrich_python,
        data_root=data_root,
        out_dir=out_dir,
        top_k=args.enrich_top_k,
        serper_num=args.enrich_serper_num,
        predictions_glob=dedup_pred_path.name,
    )

    log = dict(sampling_log)
    log["paddle_python"] = args.paddle_python
    log["unsloth_python"] = args.unsloth_python
    log["enrich_python"] = args.enrich_python
    (data_root / "sampling_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] sampling log: {data_root / 'sampling_log.json'}")


if __name__ == "__main__":
    main()
