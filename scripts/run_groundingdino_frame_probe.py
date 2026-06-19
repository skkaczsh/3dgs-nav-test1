#!/usr/bin/env python3
"""Run a small GroundingDINO frame-level probe for fine-object candidates.

This is intentionally a probe, not a production segmentation stage.  It checks
whether a text-conditioned detector can provide narrower source candidates for
fine objects such as railings and cars before we spend compute on SAM/mask
refinement and point-cloud projection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_PROMPTS = {
    "railing": [
        "railing",
        "guardrail",
        "handrail",
        "metal fence",
        "stair railing",
        "safety railing",
    ],
    "car": [
        "car",
        "parked car",
        "vehicle",
        "van",
        "truck",
    ],
}


def iter_frame_paths(frame_root: Path, cams: list[int], start: int, end: int, stride: int):
    wanted = set(range(start, end + 1, max(stride, 1)))
    for cam_id in cams:
        cam_dir = frame_root / f"cam{cam_id}"
        for path in sorted(cam_dir.glob("frame_*.jpg")):
            try:
                frame_id = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            if frame_id in wanted:
                yield cam_id, frame_id, path


def parse_prompt_args(values: list[str]) -> dict[str, list[str]]:
    prompts = {key: list(items) for key, items in DEFAULT_PROMPTS.items()}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"prompt override must be label=text1,text2: {raw!r}")
        label, text = raw.split("=", 1)
        label = label.strip()
        items = [item.strip() for item in text.split(",") if item.strip()]
        if not label or not items:
            raise ValueError(f"empty prompt override: {raw!r}")
        prompts[label] = items
    return prompts


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detections_for_image(
    processor,
    model,
    image: Image.Image,
    prompt_map: dict[str, list[str]],
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
    text: list[str] = []
    for label, prompts in prompt_map.items():
        for prompt in prompts:
            text.append(prompt)
    caption = ". ".join(text)
    if caption and not caption.endswith("."):
        caption += "."

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    detections: list[dict[str, Any]] = []
    scores = results.get("scores", [])
    boxes = results.get("boxes", [])
    text_labels = results.get("text_labels", results.get("labels", []))
    for score, box, text_label in zip(scores, boxes, text_labels):
        phrase = str(text_label)
        canonical = canonical_label(phrase, prompt_map)
        detections.append(
            {
                "label": canonical,
                "phrase": phrase,
                "score": float(score.detach().cpu().item() if hasattr(score, "detach") else score),
                "bbox_xyxy": [float(v) for v in box.detach().cpu().tolist()],
            }
        )
    return detections


def canonical_label(phrase: str, prompt_map: dict[str, list[str]]) -> str:
    text = phrase.lower()
    best_label = "unknown"
    best_len = 0
    for label, prompts in prompt_map.items():
        for prompt in prompts:
            p = prompt.lower()
            if p in text or text in p:
                if len(p) > best_len:
                    best_label = label
                    best_len = len(p)
    return best_label


def detection_stats(detections: list[dict[str, Any]], image_size: tuple[int, int], large_box_ratio: float = 0.12) -> dict[str, Any]:
    w, h = image_size
    counts = Counter(det["label"] for det in detections)
    area_by_label: Counter[str] = Counter()
    large_by_label: Counter[str] = Counter()
    for det in detections:
        x0, y0, x1, y1 = det["bbox_xyxy"]
        x0 = min(max(0.0, float(x0)), float(w))
        y0 = min(max(0.0, float(y0)), float(h))
        x1 = min(max(0.0, float(x1)), float(w))
        y1 = min(max(0.0, float(y1)), float(h))
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        label = str(det["label"])
        area_by_label[label] += int(round(area))
        if area / float(max(w * h, 1)) >= large_box_ratio:
            large_by_label[label] += 1
    total = float(max(w * h, 1))
    return {
        "detection_counts": dict(counts),
        "box_area_pixels": dict(area_by_label),
        "box_area_ratios": {key: value / total for key, value in area_by_label.items()},
        "large_box_counts": dict(large_by_label),
    }


def aggregate_detection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    large_counts = Counter()
    area_sum = Counter()
    area_max: dict[str, float] = {}
    for row in rows:
        counts.update(row.get("detection_counts", {}))
        large_counts.update(row.get("large_box_counts", {}))
        for label, ratio in (row.get("box_area_ratios") or {}).items():
            area_sum[str(label)] += float(ratio)
            area_max[str(label)] = max(area_max.get(str(label), 0.0), float(ratio))
    return {
        "detection_counts": dict(counts),
        "large_box_counts": dict(large_counts),
        "mean_box_area_ratio_by_label": {
            label: float(area_sum[label] / max(counts[label], 1))
            for label in sorted(counts)
        },
        "max_box_area_ratio_by_label": area_max,
    }


def annotate(image_bgr: np.ndarray, detections: list[dict[str, Any]], output: Path) -> None:
    colors = {
        "railing": (0, 220, 255),
        "car": (80, 80, 255),
        "unknown": (255, 255, 255),
    }
    out = image_bgr.copy()
    for det in detections:
        x0, y0, x1, y1 = [int(round(v)) for v in det["bbox_xyxy"]]
        label = str(det["label"])
        color = colors.get(label, (255, 255, 255))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        caption = f"{label}:{det['phrase']} {float(det['score']):.2f}"
        cv2.putText(out, caption[:80], (max(0, x0), max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), out)


def make_contact_sheet(paths: list[Path], output: Path, thumb_width: int = 320, cols: int = 4) -> None:
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = thumb_width / max(image.shape[1], 1)
        resized = cv2.resize(image, (thumb_width, max(1, int(image.shape[0] * scale))))
        thumbs.append(resized)
    if not thumbs:
        return
    max_h = max(img.shape[0] for img in thumbs)
    padded = []
    for img in thumbs:
        if img.shape[0] < max_h:
            pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)
    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--large-box-ratio", type=float, default=0.12)
    parser.add_argument("--prompt", action="append", default=[], help="Override prompt group, e.g. railing=railing,handrail")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    prompt_map = parse_prompt_args(args.prompt)
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "annotated").mkdir(exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device)
    model.eval()

    frame_items = list(iter_frame_paths(args.frame_root, args.cams, args.start, args.end, args.stride))
    if args.max_images:
        frame_items = frame_items[: args.max_images]

    rows: list[dict[str, Any]] = []
    annotated_paths: list[Path] = []
    for cam_id, frame_id, image_path in frame_items:
        image = Image.open(image_path).convert("RGB")
        detections = detections_for_image(
            processor,
            model,
            image,
            prompt_map,
            device,
            args.box_threshold,
            args.text_threshold,
        )
        image_id = f"cam{cam_id}_{frame_id:06d}"
        annotated_path = args.output_dir / "annotated" / f"{image_id}_gdino.jpg"
        annotate(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), detections, annotated_path)
        annotated_paths.append(annotated_path)
        rows.append(
            {
                "image_id": image_id,
                "cam_id": cam_id,
                "frame_id": frame_id,
                "image_path": str(image_path),
                "annotated_path": str(annotated_path),
                "detections": detections,
                **detection_stats(detections, image.size, args.large_box_ratio),
            }
        )
        print(json.dumps({"image_id": image_id, "detections": len(detections), "counts": rows[-1]["detection_counts"]}, ensure_ascii=False))

    aggregate = aggregate_detection_summary(rows)
    report = {
        "frame_root": str(args.frame_root),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "device": str(device),
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "cams": args.cams,
        "image_count": len(rows),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "large_box_ratio": args.large_box_ratio,
        "prompts": prompt_map,
        **aggregate,
        "items": rows,
    }
    (args.output_dir / "groundingdino_frame_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_path = args.output_dir / "groundingdino_frame_probe.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    make_contact_sheet(annotated_paths[:48], args.output_dir / "groundingdino_frame_probe_contact.jpg")
    print(json.dumps({"output_dir": str(args.output_dir), "image_count": len(rows), "detection_counts": report["detection_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
