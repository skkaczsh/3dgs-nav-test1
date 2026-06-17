#!/usr/bin/env python3
"""Segment priority large surfaces/objects from extracted frames.

This stage intentionally removes stable, high-priority scene regions before
free-form residual object clustering:

- ground / road
- wall / building
- grass / vegetation
- car / vehicle
- railing / fence
- sky

It uses universal semantic segmentation weights by default. The output is a
compact priority-id PNG and a report JSONL per image. Later DINOv3/dino.txt
providers can write the same priority-id schema without changing projection and
residual clustering stages.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


PRIORITY_IDS = {
    "background": 0,
    "ground": 1,
    "wall": 2,
    "grass": 3,
    "car": 4,
    "railing": 5,
    "sky": 6,
}

PALETTE = {
    0: (0, 0, 0),
    1: (196, 168, 112),
    2: (120, 150, 180),
    3: (80, 160, 80),
    4: (235, 90, 80),
    5: (240, 210, 60),
    6: (90, 170, 235),
}

MODEL_ALIASES = {
    "mapillary": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    "cityscapes": "facebook/mask2former-swin-large-cityscapes-semantic",
    "ade20k": "facebook/mask2former-swin-large-ade-semantic",
}


def normalize(label: str) -> str:
    return label.lower().replace("_", " ").replace("-", " ").replace("/", " ")


def map_label(label: str) -> str | None:
    text = normalize(label)
    if "sky" in text:
        return "sky"
    if any(t in text for t in ["car", "truck", "bus", "vehicle", "van"]):
        return "car"
    if any(t in text for t in ["fence", "railing", "guard rail", "guardrail", "handrail", "rail track"]):
        return "railing"
    if any(t in text for t in ["grass", "vegetation", "terrain", "plant", "tree"]):
        return "grass"
    if any(t in text for t in ["road", "sidewalk", "ground", "floor", "earth", "parking", "pavement"]):
        return "ground"
    if any(t in text for t in ["wall", "building", "bridge", "tunnel", "polegroup"]):
        return "wall"
    return None


def colorize(priority: np.ndarray) -> Image.Image:
    rgb = np.zeros((priority.shape[0], priority.shape[1], 3), dtype=np.uint8)
    for idx, color in PALETTE.items():
        rgb[priority == idx] = color
    return Image.fromarray(rgb)


def overlay(image: Image.Image, priority: np.ndarray, alpha: float) -> Image.Image:
    mask_rgb = colorize(priority).convert("RGB")
    return Image.blend(image.convert("RGB"), mask_rgb, alpha)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_images(frame_root: Path, cams: list[int], start: int, end: int, stride: int):
    wanted = set(range(start, end + 1, stride))
    for cam_id in cams:
        cam_dir = frame_root / f"cam{cam_id}"
        for path in sorted(cam_dir.glob("frame_*.jpg")):
            try:
                frame_id = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            if frame_id in wanted:
                yield cam_id, frame_id, path


def batched(items: list, batch_size: int):
    for i in range(0, len(items), max(1, batch_size)):
        yield items[i:i + max(1, batch_size)]


def postprocess_priority(pred: np.ndarray, id2label: dict[int, str]) -> tuple[np.ndarray, dict]:
    priority = np.zeros_like(pred, dtype=np.uint8)
    raw_counts = Counter()
    mapped_counts = Counter()
    for raw_id in np.unique(pred):
        label = id2label.get(int(raw_id), f"id_{raw_id}")
        count = int((pred == raw_id).sum())
        raw_counts[label] += count
        mapped = map_label(label)
        if mapped:
            priority[pred == raw_id] = PRIORITY_IDS[mapped]
            mapped_counts[mapped] += count
    total = float(pred.size)
    return priority, {
        "raw_label_counts": dict(raw_counts),
        "priority_counts": dict(mapped_counts),
        "priority_ratios": {k: v / total for k, v in mapped_counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model", default="mapillary",
                        help="Alias: mapillary/cityscapes/ade20k, or HF model id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for faster inference.")
    parser.add_argument("--report-suffix", default="")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    model_id = MODEL_ALIASES.get(args.model, args.model)
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "priority").mkdir(exist_ok=True)
    (args.output_dir / "overlay").mkdir(exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id, use_safetensors=True
    ).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    images = list(iter_images(args.frame_root, args.cams, args.start, args.end, args.stride))
    if args.max_images:
        images = images[:args.max_images]
    report_name = "priority_segmentation_report"
    if args.report_suffix:
        report_name += f"_{args.report_suffix}"
    report_path = args.output_dir / f"{report_name}.jsonl"
    summary = {
        "frame_root": str(args.frame_root),
        "output_dir": str(args.output_dir),
        "model": model_id,
        "device": str(device),
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "cams": args.cams,
        "image_count": len(images),
        "batch_size": args.batch_size,
        "amp": args.amp,
        "report_path": str(report_path),
    }
    (args.output_dir / "priority_segmentation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    processed = 0
    with report_path.open("a" if args.skip_existing else "w", encoding="utf-8") as report_f:
        for batch in batched(images, args.batch_size):
            pending = []
            pil_images = []
            for cam_id, frame_id, image_path in batch:
                image_id = f"cam{cam_id}_{frame_id:06d}"
                priority_path = args.output_dir / "priority" / f"{image_id}_priority.png"
                overlay_path = args.output_dir / "overlay" / f"{image_id}_overlay.jpg"
                if args.skip_existing and priority_path.exists():
                    continue
                image = Image.open(image_path).convert("RGB")
                pending.append((cam_id, frame_id, image_path, image_id, priority_path, overlay_path, image))
                pil_images.append(image)
            if not pending:
                continue

            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            amp_context = (
                torch.autocast(device_type="cuda", enabled=True)
                if args.amp and device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), amp_context:
                outputs = model(**inputs)
            preds = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[item[-1].size[::-1] for item in pending]
            )
            for item, pred_tensor in zip(pending, preds):
                cam_id, frame_id, image_path, image_id, priority_path, overlay_path, image = item
                pred = pred_tensor.detach().cpu().numpy().astype(np.int32)
                priority, stats = postprocess_priority(pred, id2label)
                Image.fromarray(priority).save(priority_path)
                overlay(image, priority, args.overlay_alpha).save(overlay_path, quality=92)
                row = {
                    "image_id": image_id,
                    "cam_id": cam_id,
                    "frame_id": frame_id,
                    "image_path": str(image_path),
                    "priority_path": str(priority_path),
                    "overlay_path": str(overlay_path),
                    **stats,
                }
                report_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                processed += 1
                if processed == 1 or processed % 20 == 0:
                    print(json.dumps({"processed": processed, "image_id": image_id, "priority": row["priority_counts"]}, ensure_ascii=False))

    print(json.dumps({"processed": processed, "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
