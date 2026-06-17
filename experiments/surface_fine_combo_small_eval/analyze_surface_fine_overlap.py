#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


TARGET_CLASSES = ["background", "floor_ground", "wall", "ceiling", "building", "sky"]
PALETTE = {
    "background": (0, 0, 0),
    "floor_ground": (230, 170, 40),
    "wall": (70, 130, 180),
    "ceiling": (180, 120, 220),
    "building": (220, 80, 80),
    "sky": (100, 180, 255),
}


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def box_mask(box_xyxy: list[float], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = box_xyxy
    xi1 = max(0, min(w - 1, int(np.floor(x1))))
    yi1 = max(0, min(h - 1, int(np.floor(y1))))
    xi2 = max(0, min(w, int(np.ceil(x2))))
    yi2 = max(0, min(h, int(np.ceil(y2))))
    mask = np.zeros((h, w), dtype=bool)
    if xi2 > xi1 and yi2 > yi1:
        mask[yi1:yi2, xi1:xi2] = True
    return mask


def load_surface_map(path: Path) -> np.ndarray:
    return np.load(path).astype(np.uint8)


def class_name(idx: int) -> str:
    return TARGET_CLASSES[int(idx)]


def overlap_stats(mask: np.ndarray, mapped: np.ndarray) -> dict[str, float]:
    masked = mapped[mask]
    if masked.size == 0:
        return {}
    counts = Counter(int(x) for x in masked.tolist())
    total = float(masked.size)
    return {class_name(k): float(v / total) for k, v in counts.items()}


def render_panel(
    image: Image.Image,
    detector_boxes: Image.Image,
    sam2_masks: Image.Image,
    surface_overlay: Image.Image,
    lines: list[str],
) -> Image.Image:
    width, height = image.size
    header_h = 170
    panel = Image.new("RGB", (width * 4, height + header_h), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    tiles = [
        ("original", image),
        ("surface_overlay", surface_overlay),
        ("detector_boxes", detector_boxes),
        ("sam2_masks", sam2_masks),
    ]
    for idx, (title, tile) in enumerate(tiles):
        x = idx * width
        panel.paste(tile.convert("RGB"), (x, 30))
        draw.text((x + 8, 8), title, fill=(255, 255, 255))
    y = height + 40
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255))
        y += 18
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-output-dir", type=Path, required=True)
    parser.add_argument("--fine-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "visualizations"
    visual_dir.mkdir(exist_ok=True)

    surface_predictions = args.surface_output_dir / "predictions"
    fine_per_sample = args.fine_output_dir / "per_sample"

    rows: list[dict[str, object]] = []
    aggregate_focus_class: dict[str, Counter[str]] = defaultdict(Counter)

    for result_path in sorted(fine_per_sample.glob("*/result.json")):
        sample_id = result_path.parent.name
        result = json.loads(result_path.read_text(encoding="utf-8"))
        mapped_path = surface_predictions / "mask2former_ade20k" / sample_id / "mapped_semantic.npy"
        overlay_path = surface_predictions / "mask2former_ade20k" / sample_id / "overlay.png"
        if not mapped_path.exists():
            continue
        mapped = load_surface_map(mapped_path)
        original = Image.open(result_path.parent / "original.png").convert("RGB")
        detector_boxes = Image.open(result_path.parent / "detector_boxes.png").convert("RGB")
        sam2_masks = Image.open(result_path.parent / "sam2_masks.png").convert("RGB")
        surface_overlay = Image.open(overlay_path).convert("RGB")

        det_rows = []
        lines = [sample_id]
        for det in result.get("detections", []):
            if "mask_path" in det:
                mask = load_mask(Path(det["mask_path"]))
                region_type = "mask"
            else:
                mask = box_mask(det["box_xyxy"], mapped.shape)
                region_type = "box"
            overlaps = overlap_stats(mask, mapped)
            dominant_class = max(overlaps.items(), key=lambda kv: kv[1])[0] if overlaps else "none"
            aggregate_focus_class[str(det["focus"])][dominant_class] += 1
            det_row = {
                "focus": det["focus"],
                "phrase": det["phrase"],
                "region_type": region_type,
                "mask_area": det["mask_area"],
                "dominant_surface_class": dominant_class,
                "surface_overlap": overlaps,
            }
            det_rows.append(det_row)
            compact = ", ".join(
                f"{k}:{v:.2f}" for k, v in sorted(overlaps.items(), key=lambda kv: kv[1], reverse=True)[:3]
            )
            lines.append(f"{det['focus']} | {det['phrase']} | {compact}")

        panel = render_panel(original, detector_boxes, sam2_masks, surface_overlay, lines)
        panel.save(visual_dir / f"{sample_id}_panel.png")
        rows.append({"sample_id": sample_id, "detections": det_rows})

    summary = {
        "surface_model": "mask2former_ade20k",
        "sample_count": len(rows),
        "samples": rows,
        "focus_dominant_surface_counts": {
            focus: dict(counter) for focus, counter in aggregate_focus_class.items()
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md = [
        "# Surface Fine Combo Small Eval",
        "",
        f"- surface_model: `mask2former_ade20k`",
        f"- sample_count: `{len(rows)}`",
        "",
        "## Focus vs dominant surface class",
        "",
        "| focus | dominant surface counts |",
        "| --- | --- |",
    ]
    for focus, counter in sorted(aggregate_focus_class.items()):
        payload = ", ".join(f"{k}={v}" for k, v in counter.most_common())
        md.append(f"| {focus} | {payload} |")
    (args.output_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
