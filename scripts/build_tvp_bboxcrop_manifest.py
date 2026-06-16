#!/usr/bin/env python3
"""Build a TVP bbox-crop sanity manifest from an existing TVP manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--pad", type=int, default=48)
    parser.add_argument("--image-field", default="image_path")
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])[: args.limit]

    args.output_image_dir.mkdir(parents=True, exist_ok=True)
    out_samples = []

    for sample in samples:
        bbox = sample.get("bbox")
        image_path = sample.get(args.image_field) or ""
        if not bbox or not image_path:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        x1 = clamp(x1 - args.pad, 0, width)
        y1 = clamp(y1 - args.pad, 0, height)
        x2 = clamp(x2 + args.pad, 0, width)
        y2 = clamp(y2 + args.pad, 0, height)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        crop_path = args.output_image_dir / f"{sample['id']}.png"
        crop.save(crop_path)

        updated = dict(sample)
        updated["image_path"] = str(crop_path)
        updated["crop_bbox_xyxy"] = [x1, y1, x2, y2]
        updated["crop_pad"] = args.pad
        updated["crop_source_field"] = args.image_field
        out_samples.append(updated)

    output = {
        "schema": "tvp_bboxcrop_manifest_v1",
        "source_manifest": str(args.input_manifest),
        "sample_count": len(out_samples),
        "pad": args.pad,
        "image_field": args.image_field,
        "samples": out_samples,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_manifest)
    print(json.dumps({"sample_count": len(out_samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
