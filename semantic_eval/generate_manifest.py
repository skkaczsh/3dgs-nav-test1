#!/usr/bin/env python3
"""Generate a fixed image manifest for semantic evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def image_id_from_path(path: Path) -> str:
    return path.stem


def sky_image_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_sky_vis"):
        return stem[:-8]
    if stem.endswith("_sky"):
        return stem[:-4]
    return stem


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return items
    if count == 1:
        return [items[0]]
    picks: list[Path] = []
    last_index = len(items) - 1
    for i in range(count):
        idx = round(i * last_index / (count - 1))
        picks.append(items[idx])
    # round() can collide for tiny lists; preserve order and top up if needed.
    seen = {p for p in picks}
    if len(picks) != len(seen):
        picks = []
        seen = set()
    for p in items:
        if len(picks) >= count:
            break
        if p not in seen:
            picks.append(p)
            seen.add(p)
    return picks


def build_manifest(images_dir: Path, sky_masks_dir: Path, limit: int) -> dict:
    images = sorted(images_dir.glob("*.png"))
    image_by_id = {image_id_from_path(p): p for p in images}

    sky_masks: dict[str, Path] = {}
    if sky_masks_dir.exists():
        for p in sorted(sky_masks_dir.glob("*_sky.png")):
            sky_masks[sky_image_id(p)] = p

    items: list[dict] = []
    used: set[str] = set()

    for image_id in sorted(sky_masks):
        if image_id not in image_by_id:
            continue
        items.append({
            "image_id": image_id,
            "image_path": str(image_by_id[image_id]),
            "sky_mask_path": str(sky_masks[image_id]),
            "reason": "existing_sky_mask",
        })
        used.add(image_id)
        if len(items) >= limit:
            break

    remaining = [p for p in images if image_id_from_path(p) not in used]
    for p in evenly_spaced(remaining, limit - len(items)):
        image_id = image_id_from_path(p)
        items.append({
            "image_id": image_id,
            "image_path": str(p),
            "sky_mask_path": str(sky_masks[image_id]) if image_id in sky_masks else None,
            "reason": "stratified_fill",
        })

    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "images_dir": str(images_dir),
        "sky_masks_dir": str(sky_masks_dir),
        "limit": limit,
        "total_images_available": len(images),
        "total_sky_masks_available": len(sky_masks),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semantic eval manifest")
    parser.add_argument("--images-dir", type=Path,
                        default=Path("/root/epfs/manifold_3dgs_project/processed/images"))
    parser.add_argument("--sky-masks-dir", type=Path,
                        default=Path("/root/epfs/manifold_3dgs_project/processed/sky_masks"))
    parser.add_argument("--output", type=Path,
                        default=Path("/root/epfs/manifold_3dgs_project/processed/semantic_eval_20260605/manifest.json"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    manifest = build_manifest(args.images_dir, args.sky_masks_dir, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Wrote {len(manifest['items'])} items to {args.output}")


if __name__ == "__main__":
    main()

