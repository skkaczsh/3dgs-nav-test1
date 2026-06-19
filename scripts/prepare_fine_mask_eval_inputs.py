#!/usr/bin/env python3
"""Prepare a stable input package for fine-mask evaluation.

The manifest stores source paths from the remote workdir.  This script turns it
into a flat, reproducible input directory with symlinks or copies for images,
current masks, and review crops.  It deliberately does not run SAM2; it prepares
the exact sample set that SAM2/Python or SAM2/TensorRT candidates should use.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_path(raw: str | None, source_prefix: str, target_prefix: str) -> Path | None:
    if not raw:
        return None
    text = str(raw)
    if source_prefix:
        if not text.startswith(source_prefix):
            return Path(text)
        text = target_prefix + text[len(source_prefix) :]
    return Path(text)


def link_or_copy(src: Path | None, dst: Path, mode: str, overwrite: bool) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return True
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return True


def suffix_for(path: Path | None, fallback: str) -> str:
    if path is None:
        return fallback
    suffix = path.suffix
    return suffix if suffix else fallback


def prepare(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_dir
    images_dir = out / "images"
    masks_dir = out / "current_masks"
    crops_dir = out / "crops"
    meta_dir = out / "metadata"
    for path in [images_dir, masks_dir, crops_dir, meta_dir]:
        path.mkdir(parents=True, exist_ok=True)

    items = []
    missing: list[dict[str, str]] = []
    for item in manifest.get("items", []):
        sample_id = str(item["sample_id"])
        image_src = rewrite_path(item.get("image_path"), args.source_prefix, args.target_prefix)
        mask_src = rewrite_path(item.get("current_mask_path"), args.source_prefix, args.target_prefix)
        crop_src = rewrite_path(item.get("crop_path"), args.source_prefix, args.target_prefix)

        image_dst = images_dir / f"{sample_id}{suffix_for(image_src, '.jpg')}"
        mask_dst = masks_dir / f"{sample_id}{suffix_for(mask_src, '.png')}"
        crop_dst = crops_dir / f"{sample_id}{suffix_for(crop_src, '.jpg')}"

        image_ok = link_or_copy(image_src, image_dst, args.mode, args.overwrite)
        mask_ok = link_or_copy(mask_src, mask_dst, args.mode, args.overwrite)
        crop_ok = link_or_copy(crop_src, crop_dst, args.mode, args.overwrite)

        if not image_ok:
            missing.append({"sample_id": sample_id, "kind": "image", "path": str(image_src)})
        if not mask_ok:
            missing.append({"sample_id": sample_id, "kind": "current_mask", "path": str(mask_src)})
        if crop_src and not crop_ok:
            missing.append({"sample_id": sample_id, "kind": "crop", "path": str(crop_src)})

        item_out = {
            **item,
            "prepared_image": str(image_dst),
            "prepared_current_mask": str(mask_dst) if mask_ok else None,
            "prepared_crop": str(crop_dst) if crop_ok else None,
            "source_image_exists": image_ok,
            "source_mask_exists": mask_ok,
            "source_crop_exists": crop_ok,
        }
        items.append(item_out)
        (meta_dir / f"{sample_id}.json").write_text(json.dumps(item_out, ensure_ascii=False, indent=2), encoding="utf-8")

    image_list = out / "images.txt"
    image_list.write_text(
        "".join(str(row["prepared_image"]) + "\n" for row in items if row["source_image_exists"]),
        encoding="utf-8",
    )

    report = {
        "source_manifest": str(args.manifest),
        "output_dir": str(out),
        "mode": args.mode,
        "sample_count": len(items),
        "ready_images": sum(1 for row in items if row["source_image_exists"]),
        "ready_current_masks": sum(1 for row in items if row["source_mask_exists"]),
        "ready_crops": sum(1 for row in items if row["source_crop_exists"]),
        "missing_count": len(missing),
        "missing": missing[:100],
        "image_list": str(image_list),
        "items": items,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--source-prefix", default="")
    parser.add_argument("--target-prefix", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = prepare(read_manifest(args.manifest), args)
    report_path = args.report or (args.output_dir / "prepare_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "sample_count": report["sample_count"],
                "ready_images": report["ready_images"],
                "ready_current_masks": report["ready_current_masks"],
                "ready_crops": report["ready_crops"],
                "missing_count": report["missing_count"],
                "image_list": report["image_list"],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["missing_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
