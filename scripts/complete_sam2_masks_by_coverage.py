#!/usr/bin/env python3
"""Complete sparse SAM2 mask outputs by prompting uncovered valid regions.

The normal SAM2 AMG/TensorRT path can leave large valid image regions uncovered.
Those holes later become ``other`` in semantic completion. This script keeps the
existing masks, measures non-sky coverage, and, when coverage is below a target,
adds point-prompt SAM2 masks sampled from uncovered connected components.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_SAM2_ROOT = Path("/root/epfs/vlm_seg_project/segment-anything-2")
DEFAULT_CHECKPOINT = Path("/root/epfs/vlm_seg_project/weights/sam2_hiera_large.pt")
DEFAULT_CONFIG = "sam2_hiera_l.yaml"


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    h, w = [int(x) for x in rle["size"]]
    flat = np.empty(h * w, dtype=bool)
    idx = 0
    value = False
    for count in rle["counts"]:
        next_idx = idx + int(count)
        flat[idx:next_idx] = value
        idx = next_idx
        value = not value
    if idx < flat.size:
        flat[idx:] = False
    return flat.reshape(w, h).T


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    flat = mask.T.reshape(-1)
    counts: list[int] = []
    value = False
    run = 0
    for pixel in flat:
        pixel = bool(pixel)
        if pixel == value:
            run += 1
        else:
            counts.append(run)
            run = 1
            value = pixel
    counts.append(run)
    return {"size": [int(h), int(w)], "counts": counts}


def load_masks(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("masks", data if isinstance(data, list) else [])
    masks = []
    for item in items:
        seg = item.get("segmentation")
        if isinstance(seg, dict) and "counts" in seg and "size" in seg:
            masks.append(decode_rle(seg))
        else:
            masks.append(np.asarray(seg, dtype=bool))
    return data if isinstance(data, dict) else {}, list(items), masks


def union_mask(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for mask in masks:
        if mask.shape != shape:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(shape[::-1], Image.Resampling.NEAREST)) > 0
        out |= mask
    return out


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def load_valid_mask(path: Path | None, image: Image.Image, threshold: int, invalid_dark_threshold: int) -> np.ndarray:
    shape = (image.height, image.width)
    if not path:
        valid = np.ones(shape, dtype=bool)
    else:
        sky = np.asarray(Image.open(path).convert("L"))
        if sky.shape != shape:
            sky = np.asarray(Image.fromarray(sky).resize(shape[::-1], Image.Resampling.NEAREST))
        valid = sky < threshold
    if invalid_dark_threshold >= 0:
        rgb = np.asarray(image.convert("RGB"))
        valid &= ~(rgb.max(axis=2) <= invalid_dark_threshold)
    return valid


def find_sky_mask(sky_dir: Path | None, image_id: str) -> Path | None:
    if not sky_dir:
        return None
    stem = image_id
    parts = stem.split("_")
    if len(parts) == 2 and parts[0].startswith("cam"):
        cam = parts[0]
        frame = int(parts[1])
        candidates = [
            sky_dir / f"{cam}_{frame:07d}_sky.png",
            sky_dir / f"{cam}_{frame:06d}_sky.png",
            sky_dir / f"{cam}_{frame:05d}_sky.png",
            sky_dir / f"{cam}_{frame:04d}_sky.png",
            sky_dir / cam / f"frame_{frame:06d}.png",
            sky_dir / cam / f"frame_{frame:04d}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    direct = sky_dir / f"{image_id}_sky.png"
    return direct if direct.exists() else None


def connected_component_seed_points(mask: np.ndarray, min_area: int, max_points: int, area_per_point: int) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - production env has OpenCV.
        raise RuntimeError("OpenCV is required for connected-component seed selection") from exc

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        components.append((area, label))
    components.sort(reverse=True)

    seeds: list[dict[str, Any]] = []
    for area, label in components:
        if len(seeds) >= max_points:
            break
        comp = labels == label
        count = max(1, min(6, int(np.ceil(area / max(area_per_point, 1)))))
        ys, xs = np.where(comp)
        if count == 1:
            dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(dist.argmax()), dist.shape)
            seeds.append({"point": [int(x), int(y)], "component_area": area})
            continue
        order = np.argsort(xs + ys * mask.shape[1])
        splits = np.array_split(order, count)
        for split in splits:
            if len(seeds) >= max_points or len(split) == 0:
                break
            sx = xs[split]
            sy = ys[split]
            seeds.append({"point": [int(np.median(sx)), int(np.median(sy))], "component_area": area})
    return seeds[:max_points]


def load_predictor(args: argparse.Namespace):
    sys.path.insert(0, str(args.sam2_root))
    sys.path.insert(0, str(args.sam2_root.parent))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(str(args.config), str(args.checkpoint), device=args.device)
    predictor = SAM2ImagePredictor(model)
    return predictor, torch


def predict_one_seed(predictor: Any, point: list[int], valid: np.ndarray) -> tuple[np.ndarray, float]:
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray([point], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int32),
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    mask = np.asarray(masks[best_idx], dtype=bool) & valid
    return mask, float(scores[best_idx])


def render_mask_previews(image: Image.Image, masks: list[np.ndarray], output_png: Path, numbered_png: Path) -> None:
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
    ]
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_num = ImageDraw.Draw(base)
    font = ImageFont.load_default()
    for i, mask in enumerate(masks, 1):
        color = palette[(i - 1) % len(palette)]
        alpha = Image.fromarray((mask.astype(np.uint8) * 96), mode="L")
        layer = Image.new("RGBA", base.size, (*color, 0))
        layer.putalpha(alpha)
        overlay.alpha_composite(layer)
        ys, xs = np.where(mask)
        if len(xs):
            draw_num.text((int(np.median(xs)), int(np.median(ys))), str(i), fill=(255, 255, 255, 255), font=font)
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_png)
    Image.alpha_composite(base, overlay).convert("RGB").save(numbered_png)


def process_image(image_path: Path, args: argparse.Namespace, predictor: Any | None = None) -> dict[str, Any]:
    image_id = image_path.stem
    input_json = args.input_mask_dir / f"{image_id}_sam_masks.json"
    if not input_json.exists():
        return {"image_id": image_id, "status": "missing_input_mask"}
    image = Image.open(image_path).convert("RGB")
    shape = (image.height, image.width)
    source_data, items, masks = load_masks(input_json)
    valid = load_valid_mask(
        find_sky_mask(args.sky_mask_dir, image_id),
        image,
        args.sky_threshold,
        args.invalid_dark_threshold,
    )
    union = union_mask(masks, shape) & valid
    valid_area = int(valid.sum())
    before_coverage = float(union.sum() / max(valid_area, 1))
    rgb = np.asarray(image)
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    median_luma = float(np.median(luma[valid])) if valid_area else 0.0
    added: list[dict[str, Any]] = []

    low_luma_skip = median_luma < args.min_median_luma_for_completion
    if before_coverage < args.target_coverage and not low_luma_skip:
        if predictor is None:
            raise RuntimeError("predictor is required when coverage completion is needed")
        predictor.set_image(np.asarray(image))
        for seed in connected_component_seed_points(
            valid & ~union,
            min_area=args.min_uncovered_component_area,
            max_points=args.max_prompt_points,
            area_per_point=args.uncovered_area_per_point,
        ):
            mask, score = predict_one_seed(predictor, seed["point"], valid)
            overlap = int((mask & union).sum())
            area = int(mask.sum())
            new_area = int((mask & ~union).sum())
            if area < args.min_mask_area or new_area < args.min_new_area:
                continue
            if overlap / max(area, 1) > args.max_overlap_ratio:
                continue
            union |= mask
            masks.append(mask)
            added_item = {
                "segmentation": encode_rle(mask),
                "bbox": bbox_from_mask(mask),
                "area": area,
                "predicted_iou": score,
                "stability_score": 1.0,
                "point_coords": [seed["point"]],
                "crop_box": [0, 0, image.width, image.height],
                "source": "coverage_completion_point_prompt",
                "new_area": new_area,
                "component_area": int(seed["component_area"]),
            }
            items.append(added_item)
            added.append({k: added_item[k] for k in ["bbox", "area", "predicted_iou", "point_coords", "new_area", "component_area"]})
            if float(union.sum() / max(valid_area, 1)) >= args.target_coverage:
                break

    after_coverage = float(union.sum() / max(valid_area, 1))
    output_json = args.output_dir / f"{image_id}_sam_masks.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    out = dict(source_data)
    out.update(
        {
            "image_name": image_id,
            "original_image": str(image_path),
            "num_masks": len(items),
            "masked_ratio": after_coverage,
            "valid_area": valid_area,
            "coverage_completion": {
                "enabled": True,
                "before_coverage": before_coverage,
                "after_coverage": after_coverage,
                "target_coverage": args.target_coverage,
                "invalid_dark_threshold": args.invalid_dark_threshold,
                "median_luma": median_luma,
                "min_median_luma_for_completion": args.min_median_luma_for_completion,
                "low_luma_skip_completion": low_luma_skip,
                "added_masks": len(added),
                "added": added,
            },
            "masks": items,
        }
    )
    output_json.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    render_mask_previews(
        image,
        masks,
        args.output_dir / f"{image_id}_sam_masks.png",
        args.output_dir / f"{image_id}_numbered.png",
    )
    (args.output_dir / f"{image_id}_sam_done.flag").write_text("ok\n", encoding="utf-8")
    return {
        "image_id": image_id,
        "status": "ok",
        "before_coverage": before_coverage,
        "after_coverage": after_coverage,
        "median_luma": median_luma,
        "low_luma_skip_completion": low_luma_skip,
        "input_masks": len(masks) - len(added),
        "output_masks": len(masks),
        "added_masks": len(added),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Glob of input images.")
    parser.add_argument("--input-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sky-mask-dir", type=Path, default=None)
    parser.add_argument("--sky-threshold", type=int, default=128)
    parser.add_argument("--invalid-dark-threshold", type=int, default=12, help="Exclude undistortion black-border pixels from coverage; set -1 to disable.")
    parser.add_argument("--min-median-luma-for-completion", type=float, default=25.0, help="Skip point-prompt completion on severely underexposed valid regions.")
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--min-uncovered-component-area", type=int, default=8000)
    parser.add_argument("--uncovered-area-per-point", type=int, default=120000)
    parser.add_argument("--max-prompt-points", type=int, default=16)
    parser.add_argument("--min-mask-area", type=int, default=500)
    parser.add_argument("--min-new-area", type=int, default=2000)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.images.startswith("@"):
        list_path = Path(args.images[1:])
        images = [Path(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        images = [Path(p) for p in sorted(glob.glob(args.images))]
    if not images:
        raise SystemExit(f"no images matched {args.images}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    need_predictor = False
    for image_path in images:
        image_id = image_path.stem
        out_json = args.output_dir / f"{image_id}_sam_masks.json"
        if out_json.exists() and not args.overwrite:
            continue
        input_json = args.input_mask_dir / f"{image_id}_sam_masks.json"
        if not input_json.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        _, _, masks = load_masks(input_json)
        valid = load_valid_mask(
            find_sky_mask(args.sky_mask_dir, image_id),
            image,
            args.sky_threshold,
            args.invalid_dark_threshold,
        )
        coverage = float((union_mask(masks, (image.height, image.width)) & valid).sum() / max(int(valid.sum()), 1))
        rgb = np.asarray(image)
        luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        median_luma = float(np.median(luma[valid])) if int(valid.sum()) else 0.0
        if coverage < args.target_coverage and median_luma >= args.min_median_luma_for_completion:
            need_predictor = True
            break

    predictor = None
    if need_predictor:
        predictor, torch = load_predictor(args)
    rows = []
    for image_path in images:
        image_id = image_path.stem
        out_json = args.output_dir / f"{image_id}_sam_masks.json"
        if out_json.exists() and not args.overwrite:
            rows.append({"image_id": image_id, "status": "skipped_existing"})
            continue
        with (torch.inference_mode() if predictor is not None else nullcontext()):
            rows.append(process_image(image_path, args, predictor=predictor))
        print(json.dumps(rows[-1], ensure_ascii=False))

    ok = [row for row in rows if row.get("status") == "ok"]
    report = {
        "images": len(rows),
        "ok_images": len(ok),
        "target_coverage": args.target_coverage,
        "mean_before_coverage": float(np.mean([row["before_coverage"] for row in ok])) if ok else 0.0,
        "mean_after_coverage": float(np.mean([row["after_coverage"] for row in ok])) if ok else 0.0,
        "added_masks": int(sum(int(row.get("added_masks", 0)) for row in ok)),
        "rows": rows,
    }
    report_path = args.report or (args.output_dir / "coverage_completion_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["images", "ok_images", "mean_before_coverage", "mean_after_coverage", "added_masks"]}, ensure_ascii=False, indent=2))


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args: Any) -> None:
        return None


if __name__ == "__main__":
    main()
