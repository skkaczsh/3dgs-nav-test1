#!/usr/bin/env python3
"""Build single-frame mask QA matrices from first-touch depth guidance.

This script is intentionally image-local.  It compares masks produced by:

- depth connectivity only
- depth + color-gradient gated BFS
- PCA spectral-vector gated BFS
- optional SAM2 masks split by the depth/color components

The output is a set of triptychs: source image, mask overlay, labeled map.  It is
for QA and algorithm selection; production code should reuse the selected gates
inside the 3D patch pipeline rather than consuming these PNG labels directly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scripts.sam_rle import decode_rle


PALETTE = np.asarray(
    [
        (0, 0, 0),
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
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (0, 0, 128),
    ],
    dtype=np.uint8,
)


def load_sam_masks(path: Path | None) -> list[np.ndarray]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("masks", data if isinstance(data, list) else [])
    masks = []
    for item in items:
        seg = item.get("segmentation", item)
        if isinstance(seg, dict) and "counts" in seg and "size" in seg:
            masks.append(decode_rle(seg))
        elif isinstance(seg, list):
            masks.append(np.asarray(seg, dtype=bool))
    return masks


def image_id_from_npz(path: Path) -> str:
    name = path.name
    suffix = "_geometry.npz"
    if not name.endswith(suffix):
        raise ValueError(f"unexpected geometry npz name: {path}")
    return name[: -len(suffix)]


def parse_image_id(image_id: str) -> tuple[int, int]:
    cam, frame = image_id.split("_", 1)
    return int(cam.replace("cam", "")), int(frame)


def resize_array_nearest(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(arr)
    return np.asarray(image.resize(size, Image.Resampling.NEAREST))


def resize_array_linear(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(arr)
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR))


def resize_float_nearest(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST), dtype=np.float32)


def scale_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max(width, height) <= max_side:
        return width, height
    scale = float(max_side) / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def load_source_image(frame_dir: Path | None, guidance_dir: Path, image_id: str, shape: tuple[int, int]) -> np.ndarray:
    cam_id, frame_id = parse_image_id(image_id)
    candidates = []
    if frame_dir is not None:
        candidates.extend(
            [
                frame_dir / f"cam{cam_id}" / f"frame_{frame_id:06d}.jpg",
                frame_dir / f"cam{cam_id}" / f"frame_{frame_id:06d}.png",
            ]
        )
    candidates.extend(
        [
            guidance_dir / "rendered_rgb" / f"{image_id}_rendered_rgb.jpg",
            guidance_dir / "depth_viz" / f"{image_id}_depth.jpg",
        ]
    )
    for path in candidates:
        if path.exists():
            rgb = np.asarray(Image.open(path).convert("RGB"))
            if rgb.shape[:2] != shape:
                rgb = resize_array_linear(rgb, (shape[1], shape[0]))
        if rgb.max() > 0:
            return rgb
    triptych_path = guidance_dir.parent / "triptych" / f"{image_id}_triptych.jpg"
    if triptych_path.exists():
        sheet = np.asarray(Image.open(triptych_path).convert("RGB"))
        crop_w = sheet.shape[1] // 3
        rgb = sheet[:, :crop_w]
        if rgb.shape[:2] != shape:
            rgb = resize_array_linear(rgb, (shape[1], shape[0]))
        if rgb.max() > 0:
            return rgb
    return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)


def normalize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(depth.shape, dtype=np.float32)
    values = depth[valid]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return out
    lo, hi = np.percentile(values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(values.max())
        lo = float(values.min())
    if hi <= lo:
        return out
    out[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def color_gradient_edges(rgb: np.ndarray, threshold: float) -> np.ndarray:
    rgbf = rgb.astype(np.float32)
    dx = np.zeros(rgb.shape[:2], dtype=np.float32)
    dy = np.zeros(rgb.shape[:2], dtype=np.float32)
    dx[:, 1:] = np.linalg.norm(rgbf[:, 1:] - rgbf[:, :-1], axis=2)
    dy[1:, :] = np.linalg.norm(rgbf[1:, :] - rgbf[:-1, :], axis=2)
    mag = np.maximum(dx, dy)
    return mag >= float(threshold)


def spectral_scalar(depth01: np.ndarray, rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w = depth01.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    features = np.stack(
        [
            depth01,
            rgb[:, :, 0].astype(np.float32) / 255.0,
            rgb[:, :, 1].astype(np.float32) / 255.0,
            rgb[:, :, 2].astype(np.float32) / 255.0,
            xx / max(w - 1, 1),
            yy / max(h - 1, 1),
        ],
        axis=2,
    )
    sample = features[valid]
    out = np.zeros((h, w), dtype=np.float32)
    if sample.shape[0] < 8:
        return out
    if sample.shape[0] > 20000:
        step = max(1, sample.shape[0] // 20000)
        sample_fit = sample[::step]
    else:
        sample_fit = sample
    mean = sample_fit.mean(axis=0)
    centered = sample_fit - mean
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    projected = (features.reshape(-1, features.shape[2]) - mean) @ axis
    projected = projected.reshape(h, w).astype(np.float32)
    pv = projected[valid]
    lo, hi = np.percentile(pv, [1, 99])
    if hi > lo:
        out[valid] = np.clip((projected[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def connected_components_gated(
    valid: np.ndarray,
    features: np.ndarray | None,
    blocked: np.ndarray | None,
    feature_threshold: float,
    min_area: int,
) -> np.ndarray:
    h, w = valid.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    for y in range(h):
        for x in range(w):
            if not valid[y, x] or labels[y, x] != 0:
                continue
            current += 1
            labels[y, x] = current
            q: deque[tuple[int, int]] = deque([(y, x)])
            area = 0
            while q:
                cy, cx = q.popleft()
                area += 1
                base = features[cy, cx] if features is not None else None
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if labels[ny, nx] != 0 or not valid[ny, nx]:
                        continue
                    if blocked is not None and (blocked[cy, cx] or blocked[ny, nx]):
                        continue
                    if features is not None:
                        dist = float(np.linalg.norm(base - features[ny, nx]))
                        if dist > feature_threshold:
                            continue
                    labels[ny, nx] = current
                    q.append((ny, nx))
            if area < min_area:
                labels[labels == current] = 0
                current -= 1
    return relabel_by_area(labels)


def relabel_by_area(labels: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    order = [int(i) for i in ids[np.argsort(-counts)]]
    out = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(order, 1):
        out[labels == old_id] = new_id
    return out


def labels_from_sam_split(
    masks: list[np.ndarray],
    base_labels: np.ndarray,
    valid: np.ndarray,
    min_area: int,
    target_size: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(base_labels.shape, dtype=np.int32)
    current = 0
    for mask in masks:
        if mask.shape != valid.shape:
            mask = resize_array_nearest(mask.astype(np.uint8) * 255, target_size) > 0
        candidate = mask & valid
        for component_id in np.unique(base_labels[candidate]):
            if component_id <= 0:
                continue
            part = candidate & (base_labels == component_id)
            if int(part.sum()) < min_area:
                continue
            current += 1
            out[part] = current
    return relabel_by_area(out)


def label_to_rgb(labels: np.ndarray) -> np.ndarray:
    out = PALETTE[labels % len(PALETTE)].copy()
    out[labels == 0] = 0
    return out


def boundary(labels: np.ndarray) -> np.ndarray:
    b = np.zeros(labels.shape, dtype=bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    return b & (labels > 0)


def overlay_labels(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    color = label_to_rgb(labels).astype(np.float32)
    base = rgb.astype(np.float32)
    mask = labels > 0
    out = base.copy()
    out[mask] = base[mask] * 0.58 + color[mask] * 0.42
    b = boundary(labels)
    out[b] = (255, 255, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_title(img: Image.Image, title: str) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 26), fill=(0, 0, 0))
    draw.text((8, 7), title, fill=(255, 255, 255))


def save_triptych(source: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    left = Image.fromarray(source)
    mid = Image.fromarray(overlay_labels(source, labels))
    right = Image.fromarray(label_to_rgb(labels))
    draw_title(left, f"{title}: source")
    draw_title(mid, "mask overlay")
    draw_title(right, "labeled components")
    sheet = Image.new("RGB", (left.width * 3, left.height), (0, 0, 0))
    sheet.paste(left, (0, 0))
    sheet.paste(mid, (left.width, 0))
    sheet.paste(right, (left.width * 2, 0))
    sheet.save(path)


def component_summary(labels: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    return {
        "component_count": int(len(ids)),
        "covered_pixels": int((labels > 0).sum()),
        "valid_pixels": int(valid.sum()),
        "coverage": float((labels > 0).sum() / max(int(valid.sum()), 1)),
        "top_areas": [int(x) for x in sorted(counts.tolist(), reverse=True)[:10]],
    }


def process_one(
    npz_path: Path,
    guidance_dir: Path,
    frame_dir: Path | None,
    sam_mask_dir: Path | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_id = image_id_from_npz(npz_path)
    data = np.load(npz_path)
    depth = np.asarray(data["depth"], dtype=np.float32)
    valid = (np.asarray(data["valid"]) > 0) & np.isfinite(depth) & (depth > 0)
    h, w = depth.shape
    out_w, out_h = scale_size(w, h, args.max_side)
    target_size = (out_w, out_h)
    if (out_w, out_h) != (w, h):
        depth = resize_float_nearest(depth, target_size)
        valid = resize_array_nearest(valid.astype(np.uint8) * 255, target_size) > 0
    source = load_source_image(frame_dir, guidance_dir, image_id, depth.shape)
    if source.shape[1] != out_w or source.shape[0] != out_h:
        source = resize_array_linear(source, target_size)
    depth_edge = np.asarray(data.get("edge", np.zeros((h, w), dtype=np.uint8)))
    if depth_edge.shape != valid.shape:
        depth_edge = resize_array_nearest(depth_edge, target_size)
    color_edge = color_gradient_edges(source, args.color_edge_threshold)
    saved_color_edge = np.asarray(data.get("color_edge", np.zeros((h, w), dtype=np.uint8)))
    if saved_color_edge.shape != valid.shape:
        saved_color_edge = resize_array_nearest(saved_color_edge, target_size)
    color_edge |= saved_color_edge > 0
    depth01 = normalize_depth(depth, valid)
    rgb01 = source.astype(np.float32) / 255.0
    spectral01 = spectral_scalar(depth01, source, valid)

    depth_features = depth01[:, :, None]
    color_features = np.dstack([depth01 * args.depth_feature_weight, rgb01 * args.color_feature_weight])
    spectral_features = np.dstack([depth01 * args.depth_feature_weight, spectral01 * args.spectral_feature_weight])
    depth_block = depth_edge > 0
    depth_color_block = depth_block | color_edge

    variants: dict[str, np.ndarray] = {
        "depth_connectivity": connected_components_gated(
            valid, depth_features, depth_block, args.depth_threshold, args.min_area
        ),
        "depth_color_bfs": connected_components_gated(
            valid, color_features, depth_color_block, args.color_feature_threshold, args.min_area
        ),
        "spectral_bfs": connected_components_gated(
            valid, spectral_features, depth_color_block, args.spectral_feature_threshold, args.min_area
        ),
    }

    sam_masks: list[np.ndarray] = []
    if sam_mask_dir is not None:
        for candidate in (
            sam_mask_dir / f"{image_id}_masks.json",
            sam_mask_dir / f"{image_id}.json",
            sam_mask_dir / image_id / "masks.json",
        ):
            if candidate.exists():
                sam_masks = load_sam_masks(candidate)
                break
    if sam_masks:
        variants["sam2_depth_color_split"] = labels_from_sam_split(
            sam_masks, variants["depth_color_bfs"], valid, args.min_area, target_size
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_summary: dict[str, Any] = {"image_id": image_id, "variants": {}}
    for name, labels in variants.items():
        save_triptych(source, labels, output_dir / f"{image_id}_{name}_triptych.jpg", name)
        sample_summary["variants"][name] = component_summary(labels, valid)

    return sample_summary


def build_index(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Mask Matrix QA</title>",
        "<style>body{font-family:system-ui;background:#111;color:#eee} img{max-width:100%;display:block;margin:8px 0 24px} code{color:#9cf}</style>",
        "</head><body><h1>Single-frame Mask Matrix QA</h1>",
    ]
    for row in rows:
        html.append(f"<h2>{row['image_id']}</h2>")
        for name, summary in row["variants"].items():
            rel = f"{row['image_id']}_{name}_triptych.jpg"
            html.append(f"<h3><code>{name}</code> coverage={summary['coverage']:.3f} components={summary['component_count']}</h3>")
            html.append(f"<img src='{rel}' />")
    html.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(html), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance-dir", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--sam-mask-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--depth-threshold", type=float, default=0.035)
    parser.add_argument("--color-edge-threshold", type=float, default=42.0)
    parser.add_argument("--color-feature-threshold", type=float, default=0.20)
    parser.add_argument("--spectral-feature-threshold", type=float, default=0.11)
    parser.add_argument("--depth-feature-weight", type=float, default=1.0)
    parser.add_argument("--color-feature-weight", type=float, default=0.65)
    parser.add_argument("--spectral-feature-weight", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_dir = args.guidance_dir / "maps"
    if args.image_id:
        npz_paths = [map_dir / f"{image_id}_geometry.npz" for image_id in args.image_id]
    else:
        npz_paths = sorted(map_dir.glob("*_geometry.npz"))[: args.limit]
    rows = []
    for npz_path in npz_paths:
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        rows.append(process_one(npz_path, args.guidance_dir, args.frame_dir, args.sam_mask_dir, args.output_dir, args))
    (args.output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    build_index(args.output_dir, rows)
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
