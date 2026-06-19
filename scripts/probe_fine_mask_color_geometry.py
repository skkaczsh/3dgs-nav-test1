#!/usr/bin/env python3
"""Probe color/shape consistency for risky fine-object masks.

This is the first stage of the proposed vision-depth split guard.  It does not
run SAM2.  It measures whether current fine masks look like thin, coherent
objects in the undistorted image or like broad mixed regions that probably
swallow adjacent wall/floor/stair surfaces.

Depth support can be added later by extending the per-sample metrics with a
projected sparse depth map.  The JSON schema deliberately keeps sample ids,
object ids, and target ids from the fine-mask manifest so the report can be
joined with frame-local target QA.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PRIORITY_LABELS = {
    "residual": 0,
    "ground": 1,
    "floor": 1,
    "wall": 2,
    "grass": 3,
    "car": 4,
    "railing": 5,
    "sky": 6,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def label_id(label: str) -> int:
    text = str(label or "").strip().lower()
    if text.isdigit():
        return int(text)
    return PRIORITY_LABELS.get(text, 0)


def clip_bbox(xyxy: list[Any], width: int, height: int, pad: int = 0) -> tuple[int, int, int, int]:
    if len(xyxy) != 4:
        return 0, 0, width - 1, height - 1
    x0, y0, x1, y1 = [int(round(float(v))) for v in xyxy]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width - 1, x1 + pad)
    y1 = min(height - 1, y1 + pad)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def largest_component_stats(mask: np.ndarray) -> dict[str, Any]:
    if mask.sum() == 0:
        return {"components": 0, "largest_area": 0, "largest_ratio": 0.0, "min_rect_aspect": 0.0}
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    if not areas:
        return {"components": 0, "largest_area": 0, "largest_ratio": 0.0, "min_rect_aspect": 0.0}
    largest_label = 1 + int(np.argmax(areas))
    largest_area = max(areas)
    pts_yx = np.column_stack(np.where(labels == largest_label))
    if len(pts_yx) >= 5:
        pts_xy = pts_yx[:, ::-1].astype(np.float32)
        (_cx, _cy), (w, h), _angle = cv2.minAreaRect(pts_xy)
        short = max(min(float(w), float(h)), 1e-6)
        aspect = max(float(w), float(h)) / short
    else:
        aspect = 0.0
    return {
        "components": int(n - 1),
        "largest_area": int(largest_area),
        "largest_ratio": float(largest_area / max(int(mask.sum()), 1)),
        "min_rect_aspect": float(aspect),
    }


def boundary_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
    return dilated & ~eroded


def lab_stats(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        return {"lab_std_mean": 0.0, "lab_l_std": 0.0, "boundary_lab_contrast": 0.0, "boundary_sobel_mean": 0.0}
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    masked = lab[mask]
    lab_std = masked.std(axis=0)
    boundary = boundary_mask(mask, radius=2)
    inside_edge = boundary & mask
    outside_edge = boundary & ~mask
    if inside_edge.sum() and outside_edge.sum():
        contrast = float(np.linalg.norm(lab[inside_edge].mean(axis=0) - lab[outside_edge].mean(axis=0)))
    else:
        contrast = 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sx * sx + sy * sy)
    boundary_sobel = float(sobel[boundary].mean()) if boundary.sum() else 0.0
    return {
        "lab_std_mean": float(lab_std.mean()),
        "lab_l_std": float(lab_std[0]),
        "boundary_lab_contrast": contrast,
        "boundary_sobel_mean": boundary_sobel,
    }


def risk_flags(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    flags = []
    if metrics["bbox_area_ratio"] >= args.large_bbox_ratio:
        flags.append("large_bbox")
    if metrics["mask_fill_ratio"] >= args.high_fill_ratio:
        flags.append("high_fill_ratio")
    if metrics["component_count"] > args.max_components:
        flags.append("fragmented_mask")
    if metrics["min_rect_aspect"] < args.min_thin_aspect:
        flags.append("not_thin")
    if metrics["boundary_lab_contrast"] < args.min_boundary_lab_contrast:
        flags.append("weak_color_boundary")
    if metrics["lab_std_mean"] > args.max_lab_std_mean:
        flags.append("high_internal_color_variance")
    return flags


def crop_or_pad(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[y0 : y1 + 1, x0 : x1 + 1].copy()


def make_preview(image_bgr: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], output: Path, title: str) -> None:
    x0, y0, x1, y1 = bbox
    image_box = image_bgr.copy()
    cv2.rectangle(image_box, (x0, y0), (x1, y1), (0, 255, 255), 3)
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[:, :] = (0, 255, 255)
    overlay = np.where(mask[:, :, None], cv2.addWeighted(image_bgr, 0.45, color, 0.55, 0), overlay)
    boundary = boundary_mask(mask, radius=2)
    edge = np.zeros_like(image_bgr)
    edge[boundary] = (0, 255, 255)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edge[:, :, 0] = np.maximum(edge[:, :, 0], gray)
    edge[:, :, 1] = np.maximum(edge[:, :, 1], gray)
    edge[:, :, 2] = np.maximum(edge[:, :, 2], gray)
    panels = [crop_or_pad(panel, bbox) for panel in [image_box, overlay, edge]]
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    preview = np.hstack(padded)
    cv2.putText(preview, title[:160], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), preview)


def process_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(item["prepared_image"])
    mask_path = Path(item["prepared_current_mask"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    priority = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or priority is None:
        return {"sample_id": item["sample_id"], "status": "missing_image_or_mask"}
    if priority.shape[:2] != image.shape[:2]:
        priority = cv2.resize(priority, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    lid = label_id(item.get("semantic_label"))
    mask = priority == lid
    bbox_xyxy = (item.get("bbox_2d") or {}).get("xyxy") or [0, 0, image.shape[1] - 1, image.shape[0] - 1]
    bbox = clip_bbox(bbox_xyxy, image.shape[1], image.shape[0], pad=args.bbox_pad)
    x0, y0, x1, y1 = bbox
    bbox_mask = np.zeros(mask.shape, dtype=bool)
    bbox_mask[y0 : y1 + 1, x0 : x1 + 1] = True
    local_mask = mask & bbox_mask
    image_area = int(image.shape[0] * image.shape[1])
    bbox_area = int((x1 - x0 + 1) * (y1 - y0 + 1))
    mask_area = int(local_mask.sum())
    component = largest_component_stats(local_mask)
    color_metrics = lab_stats(image, local_mask)
    metrics = {
        "sample_id": item["sample_id"],
        "object_id": item.get("object_id"),
        "target_id": item.get("target_id"),
        "semantic_label": item.get("semantic_label"),
        "frame_id": item.get("frame_id"),
        "cam_id": item.get("cam_id"),
        "status": "ok",
        "bbox_xyxy_padded": [int(x0), int(y0), int(x1), int(y1)],
        "bbox_area": bbox_area,
        "bbox_area_ratio": float(bbox_area / max(image_area, 1)),
        "mask_area_in_bbox": mask_area,
        "mask_fill_ratio": float(mask_area / max(bbox_area, 1)),
        "component_count": int(component["components"]),
        "largest_component_ratio": float(component["largest_ratio"]),
        "min_rect_aspect": float(component["min_rect_aspect"]),
        **color_metrics,
    }
    flags = risk_flags(metrics, args)
    metrics["risk_flags"] = flags
    metrics["risk_score"] = int(len(flags))
    preview = args.output_dir / "previews" / f"{item['sample_id']}_color_geom_probe.jpg"
    title = f"{item['sample_id']} flags={','.join(flags) or 'ok'} fill={metrics['mask_fill_ratio']:.2f} aspect={metrics['min_rect_aspect']:.1f}"
    make_preview(image, local_mask, bbox, preview, title)
    metrics["preview_path"] = str(preview)
    return metrics


def make_contact_sheet(paths: list[Path], output: Path, thumb_width: int = 420, cols: int = 2) -> None:
    thumbs = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        scale = thumb_width / max(img.shape[1], 1)
        thumbs.append(cv2.resize(img, (thumb_width, max(1, int(img.shape[0] * scale)))))
    if not thumbs:
        return
    max_h = max(t.shape[0] for t in thumbs)
    padded = []
    for img in thumbs:
        if img.shape[0] < max_h:
            img = np.vstack([img, np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)])
        padded.append(img)
    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bbox-pad", type=int, default=16)
    parser.add_argument("--large-bbox-ratio", type=float, default=0.10)
    parser.add_argument("--high-fill-ratio", type=float, default=0.18)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--min-thin-aspect", type=float, default=3.0)
    parser.add_argument("--min-boundary-lab-contrast", type=float, default=8.0)
    parser.add_argument("--max-lab-std-mean", type=float, default=38.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = read_json(args.prepared_report)
    rows = [process_item(item, args) for item in prepared.get("items", [])]
    rows.sort(key=lambda row: (-int(row.get("risk_score", 0)), -float(row.get("mask_fill_ratio", 0.0)), str(row.get("sample_id", ""))))
    flags = Counter(flag for row in rows for flag in row.get("risk_flags", []))
    preview_paths = [Path(row["preview_path"]) for row in rows if row.get("preview_path")]
    contact = args.output_dir / "fine_mask_color_geometry_contact.jpg"
    make_contact_sheet(preview_paths[:40], contact)
    report = {
        "prepared_report": str(args.prepared_report),
        "output_dir": str(args.output_dir),
        "sample_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "risk_flag_counts": dict(flags),
        "contact_sheet": str(contact),
        "items": rows,
    }
    (args.output_dir / "fine_mask_color_geometry_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "fine_mask_color_geometry_probe.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sample_count": report["sample_count"],
                "ok_count": report["ok_count"],
                "risk_flag_counts": report["risk_flag_counts"],
                "contact_sheet": report["contact_sheet"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
