#!/usr/bin/env python3
"""Probe color/shape consistency for risky fine-object masks.

This is the first stage of the proposed vision-depth split guard.  It does not
run SAM2.  It measures whether current fine masks look like thin, coherent
objects in the undistorted image or like broad mixed regions that probably
swallow adjacent wall/floor/stair surfaces.

When an .lx file and image calibration are provided, this also projects the
same-frame point cloud back into the undistorted image and measures sparse
depth support.  This catches broad masks that swallow multiple depth layers
such as a foreground railing plus a background wall or ground surface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def depth_layer_stats(depths: np.ndarray, gap_threshold: float, min_points: int) -> dict[str, Any]:
    depths = np.asarray(depths, dtype=np.float32)
    depths = depths[np.isfinite(depths)]
    if len(depths) == 0:
        return {
            "depth_point_count": 0,
            "depth_min": 0.0,
            "depth_max": 0.0,
            "depth_p10": 0.0,
            "depth_p50": 0.0,
            "depth_p90": 0.0,
            "depth_span_p90_p10": 0.0,
            "depth_max_gap": 0.0,
            "depth_layer_count": 0,
            "depth_layer_sizes": [],
        }
    sorted_depths = np.sort(depths)
    gaps = np.diff(sorted_depths)
    split_at = np.where(gaps > gap_threshold)[0] + 1
    chunks = np.split(sorted_depths, split_at)
    layer_sizes = [int(len(chunk)) for chunk in chunks if len(chunk) >= min_points]
    p10, p50, p90 = np.percentile(sorted_depths, [10, 50, 90])
    return {
        "depth_point_count": int(len(sorted_depths)),
        "depth_min": float(sorted_depths[0]),
        "depth_max": float(sorted_depths[-1]),
        "depth_p10": float(p10),
        "depth_p50": float(p50),
        "depth_p90": float(p90),
        "depth_span_p90_p10": float(p90 - p10),
        "depth_max_gap": float(gaps.max()) if len(gaps) else 0.0,
        "depth_layer_count": int(len(layer_sizes)),
        "depth_layer_sizes": layer_sizes,
    }


def sparse_depth_mask_metrics(
    uu: np.ndarray,
    vv: np.ndarray,
    depths: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    in_bbox = (uu >= x0) & (uu <= x1) & (vv >= y0) & (vv <= y1)
    bbox_count = int(in_bbox.sum())
    if bbox_count == 0:
        metrics = depth_layer_stats(np.array([], dtype=np.float32), args.depth_gap_threshold, args.depth_layer_min_points)
        metrics.update({
            "depth_projected_points_bbox": 0,
            "depth_projected_points_mask": 0,
            "depth_mask_point_ratio": 0.0,
        })
        return metrics
    inside_mask = in_bbox & mask[vv, uu]
    mask_depths = depths[inside_mask]
    metrics = depth_layer_stats(mask_depths, args.depth_gap_threshold, args.depth_layer_min_points)
    metrics.update({
        "depth_projected_points_bbox": bbox_count,
        "depth_projected_points_mask": int(inside_mask.sum()),
        "depth_mask_point_ratio": float(inside_mask.sum() / max(bbox_count, 1)),
    })
    return metrics


class SameFrameDepthProjector:
    def __init__(self, lx_path: Path, scan_image_dir: Path, frame_start: int, frame_end: int, min_depth: float):
        os.environ["SCAN_IMAGE_DIR"] = str(scan_image_dir)
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import config  # type: ignore
        import project_priority_masks_to_lx as lxroute  # type: ignore

        self.config = config
        self.lxroute = lxroute
        self.min_depth = float(min_depth)
        self.sections = lxroute.read_lx_sections(lx_path)
        self.poses = {row["frame_id"]: row for row in config.load_img_pos(frame_start, frame_end)}
        self.handle = lx_path.open("rb")
        self.cache: dict[int, np.ndarray] = {}

    def frame_points(self, frame_id: int) -> np.ndarray | None:
        if frame_id in self.cache:
            return self.cache[frame_id]
        if frame_id < 0 or frame_id >= len(self.sections):
            return None
        points = self.lxroute.read_lx_points(self.handle, self.sections[frame_id])
        if len(self.cache) > 8:
            self.cache.clear()
        self.cache[frame_id] = points
        return points

    def project(self, frame_id: int, cam_id: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        pose = self.poses.get(int(frame_id))
        points = self.frame_points(int(frame_id))
        if pose is None or points is None or len(points) == 0:
            return None
        p_lidar = self.lxroute.transform_world_to_lidar(points, pose)
        t_cl = self.config.Tcl[int(cam_id)]
        p_cam = (t_cl[:3, :3] @ p_lidar.T + t_cl[:3, 3:]).T
        z = p_cam[:, 2]
        valid = z > self.min_depth
        if not np.any(valid):
            return None
        valid_idx = np.where(valid)[0]
        uv_h = (self.config.CAMERA_PARAMS[int(cam_id)]["K"] @ p_cam[valid].T).T
        u = uv_h[:, 0] / uv_h[:, 2]
        v = uv_h[:, 1] / uv_h[:, 2]
        in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_img):
            return None
        idx = valid_idx[in_img]
        uu = np.clip(np.rint(u[in_img]).astype(np.int32), 0, width - 1)
        vv = np.clip(np.rint(v[in_img]).astype(np.int32), 0, height - 1)
        depths = z[valid][in_img].astype(np.float32)
        keep = self.lxroute.zbuffer_visible(idx, uu, vv, depths, width)
        return uu[keep], vv[keep], depths[keep]


def make_depth_projector(items: list[dict[str, Any]], args: argparse.Namespace) -> SameFrameDepthProjector | None:
    if not args.lx_path or not args.scan_image_dir:
        return None
    frames = [int(item["frame_id"]) for item in items if item.get("frame_id") is not None]
    if not frames:
        return None
    return SameFrameDepthProjector(
        lx_path=args.lx_path,
        scan_image_dir=args.scan_image_dir,
        frame_start=min(frames),
        frame_end=max(frames),
        min_depth=args.min_depth,
    )


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
    if "depth_projected_points_mask" in metrics:
        if metrics["depth_projected_points_mask"] < args.min_depth_mask_points:
            flags.append("low_depth_support")
        if metrics["depth_mask_point_ratio"] < args.min_depth_mask_support_ratio:
            flags.append("weak_depth_mask_support")
        if metrics["depth_layer_count"] > 1:
            flags.append("multi_depth_layers")
        if metrics["depth_span_p90_p10"] > args.max_depth_span_p90_p10:
            flags.append("large_depth_span")
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


def process_item(item: dict[str, Any], args: argparse.Namespace, depth_projector: SameFrameDepthProjector | None = None) -> dict[str, Any]:
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
    depth_metrics: dict[str, Any] = {}
    if depth_projector is not None and item.get("frame_id") is not None and item.get("cam_id") is not None:
        projected = depth_projector.project(int(item["frame_id"]), int(item["cam_id"]), image.shape[1], image.shape[0])
        if projected is not None:
            uu, vv, depths = projected
            depth_metrics = sparse_depth_mask_metrics(uu, vv, depths, local_mask, bbox, args)
        else:
            depth_metrics = {
                **depth_layer_stats(np.array([], dtype=np.float32), args.depth_gap_threshold, args.depth_layer_min_points),
                "depth_projected_points_bbox": 0,
                "depth_projected_points_mask": 0,
                "depth_mask_point_ratio": 0.0,
            }
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
        **depth_metrics,
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
    parser.add_argument("--lx-path", type=Path, help="MANIFOLD .lx file for same-frame sparse depth support")
    parser.add_argument("--scan-image-dir", type=Path, help="Dataset image directory containing img_pos.txt and cam_in_ex.txt")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--depth-gap-threshold", type=float, default=0.60)
    parser.add_argument("--depth-layer-min-points", type=int, default=3)
    parser.add_argument("--min-depth-mask-points", type=int, default=8)
    parser.add_argument("--min-depth-mask-support-ratio", type=float, default=0.05)
    parser.add_argument("--max-depth-span-p90-p10", type=float, default=1.20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = read_json(args.prepared_report)
    items = prepared.get("items", [])
    depth_projector = make_depth_projector(items, args)
    rows = [process_item(item, args, depth_projector) for item in items]
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
