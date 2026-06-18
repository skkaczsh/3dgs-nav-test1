#!/usr/bin/env python3
"""Build frame-local Target records from priority masks and MANIFOLD .lx.

This is the provenance-preserving route for parking-scene semantics:

1. read one raw .lx section whose points are already in world coordinates;
2. project that same frame into the undistorted camera image;
3. sample the same-frame priority mask with z-buffer visibility;
4. split each priority class into 3D connected components;
5. write Target JSONL/PLY records that keep frame/camera/mask provenance.

The important invariant is that every target is produced from the same LiDAR
section and camera frame. Global objects may later be fused from these targets,
but image evidence must not be reconstructed by projecting global objects into
unrelated frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
try:
    import cv2
except ModuleNotFoundError:  # Keep --help and pure utility tests usable on local machines.
    cv2 = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from project_priority_masks_to_lx import (
    PRIORITY_COLORS,
    PRIORITY_NAMES,
    frame_path,
    priority_path,
    read_lx_points,
    read_lx_sections,
    transform_world_to_lidar,
    zbuffer_visible,
)


PARENT_BY_PRIORITY = {
    0: "residual",
    1: "surface",
    2: "surface",
    3: "vegetation",
    4: "object",
    5: "structure",
    6: "background",
}

DEFAULT_VOXEL_BY_LABEL = {
    "residual": 0.16,
    "ground": 0.30,
    "wall": 0.24,
    "grass": 0.20,
    "car": 0.18,
    "railing": 0.10,
}


def parse_labels(values: list[str], include_residual: bool) -> set[int]:
    labels: set[int] = set()
    by_name = {name: idx for idx, name in PRIORITY_NAMES.items()}
    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        if text.isdigit():
            labels.add(int(text))
        elif text in by_name:
            labels.add(int(by_name[text]))
        else:
            raise ValueError(f"Unknown priority label: {value}; valid={sorted(by_name)}")
    labels.discard(6)
    if not include_residual:
        labels.discard(0)
    return labels


def connectivity_voxel_size(label_name: str, args: argparse.Namespace) -> float:
    override = {
        "ground": args.ground_voxel,
        "wall": args.wall_voxel,
        "grass": args.grass_voxel,
        "car": args.car_voxel,
        "railing": args.railing_voxel,
        "residual": args.residual_voxel,
    }.get(label_name)
    if override and override > 0:
        return float(override)
    return float(DEFAULT_VOXEL_BY_LABEL.get(label_name, args.voxel_size))


def connected_components(points: np.ndarray, voxel_size: float, min_points: int) -> tuple[list[np.ndarray], np.ndarray]:
    if len(points) == 0:
        return [], np.zeros(0, dtype=bool)
    voxels = np.floor(points / float(voxel_size)).astype(np.int64)
    by_voxel: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, voxel in enumerate(voxels):
        by_voxel[(int(voxel[0]), int(voxel[1]), int(voxel[2]))].append(i)

    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    visited: set[tuple[int, int, int]] = set()
    components: list[np.ndarray] = []
    residual = np.zeros(len(points), dtype=bool)
    for start in by_voxel:
        if start in visited:
            continue
        queue: deque[tuple[int, int, int]] = deque([start])
        visited.add(start)
        comp_voxels: list[tuple[int, int, int]] = []
        while queue:
            voxel = queue.popleft()
            comp_voxels.append(voxel)
            for dx, dy, dz in offsets:
                nxt = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                if nxt in by_voxel and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        comp = np.asarray([i for voxel in comp_voxels for i in by_voxel[voxel]], dtype=np.int64)
        if len(comp) >= int(min_points):
            components.append(comp)
        else:
            residual[comp] = True
    components.sort(key=len, reverse=True)
    return components, residual


def pca_stats(points: np.ndarray) -> dict[str, Any]:
    if len(points) < 3:
        return {
            "normal": [0.0, 0.0, 1.0],
            "eigenvalues": [0.0, 0.0, 0.0],
            "linearity": 0.0,
            "planarity": 0.0,
            "scattering": 0.0,
        }
    centered = points.astype(np.float64) - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    denom = max(float(vals[0]), 1e-12)
    normal = vecs[:, -1]
    if normal[2] < 0:
        normal = -normal
    return {
        "normal": [float(x) for x in normal],
        "eigenvalues": [float(x) for x in vals],
        "linearity": float((vals[0] - vals[1]) / denom),
        "planarity": float((vals[1] - vals[2]) / denom),
        "scattering": float(vals[2] / denom),
    }


def bbox2d(uu: np.ndarray, vv: np.ndarray) -> dict[str, Any]:
    if len(uu) == 0:
        return {"xyxy": [0, 0, 0, 0], "area": 0}
    x0, x1 = int(uu.min()), int(uu.max())
    y0, y1 = int(vv.min()), int(vv.max())
    return {"xyxy": [x0, y0, x1, y1], "area": int(max(x1 - x0 + 1, 0) * max(y1 - y0 + 1, 0))}


def target_summary(points: np.ndarray, colors: np.ndarray, uu: np.ndarray, vv: np.ndarray) -> dict[str, Any]:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    mean_color = colors.mean(axis=0) if len(colors) else np.zeros(3)
    return {
        "bbox_3d": {"min": [float(x) for x in bbox_min], "max": [float(x) for x in bbox_max]},
        "centroid": [float(x) for x in points.mean(axis=0)],
        "mean_color": [float(x) for x in mean_color],
        "bbox_2d": bbox2d(uu, vv),
        "pca": pca_stats(points),
        "cluster_size": int(len(points)),
    }


def project_one_camera(points: np.ndarray, pose: dict[str, Any], frame_id: int, cam_id: int, args: argparse.Namespace) -> dict[str, Any] | None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to read images and priority masks.")
    mask_file = priority_path(args.priority_dir, cam_id, frame_id, args.priority_suffix)
    img_file = frame_path(args.frame_root, cam_id, frame_id)
    if not mask_file.exists() or not img_file.exists():
        return None
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    if mask is None or image is None:
        return None
    h, w = mask.shape[:2]

    p_lidar = transform_world_to_lidar(points, pose)
    t_cl = config.Tcl[cam_id]
    p_cam = (t_cl[:3, :3] @ p_lidar.T + t_cl[:3, 3:]).T
    z = p_cam[:, 2]
    valid = z > args.min_depth
    if not np.any(valid):
        return None
    valid_idx = np.where(valid)[0]
    uv_h = (config.CAMERA_PARAMS[cam_id]["K"] @ p_cam[valid].T).T
    u = uv_h[:, 0] / uv_h[:, 2]
    v = uv_h[:, 1] / uv_h[:, 2]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_img):
        return None

    point_indices = valid_idx[in_img]
    uu = np.clip(np.rint(u[in_img]).astype(np.int32), 0, w - 1)
    vv = np.clip(np.rint(v[in_img]).astype(np.int32), 0, h - 1)
    depths = z[valid][in_img].astype(np.float32)
    if args.zbuffer:
        keep = zbuffer_visible(point_indices, uu, vv, depths, w)
        point_indices, uu, vv, depths = point_indices[keep], uu[keep], vv[keep], depths[keep]
    if len(point_indices) == 0:
        return None

    labels = mask[vv, uu].astype(np.uint8)
    rgb = image[vv, uu][:, ::-1]
    return {
        "image_path": img_file,
        "mask_path": mask_file,
        "point_indices": point_indices,
        "uu": uu,
        "vv": vv,
        "depths": depths,
        "labels": labels,
        "colors": rgb,
        "image_shape": [int(h), int(w)],
    }


def write_ply_header(path: Path, count: int) -> None:
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int target\n"
        "property uchar priority\n"
        "property int frame\n"
        "property int camera\n"
        "property int point_index\n"
        "end_header\n"
    )
    path.write_text(header, encoding="utf-8")


def append_target_points(handle, target_index: int, frame_id: int, cam_id: int, label_id: int, points: np.ndarray, point_indices: np.ndarray) -> None:
    color = PRIORITY_COLORS.get(label_id, (200, 200, 200))
    for point, point_index in zip(points, point_indices):
        handle.write(
            f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
            f"{color[0]} {color[1]} {color[2]} {target_index} {label_id} "
            f"{frame_id} {cam_id} {int(point_index)}\n"
        )


def build_frame_targets(points: np.ndarray, pose: dict[str, Any], frame_id: int, args: argparse.Namespace, target_start: int) -> tuple[list[dict[str, Any]], list[tuple[int, int, int, np.ndarray, np.ndarray]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ply_chunks: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    label_ids = parse_labels(args.labels, args.include_residual)
    per_cam_stats = []
    target_index = target_start

    for cam_id in args.cams:
        projected = project_one_camera(points, pose, frame_id, cam_id, args)
        if projected is None:
            per_cam_stats.append({"cam_id": cam_id, "status": "missing_or_no_projection"})
            continue

        labels = projected["labels"]
        point_indices = projected["point_indices"]
        colors = projected["colors"]
        uu = projected["uu"]
        vv = projected["vv"]
        cam_hist = Counter(int(x) for x in labels.tolist())
        made = 0
        small_residual = 0

        for label_id in sorted(label_ids):
            class_mask = labels == label_id
            if not np.any(class_mask):
                continue
            local_indices = np.where(class_mask)[0]
            class_points = points[point_indices[local_indices]]
            label_name = PRIORITY_NAMES.get(label_id, str(label_id))
            voxel = connectivity_voxel_size(label_name, args)
            min_points = args.surface_min_points if PARENT_BY_PRIORITY.get(label_id) == "surface" else args.min_target_points
            components, residual = connected_components(class_points, voxel, min_points)
            small_residual += int(residual.sum())
            for comp_id, comp in enumerate(components):
                selected = local_indices[comp]
                target_points = points[point_indices[selected]]
                target_colors = colors[selected]
                target_point_indices = point_indices[selected]
                target_id = f"pt_{frame_id:06d}_cam{cam_id}_p{label_id}_cc{comp_id:03d}"
                summary = target_summary(target_points, target_colors, uu[selected], vv[selected])
                row = {
                    "target_id": target_id,
                    "target_index": int(target_index),
                    "frame_id": int(frame_id),
                    "cam_id": int(cam_id),
                    "mask_id": int(label_id),
                    "priority_label_id": int(label_id),
                    "label": label_name,
                    "raw_label": label_name,
                    "parent_class": PARENT_BY_PRIORITY.get(label_id, "other"),
                    "confidence": 1.0,
                    "image_path": str(projected["image_path"]),
                    "mask_path": str(projected["mask_path"]),
                    "raw_frame_ply": "",
                    "colored_frame_ply": "",
                    "point_indices": [int(x) for x in target_point_indices.tolist()],
                    "source_point_count": int(len(points)),
                    "projected_class_points": int(class_mask.sum()),
                    "component_index": int(comp_id),
                    "connectivity_voxel_size": float(voxel),
                    **summary,
                }
                rows.append(row)
                ply_chunks.append((target_index, label_id, cam_id, target_points, target_point_indices))
                target_index += 1
                made += 1

        per_cam_stats.append({
            "cam_id": int(cam_id),
            "status": "ok",
            "projected_points": int(len(point_indices)),
            "label_counts": {PRIORITY_NAMES.get(k, str(k)): int(v) for k, v in sorted(cam_hist.items())},
            "targets": int(made),
            "small_target_residual_points": int(small_residual),
        })

    report = {
        "frame_id": int(frame_id),
        "raw_points": int(len(points)),
        "targets": int(len(rows)),
        "target_points": int(sum(row["cluster_size"] for row in rows)),
        "per_cam": per_cam_stats,
    }
    return rows, ply_chunks, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lx", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--priority-dir", type=Path, required=True)
    parser.add_argument("--priority-suffix", default="_priority")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--labels", nargs="+", default=["ground", "wall", "grass", "car", "railing"])
    parser.add_argument("--include-residual", action="store_true")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--zbuffer", action="store_true", default=True)
    parser.add_argument("--voxel-size", type=float, default=0.16)
    parser.add_argument("--ground-voxel", type=float, default=0.30)
    parser.add_argument("--wall-voxel", type=float, default=0.24)
    parser.add_argument("--grass-voxel", type=float, default=0.20)
    parser.add_argument("--car-voxel", type=float, default=0.18)
    parser.add_argument("--railing-voxel", type=float, default=0.10)
    parser.add_argument("--residual-voxel", type=float, default=0.16)
    parser.add_argument("--min-target-points", type=int, default=20)
    parser.add_argument("--surface-min-points", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets_path = args.output_dir / "frame_targets.jsonl"
    frame_report_path = args.output_dir / "frame_target_report.jsonl"
    ply_path = args.output_dir / "frame_targets.ply"
    summary_path = args.output_dir / "frame_target_summary.json"
    if args.resume and targets_path.exists() and frame_report_path.exists() and ply_path.exists():
        print(json.dumps({"status": "skip_existing", "output_dir": str(args.output_dir)}, ensure_ascii=False))
        return

    sections = read_lx_sections(args.lx)
    poses = {row["frame_id"]: row for row in config.load_img_pos(args.start, args.end)}
    end = min(args.end, len(sections) - 1)
    frame_ids = [i for i in range(args.start, end + 1, max(args.stride, 1)) if i in poses and i < len(sections)]
    if not frame_ids:
        raise SystemExit("No overlapping .lx sections, img_pos rows, and requested frame range.")

    tmp_fd, tmp_name = tempfile.mkstemp(prefix="frame_targets_", suffix=".plybody", dir=str(args.output_dir))
    os.close(tmp_fd)
    tmp_body = Path(tmp_name)
    target_count = 0
    point_count = 0
    label_counts = Counter()
    frame_reports: list[dict[str, Any]] = []

    with args.lx.open("rb") as lx_f, targets_path.open("w", encoding="utf-8") as target_f, frame_report_path.open("w", encoding="utf-8") as report_f, tmp_body.open("w", encoding="utf-8") as ply_body:
        for n, frame_id in enumerate(frame_ids, start=1):
            points = read_lx_points(lx_f, sections[frame_id])
            rows, chunks, frame_report = build_frame_targets(points, poses[frame_id], frame_id, args, target_count)
            for row in rows:
                target_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                label_counts[row["label"]] += 1
            for target_index, label_id, cam_id, target_points, point_indices in chunks:
                append_target_points(ply_body, target_index, frame_id, cam_id, label_id, target_points, point_indices)
                point_count += int(len(target_points))
            target_count += len(rows)
            frame_reports.append(frame_report)
            report_f.write(json.dumps(frame_report, ensure_ascii=False) + "\n")
            if n == 1 or n % args.progress_every == 0:
                print(json.dumps({"processed": n, **frame_report}, ensure_ascii=False))

    write_ply_header(ply_path, point_count)
    with ply_path.open("a", encoding="utf-8") as out, tmp_body.open("r", encoding="utf-8") as body:
        for chunk in iter(lambda: body.read(1024 * 1024), ""):
            out.write(chunk)
    tmp_body.unlink(missing_ok=True)

    summary = {
        "lx": str(args.lx),
        "frame_root": str(args.frame_root),
        "priority_dir": str(args.priority_dir),
        "output_dir": str(args.output_dir),
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "cams": args.cams,
        "labels": sorted(PRIORITY_NAMES.get(x, str(x)) for x in parse_labels(args.labels, args.include_residual)),
        "frame_count": len(frame_ids),
        "target_count": int(target_count),
        "target_point_count": int(point_count),
        "target_label_counts": dict(label_counts),
        "targets_jsonl": str(targets_path),
        "targets_ply": str(ply_path),
        "frame_report_jsonl": str(frame_report_path),
        "elapsed_sec": float(time.time() - t0),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
