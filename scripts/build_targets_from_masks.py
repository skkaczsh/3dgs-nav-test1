#!/usr/bin/env python3
"""Build 3D Target records from 2D instance masks on the validated route.

Each target is one connected 3D component produced by projecting one semantic
instance mask into one frame point cloud. The projection uses the same
img_pos.txt + cam_in_ex.txt + Tcl + Til chain as project_color.py and
project_semantic.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from project_color import load_ply_xyz
from project_semantic import LABEL_COLORS, LABEL_NAMES, zbuffer_visible_indices


LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}
SKIP_LABELS = {"sky", "ignore"}
PARENT_CLASSES = {
    "floor": "surface",
    "wall": "surface",
    "ceiling": "surface",
    "road": "surface",
    "building": "structure",
    "railing": "structure",
    "pipe": "structure",
    "equipment": "object",
    "furniture": "object",
    "tree": "vegetation",
    "grass": "vegetation",
    "person": "dynamic",
    "car": "dynamic",
    "water": "background",
    "other": "other",
    "unknown": "unknown",
}


def frames_with_combo(base: Path, combo: str) -> list[int]:
    frames = set()
    images_dir = base / "images"
    if not images_dir.exists():
        return []
    for path in images_dir.glob(f"cam*_*/*/instance.png"):
        if path.parent.name != combo:
            continue
        try:
            frames.add(int(path.parent.parent.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(frames)


def artifact_dir(base: Path, combo: str, cam_id: int, frame_id: int) -> Path:
    return base / "images" / f"cam{cam_id}_{frame_id:06d}" / combo


def read_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header = []
        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = "ascii"
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header:
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element "):
                in_vertex = False
            elif in_vertex and line.startswith("property "):
                _, ptype, name = line.split()[:3]
                properties.append((ptype, name))
        if fmt != "ascii":
            points = load_ply_xyz(str(path))
            return points.astype(np.float32), np.zeros((len(points), 3), dtype=np.uint8)
        rows = []
        for raw in f:
            parts = raw.decode("utf-8", errors="replace").strip().split()
            if len(parts) >= 3:
                rows.append(parts)
        if vertex_count and len(rows) > vertex_count:
            rows = rows[:vertex_count]
    if not rows:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    names = [name for _, name in properties]
    data = np.array(rows, dtype=np.float64)
    points = data[:, :3].astype(np.float32)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    try:
        ridx, gidx, bidx = names.index("red"), names.index("green"), names.index("blue")
        colors = np.clip(data[:, [ridx, gidx, bidx]], 0, 255).astype(np.uint8)
    except ValueError:
        pass
    return points, colors


def transform_project(points: np.ndarray, frame_id: int, cam_id: int, config, min_depth: float):
    pose_data = config.load_img_pos(frame_id, frame_id)
    if not pose_data:
        return None
    T = pose_data[0]["T_world_robot"]
    R_rw = T[:3, :3]
    t_rw = T[:3, 3]
    R_wr = R_rw.T
    t_wr = (-R_wr @ t_rw).reshape(3)
    R_li = config.Til[:3, :3].T
    t_li = (-R_li @ config.Til[:3, 3]).reshape(3)
    T_cl = config.Tcl[cam_id]

    p_robot = (R_wr @ points.T + t_wr.reshape(3, 1)).T
    p_lidar = (R_li @ p_robot.T + t_li.reshape(3, 1)).T
    p_cam = (T_cl[:3, :3] @ p_lidar.T + T_cl[:3, 3].reshape(3, 1)).T
    z = p_cam[:, 2]
    valid = z > min_depth
    if not np.any(valid):
        return None
    uv_h = (config.CAMERA_PARAMS[cam_id]["K"] @ p_cam[valid].T).T
    u = uv_h[:, 0] / uv_h[:, 2]
    v = uv_h[:, 1] / uv_h[:, 2]
    idx = np.where(valid)[0]
    return idx, u, v, z[valid]


def connected_components(points: np.ndarray, voxel_size: float, min_points: int) -> tuple[list[np.ndarray], np.ndarray]:
    if len(points) == 0:
        return [], np.zeros(0, dtype=bool)
    voxels = np.floor(points / voxel_size).astype(np.int64)
    voxel_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for i, voxel in enumerate(voxels):
        voxel_to_indices.setdefault(tuple(int(x) for x in voxel), []).append(i)
    visited: set[tuple[int, int, int]] = set()
    components = []
    residual = np.zeros(len(points), dtype=bool)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for start in voxel_to_indices:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        comp_voxels = []
        while queue:
            voxel = queue.popleft()
            comp_voxels.append(voxel)
            for dx, dy, dz in offsets:
                nb = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                if nb in voxel_to_indices and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        comp_indices = np.array([i for voxel in comp_voxels for i in voxel_to_indices[voxel]], dtype=np.int64)
        if len(comp_indices) >= min_points:
            components.append(comp_indices)
        else:
            residual[comp_indices] = True
    components.sort(key=len, reverse=True)
    return components, residual


def pca_stats(points: np.ndarray) -> dict:
    if len(points) < 3:
        return {"normal": [0.0, 0.0, 1.0], "eigenvalues": [0.0, 0.0, 0.0], "linearity": 0.0, "planarity": 0.0}
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    normal = vecs[:, -1]
    if normal[2] < 0:
        normal = -normal
    denom = float(vals[0]) if vals[0] > 1e-12 else 1.0
    return {
        "normal": [float(x) for x in normal],
        "eigenvalues": [float(x) for x in vals],
        "linearity": float((vals[0] - vals[1]) / denom),
        "planarity": float((vals[1] - vals[2]) / denom),
    }


def summarize_points(points: np.ndarray, colors: np.ndarray) -> dict:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    centroid = points.mean(axis=0)
    mean_color = colors.mean(axis=0) if len(colors) else np.zeros(3)
    stats = pca_stats(points)
    return {
        "bbox_3d": {"min": [float(x) for x in bbox_min], "max": [float(x) for x in bbox_max]},
        "centroid": [float(x) for x in centroid],
        "mean_color": [float(x) for x in mean_color],
        "pca": stats,
        "cluster_size": int(len(points)),
    }


def label_to_id(label: str) -> int:
    return int(LABEL_IDS.get(label, 0))


def stable_color(target_id: int, label: str) -> tuple[int, int, int]:
    label_id = label_to_id(label)
    if label_id in LABEL_COLORS:
        return LABEL_COLORS[label_id]
    rng = np.random.default_rng(abs(hash((target_id, label))) % (2**32))
    return tuple(int(x) for x in rng.integers(60, 240, 3))


def write_targets_ply(path: Path, rows: list[dict], points_by_target: dict[str, tuple[np.ndarray, str]]) -> None:
    total = sum(len(points) for points, _ in points_by_target.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int target\nproperty uchar semantic\n")
        f.write("property int frame\nproperty int camera\nproperty int mask\n")
        f.write("end_header\n")
        for row in rows:
            points, label = points_by_target[row["target_id"]]
            color = stable_color(row["target_index"], label)
            sem = label_to_id(label)
            for p in points:
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]} {row['target_index']} {sem} "
                    f"{row['frame_id']} {row['cam_id']} {row['mask_id']}\n"
                )


def write_residual_ply(path: Path, residual_rows: list[dict]) -> None:
    total = sum(len(row["points"]) for row in residual_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("property uchar semantic\n")
        f.write("property int frame\nproperty int camera\nproperty int mask\n")
        f.write("property int point_index\n")
        f.write("end_header\n")
        for row in residual_rows:
            label = row["label"]
            sem = label_to_id(label)
            color = LABEL_COLORS.get(sem, LABEL_COLORS.get(0, (128, 128, 128)))
            for point, visual, point_index in zip(row["points"], row["colors"], row["point_indices"]):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} "
                    f"{int(visual[0])} {int(visual[1])} {int(visual[2])} "
                    f"{sem} {row['frame_id']} {row['cam_id']} {row['mask_id']} {int(point_index)}\n"
                )


def load_labels(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "labels" in raw:
        raw = raw["labels"]
    labels = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                mask_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                labels[mask_id] = str(value.get("label") or value.get("name") or "unknown")
            else:
                labels[mask_id] = str(value)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                mask_id = int(item.get("id", item.get("mask_id", 0)))
                labels[mask_id] = str(item.get("label") or item.get("name") or "unknown")
    return labels


def process_frame(frame_id: int, args: argparse.Namespace, config) -> dict:
    target_path = args.output_dir / "targets" / f"targets_frame_{frame_id:04d}.jsonl"
    report_path = args.output_dir / "reports" / f"targets_frame_{frame_id:04d}_report.json"
    ply_path = args.output_dir / "targets" / f"targets_frame_{frame_id:04d}.ply"
    residual_ply_path = args.output_dir / "residuals" / f"residuals_frame_{frame_id:04d}.ply"
    if args.resume and target_path.exists() and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    color_path = args.color_dir / f"frame_{frame_id:04d}.ply"
    raw_path = Path(config.EXTRACTED_DIR) / f"section_{frame_id:04d}.ply"
    if not color_path.exists():
        return {"frame_id": frame_id, "status": "missing_color_ply", "color_path": str(color_path)}
    points, colors = read_colored_ply(color_path)
    if len(points) == 0:
        return {"frame_id": frame_id, "status": "empty_color_ply", "color_path": str(color_path)}

    rows: list[dict] = []
    points_by_target: dict[str, tuple[np.ndarray, str]] = {}
    residual_rows: list[dict] = []
    residual_points = 0
    residual_label_counts = Counter()
    masks_seen = 0
    masks_missing = 0
    skipped_masks = Counter()
    target_index_base = args.target_index_base + frame_id * args.target_index_frame_stride

    for cam_id in args.cams:
        combo_dir = artifact_dir(args.semantic_eval_dir, args.combo, cam_id, frame_id)
        instance_path = combo_dir / "instance.png"
        labels_path = combo_dir / "labels.json"
        if not instance_path.exists() or not labels_path.exists():
            masks_missing += 1
            continue
        instance = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
        if instance is None:
            masks_missing += 1
            continue
        if instance.ndim == 3:
            instance = instance[:, :, 0]
        labels = load_labels(labels_path)
        projected = transform_project(points, frame_id, cam_id, config, args.min_depth)
        if projected is None:
            continue
        point_idx, u, v, depths = projected
        height, width = instance.shape[:2]
        in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_img):
            continue
        point_idx = point_idx[in_img]
        uu = np.clip(np.rint(u[in_img]).astype(np.int32), 0, width - 1)
        vv = np.clip(np.rint(v[in_img]).astype(np.int32), 0, height - 1)
        depths = depths[in_img]
        if args.zbuffer_visible:
            visible = zbuffer_visible_indices(point_idx, np.column_stack([uu, vv]), depths, width)
            point_idx, uu, vv = point_idx[visible], uu[visible], vv[visible]
        if len(point_idx) == 0:
            continue

        instance_ids = instance[vv, uu].astype(np.int64)
        for mask_id in sorted(int(x) for x in np.unique(instance_ids) if int(x) != 0):
            masks_seen += 1
            label = labels.get(mask_id, "unknown")
            if label in SKIP_LABELS or label in args.skip_labels:
                skipped_masks[label] += 1
                continue
            selected = point_idx[instance_ids == mask_id]
            if len(selected) == 0:
                continue
            comps, residual = connected_components(points[selected], args.voxel_size, args.min_target_points)
            residual_points += int(residual.sum())
            if args.write_residual_ply and np.any(residual):
                residual_idx = selected[residual]
                residual_rows.append(
                    {
                        "frame_id": int(frame_id),
                        "cam_id": int(cam_id),
                        "mask_id": int(mask_id),
                        "label": label,
                        "point_indices": [int(x) for x in residual_idx.tolist()],
                        "points": points[residual_idx],
                        "colors": colors[residual_idx],
                    }
                )
                residual_label_counts[label] += int(residual.sum())
            for comp_id, comp_local in enumerate(comps):
                global_idx = selected[comp_local]
                target_id = f"t_{frame_id:04d}_cam{cam_id}_m{mask_id:04d}_c{comp_id:02d}"
                target_index = target_index_base + len(rows) + 1
                pts = points[global_idx]
                cols = colors[global_idx]
                summary = summarize_points(pts, cols)
                row = {
                    "target_id": target_id,
                    "target_index": int(target_index),
                    "frame_id": int(frame_id),
                    "cam_id": int(cam_id),
                    "mask_id": int(mask_id),
                    "component_id": int(comp_id),
                    "label": label,
                    "semantic_id": label_to_id(label),
                    "parent_class": PARENT_CLASSES.get(label, "other"),
                    "confidence": 1.0,
                    "image_path": str(combo_dir / "image.png"),
                    "mask_path": str(instance_path),
                    "raw_frame_ply": str(raw_path),
                    "colored_frame_ply": str(color_path),
                    "point_indices": [int(x) for x in global_idx.tolist()],
                    **summary,
                }
                rows.append(row)
                points_by_target[target_id] = (pts, label)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.write_ply and rows:
        write_targets_ply(ply_path, rows, points_by_target)
    if args.write_residual_ply and residual_rows:
        write_residual_ply(residual_ply_path, residual_rows)
    label_counts = Counter(row["label"] for row in rows)
    report = {
        "frame_id": int(frame_id),
        "status": "ok",
        "points": int(len(points)),
        "targets": int(len(rows)),
        "target_points": int(sum(row["cluster_size"] for row in rows)),
        "small_target_residual_points": int(residual_points),
        "small_target_residual_label_counts": dict(residual_label_counts),
        "masks_seen": int(masks_seen),
        "masks_missing_cameras": int(masks_missing),
        "skipped_masks": dict(skipped_masks),
        "label_counts": dict(label_counts),
        "target_jsonl": str(target_path),
        "target_ply": str(ply_path) if args.write_ply and rows else "",
        "residual_ply": str(residual_ply_path) if args.write_residual_ply and residual_rows else "",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-eval-dir", type=Path, required=True)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--color-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=999)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--frames-from-semantic-dir", action="store_true")
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--min-target-points", type=int, default=20)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--zbuffer-visible", action="store_true", default=True)
    parser.add_argument("--no-zbuffer-visible", dest="zbuffer_visible", action="store_false")
    parser.add_argument("--skip-labels", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--write-residual-ply", action="store_true")
    parser.add_argument("--target-index-base", type=int, default=0)
    parser.add_argument("--target-index-frame-stride", type=int, default=100000)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.frames_from_semantic_dir:
        frames = [x for x in frames_with_combo(args.semantic_eval_dir, args.combo) if args.start <= x <= args.end]
    elif args.frames is not None:
        frames = sorted(set(args.frames))
    else:
        frames = list(range(args.start, args.end + 1, args.stride))

    reports = []
    for frame_id in frames:
        row = process_frame(frame_id, args, config)
        reports.append(row)
        if row.get("status") == "ok":
            print(
                f"frame={frame_id} targets={row['targets']} "
                f"target_points={row['target_points']} residual={row['small_target_residual_points']}"
            )
        else:
            print(f"frame={frame_id} status={row.get('status')}")

    merged_jsonl = args.output_dir / "targets" / "targets_all.jsonl"
    merged_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with merged_jsonl.open("w", encoding="utf-8") as out:
        for report in reports:
            path = report.get("target_jsonl")
            if path and Path(path).exists():
                out.write(Path(path).read_text(encoding="utf-8"))

    ok = [r for r in reports if r.get("status") == "ok"]
    report = {
        "semantic_eval_dir": str(args.semantic_eval_dir),
        "combo": args.combo,
        "color_dir": str(args.color_dir),
        "frames": reports,
        "targets_jsonl": str(merged_jsonl),
        "summary": {
            "frame_count": len(reports),
            "ok_count": len(ok),
            "target_count": int(sum(r.get("targets", 0) for r in ok)),
            "target_points": int(sum(r.get("target_points", 0) for r in ok)),
            "small_target_residual_points": int(sum(r.get("small_target_residual_points", 0) for r in ok)),
            "avg_targets_per_ok_frame": float(np.mean([r.get("targets", 0) for r in ok])) if ok else 0.0,
        },
    }
    out_report = args.output_dir / "reports" / "target_build_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={out_report}")


if __name__ == "__main__":
    main()
