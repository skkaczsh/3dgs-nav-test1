#!/usr/bin/env python3
"""Build image evidence crops for object-level review/DINO stages.

Objects in the full-scene dataset are global point-cloud components. DINO or
human visual review needs image evidence: which frame/camera best sees each
object, where it projects, and a crop to run detector prompts on.

This script projects sampled object points into already-undistorted camera
frames using the same config.py calibration chain as the projection route.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def configure_dataset_from_cli() -> None:
    """Set config's dataset paths before importing the shared calibration module."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path)
    args, _unknown = parser.parse_known_args()
    if args.data_dir:
        data_dir = args.data_dir.resolve()
        os.environ["SCAN_DATA_DIR"] = str(data_dir)
        os.environ["SCAN_IMAGE_DIR"] = str(data_dir / "image")


configure_dataset_from_cli()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from project_priority_masks_to_lx import read_lx_points, read_lx_sections


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_source_frame_support(path: Path) -> dict[int, list[int]]:
    """Load conservative per-superpoint source-frame candidates."""
    support: dict[int, list[int]] = {}
    for row in read_jsonl(path):
        object_id = int(row["object_id"])
        frames = [int(item["frame_id"]) for item in row.get("top_source_frames", [])]
        if frames:
            support[object_id] = frames
    return support


def parse_ply_header(path: Path) -> tuple[list[str], int]:
    props: list[str] = []
    vertex_count = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            parts = s.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif s == "end_header":
                break
    return props, vertex_count


def load_object_point_samples(
    ply_path: Path,
    object_ids: set[int],
    max_points_per_object: int,
    seed: int,
) -> dict[int, np.ndarray]:
    props, _vertex_count = parse_ply_header(ply_path)
    idx = {name: i for i, name in enumerate(props)}
    object_col = idx.get("object", idx.get("object_id"))
    if object_col is None:
        raise ValueError(f"PLY missing object/object_id field: {ply_path}")
    required = {"x", "y", "z"}
    if not required.issubset(idx):
        raise ValueError(f"PLY missing xyz fields: {ply_path}")

    rng = np.random.default_rng(seed)
    samples: dict[int, list[list[float]]] = {oid: [] for oid in object_ids}
    seen = Counter()
    in_body = False
    with ply_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not in_body:
                if line.strip() == "end_header":
                    in_body = True
                continue
            parts = line.strip().split()
            if len(parts) <= object_col:
                continue
            try:
                object_id = int(round(float(parts[object_col])))
            except ValueError:
                continue
            if object_id not in object_ids:
                continue
            seen[object_id] += 1
            point = [
                float(parts[idx["x"]]),
                float(parts[idx["y"]]),
                float(parts[idx["z"]]),
            ]
            if "source_frame" in idx:
                point.append(float(parts[idx["source_frame"]]))
            bucket = samples[object_id]
            if len(bucket) < max_points_per_object:
                bucket.append(point)
            else:
                replace_at = int(rng.integers(0, seen[object_id]))
                if replace_at < max_points_per_object:
                    bucket[replace_at] = point

    return {oid: np.asarray(points, dtype=np.float32) for oid, points in samples.items() if points}


def points_for_source_frame(points: np.ndarray, frame_id: int) -> np.ndarray:
    """Use only raw points actually scanned in a source-supported frame."""
    if points.shape[1] == 3:
        return points
    return points[np.rint(points[:, 3]).astype(np.int32) == frame_id, :3]


def available_source_frames(points: np.ndarray, min_points: int = 1) -> set[int] | None:
    if points.shape[1] == 3:
        return None
    frames, counts = np.unique(np.rint(points[:, 3]).astype(np.int32), return_counts=True)
    return set(frames[counts >= min_points].tolist())


def transform_world_to_lidar(points_world: np.ndarray, pose: dict[str, Any]) -> np.ndarray:
    T = pose["T_world_robot"]
    R_rw = T[:3, :3]
    t_rw = T[:3, 3]
    R_wr = R_rw.T
    t_wr = (-R_wr @ t_rw).reshape(3)
    R_li = config.Til[:3, :3].T
    t_li = (-R_li @ config.Til[:3, 3]).reshape(3)
    points64 = points_world.astype(np.float64, copy=False)
    p_robot = (R_wr @ points64.T + t_wr.reshape(3, 1)).T
    return (R_li @ p_robot.T + t_li.reshape(3, 1)).T


def project_points(points_world: np.ndarray, pose: dict[str, Any], cam_id: int, min_depth: float) -> tuple[np.ndarray, np.ndarray]:
    p_lidar = transform_world_to_lidar(points_world, pose)
    t_cl = config.Tcl[cam_id]
    p_cam = (t_cl[:3, :3] @ p_lidar.T + t_cl[:3, 3:]).T
    z = p_cam[:, 2]
    valid = z > min_depth
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)
    uv_h = (config.CAMERA_PARAMS[cam_id]["K"] @ p_cam[valid].T).T
    uv = np.column_stack([uv_h[:, 0] / uv_h[:, 2], uv_h[:, 1] / uv_h[:, 2]]).astype(np.float32)
    depth = z[valid].astype(np.float32)
    return uv, depth


def world_up_image_hint(points_world: np.ndarray, pose: dict[str, Any], cam_id: int) -> dict[str, Any]:
    """Express world-up in this camera's pixel coordinates near an object."""
    if len(points_world) == 0:
        return {"world_up_image_hint": "unavailable"}
    center_world = points_world.mean(axis=0, keepdims=True)
    center_lidar = transform_world_to_lidar(center_world, pose)[0]
    t_cl = config.Tcl[cam_id]
    center_cam = t_cl[:3, :3] @ center_lidar + t_cl[:3, 3]
    if center_cam[2] <= 0.1:
        return {"world_up_image_hint": "unavailable"}

    r_world_to_camera = t_cl[:3, :3] @ config.Til[:3, :3].T @ pose["T_world_robot"][:3, :3].T
    up_cam = r_world_to_camera @ np.array([0.0, 0.0, 1.0])
    k = config.CAMERA_PARAMS[cam_id]["K"]

    def pixel(point_cam: np.ndarray) -> np.ndarray:
        uv_h = k @ point_cam
        return uv_h[:2] / uv_h[2]

    base = pixel(center_cam)
    raised = pixel(center_cam + up_cam)
    delta = raised - base
    length = float(np.linalg.norm(delta))
    if length < 1e-3 or not np.isfinite(length):
        return {"world_up_image_hint": "unavailable"}
    unit = delta / length
    horizontal = "right" if unit[0] > 0.35 else ("left" if unit[0] < -0.35 else "center")
    vertical = "down" if unit[1] > 0.35 else ("up" if unit[1] < -0.35 else "center")
    return {
        "world_up_image_hint": f"{vertical}_{horizontal}",
        "world_up_image_unit_xy": [round(float(unit[0]), 3), round(float(unit[1]), 3)],
    }


def camera_pose_context(points_world: np.ndarray, pose: dict[str, Any], cam_id: int) -> dict[str, Any]:
    """Return calibrated pose facts for a projected object observation.

    Image pixels alone cannot tell a VLM whether its top edge is world-up when
    the rig is rolled or pitched. These values are derived from the same
    world->LiDAR->camera chain used for the projection, so downstream review
    can treat them as geometry evidence rather than visual guesswork.
    """
    if len(points_world) == 0:
        return {"camera_pose_hint": "unavailable"}

    T = pose["T_world_robot"]
    r_rw = T[:3, :3]
    t_rw = T[:3, 3]
    r_wr = r_rw.T
    t_wr = -r_wr @ t_rw
    r_lw = config.Til[:3, :3].T @ r_wr
    t_lw = config.Til[:3, :3].T @ t_wr - config.Til[:3, :3].T @ config.Til[:3, 3]
    t_cl = config.Tcl[cam_id]
    r_cw = t_cl[:3, :3] @ r_lw
    t_cw = t_cl[:3, :3] @ t_lw + t_cl[:3, 3]

    camera_center = -r_cw.T @ t_cw
    camera_forward = r_cw.T @ np.array([0.0, 0.0, 1.0])
    camera_image_up = r_cw.T @ np.array([0.0, -1.0, 0.0])
    centroid = points_world.mean(axis=0).astype(np.float64, copy=False)
    view = centroid - camera_center
    distance = float(np.linalg.norm(view))
    if distance <= 1e-6:
        return {"camera_pose_hint": "unavailable"}
    view_unit = view / distance
    world_up = np.array([0.0, 0.0, 1.0])
    target_elevation = math.degrees(math.asin(float(np.clip(np.dot(view_unit, world_up), -1.0, 1.0))))
    forward_elevation = math.degrees(math.asin(float(np.clip(np.dot(camera_forward, world_up), -1.0, 1.0))))
    return {
        "camera_pose_hint": "calibrated",
        "camera_center_world": [round(float(v), 3) for v in camera_center],
        "camera_forward_world_unit": [round(float(v), 4) for v in camera_forward],
        "camera_image_up_world_unit": [round(float(v), 4) for v in camera_image_up],
        "object_view_direction_world_unit": [round(float(v), 4) for v in view_unit],
        "object_camera_distance_m": round(distance, 3),
        "object_relative_height_m": round(float(view[2]), 3),
        "object_view_elevation_deg": round(target_elevation, 2),
        "camera_forward_elevation_deg": round(forward_elevation, 2),
    }


def draw_world_up_arrow(image: np.ndarray, hint: dict[str, Any]) -> None:
    unit = hint.get("world_up_image_unit_xy")
    if not isinstance(unit, list) or len(unit) != 2:
        return
    start = (80, 90)
    end = (int(round(start[0] + 42 * float(unit[0]))), int(round(start[1] + 42 * float(unit[1]))))
    cv2.arrowedLine(image, start, end, (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.24)
    cv2.putText(image, "WORLD UP", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2, cv2.LINE_AA)


def build_frame_depth_buffer(
    points_world: np.ndarray,
    pose: dict[str, Any],
    cam_id: int,
    min_depth: float,
    width: int,
    height: int,
) -> np.ndarray:
    uv, depth = project_points(points_world, pose, cam_id, min_depth)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    if len(uv) == 0:
        return depth_buffer
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not np.any(in_img):
        return depth_buffer
    uu = np.clip(np.rint(uv[in_img, 0]).astype(np.int32), 0, width - 1)
    vv = np.clip(np.rint(uv[in_img, 1]).astype(np.int32), 0, height - 1)
    np.minimum.at(depth_buffer, (vv, uu), depth[in_img])
    return depth_buffer


def min_depth_neighborhood(depth_buffer: np.ndarray, uu: np.ndarray, vv: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return depth_buffer[vv, uu]
    height, width = depth_buffer.shape[:2]
    out = np.full(len(uu), np.inf, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        yy = np.clip(vv + dy, 0, height - 1)
        for dx in range(-radius, radius + 1):
            xx = np.clip(uu + dx, 0, width - 1)
            out = np.minimum(out, depth_buffer[yy, xx])
    return out


def lx_section_points(lx_handle, sections: list[dict[str, Any]], frame_id: int) -> np.ndarray | None:
    if frame_id < 0 or frame_id >= len(sections):
        return None
    return read_lx_points(lx_handle, sections[frame_id])


def frame_path(frame_root: Path, cam_id: int, frame_id: int) -> Path:
    return frame_root / f"cam{cam_id}" / f"frame_{frame_id:06d}.jpg"


def priority_mask_path(priority_dir: Path, cam_id: int, frame_id: int, suffix: str) -> Path:
    return priority_dir / "priority" / f"cam{cam_id}_{frame_id:06d}{suffix}.png"


def global_depth_map_path(depth_map_dir: Path, cam_id: int, frame_id: int) -> Path:
    return depth_map_dir / f"cam{cam_id}_{frame_id:06d}_geometry.npz"


def load_global_depth_map(path: Path) -> np.ndarray:
    """Load a pre-rendered full-cloud first-touch depth map."""
    with np.load(path) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
        valid = np.asarray(data.get("valid", depth > 0), dtype=bool)
    return np.where(valid & np.isfinite(depth) & (depth > 0), depth, np.inf)


def choose_frame_pool(
    points: np.ndarray,
    poses: list[dict[str, Any]],
    max_frames: int,
    max_distance: float,
    mode: str,
    min_depth: float,
) -> list[dict[str, Any]]:
    if mode == "projected":
        scored = []
        for pose in poses:
            best = 0
            for cam_id in (0, 1, 2):
                uv, _depth = project_points(points, pose, cam_id, min_depth)
                if len(uv):
                    inside = (uv[:, 0] >= 0) & (uv[:, 0] < config.IMAGE_WIDTH)
                    inside &= (uv[:, 1] >= 0) & (uv[:, 1] < config.IMAGE_HEIGHT)
                    best = max(best, int(inside.sum()))
            if best:
                scored.append((-best, pose))
        scored.sort(key=lambda item: item[0])
        return [pose for _score, pose in scored[:max_frames]]

    centroid = points[:, :3].mean(axis=0)
    scored = []
    for pose in poses:
        dist = float(np.linalg.norm(np.asarray(pose["pos"], dtype=np.float64) - centroid.astype(np.float64)))
        if max_distance > 0 and dist > max_distance:
            continue
        scored.append((dist, pose))
    scored.sort(key=lambda item: item[0])
    return [pose for _dist, pose in scored[:max_frames]]


def choose_source_frame_pool(
    object_id: int,
    support: dict[int, list[int]],
    poses_by_frame: dict[int, dict[str, Any]],
    max_frames: int,
    available_frames: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Return only poses whose raw LiDAR section contributed to the object."""
    return [
        poses_by_frame[frame_id]
        for frame_id in support.get(object_id, [])
        if frame_id in poses_by_frame and (available_frames is None or frame_id in available_frames)
    ][:max_frames]


def pose_for_evidence(row: dict[str, Any], poses_by_frame: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Return the calibration pose that produced this evidence crop."""
    return poses_by_frame[int(row["frame_id"])]


def remember_depth_buffer(
    cache: OrderedDict[tuple[int, int], np.ndarray],
    key: tuple[int, int],
    value: np.ndarray,
    max_entries: int,
) -> None:
    if max_entries <= 0:
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def crop_with_margin(image: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w - 1, x1 + margin)
    y1 = min(h - 1, y1 + margin)
    return image[y0:y1 + 1, x0:x1 + 1].copy(), (x0, y0, x1, y1)


def bbox_from_points(
    uv: np.ndarray,
    width: int,
    height: int,
    percentile: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], float]:
    raw_x0, raw_y0 = np.floor(uv.min(axis=0)).astype(int)
    raw_x1, raw_y1 = np.ceil(uv.max(axis=0)).astype(int)
    raw_bbox = (
        int(max(0, raw_x0)),
        int(max(0, raw_y0)),
        int(min(width - 1, raw_x1)),
        int(min(height - 1, raw_y1)),
    )
    if percentile <= 0:
        return raw_bbox, raw_bbox, 1.0

    lo = max(0.0, min(49.0, percentile))
    hi = 100.0 - lo
    q0 = np.floor(np.percentile(uv, lo, axis=0)).astype(int)
    q1 = np.ceil(np.percentile(uv, hi, axis=0)).astype(int)
    x0 = int(max(0, q0[0]))
    y0 = int(max(0, q0[1]))
    x1 = int(min(width - 1, q1[0]))
    y1 = int(min(height - 1, q1[1]))
    if x1 < x0 or y1 < y0:
        return raw_bbox, raw_bbox, 1.0
    inlier = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    return (x0, y0, x1, y1), raw_bbox, float(inlier.mean()) if len(inlier) else 0.0


def evidence_score(projected_points: int, bbox_area: float, bbox_area_ratio: float, median_depth: float, score_mode: str) -> float:
    depth = max(float(median_depth), 1.0)
    if score_mode == "tight":
        return float(projected_points / depth / math.sqrt(max(bbox_area_ratio, 0.002)))
    return float(projected_points * math.log1p(max(bbox_area, 0.0)) / depth)


def sampled_projection_payload(uv: np.ndarray, depth: np.ndarray, max_samples: int) -> dict[str, Any]:
    if max_samples <= 0 or len(uv) == 0:
        return {}
    if len(uv) <= max_samples:
        take = np.arange(len(uv), dtype=np.int32)
    else:
        take = np.linspace(0, len(uv) - 1, max_samples).round().astype(np.int32)
    uv_take = uv[take].astype(float)
    depth_take = depth[take].astype(float)
    return {
        "projected_uv_samples": [[round(float(x), 3), round(float(y), 3)] for x, y in uv_take],
        "projected_depth_samples": [round(float(z), 4) for z in depth_take],
    }


def make_contact_sheet(rows: list[dict[str, Any]], output_path: Path, thumb_size: int = 180, cols: int = 6) -> None:
    thumbs = []
    labels = []
    for row in rows:
        if int(row.get("rank", 999)) != 1:
            continue
        crop_path = Path(row["crop_path"])
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min(thumb_size / max(w, 1), thumb_size / max(h, 1))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        canvas = np.zeros((thumb_size + 34, thumb_size, 3), dtype=np.uint8)
        y = (thumb_size - resized.shape[0]) // 2
        x = (thumb_size - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        label = f"{row['object_id']} {row.get('semantic_label', '')}"
        cv2.putText(canvas, label[:24], (4, thumb_size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"f{row['frame_id']} c{row['cam_id']}", (4, thumb_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        thumbs.append(canvas)
        labels.append(label)
    if not thumbs:
        return
    rows_count = math.ceil(len(thumbs) / cols)
    sheet = np.zeros((rows_count * (thumb_size + 34), cols * thumb_size, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * (thumb_size + 34):(r + 1) * (thumb_size + 34), c * thumb_size:(c + 1) * thumb_size] = thumb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Dataset root supplying image/img_pos.txt and image/cam_in_ex.txt.")
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--object-ply", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--priority-dir", type=Path, default=None, help="Optional priority/refined mask dir. When set, projected sky pixels are rejected.")
    parser.add_argument("--priority-suffix", default="_priority_refined", help="Priority mask suffix before .png.")
    parser.add_argument("--lx", type=Path, default=None, help="Optional MANIFOLD .lx stream. When set, evidence points must be visible in the same frame section depth buffer.")
    parser.add_argument("--depth-tolerance", type=float, default=0.45, help="Max object-vs-frame depth difference when --lx is enabled.")
    parser.add_argument("--depth-neighborhood", type=int, default=1, help="Pixel radius for local section depth lookup when --lx is enabled.")
    parser.add_argument("--max-depth-cache-entries", type=int, default=48,
                        help="Bound cached frame/camera depth maps; 0 disables caching.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-frame-pool", type=int, default=80)
    parser.add_argument("--max-frame-distance", type=float, default=0.0)
    parser.add_argument("--view-selection", choices=("nearby", "projected"), default="nearby",
                        help="Use exact sampled-point camera visibility before depth gating for review candidates.")
    parser.add_argument("--source-frame-support", type=Path, default=None,
                        help="Optional Superpoint source-frame JSONL. When set, only use frames whose raw section contributed to the object.")
    parser.add_argument(
        "--global-visibility",
        action="store_true",
        help="Select views from all sampled poses and project the whole object, not only its source frame.",
    )
    parser.add_argument(
        "--global-depth-map-dir",
        type=Path,
        default=None,
        help="Pre-rendered full-cloud first-touch maps. Required with --global-visibility.",
    )
    parser.add_argument("--max-points-per-object", type=int, default=2500)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--min-projected-points", type=int, default=20)
    parser.add_argument("--min-bbox-area", type=float, default=1600.0)
    parser.add_argument("--bbox-percentile", type=float, default=0.0, help="Use percentile bbox, e.g. 2 means 2nd-98th percentile. 0 keeps min/max.")
    parser.add_argument("--max-bbox-area-ratio", type=float, default=0.0, help="Reject evidence boxes larger than this image-area ratio when >0.")
    parser.add_argument("--max-sky-ratio", type=float, default=0.0, help="Reject evidence when sky-filtered projected ratio exceeds this threshold. 0 disables ratio rejection.")
    parser.add_argument("--score-mode", choices=["legacy", "tight"], default="legacy")
    parser.add_argument("--crop-margin", type=int, default=48)
    parser.add_argument("--save-projected-samples", type=int, default=0, help="Store up to this many projected uv/depth samples per evidence row for patch-level feature binding.")
    parser.add_argument("--limit-objects", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.global_visibility and args.source_frame_support:
        raise SystemExit("--global-visibility and --source-frame-support are mutually exclusive")
    if args.global_visibility and args.lx:
        raise SystemExit("--global-visibility requires full-cloud depth maps, not frame-local --lx depth")
    if args.global_visibility and args.global_depth_map_dir is None:
        raise SystemExit("--global-visibility requires --global-depth-map-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects = read_jsonl(args.objects_jsonl)
    if args.limit_objects:
        objects = objects[:args.limit_objects]
    object_ids = {int(obj["object_id"]) for obj in objects}
    point_samples = load_object_point_samples(args.object_ply, object_ids, args.max_points_per_object, args.seed)
    if not point_samples:
        raise SystemExit("No object points matched candidate objects.")

    pose_end = args.end
    if pose_end is None:
        all_poses = config.load_img_pos(args.start, None)
        if not all_poses:
            raise SystemExit("No poses loaded from img_pos.txt.")
        pose_end = int(all_poses[-1]["frame_id"])
    all_poses = config.load_img_pos(args.start, pose_end)
    poses = all_poses if args.source_frame_support else [
        p for p in all_poses if int(p["frame_id"]) % args.frame_stride == 0
    ]
    poses_by_frame = {int(p["frame_id"]): p for p in all_poses}
    source_frame_support = read_source_frame_support(args.source_frame_support) if args.source_frame_support else {}
    object_map = {int(obj["object_id"]): obj for obj in objects}
    lx_sections = read_lx_sections(args.lx) if args.lx else []
    lx_handle = args.lx.open("rb") if args.lx else None
    depth_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    try:
        rows = []
        missing_points = []
        failure_counts = Counter()
        objects_without_evidence = []
        for object_id in sorted(object_ids):
            obj = object_map[object_id]
            points = point_samples.get(object_id)
            if points is None or len(points) == 0:
                missing_points.append(object_id)
                failure_counts["missing_points"] += 1
                continue
            if args.source_frame_support:
                frame_pool = choose_source_frame_pool(
                    object_id, source_frame_support, poses_by_frame, args.max_frame_pool,
                    available_source_frames(points, args.min_projected_points),
                )
                frame_selection = "source_support"
            else:
                frame_pool = choose_frame_pool(
                    points, poses, args.max_frame_pool, args.max_frame_distance,
                    args.view_selection, args.min_depth,
                )
                frame_selection = "global_" + args.view_selection if args.global_visibility else args.view_selection
            object_failures = Counter()
            object_attempts = 0
            object_accepted = 0
            if not frame_pool:
                object_failures["empty_frame_pool"] += 1
            obs = []
            for pose in frame_pool:
                frame_id = int(pose["frame_id"])
                frame_points = points[:, :3] if args.global_visibility else points_for_source_frame(points, frame_id)
                if len(frame_points) < args.min_projected_points:
                    object_failures["no_points_from_source_frame"] += 1
                    continue
                frame_section_points: np.ndarray | None = None
                if lx_handle is not None:
                    frame_section_points = lx_section_points(lx_handle, lx_sections, frame_id)
                    if frame_section_points is None or len(frame_section_points) == 0:
                        object_failures["missing_lx_section"] += 1
                        continue
                for cam_id in args.cams:
                    object_attempts += 1
                    img_path = frame_path(args.frame_root, cam_id, frame_id)
                    if not img_path.exists():
                        object_failures["missing_image"] += 1
                        continue
                    uv, depth = project_points(frame_points, pose, cam_id, args.min_depth)
                    if len(uv) < args.min_projected_points:
                        object_failures["low_projected_before_image_filter"] += 1
                        continue
                    w = config.IMAGE_WIDTH
                    h = config.IMAGE_HEIGHT
                    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                    if int(in_img.sum()) < args.min_projected_points:
                        object_failures["low_projected_in_image"] += 1
                        continue
                    uv_in = uv[in_img]
                    depth_in = depth[in_img]

                    depth_filtered_points = 0
                    depth_visible_ratio = 1.0
                    if args.global_depth_map_dir is not None:
                        cache_key = (frame_id, int(cam_id))
                        depth_buffer = depth_cache.get(cache_key)
                        if depth_buffer is None:
                            depth_path = global_depth_map_path(args.global_depth_map_dir, cam_id, frame_id)
                            if not depth_path.exists():
                                object_failures["missing_global_depth_map"] += 1
                                continue
                            depth_buffer = load_global_depth_map(depth_path)
                            remember_depth_buffer(depth_cache, cache_key, depth_buffer, args.max_depth_cache_entries)
                        else:
                            depth_cache.move_to_end(cache_key)
                        uu_depth = np.clip(np.rint(uv_in[:, 0]).astype(np.int32), 0, config.IMAGE_WIDTH - 1)
                        vv_depth = np.clip(np.rint(uv_in[:, 1]).astype(np.int32), 0, config.IMAGE_HEIGHT - 1)
                        local_depth = min_depth_neighborhood(depth_buffer, uu_depth, vv_depth, args.depth_neighborhood)
                        depth_keep = np.isfinite(local_depth) & (np.abs(depth_in - local_depth) <= args.depth_tolerance)
                        depth_filtered_points = int((~depth_keep).sum())
                        depth_visible_ratio = float(depth_keep.mean()) if len(depth_keep) else 0.0
                        uv_in = uv_in[depth_keep]
                        depth_in = depth_in[depth_keep]
                        if len(uv_in) < args.min_projected_points:
                            object_failures["low_projected_after_depth_filter"] += 1
                            continue
                    elif frame_section_points is not None:
                        cache_key = (frame_id, int(cam_id))
                        depth_buffer = depth_cache.get(cache_key)
                        if depth_buffer is None:
                            depth_buffer = build_frame_depth_buffer(
                                frame_section_points,
                                pose,
                                cam_id,
                                args.min_depth,
                                config.IMAGE_WIDTH,
                                config.IMAGE_HEIGHT,
                            )
                            remember_depth_buffer(depth_cache, cache_key, depth_buffer, args.max_depth_cache_entries)
                        else:
                            depth_cache.move_to_end(cache_key)
                        uu_depth = np.clip(np.rint(uv_in[:, 0]).astype(np.int32), 0, config.IMAGE_WIDTH - 1)
                        vv_depth = np.clip(np.rint(uv_in[:, 1]).astype(np.int32), 0, config.IMAGE_HEIGHT - 1)
                        local_depth = min_depth_neighborhood(depth_buffer, uu_depth, vv_depth, args.depth_neighborhood)
                        depth_keep = np.isfinite(local_depth) & (np.abs(depth_in - local_depth) <= args.depth_tolerance)
                        depth_filtered_points = int((~depth_keep).sum())
                        depth_visible_ratio = float(depth_keep.mean()) if len(depth_keep) else 0.0
                        uv_in = uv_in[depth_keep]
                        depth_in = depth_in[depth_keep]
                        if len(uv_in) < args.min_projected_points:
                            object_failures["low_projected_after_depth_filter"] += 1
                            continue

                    sky_filtered_points = 0
                    sky_ratio = 0.0
                    if args.priority_dir is not None:
                        pri_path = priority_mask_path(args.priority_dir, cam_id, frame_id, args.priority_suffix)
                        if not pri_path.exists():
                            object_failures["missing_priority_mask"] += 1
                            continue
                        priority = cv2.imread(str(pri_path), cv2.IMREAD_GRAYSCALE)
                        if priority is None:
                            object_failures["priority_mask_read_failed"] += 1
                            continue
                        if priority.shape[:2] != (config.IMAGE_HEIGHT, config.IMAGE_WIDTH):
                            priority = cv2.resize(priority, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
                        uu = np.clip(np.rint(uv_in[:, 0]).astype(np.int32), 0, config.IMAGE_WIDTH - 1)
                        vv = np.clip(np.rint(uv_in[:, 1]).astype(np.int32), 0, config.IMAGE_HEIGHT - 1)
                        non_sky = priority[vv, uu] != 6
                        sky_filtered_points = int((~non_sky).sum())
                        sky_ratio = sky_filtered_points / max(len(non_sky), 1)
                        if args.max_sky_ratio > 0 and sky_ratio > args.max_sky_ratio:
                            object_failures["sky_ratio_too_high"] += 1
                            continue
                        uv_in = uv_in[non_sky]
                        depth_in = depth_in[non_sky]
                        if len(uv_in) < args.min_projected_points:
                            object_failures["low_projected_after_sky_filter"] += 1
                            continue

                    bbox, raw_bbox, bbox_inlier_ratio = bbox_from_points(uv_in, w, h, args.bbox_percentile)
                    x0, y0, x1, y1 = bbox
                    rx0, ry0, rx1, ry1 = raw_bbox
                    area = float(max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1))
                    raw_area = float(max(0, rx1 - rx0 + 1) * max(0, ry1 - ry0 + 1))
                    area_ratio = area / float(max(1, w * h))
                    if args.max_bbox_area_ratio > 0 and area_ratio > args.max_bbox_area_ratio:
                        object_failures["bbox_too_large"] += 1
                        continue
                    if area < args.min_bbox_area:
                        object_failures["bbox_too_small"] += 1
                        continue
                    score = evidence_score(len(uv_in), area, area_ratio, float(np.median(depth_in)), args.score_mode)
                    obs.append({
                        "object_id": object_id,
                        "frame_id": frame_id,
                        "cam_id": int(cam_id),
                        "image_path": str(img_path),
                        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                        "raw_bbox_xyxy": [int(rx0), int(ry0), int(rx1), int(ry1)],
                        "projected_points": int(len(uv_in)),
                        "sample_points": int(len(frame_points)),
                        "bbox_area": area,
                        "raw_bbox_area": raw_area,
                        "bbox_area_ratio": area_ratio,
                        "bbox_inlier_ratio": bbox_inlier_ratio,
                        "median_depth": float(np.median(depth_in)),
                        "sky_filtered_points": sky_filtered_points,
                        "sky_filtered_ratio": sky_ratio,
                        "depth_filtered_points": depth_filtered_points,
                        "depth_visible_ratio": depth_visible_ratio,
                        "score": score,
                        "frame_selection": frame_selection,
                        "uv": uv_in,
                        "depth": depth_in,
                        "points": frame_points,
                    })
            obs.sort(key=lambda row: row["score"], reverse=True)
            for rank, row in enumerate(obs[:args.top_k], 1):
                image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
                if image is None:
                    object_failures["crop_image_read_failed"] += 1
                    continue
                bbox = tuple(row["bbox_xyxy"])
                crop, crop_bbox = crop_with_margin(image, bbox, args.crop_margin)
                overlay = image.copy()
                evidence_points = row["points"]
                evidence_pose = pose_for_evidence(row, poses_by_frame)
                up_hint = world_up_image_hint(evidence_points, evidence_pose, row["cam_id"])
                pose_hint = camera_pose_context(evidence_points, evidence_pose, row["cam_id"])
                draw_world_up_arrow(overlay, up_hint)
                x0, y0, x1, y1 = bbox
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 3)
                for uv in row["uv"][::max(1, len(row["uv"]) // 400)]:
                    cv2.circle(overlay, (int(round(uv[0])), int(round(uv[1]))), 2, (0, 0, 255), -1)
                object_dir = args.output_dir / "objects" / str(object_id)
                object_dir.mkdir(parents=True, exist_ok=True)
                stem = f"obj{object_id}_rank{rank}_cam{row['cam_id']}_frame{row['frame_id']:06d}"
                crop_path = object_dir / f"{stem}_crop.jpg"
                overlay_path = object_dir / f"{stem}_overlay.jpg"
                cv2.imwrite(str(crop_path), crop)
                cv2.imwrite(str(overlay_path), overlay)
                out = {
                    **{k: v for k, v in row.items() if k not in {"uv", "depth", "points"}},
                    "rank": rank,
                    "crop_path": str(crop_path),
                    "overlay_path": str(overlay_path),
                    "crop_bbox_xyxy": list(crop_bbox),
                    **up_hint,
                    **pose_hint,
                    **sampled_projection_payload(row["uv"], row["depth"], args.save_projected_samples),
                    "semantic_label": obj.get("semantic_label", ""),
                    "scene_context": obj.get("scene_context", ""),
                    "downstream_stage": obj.get("downstream_stage", ""),
                    "review_priority": obj.get("review_priority", ""),
                    "dino_prompt_group": obj.get("dino_prompt_group", ""),
                    "dino_prompts": obj.get("dino_prompts", []),
                }
                rows.append(out)
                object_accepted += 1

            if object_accepted == 0:
                failure_counts.update(object_failures)
                objects_without_evidence.append({
                    "object_id": object_id,
                    "semantic_label": obj.get("semantic_label", ""),
                    "candidate_label": obj.get("candidate_label", ""),
                    "dino_prompt_group": obj.get("dino_prompt_group", ""),
                    "attempts": object_attempts,
                    "top_failure_reasons": dict(object_failures.most_common(5)),
                })
    finally:
        if lx_handle is not None:
            lx_handle.close()

    manifest = args.output_dir / "object_image_evidence.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    make_contact_sheet(rows, args.output_dir / "object_image_evidence_contact.jpg")
    report = {
        "objects_jsonl": str(args.objects_jsonl),
        "object_ply": str(args.object_ply),
        "frame_root": str(args.frame_root),
        "calibration_file": str(config.CALIB_FILE),
        "img_pos_file": str(config.IMG_POS_FILE),
        "output_dir": str(args.output_dir),
        "candidate_objects": len(objects),
        "objects_with_points": len(point_samples),
        "objects_with_evidence": len(set(int(row["object_id"]) for row in rows)),
        "evidence_rows": len(rows),
        "missing_point_objects": missing_points,
        "objects_without_evidence": objects_without_evidence,
        "failure_counts_for_objects_without_evidence": dict(failure_counts),
        "params": {
            "frame_range": [args.start, pose_end],
            "frame_stride": args.frame_stride,
            "cams": args.cams,
            "top_k": args.top_k,
            "max_frame_pool": args.max_frame_pool,
            "max_frame_distance": args.max_frame_distance,
            "view_selection": args.view_selection,
            "source_frame_support": str(args.source_frame_support) if args.source_frame_support else "",
            "global_visibility": args.global_visibility,
            "global_depth_map_dir": str(args.global_depth_map_dir) if args.global_depth_map_dir else "",
            "max_points_per_object": args.max_points_per_object,
            "min_projected_points": args.min_projected_points,
            "min_bbox_area": args.min_bbox_area,
            "bbox_percentile": args.bbox_percentile,
            "max_bbox_area_ratio": args.max_bbox_area_ratio,
            "priority_dir": str(args.priority_dir) if args.priority_dir else "",
            "priority_suffix": args.priority_suffix,
            "max_sky_ratio": args.max_sky_ratio,
            "lx": str(args.lx) if args.lx else "",
            "depth_tolerance": args.depth_tolerance,
            "depth_neighborhood": args.depth_neighborhood,
            "max_depth_cache_entries": args.max_depth_cache_entries,
            "score_mode": args.score_mode,
            "save_projected_samples": args.save_projected_samples,
        },
        "label_counts": dict(Counter(row.get("semantic_label", "") for row in rows if row.get("rank") == 1)),
    }
    (args.output_dir / "object_image_evidence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
