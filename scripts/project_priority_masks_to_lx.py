#!/usr/bin/env python3
"""Project priority segmentation masks onto MANIFOLD .lx sections.

Inputs:
  - MANIFOLD .lx stream sections, already in world coordinates
  - undistorted frame JPGs from extract_undistorted_frames_jpeg.py
  - priority PNGs from segment_priority_classes.py

Outputs:
  - priority_points.ply: all visible non-sky points, colored by priority class
  - residual_points_rgb.ply: visible non-sky points with priority == background
  - report JSON with class counts and projection statistics

The sky class is hard-filtered before any point is exported. This keeps sky
pixels from contaminating residual target clustering.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


LX_HEADER_SIZE = 48
LX_COUNT_SIZE = 4
LX_POINT_SIZE = 16

PRIORITY_NAMES = {
    0: "residual",
    1: "ground",
    2: "wall",
    3: "grass",
    4: "car",
    5: "railing",
    6: "sky",
}

PRIORITY_COLORS = {
    0: (128, 128, 128),
    1: (196, 168, 112),
    2: (120, 150, 180),
    3: (80, 160, 80),
    4: (235, 90, 80),
    5: (240, 210, 60),
    6: (90, 170, 235),
}


def read_lx_sections(lx_path: Path) -> list[dict]:
    sections = []
    file_size = os.path.getsize(lx_path)
    offset = 0
    section_idx = 0
    with lx_path.open("rb") as f:
        while offset + LX_HEADER_SIZE + LX_COUNT_SIZE <= file_size:
            f.seek(offset + LX_HEADER_SIZE)
            count_raw = f.read(LX_COUNT_SIZE)
            if len(count_raw) < LX_COUNT_SIZE:
                break
            count = struct.unpack("<I", count_raw)[0]
            if count == 0 or count > 50_000_000:
                break
            data_offset = offset + LX_HEADER_SIZE + LX_COUNT_SIZE
            next_offset = data_offset + count * LX_POINT_SIZE
            if next_offset > file_size + 16:
                break
            sections.append({
                "index": section_idx,
                "offset": offset,
                "data_offset": data_offset,
                "count": count,
            })
            offset = next_offset
            section_idx += 1
    return sections


def read_lx_points(handle, section: dict) -> np.ndarray:
    handle.seek(section["data_offset"])
    raw = handle.read(section["count"] * LX_POINT_SIZE)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("marker", "<u4")])
    data = np.frombuffer(raw, dtype=dtype)
    points = np.empty((len(data), 3), dtype=np.float32)
    points[:, 0] = data["x"]
    points[:, 1] = data["y"]
    points[:, 2] = data["z"]
    return points


def priority_path(base: Path, cam_id: int, frame_id: int) -> Path:
    return base / "priority" / f"cam{cam_id}_{frame_id:06d}_priority.png"


def frame_path(base: Path, cam_id: int, frame_id: int) -> Path:
    return base / f"cam{cam_id}" / f"frame_{frame_id:06d}.jpg"


def zbuffer_visible(point_indices: np.ndarray, uu: np.ndarray, vv: np.ndarray, depths: np.ndarray, width: int) -> np.ndarray:
    if len(point_indices) == 0:
        return np.zeros(0, dtype=bool)
    pixel_idx = vv.astype(np.int64) * int(width) + uu.astype(np.int64)
    order = np.lexsort((depths, pixel_idx))
    sorted_pixel_idx = pixel_idx[order]
    first = np.r_[True, sorted_pixel_idx[1:] != sorted_pixel_idx[:-1]]
    keep_order = order[first]
    keep = np.zeros(len(point_indices), dtype=bool)
    keep[keep_order] = True
    return keep


def transform_world_to_lidar(points_world: np.ndarray, pose: dict) -> np.ndarray:
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


def project_frame(points: np.ndarray, pose: dict, frame_id: int, args: argparse.Namespace) -> dict:
    labels = np.full(len(points), 255, dtype=np.uint8)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    best_depth = np.full(len(points), np.inf, dtype=np.float32)
    visible_non_sky = np.zeros(len(points), dtype=bool)
    p_lidar = transform_world_to_lidar(points, pose)

    masks_found = 0
    masks_missing = 0
    sky_samples = 0
    zbuffer_kept = 0

    for cam_id in args.cams:
        mask_file = priority_path(args.priority_dir, cam_id, frame_id)
        img_file = frame_path(args.frame_root, cam_id, frame_id)
        if not mask_file.exists() or not img_file.exists():
            masks_missing += 1
            continue
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        if mask is None or image is None:
            masks_missing += 1
            continue
        masks_found += 1
        h, w = mask.shape[:2]

        t_cl = config.Tcl[cam_id]
        p_cam = (t_cl[:3, :3] @ p_lidar.T + t_cl[:3, 3:]).T
        z = p_cam[:, 2]
        valid = z > args.min_depth
        if not np.any(valid):
            continue

        valid_idx = np.where(valid)[0]
        uv_h = (config.CAMERA_PARAMS[cam_id]["K"] @ p_cam[valid].T).T
        u = uv_h[:, 0] / uv_h[:, 2]
        v = uv_h[:, 1] / uv_h[:, 2]
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_img):
            continue

        idx = valid_idx[in_img]
        uu = np.clip(np.rint(u[in_img]).astype(np.int32), 0, w - 1)
        vv = np.clip(np.rint(v[in_img]).astype(np.int32), 0, h - 1)
        depths = z[valid][in_img].astype(np.float32)
        if args.zbuffer:
            keep = zbuffer_visible(idx, uu, vv, depths, w)
            idx, uu, vv, depths = idx[keep], uu[keep], vv[keep], depths[keep]
        if len(idx) == 0:
            continue
        zbuffer_kept += int(len(idx))

        sampled = mask[vv, uu].astype(np.uint8)
        non_sky = sampled != 6
        sky_samples += int((~non_sky).sum())
        if not np.any(non_sky):
            continue
        idx, uu, vv, depths, sampled = idx[non_sky], uu[non_sky], vv[non_sky], depths[non_sky], sampled[non_sky]
        rgb = image[vv, uu][:, ::-1]

        closer = depths < best_depth[idx]
        if not np.any(closer):
            continue
        out_idx = idx[closer]
        labels[out_idx] = sampled[closer]
        colors[out_idx] = rgb[closer]
        best_depth[out_idx] = depths[closer]
        visible_non_sky[out_idx] = True

    return {
        "labels": labels,
        "colors": colors,
        "visible_non_sky": visible_non_sky,
        "masks_found": masks_found,
        "masks_missing": masks_missing,
        "sky_samples": sky_samples,
        "zbuffer_kept": zbuffer_kept,
    }


def append_binary_xyzrgb_label(handle, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("priority", "u1"),
    ])
    data = np.empty(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["red"] = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"] = colors[:, 2]
    data["priority"] = labels
    handle.write(data.tobytes())


def write_binary_ply_from_body(body_path: Path, output_path: Path, vertex_count: int) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar priority\n"
        "end_header\n"
    ).encode("ascii")
    with output_path.open("wb") as out, body_path.open("rb") as body:
        out.write(header)
        while True:
            chunk = body.read(16 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def color_by_priority(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for label, color in PRIORITY_COLORS.items():
        colors[labels == label] = color
    return colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lx", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--priority-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--zbuffer", action="store_true", default=True)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sections = read_lx_sections(args.lx)
    poses = {row["frame_id"]: row for row in config.load_img_pos(args.start, args.end)}
    if args.end is None:
        args.end = min(len(sections), max(poses) + 1 if poses else 0) - 1
    frame_ids = [i for i in range(args.start, args.end + 1, max(args.stride, 1)) if i < len(sections) and i in poses]
    if not frame_ids:
        raise SystemExit("No overlapping .lx sections, img_pos rows, and frame range.")

    priority_body = Path(tempfile.mkstemp(prefix="priority_points_", suffix=".bin")[1])
    residual_body = Path(tempfile.mkstemp(prefix="residual_points_", suffix=".bin")[1])
    priority_count = 0
    residual_count = 0
    raw_points = 0
    visible_points = 0
    priority_hist = Counter()
    per_frame = []

    with args.lx.open("rb") as lx_f, priority_body.open("wb") as pri_f, residual_body.open("wb") as res_f:
        for n, frame_id in enumerate(frame_ids, start=1):
            points = read_lx_points(lx_f, sections[frame_id])
            raw_points += int(len(points))
            result = project_frame(points, poses[frame_id], frame_id, args)
            labels = result["labels"]
            colors = result["colors"]
            visible = result["visible_non_sky"]
            visible_points += int(visible.sum())

            valid_labels = labels[visible]
            hist = Counter(int(x) for x in valid_labels.tolist())
            priority_hist.update(hist)

            priority_mask = visible & (labels != 0) & (labels != 255)
            residual_mask = visible & (labels == 0)
            if np.any(priority_mask):
                pri_labels = labels[priority_mask]
                append_binary_xyzrgb_label(
                    pri_f,
                    points[priority_mask],
                    color_by_priority(pri_labels),
                    pri_labels,
                )
                priority_count += int(priority_mask.sum())
            if np.any(residual_mask):
                append_binary_xyzrgb_label(
                    res_f,
                    points[residual_mask],
                    colors[residual_mask],
                    labels[residual_mask],
                )
                residual_count += int(residual_mask.sum())

            row = {
                "frame_id": frame_id,
                "raw_points": int(len(points)),
                "visible_non_sky": int(visible.sum()),
                "priority_points": int(priority_mask.sum()),
                "residual_points": int(residual_mask.sum()),
                "priority_counts": {PRIORITY_NAMES.get(k, str(k)): int(v) for k, v in sorted(hist.items())},
                "masks_found": result["masks_found"],
                "masks_missing": result["masks_missing"],
                "sky_samples": result["sky_samples"],
                "zbuffer_kept": result["zbuffer_kept"],
            }
            per_frame.append(row)
            if n == 1 or n % args.progress_every == 0:
                print(json.dumps({"processed": n, **row}, ensure_ascii=False))

    priority_ply = args.output_dir / "priority_points.ply"
    residual_ply = args.output_dir / "residual_points_rgb.ply"
    write_binary_ply_from_body(priority_body, priority_ply, priority_count)
    write_binary_ply_from_body(residual_body, residual_ply, residual_count)
    priority_body.unlink(missing_ok=True)
    residual_body.unlink(missing_ok=True)

    report = {
        "lx": str(args.lx),
        "frame_root": str(args.frame_root),
        "priority_dir": str(args.priority_dir),
        "output_dir": str(args.output_dir),
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "cams": args.cams,
        "frame_count": len(frame_ids),
        "raw_points": raw_points,
        "visible_non_sky_points": visible_points,
        "visible_non_sky_ratio": visible_points / max(raw_points, 1),
        "priority_points": priority_count,
        "residual_points": residual_count,
        "priority_ratio_of_visible": priority_count / max(visible_points, 1),
        "residual_ratio_of_visible": residual_count / max(visible_points, 1),
        "priority_counts": {PRIORITY_NAMES.get(k, str(k)): int(v) for k, v in sorted(priority_hist.items())},
        "priority_ply": str(priority_ply),
        "residual_ply": str(residual_ply),
        "elapsed_sec": time.time() - t0,
        "per_frame": per_frame,
    }
    (args.output_dir / "priority_projection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in [
        "frame_count", "raw_points", "visible_non_sky_points", "priority_points",
        "residual_points", "priority_counts", "elapsed_sec",
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
