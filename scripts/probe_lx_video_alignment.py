#!/usr/bin/env python3
"""Probe MANIFOLD .lx section to video-frame alignment.

This is a dataset QA tool, not a production projector.  It overlays projected
LiDAR rings from selected .lx sections onto candidate video frames with several
offsets.  The output makes frame-sync mistakes visible before we build priority
masks, targets, or reverse-depth guidance on top of a bad image cache.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from project_priority_masks_to_lx import (
    read_lx_points,
    read_lx_sections,
    transform_world_to_lidar,
    zbuffer_visible,
)


def undistort_maps(cam_id: int) -> tuple[np.ndarray, np.ndarray]:
    params = config.CAMERA_PARAMS[cam_id]
    return cv2.fisheye.initUndistortRectifyMap(
        params["K"],
        params["D"],
        np.eye(3),
        params["K"],
        (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        cv2.CV_16SC2,
    )


def project_points(points_world: np.ndarray, pose: dict[str, Any], cam_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_lidar = transform_world_to_lidar(points_world, pose)
    t_cl = config.Tcl[cam_id]
    p_cam = (t_cl[:3, :3] @ p_lidar.T + t_cl[:3, 3:]).T
    z = p_cam[:, 2]
    valid = z > 0.1
    if not np.any(valid):
        return np.empty(0), np.empty(0), np.empty(0)
    uv_h = (config.CAMERA_PARAMS[cam_id]["K"] @ p_cam[valid].T).T
    return uv_h[:, 0] / uv_h[:, 2], uv_h[:, 1] / uv_h[:, 2], z[valid].astype(np.float32)


def visible_pixels(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_img):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    local_idx = np.arange(len(u), dtype=np.int32)[in_img]
    uu = np.clip(np.rint(u[in_img]).astype(np.int32), 0, width - 1)
    vv = np.clip(np.rint(v[in_img]).astype(np.int32), 0, height - 1)
    depths = z[in_img].astype(np.float32)
    keep = zbuffer_visible(local_idx, uu, vv, depths, width)
    return uu[keep], vv[keep], depths[keep]


def edge_hit_ratio(image_bgr: np.ndarray, uu: np.ndarray, vv: np.ndarray, dilation_px: int) -> float:
    if len(uu) == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    if dilation_px > 1:
        kernel = np.ones((dilation_px, dilation_px), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel)
    return float((edges[vv, uu] > 0).mean())


def draw_overlay(image_bgr: np.ndarray, uu: np.ndarray, vv: np.ndarray, title: str, dot_px: int) -> np.ndarray:
    overlay = image_bgr.copy()
    dot = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    if len(uu):
        dot[vv, uu] = 255
        if dot_px > 1:
            dot = cv2.dilate(dot, np.ones((dot_px, dot_px), dtype=np.uint8))
        overlay[dot > 0] = (0, 255, 0)
    thumb = cv2.resize(overlay, (400, 324))
    cv2.putText(thumb, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(thumb, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


def read_frame_opencv_index(cap: cv2.VideoCapture, frame_id: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, raw = cap.read()
    if not ok or raw is None:
        return None
    return raw


def read_frame_ffmpeg_time(video_path: str, rel_ts: float) -> np.ndarray | None:
    """Read one video frame with the same timestamp-style access as extract_frames.py."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{rel_ts:.4f}",
        "-i",
        video_path,
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    data = np.frombuffer(result.stdout, dtype=np.uint8)
    raw = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return raw


def read_candidate_frame(
    cap: cv2.VideoCapture,
    video_path: str,
    frame_id: int,
    offset: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, int, float]:
    video_idx = int(round(frame_id * args.index_scale + args.index_shift + offset))
    rel_ts = float(video_idx) * float(args.time_scale)
    if args.read_mode == "opencv-index":
        return read_frame_opencv_index(cap, video_idx), video_idx, rel_ts
    if args.read_mode == "ffmpeg-time":
        return read_frame_ffmpeg_time(video_path, rel_ts), video_idx, rel_ts
    raise ValueError(f"Unsupported read mode: {args.read_mode}")


def probe_one(
    lx_handle,
    section: dict[str, Any],
    pose: dict[str, Any],
    frame_id: int,
    cam_id: int,
    offsets: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    points = read_lx_points(lx_handle, section)
    u, v, z = project_points(points, pose, cam_id)
    cap = cv2.VideoCapture(config.VIDEO_FILES[cam_id])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video cam{cam_id}: {config.VIDEO_FILES[cam_id]}")
    map1, map2 = undistort_maps(cam_id)
    rows = []
    panels = []
    for offset in offsets:
        video_idx = int(round(frame_id * args.index_scale + args.index_shift + offset))
        if video_idx < 0:
            continue
        raw, video_idx, rel_ts = read_candidate_frame(
            cap,
            config.VIDEO_FILES[cam_id],
            frame_id,
            offset,
            args,
        )
        if raw is None:
            continue
        image = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
        uu, vv, _depth = visible_pixels(u, v, z, image.shape[1], image.shape[0])
        edge_hit = edge_hit_ratio(image, uu, vv, args.edge_dilation_px)
        row = {
            "frame_id": int(frame_id),
            "cam_id": int(cam_id),
            "video_idx": video_idx,
            "video_rel_ts": rel_ts,
            "offset": int(offset),
            "visible": int(len(uu)),
            "edge_hit": edge_hit,
        }
        rows.append(row)
        if len(panels) < args.max_panels_per_probe:
            title = f"f={frame_id} c={cam_id} v={video_idx} {args.read_mode} edge={edge_hit:.3f}"
            panels.append(draw_overlay(image, uu, vv, title, args.dot_px))
    cap.release()
    best = max(rows, key=lambda r: r["edge_hit"]) if rows else None
    return {"frame_id": int(frame_id), "cam_id": int(cam_id), "best": best, "rows": rows, "panels": panels}


def write_sheet(panels: list[np.ndarray], output: Path, cols: int) -> None:
    if not panels:
        return
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--offsets", type=int, nargs="+", default=[-1200, -1000, -800, -600, -400, -200, 0, 200, 400])
    parser.add_argument("--edge-dilation-px", type=int, default=9)
    parser.add_argument("--dot-px", type=int, default=7)
    parser.add_argument("--sheet-cols", type=int, default=3)
    parser.add_argument("--max-panels-per-probe", type=int, default=9)
    parser.add_argument("--read-mode", choices=["opencv-index", "ffmpeg-time"], default="opencv-index")
    parser.add_argument("--index-scale", type=float, default=1.0, help="video_idx = frame_id * index_scale + index_shift + offset")
    parser.add_argument("--index-shift", type=float, default=0.0, help="video_idx = frame_id * index_scale + index_shift + offset")
    parser.add_argument("--time-scale", type=float, default=0.1, help="ffmpeg timestamp seconds per video_idx in ffmpeg-time mode")
    args = parser.parse_args()

    sections = read_lx_sections(args.lx_file)
    poses = {row["frame_id"]: row for row in config.load_img_pos(min(args.frames), max(args.frames))}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    all_panels = []
    with args.lx_file.open("rb") as lx_handle:
        for frame_id in args.frames:
            if frame_id >= len(sections) or frame_id not in poses:
                continue
            for cam_id in args.cams:
                result = probe_one(lx_handle, sections[frame_id], poses[frame_id], frame_id, cam_id, args.offsets, args)
                all_panels.extend(result.pop("panels"))
                reports.append(result)

    report = {
        "lx_file": str(args.lx_file),
        "frames": args.frames,
        "cams": args.cams,
        "offsets": args.offsets,
        "read_mode": args.read_mode,
        "index_scale": args.index_scale,
        "index_shift": args.index_shift,
        "time_scale": args.time_scale,
        "calib_file": config.CALIB_FILE,
        "video_dir": config.VIDEO_DIR,
        "reports": reports,
    }
    (args.output_dir / "alignment_probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_sheet(all_panels, args.output_dir / "alignment_probe_sheet.jpg", args.sheet_cols)
    print(json.dumps({k: report[k] for k in ("frames", "cams", "offsets", "read_mode")}, ensure_ascii=False, indent=2))
    for row in reports:
        print(json.dumps({"frame_id": row["frame_id"], "cam_id": row["cam_id"], "best": row["best"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
