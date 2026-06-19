#!/usr/bin/env python3
"""Calibrate MANIFOLD .lx section id to camera video-frame id mapping.

This is a dataset gate for image-based semantic projection.  It searches video
frames near selected .lx sections, scores LiDAR ring projections against image
edges, and writes an explicit sync report.  If the best matches cannot be
explained by a stable affine mapping per camera, the dataset is rejected for
production semantic projection until a better timing source is provided.

The score is intentionally simple and inspectable:

- project same-section LiDAR points into an undistorted candidate image;
- compute Canny edges and a distance transform to nearest edge;
- reward projected points that lie near image edges;
- keep visible point count as supporting evidence, not as the main score.

Outputs:

- sync_calibration_report.json
- sync_candidates.jsonl
- sync_best.jsonl
- sync_fit.json
- sync_probe_sheet.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CandidateScore:
    frame_id: int
    cam_id: int
    video_idx: int
    offset: int
    visible: int
    edge_hit: float
    edge_distance_mean: float
    edge_distance_p50: float
    score: float


def parse_int_range(text: str) -> list[int]:
    values: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [int(x) for x in chunk.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"Bad range chunk: {chunk}")
            start, end = parts[0], parts[1]
            step = parts[2] if len(parts) == 3 else 1
            if step == 0:
                raise ValueError(f"Zero step in range chunk: {chunk}")
            stop = end + (1 if step > 0 else -1)
            values.extend(range(start, stop, step))
        else:
            values.append(int(chunk))
    return sorted(set(values))


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


def project_points(
    points_world: np.ndarray,
    pose: dict[str, Any],
    cam_id: int,
    min_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_lidar = transform_world_to_lidar(points_world, pose)
    t_cl = config.Tcl[cam_id]
    p_cam = (t_cl[:3, :3] @ p_lidar.T + t_cl[:3, 3:]).T
    z = p_cam[:, 2]
    valid = z > float(min_depth)
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


def read_frame(cap: cv2.VideoCapture, video_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_idx))
    ok, raw = cap.read()
    if not ok or raw is None:
        return None
    return raw


def edge_distance_score(
    image_bgr: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    edge_dilation_px: int,
    distance_sigma: float,
) -> tuple[float, float, float, float]:
    if len(uu) == 0:
        return 0.0, float("inf"), float("inf"), 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    if edge_dilation_px > 1:
        edges_hit = cv2.dilate(edges, np.ones((edge_dilation_px, edge_dilation_px), dtype=np.uint8))
    else:
        edges_hit = edges
    hit = float((edges_hit[vv, uu] > 0).mean())
    # distanceTransform measures distance to zero pixels, so invert edge mask:
    # edge pixels become zero distance, non-edge pixels are positive.
    non_edge = (edges == 0).astype(np.uint8)
    dist = cv2.distanceTransform(non_edge, cv2.DIST_L2, 3)
    sampled = dist[vv, uu].astype(np.float32)
    mean_dist = float(sampled.mean())
    p50_dist = float(np.percentile(sampled, 50))
    sigma = max(float(distance_sigma), 1e-3)
    proximity = float(np.exp(-sampled / sigma).mean())
    score = 0.65 * proximity + 0.35 * hit
    return hit, mean_dist, p50_dist, score


def draw_panel(
    image_bgr: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    score: CandidateScore,
    dot_px: int,
) -> np.ndarray:
    overlay = image_bgr.copy()
    dot = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    if len(uu):
        dot[vv, uu] = 255
        if dot_px > 1:
            dot = cv2.dilate(dot, np.ones((dot_px, dot_px), dtype=np.uint8))
        overlay[dot > 0] = (0, 255, 0)
    thumb = cv2.resize(overlay, (420, 340))
    title = (
        f"s={score.frame_id} c={score.cam_id} v={score.video_idx} "
        f"off={score.offset} score={score.score:.3f} d={score.edge_distance_mean:.1f}"
    )
    cv2.putText(thumb, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(thumb, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


def score_candidates_for_probe(
    lx_handle,
    section: dict[str, Any],
    pose: dict[str, Any],
    frame_id: int,
    cam_id: int,
    offsets: list[int],
    args: argparse.Namespace,
) -> tuple[list[CandidateScore], list[np.ndarray]]:
    points = read_lx_points(lx_handle, section)
    u, v, z = project_points(points, pose, cam_id, args.min_depth)
    cap = cv2.VideoCapture(config.VIDEO_FILES[cam_id])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video cam{cam_id}: {config.VIDEO_FILES[cam_id]}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    map1, map2 = undistort_maps(cam_id)
    scores: list[CandidateScore] = []
    panel_payloads: list[tuple[CandidateScore, np.ndarray, np.ndarray, np.ndarray]] = []
    for offset in offsets:
        video_idx = int(round(frame_id * args.index_scale + args.index_shift + offset))
        if video_idx < 0 or (frame_count and video_idx >= frame_count):
            continue
        raw = read_frame(cap, video_idx)
        if raw is None:
            continue
        image = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
        uu, vv, _depth = visible_pixels(u, v, z, image.shape[1], image.shape[0])
        hit, mean_dist, p50_dist, score = edge_distance_score(
            image,
            uu,
            vv,
            args.edge_dilation_px,
            args.distance_sigma,
        )
        item = CandidateScore(
            frame_id=int(frame_id),
            cam_id=int(cam_id),
            video_idx=int(video_idx),
            offset=int(offset),
            visible=int(len(uu)),
            edge_hit=hit,
            edge_distance_mean=mean_dist,
            edge_distance_p50=p50_dist,
            score=score,
        )
        scores.append(item)
        panel_payloads.append((item, image, uu, vv))
    cap.release()
    scores.sort(key=lambda x: x.score, reverse=True)
    top = {(s.video_idx, s.offset) for s in scores[: args.panels_per_probe]}
    # Always include the direct frame-id candidate when present.
    top.add((int(round(frame_id * args.index_scale + args.index_shift)), 0))
    panels: list[np.ndarray] = []
    for item, image, uu, vv in panel_payloads:
        if (item.video_idx, item.offset) in top and len(panels) < args.panels_per_probe + 1:
            panels.append(draw_panel(image, uu, vv, item, args.dot_px))
    return scores, panels


def fit_affine_mapping(
    best_rows: list[dict[str, Any]],
    max_rmse: float,
    max_abs_residual: float,
    min_samples: int,
) -> dict[str, Any]:
    if len(best_rows) < min_samples:
        return {
            "status": "insufficient_samples",
            "sample_count": len(best_rows),
            "accepted": False,
        }
    x = np.asarray([row["frame_id"] for row in best_rows], dtype=np.float64)
    y = np.asarray([row["video_idx"] for row in best_rows], dtype=np.float64)
    if np.ptp(x) <= 0:
        return {"status": "degenerate_samples", "sample_count": len(best_rows), "accepted": False}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    residual = y - pred
    rmse = float(np.sqrt(np.mean(residual * residual)))
    max_abs = float(np.max(np.abs(residual)))
    monotonic = bool(slope > 0)
    accepted = monotonic and rmse <= max_rmse and max_abs <= max_abs_residual
    return {
        "status": "accepted" if accepted else "rejected_unstable_fit",
        "accepted": accepted,
        "sample_count": len(best_rows),
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": rmse,
        "max_abs_residual": max_abs,
        "residuals": [
            {
                "frame_id": int(row["frame_id"]),
                "video_idx": int(row["video_idx"]),
                "predicted_video_idx": float(pred_i),
                "residual": float(res_i),
                "score": float(row["score"]),
            }
            for row, pred_i, res_i in zip(best_rows, pred, residual)
        ],
    }


def annotate_direct_rank(candidate_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_probe: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_probe.setdefault((int(row["frame_id"]), int(row["cam_id"])), []).append(row)
    ranks = []
    for key, rows in by_probe.items():
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        direct_rank = None
        direct_score = None
        direct_video_idx = None
        for i, row in enumerate(rows, start=1):
            if int(row["offset"]) == 0:
                direct_rank = i
                direct_score = float(row["score"])
                direct_video_idx = int(row["video_idx"])
                break
        for best in best_rows:
            if (int(best["frame_id"]), int(best["cam_id"])) == key:
                best["direct_rank"] = direct_rank
                best["direct_score"] = direct_score
                best["direct_video_idx"] = direct_video_idx
                if direct_rank is not None:
                    ranks.append(int(direct_rank))
                break
    if not ranks:
        return {"count": 0}
    values = np.asarray(ranks, dtype=np.float64)
    return {
        "count": int(len(ranks)),
        "min": int(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "max": int(np.max(values)),
    }


def write_sheet(panels: list[np.ndarray], output: Path, cols: int) -> None:
    if not panels:
        return
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def as_dict(item: CandidateScore) -> dict[str, Any]:
    return {
        "frame_id": item.frame_id,
        "cam_id": item.cam_id,
        "video_idx": item.video_idx,
        "offset": item.offset,
        "visible": item.visible,
        "edge_hit": item.edge_hit,
        "edge_distance_mean": item.edge_distance_mean,
        "edge_distance_p50": item.edge_distance_p50,
        "score": item.score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", help="Explicit section ids to probe.")
    parser.add_argument("--frame-range", help="Alternative to --frames. Format: start:end:step")
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--offsets", default="-1200:600:100", help="Comma/range list, e.g. -1200:600:100 or -800,-400,0")
    parser.add_argument("--index-scale", type=float, default=1.0)
    parser.add_argument("--index-shift", type=float, default=0.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--edge-dilation-px", type=int, default=9)
    parser.add_argument("--distance-sigma", type=float, default=8.0)
    parser.add_argument("--dot-px", type=int, default=7)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--panels-per-probe", type=int, default=4)
    parser.add_argument("--max-fit-rmse", type=float, default=150.0)
    parser.add_argument("--max-fit-abs-residual", type=float, default=300.0)
    parser.add_argument("--min-fit-samples", type=int, default=4)
    args = parser.parse_args()

    if args.frames:
        frames = sorted(set(args.frames))
    elif args.frame_range:
        frames = parse_int_range(args.frame_range)
    else:
        raise SystemExit("Provide --frames or --frame-range.")
    offsets = parse_int_range(args.offsets)
    if not offsets:
        raise SystemExit("No offsets provided.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sections = read_lx_sections(args.lx_file)
    poses = {row["frame_id"]: row for row in config.load_img_pos(min(frames), max(frames))}

    candidate_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    with args.lx_file.open("rb") as lx_handle:
        for frame_id in frames:
            if frame_id >= len(sections) or frame_id not in poses:
                continue
            for cam_id in args.cams:
                scores, probe_panels = score_candidates_for_probe(
                    lx_handle,
                    sections[frame_id],
                    poses[frame_id],
                    frame_id,
                    cam_id,
                    offsets,
                    args,
                )
                candidate_rows.extend(as_dict(item) for item in scores)
                if scores:
                    best_rows.append(as_dict(scores[0]))
                panels.extend(probe_panels)

    fit_by_cam = {}
    for cam_id in args.cams:
        cam_best = [row for row in best_rows if int(row["cam_id"]) == int(cam_id)]
        fit_by_cam[str(cam_id)] = fit_affine_mapping(
            cam_best,
            args.max_fit_rmse,
            args.max_fit_abs_residual,
            args.min_fit_samples,
        )
    direct_rank_summary = annotate_direct_rank(candidate_rows, best_rows)
    all_accepted = bool(fit_by_cam) and all(item.get("accepted") for item in fit_by_cam.values())
    report = {
        "status": "accepted" if all_accepted else "rejected",
        "lx_file": str(args.lx_file),
        "frames": frames,
        "cams": args.cams,
        "offsets": offsets,
        "index_scale": args.index_scale,
        "index_shift": args.index_shift,
        "video_dir": config.VIDEO_DIR,
        "calib_file": config.CALIB_FILE,
        "fit_by_cam": fit_by_cam,
        "candidate_count": len(candidate_rows),
        "best_count": len(best_rows),
        "direct_rank_summary": direct_rank_summary,
    }

    with (args.output_dir / "sync_candidates.jsonl").open("w", encoding="utf-8") as f:
        for row in candidate_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "sync_best.jsonl").open("w", encoding="utf-8") as f:
        for row in best_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "sync_fit.json").write_text(
        json.dumps({"fit_by_cam": fit_by_cam}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "sync_calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_sheet(panels, args.output_dir / "sync_probe_sheet.jpg", args.sheet_cols)
    print(json.dumps({
        "status": report["status"],
        "candidate_count": len(candidate_rows),
        "best_count": len(best_rows),
        "fit_by_cam": fit_by_cam,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
