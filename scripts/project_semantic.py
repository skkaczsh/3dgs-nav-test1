#!/usr/bin/env python3
"""Project 2D semantic PNGs onto new_route section point clouds.

Input semantic artifacts come from semantic_eval/run_eval.py:
  <semantic-eval>/images/cam0_000000/<combo>/semantic.png

This script uses the same calibrated world->camera projection as project_color.py
and writes per-section PLY files with an extra `semantic` property plus a QA JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import cv2
except ModuleNotFoundError:  # Allows importing label/z-buffer helpers without OpenCV.
    cv2 = None

try:
    from project_color import load_ply_xyz
except ModuleNotFoundError:
    from scripts.project_color import load_ply_xyz
from scripts.semantic_label_contract import SEMANTIC_COLORS as LABEL_COLORS
from scripts.semantic_label_contract import SEMANTIC_TO_LABEL as LABEL_NAMES


def semantic_path(base: Path, combo: str, cam_id: int, frame_id: int) -> Path:
    return base / "images" / f"cam{cam_id}_{frame_id:06d}" / combo / "semantic.png"


def frames_with_semantic(base: Path, combo: str) -> list[int]:
    frames = set()
    images_dir = base / "images"
    if not images_dir.exists():
        return []
    for sem_path in images_dir.glob(f"cam*_*/*/semantic.png"):
        if sem_path.parent.name != combo:
            continue
        image_id = sem_path.parent.parent.name
        try:
            frames.add(int(image_id.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(frames)


def write_semantic_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    colors = np.array([LABEL_COLORS.get(int(x), LABEL_COLORS[0]) for x in labels], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("end_header\n")
        for p, c, label in zip(points, colors, labels):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {int(label)}\n"
            )


def read_semantic_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header_lines = 0
    with path.open("rb") as f:
        for raw in f:
            header_lines += 1
            if raw.strip() == b"end_header":
                break
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    points = data[:, :3].astype(np.float32)
    labels = data[:, 6].astype(np.uint8) if data.shape[1] >= 7 else np.zeros(len(points), dtype=np.uint8)
    return points, labels


def zbuffer_visible_indices(point_indices: np.ndarray, pixels: np.ndarray, depths: np.ndarray, width: int) -> np.ndarray:
    """Return the nearest projected point for each image pixel."""
    if len(point_indices) == 0:
        return np.zeros(0, dtype=bool)
    pixel_idx = pixels[:, 1].astype(np.int64) * int(width) + pixels[:, 0].astype(np.int64)
    order = np.lexsort((depths, pixel_idx))
    sorted_pixel_idx = pixel_idx[order]
    first = np.r_[True, sorted_pixel_idx[1:] != sorted_pixel_idx[:-1]]
    keep_order = order[first]
    keep = np.zeros(len(point_indices), dtype=bool)
    keep[keep_order] = True
    return keep


def process_frame(frame_id: int, args: argparse.Namespace, config) -> dict:
    pcd_path = Path(config.EXTRACTED_DIR) / f"section_{frame_id:04d}.ply"
    if not pcd_path.exists():
        return {"frame": frame_id, "status": "missing_pcd"}

    points = load_ply_xyz(str(pcd_path))
    if args.max_points and len(points) > args.max_points:
        rng = np.random.default_rng(args.seed + frame_id)
        idx = rng.choice(len(points), args.max_points, replace=False)
        points = points[idx]

    pose_data = config.load_img_pos(frame_id, frame_id)
    if not pose_data:
        return {"frame": frame_id, "status": "missing_pose", "points": int(len(points))}
    T = pose_data[0]["T_world_robot"]

    labels = np.zeros(len(points), dtype=np.uint8)
    best_depth = np.full(len(points), np.inf, dtype=np.float64)
    observation_count = np.zeros(len(points), dtype=np.uint8)

    R_rw = T[:3, :3]
    t_rw = T[:3, 3]
    R_wr = R_rw.T
    t_wr = (-R_wr @ t_rw).reshape(3)
    R_li = config.Til[:3, :3].T
    t_li = (-R_li @ config.Til[:3, 3]).reshape(3)

    semantic_found = 0
    semantic_missing = 0
    for cam_id in args.cams:
        sem_path = semantic_path(args.semantic_eval_dir, args.combo, cam_id, frame_id)
        if not sem_path.exists():
            semantic_missing += 1
            continue
        semantic_found += 1
        if cv2 is None:
            raise RuntimeError("OpenCV is required to load semantic PNGs in project_semantic.py")
        sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        H, W = sem.shape[:2]

        T_cl = config.Tcl[cam_id]
        P_robot = (R_wr @ points.T + t_wr.reshape(3, 1)).T
        P_lidar = (R_li @ P_robot.T + t_li.reshape(3, 1)).T
        P_cam = (T_cl[:3, :3] @ P_lidar.T + T_cl[:3, 3].reshape(3, 1)).T

        z = P_cam[:, 2]
        valid = z > args.min_depth
        if not np.any(valid):
            continue
        uv_h = (config.CAMERA_PARAMS[cam_id]["K"] @ P_cam[valid].T).T
        u = uv_h[:, 0] / uv_h[:, 2]
        v = uv_h[:, 1] / uv_h[:, 2]
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_img):
            continue

        idx = np.where(valid)[0][in_img]
        uu = np.clip(np.rint(u[in_img]).astype(np.int32), 0, W - 1)
        vv = np.clip(np.rint(v[in_img]).astype(np.int32), 0, H - 1)
        depths = z[valid][in_img]
        if args.zbuffer_visible:
            visible = zbuffer_visible_indices(idx, np.column_stack([uu, vv]), depths, W)
            idx, uu, vv, depths = idx[visible], uu[visible], vv[visible], depths[visible]
            if len(idx) == 0:
                continue
        sampled = sem[vv, uu].astype(np.uint8)

        if args.ignore_sky:
            keep = sampled != 11
            idx, sampled, depths = idx[keep], sampled[keep], depths[keep]
        if args.ignore_ids:
            ignore = np.isin(sampled, np.array(args.ignore_ids, dtype=np.uint8))
            idx, sampled, depths = idx[~ignore], sampled[~ignore], depths[~ignore]
        if len(idx) == 0:
            continue

        if args.ignore_sky or args.ignore_ids:
            # Recompute depths for kept idx without relying on previous masks.
            depths = P_cam[idx, 2]
        closer = depths < best_depth[idx]
        update_idx = idx[closer]
        labels[update_idx] = sampled[closer]
        best_depth[update_idx] = depths[closer]
        observation_count[idx] = np.clip(observation_count[idx] + 1, 0, 255)

    labeled = labels != 0
    out_path = args.output_dir / f"semantic_frame_{frame_id:04d}.ply"
    if args.write_ply:
        write_semantic_ply(out_path, points, labels)

    counts = Counter(int(x) for x in labels[labeled])
    return {
        "frame": frame_id,
        "status": "ok",
        "points": int(len(points)),
        "semantic_found": semantic_found,
        "semantic_missing": semantic_missing,
        "labeled_points": int(labeled.sum()),
        "labeled_ratio": float(labeled.sum() / max(len(points), 1)),
        "unknown_points": int((labels == 0).sum()),
        "sky_points": int((labels == 11).sum()),
        "ignore_points": int((labels == 255).sum()),
        "mean_observation_count": float(observation_count.mean()) if len(points) else 0.0,
        "label_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "label_names": {str(k): LABEL_NAMES.get(k, "unknown") for k in sorted(counts)},
        "output": str(out_path) if args.write_ply else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-eval-dir", type=Path, required=True)
    parser.add_argument("--combo", default="sky_sam3_rules_qwen_review")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--frames-from-semantic-dir", action="store_true")
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--zbuffer-visible", action="store_true", default=True,
                        help="Keep only the nearest projected point per camera pixel before sampling semantic labels.")
    parser.add_argument("--no-zbuffer-visible", dest="zbuffer_visible", action="store_false")
    parser.add_argument("--ignore-sky", action="store_true", default=True)
    parser.add_argument("--include-sky", dest="ignore_sky", action="store_false")
    parser.add_argument("--ignore-ids", type=int, nargs="*", default=[255])
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--write-merged-ply", action="store_true")
    parser.add_argument("--merged-name", default="semantic_points_merged.ply")
    args = parser.parse_args()
    if args.write_merged_ply:
        args.write_ply = True

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.frames_from_semantic_dir:
        frame_ids = frames_with_semantic(args.semantic_eval_dir, args.combo)
    elif args.frames is not None:
        frame_ids = sorted(set(args.frames))
    else:
        frame_ids = list(range(args.start, args.end + 1, args.stride))
    for frame_id in frame_ids:
        row = process_frame(frame_id, args, config)
        rows.append(row)
        if row.get("status") == "ok":
            print(f"frame={frame_id} labeled={row['labeled_ratio']:.3f} found={row['semantic_found']} missing={row['semantic_missing']}")
        else:
            print(f"frame={frame_id} status={row.get('status')}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    merged = {"available": False}
    if args.write_merged_ply:
        frame_points = []
        frame_labels = []
        for row in ok_rows:
            out = row.get("output")
            if not out:
                continue
            out_path = Path(out)
            if not out_path.exists():
                continue
            points, labels = read_semantic_ply(out_path)
            frame_points.append(points)
            frame_labels.append(labels)
        if frame_points:
            merged_points = np.concatenate(frame_points, axis=0)
            merged_labels = np.concatenate(frame_labels, axis=0)
            merged_path = args.output_dir / args.merged_name
            write_semantic_ply(merged_path, merged_points, merged_labels)
            merged = {
                "available": True,
                "output": str(merged_path),
                "points": int(len(merged_points)),
                "labeled_points": int((merged_labels != 0).sum()),
                "labeled_ratio": float((merged_labels != 0).sum() / max(len(merged_labels), 1)),
            }
    report = {
        "semantic_eval_dir": str(args.semantic_eval_dir),
        "combo": args.combo,
        "frames": rows,
        "merged": merged,
        "summary": {
            "frame_count": len(rows),
            "ok_count": len(ok_rows),
            "avg_labeled_ratio": float(np.mean([r["labeled_ratio"] for r in ok_rows])) if ok_rows else 0.0,
            "avg_sky_points": float(np.mean([r["sky_points"] for r in ok_rows])) if ok_rows else 0.0,
            "avg_ignore_points": float(np.mean([r["ignore_points"] for r in ok_rows])) if ok_rows else 0.0,
            "total_labeled_points": int(sum(r["labeled_points"] for r in ok_rows)),
            "total_points": int(sum(r["points"] for r in ok_rows)),
        },
    }
    (args.output_dir / "semantic_projection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote={args.output_dir / 'semantic_projection_report.json'}")


if __name__ == "__main__":
    main()
