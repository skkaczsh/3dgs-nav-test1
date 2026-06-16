#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SOURCE_GROUNDED_BOX_GROWTH = 4
FOCUS_SEMANTIC = {
    "railing": 9,
    "pipe": 15,
    "equipment": 16,
    "hvac": 16,
}
FOCUS_COLORS = {
    "railing": (255, 210, 40),
    "pipe": (255, 165, 0),
    "equipment": (255, 0, 255),
    "hvac": (0, 200, 255),
}


def add_script_path(script_dir: Path) -> None:
    text = str(script_dir)
    if text not in sys.path:
        sys.path.insert(0, text)


def candidate_color(candidate_id: int, focus: str) -> tuple[int, int, int]:
    base = np.array(FOCUS_COLORS.get(focus, (255, 255, 255)), dtype=np.int32)
    rng = np.random.default_rng(candidate_id * 101 + 23)
    delta = rng.integers(-24, 25, size=3)
    color = np.clip(base + delta, 40, 255)
    return int(color[0]), int(color[1]), int(color[2])


def pca_summary(points: np.ndarray) -> dict:
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    denom = float(eigvals[0]) if eigvals[0] > 1e-9 else 1.0
    return {
        "centroid": [float(x) for x in centroid],
        "bbox_3d": {
            "min": [float(x) for x in points.min(axis=0)],
            "max": [float(x) for x in points.max(axis=0)],
        },
        "linearity": float((eigvals[0] - eigvals[1]) / denom),
        "planarity": float((eigvals[1] - eigvals[2]) / denom),
        "pca_eigenvalues": [float(x) for x in eigvals],
    }


def write_focus_ply(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("property int accepted_candidate\n")
        f.write("property uchar source_type\n")
        f.write("property int source_cluster\n")
        f.write("property int subcluster\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("property int frame\n")
        f.write("property int camera\n")
        f.write("property int mask\n")
        f.write("property int point_index\n")
        f.write("property uchar seed_flag\n")
        f.write("end_header\n")
        for row in rows:
            p = row["point"]
            c = row["candidate_color"]
            v = row["visual_color"]
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {int(row['semantic'])} "
                f"{int(row['accepted_candidate'])} {SOURCE_GROUNDED_BOX_GROWTH} {int(row['source_cluster'])} -1 "
                f"{int(v[0])} {int(v[1])} {int(v[2])} "
                f"{int(row['frame'])} {int(row['camera'])} {int(row['mask'])} {int(row['point_index'])} "
                f"{int(row['seed_flag'])}\n"
            )


def mask_distance(mask: np.ndarray) -> np.ndarray:
    zeros = (mask == 0).astype(np.uint8)
    return cv2.distanceTransform(zeros, cv2.DIST_L2, 3)


def expanded_box(box_xyxy: list[float], width: int, height: int, pad: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box_xyxy]
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width - 1, x1 + pad),
        min(height - 1, y1 + pad),
    )


def keep_connected_to_seed(
    candidate_indices: np.ndarray,
    points: np.ndarray,
    seed_mask_local: np.ndarray,
    *,
    connected_components,
    voxel_size: float,
    min_component_points: int,
) -> np.ndarray:
    if len(candidate_indices) == 0:
        return np.empty(0, dtype=np.int64)
    candidate_points = points[candidate_indices]
    components, _ = connected_components(candidate_points, voxel_size, min_component_points)
    if not components:
        return candidate_indices[seed_mask_local]
    keep_parts = []
    for comp in components:
        if np.any(seed_mask_local[comp]):
            keep_parts.append(candidate_indices[comp])
    if not keep_parts:
        return candidate_indices[seed_mask_local]
    merged = np.concatenate(keep_parts)
    return np.unique(merged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-jsonl", required=True)
    parser.add_argument("--script-dir", required=True)
    parser.add_argument("--color-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--box-expand-px", type=int, default=12)
    parser.add_argument("--mask-distance-px", type=float, default=18.0)
    parser.add_argument("--depth-lower-quantile", type=float, default=0.05)
    parser.add_argument("--depth-upper-quantile", type=float, default=0.95)
    parser.add_argument("--depth-slack", type=float, default=0.45)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--min-component-points", type=int, default=8)
    parser.add_argument("--zbuffer-visible", action="store_true", default=True)
    args = parser.parse_args()

    add_script_path(Path(args.script_dir))
    import config  # type: ignore
    from build_targets_from_masks import connected_components, read_colored_ply, transform_project  # type: ignore
    from project_semantic import zbuffer_visible_indices  # type: ignore

    sample_id_filter = set(args.sample_ids)
    accepted_rows = []
    with open(args.accepted_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if sample_id_filter and row["sample_id"] not in sample_id_filter:
                continue
            accepted_rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    accepted_candidates = []
    all_rows = []
    for row in accepted_rows:
        frame = int(row["frame"])
        cam = int(row["cam"])
        focus = str(row["focus"])
        candidate_id = 400000 + len(accepted_candidates) + 1
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            reports.append({"sample_id": row["sample_id"], "status": "missing_mask"})
            continue
        color_ply = Path(args.color_dir) / f"frame_{frame:04d}.ply"
        if not color_ply.exists():
            reports.append({"sample_id": row["sample_id"], "status": "missing_color_ply", "frame": frame})
            continue
        points, rgb = read_colored_ply(color_ply)
        projected = transform_project(points, frame, cam, config, args.min_depth)
        if projected is None:
            reports.append({"sample_id": row["sample_id"], "status": "no_projection", "frame": frame, "cam": cam})
            continue
        idx, u, v, depth = projected
        h, w = mask.shape[:2]
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = idx[in_img]
        if len(idx) == 0:
            reports.append({"sample_id": row["sample_id"], "status": "empty_image_overlap", "frame": frame, "cam": cam})
            continue
        uu = np.clip(np.rint(u[in_img]).astype(np.int32), 0, w - 1)
        vv = np.clip(np.rint(v[in_img]).astype(np.int32), 0, h - 1)
        dd = depth[in_img]
        if args.zbuffer_visible:
            visible = zbuffer_visible_indices(idx, np.column_stack([uu, vv]), dd, w)
            idx, uu, vv, dd = idx[visible], uu[visible], vv[visible], dd[visible]

        x0, y0, x1, y1 = expanded_box(row["box_xyxy"], w, h, args.box_expand_px)
        in_box = (uu >= x0) & (uu <= x1) & (vv >= y0) & (vv <= y1)
        if not np.any(in_box):
            reports.append({"sample_id": row["sample_id"], "status": "box_no_points", "frame": frame, "cam": cam, "focus": focus})
            continue

        box_idx = idx[in_box]
        box_uu = uu[in_box]
        box_vv = vv[in_box]
        box_dd = dd[in_box]
        seed_local = mask[box_vv, box_uu] > 0
        seed_count = int(seed_local.sum())
        if seed_count == 0:
            reports.append({"sample_id": row["sample_id"], "status": "mask_no_points", "frame": frame, "cam": cam, "focus": focus})
            continue

        dist_map = mask_distance(mask)
        box_dist = dist_map[box_vv, box_uu]
        seed_depths = box_dd[seed_local]
        depth_lo = float(np.quantile(seed_depths, args.depth_lower_quantile) - args.depth_slack)
        depth_hi = float(np.quantile(seed_depths, args.depth_upper_quantile) + args.depth_slack)
        near_mask = box_dist <= args.mask_distance_px
        depth_ok = (box_dd >= depth_lo) & (box_dd <= depth_hi)
        candidate_local = near_mask & depth_ok
        candidate_indices = box_idx[candidate_local]
        candidate_seed_local = seed_local[candidate_local]
        if len(candidate_indices) == 0:
            candidate_indices = box_idx[seed_local]
            candidate_seed_local = np.ones(len(candidate_indices), dtype=bool)

        kept_idx = keep_connected_to_seed(
            candidate_indices,
            points,
            candidate_seed_local,
            connected_components=connected_components,
            voxel_size=args.voxel_size,
            min_component_points=args.min_component_points,
        )
        if len(kept_idx) == 0:
            kept_idx = box_idx[seed_local]

        kept_points = points[kept_idx]
        kept_rgb = rgb[kept_idx]
        seed_idx_set = set(int(i) for i in box_idx[seed_local].tolist())
        focus_color = np.array(candidate_color(candidate_id, focus), dtype=np.uint8)
        focus_id = FOCUS_SEMANTIC.get(focus, 1)

        ply_path = output_dir / f"{row['sample_id']}_{focus}_grown_projected.ply"
        point_rows = []
        for point, visual, point_index in zip(kept_points, kept_rgb, kept_idx):
            point_rows.append(
                {
                    "point": point,
                    "candidate_color": focus_color,
                    "visual_color": visual,
                    "semantic": focus_id,
                    "accepted_candidate": candidate_id,
                    "source_cluster": candidate_id,
                    "frame": frame,
                    "camera": cam,
                    "mask": 0,
                    "point_index": int(point_index),
                    "seed_flag": 1 if int(point_index) in seed_idx_set else 0,
                }
            )
        write_focus_ply(ply_path, point_rows)
        all_rows.extend(point_rows)

        stats = pca_summary(kept_points)
        seed_point_count = sum(1 for r in point_rows if r["seed_flag"])
        mean_color = kept_rgb.astype(np.float32).mean(axis=0)
        candidate_row = {
            "candidate_id": candidate_id,
            "source_type": "grounded_box_growth",
            "source_cluster": candidate_id,
            "subcluster": -1,
            "semantic_id": focus_id,
            "focus": focus,
            "phrase": row["phrase"],
            "frame": frame,
            "cam": cam,
            "mask_id": 0,
            "points": int(len(kept_points)),
            "seed_points": int(seed_point_count),
            "growth_gain": float(len(kept_points) / max(seed_point_count, 1)),
            "mean_visual_color": [float(x) for x in mean_color],
            "grounding_score": float(row["grounding_score"]),
            "sam_score": float(row["sam_score"]),
            "mask_area_ratio": float(row["mask_area_ratio"]),
            "box_area_ratio": float(row["box_area_ratio"]),
            "box_aspect_ratio": float(row["box_aspect_ratio"]),
            "mask_bbox_fill_ratio": float(row.get("mask_bbox_fill_ratio", 0.0)),
            "largest_component_ratio": float(row.get("largest_component_ratio", 0.0)),
            "minrect_aspect_ratio": float(row.get("minrect_aspect_ratio", 0.0)),
            "component_count": int(row.get("component_count", 0)),
            "box_xyxy": row["box_xyxy"],
            "mask_path": row["mask_path"],
            "ply_path": str(ply_path),
            "grow_params": {
                "box_expand_px": int(args.box_expand_px),
                "mask_distance_px": float(args.mask_distance_px),
                "depth_slack": float(args.depth_slack),
                "voxel_size": float(args.voxel_size),
                "min_component_points": int(args.min_component_points),
            },
            **stats,
        }
        accepted_candidates.append(candidate_row)
        reports.append(
            {
                "sample_id": row["sample_id"],
                "candidate_id": candidate_id,
                "frame": frame,
                "cam": cam,
                "focus": focus,
                "status": "ok",
                "projected_points": int(len(kept_points)),
                "seed_points": int(seed_point_count),
                "box_points": int(len(box_idx)),
                "candidate_box_points": int(len(candidate_indices)),
                "growth_gain": float(len(kept_points) / max(seed_point_count, 1)),
                "point_ratio": float(len(kept_points) / max(len(points), 1)),
                "depth_range": [depth_lo, depth_hi],
                "ply_path": str(ply_path),
                "mask_path": row["mask_path"],
                "grounding_score": row["grounding_score"],
                "mask_area_ratio": row["mask_area_ratio"],
            }
        )

    accepted_ply_path = output_dir / "accepted_points.ply"
    if all_rows:
        write_focus_ply(accepted_ply_path, all_rows)
    accepted_report = {
        "accepted_jsonl": str(args.accepted_jsonl),
        "output_ply": str(accepted_ply_path),
        "candidate_count": int(len(accepted_candidates)),
        "accepted_points": int(len(all_rows)),
        "candidate_counts": {"grounded_box_growth": int(len(accepted_candidates))},
        "point_counts": {"grounded_box_growth": int(len(all_rows))},
        "top_candidates": sorted(accepted_candidates, key=lambda x: x["points"], reverse=True),
    }
    (output_dir / "accepted_report.json").write_text(json.dumps(accepted_report, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "projection_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print(output_dir / "projection_report.json")


if __name__ == "__main__":
    main()
