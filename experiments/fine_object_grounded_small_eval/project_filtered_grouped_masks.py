#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def add_script_path(script_dir: Path) -> None:
    text = str(script_dir)
    if text not in sys.path:
        sys.path.insert(0, text)


FOCUS_COLORS = {
    "railing": (255, 210, 40),
    "pipe": (255, 165, 0),
    "equipment": (255, 0, 255),
    "hvac": (0, 200, 255),
}
SOURCE_GROUNDED_FINE = 3


def candidate_color(candidate_id: int, focus: str) -> tuple[int, int, int]:
    base = np.array(FOCUS_COLORS.get(focus, (255, 255, 255)), dtype=np.int32)
    rng = np.random.default_rng(candidate_id * 97 + 17)
    delta = rng.integers(-28, 29, size=3)
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
        f.write("end_header\n")
        for row in rows:
            p = row["point"]
            c = row["candidate_color"]
            v = row["visual_color"]
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {int(row['semantic'])} "
                f"{int(row['accepted_candidate'])} {SOURCE_GROUNDED_FINE} {int(row['source_cluster'])} -1 "
                f"{int(v[0])} {int(v[1])} {int(v[2])} "
                f"{int(row['frame'])} {int(row['camera'])} {int(row['mask'])} {int(row['point_index'])}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-jsonl", required=True)
    parser.add_argument("--script-dir", required=True)
    parser.add_argument("--color-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--zbuffer-visible", action="store_true", default=True)
    args = parser.parse_args()

    add_script_path(Path(args.script_dir))
    import config  # type: ignore
    from build_targets_from_masks import read_colored_ply, transform_project  # type: ignore
    from project_semantic import zbuffer_visible_indices  # type: ignore

    accepted_rows = []
    with open(args.accepted_jsonl) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if not args.sample_ids or row["sample_id"] in set(args.sample_ids):
                    accepted_rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    accepted_candidates = []
    all_rows = []
    for row in accepted_rows:
        frame = int(row["frame"])
        cam = int(row["cam"])
        focus = row["focus"]
        candidate_id = 300000 + len(accepted_candidates) + 1
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
            idx, uu, vv = idx[visible], uu[visible], vv[visible]
        keep = mask[vv, uu] > 0
        kept_idx = idx[keep]
        if len(kept_idx) == 0:
            reports.append({"sample_id": row["sample_id"], "status": "mask_no_points", "frame": frame, "cam": cam, "focus": focus})
            continue
        kept_points = points[kept_idx]
        kept_rgb = rgb[kept_idx]
        focus_color = np.array(candidate_color(candidate_id, focus), dtype=np.uint8)
        focus_id = {"railing": 9, "pipe": 15, "equipment": 16, "hvac": 16}.get(focus, 1)
        ply_path = output_dir / f"{row['sample_id']}_{focus}_projected.ply"
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
                }
            )
        write_focus_ply(ply_path, point_rows)
        all_rows.extend(point_rows)
        stats = pca_summary(kept_points)
        candidate_row = {
            "candidate_id": candidate_id,
            "source_type": "grounded_detector_mask",
            "source_cluster": candidate_id,
            "subcluster": -1,
            "semantic_id": focus_id,
            "focus": focus,
            "phrase": row["phrase"],
            "frame": frame,
            "cam": cam,
            "mask_id": 0,
            "points": int(len(kept_points)),
            "mean_visual_color": [float(x) for x in kept_rgb.astype(np.float32).mean(axis=0)],
            "grounding_score": float(row["grounding_score"]),
            "sam_score": float(row["sam_score"]),
            "mask_area_ratio": float(row["mask_area_ratio"]),
            "box_area_ratio": float(row["box_area_ratio"]),
            "box_aspect_ratio": float(row["box_aspect_ratio"]),
            "mask_bbox_fill_ratio": float(row.get("mask_bbox_fill_ratio", 0.0)),
            "largest_component_ratio": float(row.get("largest_component_ratio", 0.0)),
            "minrect_aspect_ratio": float(row.get("minrect_aspect_ratio", 0.0)),
            "component_count": int(row.get("component_count", 0)),
            "mask_path": row["mask_path"],
            "ply_path": str(ply_path),
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
                "total_points": int(len(points)),
                "point_ratio": float(len(kept_points) / max(len(points), 1)),
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
        "candidate_counts": {"grounded_detector_mask": int(len(accepted_candidates))},
        "point_counts": {"grounded_detector_mask": int(len(all_rows))},
        "top_candidates": sorted(accepted_candidates, key=lambda x: x["points"], reverse=True),
    }
    (output_dir / "accepted_report.json").write_text(json.dumps(accepted_report, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "projection_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print(output_dir / "projection_report.json")


if __name__ == "__main__":
    main()
