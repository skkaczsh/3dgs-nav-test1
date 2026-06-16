#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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


def write_focus_ply(path: Path, points: np.ndarray, colors: np.ndarray, focus_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("end_header\n")
        for p, c, s in zip(points, colors, focus_ids):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} {int(s)}\n")


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
    for row in accepted_rows:
        frame = int(row["frame"])
        cam = int(row["cam"])
        focus = row["focus"]
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
        focus_color = np.tile(np.array(FOCUS_COLORS.get(focus, (255, 255, 255)), dtype=np.uint8), (len(kept_points), 1))
        focus_id = {"railing": 9, "pipe": 15, "equipment": 16, "hvac": 16}.get(focus, 1)
        focus_ids = np.full(len(kept_points), focus_id, dtype=np.uint8)
        ply_path = output_dir / f"{row['sample_id']}_{focus}_projected.ply"
        write_focus_ply(ply_path, kept_points, focus_color, focus_ids)
        reports.append(
            {
                "sample_id": row["sample_id"],
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

    (output_dir / "projection_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print(output_dir / "projection_report.json")


if __name__ == "__main__":
    main()
