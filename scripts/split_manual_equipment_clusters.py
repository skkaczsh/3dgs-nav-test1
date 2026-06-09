#!/usr/bin/env python3
"""Split manual-review equipment clusters by 3D connectivity and visual color.

Input is the hygiene status PLY from apply_oversized_mask_hygiene.py. The script
selects manual-review equipment points, then forms connected components in a
voxel grid where neighboring voxels must also have similar mean visual RGB.
Output is a non-destructive QA PLY plus a JSON report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np


EQUIPMENT_SEMANTIC_ID = 16
MANUAL_REVIEW_STATUS = 2


def read_ascii_ply(path: Path) -> tuple[list[str], int, np.ndarray]:
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
                in_vertex = True
            elif s.startswith("element "):
                in_vertex = False
            elif in_vertex and s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    if vertex_count == 0:
        return props, header_lines, np.empty((0, len(props)), dtype=np.float32)
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return props, header_lines, data


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
        "pca_eigenvalues": [float(x) for x in eigvals],
        "linearity": float((eigvals[0] - eigvals[1]) / denom),
        "planarity": float((eigvals[1] - eigvals[2]) / denom),
    }


def subcluster_color(subcluster_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(subcluster_id + 17017)
    return tuple(int(x) for x in rng.integers(60, 245, 3))


def build_voxels(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    cells = np.floor(points / voxel_size).astype(np.int32)
    voxel_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for idx, cell in enumerate(cells):
        voxel_to_indices.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(idx)
    voxel_color = {
        voxel: colors[indices].astype(np.float32).mean(axis=0) for voxel, indices in voxel_to_indices.items()
    }
    return voxel_to_indices, voxel_color


def connected_components(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    max_color_distance: float,
    min_points: int,
) -> tuple[list[np.ndarray], int]:
    if len(points) == 0:
        return [], 0
    voxel_to_indices, voxel_color = build_voxels(points, colors, voxel_size)
    remaining = set(voxel_to_indices)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    comps: list[np.ndarray] = []
    small_points = 0
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        comp_voxels = [start]
        while queue:
            voxel = queue.popleft()
            base_color = voxel_color[voxel]
            x, y, z = voxel
            for dx, dy, dz in offsets:
                nb = (x + dx, y + dy, z + dz)
                if nb not in remaining:
                    continue
                if float(np.linalg.norm(base_color - voxel_color[nb])) > max_color_distance:
                    continue
                remaining.remove(nb)
                queue.append(nb)
                comp_voxels.append(nb)
        indices = np.array([i for voxel in comp_voxels for i in voxel_to_indices[voxel]], dtype=np.int64)
        if len(indices) >= min_points:
            comps.append(indices)
        else:
            small_points += int(len(indices))
    comps.sort(key=len, reverse=True)
    return comps, small_points


def write_ply(path: Path, rows: list[dict]) -> None:
    total = sum(len(row["points"]) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("property int source_cluster\n")
        f.write("property int subcluster\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("end_header\n")
        for row in rows:
            color = subcluster_color(int(row["subcluster_id"]))
            for point, visual in zip(row["points"], row["visual_colors"]):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]} {EQUIPMENT_SEMANTIC_ID} "
                    f"{int(row['source_cluster'])} {int(row['subcluster_id'])} "
                    f"{int(visual[0])} {int(visual[1])} {int(visual[2])}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygiene-status-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.06)
    parser.add_argument("--max-color-distance", type=float, default=45.0)
    parser.add_argument("--min-subcluster-points", type=int, default=80)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    props, _, data = read_ascii_ply(args.hygiene_status_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {"x", "y", "z", "semantic", "cluster", "visual_red", "visual_green", "visual_blue", "cluster_status"}
    if not required.issubset(idx):
        raise ValueError(f"missing required fields. required={required} available={props}")

    semantic = data[:, idx["semantic"]].astype(np.int32)
    status = data[:, idx["cluster_status"]].astype(np.int32)
    selected = (semantic == EQUIPMENT_SEMANTIC_ID) & (status == MANUAL_REVIEW_STATUS)
    points = data[selected][:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    colors = data[selected][:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]].astype(np.uint8)
    source_clusters = data[selected][:, idx["cluster"]].astype(np.int32)

    rows = []
    report_rows = []
    small_by_source = Counter()
    subcluster_id = 1
    for source_cluster in sorted(set(int(x) for x in source_clusters.tolist())):
        local = np.where(source_clusters == source_cluster)[0]
        pts = points[local]
        vis = colors[local]
        comps, small = connected_components(
            pts,
            vis,
            args.voxel_size,
            args.max_color_distance,
            args.min_subcluster_points,
        )
        small_by_source[str(source_cluster)] = int(small)
        for comp in comps:
            comp_pts = pts[comp]
            comp_vis = vis[comp]
            summary = pca_summary(comp_pts)
            row = {
                "subcluster_id": int(subcluster_id),
                "source_cluster": int(source_cluster),
                "points": int(len(comp)),
                "mean_visual_color": [float(x) for x in comp_vis.astype(np.float32).mean(axis=0)],
                **summary,
            }
            report_rows.append(row)
            rows.append(
                {
                    "subcluster_id": int(subcluster_id),
                    "source_cluster": int(source_cluster),
                    "points": comp_pts,
                    "visual_colors": comp_vis,
                }
            )
            subcluster_id += 1

    report_rows.sort(key=lambda row: row["points"], reverse=True)
    write_ply(args.output_ply, rows)
    report = {
        "hygiene_status_ply": str(args.hygiene_status_ply),
        "output_ply": str(args.output_ply),
        "params": {
            "voxel_size": args.voxel_size,
            "max_color_distance": args.max_color_distance,
            "min_subcluster_points": args.min_subcluster_points,
        },
        "selected_points": int(len(points)),
        "source_cluster_count": int(len(set(int(x) for x in source_clusters.tolist()))),
        "subcluster_count": int(len(report_rows)),
        "clustered_points": int(sum(row["points"] for row in report_rows)),
        "small_points": int(sum(small_by_source.values())),
        "small_by_source_cluster": dict(small_by_source),
        "top_subclusters": report_rows[: args.top_n],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_points": report["selected_points"],
                "source_cluster_count": report["source_cluster_count"],
                "subcluster_count": report["subcluster_count"],
                "clustered_points": report["clustered_points"],
                "small_points": report["small_points"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
