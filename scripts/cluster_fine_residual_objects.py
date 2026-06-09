#!/usr/bin/env python3
"""Cluster unassigned fine-object residual points for second-stage QA.

Input is the residual assignment PLY from assign_residuals_to_surface_objects.py.
Only unassigned points with selected semantic ids are clustered. The output PLY
is a QA artifact, not a replacement for the main object fusion result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from analyze_residual_absorbability import LABEL_COLORS, SEMANTIC_IDS, SEMANTIC_NAMES


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
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if len(data) != vertex_count:
        raise ValueError(f"vertex count mismatch: {path} header={vertex_count} rows={len(data)}")
    return props, header_lines, data


def connected_components(points: np.ndarray, voxel_size: float, min_points: int) -> tuple[list[np.ndarray], int]:
    if len(points) == 0:
        return [], 0
    cells = np.floor(points / voxel_size).astype(np.int32)
    cell_points: dict[tuple[int, int, int], list[int]] = {}
    for idx, cell in enumerate(cells):
        cell_points.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(idx)

    remaining = set(cell_points)
    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    components = []
    small_points = 0
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        indices = []
        while queue:
            cell = queue.popleft()
            indices.extend(cell_points[cell])
            x, y, z = cell
            for dx, dy, dz in neighbors:
                nb = (x + dx, y + dy, z + dz)
                if nb in remaining:
                    remaining.remove(nb)
                    queue.append(nb)
        comp = np.array(indices, dtype=np.int64)
        if len(comp) >= min_points:
            components.append(comp)
        else:
            small_points += int(len(comp))
    components.sort(key=len, reverse=True)
    return components, small_points


def cluster_color(cluster_id: int, semantic_id: int) -> tuple[int, int, int]:
    base = np.array(LABEL_COLORS.get(semantic_id, LABEL_COLORS[0]), dtype=np.int32)
    rng = np.random.default_rng(cluster_id)
    jitter = rng.integers(-35, 36, 3)
    color = np.clip(base + jitter, 40, 255)
    return int(color[0]), int(color[1]), int(color[2])


def write_cluster_ply(path: Path, rows: list[dict]) -> None:
    total = sum(len(row["points"]) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("property int cluster\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("end_header\n")
        for row in rows:
            color = cluster_color(int(row["cluster_id"]), int(row["semantic_id"]))
            for point, visual in zip(row["points"], row["visual_colors"]):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]} {int(row['semantic_id'])} {int(row['cluster_id'])} "
                    f"{int(visual[0])} {int(visual[1])} {int(visual[2])}\n"
                )


def pca_summary(points: np.ndarray) -> dict:
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    denom = float(eigvals[0]) if eigvals[0] > 1e-9 else 1.0
    linearity = float((eigvals[0] - eigvals[1]) / denom)
    planarity = float((eigvals[1] - eigvals[2]) / denom)
    return {
        "centroid": [float(x) for x in centroid],
        "bbox_3d": {
            "min": [float(x) for x in points.min(axis=0)],
            "max": [float(x) for x in points.max(axis=0)],
        },
        "pca_eigenvalues": [float(x) for x in eigvals],
        "linearity": linearity,
        "planarity": planarity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-assignment-ply", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["equipment", "railing"])
    parser.add_argument("--voxel-size", type=float, default=0.12)
    parser.add_argument("--min-cluster-points", type=int, default=50)
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    wanted_ids = {SEMANTIC_IDS[label] for label in args.labels}
    props, _, data = read_ascii_ply(args.residual_assignment_ply)
    idx = {name: i for i, name in enumerate(props)}
    assignment_status = data[:, idx["assignment_status"]].astype(np.int32)
    original_semantic = data[:, idx["original_semantic"]].astype(np.int32)
    selected = (assignment_status == 0) & np.isin(original_semantic, np.array(sorted(wanted_ids), dtype=np.int32))

    points = data[selected][:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    visuals = data[selected][:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]].astype(np.uint8)
    semantics = original_semantic[selected]

    rows = []
    cluster_ply_rows = []
    cluster_id = 1
    by_label = Counter()
    clustered_by_label = Counter()
    small_by_label = Counter()
    for semantic_id in sorted(wanted_ids):
        label = SEMANTIC_NAMES.get(int(semantic_id), "unknown")
        local_idx = np.where(semantics == semantic_id)[0]
        by_label[label] = int(len(local_idx))
        comps, small_points = connected_components(points[local_idx], args.voxel_size, args.min_cluster_points)
        small_by_label[label] = int(small_points)
        for comp in comps:
            global_idx = local_idx[comp]
            pts = points[global_idx]
            vis = visuals[global_idx]
            clustered_by_label[label] += int(len(global_idx))
            summary = pca_summary(pts)
            row = {
                "cluster_id": int(cluster_id),
                "semantic_id": int(semantic_id),
                "label": label,
                "points": int(len(global_idx)),
                "mean_visual_color": [float(x) for x in vis.astype(np.float32).mean(axis=0)],
                **summary,
            }
            rows.append(row)
            if args.write_ply:
                cluster_ply_rows.append(
                    {
                        "cluster_id": cluster_id,
                        "semantic_id": int(semantic_id),
                        "points": pts,
                        "visual_colors": vis,
                    }
                )
            cluster_id += 1

    rows.sort(key=lambda r: r["points"], reverse=True)
    if args.write_ply:
        write_cluster_ply(args.output_ply, cluster_ply_rows)

    report = {
        "residual_assignment_ply": str(args.residual_assignment_ply),
        "output_ply": str(args.output_ply) if args.write_ply else "",
        "labels": args.labels,
        "params": {
            "voxel_size": args.voxel_size,
            "min_cluster_points": args.min_cluster_points,
        },
        "selected_points": int(len(points)),
        "cluster_count": int(len(rows)),
        "clustered_points": int(sum(r["points"] for r in rows)),
        "small_cluster_points": int(sum(small_by_label.values())),
        "by_label": dict(by_label),
        "clustered_by_label": dict(clustered_by_label),
        "small_by_label": dict(small_by_label),
        "top_clusters": rows[: args.top_n],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "selected_points": report["selected_points"],
            "cluster_count": report["cluster_count"],
            "clustered_points": report["clustered_points"],
            "small_cluster_points": report["small_cluster_points"],
            "by_label": report["by_label"],
            "clustered_by_label": report["clustered_by_label"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
