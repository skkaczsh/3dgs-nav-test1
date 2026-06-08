#!/usr/bin/env python3
"""Merge scattered semantic residuals into stable surfaces using geometry and visual color."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


LABEL_NAMES = {
    0: "unknown",
    1: "other",
    2: "wall",
    3: "floor",
    4: "ceiling",
    5: "grass",
    6: "tree",
    7: "person",
    8: "car",
    9: "railing",
    10: "building",
    11: "sky",
    12: "road",
    13: "water",
    14: "furniture",
    15: "pipe",
    16: "equipment",
    254: "ambiguous",
    255: "ignore",
}

LABEL_COLORS = {
    0: (128, 128, 128),
    1: (160, 160, 160),
    2: (200, 200, 200),
    3: (139, 100, 60),
    4: (240, 240, 240),
    5: (80, 180, 80),
    6: (20, 120, 40),
    7: (255, 80, 80),
    8: (60, 120, 255),
    9: (255, 210, 40),
    10: (190, 170, 140),
    11: (135, 206, 250),
    12: (80, 80, 80),
    13: (30, 160, 220),
    14: (120, 80, 200),
    15: (255, 165, 0),
    16: (255, 0, 255),
    254: (120, 60, 255),
    255: (30, 30, 30),
}


def read_ascii_ply(path: Path) -> tuple[list[str], np.ndarray]:
    props: list[str] = []
    vertex_count = None
    header_lines = 0
    with path.open() as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
            elif s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"missing vertex count: {path}")
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if len(data) != vertex_count:
        raise ValueError(f"vertex count mismatch for {path}: header={vertex_count} rows={len(data)}")
    return props, data


def label_counts(labels: np.ndarray) -> dict:
    counts = Counter(int(x) for x in labels)
    return {
        str(k): {"name": LABEL_NAMES.get(int(k), "unknown"), "points": int(v)}
        for k, v in sorted(counts.items())
    }


def connected_clusters(points: np.ndarray, voxel_size: float) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    cells = np.floor(points / voxel_size).astype(np.int32)
    cell_points: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, cell in enumerate(cells):
        cell_points.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(idx)
    remaining = set(cell_points)
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    clusters: list[np.ndarray] = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        indices: list[int] = []
        while queue:
            cell = queue.popleft()
            indices.extend(cell_points[cell])
            x, y, z = cell
            for dx, dy, dz in neighbors:
                nb = (x + dx, y + dy, z + dz)
                if nb in remaining:
                    remaining.remove(nb)
                    queue.append(nb)
        clusters.append(np.array(indices, dtype=np.int64))
    clusters.sort(key=len, reverse=True)
    return clusters


def classify_clusters(
    points: np.ndarray,
    labels: np.ndarray,
    stable_labels: set[int],
    voxel_size: float,
    stable_min_cluster_points: int,
    candidate_max_cluster_points: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    stable_surface = np.zeros(len(points), dtype=bool)
    candidate = np.zeros(len(points), dtype=bool)
    cluster_size = np.zeros(len(points), dtype=np.int32)
    cluster_rows = []
    for label in sorted(set(int(x) for x in labels)):
        local = np.where(labels == label)[0]
        for cluster in connected_clusters(points[local], voxel_size):
            idx = local[cluster]
            size = int(len(idx))
            cluster_size[idx] = size
            is_stable = label in stable_labels and size >= stable_min_cluster_points
            if is_stable:
                stable_surface[idx] = True
            elif size <= candidate_max_cluster_points or label == 0:
                candidate[idx] = True
            cluster_rows.append(
                {
                    "label": int(label),
                    "name": LABEL_NAMES.get(int(label), "unknown"),
                    "points": size,
                    "stable_surface": bool(is_stable),
                }
            )
    return stable_surface, candidate, {"clusters": sorted(cluster_rows, key=lambda x: x["points"], reverse=True)[:200]}


def maybe_surface_label(
    point: np.ndarray,
    color: np.ndarray,
    stable_points: np.ndarray,
    stable_colors: np.ndarray,
    stable_labels: np.ndarray,
    neighbor_ids: Iterable[int],
    min_plane_neighbors: int,
    max_plane_distance: float,
    max_plane_rms: float,
    max_color_distance: float,
) -> tuple[int, dict] | None:
    ids = np.array(list(neighbor_ids), dtype=np.int64)
    if len(ids) < min_plane_neighbors:
        return None
    local_points = stable_points[ids]
    centroid = local_points.mean(axis=0)
    centered = local_points - centroid
    cov = centered.T @ centered / max(len(local_points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    plane_dist = float(abs((point - centroid) @ normal))
    plane_rms = float(np.sqrt(max(float(np.min(eigvals)), 0.0)))
    if plane_dist > max_plane_distance or plane_rms > max_plane_rms:
        return None
    mean_color = stable_colors[ids].astype(np.float32).mean(axis=0)
    color_dist = float(np.linalg.norm(color.astype(np.float32) - mean_color))
    if color_dist > max_color_distance:
        return None
    label = Counter(int(x) for x in stable_labels[ids]).most_common(1)[0][0]
    return label, {"plane_distance": plane_dist, "plane_rms": plane_rms, "color_distance": color_dist}


def write_surface_ply(
    path: Path,
    points: np.ndarray,
    visual_colors: np.ndarray,
    original_labels: np.ndarray,
    consolidated_labels: np.ndarray,
    status: np.ndarray,
) -> None:
    semantic_colors = np.array([LABEL_COLORS.get(int(x), LABEL_COLORS[0]) for x in consolidated_labels], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("property uchar consolidated_semantic\n")
        f.write("property uchar surface_status\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("end_header\n")
        for p, sc, orig, label, st, vc in zip(
            points, semantic_colors, original_labels, consolidated_labels, status, visual_colors
        ):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(sc[0])} {int(sc[1])} {int(sc[2])} "
                f"{int(orig)} {int(label)} {int(st)} "
                f"{int(vc[0])} {int(vc[1])} {int(vc[2])}\n"
            )


def process_frame(frame_id: int, args: argparse.Namespace) -> dict:
    sem_path = args.semantic_dir / f"semantic_frame_{frame_id:04d}.ply"
    color_path = args.color_dir / f"frame_{frame_id:04d}.ply"
    if not sem_path.exists() or not color_path.exists():
        return {"frame": frame_id, "status": "missing", "semantic_exists": sem_path.exists(), "color_exists": color_path.exists()}
    sem_props, sem_data = read_ascii_ply(sem_path)
    color_props, color_data = read_ascii_ply(color_path)
    sem_idx = {name: i for i, name in enumerate(sem_props)}
    color_idx = {name: i for i, name in enumerate(color_props)}
    points = sem_data[:, [sem_idx["x"], sem_idx["y"], sem_idx["z"]]].astype(np.float32)
    labels = sem_data[:, sem_idx["semantic"]].astype(np.uint8)
    color_points = color_data[:, [color_idx["x"], color_idx["y"], color_idx["z"]]].astype(np.float32)
    visual_colors = color_data[:, [color_idx["red"], color_idx["green"], color_idx["blue"]]].astype(np.uint8)
    if len(points) != len(color_points) or not np.allclose(points, color_points, atol=1e-4):
        raise ValueError(f"semantic/color point order mismatch for frame {frame_id}")

    stable_surface, candidate, cluster_report = classify_clusters(
        points,
        labels,
        set(args.stable_labels),
        args.voxel_size,
        args.stable_min_cluster_points,
        args.candidate_max_cluster_points,
    )
    consolidated = labels.copy()
    status = np.zeros(len(points), dtype=np.uint8)
    status[stable_surface] = 1
    status[candidate] = 3
    if np.any(stable_surface) and np.any(candidate):
        stable_points = points[stable_surface]
        stable_colors = visual_colors[stable_surface]
        stable_labels = labels[stable_surface]
        tree = cKDTree(stable_points)
        candidate_idx = np.where(candidate)[0]
        assimilated = 0
        metrics = []
        for idx in candidate_idx:
            neighbor_ids = tree.query_ball_point(points[idx], args.neighbor_radius)
            result = maybe_surface_label(
                points[idx],
                visual_colors[idx],
                stable_points,
                stable_colors,
                stable_labels,
                neighbor_ids,
                args.min_plane_neighbors,
                args.max_plane_distance,
                args.max_plane_rms,
                args.max_color_distance,
            )
            if result is None:
                continue
            label, metric = result
            consolidated[idx] = label
            status[idx] = 2
            assimilated += 1
            metrics.append(metric)
    else:
        assimilated = 0
        metrics = []

    out_path = args.output_dir / f"surface_consolidated_frame_{frame_id:04d}.ply"
    write_surface_ply(out_path, points, visual_colors, labels, consolidated, status)
    return {
        "frame": frame_id,
        "status": "ok",
        "points": int(len(points)),
        "semantic_points": int((labels != 0).sum()),
        "stable_surface_points": int(stable_surface.sum()),
        "candidate_points": int(candidate.sum()),
        "assimilated_points": int(assimilated),
        "assimilated_ratio_of_candidates": float(assimilated / max(candidate.sum(), 1)),
        "original_label_counts": label_counts(labels),
        "consolidated_label_counts": label_counts(consolidated),
        "metric_means": {
            key: float(np.mean([m[key] for m in metrics])) if metrics else 0.0
            for key in ["plane_distance", "plane_rms", "color_distance"]
        },
        **cluster_report,
    }


def frames_from_report(report_path: Path) -> list[int]:
    data = json.loads(report_path.read_text())
    return [int(row["frame"]) for row in data.get("frames", []) if row.get("status") == "ok"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--color-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--frames-from-report", type=Path, default=None)
    parser.add_argument("--stable-labels", type=int, nargs="*", default=[2, 3, 10, 12])
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--stable-min-cluster-points", type=int, default=1000)
    parser.add_argument("--candidate-max-cluster-points", type=int, default=120)
    parser.add_argument("--neighbor-radius", type=float, default=0.28)
    parser.add_argument("--min-plane-neighbors", type=int, default=30)
    parser.add_argument("--max-plane-distance", type=float, default=0.08)
    parser.add_argument("--max-plane-rms", type=float, default=0.06)
    parser.add_argument("--max-color-distance", type=float, default=45.0)
    parser.add_argument("--merged-name", default="surface_consolidated_points.ply")
    args = parser.parse_args()

    if args.frames_from_report:
        frames = frames_from_report(args.frames_from_report)
    elif args.frames:
        frames = sorted(set(args.frames))
    else:
        frames = sorted(
            int(path.stem.rsplit("_", 1)[1])
            for path in args.semantic_dir.glob("semantic_frame_*.ply")
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [process_frame(frame, args) for frame in frames]

    merged_points = []
    merged_visual_colors = []
    merged_original = []
    merged_consolidated = []
    merged_status = []
    for frame in frames:
        path = args.output_dir / f"surface_consolidated_frame_{frame:04d}.ply"
        if not path.exists():
            continue
        props, data = read_ascii_ply(path)
        idx = {name: i for i, name in enumerate(props)}
        merged_points.append(data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32))
        merged_visual_colors.append(data[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]].astype(np.uint8))
        merged_original.append(data[:, idx["semantic"]].astype(np.uint8))
        merged_consolidated.append(data[:, idx["consolidated_semantic"]].astype(np.uint8))
        merged_status.append(data[:, idx["surface_status"]].astype(np.uint8))
    if merged_points:
        write_surface_ply(
            args.output_dir / args.merged_name,
            np.concatenate(merged_points),
            np.concatenate(merged_visual_colors),
            np.concatenate(merged_original),
            np.concatenate(merged_consolidated),
            np.concatenate(merged_status),
        )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    report = {
        "semantic_dir": str(args.semantic_dir),
        "color_dir": str(args.color_dir),
        "frames": rows,
        "params": {
            "stable_labels": args.stable_labels,
            "voxel_size": args.voxel_size,
            "stable_min_cluster_points": args.stable_min_cluster_points,
            "candidate_max_cluster_points": args.candidate_max_cluster_points,
            "neighbor_radius": args.neighbor_radius,
            "min_plane_neighbors": args.min_plane_neighbors,
            "max_plane_distance": args.max_plane_distance,
            "max_plane_rms": args.max_plane_rms,
            "max_color_distance": args.max_color_distance,
        },
        "summary": {
            "frame_count": len(rows),
            "ok_count": len(ok_rows),
            "points": int(sum(row.get("points", 0) for row in ok_rows)),
            "stable_surface_points": int(sum(row.get("stable_surface_points", 0) for row in ok_rows)),
            "candidate_points": int(sum(row.get("candidate_points", 0) for row in ok_rows)),
            "assimilated_points": int(sum(row.get("assimilated_points", 0) for row in ok_rows)),
        },
    }
    (args.output_dir / "surface_consolidation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
