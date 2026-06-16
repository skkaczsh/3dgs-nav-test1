#!/usr/bin/env python3
"""Split mixed surface Targets into plane-consistent child Targets.

This post-process operates on Target JSONL files produced by
build_targets_from_masks.py / geometry_guard_targets.py. It does not rerun
SAM/VLM/projection. For large surface targets, it loads the source frame PLY,
fits multiple planes with deterministic RANSAC, then writes child targets whose
point sets are more geometrically coherent before object fusion.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from build_targets_from_masks import connected_components, read_colored_ply, summarize_points


SURFACE_LABELS = {"floor", "wall", "building", "ceiling", "road"}
SEMANTIC_IDS = {
    "unknown": 0,
    "other": 1,
    "wall": 2,
    "floor": 3,
    "ceiling": 4,
    "grass": 5,
    "tree": 6,
    "person": 7,
    "car": 8,
    "railing": 9,
    "building": 10,
    "sky": 11,
    "road": 12,
    "water": 13,
    "furniture": 14,
    "pipe": 15,
    "equipment": 16,
    "ignore": 255,
}


def iter_target_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("targets_frame_*.jsonl") if p.name != "targets_all.jsonl")
    return [path]


def normal_abs_z(points: np.ndarray) -> float:
    stats = summarize_points(points, np.zeros((len(points), 3), dtype=np.uint8)).get("pca", {})
    normal = stats.get("normal") or [0.0, 0.0, 1.0]
    try:
        return abs(float(normal[2]))
    except (TypeError, ValueError, IndexError):
        return 1.0


def bbox_xy_area(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return float(max(hi[0] - lo[0], 0.0) * max(hi[1] - lo[1], 0.0))


def bbox_extents(points: np.ndarray) -> tuple[float, float, float]:
    if len(points) == 0:
        return 0.0, 0.0, 0.0
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return (
        float(max(hi[0] - lo[0], 0.0)),
        float(max(hi[1] - lo[1], 0.0)),
        float(max(hi[2] - lo[2], 0.0)),
    )


def label_from_geometry(old_label: str, points: np.ndarray, args: argparse.Namespace) -> str:
    source_label = old_label
    if old_label == "ceiling":
        return old_label
    z = normal_abs_z(points)
    x_extent, y_extent, z_extent = bbox_extents(points)
    xy_area = x_extent * y_extent
    minor_extent = min(abs(x_extent), abs(y_extent))
    major_extent = max(abs(x_extent), abs(y_extent), 1e-6)
    aspect_ratio = major_extent / max(minor_extent, 1e-6)
    centroid_z = float(points[:, 2].mean()) if len(points) else 0.0

    if (
        args.enable_ceiling_heuristic
        and source_label in set(args.ceiling_source_labels)
        and z >= args.floor_normal_z
        and centroid_z >= args.ceiling_min_z
        and xy_area <= args.ceiling_max_xy_area
        and z_extent <= args.ceiling_max_z_extent
        and minor_extent >= args.ceiling_min_minor_extent
        and aspect_ratio <= args.ceiling_max_aspect_ratio
    ):
        return "ceiling"
    if z >= args.floor_normal_z:
        return "floor"
    if z <= args.wall_normal_z:
        return "wall"
    return "building" if old_label == "building" else old_label


def fit_plane(points: np.ndarray, rng: np.random.Generator, args: argparse.Namespace) -> np.ndarray | None:
    n = len(points)
    if n < 3:
        return None
    sample_points = points
    if n > args.max_fit_points:
        sample_points = points[rng.choice(n, size=args.max_fit_points, replace=False)]
    best_normal = None
    best_d = 0.0
    best_count = 0
    m = len(sample_points)
    for _ in range(args.ransac_iters):
        ids = rng.choice(m, size=3, replace=False)
        p0, p1, p2 = sample_points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p0))
        distances = np.abs(points @ normal + d)
        count = int(np.count_nonzero(distances <= args.plane_distance))
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d
    if best_normal is None or best_count < args.min_plane_points:
        return None
    return np.array([best_normal[0], best_normal[1], best_normal[2], best_d], dtype=np.float64)


def split_points_by_planes(points: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    rng = np.random.default_rng(args.seed + len(points))
    remaining = np.arange(len(points), dtype=np.int64)
    components: list[np.ndarray] = []
    for _ in range(args.max_planes):
        if len(remaining) < args.min_plane_points:
            break
        plane = fit_plane(points[remaining], rng, args)
        if plane is None:
            break
        normal = plane[:3]
        d = float(plane[3])
        distances = np.abs(points[remaining] @ normal + d)
        inliers_local = np.where(distances <= args.plane_distance)[0]
        if len(inliers_local) < args.min_plane_points:
            break
        inliers = remaining[inliers_local]
        conn, residual = connected_components(points[inliers], args.voxel_size, args.min_component_points)
        for comp in conn:
            components.append(inliers[comp])
        keep = np.ones(len(remaining), dtype=bool)
        keep[inliers_local] = False
        remaining = remaining[keep]
    if len(remaining) >= args.min_residual_points:
        conn, _ = connected_components(points[remaining], args.voxel_size, args.min_component_points)
        for comp in conn:
            components.append(remaining[comp])
    components.sort(key=len, reverse=True)
    return components


def update_child(row: dict, suffix: str, point_indices: np.ndarray, frame_points: np.ndarray,
                 frame_colors: np.ndarray, args: argparse.Namespace) -> dict:
    child = dict(row)
    child["split_parent_target_id"] = row.get("target_id")
    child["target_id"] = f"{row.get('target_id')}_{suffix}"
    child["point_indices"] = [int(x) for x in point_indices.tolist()]
    points = frame_points[point_indices]
    colors = frame_colors[point_indices] if len(frame_colors) else np.zeros((len(points), 3), dtype=np.uint8)
    child.update(summarize_points(points, colors))
    old_label = str(row.get("label") or "unknown")
    new_label = label_from_geometry(old_label, points, args)
    if new_label != old_label:
        child["pre_surface_split_label"] = old_label
        child["label"] = new_label
        child["semantic_id"] = SEMANTIC_IDS.get(new_label, 0)
        if new_label in {"floor", "wall", "ceiling", "road"}:
            child["parent_class"] = "surface"
        elif new_label == "building":
            child["parent_class"] = "structure"
    child["surface_split_reason"] = "plane_component"
    return child


def process_file(src: Path, dst: Path, frame_cache: dict[str, tuple[np.ndarray, np.ndarray]],
                 args: argparse.Namespace) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["targets"] += 1
            label = str(row.get("label") or "unknown")
            cluster_size = int(row.get("cluster_size") or len(row.get("point_indices") or []))
            if label not in SURFACE_LABELS or cluster_size < args.min_split_points:
                row["surface_split_reason"] = "passthrough"
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["passthrough"] += 1
                continue
            frame_ply = str(row.get("colored_frame_ply") or "")
            indices = np.array(row.get("point_indices") or [], dtype=np.int64)
            if not frame_ply or indices.size < args.min_split_points:
                row["surface_split_reason"] = "missing_frame_or_points"
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["passthrough"] += 1
                continue
            if frame_ply not in frame_cache:
                frame_cache[frame_ply] = read_colored_ply(Path(frame_ply))
            frame_points, frame_colors = frame_cache[frame_ply]
            valid = indices[(indices >= 0) & (indices < len(frame_points))]
            if len(valid) < args.min_split_points:
                row["surface_split_reason"] = "invalid_indices"
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["passthrough"] += 1
                continue
            components = split_points_by_planes(frame_points[valid], args)
            components = [valid[comp] for comp in components if len(comp) >= args.min_component_points]
            if len(components) < 2:
                row["surface_split_reason"] = "single_plane"
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["single_plane"] += 1
                continue
            counts["split_targets"] += 1
            counts["children"] += len(components)
            for i, comp_indices in enumerate(components):
                child = update_child(row, f"p{i:02d}", comp_indices, frame_points, frame_colors, args)
                counts[f"child_label:{child.get('label')}"] += 1
                if child.get("label") != label:
                    counts[f"child_change:{label}->{child.get('label')}"] += 1
                fout.write(json.dumps(child, ensure_ascii=False) + "\n")
    return {"source": str(src), "output": str(dst), "counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-targets", type=Path, required=True)
    parser.add_argument("--output-targets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-split-points", type=int, default=240)
    parser.add_argument("--min-plane-points", type=int, default=120)
    parser.add_argument("--min-component-points", type=int, default=40)
    parser.add_argument("--min-residual-points", type=int, default=120)
    parser.add_argument("--plane-distance", type=float, default=0.055)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--max-planes", type=int, default=4)
    parser.add_argument("--ransac-iters", type=int, default=96)
    parser.add_argument("--max-fit-points", type=int, default=1200)
    parser.add_argument("--floor-normal-z", type=float, default=0.72)
    parser.add_argument("--wall-normal-z", type=float, default=0.40)
    parser.add_argument("--enable-ceiling-heuristic", action="store_true")
    parser.add_argument("--ceiling-source-labels", nargs="+", default=["floor", "building"])
    parser.add_argument("--ceiling-min-z", type=float, default=2.0)
    parser.add_argument("--ceiling-max-xy-area", type=float, default=8.0)
    parser.add_argument("--ceiling-max-z-extent", type=float, default=0.35)
    parser.add_argument("--ceiling-min-minor-extent", type=float, default=0.30)
    parser.add_argument("--ceiling-max-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    files = iter_target_files(args.input_targets)
    if not files:
        raise SystemExit(f"no target jsonl files found: {args.input_targets}")

    total = Counter()
    reports = []
    frame_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for src in files:
        dst = args.output_targets / src.name if args.input_targets.is_dir() else args.output_targets
        report = process_file(src, dst, frame_cache, args)
        reports.append(report)
        total.update(report["counts"])

    if args.input_targets.is_dir():
        merged = args.output_targets / "targets_all.jsonl"
        with merged.open("w", encoding="utf-8") as fout:
            for p in sorted(args.output_targets.glob("targets_frame_*.jsonl")):
                if p.name != "targets_all.jsonl":
                    fout.write(p.read_text(encoding="utf-8"))

    summary = {
        "input_targets": str(args.input_targets),
        "output_targets": str(args.output_targets),
        "files": len(files),
        "summary": dict(total),
        "split_ratio": float(total.get("split_targets", 0) / max(total.get("targets", 0), 1)),
        "file_reports": reports[:20],
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["files", "summary", "split_ratio"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
