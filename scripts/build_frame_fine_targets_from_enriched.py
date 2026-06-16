#!/usr/bin/env python3
"""Build per-frame fine-object Target JSONL records from enriched fine points."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from project_semantic import LABEL_NAMES


PARENT_BY_LABEL = {
    "wall": "surface",
    "floor": "surface",
    "ceiling": "surface",
    "road": "surface",
    "building": "structure",
    "railing": "fine_object",
    "pipe": "fine_object",
    "equipment": "fine_object",
    "furniture": "fine_object",
    "tree": "vegetation",
    "grass": "vegetation",
    "sky": "background",
    "unknown": "unknown",
    "ignore": "ignore",
}


def read_ascii_ply(path: Path) -> tuple[list[str], np.ndarray]:
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
        return props, np.empty((0, len(props)), dtype=np.float64)
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return props, data


def neighbor_offsets() -> list[tuple[int, int, int]]:
    return [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]


NEIGHBORS = neighbor_offsets()


def connected_components(points: np.ndarray, voxel_size: float, min_points: int) -> tuple[list[np.ndarray], np.ndarray]:
    if len(points) == 0:
        return [], np.zeros(0, dtype=bool)
    voxels = np.floor(points / voxel_size).astype(np.int64)
    by_voxel: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, v in enumerate(voxels):
        by_voxel[(int(v[0]), int(v[1]), int(v[2]))].append(i)
    visited: set[tuple[int, int, int]] = set()
    components: list[np.ndarray] = []
    residual = np.zeros(len(points), dtype=bool)
    for start in by_voxel:
        if start in visited:
            continue
        q: deque[tuple[int, int, int]] = deque([start])
        visited.add(start)
        comp_indices: list[int] = []
        while q:
            cell = q.popleft()
            comp_indices.extend(by_voxel[cell])
            for off in NEIGHBORS:
                nxt = (cell[0] + off[0], cell[1] + off[1], cell[2] + off[2])
                if nxt in by_voxel and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        comp = np.array(sorted(comp_indices), dtype=np.int64)
        if len(comp) >= min_points:
            components.append(comp)
        else:
            residual[comp] = True
    components.sort(key=len, reverse=True)
    return components, residual


def pca_summary(points: np.ndarray) -> dict:
    if len(points) < 3:
        return {"normal": [0.0, 0.0, 1.0], "linearity": 0.0, "planarity": 0.0, "scattering": 0.0}
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    denom = max(float(vals[0]), 1e-12)
    normal = vecs[:, -1]
    return {
        "normal": [float(x) for x in normal.tolist()],
        "linearity": float((vals[0] - vals[1]) / denom),
        "planarity": float((vals[1] - vals[2]) / denom),
        "scattering": float(vals[2] / denom),
    }


def dominant_vote(counter: Counter[int]) -> tuple[int, float]:
    if not counter:
        return -1, 0.0
    winner, votes = max(counter.items(), key=lambda kv: (kv[1], kv[0]))
    total = max(sum(counter.values()), 1)
    return int(winner), float(votes / total)


def bbox_gap(a_xyz: np.ndarray, b_xyz: np.ndarray) -> float:
    amin = a_xyz.min(axis=0)
    amax = a_xyz.max(axis=0)
    bmin = b_xyz.min(axis=0)
    bmax = b_xyz.max(axis=0)
    gap = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    return float(np.linalg.norm(gap))


def merge_components_with_same_candidate(
    rows: np.ndarray,
    components: list[np.ndarray],
    idx: dict[str, int],
    args: argparse.Namespace,
) -> list[np.ndarray]:
    if len(components) <= 1:
        return components

    ratio_min = float(getattr(args, "in_frame_candidate_ratio", 0.8))
    centroid_limit = float(getattr(args, "in_frame_centroid_distance", 0.5))
    bbox_limit = float(getattr(args, "in_frame_bbox_distance", 0.5))
    color_limit = float(getattr(args, "in_frame_color_distance", 80.0))

    summaries: list[dict] = []
    for comp in components:
        comp_rows = rows[comp]
        comp_xyz = comp_rows[:, [idx["x"], idx["y"], idx["z"]]]
        comp_rgb = comp_rows[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]]
        candidate_counts = Counter(int(x) for x in comp_rows[:, idx["accepted_candidate"]].tolist())
        source_cluster_counts = Counter(int(x) for x in comp_rows[:, idx["source_cluster"]].tolist())
        dominant_candidate, candidate_ratio = dominant_vote(candidate_counts)
        dominant_source_cluster, source_ratio = dominant_vote(source_cluster_counts)
        summaries.append(
            {
                "indices": comp,
                "xyz": comp_xyz,
                "centroid": comp_xyz.mean(axis=0),
                "mean_color": comp_rgb.mean(axis=0),
                "dominant_candidate": dominant_candidate,
                "candidate_ratio": candidate_ratio,
                "dominant_source_cluster": dominant_source_cluster,
                "source_ratio": source_ratio,
            }
        )

    parent = list(range(len(summaries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(summaries)):
        a = summaries[i]
        if a["dominant_candidate"] <= 0 or a["candidate_ratio"] < ratio_min:
            continue
        for j in range(i + 1, len(summaries)):
            b = summaries[j]
            if b["dominant_candidate"] != a["dominant_candidate"] or b["candidate_ratio"] < ratio_min:
                continue
            centroid_dist = float(np.linalg.norm(a["centroid"] - b["centroid"]))
            bbox_dist = bbox_gap(a["xyz"], b["xyz"])
            color_dist = float(np.linalg.norm(a["mean_color"] - b["mean_color"]))
            same_source_cluster = (
                a["dominant_source_cluster"] > 0
                and b["dominant_source_cluster"] > 0
                and a["dominant_source_cluster"] == b["dominant_source_cluster"]
                and a["source_ratio"] >= ratio_min
                and b["source_ratio"] >= ratio_min
            )
            same_source_and_near = same_source_cluster and (
                centroid_dist <= centroid_limit or bbox_dist <= bbox_limit
            )
            if same_source_and_near or (
                color_dist <= color_limit and (centroid_dist <= centroid_limit or bbox_dist <= bbox_limit)
            ):
                union(i, j)

    merged: dict[int, list[np.ndarray]] = defaultdict(list)
    for i, summary in enumerate(summaries):
        merged[find(i)].append(summary["indices"])

    merged_components = [
        np.array(sorted(np.concatenate(parts).tolist()), dtype=np.int64)
        for parts in merged.values()
    ]
    merged_components.sort(key=len, reverse=True)
    return merged_components


def object_color(target_number: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(target_number * 97 + 311)
    return tuple(int(x) for x in rng.integers(70, 245, 3))


def colored_frame_ply_path(frame: int, args: argparse.Namespace) -> str:
    colored_dir = getattr(args, "colored_frame_dir", None)
    if not colored_dir:
        return ""
    return str(Path(colored_dir) / f"frame_{frame:04d}.ply")


def build_targets(props: list[str], data: np.ndarray, args: argparse.Namespace) -> tuple[list[dict], dict, list[tuple[np.ndarray, int]]]:
    idx = {name: i for i, name in enumerate(props)}
    required = {
        "x",
        "y",
        "z",
        "semantic",
        "accepted_candidate",
        "source_type",
        "source_cluster",
        "subcluster",
        "visual_red",
        "visual_green",
        "visual_blue",
        "frame",
        "camera",
        "mask",
        "point_index",
    }
    if not required.issubset(idx):
        raise ValueError(f"missing required enriched PLY fields. required={required} available={props}")

    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    skipped = Counter()
    for row_i, row in enumerate(data):
        frame = int(row[idx["frame"]])
        camera = int(row[idx["camera"]])
        mask = int(row[idx["mask"]])
        semantic = int(row[idx["semantic"]])
        if frame < 0 or camera < 0 or mask < 0:
            skipped["missing_trace"] += 1
            continue
        if semantic in {0, 11, 255}:
            skipped["ignored_semantic"] += 1
            continue
        groups[(frame, camera, mask, semantic)].append(row_i)

    targets: list[dict] = []
    target_points: list[tuple[np.ndarray, int]] = []
    residual_points = 0
    group_component_counts = Counter()
    for (frame, camera, mask, semantic), row_indices in sorted(groups.items()):
        rows = data[np.array(row_indices, dtype=np.int64)]
        xyz = rows[:, [idx["x"], idx["y"], idx["z"]]]
        components, residual = connected_components(xyz, args.voxel_size, args.min_target_points)
        components = merge_components_with_same_candidate(rows, components, idx, args)
        residual_points += int(residual.sum())
        group_component_counts[len(components)] += 1
        for component_number, comp in enumerate(components):
            comp_rows = rows[comp]
            comp_xyz = comp_rows[:, [idx["x"], idx["y"], idx["z"]]]
            visual = comp_rows[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]]
            label = LABEL_NAMES.get(semantic, "unknown")
            candidate_counts = Counter(int(x) for x in comp_rows[:, idx["accepted_candidate"]].tolist())
            source_cluster_counts = Counter(int(x) for x in comp_rows[:, idx["source_cluster"]].tolist())
            subcluster_counts = Counter(int(x) for x in comp_rows[:, idx["subcluster"]].tolist())
            point_indices = sorted(set(int(x) for x in comp_rows[:, idx["point_index"]].tolist() if int(x) >= 0))
            target_id = f"fine_t_{frame:06d}_cam{camera}_mask{mask:04d}_sem{semantic}_cc{component_number:02d}"
            target = {
                "target_id": target_id,
                "frame_id": int(frame),
                "cam_id": int(camera),
                "mask_id": int(mask),
                "label": label,
                "label_id": int(semantic),
                "parent_class": PARENT_BY_LABEL.get(label, "other"),
                "confidence": 1.0,
                "image_path": "",
                "mask_path": "",
                "raw_frame_ply": colored_frame_ply_path(int(frame), args),
                "colored_frame_ply": colored_frame_ply_path(int(frame), args),
                "point_indices": point_indices,
                "bbox_3d": {
                    "min": [float(x) for x in comp_xyz.min(axis=0)],
                    "max": [float(x) for x in comp_xyz.max(axis=0)],
                },
                "centroid": [float(x) for x in comp_xyz.mean(axis=0)],
                "mean_color": [float(x) for x in visual.mean(axis=0)],
                "pca": pca_summary(comp_xyz),
                "cluster_size": int(len(comp_xyz)),
                "accepted_candidate_votes": {str(k): int(v) for k, v in candidate_counts.items()},
                "source_cluster_votes": {str(k): int(v) for k, v in source_cluster_counts.items()},
                "subcluster_votes": {str(k): int(v) for k, v in subcluster_counts.items()},
            }
            targets.append(target)
            target_points.append((comp_xyz, len(targets)))

    report = {
        "source_points": int(len(data)),
        "groups": int(len(groups)),
        "targets": int(len(targets)),
        "target_points": int(sum(t["cluster_size"] for t in targets)),
        "small_residual_points": int(residual_points),
        "skipped_points": dict(skipped),
        "group_component_counts": {str(k): int(v) for k, v in sorted(group_component_counts.items())},
        "label_counts": dict(Counter(t["label"] for t in targets)),
        "target_point_stats": {
            "min": int(min([t["cluster_size"] for t in targets], default=0)),
            "max": int(max([t["cluster_size"] for t in targets], default=0)),
            "mean": float(np.mean([t["cluster_size"] for t in targets])) if targets else 0.0,
        },
    }
    return sorted(targets, key=lambda t: (t["frame_id"], t["cam_id"], t["mask_id"], t["target_id"])), report, target_points


def write_targets(output_dir: Path, targets: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for target in targets:
        by_frame[int(target["frame_id"])].append(target)
    for frame, rows in sorted(by_frame.items()):
        with (output_dir / f"targets_frame_{frame:06d}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "targets_all.jsonl").open("w", encoding="utf-8") as f:
        for row in targets:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_target_ply(path: Path, target_points: list[tuple[np.ndarray, int]]) -> int:
    total = int(sum(len(points) for points, _ in target_points))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int target\n")
        f.write("end_header\n")
        for points, target_number in target_points:
            color = object_color(target_number)
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {color[0]} {color[1]} {color[2]} {target_number}\n")
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colored-frame-dir", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--min-target-points", type=int, default=5)
    parser.add_argument("--in-frame-candidate-ratio", type=float, default=0.8)
    parser.add_argument("--in-frame-centroid-distance", type=float, default=0.5)
    parser.add_argument("--in-frame-bbox-distance", type=float, default=0.5)
    parser.add_argument("--in-frame-color-distance", type=float, default=80.0)
    parser.add_argument("--write-ply", action="store_true")
    args = parser.parse_args()

    props, data = read_ascii_ply(args.enriched_ply)
    targets, report, target_points = build_targets(props, data, args)
    write_targets(args.output_dir, targets)
    if args.write_ply:
        report["target_ply"] = str(args.output_dir / "frame_fine_targets.ply")
        report["target_ply_points"] = write_target_ply(args.output_dir / "frame_fine_targets.ply", target_points)
    report["enriched_ply"] = str(args.enriched_ply)
    report["output_dir"] = str(args.output_dir)
    report["params"] = {
        "voxel_size": args.voxel_size,
        "min_target_points": args.min_target_points,
        "in_frame_candidate_ratio": args.in_frame_candidate_ratio,
        "in_frame_centroid_distance": args.in_frame_centroid_distance,
        "in_frame_bbox_distance": args.in_frame_bbox_distance,
        "in_frame_color_distance": args.in_frame_color_distance,
    }
    report_path = args.output_dir / "frame_fine_targets_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["source_points", "groups", "targets", "target_points", "small_residual_points"]}, indent=2))


if __name__ == "__main__":
    main()
