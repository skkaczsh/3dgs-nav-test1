#!/usr/bin/env python3
"""Create a small, geometry-stratified Superpoint evidence sample."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

try:
    from scripts.build_raw_lx_voxel_cloud import read_lx_points, read_lx_sections
except ModuleNotFoundError:  # Supports direct `python scripts/...` execution.
    from build_raw_lx_voxel_cloud import read_lx_points, read_lx_sections


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def supported_ids(path: Path) -> set[int]:
    return {
        int(row["object_id"])
        for row in read_jsonl(path)
        if row.get("top_source_frames")
    }


def select_rows(
    rows: list[dict[str, Any]], support: set[int], per_geometry: int, min_object_points: int, seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["object_id"]) in support and int(row.get("count") or 0) >= min_object_points:
            groups[str(row.get("geometry_type") or "unknown")].append(row)
    rng = np.random.default_rng(seed)
    selected = []
    for geometry in sorted(groups):
        candidates = groups[geometry]
        take = min(per_geometry, len(candidates))
        if take:
            selected.extend(candidates[i] for i in rng.choice(len(candidates), size=take, replace=False))
    return sorted(selected, key=lambda row: int(row["object_id"]))


def source_frame_ids(path: Path, selected_ids: set[int]) -> dict[int, list[int]]:
    """Return the proven raw-section frames for each selected Superpoint."""
    out: dict[int, list[int]] = {}
    for row in read_jsonl(path):
        object_id = int(row["object_id"])
        if object_id not in selected_ids:
            continue
        frames = [int(item["frame_id"]) for item in row.get("top_source_frames", [])]
        if frames:
            out[object_id] = frames
    return out


def reservoir_update(
    points: dict[int, np.ndarray],
    keys: dict[int, np.ndarray],
    object_id: int,
    new_points: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> None:
    """Keep an unbiased bounded sample without retaining all source points."""
    if len(new_points) == 0:
        return
    new_keys = rng.random(len(new_points))
    old_points = points.get(object_id, np.empty((0, 3), dtype=np.float32))
    old_keys = keys.get(object_id, np.empty(0, dtype=np.float64))
    all_points = np.vstack([old_points, new_points.astype(np.float32, copy=False)])
    all_keys = np.concatenate([old_keys, new_keys])
    if len(all_points) > max_points:
        keep = np.argpartition(all_keys, -max_points)[-max_points:]
        all_points, all_keys = all_points[keep], all_keys[keep]
    points[object_id], keys[object_id] = all_points, all_keys


def source_aware_samples(
    reference_xyz: np.ndarray,
    labels: np.ndarray,
    source_support: dict[int, list[int]],
    lx_path: Path,
    max_match_distance: float,
    max_points_per_object: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """Materialize only raw points that actually supported selected labels.

    The LX coordinate audit established that these section points are already
    world coordinates. Reusing the provenance KD-tree match keeps image
    evidence and source support on the same spatial support rather than
    sampling arbitrary distant parts of a global Superpoint.
    """
    tree = cKDTree(reference_xyz, compact_nodes=False, balanced_tree=False)
    selected = set(source_support)
    sections = read_lx_sections(lx_path)
    frames = sorted({frame for values in source_support.values() for frame in values})
    samples: dict[tuple[int, int], np.ndarray] = {}
    keys: dict[tuple[int, int], np.ndarray] = {}
    rng = np.random.default_rng(seed)
    raw_points = matched_points = selected_points = 0
    with lx_path.open("rb") as handle:
        for frame_id in frames:
            if frame_id < 0 or frame_id >= len(sections):
                continue
            raw = read_lx_points(handle, sections[frame_id])
            raw_points += len(raw)
            if not len(raw):
                continue
            distances, nearest = tree.query(raw, k=1, workers=-1)
            keep = distances <= float(max_match_distance)
            matched_points += int(keep.sum())
            if not np.any(keep):
                continue
            matched_labels = labels[nearest[keep]]
            matched_xyz = raw[keep]
            for object_id in np.unique(matched_labels):
                oid = int(object_id)
                if oid not in selected or frame_id not in source_support[oid]:
                    continue
                object_points = matched_xyz[matched_labels == oid]
                selected_points += len(object_points)
                # Keep frame membership. A global Superpoint sample cannot be
                # projected into a source frame as though every point was seen there.
                per_frame_limit = max(1, max_points_per_object // max(1, len(source_support[oid])))
                framed = np.column_stack((object_points, np.full(len(object_points), frame_id, dtype=np.float32)))
                reservoir_update(samples, keys, (oid, frame_id), framed, per_frame_limit, rng)
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for (object_id, _frame_id), points in samples.items():
        grouped[object_id].append(points)
    return {object_id: np.vstack(chunks) for object_id, chunks in grouped.items()}, {
        "source_frames": len(frames),
        "raw_lx_points": int(raw_points),
        "matched_reference_points": int(matched_points),
        "selected_source_points": int(selected_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path)
    parser.add_argument("--source-support", type=Path)
    parser.add_argument("--selected-objects-jsonl", type=Path, help="Use a preselected candidate set instead of random geometry sampling.")
    parser.add_argument("--source-aware", action="store_true",
                        help="Sample only raw LX points from each object's proven source frames.")
    parser.add_argument("--lx", type=Path, help="MANIFOLD .lx required by --source-aware.")
    parser.add_argument("--max-match-distance", type=float, default=0.05,
                        help="Reference KD-tree radius for --source-aware; match provenance default.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-geometry", type=int, default=20)
    parser.add_argument("--min-object-points", type=int, default=0)
    parser.add_argument("--max-points-per-object", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.selected_objects_jsonl:
        selected = read_jsonl(args.selected_objects_jsonl)
        support = set()
        selection_source = "selected_objects_jsonl"
    else:
        if not args.objects_jsonl or not args.source_support:
            raise SystemExit("--objects-jsonl and --source-support are required without --selected-objects-jsonl")
        support = supported_ids(args.source_support)
        selected = select_rows(read_jsonl(args.objects_jsonl), support, args.per_geometry, args.min_object_points, args.seed)
        selection_source = "random_geometry_sample"
    selected_ids = np.array([int(row["object_id"]) for row in selected], dtype=np.int32)
    labels = np.load(args.labels).astype(np.int32, copy=False)
    vertex = PlyData.read(str(args.reference_ply))["vertex"].data
    if len(vertex) != len(labels):
        raise ValueError(f"reference/label count mismatch: {len(vertex)} != {len(labels)}")

    chunks = []
    source_report: dict[str, int] = {}
    if args.source_aware:
        if not args.source_support or not args.lx:
            raise SystemExit("--source-aware requires --source-support and --lx")
        xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
        support = source_frame_ids(args.source_support, set(selected_ids.tolist()))
        samples, source_report = source_aware_samples(
            xyz, labels, support, args.lx, args.max_match_distance, args.max_points_per_object, args.seed,
        )
        for object_id, points in samples.items():
            sample = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("object", "i4"), ("source_frame", "i4")])
            sample["x"], sample["y"], sample["z"] = points[:, 0], points[:, 1], points[:, 2]
            sample["object"] = object_id
            sample["source_frame"] = points[:, 3].astype(np.int32, copy=False)
            chunks.append(sample)
    else:
        rng = np.random.default_rng(args.seed)
        for object_id in selected_ids:
            indices = np.flatnonzero(labels == object_id)
            if len(indices) > args.max_points_per_object:
                indices = rng.choice(indices, size=args.max_points_per_object, replace=False)
            sample = np.empty(len(indices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("object", "i4")])
            sample["x"], sample["y"], sample["z"] = vertex["x"][indices], vertex["y"][indices], vertex["z"][indices]
            sample["object"] = object_id
            chunks.append(sample)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(np.concatenate(chunks) if chunks else np.empty(0, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("object", "i4")]), "vertex")], text=True).write(str(output / "object_samples.ply"))
    with (output / "objects.jsonl").open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "selected_objects": len(selected), "selection_source": selection_source,
        "sampling_mode": "source_aware_lx" if args.source_aware else "reference_uniform",
        "per_geometry": args.per_geometry, "min_object_points": args.min_object_points,
        "supported_pool": len(support), "sample_points": int(sum(len(chunk) for chunk in chunks)),
        "objects_with_samples": len({int(chunk["object"][0]) for chunk in chunks if len(chunk)}),
        **source_report,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
