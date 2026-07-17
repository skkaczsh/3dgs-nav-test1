#!/usr/bin/env python3
"""Run the official Superpoint Graph partition on one PLY file."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.geometry_input_contract import geometry_only_semantic_fields
from scripts.optimize_patch_graph_energy import compute_patch_stats, read_region_input


def configure_official_spg_backend(root: Path) -> Path:
    """Expose the official partition extension modules from one explicit root."""
    partition = root / "partition"
    if not partition.is_dir():
        raise RuntimeError(
            f"official Superpoint Graph partition directory is missing: {partition}. "
            "Run scripts/setup_official_superpoint_graph.sh first."
        )
    if str(partition) not in sys.path:
        sys.path.insert(0, str(partition))
    return partition


def read_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype("float32")
    if all(name in vertex.dtype.names for name in ("red", "green", "blue")):
        rgb = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype("uint8")
    else:
        rgb = np.zeros((len(xyz), 3), dtype="uint8")
    return np.ascontiguousarray(xyz), rgb


def crop_points(
    xyz: np.ndarray, rgb: np.ndarray, bbox_min: list[float] | None, bbox_max: list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial smoke crops preserve local density and cannot reuse global labels."""
    if bbox_min is None and bbox_max is None:
        return xyz, rgb
    if bbox_min is None or bbox_max is None:
        raise ValueError("--bbox-min and --bbox-max must be provided together")
    low = np.asarray(bbox_min, dtype=np.float32)
    high = np.asarray(bbox_max, dtype=np.float32)
    if np.any(high <= low):
        raise ValueError("--bbox-max must be strictly greater than --bbox-min")
    keep = np.all((xyz >= low) & (xyz <= high), axis=1)
    if not np.any(keep):
        raise ValueError("spatial crop contains no points")
    return np.ascontiguousarray(xyz[keep]), np.ascontiguousarray(rgb[keep])


def progress(stage: str, **values: object) -> None:
    """Emit durable stages; Cut Pursuit's carriage-return progress is not parseable."""
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def configure_omp_threads(threads: int, allow_nondeterministic: bool) -> None:
    """Keep Cut Pursuit ownership reproducible unless exploration opts out."""
    if threads < 1:
        raise ValueError("--omp-threads must be positive")
    if threads != 1 and not allow_nondeterministic:
        raise ValueError(
            "multi-threaded Cut Pursuit is nondeterministic; pass "
            "--allow-nondeterministic-omp only for non-production experiments"
        )
    os.environ["OMP_NUM_THREADS"] = str(threads)


def write_random_color_ply(path: Path, xyz: np.ndarray, labels: np.ndarray) -> None:
    rng = random.Random(0)
    unique, inverse = np.unique(labels, return_inverse=True)
    colors = np.array([[rng.randrange(256), rng.randrange(256), rng.randrange(256)] for _ in unique], dtype="uint8")
    rgb = colors[inverse]

    vertex = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("object", "i4"), ("semantic", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    vertex["object"] = labels.astype("int32", copy=False)
    vertex["semantic"] = 0
    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(str(path))


def geometry_by_object(xyz: np.ndarray, labels: np.ndarray, region_input: Path | None) -> dict[int, dict[str, object]]:
    """Derive non-semantic geometry from the same points that Cut Pursuit segmented."""
    if region_input is not None:
        arrays, _src, _dst = read_region_input(region_input)
        if len(arrays["xyz"]) != len(xyz):
            raise ValueError(f"region input count differs from PLY: {len(arrays['xyz'])} != {len(xyz)}")
        probes = np.unique(np.array([0, len(xyz) // 2, len(xyz) - 1], dtype=np.int64))
        delta = float(np.abs(arrays["xyz"][probes] - xyz[probes]).max())
        if delta > 0.002:
            raise ValueError(f"region input does not share PLY order (max probe delta={delta:.6f}m)")
        return {
            int(pid): {"geometry_type": stat.geometry_type, "source": "region_input"}
            for pid, stat in compute_patch_stats(arrays, labels).items()
        }

    # ponytail: PCA is enough here.  Keep richer structural fields as evidence,
    # not a second segmentation whose labels could contradict Superpoints.
    label_ids = labels.astype(np.int64, copy=False)
    count = np.bincount(label_ids)
    mean = np.column_stack([
        np.bincount(label_ids, weights=xyz[:, axis], minlength=len(count)) / np.maximum(count, 1)
        for axis in range(3)
    ])
    second = np.empty((len(count), 3, 3), dtype=np.float64)
    for row in range(3):
        for col in range(3):
            second[:, row, col] = np.bincount(
                label_ids,
                weights=xyz[:, row].astype(np.float64) * xyz[:, col].astype(np.float64),
                minlength=len(count),
            ) / np.maximum(count, 1)
    covariance = second - mean[:, :, None] * mean[:, None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    result: dict[int, dict[str, object]] = {}
    for object_id in np.flatnonzero(count):
        values = np.maximum(eigenvalues[object_id], 0.0)
        scale = max(float(values[2]), 1e-9)
        planarity = float((values[1] - values[0]) / scale)
        linearity = float((values[2] - values[1]) / scale)
        normal = eigenvectors[object_id, :, 0]
        verticality = 1.0 - abs(float(normal[2]))
        if count[object_id] < 10:
            geometry_type = "unknown"
        elif linearity >= 0.65 and planarity < 0.35:
            geometry_type = "thin_linear"
        elif planarity >= 0.55 and verticality <= 0.28:
            geometry_type = "horizontal"
        elif planarity >= 0.55 and verticality >= 0.62:
            geometry_type = "vertical"
        else:
            geometry_type = "rough_mixed"
        result[int(object_id)] = {
            "geometry_type": geometry_type,
            "source": "superpoint_pca",
            "normal": np.round(normal, 5).tolist(),
            "planarity": round(planarity, 5),
            "linearity": round(linearity, 5),
            "verticality": round(verticality, 5),
        }
    return result


def write_objects_jsonl(path: Path, xyz: np.ndarray, labels: np.ndarray, geometry: dict[int, dict[str, object]]) -> None:
    """Write object metadata in O(points + superpoints), never one full scan per id."""
    label_ids = labels.astype(np.int64, copy=False)
    count = np.bincount(label_ids)
    active = np.flatnonzero(count)
    sums = np.column_stack([
        np.bincount(label_ids, weights=xyz[:, axis], minlength=len(count))
        for axis in range(3)
    ])
    minimum = np.full((len(count), 3), np.inf, dtype=np.float32)
    maximum = np.full((len(count), 3), -np.inf, dtype=np.float32)
    for axis in range(3):
        np.minimum.at(minimum[:, axis], label_ids, xyz[:, axis])
        np.maximum.at(maximum[:, axis], label_ids, xyz[:, axis])
    centroids = sums / np.maximum(count[:, None], 1)
    with path.open("w", encoding="utf-8") as f:
        for label in active:
            geometry_meta = geometry.get(int(label), {"geometry_type": "unknown", "source": "unavailable"})
            geometry_type = str(geometry_meta["geometry_type"])
            row = {
                "object_id": int(label),
                "label": "official_superpoint",
                "count": int(count[label]),
                "bbox_min": minimum[label].round(4).tolist(),
                "bbox_max": maximum[label].round(4).tolist(),
                "centroid": centroids[label].round(4).tolist(),
                "geometry_type": geometry_type,
                "geometry_features": geometry_meta,
                **geometry_only_semantic_fields(geometry_type),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k-nn-adj", type=int, default=10)
    parser.add_argument("--k-nn-geof", type=int, default=45)
    parser.add_argument("--reg-strength", type=float, default=0.1)
    parser.add_argument("--lambda-edge-weight", type=float, default=1.0)
    parser.add_argument("--stride-preview", type=int, default=10)
    parser.add_argument("--omp-threads", type=int, default=1,
                        help="Cut Pursuit OpenMP threads; production default 1 is reproducible.")
    parser.add_argument("--allow-nondeterministic-omp", action="store_true",
                        help="Required with --omp-threads > 1 for exploratory speed experiments.")
    parser.add_argument("--region-input", type=Path, help="Optional same-order GPRG input for geometry-only object metadata.")
    parser.add_argument("--labels-input", type=Path, help="Reuse an existing same-order official_superpoints_labels.npy instead of rerunning Cut Pursuit.")
    parser.add_argument("--bbox-min", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Optional spatial smoke-crop lower bound; cannot reuse global labels/metadata.")
    parser.add_argument("--bbox-max", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Optional spatial smoke-crop upper bound; cannot reuse global labels/metadata.")
    parser.add_argument(
        "--superpoint-graph-root", type=Path,
        default=Path(os.environ.get("SUPERPOINT_GRAPH_ROOT", REPO_ROOT / "third_party" / "superpoint_graph")),
        help="Official loicland/superpoint_graph checkout containing compiled partition modules.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_omp_threads(args.omp_threads, args.allow_nondeterministic_omp)
    progress("runtime", omp_threads=args.omp_threads, deterministic=args.omp_threads == 1)

    if (args.bbox_min is None) != (args.bbox_max is None):
        raise ValueError("--bbox-min and --bbox-max must be provided together")
    if args.bbox_min is not None and (args.labels_input or args.region_input):
        raise ValueError("a spatial smoke crop cannot reuse global --labels-input or --region-input")
    progress("load_input", input=str(args.input))
    xyz, rgb = read_ply_xyz_rgb(args.input)
    input_points = len(xyz)
    xyz, _rgb = crop_points(xyz, rgb, args.bbox_min, args.bbox_max)
    progress("input_ready", input_points=int(input_points), active_points=int(len(xyz)))
    if args.labels_input:
        progress("load_labels", labels_input=str(args.labels_input))
        labels = np.load(args.labels_input).astype(np.uint32, copy=False)
        if len(labels) != len(xyz):
            raise ValueError(f"labels count differs from PLY: {len(labels)} != {len(xyz)}")
    else:
        configure_official_spg_backend(args.superpoint_graph_root)
        try:
            from graphs import compute_graph_nn_2
            import libcp
            import libply_c
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "official Superpoint Graph extensions are unavailable. "
                "Run scripts/setup_official_superpoint_graph.sh and pass --superpoint-graph-root if needed."
            ) from exc

        progress("build_knn", k_adj=args.k_nn_adj, k_geof=args.k_nn_geof)
        graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
        progress("compute_geof")
        geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype("float32")
        geof[:, 3] = 2.0 * geof[:, 3]
        edge_weight = np.array(
            1.0 / (args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])),
            dtype="float32",
        )
        progress("cut_pursuit", regularization=args.reg_strength, edges=int(len(graph_nn["source"])))
        components, in_component = libcp.cutpursuit(
            geof,
            graph_nn["source"],
            graph_nn["target"],
            edge_weight,
            args.reg_strength,
        )
        labels = np.asarray(in_component, dtype=np.uint32)
        progress("cut_pursuit_done", components=int(len(components)))
    progress("summarize_geometry")
    geometry = geometry_by_object(xyz, labels, args.region_input)

    labels_path = args.output_dir / "official_superpoints_labels.npy"
    np.save(labels_path, labels)

    progress("write_outputs")
    full_ply = args.output_dir / "official_superpoints_random_color.ply"
    write_random_color_ply(full_ply, xyz, labels)

    stride = max(1, args.stride_preview)
    preview_ply = args.output_dir / f"official_superpoints_random_color_stride{stride}.ply"
    write_random_color_ply(preview_ply, xyz[::stride], labels[::stride])
    write_objects_jsonl(args.output_dir / "official_superpoints_objects.jsonl", xyz, labels, geometry)

    counts = np.bincount(labels)
    report = {
        "input": str(args.input),
        "input_points": int(input_points),
        "points": int(len(xyz)),
        "superpoints": int(labels.max() + 1) if len(labels) else 0,
        "nonempty_superpoints": int((counts > 0).sum()),
        "median_points_per_superpoint": float(np.median(counts[counts > 0])) if np.any(counts > 0) else 0.0,
        "largest_superpoints": sorted([int(x) for x in counts], reverse=True)[:20],
        "params": {
            "k_nn_adj": args.k_nn_adj,
            "k_nn_geof": args.k_nn_geof,
            "reg_strength": args.reg_strength,
            "lambda_edge_weight": args.lambda_edge_weight,
            "omp_threads": args.omp_threads,
            "deterministic": args.omp_threads == 1,
            "region_input": str(args.region_input) if args.region_input else None,
            "labels_input": str(args.labels_input) if args.labels_input else None,
            "superpoint_graph_root": str(args.superpoint_graph_root),
            "bbox_min": args.bbox_min,
            "bbox_max": args.bbox_max,
        },
        "outputs": {
            "labels": str(labels_path),
            "full_ply": str(full_ply),
            "preview_ply": str(preview_ply),
        },
    }
    (args.output_dir / "official_superpoints_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    progress("complete", superpoints=report["superpoints"], outputs=report["outputs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
