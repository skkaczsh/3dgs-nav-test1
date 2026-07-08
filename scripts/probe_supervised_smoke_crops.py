#!/usr/bin/env python3
"""Small feature-separability probe for supervised smoke crop inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "xyz": ("x", "y", "z"),
    "xyz_rgb": ("x", "y", "z", "red", "green", "blue"),
    "xyz_normal": ("x", "y", "z", "normal_x", "normal_y", "normal_z"),
    "xyz_rgb_normal_height": (
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
        "normal_x",
        "normal_y",
        "normal_z",
        "height",
    ),
}


def read_ascii_ply(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as fh:
        header: list[str] = []
        props: list[str] = []
        vertex_count = 0
        for line in fh:
            line = line.strip()
            header.append(line)
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[1])
            elif line.startswith("property "):
                props.append(line.rsplit(" ", 1)[1])
            elif line == "end_header":
                break
        rows = np.loadtxt(fh, dtype=np.float32, max_rows=vertex_count)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return props, rows


def add_derived(props: list[str], rows: np.ndarray) -> dict[str, np.ndarray]:
    data = {name: rows[:, i] for i, name in enumerate(props)}
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    z = xyz[:, 2]
    data["height"] = z - float(np.percentile(z, 2))
    return data


def sampled_indices(n: int, max_points: int) -> np.ndarray:
    if n > max_points:
        return np.linspace(0, n - 1, max_points, dtype=np.int64)
    return np.arange(n)


def local_normals(xyz: np.ndarray, sample: np.ndarray, k: int = 16) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k, len(xyz))).fit(xyz)
    neighbor_idx = nn.kneighbors(xyz[sample], return_distance=False)
    pts = xyz[neighbor_idx]
    centered = pts - pts.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(1, pts.shape[1] - 1)
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0].astype(np.float32)
    normals *= np.sign(normals[:, 2:3] + 1e-6)
    return normals


def feature_matrix(data: dict[str, np.ndarray], names: tuple[str, ...], max_points: int) -> np.ndarray:
    n = len(next(iter(data.values())))
    idx = sampled_indices(n, max_points)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    normals = None
    if any(name.startswith("normal_") for name in names):
        normals = local_normals(xyz, idx)
    cols = []
    for name in names:
        if name == "normal_x":
            cols.append(normals[:, 0])
        elif name == "normal_y":
            cols.append(normals[:, 1])
        elif name == "normal_z":
            cols.append(normals[:, 2])
        else:
            cols.append(data[name][idx])
    x = np.column_stack(cols).astype(np.float32)
    for c in ("red", "green", "blue"):
        if c in names:
            x[:, names.index(c)] /= 255.0
    return x


def probe_crop(path: Path, *, max_points: int, clusters: int) -> dict[str, Any]:
    props, rows = read_ascii_ply(path)
    data = add_derived(props, rows)
    xyz = np.column_stack([data["x"], data["y"], data["z"]])
    rgb = np.column_stack([data["red"], data["green"], data["blue"]]) if "red" in data else np.empty((rows.shape[0], 0))
    out: dict[str, Any] = {
        "point_count": int(rows.shape[0]),
        "bbox_min": xyz.min(axis=0).round(4).tolist(),
        "bbox_max": xyz.max(axis=0).round(4).tolist(),
        "extent": (xyz.max(axis=0) - xyz.min(axis=0)).round(4).tolist(),
        "rgb_mean": rgb.mean(axis=0).round(2).tolist() if rgb.size else [],
        "feature_sets": {},
    }
    for feature_id, names in FEATURE_SETS.items():
        x = feature_matrix(data, names, max_points)
        k = min(clusters, max(1, len(x)))
        scaled = StandardScaler().fit_transform(x)
        labels = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10, batch_size=8192).fit_predict(scaled)
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        probs = counts / max(1.0, counts.sum())
        entropy = float(-(probs * np.log2(np.clip(probs, 1e-9, 1.0))).sum())
        out["feature_sets"][feature_id] = {
            "sampled_points": int(len(x)),
            "clusters": int(k),
            "largest_cluster_ratio": float(probs.max()),
            "entropy": entropy,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--clusters", type=int, default=8)
    args = parser.parse_args()

    report_path = args.crop_dir / "crop_export_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    crops: list[dict[str, Any]] = []
    for crop in report.get("crops", []):
        ply = args.crop_dir / Path(str(crop["output_ply"])).name
        item = {
            "id": crop["id"],
            "geometry_type": crop.get("geometry_type"),
            "sha256": crop.get("sha256"),
            "probe": probe_crop(ply, max_points=args.max_points, clusters=args.clusters),
        }
        crops.append(item)
    output = {
        "schema": "pointcloud-supervised-smoke-feature-probe/v1",
        "crop_dir": str(args.crop_dir),
        "crop_count": len(crops),
        "max_points_per_crop": args.max_points,
        "clusters": args.clusters,
        "crops": crops,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(args.output), "crop_count": len(crops)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
