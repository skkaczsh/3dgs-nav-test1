#!/usr/bin/env python3
"""Analyze whether Sonata PCA colors give useful region separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def read_ascii_ply(path: Path) -> tuple[list[str], np.ndarray]:
    props: list[str] = []
    vertex_count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("element vertex "):
                vertex_count = int(s.split()[-1])
            elif s.startswith("property "):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
        rows = np.loadtxt(fh, dtype=np.float32, max_rows=vertex_count)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return props, rows


def cluster_report(xyz: np.ndarray, feat: np.ndarray, clusters: int) -> dict:
    x = StandardScaler().fit_transform(feat.astype(np.float32))
    labels = MiniBatchKMeans(n_clusters=clusters, random_state=0, n_init=10, batch_size=8192).fit_predict(x)
    counts = np.bincount(labels, minlength=clusters).astype(np.float64)
    probs = counts / max(1.0, counts.sum())
    nn = NearestNeighbors(n_neighbors=min(8, len(xyz))).fit(xyz)
    neigh = nn.kneighbors(xyz, return_distance=False)[:, 1:]
    same = labels[neigh] == labels[:, None]
    local_agreement = float(same.mean()) if same.size else 0.0
    return {
        "clusters": int(clusters),
        "largest_cluster_ratio": float(probs.max()),
        "entropy": float(-(probs * np.log2(np.clip(probs, 1e-9, 1.0))).sum()),
        "local_neighbor_label_agreement": local_agreement,
        "cluster_counts": counts.astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonata-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=8)
    args = parser.parse_args()

    props, rows = read_ascii_ply(args.sonata_ply)
    col = {name: i for i, name in enumerate(props)}
    xyz = rows[:, [col["x"], col["y"], col["z"]]]
    rgb = rows[:, [col["red"], col["green"], col["blue"]]] / 255.0
    z = xyz[:, 2:3]
    xyz_height = np.column_stack([xyz, z - np.percentile(z, 2)])
    report = {
        "schema": "sonata-smoke-pca-analysis/v1",
        "input": str(args.sonata_ply),
        "point_count": int(len(rows)),
        "bbox_min": xyz.min(axis=0).round(4).tolist(),
        "bbox_max": xyz.max(axis=0).round(4).tolist(),
        "extent": (xyz.max(axis=0) - xyz.min(axis=0)).round(4).tolist(),
        "cluster_reports": {
            "sonata_pca_rgb": cluster_report(xyz, rgb, args.clusters),
            "xyz_height": cluster_report(xyz, xyz_height, args.clusters),
            "xyz_height_plus_sonata_pca_rgb": cluster_report(xyz, np.column_stack([xyz_height, rgb]), args.clusters),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(args.output), "point_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
