#!/usr/bin/env python3
"""Pool point-level descriptors into patch-level feature descriptors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_patch_graph_energy import read_labels


def load_point_features(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    features = data["features"].astype(np.float64, copy=False)
    if features.ndim != 2:
        raise ValueError("point feature NPZ must contain features[N,D]")
    point_indices = data["point_indices"].astype(np.int64, copy=False) if "point_indices" in data else None
    if point_indices is not None and len(point_indices) != len(features):
        raise ValueError("point_indices length must match features")
    return features, point_indices


def normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1)
    out = np.zeros_like(value, dtype=np.float32)
    ok = norm > 1e-9
    out[ok] = value[ok] / norm[ok, None]
    return out


def pool(labels: np.ndarray, features: np.ndarray, point_indices: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if point_indices is not None:
        labels = labels[point_indices]
    if len(labels) != len(features):
        raise ValueError(f"label count {len(labels)} does not match feature count {len(features)}")
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order].astype(np.int64, copy=False)
    sorted_features = features[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
    patch_ids = sorted_labels[starts]
    sums = np.add.reduceat(sorted_features, starts, axis=0)
    counts = np.diff(np.r_[starts, len(sorted_labels)]).astype(np.int64)
    means = sums / counts[:, None]
    return patch_ids, normalize_rows(means), counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--point-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = read_labels(args.labels)
    features, point_indices = load_point_features(args.point_features)
    patch_ids, pooled, counts = pool(labels, features, point_indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, patch_ids=patch_ids, features=pooled, counts=counts)
    report = {
        "schema": "patch-feature-pool/v1",
        "output": str(args.output),
        "input_point_count": int(len(features)),
        "patch_count": int(len(patch_ids)),
        "feature_dim": int(pooled.shape[1]),
        "used_point_indices": bool(point_indices is not None),
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
