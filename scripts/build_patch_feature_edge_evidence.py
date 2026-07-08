#!/usr/bin/env python3
"""Build SPG external edge evidence from patch-level feature descriptors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_patch_graph_energy import build_edge_counts, read_labels, read_region_input


def normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1)
    out = np.zeros_like(value, dtype=np.float64)
    ok = norm > 1e-9
    out[ok] = value[ok] / norm[ok, None]
    return out


def load_patch_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    patch_ids = data["patch_ids"].astype(np.int64, copy=False)
    features = data["features"].astype(np.float64, copy=False)
    if features.ndim != 2 or len(patch_ids) != len(features):
        raise ValueError("feature NPZ must contain patch_ids[N] and features[N,D]")
    return patch_ids, normalize_rows(features)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--patch-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    edge_counts = build_edge_counts(labels, src, dst)
    patch_ids, features = load_patch_features(args.patch_features)
    by_id = {int(patch_id): features[i] for i, patch_id in enumerate(patch_ids.tolist())}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = 0
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["patch_a", "patch_b", "similarity", "contact_points"])
        writer.writeheader()
        for a, b in sorted(edge_counts):
            fa = by_id.get(int(a))
            fb = by_id.get(int(b))
            if fa is None or fb is None:
                missing += 1
                continue
            similarity = float(np.dot(fa, fb))
            row = {
                "patch_a": int(a),
                "patch_b": int(b),
                "similarity": max(0.0, min(1.0, (similarity + 1.0) * 0.5)),
                "contact_points": int(edge_counts[(a, b)]),
            }
            writer.writerow(row)
            rows.append(row)

    report = {
        "schema": "patch-feature-edge-evidence/v1",
        "output": str(args.output),
        "touch_edge_count": int(len(edge_counts)),
        "written_edge_count": int(len(rows)),
        "missing_feature_edge_count": int(missing),
        "patch_feature_count": int(len(patch_ids)),
    }
    (args.output.with_suffix(args.output.suffix + ".report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
