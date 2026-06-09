#!/usr/bin/env python3
"""Sweep fine residual clustering parameters without writing point clouds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_residual_absorbability import SEMANTIC_IDS, SEMANTIC_NAMES
from cluster_fine_residual_objects import connected_components, pca_summary, read_ascii_ply


def parse_config(text: str) -> dict:
    values = {}
    for part in text.split(","):
        key, raw = part.split("=", 1)
        values[key.strip()] = float(raw)
    return {
        "name": text,
        "voxel_size": float(values["voxel"]),
        "min_cluster_points": int(values["min"]),
    }


def load_selected_points(path: Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    wanted_ids = {SEMANTIC_IDS[label] for label in labels}
    props, _, data = read_ascii_ply(path)
    idx = {name: i for i, name in enumerate(props)}
    assignment_status = data[:, idx["assignment_status"]].astype(np.int32)
    original_semantic = data[:, idx["original_semantic"]].astype(np.int32)
    selected = (assignment_status == 0) & np.isin(original_semantic, np.array(sorted(wanted_ids), dtype=np.int32))
    points = data[selected][:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    semantics = original_semantic[selected].astype(np.int32)
    return points, semantics


def run_config(points: np.ndarray, semantics: np.ndarray, config: dict, top_n: int) -> dict:
    by_label = Counter()
    clustered_by_label = Counter()
    small_by_label = Counter()
    rows = []
    for semantic_id in sorted(set(int(x) for x in semantics)):
        label = SEMANTIC_NAMES.get(semantic_id, "unknown")
        local_idx = np.where(semantics == semantic_id)[0]
        by_label[label] = int(len(local_idx))
        comps, small_points = connected_components(points[local_idx], config["voxel_size"], config["min_cluster_points"])
        small_by_label[label] = int(small_points)
        for comp in comps:
            pts = points[local_idx[comp]]
            clustered_by_label[label] += int(len(comp))
            summary = pca_summary(pts)
            rows.append(
                {
                    "label": label,
                    "points": int(len(comp)),
                    "linearity": summary["linearity"],
                    "planarity": summary["planarity"],
                    "bbox_3d": summary["bbox_3d"],
                }
            )
    rows.sort(key=lambda r: r["points"], reverse=True)
    clustered_points = sum(int(r["points"]) for r in rows)
    return {
        **config,
        "selected_points": int(len(points)),
        "cluster_count": int(len(rows)),
        "clustered_points": int(clustered_points),
        "clustered_ratio": float(clustered_points / max(len(points), 1)),
        "small_cluster_points": int(sum(small_by_label.values())),
        "by_label": dict(by_label),
        "clustered_by_label": dict(clustered_by_label),
        "small_by_label": dict(small_by_label),
        "top_clusters": rows[:top_n],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-assignment-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["equipment", "railing"])
    parser.add_argument("--config", action="append", required=True, help="voxel=<m>,min=<points>")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    points, semantics = load_selected_points(args.residual_assignment_ply, args.labels)
    configs = [parse_config(text) for text in args.config]
    report = {
        "residual_assignment_ply": str(args.residual_assignment_ply),
        "labels": args.labels,
        "configs": [run_config(points, semantics, cfg, args.top_n) for cfg in configs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        [
            {
                "name": row["name"],
                "cluster_count": row["cluster_count"],
                "clustered_ratio": row["clustered_ratio"],
                "largest": row["top_clusters"][0] if row["top_clusters"] else None,
                "small_cluster_points": row["small_cluster_points"],
            }
            for row in report["configs"]
        ],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
