#!/usr/bin/env python3
"""Build a combined accepted fine-object QA PLY.

Accepted fine objects currently come from two non-overlapping sources:

1. hygiene status PLY points with cluster_status == fine_object_candidate
2. manual equipment subclusters whose review action is fine_candidate

The output is a QA artifact for visual inspection and later object fusion.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


FINE_OBJECT_STATUS = 1
SOURCE_HYGIENE_CLUSTER = 1
SOURCE_MANUAL_EQUIPMENT_SUBCLUSTER = 2


def read_ascii_ply(path: Path) -> tuple[list[str], int, np.ndarray]:
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
        return props, header_lines, np.empty((0, len(props)), dtype=np.float32)
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return props, header_lines, data


def pca_summary(points: np.ndarray) -> dict:
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    denom = float(eigvals[0]) if eigvals[0] > 1e-9 else 1.0
    return {
        "centroid": [float(x) for x in centroid],
        "bbox_3d": {
            "min": [float(x) for x in points.min(axis=0)],
            "max": [float(x) for x in points.max(axis=0)],
        },
        "linearity": float((eigvals[0] - eigvals[1]) / denom),
        "planarity": float((eigvals[1] - eigvals[2]) / denom),
        "pca_eigenvalues": [float(x) for x in eigvals],
    }


def candidate_color(candidate_id: int, source_type: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(candidate_id * 97 + source_type * 7919)
    lo = np.array([70, 90, 80], dtype=np.int32)
    hi = np.array([245, 235, 245], dtype=np.int32)
    color = rng.integers(lo, hi + 1)
    return int(color[0]), int(color[1]), int(color[2])


def load_fine_subclusters(path: Path) -> set[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["subcluster_id"])
        for row in raw.get("subclusters", [])
        if row.get("recommended_action") == "fine_candidate"
    }


def collect_hygiene_candidates(path: Path) -> tuple[list[dict], list[dict]]:
    props, _, data = read_ascii_ply(path)
    idx = {name: i for i, name in enumerate(props)}
    required = {"x", "y", "z", "semantic", "cluster", "visual_red", "visual_green", "visual_blue", "cluster_status"}
    if not required.issubset(idx):
        raise ValueError(f"missing required fields in {path}: {props}")
    status = data[:, idx["cluster_status"]].astype(np.int32)
    selected = data[status == FINE_OBJECT_STATUS]
    rows = []
    report_rows = []
    for cluster_id in sorted(set(int(x) for x in selected[:, idx["cluster"]].astype(np.int32).tolist())):
        local = selected[selected[:, idx["cluster"]].astype(np.int32) == cluster_id]
        points = local[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
        visual = local[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]].astype(np.uint8)
        semantic_id = int(round(float(local[0, idx["semantic"]])))
        candidate_id = 100000 + cluster_id
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_type": SOURCE_HYGIENE_CLUSTER,
                "source_cluster": cluster_id,
                "subcluster": -1,
                "semantic_id": semantic_id,
                "points": points,
                "visual_colors": visual,
            }
        )
        report_rows.append(
            {
                "candidate_id": candidate_id,
                "source_type": "hygiene_cluster",
                "source_cluster": cluster_id,
                "subcluster": -1,
                "semantic_id": semantic_id,
                "points": int(len(points)),
                "mean_visual_color": [float(x) for x in visual.astype(np.float32).mean(axis=0)],
                **pca_summary(points),
            }
        )
    return rows, report_rows


def collect_manual_candidates(path: Path, fine_subclusters: set[int]) -> tuple[list[dict], list[dict]]:
    props, _, data = read_ascii_ply(path)
    idx = {name: i for i, name in enumerate(props)}
    required = {"x", "y", "z", "semantic", "source_cluster", "subcluster", "visual_red", "visual_green", "visual_blue"}
    if not required.issubset(idx):
        raise ValueError(f"missing required fields in {path}: {props}")
    subclusters = data[:, idx["subcluster"]].astype(np.int32)
    selected = data[np.isin(subclusters, np.array(sorted(fine_subclusters), dtype=np.int32))]
    rows = []
    report_rows = []
    for subcluster in sorted(fine_subclusters):
        local = selected[selected[:, idx["subcluster"]].astype(np.int32) == subcluster]
        if len(local) == 0:
            continue
        points = local[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
        visual = local[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]].astype(np.uint8)
        semantic_id = int(round(float(local[0, idx["semantic"]])))
        source_cluster = int(round(float(local[0, idx["source_cluster"]])))
        candidate_id = 200000 + subcluster
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_type": SOURCE_MANUAL_EQUIPMENT_SUBCLUSTER,
                "source_cluster": source_cluster,
                "subcluster": subcluster,
                "semantic_id": semantic_id,
                "points": points,
                "visual_colors": visual,
            }
        )
        report_rows.append(
            {
                "candidate_id": candidate_id,
                "source_type": "manual_equipment_subcluster",
                "source_cluster": source_cluster,
                "subcluster": subcluster,
                "semantic_id": semantic_id,
                "points": int(len(points)),
                "mean_visual_color": [float(x) for x in visual.astype(np.float32).mean(axis=0)],
                **pca_summary(points),
            }
        )
    return rows, report_rows


def write_ply(path: Path, rows: list[dict]) -> None:
    total = sum(len(row["points"]) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("property int accepted_candidate\n")
        f.write("property uchar source_type\n")
        f.write("property int source_cluster\n")
        f.write("property int subcluster\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("end_header\n")
        for row in rows:
            color = candidate_color(int(row["candidate_id"]), int(row["source_type"]))
            for point, visual in zip(row["points"], row["visual_colors"]):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]} {int(row['semantic_id'])} "
                    f"{int(row['candidate_id'])} {int(row['source_type'])} "
                    f"{int(row['source_cluster'])} {int(row['subcluster'])} "
                    f"{int(visual[0])} {int(visual[1])} {int(visual[2])}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygiene-status-ply", type=Path, required=True)
    parser.add_argument("--manual-subclusters-ply", type=Path, required=True)
    parser.add_argument("--manual-subcluster-review-json", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    hygiene_rows, hygiene_report = collect_hygiene_candidates(args.hygiene_status_ply)
    fine_subclusters = load_fine_subclusters(args.manual_subcluster_review_json)
    manual_rows, manual_report = collect_manual_candidates(args.manual_subclusters_ply, fine_subclusters)
    rows = hygiene_rows + manual_rows
    report_rows = hygiene_report + manual_report
    report_rows.sort(key=lambda row: row["points"], reverse=True)
    write_ply(args.output_ply, rows)
    counts = Counter(row["source_type"] for row in report_rows)
    point_counts = Counter()
    for row in report_rows:
        point_counts[row["source_type"]] += int(row["points"])
    report = {
        "hygiene_status_ply": str(args.hygiene_status_ply),
        "manual_subclusters_ply": str(args.manual_subclusters_ply),
        "manual_subcluster_review_json": str(args.manual_subcluster_review_json),
        "output_ply": str(args.output_ply),
        "candidate_count": int(len(report_rows)),
        "accepted_points": int(sum(row["points"] for row in report_rows)),
        "candidate_counts": dict(counts),
        "point_counts": dict(point_counts),
        "top_candidates": report_rows[: args.top_n],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": report["candidate_count"],
                "accepted_points": report["accepted_points"],
                "candidate_counts": report["candidate_counts"],
                "point_counts": report["point_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
