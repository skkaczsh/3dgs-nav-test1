#!/usr/bin/env python3
"""Propose cross accepted-candidate object merge pairs for manual/VLM review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_objects(path: Path) -> list[dict]:
    files = sorted(path.glob("long_objects.jsonl")) if path.is_dir() else [path]
    rows: list[dict] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def bbox_distance(a: dict, b: dict) -> float:
    amin = np.array(a["bbox_3d"]["min"], dtype=np.float64)
    amax = np.array(a["bbox_3d"]["max"], dtype=np.float64)
    bmin = np.array(b["bbox_3d"]["min"], dtype=np.float64)
    bmax = np.array(b["bbox_3d"]["max"], dtype=np.float64)
    gap = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    return float(np.linalg.norm(gap))


def frame_gap(a: dict, b: dict) -> int:
    if int(b["frame_min"]) > int(a["frame_max"]):
        return int(b["frame_min"] - a["frame_max"])
    if int(a["frame_min"]) > int(b["frame_max"]):
        return int(a["frame_min"] - b["frame_max"])
    return 0


def overlap_ratio(a: dict, b: dict) -> float:
    amin = np.array(a["bbox_3d"]["min"], dtype=np.float64)
    amax = np.array(a["bbox_3d"]["max"], dtype=np.float64)
    bmin = np.array(b["bbox_3d"]["min"], dtype=np.float64)
    bmax = np.array(b["bbox_3d"]["max"], dtype=np.float64)
    inter_min = np.maximum(amin, bmin)
    inter_max = np.minimum(amax, bmax)
    inter = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter))
    avol = float(np.prod(np.maximum(0.0, amax - amin)))
    bvol = float(np.prod(np.maximum(0.0, bmax - bmin)))
    return inter_vol / max(min(avol, bvol), 1e-12)


def proposal_score(row: dict) -> float:
    return (
        row["centroid_distance"]
        + row["bbox_distance"] * 2.0
        + row["color_distance"] / 255.0
        + min(row["frame_gap"], 300) / 1000.0
        - row["bbox_overlap_ratio"]
    )


def propose(objects: list[dict], args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if a["label"] != b["label"]:
                continue
            if a.get("dominant_accepted_candidate") == b.get("dominant_accepted_candidate"):
                continue
            centroid_dist = float(np.linalg.norm(np.array(a["centroid"]) - np.array(b["centroid"])))
            bd = bbox_distance(a, b)
            color_dist = float(np.linalg.norm(np.array(a["mean_color"]) - np.array(b["mean_color"])))
            fg = frame_gap(a, b)
            ov = overlap_ratio(a, b)
            same_source = bool(a.get("dominant_source_cluster") and a.get("dominant_source_cluster") == b.get("dominant_source_cluster"))
            spatial_ok = centroid_dist <= args.centroid_distance or bd <= args.bbox_distance or ov >= args.min_bbox_overlap
            if not spatial_ok:
                continue
            if color_dist > args.color_distance:
                continue
            if fg > args.frame_gap and not same_source:
                continue
            row = {
                "object_a": a["long_object_id"],
                "object_b": b["long_object_id"],
                "label": a["label"],
                "candidate_a": a.get("dominant_accepted_candidate", ""),
                "candidate_b": b.get("dominant_accepted_candidate", ""),
                "candidate_ratio_a": float(a.get("dominant_accepted_candidate_ratio", 0.0)),
                "candidate_ratio_b": float(b.get("dominant_accepted_candidate_ratio", 0.0)),
                "source_a": a.get("dominant_source_cluster", ""),
                "source_b": b.get("dominant_source_cluster", ""),
                "same_source_cluster": bool(same_source),
                "point_count_a": int(a.get("point_count", 0)),
                "point_count_b": int(b.get("point_count", 0)),
                "tracklet_count_a": int(a.get("tracklet_count", 0)),
                "tracklet_count_b": int(b.get("tracklet_count", 0)),
                "frame_min_a": int(a.get("frame_min", 0)),
                "frame_max_a": int(a.get("frame_max", 0)),
                "frame_min_b": int(b.get("frame_min", 0)),
                "frame_max_b": int(b.get("frame_max", 0)),
                "frame_gap": int(fg),
                "centroid_distance": centroid_dist,
                "bbox_distance": bd,
                "bbox_overlap_ratio": ov,
                "color_distance": color_dist,
            }
            row["score"] = proposal_score(row)
            if same_source and row["score"] <= args.auto_review_score:
                row["review_priority"] = "high"
            elif row["score"] <= args.auto_review_score:
                row["review_priority"] = "medium"
            else:
                row["review_priority"] = "low"
            rows.append(row)
    rows.sort(key=lambda r: (r["score"], -min(r["point_count_a"], r["point_count_b"])))
    return rows[: args.max_proposals]


def write_outputs(output_dir: Path, proposals: list[dict], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "cross_candidate_merge_proposals.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in proposals:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = output_dir / "cross_candidate_merge_proposals.csv"
    if proposals:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(proposals[0].keys()))
            writer.writeheader()
            writer.writerows(proposals)
    report = {
        "objects_jsonl": str(args.objects),
        "output_jsonl": str(jsonl_path),
        "output_csv": str(csv_path),
        "proposal_count": int(len(proposals)),
        "priority_counts": {
            key: int(sum(1 for row in proposals if row["review_priority"] == key))
            for key in ("high", "medium", "low")
        },
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "top_proposals": proposals[:50],
    }
    (output_dir / "cross_candidate_merge_proposals_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["proposal_count", "priority_counts"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--centroid-distance", type=float, default=1.2)
    parser.add_argument("--bbox-distance", type=float, default=0.35)
    parser.add_argument("--min-bbox-overlap", type=float, default=0.05)
    parser.add_argument("--color-distance", type=float, default=80.0)
    parser.add_argument("--frame-gap", type=int, default=360)
    parser.add_argument("--auto-review-score", type=float, default=1.2)
    parser.add_argument("--max-proposals", type=int, default=200)
    args = parser.parse_args()

    objects = load_objects(args.objects)
    proposals = propose(objects, args)
    write_outputs(args.output_dir, proposals, args)


if __name__ == "__main__":
    main()
