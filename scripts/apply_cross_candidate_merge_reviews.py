#!/usr/bin/env python3
"""Apply reviewed cross-candidate merge decisions to long objects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {v: v for v in values}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        self.parent[max(ra, rb)] = min(ra, rb)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_reviews(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return load_jsonl(path)


def review_decision(row: dict) -> tuple[str, float]:
    vlm = row.get("vlm", {})
    if isinstance(vlm, dict):
        decision = vlm.get("decision", row.get("decision", ""))
        confidence = vlm.get("confidence", row.get("confidence", 0))
    else:
        decision = row.get("decision", "")
        confidence = row.get("confidence", 0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return str(decision).strip().lower(), confidence


def object_pair(row: dict) -> tuple[str, str]:
    if row.get("object_a") and row.get("object_b"):
        return str(row["object_a"]), str(row["object_b"])
    proposal = row.get("proposal", {})
    return str(proposal.get("object_a", "")), str(proposal.get("object_b", ""))


def merge_votes(objects: list[dict], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for obj in objects:
        for vote_key, value in obj.get(key, {}).items():
            merged[str(vote_key)] = merged.get(str(vote_key), 0) + int(value)
    return dict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True))


def weighted_mean(objects: list[dict], key: str, dims: int) -> list[float]:
    total = sum(max(0, int(obj.get("point_count", 0))) for obj in objects)
    if total <= 0:
        return [0.0] * dims
    accum = [0.0] * dims
    for obj in objects:
        weight = max(0, int(obj.get("point_count", 0)))
        values = obj.get(key, [0.0] * dims)
        for idx in range(dims):
            accum[idx] += float(values[idx]) * weight
    return [v / total for v in accum]


def aggregate_group(group_id: str, objects: list[dict], prefix: str) -> dict[str, Any]:
    objects = sorted(objects, key=lambda row: row.get("long_object_id", ""))
    point_count = sum(int(obj.get("point_count", 0)) for obj in objects)
    tracklet_ids = []
    for obj in objects:
        tracklet_ids.extend(obj.get("tracklet_ids", []))
    tracklet_ids = list(dict.fromkeys(tracklet_ids))
    bbox_mins = [obj.get("bbox_3d", {}).get("min", [0, 0, 0]) for obj in objects]
    bbox_maxs = [obj.get("bbox_3d", {}).get("max", [0, 0, 0]) for obj in objects]
    label_votes = merge_votes(objects, "label_votes")
    if not label_votes:
        label_votes = {}
        for obj in objects:
            label = str(obj.get("label", "unknown"))
            label_votes[label] = label_votes.get(label, 0) + int(obj.get("point_count", 0))
    dominant_label = max(label_votes.items(), key=lambda kv: kv[1])[0] if label_votes else "unknown"
    accepted_votes = merge_votes(objects, "accepted_candidate_votes")
    source_votes = merge_votes(objects, "source_cluster_votes")
    return {
        "long_object_id": f"{prefix}_{group_id}",
        "source_long_object_ids": [obj.get("long_object_id") for obj in objects],
        "source_object_count": len(objects),
        "label": dominant_label,
        "point_count": point_count,
        "tracklet_ids": tracklet_ids,
        "tracklet_count": len(tracklet_ids),
        "target_count": sum(int(obj.get("target_count", 0)) for obj in objects),
        "frame_min": min(int(obj.get("frame_min", 0)) for obj in objects),
        "frame_max": max(int(obj.get("frame_max", 0)) for obj in objects),
        "frame_count": sum(int(obj.get("frame_count", 0)) for obj in objects),
        "bbox_3d": {
            "min": [min(float(v[idx]) for v in bbox_mins) for idx in range(3)],
            "max": [max(float(v[idx]) for v in bbox_maxs) for idx in range(3)],
        },
        "centroid": weighted_mean(objects, "centroid", 3),
        "mean_color": weighted_mean(objects, "mean_color", 3),
        "label_votes": label_votes,
        "accepted_candidate_votes": accepted_votes,
        "dominant_accepted_candidate": max(accepted_votes.items(), key=lambda kv: kv[1])[0] if accepted_votes else "",
        "source_cluster_votes": source_votes,
        "dominant_source_cluster": max(source_votes.items(), key=lambda kv: kv[1])[0] if source_votes else "",
        "status": "review_merged_object" if len(objects) > 1 else objects[0].get("status", "unchanged_object"),
    }


def apply_reviews(objects: list[dict], reviews: list[dict], min_confidence: float) -> tuple[list[dict], list[dict]]:
    object_ids = [str(obj["long_object_id"]) for obj in objects]
    known = set(object_ids)
    uf = UnionFind(object_ids)
    decisions = []
    for row in reviews:
        a, b = object_pair(row)
        decision, confidence = review_decision(row)
        accepted = decision == "merge" and confidence >= min_confidence and a in known and b in known
        if accepted:
            uf.union(a, b)
        decisions.append(
            {
                "review_id": row.get("review_id", ""),
                "object_a": a,
                "object_b": b,
                "decision": decision,
                "confidence": confidence,
                "accepted": accepted,
            }
        )
    groups: dict[str, list[dict]] = {}
    for obj in objects:
        root = uf.find(str(obj["long_object_id"]))
        groups.setdefault(root, []).append(obj)
    merged = []
    for idx, (_, group) in enumerate(sorted(groups.items()), start=1):
        merged.append(aggregate_group(f"{idx:04d}", group, "review_obj"))
    return merged, decisions


def write_outputs(merged: list[dict], decisions: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    objects_path = output_dir / "review_merged_long_objects.jsonl"
    with objects_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    decisions_path = output_dir / "review_merge_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "objects_path": str(objects_path),
        "decisions_path": str(decisions_path),
        "object_count": len(merged),
        "accepted_merge_count": sum(1 for row in decisions if row["accepted"]),
        "review_count": len(decisions),
        "merged_group_count": sum(1 for row in merged if row["source_object_count"] > 1),
    }
    (output_dir / "review_merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    args = parser.parse_args()

    objects = load_jsonl(args.objects)
    reviews = load_reviews(args.reviews)
    merged, decisions = apply_reviews(objects, reviews, args.min_confidence)
    write_outputs(merged, decisions, args.output_dir)
    print(json.dumps({"objects": len(merged), "accepted_merges": sum(d["accepted"] for d in decisions)}, indent=2))


if __name__ == "__main__":
    main()
