#!/usr/bin/env python3
"""Evaluate SAM2 edge evidence against agreement-only historical VLM labels.

Historical reviews use older image evidence and must never become production
unaries. They are useful as a frozen, deliberately conservative holdout: an
edge is eligible only when every supplied review independently assigned the
same controlled label to both endpoint objects.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def review_labels(rows: list[dict[str, Any]], min_confidence: float) -> dict[int, str]:
    labels: dict[int, str] = {}
    for row in rows:
        parsed = row.get("parsed") or {}
        label = str(parsed.get("controlled_label") or "unknown")
        confidence = float(parsed.get("confidence") or 0.0)
        if label != "unknown" and confidence >= min_confidence:
            labels[int(row["object_id"])] = label
    return labels


def consensus_labels(review_sets: list[dict[int, str]]) -> dict[int, str]:
    if not review_sets:
        return {}
    common = set.intersection(*(set(labels) for labels in review_sets))
    return {
        object_id: review_sets[0][object_id]
        for object_id in common
        if all(labels[object_id] == review_sets[0][object_id] for labels in review_sets[1:])
    }


def evidence_group(row: dict[str, Any], separation_threshold: float, same_threshold: float) -> str:
    if float(row.get("sam2_affinity") or 1.0) < separation_threshold:
        return "strong_separation"
    if float(row.get("same_mask_lcb") or 0.0) >= same_threshold:
        return "strong_same_mask"
    if float(row.get("sam2_affinity") or 1.0) == 1.0:
        return "neutral"
    return "weak_non_neutral"


def evaluate(
    edges: list[dict[str, Any]], labels: dict[int, str], separation_threshold: float, same_threshold: float,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for row in edges:
        group = evidence_group(row, separation_threshold, same_threshold)
        stats = groups.setdefault(group, {"edges": 0, "consensus_pairs": 0, "same_label_pairs": 0, "examples": []})
        stats["edges"] += 1
        object_a, object_b = int(row["object_a"]), int(row["object_b"])
        if object_a not in labels or object_b not in labels:
            continue
        stats["consensus_pairs"] += 1
        same = labels[object_a] == labels[object_b]
        stats["same_label_pairs"] += int(same)
        if len(stats["examples"]) < 20:
            stats["examples"].append({
                "object_a": object_a,
                "object_b": object_b,
                "label_a": labels[object_a],
                "label_b": labels[object_b],
                "same_label": same,
                "sam2_affinity": float(row.get("sam2_affinity") or 1.0),
                "view_count": int(row.get("view_count") or 0),
            })
    for stats in groups.values():
        pairs = int(stats["consensus_pairs"])
        stats["same_label_ratio"] = round(int(stats["same_label_pairs"]) / pairs, 6) if pairs else None
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-edges", type=Path, required=True)
    parser.add_argument("--review-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--separation-threshold", type=float, default=0.8)
    parser.add_argument("--same-mask-threshold", type=float, default=0.5)
    args = parser.parse_args()

    review_sets = [review_labels(read_jsonl(path), args.min_confidence) for path in args.review_jsonl]
    labels = consensus_labels(review_sets)
    edges = read_jsonl(args.sam2_edges)
    report = {
        "sam2_edges": str(args.sam2_edges),
        "review_jsonl": [str(path) for path in args.review_jsonl],
        "review_label_counts": [len(values) for values in review_sets],
        "consensus_label_count": len(labels),
        "min_confidence": args.min_confidence,
        "groups": evaluate(edges, labels, args.separation_threshold, args.same_mask_threshold),
        "warning": "Holdout only: historical labels must not be used as current production unaries.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"consensus_label_count": len(labels), "groups": report["groups"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
