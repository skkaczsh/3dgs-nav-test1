#!/usr/bin/env python3
"""Propagate structural anchors over short, color-compatible contact paths."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .audit_superpoint_structural_conflicts import conflict_reason
except ImportError:  # Direct script execution keeps scripts/ on sys.path.
    from audit_superpoint_structural_conflicts import conflict_reason


STRUCTURAL_LABELS = {"floor", "wall", "grass", "roof", "ceiling", "stair"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def edge_weight(row: dict[str, Any], min_faces: int, contact_faces_norm: int, color_sigma: float) -> float:
    faces = int(row["shared_voxel_faces"])
    if faces < min_faces:
        return 0.0
    contact = min(1.0, faces / float(max(contact_faces_norm, 1)))
    color = math.exp(-0.5 * (float(row["contact_rgb_distance"]) / max(color_sigma, 1e-6)) ** 2)
    return contact * color


def propagate(
    edges: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    min_faces: int,
    contact_faces_norm: int,
    color_sigma: float,
    max_hops: int,
    min_confidence: float,
    min_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    graph: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in edges:
        weight = edge_weight(row, min_faces, contact_faces_norm, color_sigma)
        if weight:
            a, b = int(row["object_a"]), int(row["object_b"])
            graph[a].append((b, weight))
            graph[b].append((a, weight))

    scores: dict[int, dict[str, float]] = defaultdict(dict)
    provenance: dict[tuple[int, str], tuple[int, int]] = {}
    geometry_by_id = {int(row["object_id"]): str(row.get("geometry_type") or "unknown") for row in anchors}
    queue: list[tuple[float, int, str, int, int]] = []
    seed_count = 0
    for row in anchors:
        label = str(row.get("anchor_label") or "unknown")
        if not bool(row.get("propagation_eligible")) or label not in STRUCTURAL_LABELS:
            continue
        object_id = int(row["object_id"])
        scores[object_id][label] = 1.0
        provenance[(object_id, label)] = (0, object_id)
        heapq.heappush(queue, (-1.0, object_id, label, 0, object_id))
        seed_count += 1

    while queue:
        neg_score, source, label, hops, seed = heapq.heappop(queue)
        score = -neg_score
        if score < scores[source].get(label, 0.0) or hops >= max_hops:
            continue
        for target, weight in graph.get(source, []):
            if conflict_reason(geometry_by_id.get(target, "unknown"), label):
                continue
            candidate = score * weight
            if candidate <= scores[target].get(label, 0.0):
                continue
            scores[target][label] = candidate
            provenance[(target, label)] = (hops + 1, seed)
            heapq.heappush(queue, (-candidate, target, label, hops + 1, seed))

    rows = []
    promoted = 0
    for object_id, posterior in sorted(scores.items()):
        ranked = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
        label, confidence = ranked[0]
        margin = confidence - (ranked[1][1] if len(ranked) > 1 else 0.0)
        eligible = confidence >= min_confidence and margin >= min_margin
        hops, seed = provenance[(object_id, label)]
        promoted += int(eligible)
        rows.append({
            "object_id": object_id,
            "structural_posterior": {key: round(value, 6) for key, value in ranked},
            "structural_candidate_label": label,
            "structural_confidence": round(confidence, 6),
            "structural_margin": round(margin, 6),
            "structural_source_anchor": seed,
            "structural_hops": hops,
            "geometry_type": geometry_by_id.get(object_id, "unknown"),
            "propagation_eligible": eligible,
            "propagation_status": "promoted" if eligible else "ambiguous_or_weak",
        })
    return rows, {"seed_count": seed_count, "nodes_with_posterior": len(rows), "promoted_nodes": promoted, "retained_edges": sum(len(neighbors) for neighbors in graph.values()) // 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-edges", type=Path, required=True)
    parser.add_argument("--anchor-posteriors", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-faces", type=int, default=10)
    parser.add_argument("--contact-faces-norm", type=int, default=100)
    parser.add_argument("--color-sigma", type=float, default=40.0)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--min-margin", type=float, default=0.15)
    args = parser.parse_args()

    rows, report = propagate(
        read_jsonl(args.contact_edges), read_jsonl(args.anchor_posteriors), args.min_faces,
        args.contact_faces_norm, args.color_sigma, args.max_hops, args.min_confidence, args.min_margin,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report.update(vars(args))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
