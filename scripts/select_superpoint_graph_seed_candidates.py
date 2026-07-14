#!/usr/bin/env python3
"""Select reviewable Superpoints that cover new contact-graph neighborhoods."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .propagate_superpoint_structural_anchors import edge_weight
except ImportError:  # Direct script execution keeps scripts/ on sys.path.
    from propagate_superpoint_structural_anchors import edge_weight


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_supported_ids(rows: list[dict[str, Any]]) -> set[int]:
    return {int(row["object_id"]) for row in rows if row.get("top_source_frames")}


def graph_candidates(
    objects: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    supported: set[int],
    reviewed: set[int],
    covered: set[int],
    min_points: int,
    per_group: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    object_by_id = {int(row["object_id"]): row for row in objects}
    graph: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for edge in edges:
        weight = edge_weight(edge, min_faces=10, contact_faces_norm=100, color_sigma=40.0)
        if weight:
            a, b = int(edge["object_a"]), int(edge["object_b"])
            graph[a].append((b, weight))
            graph[b].append((a, weight))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for object_id, row in object_by_id.items():
        if object_id not in supported or object_id in reviewed or object_id not in graph:
            continue
        if int(row.get("count") or 0) < min_points:
            continue
        score = 0.0
        one_hop = 0
        two_hop = 0
        for neighbor, weight in graph[object_id]:
            if neighbor not in covered:
                score += weight * math.log1p(int(object_by_id.get(neighbor, {}).get("count") or 0))
                one_hop += 1
            for second, weight_second in graph.get(neighbor, []):
                if second != object_id and second not in covered:
                    score += 0.5 * weight * weight_second * math.log1p(int(object_by_id.get(second, {}).get("count") or 0))
                    two_hop += 1
        if score <= 0.0:
            continue
        enriched = dict(row)
        enriched.update({
            "seed_candidate_score": round(score, 6),
            "new_one_hop_neighbors": one_hop,
            "new_two_hop_paths": two_hop,
            "selection_policy": "source_supported_graph_coverage",
        })
        group = (str(row.get("geometry_type") or "unknown"), str(row.get("structural_region_dominant") or "unknown"))
        groups[group].append(enriched)

    selected = []
    for group, rows in groups.items():
        rows.sort(key=lambda row: (-float(row["seed_candidate_score"]), -int(row.get("count") or 0), int(row["object_id"])))
        for row in rows[:per_group]:
            row["selection_group"] = {"geometry_type": group[0], "structural_region": group[1]}
            selected.append(row)
    selected.sort(key=lambda row: (-float(row["seed_candidate_score"]), -int(row.get("count") or 0), int(row["object_id"])))
    return selected[:max_candidates]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--contact-edges", type=Path, required=True)
    parser.add_argument("--source-support", type=Path, required=True)
    parser.add_argument("--reviewed-jsonl", type=Path, required=True)
    parser.add_argument("--covered-posteriors", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=500)
    parser.add_argument("--per-group", type=int, default=25)
    parser.add_argument("--max-candidates", type=int, default=300)
    args = parser.parse_args()

    objects = read_jsonl(args.objects_jsonl)
    supported = source_supported_ids(read_jsonl(args.source_support))
    reviewed = {int(row["object_id"]) for row in read_jsonl(args.reviewed_jsonl)}
    covered = {int(row["object_id"]) for row in read_jsonl(args.covered_posteriors)}
    rows = graph_candidates(
        objects, read_jsonl(args.contact_edges), supported, reviewed, covered,
        args.min_points, args.per_group, args.max_candidates,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    group_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        group = row["selection_group"]
        group_counts[f"{group['geometry_type']}|{group['structural_region']}"] += 1
    report = {
        "selected": len(rows),
        "source_supported": len(supported),
        "already_reviewed": len(reviewed),
        "already_covered": len(covered),
        "selection_groups": dict(sorted(group_counts.items())),
        "min_points": args.min_points,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
