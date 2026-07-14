#!/usr/bin/env python3
"""Group only compatible structural Superpoints into traceable spatial regions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

try:
    from scripts.propagate_superpoint_structural_anchors import edge_weight
except ModuleNotFoundError:  # Supports direct `python scripts/...` execution.
    from propagate_superpoint_structural_anchors import edge_weight


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def structural_regions(
    posteriors: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    min_faces: int,
    contact_faces_norm: int,
    color_sigma: float,
) -> list[tuple[str, list[int]]]:
    labels = {
        int(row["object_id"]): str(row["structural_candidate_label"])
        for row in posteriors if row.get("propagation_eligible")
    }
    graph: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        if not edge_weight(edge, min_faces, contact_faces_norm, color_sigma):
            continue
        a, b = int(edge["object_a"]), int(edge["object_b"])
        if labels.get(a) and labels.get(a) == labels.get(b):
            graph[a].append(b)
            graph[b].append(a)

    unseen = set(labels)
    regions = []
    while unseen:
        start = unseen.pop()
        members = [start]
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, []):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    members.append(neighbor)
                    queue.append(neighbor)
        regions.append((labels[start], sorted(members)))
    return sorted(regions, key=lambda item: (item[0], item[1][0]))


def region_row(region_id: int, label: str, members: list[int], objects: dict[int, dict[str, Any]], posteriors: dict[int, dict[str, Any]]) -> dict[str, Any]:
    bboxes = [(objects[node].get("bbox_min"), objects[node].get("bbox_max")) for node in members if node in objects]
    bounds_min = [min(float(b[0][axis]) for b in bboxes) for axis in range(3)] if bboxes else []
    bounds_max = [max(float(b[1][axis]) for b in bboxes) for axis in range(3)] if bboxes else []
    anchors = Counter(posteriors[node].get("structural_source_anchor") for node in members if posteriors[node].get("structural_source_anchor") is not None)
    return {
        "region_id": f"region:{label}:{region_id}",
        "region_label": label,
        "superpoint_ids": members,
        "superpoint_count": len(members),
        "bbox_min": bounds_min,
        "bbox_max": bounds_max,
        "source_anchor_ids": sorted(int(anchor) for anchor in anchors),
        "max_hops": max(int(posteriors[node].get("structural_hops") or 0) for node in members),
        "ownership_policy": "members remain immutable official superpoints",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-posteriors", type=Path, required=True)
    parser.add_argument("--contact-edges", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-regions", type=Path, required=True)
    parser.add_argument("--output-assignments", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-faces", type=int, default=10)
    parser.add_argument("--contact-faces-norm", type=int, default=100)
    parser.add_argument("--color-sigma", type=float, default=40.0)
    args = parser.parse_args()

    posterior_rows = read_jsonl(args.structural_posteriors)
    posterior_by_id = {int(row["object_id"]): row for row in posterior_rows}
    object_by_id = {int(row["object_id"]): row for row in read_jsonl(args.objects_jsonl)}
    grouped = structural_regions(
        posterior_rows, read_jsonl(args.contact_edges), args.min_faces,
        args.contact_faces_norm, args.color_sigma,
    )
    regions = [region_row(index, label, members, object_by_id, posterior_by_id) for index, (label, members) in enumerate(grouped)]
    assignments = [
        {"superpoint_id": node, "region_id": region["region_id"], "region_label": region["region_label"]}
        for region in regions for node in region["superpoint_ids"]
    ]
    args.output_regions.parent.mkdir(parents=True, exist_ok=True)
    args.output_regions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in regions), encoding="utf-8")
    args.output_assignments.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in assignments), encoding="utf-8")
    report = {
        "regions": len(regions),
        "assigned_superpoints": len(assignments),
        "region_labels": dict(Counter(row["region_label"] for row in regions)),
        "largest_region_superpoints": max((row["superpoint_count"] for row in regions), default=0),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
