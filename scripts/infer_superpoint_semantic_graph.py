#!/usr/bin/env python3
"""Conservatively refine reviewed Superpoint semantic posteriors on contact edges.

This stage never changes voxel ownership. It only combines VLM soft unaries on
immutable Superpoints, and deliberately refuses to use unobserved nodes as
semantic sources or bridges.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SURFACE_LABELS = {"floor", "grass", "stair", "roof", "ceiling"}
VERTICAL_LABELS = {"wall", "building"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_alpha(alpha: Any) -> dict[str, float]:
    if not isinstance(alpha, dict):
        return {}
    values: dict[str, float] = {}
    for label, value in alpha.items():
        try:
            numeric = max(float(value), 0.0)
        except (TypeError, ValueError):
            continue
        if numeric:
            values[str(label)] = numeric
    total = sum(values.values())
    return {label: value / total for label, value in values.items()} if total else {}


def edge_affinity(row: dict[str, Any], min_faces: int, min_contact_ratio: float, color_sigma: float) -> float:
    faces = max(int(row.get("shared_voxel_faces") or 0), 0)
    ratio = max(float(row.get("contact_ratio_min") or 0.0), 0.0)
    if faces < min_faces or ratio < min_contact_ratio:
        return 0.0
    support = math.sqrt(min(1.0, faces / 100.0) * min(1.0, ratio / 0.25))
    rgb = float(row.get("contact_rgb_distance") or 0.0)
    color = math.exp(-0.5 * (rgb / max(color_sigma, 1e-6)) ** 2)
    return support * color


def geometry_allows(geometry_type: str, label: str) -> bool:
    """Only veto stable surface contradictions; retain object-like alternatives."""
    if geometry_type == "horizontal" and label in VERTICAL_LABELS:
        return False
    if geometry_type == "vertical" and label in SURFACE_LABELS:
        return False
    if geometry_type == "upper_surface" and label in {"floor", "grass", "stair"}:
        return False
    return True


def infer(
    unary_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    min_faces: int = 10,
    min_contact_ratio: float = 0.01,
    color_sigma: float = 35.0,
    pairwise_weight: float = 0.35,
    promotion_margin: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = {int(row["object_id"]): row for row in unary_rows}
    graph: dict[int, list[tuple[int, float]]] = defaultdict(list)
    kept_edges = 0
    for edge in edge_rows:
        a, b = int(edge["object_a"]), int(edge["object_b"])
        if a not in rows or b not in rows:
            continue
        affinity = edge_affinity(edge, min_faces, min_contact_ratio, color_sigma)
        if affinity <= 0:
            continue
        graph[a].append((b, affinity))
        graph[b].append((a, affinity))
        kept_edges += 1

    # Only reviewed nodes carry VLM class evidence. Unobserved nodes neither
    # seed nor relay messages; observed-but-unlabeled nodes may receive a local
    # proposal but never become promoted by this stage.
    base = {object_id: normalize_alpha(row.get("alpha")) for object_id, row in rows.items()}
    out: list[dict[str, Any]] = []
    proposals = 0
    for object_id, row in sorted(rows.items()):
        scores = dict(base[object_id])
        for neighbor, affinity in graph.get(object_id, []):
            neighbor_row = rows[neighbor]
            if neighbor_row.get("state") != "reviewed":
                continue
            for label, value in base[neighbor].items():
                if geometry_allows(str(row.get("geometry_type") or "unknown"), label):
                    scores[label] = scores.get(label, 0.0) + pairwise_weight * affinity * value
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        total = sum(value for _label, value in ranked)
        posterior = {label: round(value / total, 6) for label, value in ranked} if total else {}
        label = ranked[0][0] if ranked else "unknown"
        margin = (ranked[0][1] - ranked[1][1]) / total if len(ranked) > 1 and total else (ranked[0][1] / total if total else 0.0)
        reviewed = row.get("state") == "reviewed"
        promoted = reviewed and margin >= promotion_margin
        proposals += int(bool(posterior) and not reviewed)
        out.append({
            "object_id": object_id,
            "state": row.get("state", "unobserved"),
            "geometry_type": row.get("geometry_type", "unknown"),
            "semantic_posterior": posterior,
            "semantic_candidate_label": label,
            "semantic_margin": round(float(margin), 6),
            "semantic_status": "reviewed_posterior" if promoted else (
                "local_proposal_not_promoted" if posterior else "unobserved_or_unlabeled"
            ),
        })
    return out, {
        "nodes": len(rows),
        "kept_edges": kept_edges,
        "reviewed_nodes": sum(row.get("state") == "reviewed" for row in rows.values()),
        "local_unlabeled_proposals": proposals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft-unaries", type=Path, required=True)
    parser.add_argument("--contact-edges", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-faces", type=int, default=10)
    parser.add_argument("--min-contact-ratio", type=float, default=0.01)
    parser.add_argument("--color-sigma", type=float, default=35.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.35)
    parser.add_argument("--promotion-margin", type=float, default=0.20)
    args = parser.parse_args()
    rows, report = infer(
        read_jsonl(args.soft_unaries), read_jsonl(args.contact_edges), args.min_faces,
        args.min_contact_ratio, args.color_sigma, args.pairwise_weight, args.promotion_margin,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
