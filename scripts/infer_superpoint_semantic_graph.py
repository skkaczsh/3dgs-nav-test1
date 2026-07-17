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


def positive_alpha(alpha: Any) -> dict[str, float]:
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
    return values


def edge_affinity(row: dict[str, Any], min_faces: int, min_contact_ratio: float, color_sigma: float) -> float:
    faces = max(int(row.get("shared_voxel_faces") or 0), 0)
    ratio = max(float(row.get("contact_ratio_min") or 0.0), 0.0)
    if faces < min_faces or ratio < min_contact_ratio:
        return 0.0
    support = math.sqrt(min(1.0, faces / 100.0) * min(1.0, ratio / 0.25))
    rgb = float(row.get("contact_rgb_distance") or 0.0)
    color = math.exp(-0.5 * (rgb / max(color_sigma, 1e-6)) ** 2)
    return support * color


def photometric_affinity(row: dict[str, Any] | None) -> float:
    if not row:
        return 1.0
    try:
        value = row.get("photometric_affinity")
        return 1.0 if value is None else min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 1.0


def sam2_affinity(row: dict[str, Any] | None) -> float:
    if not row:
        return 1.0
    try:
        value = row.get("sam2_affinity")
        return 1.0 if value is None else min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 1.0


def attenuation_factor(evidence_affinity: float, weight: float) -> float:
    """Apply optional image evidence as a bounded attenuation, never a boost."""
    bounded_weight = min(max(float(weight), 0.0), 1.0)
    return (1.0 - bounded_weight) + bounded_weight * evidence_affinity


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
    photometric_rows: list[dict[str, Any]] | None = None,
    photometric_weight: float = 1.0,
    sam2_rows: list[dict[str, Any]] | None = None,
    sam2_weight: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = {int(row["object_id"]): row for row in unary_rows}
    photo_by_edge = {
        (min(int(row["object_a"]), int(row["object_b"])), max(int(row["object_a"]), int(row["object_b"]))): row
        for row in (photometric_rows or [])
    }
    sam2_by_edge = {
        (min(int(row["object_a"]), int(row["object_b"])), max(int(row["object_a"]), int(row["object_b"]))): row
        for row in (sam2_rows or [])
    }
    graph: dict[int, list[tuple[int, float]]] = defaultdict(list)
    kept_edges = 0
    sam2_observed_viable_edges = 0
    sam2_strong_separation_viable_edges = 0
    for edge in edge_rows:
        a, b = int(edge["object_a"]), int(edge["object_b"])
        if a not in rows or b not in rows:
            continue
        affinity = edge_affinity(edge, min_faces, min_contact_ratio, color_sigma)
        if affinity <= 0:
            continue
        key = (min(a, b), max(a, b))
        photo = photometric_affinity(photo_by_edge.get(key))
        affinity *= attenuation_factor(photo, photometric_weight)
        if affinity <= 0:
            continue
        sam2 = sam2_affinity(sam2_by_edge.get(key))
        if key in sam2_by_edge:
            sam2_observed_viable_edges += 1
            sam2_strong_separation_viable_edges += int(sam2 < 0.8)
        affinity *= attenuation_factor(sam2, sam2_weight)
        if affinity <= 0:
            continue
        graph[a].append((b, affinity))
        graph[b].append((a, affinity))
        kept_edges += 1

    # Only reviewed nodes carry VLM class evidence. Unobserved nodes neither
    # seed nor relay messages; observed-but-unlabeled nodes may receive a local
    # proposal but never become promoted by this stage.
    # Keep the unary magnitude through message passing.  Normalizing here
    # would make a weak one-view review exert the same force as a strongly
    # supported multi-view review.  Only the final posterior is normalized.
    raw_alpha = {object_id: positive_alpha(row.get("alpha")) for object_id, row in rows.items()}
    out: list[dict[str, Any]] = []
    proposals = 0
    for object_id, row in sorted(rows.items()):
        scores = dict(raw_alpha[object_id])
        for neighbor, affinity in graph.get(object_id, []):
            neighbor_row = rows[neighbor]
            if neighbor_row.get("state") != "reviewed":
                continue
            for label, value in raw_alpha[neighbor].items():
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
        "photometric_edges": len(photo_by_edge),
        "sam2_comask_edges": len(sam2_by_edge),
        "sam2_observed_viable_edges": sam2_observed_viable_edges,
        "sam2_strong_separation_viable_edges": sam2_strong_separation_viable_edges,
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
    parser.add_argument("--photometric-edges", type=Path,
                        help="Optional repeated-view image boundary evidence; absent means no photometric adjustment.")
    parser.add_argument("--photometric-weight", type=float, default=1.0)
    parser.add_argument("--sam2-comask-edges", type=Path,
                        help="Optional repeated-view SAM2 co-mask edge evidence; absent means no SAM2 adjustment.")
    parser.add_argument("--sam2-weight", type=float, default=1.0)
    args = parser.parse_args()
    rows, report = infer(
        read_jsonl(args.soft_unaries), read_jsonl(args.contact_edges), args.min_faces,
        args.min_contact_ratio, args.color_sigma, args.pairwise_weight, args.promotion_margin,
        read_jsonl(args.photometric_edges) if args.photometric_edges else None, args.photometric_weight,
        read_jsonl(args.sam2_comask_edges) if args.sam2_comask_edges else None, args.sam2_weight,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
