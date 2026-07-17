#!/usr/bin/env python3
"""Build traceable soft semantic unaries from accepted multi-view evidence.

This is intentionally not a graph propagator. It records what each immutable
Superpoint has observed, separates no observation from uncertain observation,
and preserves an explicit unknown mass for the later graph optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def observation_weight(row: dict[str, Any]) -> float:
    """Visibility-quality weight; VLM confidence is applied separately."""
    score = max(float(row.get("score") or 0.0), 0.0)
    depth_visible = min(max(float(row.get("depth_visible_ratio") or 0.0), 0.0), 1.0)
    sky = min(max(float(row.get("sky_filtered_ratio") or 0.0), 0.0), 1.0)
    return score * depth_visible * (1.0 - sky)


def build_row(object_row: dict[str, Any], observations: list[dict[str, Any]], review: dict[str, Any] | None) -> dict[str, Any]:
    object_id = int(object_row["object_id"])
    raw_weights = [observation_weight(row) for row in observations]
    raw_sum = sum(raw_weights)
    if raw_sum > 0:
        view_weights = [weight / raw_sum for weight in raw_weights]
    elif observations:
        view_weights = [1.0 / len(observations)] * len(observations)
    else:
        view_weights = []
    effective_views = 1.0 / sum(weight * weight for weight in view_weights) if view_weights else 0.0
    parsed = (review or {}).get("parsed") or {}
    label = str(parsed.get("controlled_label") or "unknown")
    confidence = min(max(float(parsed.get("confidence") or 0.0), 0.0), 1.0)
    support = min(1.0, math.log1p(raw_sum)) if raw_sum > 0 else 0.0
    alpha: dict[str, float] = {}
    state = "unobserved"
    if observations:
        state = "observed_unlabeled"
    if parsed:
        state = "reviewed"
        alpha[label] = round(support * confidence, 6)
        alpha["unknown"] = round(support * (1.0 - confidence), 6)
    return {
        "object_id": object_id,
        "geometry_type": str(object_row.get("geometry_type") or "unknown"),
        "state": state,
        "alpha": alpha,
        "controlled_label": label if parsed else "unknown",
        "review_confidence": confidence if parsed else 0.0,
        "evidence_count": len(observations),
        "effective_view_count": round(effective_views, 4),
        "visibility_support": round(support, 6),
        "evidence_ids": [f"{int(row['frame_id'])}:{int(row['cam_id'])}:{int(row.get('rank') or 0)}" for row in observations],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.evidence_jsonl):
        observations[int(row["object_id"])].append(row)
    reviews = {int(row["object_id"]): row for row in read_jsonl(args.review_jsonl)}
    rows = [build_row(row, observations[int(row["object_id"])], reviews.get(int(row["object_id"]))) for row in read_jsonl(args.objects_jsonl)]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "objects": len(rows),
        "unobserved": sum(row["state"] == "unobserved" for row in rows),
        "observed_unlabeled": sum(row["state"] == "observed_unlabeled" for row in rows),
        "reviewed": sum(row["state"] == "reviewed" for row in rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
