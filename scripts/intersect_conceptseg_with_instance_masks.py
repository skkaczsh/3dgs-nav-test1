#!/usr/bin/env python3
"""Intersect ConceptSeg candidate masks with existing SAM2/Qwen instance masks.

This is an offline QA/refinement step. It measures whether ConceptSeg's
right-panel red mask actually overlaps the original instance mask that produced
the target. It does not change main-route semantic labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def conceptseg_red_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.int16)
    h, w, _ = arr.shape
    # ConceptSeg visualization is a 3-panel image; the rightmost panel carries
    # the final red overlay.
    panel = arr[:, (2 * w) // 3 :, :]
    red = panel[:, :, 0]
    green = panel[:, :, 1]
    blue = panel[:, :, 2]
    return (red > 110) & (red > green + 20) & (red > blue + 20)


def resized_instance_mask(path: Path, mask_id: int, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path)
    resized = img.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(resized)
    return arr == mask_id


def mask_stats(candidate: np.ndarray, instance: np.ndarray) -> dict[str, Any]:
    candidate_count = int(candidate.sum())
    instance_count = int(instance.sum())
    inter = candidate & instance
    union = candidate | instance
    intersection_count = int(inter.sum())
    union_count = int(union.sum())
    return {
        "candidate_pixels": candidate_count,
        "instance_pixels": instance_count,
        "intersection_pixels": intersection_count,
        "iou": intersection_count / union_count if union_count else 0.0,
        "candidate_inside_instance_ratio": intersection_count / candidate_count if candidate_count else 0.0,
        "instance_covered_ratio": intersection_count / instance_count if instance_count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-candidate-inside-instance", type=float, default=0.2)
    parser.add_argument("--min-instance-covered", type=float, default=0.005)
    parser.add_argument("--max-red-ratio", type=float, default=0.25)
    args = parser.parse_args()

    rows = read_jsonl(args.candidates)
    results: list[dict[str, Any]] = []
    for row in rows:
        local_assets = row.get("local_assets") or {}
        instance_path = Path(local_assets.get("instance", ""))
        output_path = args.local_output_dir / Path(row.get("output_path", "")).name
        mask_id = row.get("mask")
        errors: list[str] = []
        stats: dict[str, Any] = {}
        accept = False
        if not output_path.exists():
            errors.append(f"missing conceptseg output: {output_path}")
        if not instance_path.exists():
            errors.append(f"missing instance mask: {instance_path}")
        if mask_id is None:
            errors.append("missing mask id")
        if not errors:
            candidate = conceptseg_red_mask(output_path)
            h, w = candidate.shape
            instance = resized_instance_mask(instance_path, int(mask_id), (w, h))
            stats = mask_stats(candidate, instance)
            accept = (
                row.get("is_concept_match")
                and not row.get("is_overlarge")
                and row.get("red_overlay_ratio", 0.0) <= args.max_red_ratio
                and stats["candidate_inside_instance_ratio"] >= args.min_candidate_inside_instance
                and stats["instance_covered_ratio"] >= args.min_instance_covered
            )
        results.append(
            {
                **row,
                "intersection": stats,
                "intersection_accept": bool(accept),
                "intersection_errors": errors,
                "intersection_policy": {
                    "min_candidate_inside_instance": args.min_candidate_inside_instance,
                    "min_instance_covered": args.min_instance_covered,
                    "max_red_ratio": args.max_red_ratio,
                },
            }
        )

    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_target.setdefault(str(row.get("target_id")), []).append(row)
    target_rows = []
    for target_id, group in sorted(by_target.items()):
        accepted = [row for row in group if row["intersection_accept"]]
        target_rows.append(
            {
                "target_id": target_id,
                "review_id": group[0].get("review_id"),
                "object_id": group[0].get("object_id"),
                "frame": group[0].get("frame"),
                "cam": group[0].get("cam"),
                "mask": group[0].get("mask"),
                "candidate_count": len(group),
                "accepted_count": len(accepted),
                "accepted_concepts": sorted({row["concept_class"] for row in accepted}),
                "best_by_inside_ratio": max(
                    group,
                    key=lambda row: row.get("intersection", {}).get("candidate_inside_instance_ratio", 0.0),
                ).get("image_id"),
                "status": "has_intersection_candidate" if accepted else "no_intersection_candidate",
            }
        )

    report = {
        "candidate_count": len(results),
        "target_count": len(target_rows),
        "accepted_candidate_count": sum(1 for row in results if row["intersection_accept"]),
        "target_status_counts": dict(Counter(row["status"] for row in target_rows)),
        "accepted_concept_counts": dict(Counter(row["concept_class"] for row in results if row["intersection_accept"])),
        "error_count": sum(1 for row in results if row["intersection_errors"]),
        "interpretation": {
            "use": "Accepted rows are ConceptSeg local candidates that also overlap the original SAM2 instance mask.",
            "limitation": "This only validates 2D mask compatibility; 3D connected-component filtering is still required before point-level fusion.",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conceptseg_instance_intersections.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "conceptseg_instance_target_summary.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in target_rows) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "conceptseg_instance_intersection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
