#!/usr/bin/env python3
"""Build a fine-mask evaluation manifest from object QA evidence.

The manifest is the handoff between point-cloud QA and the next 2D fine-mask
test.  It selects risky fine-object evidence frames and records the source
image, current priority mask, 2D bbox, and object/target provenance so a SAM2
loop/coverage experiment can be run on a stable sample set.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LABELS = ["railing", "car"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def bbox_area(row: dict[str, Any]) -> int:
    bbox = row.get("bbox_2d") or {}
    xyxy = bbox.get("xyxy") or []
    if len(xyxy) != 4:
        return as_int(bbox.get("area"))
    x0, y0, x1, y1 = [as_float(v) for v in xyxy]
    return int(max(0.0, x1 - x0) * max(0.0, y1 - y0))


def row_sort_key(row: dict[str, Any]) -> tuple[float, int, int]:
    return (
        as_float(row.get("risk_score")),
        as_int(row.get("cluster_size")),
        bbox_area(row),
    )


def select_rows(
    rows: list[dict[str, Any]],
    labels: list[str],
    limit: int,
    per_object_limit: int,
) -> list[dict[str, Any]]:
    wanted = {normalize_label(label) for label in labels}
    filtered = [row for row in rows if normalize_label(row.get("semantic_label")) in wanted]
    filtered.sort(key=row_sort_key, reverse=True)
    selected: list[dict[str, Any]] = []
    per_object_counts: dict[str, int] = defaultdict(int)
    for row in filtered:
        object_id = str(row.get("object_id"))
        if per_object_limit and per_object_counts[object_id] >= per_object_limit:
            continue
        selected.append(row)
        per_object_counts[object_id] += 1
        if limit and len(selected) >= limit:
            break
    return selected


def manifest_item(row: dict[str, Any], index: int) -> dict[str, Any]:
    cam_id = as_int(row.get("cam_id"))
    frame_id = as_int(row.get("frame_id"))
    return {
        "sample_id": f"{index:04d}_obj{row.get('object_id')}_cam{cam_id}_frame{frame_id:06d}",
        "object_id": row.get("object_id"),
        "semantic_label": row.get("semantic_label"),
        "risk_score": as_float(row.get("risk_score")),
        "risk_reasons": row.get("risk_reasons") or [],
        "target_id": row.get("target_id"),
        "target_label": row.get("target_label"),
        "frame_id": frame_id,
        "cam_id": cam_id,
        "cluster_size": as_int(row.get("cluster_size")),
        "bbox_2d": row.get("bbox_2d") or {},
        "image_path": row.get("image_path"),
        "current_mask_path": row.get("mask_path"),
        "current_mask_overlay": row.get("mask_overlay"),
        "crop_path": row.get("crop_path"),
        "recommended_eval": [
            "use undistorted image as input",
            "apply existing skymask before SAM2",
            "run SAM2 loop / coverage completion inside or around bbox",
            "compare against current priority mask and projected 3D local-geometry guard",
        ],
    }


def build_manifest(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    selected = select_rows(rows, args.labels, args.limit, args.per_object_limit)
    items = [manifest_item(row, i + 1) for i, row in enumerate(selected)]
    label_counts: dict[str, int] = defaultdict(int)
    object_ids: set[str] = set()
    for item in items:
        label_counts[normalize_label(item.get("semantic_label"))] += 1
        object_ids.add(str(item.get("object_id")))
    return {
        "source_evidence_jsonl": str(args.evidence_jsonl),
        "labels": args.labels,
        "limit": args.limit,
        "per_object_limit": args.per_object_limit,
        "sample_count": len(items),
        "object_count": len(object_ids),
        "label_counts": dict(sorted(label_counts.items())),
        "method_under_test": "undistorted image + skymask + SAM2 loop / coverage completion",
        "items": items,
    }


def markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Fine Mask Evaluation Manifest",
        "",
        f"Source evidence: `{manifest['source_evidence_jsonl']}`",
        "",
        f"Method under test: `{manifest['method_under_test']}`",
        "",
        f"Samples: `{manifest['sample_count']}`",
        f"Objects: `{manifest['object_count']}`",
        f"Label counts: `{manifest['label_counts']}`",
        "",
        "| sample | label | object | frame | cam | risk | cluster | reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in manifest["items"]:
        reasons = ",".join(str(v) for v in item.get("risk_reasons", []))
        lines.append(
            f"| `{item['sample_id']}` | `{item['semantic_label']}` | {item['object_id']} | "
            f"{item['frame_id']} | {item['cam_id']} | {item['risk_score']:.1f} | "
            f"{item['cluster_size']:,} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## Next Evaluation Gate",
            "",
            "- Run the selected samples through the SAM2 loop / coverage completion path.",
            "- Reject a candidate if it swallows adjacent ground/wall/stair surfaces into the fine mask.",
            "- Promote only if bbox-local overlays improve fine-object precision without losing obvious handrail/car coverage.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--per-object-limit", type=int, default=3)
    args = parser.parse_args()

    manifest = build_manifest(read_jsonl(args.evidence_jsonl), args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "sample_count": manifest["sample_count"],
                "object_count": manifest["object_count"],
                "label_counts": manifest["label_counts"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
