#!/usr/bin/env python3
"""Build a minimal TVP side-track manifest from accepted 2D fine candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def concept_prompt(row: dict) -> str:
    concept = str(row.get("concept") or "").strip()
    answer = str(row.get("answer") or "").strip()
    answer_class = str(row.get("answer_class") or "").strip()
    parts = [x for x in [concept, answer, answer_class] if x]
    return " ; ".join(parts) if parts else "fine object"


def locate_prompt(row: dict, field: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        value = str(row.get("answer_class") or row.get("source_label") or "object").strip()
    article = "an" if value[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    if value.lower().startswith(("a ", "an ", "the ")):
        return f"Locate {value} in the image."
    return f"Locate {article} {value} in the image."


def resolve_prompt(row: dict, mode: str, locate_field: str) -> str:
    if mode == "concept":
        return concept_prompt(row)
    if mode == "locate":
        return locate_prompt(row, locate_field)
    raise ValueError(f"unsupported prompt mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-mode", choices=["concept", "locate"], default="concept")
    parser.add_argument(
        "--locate-field",
        choices=["concept", "concept_class", "answer", "answer_class", "source_label"],
        default="answer_class",
    )
    args = parser.parse_args()

    rows = []
    with args.accepted_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[: args.limit]

    samples = []
    for index, row in enumerate(rows):
        assets = row.get("remote_assets") or row.get("local_assets") or {}
        sample = {
            "id": f"tvp_{index:04d}_{row['target_id']}",
            "target_id": row["target_id"],
            "frame": int(row["frame"]),
            "cam": int(row["cam"]),
            "mask": int(row["mask"]),
            "semantic": int(row["semantic"]),
            "source_label": row.get("source_label", "unknown"),
            "concept": row.get("concept", ""),
            "concept_class": row.get("concept_class", ""),
            "answer": row.get("answer", ""),
            "answer_class": row.get("answer_class", ""),
            "prompt": resolve_prompt(row, args.prompt_mode, args.locate_field),
            "bbox": row.get("bbox"),
            "image_path": assets.get("image", ""),
            "instance_path": assets.get("instance", ""),
            "overlay_path": assets.get("overlay", ""),
            "semantic_path": assets.get("semantic", ""),
            "candidate_pixels": int(row.get("candidate_pixels", 0)),
            "instance_pixels": int(row.get("instance_pixels", 0)),
            "intersection_pixels": int(row.get("intersection_pixels", 0)),
            "candidate_inside_instance_ratio": float(row.get("candidate_inside_instance_ratio", 0.0)),
            "instance_covered_ratio": float(row.get("instance_covered_ratio", 0.0)),
            "red_overlay_ratio": float(row.get("red_overlay_ratio", 0.0)),
        }
        samples.append(sample)

    manifest = {
        "schema": "tvp_side_track_manifest_v1",
        "source": str(args.accepted_jsonl),
        "prompt_mode": args.prompt_mode,
        "locate_field": args.locate_field,
        "sample_count": len(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({"sample_count": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
