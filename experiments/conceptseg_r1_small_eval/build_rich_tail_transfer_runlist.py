#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FOCUS_TO_CONCEPT = {
    "railing": "railing or thin metal guardrail",
    "pipe": "pipe or thin utility conduit",
    "equipment": "rooftop equipment box or HVAC unit",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def image_path_from_mask(mask_path: str) -> str:
    path = Path(mask_path)
    return str(path.parent.parent / "original.png")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def sort_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("grounding_score", 0.0)),
        float(row.get("sam_score", 0.0)),
        float(row.get("minrect_aspect_ratio", 0.0)),
    )


def build_items(rows: list[dict], *, top_k: int) -> list[dict]:
    by_focus: dict[str, list[dict]] = {}
    for row in rows:
        focus = str(row.get("focus") or "").strip().lower()
        if focus not in FOCUS_TO_CONCEPT:
            continue
        by_focus.setdefault(focus, []).append(row)

    items = []
    for focus, concept in FOCUS_TO_CONCEPT.items():
        selected = sorted(by_focus.get(focus, []), key=sort_key, reverse=True)[:top_k]
        for idx, row in enumerate(selected, start=1):
            sample_id = str(row["sample_id"])
            mask_path = str(row["mask_path"])
            image_path = image_path_from_mask(mask_path)
            image_id = f"rich_tail_{focus}_{idx:02d}_{sample_id}_{slug(Path(mask_path).stem)}"
            items.append(
                {
                    "image_id": image_id,
                    "concept": concept,
                    "image_path": image_path,
                    "metadata": {
                        "focus": focus,
                        "sample_id": sample_id,
                        "frame": int(row["frame"]),
                        "cam": int(row["cam"]),
                        "phrase": str(row.get("phrase", "")),
                        "mask_path": mask_path,
                        "source_label": str(row.get("source_label", focus)),
                        "grounding_score": float(row.get("grounding_score", 0.0)),
                        "sam_score": float(row.get("sam_score", 0.0)),
                        "mask_area_ratio": float(row.get("mask_area_ratio", 0.0)),
                        "box_aspect_ratio": float(row.get("box_aspect_ratio", 0.0)),
                        "minrect_aspect_ratio": float(row.get("minrect_aspect_ratio", 0.0)),
                    },
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--railing-jsonl", type=Path, required=True)
    parser.add_argument("--pipe-jsonl", type=Path, required=True)
    parser.add_argument("--equipment-jsonl", type=Path, required=True)
    parser.add_argument("--top-k-per-focus", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    rows.extend(read_jsonl(args.railing_jsonl))
    rows.extend(read_jsonl(args.pipe_jsonl))
    rows.extend(read_jsonl(args.equipment_jsonl))
    items = build_items(rows, top_k=args.top_k_per_focus)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runlist = {
        "schema": "conceptseg_rich_tail_transfer_v1",
        "selection_policy": {
            "top_k_per_focus": args.top_k_per_focus,
            "focuses": list(FOCUS_TO_CONCEPT.keys()),
            "sort_key": ["grounding_score", "sam_score", "minrect_aspect_ratio"],
            "note": "Server-first transfer check on rich-tail accepted fine-object images.",
        },
        "sources": {
            "railing_jsonl": str(args.railing_jsonl),
            "pipe_jsonl": str(args.pipe_jsonl),
            "equipment_jsonl": str(args.equipment_jsonl),
        },
        "items": items,
    }
    summary = {
        "items": len(items),
        "by_focus": {
            focus: sum(1 for item in items if item["metadata"]["focus"] == focus)
            for focus in FOCUS_TO_CONCEPT
        },
        "runlist": str(args.output_dir / "runlist.json"),
    }
    (args.output_dir / "runlist.json").write_text(json.dumps(runlist, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
