#!/usr/bin/env python3
"""Align ConceptSeg fine-object outputs back to target/object review metadata.

This script does not promote ConceptSeg to the main semantic route. It creates
an object-level QA table so ConceptSeg masks can be used as constrained
second-stage candidates inside the existing target/object workflow.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
BBOX_RE = re.compile(r"<bbox>\s*\[?\s*([0-9.,\s-]+?)\s*\]?\s*</bbox>", re.DOTALL)

CONCEPT_TO_CLASS = {
    "railing or thin metal guardrail": "railing",
    "rooftop equipment box or HVAC unit": "equipment",
    "pipe or thin utility conduit": "pipe",
}

ANSWER_ALIASES = {
    "rail": "railing",
    "railing": "railing",
    "guardrail": "railing",
    "fence": "railing",
    "hvac": "equipment",
    "hvac unit": "equipment",
    "unit": "equipment",
    "aircon": "equipment",
    "duct": "equipment",
    "red box": "equipment",
    "pipe": "pipe",
    "pipes": "pipe",
    "conduit": "pipe",
    "cables": "pipe",
    "pole": "pipe",
    "nonexistent": "nonexistent",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_answer(text: str) -> str:
    match = ANSWER_RE.search(text or "")
    return match.group(1).strip() if match else ""


def parse_bbox(text: str) -> list[float] | None:
    match = BBOX_RE.search(text or "")
    if not match:
        return None
    parts = [part.strip() for part in match.group(1).split(",")]
    if len(parts) != 4:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def normalize_answer(answer: str) -> str:
    return ANSWER_ALIASES.get(answer.strip().lower(), "other" if answer else "empty")


def bbox_area(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def item_key(item: dict[str, Any]) -> tuple[str, str, str]:
    meta = item.get("metadata") or {}
    rep = meta.get("representative") or {}
    return (
        str(meta.get("review_id") or ""),
        str(rep.get("object_id") or ""),
        str(rep.get("target_id") or ""),
    )


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    concepts = Counter(row["concept_class"] for row in rows)
    answers = Counter(row["answer_class"] for row in rows)
    acceptable = [row for row in rows if row["is_concept_match"] and not row["is_overlarge"]]
    risky = [row for row in rows if row["is_overlarge"] or row["answer_class"] in {"other", "empty", "nonexistent"}]
    matched_classes = sorted({row["answer_class"] for row in acceptable})
    best = sorted(
        rows,
        key=lambda row: (
            row["is_concept_match"],
            not row["is_overlarge"],
            row["red_overlay_ratio"] > 0,
            -row["red_overlay_ratio"],
        ),
        reverse=True,
    )[0]
    return {
        "review_id": rows[0]["review_id"],
        "object_id": rows[0]["object_id"],
        "target_id": rows[0]["target_id"],
        "source_label": rows[0]["source_label"],
        "frame": rows[0].get("frame"),
        "cam": rows[0].get("cam"),
        "mask": rows[0].get("mask"),
        "concept_counts": dict(concepts),
        "answer_counts": dict(answers),
        "candidate_count": len(rows),
        "acceptable_candidate_count": len(acceptable),
        "risky_candidate_count": len(risky),
        "matched_classes": matched_classes,
        "is_semantically_discriminative": len(matched_classes) <= 1,
        "best_candidate": {
            key: best[key]
            for key in [
                "concept",
                "concept_class",
                "answer",
                "answer_class",
                "bbox",
                "bbox_area",
                "red_overlay_ratio",
                "is_concept_match",
                "is_overlarge",
                "output_path",
            ]
        },
        "status": "usable_candidate" if acceptable else "needs_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True, help="Merged ConceptSeg report.json.")
    parser.add_argument("--qa", type=Path, required=True, help="Structured ConceptSeg QA JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlarge-red-ratio", type=float, default=0.25)
    args = parser.parse_args()

    report = read_json(args.report)
    qa = read_json(args.qa)
    qa_by_output = {Path(row["output_path"]).name: row for row in qa.get("items_detail", [])}

    rows: list[dict[str, Any]] = []
    for item in report.get("items", []):
        meta = item.get("metadata") or {}
        rep = meta.get("representative") or {}
        target_meta = rep.get("target_meta") or {}
        stdout = item.get("stdout_tail", "") or ""
        stderr = item.get("stderr_tail", "") or ""
        text = stdout + "\n" + stderr
        answer = parse_answer(text)
        bbox = parse_bbox(text)
        output_name = Path(item.get("output_path", "")).name
        qa_row = qa_by_output.get(output_name, {})
        red_ratio = float((qa_row.get("overlay") or {}).get("red_overlay_ratio", 0.0))
        concept = str(item.get("concept") or "")
        concept_class = CONCEPT_TO_CLASS.get(concept, "other")
        answer_class = normalize_answer(answer or str(qa_row.get("answer") or ""))
        row = {
            "absolute_index": item.get("absolute_index"),
            "image_id": item.get("image_id"),
            "review_id": meta.get("review_id"),
            "object_id": rep.get("object_id"),
            "target_id": rep.get("target_id"),
            "tracklet_id": rep.get("tracklet_id"),
            "candidate": rep.get("candidate"),
            "source_label": meta.get("source_label"),
            "frame": target_meta.get("frame"),
            "cam": target_meta.get("cam"),
            "mask": target_meta.get("mask"),
            "semantic": target_meta.get("semantic"),
            "concept": concept,
            "concept_class": concept_class,
            "answer": answer or qa_row.get("answer", ""),
            "answer_class": answer_class,
            "bbox": bbox,
            "bbox_area": bbox_area(bbox),
            "red_overlay_ratio": red_ratio,
            "is_concept_match": concept_class == answer_class,
            "is_overlarge": red_ratio > args.overlarge_red_ratio,
            "mode": qa_row.get("mode"),
            "returncode": item.get("returncode"),
            "output_path": item.get("output_path"),
            "local_assets": meta.get("local_assets", {}),
            "remote_assets": meta.get("remote_assets", {}),
        }
        rows.append(row)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["review_id"]), str(row["object_id"]), str(row["target_id"]))].append(row)
    target_summaries = [summarize_group(group_rows) for group_rows in groups.values()]

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in target_summaries:
        by_object[str(summary["object_id"])].append(summary)
    object_summaries = []
    for object_id, summaries in sorted(by_object.items()):
        status_counts = Counter(summary["status"] for summary in summaries)
        answer_counts: Counter[str] = Counter()
        for summary in summaries:
            answer_counts.update(summary["answer_counts"])
        object_summaries.append(
            {
                "object_id": object_id,
                "target_count": len(summaries),
                "status_counts": dict(status_counts),
                "answer_counts": dict(answer_counts),
                "usable_target_ratio": status_counts.get("usable_candidate", 0) / max(1, len(summaries)),
                "non_discriminative_target_count": sum(
                    1 for summary in summaries if not summary.get("is_semantically_discriminative")
                ),
                "status": "usable_conceptseg_review"
                if status_counts.get("usable_candidate", 0) >= max(1, len(summaries) // 2)
                else "needs_manual_review",
            }
        )

    report_out = {
        "source_report": str(args.report),
        "source_qa": str(args.qa),
        "item_count": len(rows),
        "target_count": len(target_summaries),
        "object_count": len(object_summaries),
        "returncode_counts": dict(Counter(str(row["returncode"]) for row in rows)),
        "mode_counts": dict(Counter(str(row["mode"]) for row in rows)),
        "concept_counts": dict(Counter(row["concept_class"] for row in rows)),
        "answer_counts": dict(Counter(row["answer_class"] for row in rows)),
        "concept_match_count": sum(1 for row in rows if row["is_concept_match"]),
        "overlarge_count": sum(1 for row in rows if row["is_overlarge"]),
        "target_status_counts": dict(Counter(row["status"] for row in target_summaries)),
        "object_status_counts": dict(Counter(row["status"] for row in object_summaries)),
        "non_discriminative_target_count": sum(
            1 for row in target_summaries if not row.get("is_semantically_discriminative")
        ),
        "semantically_discriminative_target_count": sum(
            1 for row in target_summaries if row.get("is_semantically_discriminative")
        ),
        "interpretation": {
            "use": "ConceptSeg-R1 is aligned as a second-stage target/object candidate review signal.",
            "classification_warning": "A target can match multiple prompts; ConceptSeg-R1 should not be used as the target semantic classifier from these results.",
            "no_go": "Do not treat these masks as dense semantic production outputs without 3D connected-component and existing-mask intersection filters.",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conceptseg_target_candidates.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "conceptseg_target_summary.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in target_summaries) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "conceptseg_object_summary.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in object_summaries) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "conceptseg_target_object_alignment_report.json").write_text(
        json.dumps(report_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
