#!/usr/bin/env python3
"""Recover review sample -> frame/cam mapping from ConceptSeg QA artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


IMAGE_RE = re.compile(
    r"^(review_\d+)_(a\d+|b\d+)_fine_t_(\d+)_cam(\d+)_mask(\d+)_sem(\d+)_cc(\d+)_"
)


def parse_image_id(image_id: str) -> dict | None:
    m = IMAGE_RE.match(image_id)
    if not m:
        return None
    review, slot, frame, cam, mask, semantic, cluster = m.groups()
    return {
        "image_id": image_id,
        "review_id": review,
        "slot": slot,
        "sample_id": f"{review}__{slot}",
        "frame": int(frame),
        "cam": int(cam),
        "mask": int(mask),
        "semantic": int(semantic),
        "cluster": int(cluster),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.qa_json.read_text(encoding="utf-8"))
    items = raw.get("items_detail", []) if isinstance(raw, dict) else raw
    best_by_sample: dict[str, dict] = {}
    for row in items:
        parsed = parse_image_id(str(row.get("image_id", "")))
        if not parsed:
            continue
        sample_id = parsed["sample_id"]
        prev = best_by_sample.get(sample_id)
        if prev is None or parsed["frame"] < prev["frame"]:
            best_by_sample[sample_id] = {
                "image_id": f"{parsed['review_id']}_{parsed['slot']}",
                "target_id": sample_id,
                "frame": parsed["frame"],
                "cam": parsed["cam"],
                "mask": parsed["mask"],
                "semantic": parsed["semantic"],
                "source_label": row.get("concept", "unknown"),
                "qa_image_id": parsed["image_id"],
            }

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for sample_id in sorted(best_by_sample):
            f.write(json.dumps(best_by_sample[sample_id], ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "qa_json": str(args.qa_json),
                "output_jsonl": str(args.output_jsonl),
                "sample_count": len(best_by_sample),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
