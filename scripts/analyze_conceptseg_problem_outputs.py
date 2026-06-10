#!/usr/bin/env python3
"""Analyze ConceptSeg-R1 problem-sample outputs.

The ConceptSeg runner writes visualization PNGs and a report with stdout/stderr.
This QA script extracts coarse but useful signals:

- answer/bbox text from the model output
- whether SAM3 or MLLM inference was used
- red-overlay ratio on the visualization image

It is intentionally conservative. The output is not a benchmark score; it is a
side-track triage report for deciding whether ConceptSeg deserves more GPU time.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
BBOX_RE = re.compile(r"<bbox>\s*\[?([^<\\]]+)\]?\s*</bbox>", re.DOTALL)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def overlay_stats(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.int16)
    h, w, _ = arr.shape
    panels = [arr]
    if w >= 900:
        panels.extend([arr[:, : w // 3, :], arr[:, w // 3 : (2 * w) // 3, :], arr[:, (2 * w) // 3 :, :]])
    red_ratios = []
    bright_ratios = []
    unique_counts = []
    for roi in panels:
        red = roi[:, :, 0]
        green = roi[:, :, 1]
        blue = roi[:, :, 2]
        # The visual overlay is often translucent pink/red rather than pure red.
        red_overlay = (red > 110) & (red > green + 20) & (red > blue + 20)
        bright = (red + green + blue) > 45
        red_ratios.append(float(red_overlay.mean()))
        bright_ratios.append(float(bright.mean()))
        unique_counts.append(int(len(np.unique(roi.reshape(-1, 3), axis=0))))
    return {
        "exists": True,
        "size": [int(w), int(h)],
        "red_overlay_ratio": float(max(red_ratios)),
        "red_overlay_ratio_by_panel": red_ratios,
        "bright_ratio": float(max(bright_ratios)),
        "unique_colors": int(max(unique_counts)),
    }


def resolve_output_path(raw_path: str, local_output_dir: Path | None) -> Path:
    path = Path(raw_path)
    if path.exists() or local_output_dir is None:
        return path
    candidate = local_output_dir / path.name
    return candidate if candidate.exists() else path


def parse_item(row: dict, local_output_dir: Path | None) -> dict:
    stdout = row.get("stdout_tail", "") or ""
    stderr = row.get("stderr_tail", "") or ""
    text = stdout + "\n" + stderr
    answer_match = ANSWER_RE.search(text)
    bbox_match = BBOX_RE.search(text)
    if "============using MLLM inference===============" in text:
        mode = "mllm"
    elif "============using sam3 inference===============" in text:
        mode = "sam3"
    else:
        mode = "unknown"
    return {
        "image_id": row.get("image_id"),
        "concept": row.get("concept"),
        "output_path": row.get("output_path"),
        "returncode": row.get("returncode"),
        "mode": mode,
        "answer": answer_match.group(1).strip() if answer_match else "",
        "bbox": bbox_match.group(1).strip() if bbox_match else "",
        "overlay": overlay_stats(resolve_output_path(row.get("output_path", ""), local_output_dir)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-output-dir", type=Path, default=None)
    parser.add_argument("--max-good-red-ratio", type=float, default=0.25)
    args = parser.parse_args()

    data = read_json(args.report)
    rows = [parse_item(row, args.local_output_dir) for row in data.get("items", [])]
    by_concept = defaultdict(list)
    for row in rows:
        by_concept[row["concept"]].append(row)

    concept_summary = {}
    for concept, items in sorted(by_concept.items()):
        red_values = [x["overlay"].get("red_overlay_ratio", 0.0) for x in items if x["overlay"].get("exists")]
        concept_summary[concept] = {
            "count": len(items),
            "returncode_counts": dict(Counter(str(x["returncode"]) for x in items)),
            "mode_counts": dict(Counter(x["mode"] for x in items)),
            "answer_counts": dict(Counter(x["answer"] or "<empty>" for x in items)),
            "avg_red_overlay_ratio": float(np.mean(red_values)) if red_values else 0.0,
            "overlarge_red_overlay_count": int(sum(v > args.max_good_red_ratio for v in red_values)),
        }

    summary = {
        "report": str(args.report),
        "items": len(rows),
        "returncode_counts": dict(Counter(str(x["returncode"]) for x in rows)),
        "mode_counts": dict(Counter(x["mode"] for x in rows)),
        "concept_summary": concept_summary,
        "items_detail": rows,
        "interpretation": {
            "max_good_red_ratio": args.max_good_red_ratio,
            "note": "High red-overlay ratio usually means the prompt selected broad surfaces, not a clean fine-object mask.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["items", "returncode_counts", "mode_counts", "concept_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
