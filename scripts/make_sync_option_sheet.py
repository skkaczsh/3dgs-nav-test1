#!/usr/bin/env python3
"""Create a contact sheet for one sync-review option source.

This is a lightweight QA helper for synchronization review packs.  It does not
modify anchors or mark anything accepted; it only extracts options such as
`smooth_path`, `direct`, or `independent_best` from `manual_anchor_manifest.jsonl`
and renders a compact sheet for fast visual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def choose_option(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    options = row.get("options") or []
    for option in options:
        if str(option.get("review_source")) == source:
            return option
    return None


def draw_title(image: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_sheet(
    manifest: Path,
    source_dir: Path,
    option_source: str,
    output: Path,
    cols: int,
    thumb_width: int,
) -> dict[str, Any]:
    rows = read_jsonl(manifest)
    panels: list[np.ndarray] = []
    selected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        option = choose_option(row, option_source)
        if option is None:
            missing.append({
                "frame_id": int(row["frame_id"]),
                "cam_id": int(row["cam_id"]),
                "reason": f"missing_option_source={option_source}",
            })
            continue
        panel_path = source_dir / str(option["panel_path"])
        image = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
        record = {
            "frame_id": int(row["frame_id"]),
            "cam_id": int(row["cam_id"]),
            "option_idx": int(option["option_idx"]),
            "review_source": str(option["review_source"]),
            "video_idx": int(option["video_idx"]),
            "offset": int(option.get("offset", 0)),
            "score": float(option.get("score", 0.0)),
            "panel_path": str(panel_path),
        }
        selected.append(record)
        if image is None:
            missing.append({**record, "reason": "panel_image_missing"})
            continue
        scale = float(thumb_width) / max(float(image.shape[1]), 1.0)
        thumb = cv2.resize(image, (thumb_width, max(1, int(round(image.shape[0] * scale)))))
        title = (
            f"f={record['frame_id']} cam={record['cam_id']} "
            f"v={record['video_idx']} off={record['offset']} score={record['score']:.3f}"
        )
        panels.append(draw_title(thumb, title))

    output.parent.mkdir(parents=True, exist_ok=True)
    if panels:
        height = max(panel.shape[0] for panel in panels)
        width = max(panel.shape[1] for panel in panels)
        normalized = []
        for panel in panels:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[: panel.shape[0], : panel.shape[1]] = panel
            normalized.append(canvas)
        while len(normalized) % cols:
            normalized.append(np.zeros_like(normalized[0]))
        grid_rows = [np.hstack(normalized[i:i + cols]) for i in range(0, len(normalized), cols)]
        cv2.imwrite(str(output), np.vstack(grid_rows))

    report = {
        "manifest": str(manifest),
        "source_dir": str(source_dir),
        "option_source": option_source,
        "output": str(output),
        "row_count": len(rows),
        "selected_count": len(selected),
        "rendered_count": len(panels),
        "missing_count": len(missing),
        "missing": missing,
        "selected": selected,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--option-source", default="smooth_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--thumb-width", type=int, default=420)
    args = parser.parse_args()

    report = build_sheet(
        args.manifest,
        args.source_dir,
        args.option_source,
        args.output,
        args.cols,
        args.thumb_width,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
