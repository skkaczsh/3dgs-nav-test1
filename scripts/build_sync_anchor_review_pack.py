#!/usr/bin/env python3
"""Build a manual anchor review pack for LiDAR/video synchronization.

Inputs are candidate scores from `calibrate_lx_video_frame_mapping.py` and,
optionally, a smooth path from `solve_sync_path_from_candidates.py`.  For each
section/camera probe, this renders a compact set of choices:

- direct index candidate (`offset == 0`)
- independent best score
- smooth-path candidate
- top-N remaining candidates

The output is intentionally human-reviewable.  The generated
`manual_anchor_manifest.jsonl` is a template: fill `selected_video_idx` and
`anchor_status` for reliable anchors, then use it to constrain the next sync
optimizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from calibrate_lx_video_frame_mapping import (
    CandidateScore,
    draw_panel,
    project_points,
    read_frame,
    undistort_maps,
    visible_pixels,
)
from project_priority_masks_to_lx import read_lx_points, read_lx_sections


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_probe(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["frame_id"]), int(row["cam_id"])), []).append(row)
    for items in grouped.values():
        items.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return grouped


def row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["frame_id"]), int(row["cam_id"]), int(row["video_idx"]))


def choose_review_options(
    candidates: list[dict[str, Any]],
    smooth_row: dict[str, Any] | None,
    top_n: int,
) -> list[dict[str, Any]]:
    """Select deduplicated review options in stable priority order."""
    if not candidates:
        return []
    options: list[dict[str, Any]] = []

    def add(row: dict[str, Any], source: str) -> None:
        item = dict(row)
        item["review_source"] = source
        key = row_key(item)
        if key not in {row_key(existing) for existing in options}:
            options.append(item)

    direct = next((row for row in candidates if int(row.get("offset", 10**9)) == 0), None)
    if direct is not None:
        add(direct, "direct")
    add(candidates[0], "independent_best")
    if smooth_row is not None:
        add(smooth_row, "smooth_path")
    for row in candidates[:top_n]:
        add(row, "top_candidate")
    return options


def make_sheet(panels: list[np.ndarray], output: Path, cols: int) -> None:
    if not panels:
        return
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def render_option(
    lx_handle,
    sections: list[dict[str, Any]],
    poses: dict[int, dict[str, Any]],
    maps: dict[int, tuple[np.ndarray, np.ndarray]],
    caps: dict[int, cv2.VideoCapture],
    row: dict[str, Any],
    dot_px: int,
) -> np.ndarray | None:
    frame_id = int(row["frame_id"])
    cam_id = int(row["cam_id"])
    video_idx = int(row["video_idx"])
    if frame_id >= len(sections) or frame_id not in poses:
        return None
    points = read_lx_points(lx_handle, sections[frame_id])
    u, v, z = project_points(points, poses[frame_id], cam_id, 0.1)
    raw = read_frame(caps[cam_id], video_idx)
    if raw is None:
        return None
    map1, map2 = maps[cam_id]
    image = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
    uu, vv, _depth = visible_pixels(u, v, z, image.shape[1], image.shape[0])
    score = CandidateScore(
        frame_id=frame_id,
        cam_id=cam_id,
        video_idx=video_idx,
        offset=int(row.get("offset", video_idx - frame_id)),
        visible=int(row.get("visible", len(uu))),
        edge_hit=float(row.get("edge_hit", 0.0)),
        edge_distance_mean=float(row.get("edge_distance_mean", 0.0)),
        edge_distance_p50=float(row.get("edge_distance_p50", 0.0)),
        score=float(row.get("score", 0.0)),
    )
    panel = draw_panel(image, uu, vv, score, dot_px)
    source = str(row.get("review_source", "candidate"))
    cv2.putText(panel, source, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, source, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx-file", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--smooth-path-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--dot-px", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = group_by_probe(load_jsonl(args.candidates_jsonl))
    smooth = {row_key(row): row for row in load_jsonl(args.smooth_path_jsonl)}
    sections = read_lx_sections(args.lx_file)
    frame_ids = sorted({key[0] for key in candidates})
    poses = {row["frame_id"]: row for row in config.load_img_pos(min(frame_ids), max(frame_ids))}
    maps = {cam_id: undistort_maps(cam_id) for cam_id in config.VIDEO_FILES}
    caps = {cam_id: cv2.VideoCapture(path) for cam_id, path in config.VIDEO_FILES.items()}
    for cam_id, cap in caps.items():
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video cam{cam_id}: {config.VIDEO_FILES[cam_id]}")

    manifest_rows = []
    panels = []
    with args.lx_file.open("rb") as lx_handle:
        for key in sorted(candidates):
            frame_id, cam_id = key
            smooth_row = next((row for row in smooth.values() if int(row["frame_id"]) == frame_id and int(row["cam_id"]) == cam_id), None)
            options = choose_review_options(candidates[key], smooth_row, args.top_n)
            option_records = []
            for option_idx, row in enumerate(options):
                option_records.append({
                    "option_idx": option_idx,
                    "review_source": row.get("review_source"),
                    "video_idx": int(row["video_idx"]),
                    "offset": int(row.get("offset", int(row["video_idx"]) - frame_id)),
                    "score": float(row.get("score", 0.0)),
                    "edge_hit": float(row.get("edge_hit", 0.0)),
                    "edge_distance_mean": float(row.get("edge_distance_mean", 0.0)),
                })
                panel = render_option(lx_handle, sections, poses, maps, caps, row, args.dot_px)
                if panel is not None:
                    panels.append(panel)
            manifest_rows.append({
                "frame_id": frame_id,
                "cam_id": cam_id,
                "anchor_status": "unreviewed",
                "selected_video_idx": None,
                "selected_option_idx": None,
                "notes": "",
                "options": option_records,
            })
    for cap in caps.values():
        cap.release()

    with (args.output_dir / "manual_anchor_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "candidates_jsonl": str(args.candidates_jsonl),
        "smooth_path_jsonl": str(args.smooth_path_jsonl) if args.smooth_path_jsonl else None,
        "probe_count": len(manifest_rows),
        "panel_count": len(panels),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "manual_anchor_review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_sheet(panels, args.output_dir / "manual_anchor_review_sheet.jpg", args.sheet_cols)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
