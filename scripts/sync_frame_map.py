#!/usr/bin/env python3
"""Utilities for explicit section/camera -> video-frame mappings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FrameMap = dict[tuple[int, int], int]

REJECTED_STATUSES = {
    "rejected",
    "rejected_unstable_temporal_path",
    "sync_failed",
    "read_failed",
    "missing_frame_map",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def selected_video_idx(row: dict[str, Any]) -> int:
    """Return the chosen video frame index from supported sync row formats."""
    if row.get("selected_video_idx") is not None:
        return int(row["selected_video_idx"])
    if row.get("video_idx") is not None:
        return int(row["video_idx"])
    option_idx = row.get("selected_option_idx")
    if option_idx is not None:
        for option in row.get("options", []):
            if int(option.get("option_idx", -1)) == int(option_idx):
                return int(option["video_idx"])
    raise ValueError(
        "sync row missing selected_video_idx/video_idx/selected option: "
        f"frame={row.get('frame_id')} cam={row.get('cam_id')}"
    )


def row_rejection_status(row: dict[str, Any]) -> str | None:
    """Return a rejection status if a row is unsafe for production mapping."""
    for key in ("anchor_status", "cam_path_status", "status"):
        value = str(row.get(key, "")).lower()
        if value in {"", "accepted", "ok"}:
            continue
        if value in {"unreviewed", "rejected"}:
            return value
        if value.startswith("rejected"):
            return value
        if value in REJECTED_STATUSES:
            return value
    return None


def load_frame_map(path: Path | None, *, allow_rejected: bool = False) -> FrameMap:
    if path is None:
        return {}
    frame_map: FrameMap = {}
    for row in read_jsonl(path):
        if "frame_id" not in row or "cam_id" not in row:
            raise ValueError(f"sync row missing frame_id/cam_id in {path}: {row}")
        rejection = row_rejection_status(row)
        if rejection in {"rejected", "unreviewed"} and "anchor_status" in row:
            continue
        if rejection and not allow_rejected:
            raise ValueError(
                f"unsafe sync row status={rejection!r} in {path}: "
                f"frame={row.get('frame_id')} cam={row.get('cam_id')}"
            )
        key = (int(row["frame_id"]), int(row["cam_id"]))
        value = selected_video_idx(row)
        old = frame_map.get(key)
        if old is not None and old != value:
            raise ValueError(
                f"conflicting video frame for frame={key[0]} cam={key[1]}: {old} vs {value}"
            )
        frame_map[key] = value
    return frame_map


def resolve_video_idx(frame_map: FrameMap, frame_id: int, cam_id: int, fallback_to_direct: bool = True) -> int | None:
    value = frame_map.get((int(frame_id), int(cam_id)))
    if value is not None:
        return value
    if fallback_to_direct:
        return int(frame_id)
    return None
