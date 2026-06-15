#!/usr/bin/env python3
"""Filter semantic manifest items by frame range and frame stride."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FRAME_RE = re.compile(r"cam\d+_(\d+)")
CAMERA_RE = re.compile(r"(cam\d+)_")


def item_frame(item: dict) -> int | None:
    for key in ("frame_id", "frame", "image_id"):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, int):
            return value
        match = FRAME_RE.search(str(value))
        if match:
            return int(match.group(1))
    return None


def item_camera(item: dict) -> str | None:
    for key in ("camera", "cam_id", "image_id"):
        value = item.get(key)
        if value is None:
            continue
        match = CAMERA_RE.search(str(value))
        if match:
            return match.group(1)
        text = str(value)
        if text.startswith("cam") and text[3:].isdigit():
            return text
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--camera", action="append", default=None, help="Optional camera id filter, e.g. cam0. Can be repeated.")
    args = parser.parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    cameras = set(args.camera or [])

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = []
    missing_frame = 0
    for item in data.get("items", []):
        frame = item_frame(item)
        if frame is None:
            missing_frame += 1
            continue
        if cameras:
            camera = item_camera(item)
            if camera not in cameras:
                continue
        if args.start_frame <= frame <= args.end_frame and (frame - args.start_frame) % args.frame_stride == 0:
            out = dict(item)
            out["frame_id"] = int(frame)
            rows.append(out)
    report = {
        "source": str(args.input),
        "output": str(args.output),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "frame_stride": args.frame_stride,
        "cameras": sorted(cameras),
        "items": len(rows),
        "missing_frame_items": missing_frame,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"items": rows, "filter_report": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
