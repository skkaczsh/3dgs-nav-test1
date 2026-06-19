#!/usr/bin/env python3
"""Audit timing sources for MANIFOLD .lx, img_pos.txt, and camera videos.

This script does not try to fix synchronization.  It makes the available time
signals explicit so the route can decide whether image projection is safe:

- .lx section header raw float/uint fields;
- .lx section counts and offsets;
- img_pos frame ids, unix timestamps, and cam_info fields;
- video frame count/FPS/PTS summary;
- optional point-marker statistics from selected .lx sections.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


LX_HEADER_SIZE = 48
LX_COUNT_SIZE = 4
LX_POINT_SIZE = 16


def read_lx_headers(path: Path, max_sections: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    size = path.stat().st_size
    offset = 0
    section = 0
    with path.open("rb") as f:
        while offset + LX_HEADER_SIZE + LX_COUNT_SIZE <= size:
            f.seek(offset)
            header = f.read(LX_HEADER_SIZE)
            count_raw = f.read(LX_COUNT_SIZE)
            if len(header) != LX_HEADER_SIZE or len(count_raw) != LX_COUNT_SIZE:
                break
            count = struct.unpack("<I", count_raw)[0]
            if count == 0 or count > 50_000_000:
                break
            data_offset = offset + LX_HEADER_SIZE + LX_COUNT_SIZE
            next_offset = data_offset + count * LX_POINT_SIZE
            if next_offset > size + 16:
                break
            floats = struct.unpack("<12f", header)
            uints = struct.unpack("<12I", header)
            rows.append({
                "section": section,
                "offset": offset,
                "data_offset": data_offset,
                "count": int(count),
                "floats": [float(x) for x in floats],
                "uints": [int(x) for x in uints],
            })
            section += 1
            offset = next_offset
            if max_sections and section >= max_sections:
                break
    return rows


def video_summary(video_path: str, sample_frames: int) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    opened = cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    cap.release()
    pts_head = []
    if opened and sample_frames > 0:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-read_intervals",
            f"0%+#{sample_frames}",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time,pkt_duration_time",
            "-of",
            "csv=p=0",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines()[:sample_frames]:
                parts = [x for x in line.split(",") if x != ""]
                if parts:
                    try:
                        pts_head.append(float(parts[0]))
                    except ValueError:
                        pass
    return {
        "path": video_path,
        "opened": opened,
        "fps": fps,
        "frame_count": count,
        "duration_by_count": count / fps if fps > 0 else None,
        "pts_head": pts_head,
    }


def summarize_numeric(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def correlation_summary(headers: list[dict[str, Any]], poses: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(headers), len(poses))
    if n < 2:
        return {}
    pose_ts = np.asarray([poses[i]["timestamp"] for i in range(n)], dtype=np.float64)
    pose_rel = pose_ts - pose_ts[0]
    section = np.asarray([headers[i]["section"] for i in range(n)], dtype=np.float64)
    out: dict[str, Any] = {}
    for kind in ("floats", "uints"):
        fields = []
        for j in range(12):
            vals = np.asarray([headers[i][kind][j] for i in range(n)], dtype=np.float64)
            finite = np.isfinite(vals)
            if finite.sum() < 2 or np.std(vals[finite]) == 0:
                corr_ts = None
                corr_section = None
            else:
                corr_ts = float(np.corrcoef(vals[finite], pose_rel[finite])[0, 1])
                corr_section = float(np.corrcoef(vals[finite], section[finite])[0, 1])
            fields.append({
                "field": f"{kind[:-1]}{j}",
                "summary": summarize_numeric(vals[np.isfinite(vals)]),
                "corr_pose_rel_time": corr_ts,
                "corr_section": corr_section,
                "first_values": [float(x) for x in vals[:8]],
            })
        out[kind] = fields
    return out


def pose_match_summary(headers: list[dict[str, Any]], poses: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(headers), len(poses))
    if n == 0:
        return {}
    pos_err = []
    quat_err = []
    section_id_match = 0
    for i in range(n):
        h = headers[i]
        p = poses[i]
        floats = h["floats"]
        header_pos = np.asarray(floats[1:4], dtype=np.float64)
        header_quat_xyzw = np.asarray(floats[4:8], dtype=np.float64)
        pose_quat_xyzw = np.asarray([p["quat"][1], p["quat"][2], p["quat"][3], p["quat"][0]], dtype=np.float64)
        pos_err.append(float(np.linalg.norm(header_pos - p["pos"])))
        quat_err.append(float(np.linalg.norm(header_quat_xyzw - pose_quat_xyzw)))
        if int(h["uints"][8]) == int(p["frame_id"]):
            section_id_match += 1
    return {
        "sample_count": n,
        "header_pos_float1_3_vs_img_pos_error": summarize_numeric(np.asarray(pos_err)),
        "header_quat_float4_7_xyzw_vs_img_pos_error": summarize_numeric(np.asarray(quat_err)),
        "uint8_matches_frame_id_count": int(section_id_match),
        "uint8_matches_frame_id_ratio": section_id_match / max(n, 1),
    }


def marker_stats(path: Path, headers: list[dict[str, Any]], sections: list[int]) -> list[dict[str, Any]]:
    rows = []
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("marker", "<u4")])
    with path.open("rb") as f:
        by_section = {row["section"]: row for row in headers}
        for section in sections:
            row = by_section.get(section)
            if row is None:
                continue
            f.seek(row["data_offset"])
            data = np.frombuffer(f.read(row["count"] * LX_POINT_SIZE), dtype=dtype)
            markers = data["marker"].astype(np.uint32)
            top = Counter(int(x) for x in markers[: min(len(markers), 100000)].tolist()).most_common(8)
            rows.append({
                "section": int(section),
                "point_count": int(len(data)),
                "marker_min": int(markers.min()) if len(markers) else None,
                "marker_max": int(markers.max()) if len(markers) else None,
                "marker_unique_sample": int(len(set(markers[: min(len(markers), 100000)].tolist()))),
                "marker_top_sample": top,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sections", type=int, default=0)
    parser.add_argument("--marker-sections", type=int, nargs="*", default=[0, 1, 100, 1000, 3400, 5000, 6180])
    parser.add_argument("--video-pts-head", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    headers = read_lx_headers(args.lx_file, args.max_sections)
    if not headers:
        raise SystemExit(f"No .lx sections parsed: {args.lx_file}")
    poses = config.load_img_pos(0, None)
    pose_ts = np.asarray([p["timestamp"] for p in poses], dtype=np.float64)
    pose_dt = np.diff(pose_ts) if len(pose_ts) > 1 else np.empty(0)
    cam_info_counts = Counter()
    for p in poses:
        for cam_id, info in p["cam_info"].items():
            cam_info_counts[f"cam{cam_id}:{info[0]},{info[1]}"] += 1

    report = {
        "lx_file": str(args.lx_file),
        "img_pos_file": config.IMG_POS_FILE,
        "video_dir": config.VIDEO_DIR,
        "section_count": len(headers),
        "img_pos_count": len(poses),
        "first_sections": headers[:8],
        "lx_count_summary": summarize_numeric(np.asarray([row["count"] for row in headers], dtype=np.float64)),
        "img_pos_timestamp": {
            "first": float(pose_ts[0]) if len(pose_ts) else None,
            "last": float(pose_ts[-1]) if len(pose_ts) else None,
            "span": float(pose_ts[-1] - pose_ts[0]) if len(pose_ts) else None,
            "dt_summary": summarize_numeric(pose_dt),
            "dt_gt_0_15_count": int(np.count_nonzero(pose_dt > 0.15)),
            "dt_gt_0_5_count": int(np.count_nonzero(pose_dt > 0.5)),
        },
        "cam_info_top": cam_info_counts.most_common(20),
        "pose_match_summary": pose_match_summary(headers, poses),
        "correlation_summary": correlation_summary(headers, poses),
        "video_summary": {
            str(cam_id): video_summary(path, args.video_pts_head)
            for cam_id, path in config.VIDEO_FILES.items()
        },
        "marker_stats": marker_stats(args.lx_file, headers, args.marker_sections),
    }
    (args.output_dir / "timing_sources_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "lx_headers_sample.jsonl").open("w", encoding="utf-8") as f:
        for row in headers[: min(len(headers), 200)]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "section_count": report["section_count"],
        "img_pos_count": report["img_pos_count"],
        "img_pos_span": report["img_pos_timestamp"]["span"],
        "pose_match": report["pose_match_summary"],
        "video_frames": {cam: item["frame_count"] for cam, item in report["video_summary"].items()},
        "dt_gt_0_15_count": report["img_pos_timestamp"]["dt_gt_0_15_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
