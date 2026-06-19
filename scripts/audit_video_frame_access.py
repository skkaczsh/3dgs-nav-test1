#!/usr/bin/env python3
"""Audit whether video frame access methods return the same image.

Random access through OpenCV can be unreliable for some HEVC/MKV files.  The
sync calibration route uses random frame reads heavily, so this script compares:

- OpenCV `CAP_PROP_POS_FRAMES` seek;
- ffmpeg exact frame index via `select=eq(n,index)`;
- ffmpeg timestamp seek with `-ss index/fps`.

If OpenCV and exact ffmpeg disagree, sync candidate scoring must stop using
OpenCV random seek as an authoritative reader.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from calibrate_lx_video_frame_mapping import parse_int_range


def read_opencv_index(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def decode_ffmpeg_mjpeg(cmd: list[str]) -> np.ndarray | None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    data = np.frombuffer(result.stdout, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_ffmpeg_select(video_path: str, frame_idx: int) -> np.ndarray | None:
    # Comma must be escaped for ffmpeg filter parser.
    expr = f"select=eq(n\\,{int(frame_idx)})"
    return decode_ffmpeg_mjpeg([
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        expr,
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ])


def read_ffmpeg_time(video_path: str, frame_idx: int, fps: float) -> np.ndarray | None:
    rel_ts = float(frame_idx) / max(float(fps), 1e-6)
    return decode_ffmpeg_mjpeg([
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{rel_ts:.6f}",
        "-i",
        video_path,
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ])


def image_metrics(a: np.ndarray | None, b: np.ndarray | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"available": False}
    if a.shape != b.shape:
        return {"available": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = cv2.absdiff(a, b)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if float(gray_a.std()) == 0.0 or float(gray_b.std()) == 0.0:
        corr = 1.0 if np.array_equal(gray_a, gray_b) else 0.0
    else:
        corr = float(np.corrcoef(gray_a.reshape(-1).astype(np.float64), gray_b.reshape(-1).astype(np.float64))[0, 1])
    return {
        "available": True,
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "max_abs_diff": int(diff.max()),
        "gray_corr": corr,
    }


def audit_video(video_path: str, cam_id: int, frame_ids: list[int], fps: float) -> list[dict[str, Any]]:
    rows = []
    for frame_idx in frame_ids:
        opencv = read_opencv_index(video_path, frame_idx)
        exact = read_ffmpeg_select(video_path, frame_idx)
        by_time = read_ffmpeg_time(video_path, frame_idx, fps)
        rows.append({
            "cam_id": int(cam_id),
            "video_path": video_path,
            "frame_idx": int(frame_idx),
            "opencv_shape": list(opencv.shape) if opencv is not None else None,
            "ffmpeg_exact_shape": list(exact.shape) if exact is not None else None,
            "ffmpeg_time_shape": list(by_time.shape) if by_time is not None else None,
            "opencv_vs_ffmpeg_exact": image_metrics(opencv, exact),
            "ffmpeg_time_vs_ffmpeg_exact": image_metrics(by_time, exact),
        })
    return rows


def summarize(rows: list[dict[str, Any]], key: str, mad_threshold: float) -> dict[str, Any]:
    metrics = [row[key] for row in rows if row[key].get("available")]
    if not metrics:
        return {"available_count": 0, "pass": False}
    mean_abs = [float(item["mean_abs_diff"]) for item in metrics]
    corr = [float(item["gray_corr"]) for item in metrics]
    return {
        "available_count": len(metrics),
        "mean_abs_diff": {
            "max": float(max(mean_abs)),
            "mean": float(np.mean(mean_abs)),
        },
        "gray_corr": {
            "min": float(min(corr)),
            "mean": float(np.mean(corr)),
        },
        "pass": bool(max(mean_abs) <= float(mad_threshold)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", default="0,1,300,1000,2200,3400,5200,5800,6180")
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--mean-abs-diff-threshold", type=float, default=2.0)
    args = parser.parse_args()

    frame_ids = parse_int_range(args.frames)
    rows = []
    for cam_id in args.cams:
        rows.extend(audit_video(config.VIDEO_FILES[cam_id], cam_id, frame_ids, args.fps))
    report = {
        "video_dir": config.VIDEO_DIR,
        "frames": frame_ids,
        "cams": args.cams,
        "fps": args.fps,
        "thresholds": {"mean_abs_diff": args.mean_abs_diff_threshold},
        "summary": {
            "opencv_vs_ffmpeg_exact": summarize(rows, "opencv_vs_ffmpeg_exact", args.mean_abs_diff_threshold),
            "ffmpeg_time_vs_ffmpeg_exact": summarize(rows, "ffmpeg_time_vs_ffmpeg_exact", args.mean_abs_diff_threshold),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": report["summary"],
        "row_count": len(rows),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
