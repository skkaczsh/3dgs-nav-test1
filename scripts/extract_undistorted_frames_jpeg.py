#!/usr/bin/env python3
"""Extract undistorted synchronized frames into an external work directory.

The original dataset directory is treated as read-only. Inputs are selected by
the existing `config.py` environment variables:

- SCAN_IMAGE_DIR: contains cam_in_ex.txt and img_pos.txt
- SCAN_VIDEO_DIR: contains video_cam0.mkv, video_cam1.mkv, video_cam2.mkv

Output layout:

  <output>/cam0/frame_000000.jpg
  <output>/cam1/frame_000000.jpg
  <output>/cam2/frame_000000.jpg
  <output>/extract_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load_video_timestamps(video_path: str) -> list[tuple[int, float]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pkt_pts_time",
        "-of",
        "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    frames: list[tuple[int, float]] = []
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            try:
                frames.append((len(frames), float(line.split(",")[0])))
            except (ValueError, IndexError):
                continue
    if frames:
        return frames

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return [(i, i / fps) for i in range(count)]


def find_best_video_frame(target_ts_rel: float, video_ts: np.ndarray, max_delta: float) -> tuple[int | None, float, float]:
    if video_ts.size == 0:
        return None, float("inf"), float("nan")
    frame_idxs = video_ts[:, 0]
    rel_times = video_ts[:, 1]
    idx = int(np.searchsorted(rel_times, target_ts_rel))
    candidates = []
    if idx > 0:
        candidates.append((idx - 1, abs(float(rel_times[idx - 1]) - target_ts_rel)))
    if idx < len(rel_times):
        candidates.append((idx, abs(float(rel_times[idx]) - target_ts_rel)))
    if not candidates:
        return None, float("inf"), float("nan")
    best_rel_idx, best_delta = min(candidates, key=lambda item: item[1])
    rel_ts = float(rel_times[best_rel_idx])
    if best_delta > max_delta:
        return None, float(best_delta), rel_ts
    return int(frame_idxs[best_rel_idx]), float(best_delta), rel_ts


def read_frame_ffmpeg_time(video_path: str, rel_ts: float) -> np.ndarray | None:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{rel_ts:.4f}",
        "-i",
        video_path,
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    data = np.frombuffer(result.stdout, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def make_maps() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    maps = {}
    for cam_id, params in config.CAMERA_PARAMS.items():
        maps[cam_id] = cv2.fisheye.initUndistortRectifyMap(
            params["K"],
            params["D"],
            np.eye(3),
            params["K"],
            (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
            cv2.CV_16SC2,
        )
    return maps


def frame_ids(start: int, end: int, stride: int) -> list[int]:
    return list(range(start, end + 1, stride))


def extract_cam(
    cam_id: int,
    ids: list[int],
    output_dir: Path,
    quality: int,
    skip_existing: bool,
    sync_mode: str,
    time_scale: float,
    max_delta: float,
) -> dict:
    video_path = config.VIDEO_FILES[cam_id]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video cam{cam_id}: {video_path}")
    video_ts = np.asarray(load_video_timestamps(video_path), dtype=np.float64)
    maps = make_maps()
    map1, map2 = maps[cam_id]
    cam_dir = output_dir / f"cam{cam_id}"
    cam_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped = failed = 0
    first_error = None
    deltas: list[float] = []
    frame_map: list[dict[str, int | float | str]] = []
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    sequential = sync_mode == "opencv-index" and bool(ids) and all((b - a) == 1 for a, b in zip(ids, ids[1:]))
    if sequential:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ids[0]))
    for frame_id in ids:
        out = cam_dir / f"frame_{frame_id:06d}.jpg"
        if skip_existing and out.exists():
            skipped += 1
            if sequential:
                cap.grab()
            continue
        if sync_mode == "ffmpeg-time":
            target_ts = float(frame_id) * float(time_scale)
            video_idx, delta, rel_ts = find_best_video_frame(target_ts, video_ts, max_delta)
            if video_idx is None:
                failed += 1
                first_error = first_error or f"sync_failed_frame_{frame_id}"
                frame_map.append({
                    "frame_id": int(frame_id),
                    "status": "sync_failed",
                    "target_rel_ts": target_ts,
                    "video_rel_ts": rel_ts,
                    "delta": delta,
                })
                continue
            frame = read_frame_ffmpeg_time(video_path, rel_ts)
            ret = frame is not None
            deltas.append(delta)
            frame_map.append({
                "frame_id": int(frame_id),
                "status": "ok" if ret else "read_failed",
                "video_idx": int(video_idx),
                "target_rel_ts": target_ts,
                "video_rel_ts": rel_ts,
                "delta": delta,
            })
        else:
            video_idx = int(frame_id)
            if sequential:
                ret, frame = cap.read()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
                ret, frame = cap.read()
            frame_map.append({
                "frame_id": int(frame_id),
                "status": "ok" if ret else "read_failed",
                "video_idx": video_idx,
                "target_rel_ts": float(video_idx) * float(time_scale),
                "video_rel_ts": float(video_idx) * float(time_scale),
                "delta": 0.0,
            })
        if not ret or frame is None:
            failed += 1
            first_error = first_error or f"read_failed_frame_{frame_id}"
            continue
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        if cv2.imwrite(str(out), undistorted, encode_params):
            ok += 1
        else:
            failed += 1
            first_error = first_error or f"write_failed_frame_{frame_id}"
    cap.release()
    return {
        "cam_id": cam_id,
        "video": video_path,
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "first_error": first_error,
        "sync_mode": sync_mode,
        "time_scale": time_scale,
        "max_delta": max_delta,
        "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "delta_max": float(np.max(deltas)) if deltas else 0.0,
        "frame_map": frame_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sync-mode", choices=["ffmpeg-time", "opencv-index"], default="ffmpeg-time")
    parser.add_argument("--time-scale", type=float, default=0.1, help="Seconds per MANIFOLD frame id in ffmpeg-time mode.")
    parser.add_argument("--max-delta", type=float, default=0.15, help="Max allowed timestamp delta in ffmpeg-time mode.")
    args = parser.parse_args()

    if args.end is None:
        poses = config.load_img_pos(args.start, None)
        if not poses:
            raise SystemExit("No img_pos rows found.")
        args.end = int(poses[-1]["frame_id"])
    ids = frame_ids(args.start, args.end, args.stride)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"output_dir={args.output_dir}")
    print(f"frames={len(ids)} range={args.start}..{args.end} stride={args.stride}")
    print(f"sync_mode={args.sync_mode} time_scale={args.time_scale} max_delta={args.max_delta}")
    print(f"calib={config.CALIB_FILE}")
    print(f"video_dir={config.VIDEO_DIR}")

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                extract_cam,
                cam_id,
                ids,
                args.output_dir,
                args.quality,
                args.skip_existing,
                args.sync_mode,
                args.time_scale,
                args.max_delta,
            )
            for cam_id in args.cams
        ]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))

    report = {
        "output_dir": str(args.output_dir),
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "frame_count": len(ids),
        "cams": args.cams,
        "sync_mode": args.sync_mode,
        "time_scale": args.time_scale,
        "max_delta": args.max_delta,
        "quality": args.quality,
        "calib_file": config.CALIB_FILE,
        "video_dir": config.VIDEO_DIR,
        "results": sorted(results, key=lambda r: r["cam_id"]),
        "elapsed_sec": time.time() - t0,
    }
    (args.output_dir / "extract_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
