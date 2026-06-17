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
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


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


def extract_cam(cam_id: int, ids: list[int], output_dir: Path, quality: int, skip_existing: bool) -> dict:
    video_path = config.VIDEO_FILES[cam_id]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video cam{cam_id}: {video_path}")
    maps = make_maps()
    map1, map2 = maps[cam_id]
    cam_dir = output_dir / f"cam{cam_id}"
    cam_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped = failed = 0
    first_error = None
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    for frame_id in ids:
        out = cam_dir / f"frame_{frame_id:06d}.jpg"
        if skip_existing and out.exists():
            skipped += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ret, frame = cap.read()
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
    print(f"calib={config.CALIB_FILE}")
    print(f"video_dir={config.VIDEO_DIR}")

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(extract_cam, cam_id, ids, args.output_dir, args.quality, args.skip_existing)
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
