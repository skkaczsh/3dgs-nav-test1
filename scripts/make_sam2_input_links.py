#!/usr/bin/env python3
"""Create SAM2 input symlinks with semantic-eval image ids.

Camera frame files are stored as frames/camX/frame_YYYY.png, while SAM2 mask
artifacts must be named camX_00YYYY_* to match semantic manifests. This script
creates a flat symlink directory with the correct basenames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, default=Path("/root/epfs/new_route_stage1_skymask/frames"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    missing = []
    skipped = 0
    for frame_id in range(args.start, args.end + 1):
        for cam_id in args.cams:
            src = args.frames_dir / f"cam{cam_id}" / f"frame_{frame_id:04d}.png"
            dst = args.output_dir / f"cam{cam_id}_{frame_id:06d}.png"
            if not src.exists():
                missing.append({"cam": cam_id, "frame": frame_id, "path": str(src)})
                continue
            if dst.exists() or dst.is_symlink():
                if args.overwrite:
                    dst.unlink()
                else:
                    skipped += 1
                    continue
            dst.symlink_to(src)
            linked += 1
    report = {
        "frames_dir": str(args.frames_dir),
        "output_dir": str(args.output_dir),
        "range": {"start": args.start, "end": args.end},
        "cams": args.cams,
        "linked": linked,
        "skipped": skipped,
        "missing": len(missing),
        "missing_samples": missing[:20],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
