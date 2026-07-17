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


def read_views(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_views(rows: list[dict]) -> list[tuple[int, int, Path | None]]:
    """Deduplicate arbitrary calibrated views from an evidence ledger.

    Superpoint evidence is sparse and irregular by construction, so forcing it
    through a contiguous frame range either wastes SAM2 work or misses a view.
    The ledger's image_path is authoritative when present; `None` retains the
    legacy frames-dir fallback for old manifests.
    """
    views: dict[tuple[int, int], Path | None] = {}
    for row in rows:
        frame_id, cam_id = int(row["frame_id"]), int(row["cam_id"])
        key = (frame_id, cam_id)
        image_path = Path(str(row["image_path"])) if row.get("image_path") else None
        if key not in views or (views[key] is None and image_path is not None):
            views[key] = image_path
    return [(frame_id, cam_id, views[(frame_id, cam_id)]) for frame_id, cam_id in sorted(views)]


def fallback_frame_path(frames_dir: Path, cam_id: int, frame_id: int) -> Path:
    candidates = (
        frames_dir / f"cam{cam_id}" / f"frame_{frame_id:06d}.jpg",
        frames_dir / f"cam{cam_id}" / f"frame_{frame_id:06d}.png",
        frames_dir / f"cam{cam_id}" / f"frame_{frame_id:04d}.png",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, default=Path("/root/epfs/new_route_stage1_skymask/frames"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--views-jsonl", type=Path,
                        help="Use exact (frame_id, cam_id, image_path) evidence rows instead of a contiguous range.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.views_jsonl is not None and (args.start is not None or args.end is not None):
        raise SystemExit("--views-jsonl is mutually exclusive with --start/--end")
    if args.views_jsonl is None and (args.start is None or args.end is None):
        raise SystemExit("Provide either --views-jsonl or both --start and --end")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    missing = []
    skipped = 0
    if args.views_jsonl is not None:
        selected_views = evidence_views(read_views(args.views_jsonl))
    else:
        selected_views = [
            (frame_id, cam_id, None)
            for frame_id in range(args.start, args.end + 1)
            for cam_id in args.cams
        ]
    for frame_id, cam_id, evidence_path in selected_views:
        src = evidence_path if evidence_path is not None else fallback_frame_path(args.frames_dir, cam_id, frame_id)
        suffix = src.suffix.lower() if src.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".png"
        dst = args.output_dir / f"cam{cam_id}_{frame_id:06d}{suffix}"
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
        "range": {"start": args.start, "end": args.end} if args.views_jsonl is None else None,
        "cams": args.cams,
        "views_jsonl": str(args.views_jsonl) if args.views_jsonl else "",
        "selected_views": len(selected_views),
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
