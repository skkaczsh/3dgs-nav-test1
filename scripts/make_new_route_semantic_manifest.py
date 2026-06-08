#!/usr/bin/env python3
"""Build a semantic-eval manifest for new_route camera frames.

The manifest format matches semantic_eval/run_eval.py:
{"items": [{"image_id", "image_path", "sky_mask_path"}]}.
"""

import argparse
import json
from pathlib import Path


def find_mask(mask_dir: Path, cam: int, frame: int) -> Path | None:
    names = [
        f"cam{cam}_{frame:07d}_sky.png",
        f"cam{cam}_{frame:05d}_sky.png",
        f"cam{cam}_{frame:04d}_sky.png",
    ]
    for name in names:
        path = mask_dir / name
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, default=Path("/root/epfs/new_route_stage1_skymask/frames"))
    parser.add_argument("--sky-mask-dir", type=Path, default=Path("/root/epfs/new_route_data/sky_masks_color"))
    parser.add_argument("--output", type=Path, default=Path("/root/epfs/new_route_stage1_skymask/semantic_eval_0000_0500/manifest.json"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--require-sky-mask", action="store_true")
    args = parser.parse_args()

    frames = list(range(args.start, args.end + 1))
    if args.count and args.count < len(frames):
        if args.count == 1:
            frames = [frames[len(frames) // 2]]
        else:
            step = (len(frames) - 1) / (args.count - 1)
            frames = [frames[round(i * step)] for i in range(args.count)]

    items = []
    missing_masks = []
    for frame in frames:
        for cam in args.cams:
            image_path = args.frames_dir / f"cam{cam}" / f"{frame:06d}.png"
            if not image_path.exists():
                image_path = args.frames_dir / f"cam{cam}" / f"frame_{frame:04d}.png"
            if not image_path.exists():
                continue
            mask_path = find_mask(args.sky_mask_dir, cam, frame)
            if mask_path is None:
                missing_masks.append({"cam": cam, "frame": frame})
                if args.require_sky_mask:
                    continue
            items.append({
                "image_id": f"cam{cam}_{frame:06d}",
                "image_path": str(image_path),
                "sky_mask_path": str(mask_path) if mask_path else "",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "range": [args.start, args.end],
        "count_requested": args.count,
        "cams": args.cams,
        "frames_dir": str(args.frames_dir),
        "sky_mask_dir": str(args.sky_mask_dir),
        "items": items,
        "missing_masks": missing_masks,
    }
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"items={len(items)}")
    print(f"missing_masks={len(missing_masks)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
