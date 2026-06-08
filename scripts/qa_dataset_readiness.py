#!/usr/bin/env python3
"""Report readiness of the 0-999 new_route semantic dataset caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sky_exists(mask_dir: Path, cam_id: int, frame_id: int) -> bool:
    candidates = [
        mask_dir / f"cam{cam_id}_{frame_id:07d}_sky.png",
        mask_dir / f"cam{cam_id}_{frame_id:06d}_sky.png",
        mask_dir / f"cam{cam_id}_{frame_id:05d}_sky.png",
        mask_dir / f"cam{cam_id}_{frame_id:04d}_sky.png",
    ]
    return any(p.exists() for p in candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, default=Path("/root/epfs/new_route_stage1_skymask/frames"))
    parser.add_argument("--color-dir", type=Path, default=Path("/root/epfs/new_route_stage1_skymask/output"))
    parser.add_argument("--sky-mask-dir", type=Path, default=Path("/root/epfs/new_route_data/sky_masks_color"))
    parser.add_argument("--sam-masks-dir", type=Path, default=Path("/root/epfs/manifold_3dgs_project/processed/sam_masks"))
    parser.add_argument("--semantic-eval-dir", type=Path, default=Path("/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999"))
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=999)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frame_ids = list(range(args.start, args.end + 1))
    total_frames = len(frame_ids)
    total_images = total_frames * len(args.cams)

    complete_camera_frames = 0
    color_ply = 0
    sky = 0
    sam = 0
    completion = 0
    missing = {"camera_frame": [], "color_ply": [], "sky_mask": [], "sam2_mask": [], "completion": []}

    for frame_id in frame_ids:
        if all((args.frames_dir / f"cam{cam_id}" / f"frame_{frame_id:04d}.png").exists() for cam_id in args.cams):
            complete_camera_frames += 1
        elif len(missing["camera_frame"]) < 20:
            missing["camera_frame"].append(frame_id)
        if (args.color_dir / f"frame_{frame_id:04d}.ply").exists():
            color_ply += 1
        elif len(missing["color_ply"]) < 20:
            missing["color_ply"].append(frame_id)
        for cam_id in args.cams:
            image_id = f"cam{cam_id}_{frame_id:06d}"
            if sky_exists(args.sky_mask_dir, cam_id, frame_id):
                sky += 1
            elif len(missing["sky_mask"]) < 20:
                missing["sky_mask"].append(image_id)
            if (args.sam_masks_dir / f"{image_id}_sam_masks.json").exists():
                sam += 1
            elif len(missing["sam2_mask"]) < 20:
                missing["sam2_mask"].append(image_id)
            sem = args.semantic_eval_dir / "images" / image_id / args.combo / "semantic.png"
            if sem.exists():
                completion += 1
            elif len(missing["completion"]) < 20:
                missing["completion"].append(image_id)

    report = {
        "range": {"start": args.start, "end": args.end, "frames": total_frames, "images": total_images},
        "counts": {
            "complete_camera_frames": complete_camera_frames,
            "color_ply": color_ply,
            "sky_masks": sky,
            "sam2_masks": sam,
            "completion_semantic_images": completion,
        },
        "ratios": {
            "complete_camera_frames": complete_camera_frames / max(total_frames, 1),
            "color_ply": color_ply / max(total_frames, 1),
            "sky_masks": sky / max(total_images, 1),
            "sam2_masks": sam / max(total_images, 1),
            "completion_semantic_images": completion / max(total_images, 1),
        },
        "missing_samples": missing,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
