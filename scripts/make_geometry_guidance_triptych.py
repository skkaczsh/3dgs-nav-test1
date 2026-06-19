#!/usr/bin/env python3
"""Create original/depth/projected-point overlay triptychs from guidance maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    pad_h = 34
    pad = np.zeros((pad_h, out.shape[1], 3), dtype=np.uint8)
    cv2.putText(pad, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack([pad, out])


def fit_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    return cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), height), interpolation=cv2.INTER_AREA)


def overlay_points(original: np.ndarray, rendered_rgb: np.ndarray, alpha: float) -> np.ndarray:
    valid = np.any(rendered_rgb > 0, axis=2)
    blended = original.copy()
    if np.any(valid):
        mixed = cv2.addWeighted(original, 1.0 - alpha, rendered_rgb, alpha, 0.0)
        blended[valid] = mixed[valid]
    return blended


def overlay_depth_points(original: np.ndarray, npz_path: Path | None, fallback_depth_viz: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    if npz_path is not None and npz_path.exists():
        data = np.load(npz_path)
        valid = data["valid"] > 0
        depth = data["depth"].astype(np.float32)
        if np.any(valid):
            vals = depth[valid]
            lo = float(np.percentile(vals, 2))
            hi = float(np.percentile(vals, 98))
            if hi <= lo:
                hi = lo + 1.0
            norm = np.zeros(depth.shape, dtype=np.uint8)
            norm[valid] = (255.0 * (1.0 - np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0))).astype(np.uint8)
            color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            blended = original.copy()
            mixed = cv2.addWeighted(original, 1.0 - alpha, color, alpha, 0.0)
            blended[valid] = mixed[valid]
            return blended, int(np.count_nonzero(valid))
    valid = np.any(fallback_depth_viz > 0, axis=2)
    blended = original.copy()
    if np.any(valid):
        mixed = cv2.addWeighted(original, 1.0 - alpha, fallback_depth_viz, alpha, 0.0)
        blended[valid] = mixed[valid]
    return blended, int(np.count_nonzero(valid))


def make_triptych(
    original_path: Path,
    depth_path: Path,
    rendered_path: Path,
    output_path: Path,
    alpha: float,
    npz_path: Path | None = None,
) -> dict:
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_COLOR)
    rendered = cv2.imread(str(rendered_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(original_path)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if rendered is None:
        raise FileNotFoundError(rendered_path)
    if depth.shape[:2] != original.shape[:2]:
        depth = cv2.resize(depth, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    if rendered.shape[:2] != original.shape[:2]:
        rendered = cv2.resize(rendered, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    rendered_valid = int(np.count_nonzero(np.any(rendered > 0, axis=2)))
    if rendered_valid:
        overlay = overlay_points(original, rendered, alpha)
        overlay_valid = rendered_valid
        overlay_source = "rendered_rgb"
    else:
        overlay, overlay_valid = overlay_depth_points(original, npz_path, depth, alpha)
        overlay_source = "depth_valid"
    panels = [
        label_panel(original, "undistorted image"),
        label_panel(depth, "reverse-rendered depth"),
        label_panel(overlay, f"global point projection overlay ({overlay_source})"),
    ]
    height = max(panel.shape[0] for panel in panels)
    panels = [fit_height(panel, height) for panel in panels]
    triptych = np.hstack(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), triptych)
    return {
        "original_path": str(original_path),
        "depth_path": str(depth_path),
        "rendered_path": str(rendered_path),
        "output_path": str(output_path),
        "valid_render_pixels": rendered_valid,
        "valid_overlay_pixels": overlay_valid,
        "overlay_source": overlay_source,
        "image_pixels": int(original.shape[0] * original.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--max-items", type=int, default=9)
    args = parser.parse_args()

    report = json.loads(args.guidance_report.read_text(encoding="utf-8"))
    rows = [row for row in report.get("items", []) if row.get("status") == "ok"]
    outputs = []
    for row in rows[: max(args.max_items, 0)]:
        frame_id = int(row["frame_id"])
        cam_id = int(row["cam_id"])
        output = args.output_dir / f"cam{cam_id}_{frame_id:06d}_triptych.jpg"
        outputs.append(make_triptych(
            Path(row["image_path"]),
            Path(row["depth_viz_path"]),
            Path(row["rendered_rgb_path"]),
            output,
            args.alpha,
            Path(row["npz_path"]) if row.get("npz_path") else None,
        ))
    summary = {
        "guidance_report": str(args.guidance_report),
        "output_dir": str(args.output_dir),
        "item_count": len(outputs),
        "items": outputs,
    }
    (args.output_dir / "triptych_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
