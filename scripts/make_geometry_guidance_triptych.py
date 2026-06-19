#!/usr/bin/env python3
"""Create original/depth/projected-point overlay triptychs from guidance maps."""

from __future__ import annotations

import argparse
import html
import json
import os
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


def relpath_for_html(path: str, html_path: Path) -> str:
    try:
        return os.path.relpath(path, start=html_path.parent)
    except ValueError:
        return path


def write_review_html(summary: dict, html_path: Path, title: str) -> None:
    rows = []
    for item in summary.get("items", []):
        image_pixels = max(int(item.get("image_pixels") or 1), 1)
        overlay_pixels = int(item.get("valid_overlay_pixels") or 0)
        coverage = overlay_pixels / image_pixels
        src = relpath_for_html(str(item["output_path"]), html_path)
        rows.append(
            "<article class=\"card\">"
            f"<h2>{html.escape(Path(item['output_path']).stem)}</h2>"
            f"<img src=\"{html.escape(src)}\" loading=\"lazy\" />"
            "<dl>"
            f"<dt>overlay source</dt><dd>{html.escape(str(item.get('overlay_source', '')))}</dd>"
            f"<dt>overlay coverage</dt><dd>{coverage:.3f}</dd>"
            f"<dt>valid overlay pixels</dt><dd>{overlay_pixels:,}</dd>"
            "</dl>"
            "</article>"
        )
    body = "\n".join(rows)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0d1117; color: #e6edf3; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 14px 18px; background: rgba(13, 17, 23, 0.94); border-bottom: 1px solid #30363d; }}
h1 {{ margin: 0; font-size: 18px; }}
p {{ color: #9da7b3; margin: 6px 0 0; }}
main {{ padding: 16px; display: grid; gap: 18px; }}
.card {{ border: 1px solid #30363d; border-radius: 8px; background: #151b23; overflow: hidden; }}
.card h2 {{ font-size: 15px; margin: 0; padding: 10px 12px; border-bottom: 1px solid #30363d; background: #1b2430; }}
.card img {{ display: block; width: 100%; height: auto; background: #010409; }}
dl {{ display: grid; grid-template-columns: 160px 1fr; gap: 4px 12px; margin: 0; padding: 10px 12px 14px; font-size: 13px; }}
dt {{ color: #8b949e; }}
dd {{ margin: 0; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p>Diagnostic reverse-depth triptychs. Left: undistorted frame. Middle: full-cloud z-buffer depth. Right: projection overlay.</p>
</header>
<main>
{body}
</main>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--max-items", type=int, default=9)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--title", default="Reverse Depth Triptych Review")
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
    if args.html:
        write_review_html(summary, args.html, args.title)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
