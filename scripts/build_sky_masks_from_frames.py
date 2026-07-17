#!/usr/bin/env python3
"""Build sky masks from stage1 frame images with the ONNX skyseg model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def predict_sky_mask(session, input_name: str, image_bgr: np.ndarray, threshold: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (320, 320))
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    inp = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    out = session.run(None, {input_name: inp})[0]
    mask = out[0, 0]
    mask_full = cv2.resize(mask, (w, h))
    return (mask_full > threshold).astype(np.uint8) * 255


def frame_path(frames_dir: Path, cam_id: int, frame_id: int) -> Path | None:
    """Find the undistorted frame without assuming a legacy image codec."""
    frame_dir = frames_dir / f"cam{cam_id}"
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = frame_dir / f"frame_{frame_id:06d}{suffix}"
        if candidate.is_file():
            return candidate
    # The earliest extractor used four-digit names. Keep it as a read-only
    # compatibility fallback; all new output remains seven-digit SKYMask.
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = frame_dir / f"frame_{frame_id:04d}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def mask_path(output_dir: Path, cam_id: int, frame_id: int) -> Path:
    return output_dir / f"cam{cam_id}_{frame_id:07d}_sky.png"


def read_frame_ids(path: Path) -> list[int]:
    """Read a JSON array or one frame id per line, preserving no duplicates."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    values = json.loads(text) if text.startswith("[") else [line for line in text.splitlines() if line.strip()]
    return sorted({int(value) for value in values})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/root/epfs/lingbot-map/lingbot-map/skyseg_batch.onnx"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=999)
    parser.add_argument("--frame-ids-file", type=Path, default=None,
                        help="Optional JSON array or newline frame-id list. Limits inference to actual consumer poses.")
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(str(args.model), providers=[args.provider])
    input_name = sess.get_inputs()[0].name

    frame_ids = read_frame_ids(args.frame_ids_file) if args.frame_ids_file else list(range(args.start, args.end + 1))
    rows = []
    for frame_id in frame_ids:
        for cam_id in args.cams:
            src = frame_path(args.frames_dir, cam_id, frame_id)
            dst = mask_path(args.output_dir, cam_id, frame_id)
            if args.skip_existing and dst.exists():
                rows.append({"frame_id": frame_id, "cam_id": cam_id, "status": "skip", "output": str(dst)})
                continue
            if src is None:
                rows.append({"frame_id": frame_id, "cam_id": cam_id, "status": "missing_frame"})
                continue
            img = cv2.imread(str(src))
            if img is None:
                rows.append({"frame_id": frame_id, "cam_id": cam_id, "status": "bad_frame", "frame": str(src)})
                continue
            mask = predict_sky_mask(sess, input_name, img, args.threshold)
            ok = cv2.imwrite(str(dst), mask)
            rows.append({
                "frame_id": frame_id,
                "cam_id": cam_id,
                "status": "ok" if ok else "write_failed",
                "sky_ratio": float((mask > 0).sum() / max(mask.size, 1)),
                "output": str(dst),
            })
        if len(rows) and (len(rows) // max(1, len(args.cams))) % 50 == 0:
            ok_count = sum(1 for r in rows if r["status"] in {"ok", "skip"})
            print(f"frame={frame_id} done={ok_count}/{len(rows)}")

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = {
        "frames_dir": str(args.frames_dir),
        "output_dir": str(args.output_dir),
        "model": str(args.model),
        "start": args.start,
        "end": args.end,
        "frame_ids_file": str(args.frame_ids_file) if args.frame_ids_file else "",
        "requested_frame_count": len(frame_ids),
        "cams": args.cams,
        "status_counts": counts,
        "total": len(rows),
    }
    report_path = args.report or (args.output_dir / f"sky_masks_{args.start:04d}_{args.end:04d}_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
