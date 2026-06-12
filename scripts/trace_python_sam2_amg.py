#!/usr/bin/env python3
"""Trace official Python SAM2 AMG stages for C++ TensorRT parity work.

This is a diagnostic script. It runs the official SAM2AutomaticMaskGenerator
path and records stage counts using the same images as the C++ TensorRT trace.
It intentionally does not write production mask artifacts.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_SAM2_ROOT = Path("/root/epfs/vlm_seg_project/segment-anything-2")
DEFAULT_CHECKPOINT = Path("/root/epfs/vlm_seg_project/weights/sam2_hiera_large.pt")
DEFAULT_CONFIG = "sam2_hiera_l.yaml"


def project_overlap_count(anns: list[dict[str, Any]], shape: tuple[int, int], min_area: int) -> int:
    """Apply this project's post-SAM2 overlap resolution and return mask count."""
    h, w = shape
    owner = np.full((h, w), -1, dtype=np.int32)
    score_map = np.full((h, w), -1.0, dtype=np.float32)
    for i, ann in enumerate(anns):
        mask = np.asarray(ann["segmentation"], dtype=bool)
        score = float(ann.get("predicted_iou", 0.5)) * float(ann.get("stability_score", 0.5))
        better = mask & (score > score_map)
        owner[better] = i
        score_map[better] = score
    kept = 0
    for i in range(len(anns)):
        if int((owner == i).sum()) >= min_area:
            kept += 1
    return kept


def load_traced_generator(args: argparse.Namespace):
    sys.path.insert(0, str(args.sam2_root))
    sys.path.insert(0, str(args.sam2_root.parent))

    import torch
    from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.utils.amg import (
        MaskData,
        batch_iterator,
        batched_mask_to_box,
        generate_crop_boxes,
        is_box_near_crop_edge,
        mask_to_rle_pytorch,
        uncrop_boxes_xyxy,
        uncrop_masks,
        uncrop_points,
        calculate_stability_score,
    )

    class TracedSAM2AutomaticMaskGenerator(SAM2AutomaticMaskGenerator):
        def reset_trace(self) -> None:
            self.trace: dict[str, Any] = {
                "crop_boxes": 0,
                "totals": {
                    "raw_predictions": 0,
                    "after_pred_iou": 0,
                    "after_stability": 0,
                    "dropped_near_crop_edge": 0,
                    "after_crop_edge_filter": 0,
                    "before_within_crop_nms": 0,
                    "after_within_crop_nms": 0,
                    "before_cross_crop_nms": 0,
                    "after_cross_crop_nms": 0,
                    "official_output_masks": 0,
                    "after_project_overlap_resolution": 0,
                },
                "crops": [],
            }
            self._active_crop_trace: dict[str, Any] | None = None

        def generate_with_trace(self, image: np.ndarray, image_name: str, min_area: int) -> dict[str, Any]:
            self.reset_trace()
            anns = self.generate(image)
            self.trace["image_name"] = image_name
            self.trace["totals"]["official_output_masks"] = len(anns)
            self.trace["totals"]["after_project_overlap_resolution"] = project_overlap_count(
                anns, image.shape[:2], min_area
            )
            return self.trace

        def _generate_masks(self, image: np.ndarray):  # noqa: ANN001
            orig_size = image.shape[:2]
            crop_boxes, layer_idxs = generate_crop_boxes(
                orig_size, self.crop_n_layers, self.crop_overlap_ratio
            )
            self.trace["crop_boxes"] = len(crop_boxes)

            data = MaskData()
            for crop_index, (crop_box, layer_idx) in enumerate(zip(crop_boxes, layer_idxs)):
                crop_trace = {
                    "crop_index": crop_index,
                    "layer_idx": int(layer_idx),
                    "box": [int(x) for x in crop_box],
                    "raw_predictions": 0,
                    "after_pred_iou": 0,
                    "after_stability": 0,
                    "dropped_near_crop_edge": 0,
                    "after_crop_edge_filter": 0,
                    "before_within_crop_nms": 0,
                    "after_within_crop_nms": 0,
                }
                self._active_crop_trace = crop_trace
                crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
                self._active_crop_trace = None
                self.trace["crops"].append(crop_trace)
                data.cat(crop_data)
                del crop_data

            self.trace["totals"]["before_cross_crop_nms"] = len(data["boxes"]) if "boxes" in dict(data.items()) else 0
            if len(crop_boxes) > 1 and len(data["boxes"]):
                scores = 1 / box_area(data["crop_boxes"])
                scores = scores.to(data["boxes"].device)
                keep_by_nms = batched_nms(
                    data["boxes"].float(),
                    scores,
                    torch.zeros_like(data["boxes"][:, 0]),
                    iou_threshold=self.crop_nms_thresh,
                )
                data.filter(keep_by_nms)
            self.trace["totals"]["after_cross_crop_nms"] = len(data["boxes"]) if "boxes" in dict(data.items()) else 0
            data.to_numpy()
            return data

        def _process_crop(self, image: np.ndarray, crop_box: list[int], crop_layer_idx: int, orig_size: tuple[int, ...]):
            x0, y0, x1, y1 = crop_box
            cropped_im = image[y0:y1, x0:x1, :]
            cropped_im_size = cropped_im.shape[:2]
            self.predictor.set_image(cropped_im)

            points_scale = np.array(cropped_im_size)[None, ::-1]
            points_for_image = self.point_grids[crop_layer_idx] * points_scale

            data = MaskData()
            for (points,) in batch_iterator(self.points_per_batch, points_for_image):
                batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size, normalize=True)
                data.cat(batch_data)
                del batch_data
            self.predictor.reset_predictor()

            crop_trace = self._active_crop_trace
            if crop_trace is not None:
                crop_trace["before_within_crop_nms"] = len(data["boxes"])
                self.trace["totals"]["before_within_crop_nms"] += len(data["boxes"])

            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)
            if crop_trace is not None:
                crop_trace["after_within_crop_nms"] = len(data["boxes"])
                self.trace["totals"]["after_within_crop_nms"] += len(data["boxes"])

            data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
            data["points"] = uncrop_points(data["points"], crop_box)
            data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
            return data

        def _process_batch(self, points, im_size, crop_box, orig_size, normalize=False):  # noqa: ANN001
            orig_h, orig_w = orig_size
            crop_trace = self._active_crop_trace

            points = torch.as_tensor(points, dtype=torch.float32, device=self.predictor.device)
            in_points = self.predictor._transforms.transform_coords(
                points, normalize=normalize, orig_hw=im_size
            )
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            masks, iou_preds, low_res_masks = self.predictor._predict(
                in_points[:, None, :],
                in_labels[:, None],
                multimask_output=self.multimask_output,
                return_logits=True,
            )
            data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                points=points.repeat_interleave(masks.shape[1], dim=0),
                low_res_masks=low_res_masks.flatten(0, 1),
            )
            del masks
            if crop_trace is not None:
                raw_count = len(data["iou_preds"])
                crop_trace["raw_predictions"] += raw_count
                self.trace["totals"]["raw_predictions"] += raw_count

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)
            if crop_trace is not None:
                count = len(data["iou_preds"])
                crop_trace["after_pred_iou"] += count
                self.trace["totals"]["after_pred_iou"] += count

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
            if crop_trace is not None:
                count = len(data["iou_preds"])
                crop_trace["after_stability"] += count
                self.trace["totals"]["after_stability"] += count

            data["masks"] = data["masks"] > self.mask_threshold
            data["boxes"] = batched_mask_to_box(data["masks"])

            before_edge = len(data["boxes"])
            keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
            if not torch.all(keep_mask):
                data.filter(keep_mask)
            after_edge = len(data["boxes"])
            if crop_trace is not None:
                dropped = before_edge - after_edge
                crop_trace["dropped_near_crop_edge"] += dropped
                crop_trace["after_crop_edge_filter"] += after_edge
                self.trace["totals"]["dropped_near_crop_edge"] += dropped
                self.trace["totals"]["after_crop_edge_filter"] += after_edge

            data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
            data["rles"] = mask_to_rle_pytorch(data["masks"])
            del data["masks"]
            return data

    sam2 = build_sam2(str(args.config), str(args.checkpoint), device=args.device)
    return TracedSAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        min_mask_region_area=args.min_mask_area,
        output_mode="binary_mask",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--min-mask-area", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = [Path(p) for p in sorted(glob.glob(args.images))]
    if not images:
        raise SystemExit(f"no images matched {args.images}")

    generator = load_traced_generator(args)
    rows = []
    for image_path in images:
        image = np.array(Image.open(image_path).convert("RGB"))
        image_id = image_path.stem
        trace = generator.generate_with_trace(image, image_id, args.min_mask_area)
        out = args.output_dir / f"{image_id}_python_trace.json"
        out.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({
            "image_id": image_id,
            **trace["totals"],
            "crop_boxes": trace["crop_boxes"],
        })
        print(json.dumps({"image": image_id, **trace["totals"]}, ensure_ascii=False))

    summary = {"images": len(rows)}
    if rows:
        for key in [k for k in rows[0] if k != "image_id"]:
            summary[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
    report = {"summary": summary, "rows": rows}
    (args.output_dir / "python_trace_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
