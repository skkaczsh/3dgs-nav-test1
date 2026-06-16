#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as ops
from PIL import Image


def add_project_paths(project_dir: Path) -> None:
    for extra in [project_dir / "GroundingDINO", project_dir / "segment-anything-2"]:
        text = str(extra)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_and_transform_image(image_path: Path):
    import groundingdino.datasets.transforms as T

    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor


def box_cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
    boxes_abs = boxes * scale
    cx, cy, bw, bh = boxes_abs.unbind(-1)
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def sanitize_phrase(text: str) -> str:
    return " ".join(text.replace(".", " ").split())


def overlay_masks(image_rgb: np.ndarray, masks, labels, alpha: float = 0.35) -> np.ndarray:
    vis = image_rgb.copy()
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]
    for idx, (mask, label) in enumerate(zip(masks, labels)):
        color = np.array(colors[idx % len(colors)], dtype=np.uint8)
        vis[mask] = ((1 - alpha) * vis[mask] + alpha * color).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs) and len(ys):
            cv2.putText(
                vis,
                label,
                (int(xs.min()), max(20, int(ys.min()) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                tuple(int(c) for c in color.tolist()),
                1,
                cv2.LINE_AA,
            )
    return vis


def draw_boxes(image_rgb: np.ndarray, boxes_xyxy: np.ndarray, labels) -> np.ndarray:
    vis = image_rgb.copy()
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]
    for idx, (box, label) in enumerate(zip(boxes_xyxy, labels)):
        color = colors[idx % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--grounding-device", default="cpu")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--max-boxes", type=int, default=12)
    parser.add_argument("--max-boxes-per-group", type=int, default=4)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    project_dir = Path(args.project_dir)
    work_dir = Path(args.work_dir)
    add_project_paths(project_dir)

    from groundingdino.util.inference import load_model as load_grounding_model
    from groundingdino.util.inference import predict as grounding_predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    manifest = json.loads(manifest_path.read_text())
    default_prompt_text = manifest["prompt_text"]
    work_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = work_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    config_path = project_dir / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_ckpt = project_dir / "weights/groundingdino_swint_ogc.pth"
    sam2_ckpt = project_dir / "weights/sam2_hiera_large.pt"

    grounding_device = args.grounding_device
    sam_device = args.sam_device if torch.cuda.is_available() else "cpu"

    grounding_model = load_grounding_model(
        str(config_path),
        str(grounding_ckpt),
        device=grounding_device,
    )
    sam2_model = build_sam2("sam2_hiera_l.yaml", str(sam2_ckpt), device=sam_device)
    predictor = SAM2ImagePredictor(sam2_model)

    aggregate = []
    selected_sample_ids = set(args.sample_ids)
    samples = manifest["samples"]
    if selected_sample_ids:
        samples = [sample for sample in samples if sample["id"] in selected_sample_ids]

    for sample in samples:
        sample_id = sample["id"]
        sample_dir = per_sample_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        prompt_terms = sample.get("prompt_terms") or manifest.get("prompt_terms") or []
        prompt_groups = sample.get("prompt_groups") or [{"focus": "all", "terms": prompt_terms}]
        prompt_text = " . ".join(prompt_terms) if prompt_terms else default_prompt_text
        staged_candidates = [
            work_dir / "staged_samples" / sample_id / "image.png",
            work_dir.parent / "staged_samples" / sample_id / "image.png",
        ]
        image_path = next((p for p in staged_candidates if p.exists()), Path(sample["image"]))
        image_pil, image_tensor = load_and_transform_image(image_path)
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]

        group_boxes_xyxy = []
        group_logits = []
        group_phrases = []
        group_focuses = []
        for prompt_group in prompt_groups:
            group_terms = prompt_group.get("terms") or []
            group_focus = prompt_group.get("focus", "all")
            group_prompt_text = " . ".join(group_terms) if group_terms else prompt_text
            boxes, logits, phrases = grounding_predict(
                model=grounding_model,
                image=image_tensor,
                caption=group_prompt_text,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=grounding_device,
            )
            phrases = [sanitize_phrase(p) for p in phrases]
            boxes_xyxy = box_cxcywh_to_xyxy(boxes.cpu(), w, h)
            if len(boxes_xyxy):
                keep = ops.nms(boxes_xyxy, logits.cpu(), args.nms_threshold)
                keep = keep[: args.max_boxes_per_group]
                boxes_xyxy = boxes_xyxy[keep]
                logits = logits[keep]
                phrases = [phrases[i] for i in keep.tolist()]
                group_boxes_xyxy.append(boxes_xyxy)
                group_logits.append(logits.cpu())
                group_phrases.extend(phrases)
                group_focuses.extend([group_focus] * len(phrases))

        if group_boxes_xyxy:
            boxes_xyxy = torch.cat(group_boxes_xyxy, dim=0)
            logits = torch.cat(group_logits, dim=0)
            keep = ops.nms(boxes_xyxy, logits, args.nms_threshold)
            keep = keep[: args.max_boxes]
            boxes_xyxy = boxes_xyxy[keep]
            logits = logits[keep]
            phrases = [group_phrases[i] for i in keep.tolist()]
            detection_focuses = [group_focuses[i] for i in keep.tolist()]
        else:
            boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            logits = torch.zeros((0,), dtype=torch.float32)
            phrases = []
            detection_focuses = []

        predictor.set_image(image_np)
        masks = []
        sam_scores = []
        if len(boxes_xyxy):
            for box in boxes_xyxy.numpy():
                box = np.array(
                    [
                        max(0, min(w - 1, box[0])),
                        max(0, min(h - 1, box[1])),
                        max(0, min(w - 1, box[2])),
                        max(0, min(h - 1, box[3])),
                    ],
                    dtype=np.float32,
                )
                pred_masks, scores, _ = predictor.predict(box=box, multimask_output=False)
                masks.append(pred_masks[0].astype(bool))
                sam_scores.append(float(scores[0]))

        det_labels = [
            f"{focus}:{phrase} | gd={float(logit):.2f}"
            for focus, phrase, logit in zip(detection_focuses, phrases, logits)
        ]
        mask_labels = [
            f"{focus}:{phrase} | sam={score:.2f}"
            for focus, phrase, score in zip(detection_focuses, phrases, sam_scores)
        ]
        boxes_vis = draw_boxes(image_np, boxes_xyxy.numpy() if len(boxes_xyxy) else np.zeros((0, 4)), det_labels)
        masks_vis = overlay_masks(image_np, masks, mask_labels)

        cv2.imwrite(str(sample_dir / "original.png"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / "detector_boxes.png"), cv2.cvtColor(boxes_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / "sam2_masks.png"), cv2.cvtColor(masks_vis, cv2.COLOR_RGB2BGR))

        result = {
            "id": sample_id,
            "rel": sample["rel"],
            "focus": sample["focus"],
            "prompt_terms": prompt_terms,
            "prompt_groups": prompt_groups,
            "prompt_text": prompt_text,
            "label_counts": sample["label_counts"],
            "image_path": sample["image"],
            "num_boxes": int(len(boxes_xyxy)),
            "detections": [
                {
                    "focus": focus,
                    "phrase": phrase,
                    "grounding_score": float(logit),
                    "sam_score": float(sam_score),
                    "box_xyxy": [float(x) for x in box.tolist()],
                    "mask_area": int(mask.sum()),
                }
                for focus, phrase, logit, sam_score, box, mask in zip(
                    detection_focuses,
                    phrases,
                    logits.tolist(),
                    sam_scores,
                    boxes_xyxy,
                    masks,
                )
            ],
        }
        (sample_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        aggregate.append(result)

    summary = {
        "grounding_device": grounding_device,
        "sam_device": sam_device,
        "prompt_text": prompt_text,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "nms_threshold": args.nms_threshold,
        "max_boxes": args.max_boxes,
        "max_boxes_per_group": args.max_boxes_per_group,
        "sample_ids": sorted(selected_sample_ids),
        "samples": aggregate,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
