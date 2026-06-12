#!/usr/bin/env python3
"""Export SAM2 image encoder and point decoder ONNX candidates.

This is the first step toward a C++/TensorRT SAM2 runner. It intentionally
exports model subgraphs, not the full automatic mask generator; dense AMG
post-processing stays separate so that TensorRT accuracy can be measured
against the existing Python generator before promotion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


DEFAULT_SAM2_ROOT = Path("/root/epfs/vlm_seg_project/segment-anything-2")
DEFAULT_PROJECT_ROOT = Path("/root/epfs/vlm_seg_project")
DEFAULT_CHECKPOINT = Path("/root/epfs/vlm_seg_project/weights/sam2_hiera_large.pt")


class SAM2ImageEncoderExport(torch.nn.Module):
    def __init__(self, model, bb_feat_sizes):
        super().__init__()
        self.model = model
        self.bb_feat_sizes = bb_feat_sizes

    def forward(self, image):
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.bb_feat_sizes[::-1])
        ][::-1]
        return feats[0], feats[1], feats[2]


class SAM2PointDecoderExport(torch.nn.Module):
    def __init__(self, model, multimask_output: bool = True):
        super().__init__()
        self.model = model
        self.multimask_output = multimask_output

    def forward(self, image_embed, high_res_0, high_res_1, point_coords, point_labels):
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        sparse_embeddings = sparse_embeddings.to(image_embed.device)
        dense_embeddings = dense_embeddings.to(image_embed.device)
        high_res_0 = high_res_0.to(image_embed.device)
        high_res_1 = high_res_1.to(image_embed.device)
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe().to(image_embed.device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=True,
            high_res_features=[high_res_0, high_res_1],
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        return low_res_masks, iou_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default="sam2_hiera_l.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("/root/epfs/sam2_tensorrt/onnx"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--export", choices=["encoder", "decoder", "both"], default="both")
    parser.add_argument("--dynamic-decoder-batch", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    sys.path.insert(0, str(args.sam2_root))
    sys.path.insert(0, str(args.project_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(args.config, str(args.checkpoint), device=args.device)
    model = model.to(args.device)
    model.eval()
    predictor = SAM2ImagePredictor(model)
    return model, predictor


def export_encoder(args: argparse.Namespace, model, predictor) -> Path:
    wrapper = SAM2ImageEncoderExport(model, predictor._bb_feat_sizes).to(args.device).eval()
    image = torch.randn(1, 3, model.image_size, model.image_size, device=args.device)
    path = args.output_dir / "sam2_hiera_l_image_encoder.onnx"
    torch.onnx.export(
        wrapper,
        (image,),
        str(path),
        input_names=["image"],
        output_names=["high_res_0", "high_res_1", "image_embed"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    return path


def export_decoder(args: argparse.Namespace, model) -> Path:
    wrapper = SAM2PointDecoderExport(model).to(args.device).eval()
    b = args.batch_size
    image_embed = torch.randn(1, 256, 64, 64, device=args.device)
    high_res_0 = torch.randn(1, 32, 256, 256, device=args.device)
    high_res_1 = torch.randn(1, 64, 128, 128, device=args.device)
    point_coords = torch.rand(b, 1, 2, device=args.device) * float(model.image_size)
    point_labels = torch.ones(b, 1, dtype=torch.int64, device=args.device)
    path = args.output_dir / f"sam2_hiera_l_point_decoder_b{b}.onnx"
    with torch.inference_mode():
        wrapper(image_embed, high_res_0, high_res_1, point_coords, point_labels)
    dynamic_axes = None
    if args.dynamic_decoder_batch:
        dynamic_axes = {
            "point_coords": {0: "point_batch"},
            "point_labels": {0: "point_batch"},
            "low_res_masks": {0: "point_batch"},
            "iou_predictions": {0: "point_batch"},
        }
    torch.onnx.export(
        wrapper,
        (image_embed, high_res_0, high_res_1, point_coords, point_labels),
        str(path),
        input_names=["image_embed", "high_res_0", "high_res_1", "point_coords", "point_labels"],
        output_names=["low_res_masks", "iou_predictions"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, predictor = load_model(args)
    written = []
    with torch.inference_mode():
        if args.export in {"encoder", "both"}:
            written.append(export_encoder(args, model, predictor))
        if args.export in {"decoder", "both"}:
            written.append(export_decoder(args, model))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
