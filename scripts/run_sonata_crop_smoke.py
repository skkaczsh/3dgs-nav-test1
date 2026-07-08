#!/usr/bin/env python3
"""Run Sonata encoder smoke on one ASCII PLY crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

try:
    import flash_attn  # noqa: F401
except ImportError:
    flash_attn = None

import sonata


def pca_color(feat: torch.Tensor) -> np.ndarray:
    feat = feat.float()
    _, _, v = torch.pca_lowrank(feat, center=True, q=3, niter=5)
    proj = feat @ v[:, :3]
    lo = proj.min(dim=0, keepdim=True).values
    hi = proj.max(dim=0, keepdim=True).values
    rgb = ((proj - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    return rgb.cpu().numpy()


def load_point(path: Path, max_points: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    coord = np.asarray(pcd.points, dtype=np.float32)
    if len(coord) == 0:
        raise ValueError(f"empty point cloud: {path}")
    color = np.asarray(pcd.colors, dtype=np.float32)
    if len(color) != len(coord):
        color = np.zeros_like(coord)
    color = (color * 255.0).clip(0, 255).astype(np.float32)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=24))
    normal = np.asarray(pcd.normals, dtype=np.float32)
    if len(coord) > max_points:
        idx = np.linspace(0, len(coord) - 1, max_points, dtype=np.int64)
        coord, color, normal = coord[idx], color[idx], normal[idx]
    original = coord.copy()
    return {"coord": coord, "color": color, "normal": normal}, original


def upcast_point(point):
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--download-root", type=Path, default=Path.home() / ".cache" / "sonata")
    parser.add_argument("--save-feature-npz", action="store_true")
    args = parser.parse_args()

    sonata.utils.set_seed(53124)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    point, original_coord = load_point(args.input, args.max_points)
    transform = sonata.transform.default()
    point = transform(point)

    custom_config = None
    if flash_attn is None:
        custom_config = {"enc_patch_size": [1024 for _ in range(5)], "enable_flash": False}
    model = sonata.load(
        "sonata",
        repo_id="facebook/sonata",
        custom_config=custom_config,
        download_root=str(args.download_root),
    ).cuda()
    model.eval()

    with torch.inference_mode():
        for key, value in list(point.items()):
            if isinstance(value, torch.Tensor):
                point[key] = value.cuda(non_blocking=True)
        point = model(point)
        point = upcast_point(point)
        feat = point.feat
        colors = pca_color(feat)
        inverse = point.inverse.detach().cpu().numpy()
        original_colors = colors[inverse]
        original_features = feat.detach().cpu().numpy()[inverse].astype(np.float32, copy=False)

    out_ply = args.output_dir / f"{args.input.stem}_sonata_pca.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_coord)
    pcd.colors = o3d.utility.Vector3dVector(original_colors)
    o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=True)
    report = {
        "schema": "sonata-crop-smoke/v1",
        "input": str(args.input),
        "output_ply": str(out_ply),
        "input_points": int(len(original_coord)),
        "model_points": int(feat.shape[0]),
        "output_points": int(len(original_coord)),
        "feature_dim": int(feat.shape[1]),
    }
    if args.save_feature_npz:
        feature_path = args.output_dir / f"{args.input.stem}_sonata_features.npz"
        np.savez_compressed(feature_path, features=original_features)
        report["feature_npz"] = str(feature_path)
    report_path = args.output_dir / f"{args.input.stem}_sonata_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
