#!/usr/bin/env python3
"""Compare Sonata PCA evidence across a known risky patch edge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def read_ascii_ply(path: Path) -> tuple[list[str], np.ndarray]:
    props: list[str] = []
    vertex_count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("element vertex "):
                vertex_count = int(s.split()[-1])
            elif s.startswith("property "):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
        rows = np.loadtxt(fh, dtype=np.float32, max_rows=vertex_count)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return props, rows


def patch_stats(name: str, rgb: np.ndarray, dist: np.ndarray) -> dict:
    if len(rgb) == 0:
        return {"patch_id": name, "matched_points": 0}
    return {
        "patch_id": name,
        "matched_points": int(len(rgb)),
        "mean_sonata_rgb": rgb.mean(axis=0).round(4).tolist(),
        "std_sonata_rgb": rgb.std(axis=0).round(4).tolist(),
        "mean_match_distance": float(dist.mean()),
        "p90_match_distance": float(np.percentile(dist, 90)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-ply", type=Path, required=True)
    parser.add_argument("--sonata-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-a", type=int, default=70503)
    parser.add_argument("--patch-b", type=int, default=9366)
    parser.add_argument("--max-distance", type=float, default=0.03)
    args = parser.parse_args()

    risk_props, risk_rows = read_ascii_ply(args.risk_ply)
    sonata_props, sonata_rows = read_ascii_ply(args.sonata_ply)
    r = {name: i for i, name in enumerate(risk_props)}
    s = {name: i for i, name in enumerate(sonata_props)}
    risk_xyz = risk_rows[:, [r["x"], r["y"], r["z"]]]
    risk_obj = risk_rows[:, r["object"]].astype(np.int64)
    sonata_xyz = sonata_rows[:, [s["x"], s["y"], s["z"]]]
    sonata_rgb = sonata_rows[:, [s["red"], s["green"], s["blue"]]] / 255.0

    nn = NearestNeighbors(n_neighbors=1).fit(sonata_xyz)
    dist, idx = nn.kneighbors(risk_xyz)
    dist = dist[:, 0]
    idx = idx[:, 0]
    valid = dist <= args.max_distance

    out: dict = {
        "schema": "sonata-patch-edge-analysis/v1",
        "risk_ply": str(args.risk_ply),
        "sonata_ply": str(args.sonata_ply),
        "patch_pair": [args.patch_a, args.patch_b],
        "max_distance": args.max_distance,
        "risk_points": int(len(risk_rows)),
        "matched_points": int(valid.sum()),
        "matched_ratio": float(valid.mean()),
        "patch_stats": {},
    }
    patch_rgbs: dict[int, np.ndarray] = {}
    for patch_id in (args.patch_a, args.patch_b):
        mask = valid & (risk_obj == patch_id)
        rgb = sonata_rgb[idx[mask]]
        patch_rgbs[patch_id] = rgb
        out["patch_stats"][str(patch_id)] = patch_stats(str(patch_id), rgb, dist[mask])
    a_rgb = patch_rgbs[args.patch_a]
    b_rgb = patch_rgbs[args.patch_b]
    if len(a_rgb) and len(b_rgb):
        mean_a = a_rgb.mean(axis=0)
        mean_b = b_rgb.mean(axis=0)
        pooled_std = np.sqrt((a_rgb.var(axis=0).mean() + b_rgb.var(axis=0).mean()) / 2)
        out["edge_evidence"] = {
            "mean_sonata_rgb_distance": float(np.linalg.norm(mean_a - mean_b)),
            "pooled_channel_std": float(pooled_std),
            "distance_over_pooled_std": float(np.linalg.norm(mean_a - mean_b) / max(pooled_std, 1e-6)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(args.output), "matched_ratio": out["matched_ratio"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
