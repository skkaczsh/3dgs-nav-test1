#!/usr/bin/env python3
"""Create local Chinese-labeled review panels from semantic eval artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL_ZH = {
    "unknown": "未知",
    "other": "其他",
    "wall": "墙壁",
    "floor": "地面",
    "ceiling": "天花板",
    "grass": "草地",
    "tree": "树木",
    "person": "人",
    "car": "汽车",
    "railing": "栏杆",
    "building": "建筑",
    "sky": "天空",
    "road": "道路",
    "water": "水面",
    "furniture": "家具",
    "pipe": "管道",
    "equipment": "设备",
    "ignore": "忽略",
    "建筑": "建筑",
    "地面": "地面",
    "天空": "天空",
    "栏杆": "栏杆",
    "设备": "设备",
}

SEM_ID_ZH = {
    0: "未知",
    1: "其他",
    2: "墙壁",
    3: "地面",
    4: "天花板",
    5: "草地",
    6: "树木",
    7: "人",
    8: "汽车",
    9: "栏杆",
    10: "建筑",
    11: "天空",
    12: "道路",
    13: "水面",
    14: "家具",
    15: "管道",
    16: "设备",
    255: "忽略",
}

SEM_COLORS = {
    0: (120, 120, 120),
    1: (180, 160, 120),
    2: (210, 90, 90),
    3: (90, 170, 90),
    4: (160, 120, 210),
    5: (80, 190, 70),
    6: (40, 135, 65),
    7: (230, 140, 60),
    8: (230, 210, 70),
    9: (75, 170, 205),
    10: (120, 120, 220),
    11: (75, 145, 245),
    12: (70, 70, 70),
    13: (40, 170, 210),
    14: (210, 120, 170),
    15: (170, 100, 70),
    16: (210, 170, 65),
    255: (30, 30, 30),
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    bg = (0, 0, 0, 180)
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=4,
        fill=bg,
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    up = np.roll(mask, 1, axis=0)
    down = np.roll(mask, -1, axis=0)
    left = np.roll(mask, 1, axis=1)
    right = np.roll(mask, -1, axis=1)
    return mask & ~(up & down & left & right)


def label_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(np.median(xs)), int(np.median(ys))


def resize_panel(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    return image.resize((width, int(image.height * scale)), Image.Resampling.LANCZOS)


def instance_overlay(image: np.ndarray, inst: np.ndarray, labels: dict[str, str]) -> Image.Image:
    rng = np.random.default_rng(20260605)
    overlay = image.copy()
    ids = [int(x) for x in np.unique(inst) if int(x) > 0]
    colors = rng.integers(70, 235, size=(max(len(ids), 1), 3), dtype=np.uint8)
    for idx, mask_id in enumerate(ids):
        mask = inst == mask_id
        color = colors[idx].astype(np.float32)
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.42 + color * 0.58).astype(np.uint8)
        overlay[mask_boundary(mask)] = (255, 255, 255)

    out = Image.fromarray(overlay).convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    font = load_font(28)
    for mask_id in ids:
        mask = inst == mask_id
        centroid = label_centroid(mask)
        if centroid is None:
            continue
        label = LABEL_ZH.get(str(labels.get(str(mask_id), "unknown")), str(labels.get(str(mask_id), "未知")))
        draw_label(draw, centroid, f"{mask_id} {label}", font)
    if not ids:
        draw_label(draw, (32, 32), "无有效非天空 mask", font)
    return out.convert("RGB")


def semantic_overlay(image: np.ndarray, sem: np.ndarray) -> Image.Image:
    color_layer = np.zeros_like(image)
    for label_id, color in SEM_COLORS.items():
        color_layer[sem == label_id] = color
    overlay = (image.astype(np.float32) * 0.42 + color_layer.astype(np.float32) * 0.58).astype(np.uint8)
    for label_id in np.unique(sem):
        mask = sem == label_id
        if int(label_id) != 0:
            overlay[mask_boundary(mask)] = (255, 255, 255)
    out = Image.fromarray(overlay).convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    font = load_font(28)
    for label_id in sorted(int(x) for x in np.unique(sem) if int(x) not in (0, 255)):
        mask = sem == label_id
        if int(mask.sum()) < 3000:
            continue
        centroid = label_centroid(mask)
        if centroid is not None:
            draw_label(draw, centroid, SEM_ID_ZH.get(label_id, str(label_id)), font)
    return out.convert("RGB")


def title_bar(text: str, width: int, height: int = 54) -> Image.Image:
    img = Image.new("RGB", (width, height), (32, 35, 39))
    draw = ImageDraw.Draw(img)
    draw.text((18, 12), text, font=load_font(26), fill=(245, 245, 245))
    return img


def make_panel(sample_dir: Path, out_dir: Path, panel_width: int, combo: str) -> dict[str, Any]:
    combo_dir = sample_dir / combo
    image = np.array(Image.open(combo_dir / "image.png").convert("RGB"))
    inst = np.array(Image.open(combo_dir / "instance.png"))
    sem = np.array(Image.open(combo_dir / "semantic.png"))
    labels = json.loads((combo_dir / "labels.json").read_text())
    summary = json.loads((combo_dir / "summary.json").read_text())

    panels = [
        ("原图", Image.fromarray(image)),
        ("实例 mask + 中文标签", instance_overlay(image, inst, labels)),
        ("语义 mask + 中文标签", semantic_overlay(image, sem)),
    ]
    rendered = []
    for title, panel in panels:
        body = resize_panel(panel, panel_width)
        tile = Image.new("RGB", (panel_width, body.height + 54), (20, 22, 25))
        tile.paste(title_bar(title, panel_width), (0, 0))
        tile.paste(body, (0, 54))
        rendered.append(tile)

    h = max(p.height for p in rendered)
    result = Image.new("RGB", (panel_width * len(rendered), h), (20, 22, 25))
    for i, panel in enumerate(rendered):
        result.paste(panel, (i * panel_width, 0))

    image_id = sample_dir.name
    out_path = out_dir / f"{image_id}_{combo}_panel.png"
    result.save(out_path)
    return {
        "image_id": image_id,
        "path": str(out_path),
        "mask_count": summary.get("mask_count"),
        "coverage": summary.get("coverage"),
        "sky_mask_ratio": summary.get("sky_mask_ratio"),
        "parse_ok": summary.get("vlm", {}).get("parse_ok"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--combo", default="sky_sam3_qwen")
    args = parser.parse_args()

    raw_dir = args.samples_root / "raw"
    out_dir = args.samples_root / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if not (sample_dir / args.combo).exists():
            continue
        rows.append(make_panel(sample_dir, out_dir, args.panel_width, args.combo))

    index_path = args.samples_root / "sample_index.md"
    lines = ["# Local Semantic Samples", ""]
    for row in rows:
        lines.append(
            f"- `{row['image_id']}`: masks={row['mask_count']}, "
            f"coverage={row['coverage']:.3f}, sky={row['sky_mask_ratio']:.3f}, "
            f"parse={row['parse_ok']}  "
            f"[panel](panels/{Path(row['path']).name})"
        )
    index_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} panels to {out_dir}")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
