#!/usr/bin/env python3
"""Make per-proposal contact sheets from a cross-candidate review pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def image_path(rep: dict, pack_dir: Path | None = None) -> Path | None:
    for key in ("copied_overlay", "copied_image", "copied_raw_image"):
        raw_path = rep.get(key, "")
        if not raw_path:
            continue
        path = Path(raw_path)
        if path_exists(path):
            return path
        if pack_dir is not None and "assets" in path.parts:
            parts = path.parts
            asset_idx = parts.index("assets")
            remapped = pack_dir.joinpath(*parts[asset_idx:])
            if path_exists(remapped):
                return remapped
    return None


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return canvas


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def make_sheet(item: dict, output_dir: Path, tile_w: int, tile_h: int, pack_dir: Path | None = None) -> Path:
    reps = item.get("representatives", [])
    cols = max(1, min(4, len(reps)))
    rows = (len(reps) + cols - 1) // cols
    header_h = 96
    label_h = 52
    sheet = Image.new("RGB", (cols * tile_w, header_h + rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    p = item["proposal"]
    title = (
        f"{item['review_id']}  {p['object_a']} + {p['object_b']}  "
        f"score={p['score']:.3f}  candidates={p['candidate_a']}/{p['candidate_b']}"
    )
    draw_text(draw, (16, 12), title)
    draw_text(
        draw,
        (16, 44),
        "Decision options: merge / keep_split / uncertain. Rooftop scene; large ground is expected.",
    )
    for idx, rep in enumerate(reps):
        col = idx % cols
        row = idx // cols
        x = col * tile_w
        y = header_h + row * (tile_h + label_h)
        path = image_path(rep, pack_dir)
        if path is not None:
            sheet.paste(fit_image(path, tile_w, tile_h), (x, y))
        label = (
            f"{rep['side']}{rep['rep_index']} {rep['tracklet_id']} "
            f"frame={rep.get('target_meta', {}).get('frame', '')} "
            f"cam={rep.get('target_meta', {}).get('cam', '')}"
        )
        draw.rectangle([x, y + tile_h, x + tile_w, y + tile_h + label_h], fill=(245, 245, 245))
        draw_text(draw, (x + 8, y + tile_h + 8), label)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{item['review_id']}_contact_sheet.jpg"
    sheet.save(out, quality=92)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=360)
    args = parser.parse_args()

    items = load_jsonl(args.review_jsonl)
    outputs = [
        str(make_sheet(item, args.output_dir, args.tile_width, args.tile_height, args.review_jsonl.parent))
        for item in items
    ]
    report = {"item_count": len(items), "sheet_count": len(outputs), "outputs": outputs}
    (args.output_dir / "contact_sheet_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
