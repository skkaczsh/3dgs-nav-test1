#!/usr/bin/env python3
"""Create an XY preview PNG for ASCII or binary XYZRGB PLY files."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def scalar_colors(values: np.ndarray, field: str) -> np.ndarray:
    if field == "cluster_status":
        palette = {
            0: (128, 128, 128),
            1: (70, 190, 90),
            2: (255, 210, 40),
            3: (255, 120, 40),
            4: (180, 80, 255),
        }
        return np.array([palette.get(int(v), (220, 220, 220)) for v in values], dtype=np.uint8)
    valid = values >= 0
    colors = np.full((len(values), 3), (90, 90, 90), dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(values[valid].astype(np.float32), 0, 100) / 100.0
        colors[valid, 0] = np.clip(255 * clipped, 0, 255).astype(np.uint8)
        colors[valid, 1] = np.clip(220 * (1.0 - clipped), 0, 255).astype(np.uint8)
        colors[valid, 2] = np.clip(80 * (1.0 - clipped), 0, 255).astype(np.uint8)
    return colors


def read_ply_xyzrgb(path: Path, max_points: int, seed: int, color_field: str | None):
    with path.open("rb") as f:
        fmt = "ascii"
        n = 0
        header_len = 0
        props = []
        prop_types = []
        in_vertex = False
        while True:
            raw = f.readline()
            header_len += 1
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element "):
                in_vertex = False
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                prop_types.append(parts[1])
                props.append(parts[-1])
            elif line == "end_header":
                break

    if fmt == "ascii":
        data = np.loadtxt(path, skiprows=header_len, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        xyz = data[:, :3]
        if color_field:
            if color_field not in props:
                raise ValueError(f"PLY field not found: {color_field}; available={props}")
            rgb = scalar_colors(data[:, props.index(color_field)], color_field)
        elif data.shape[1] >= 6:
            rgb = np.clip(data[:, 3:6], 0, 255).astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), 210, dtype=np.uint8)
    elif fmt == "binary_little_endian":
        type_map = {
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "uchar": "u1",
            "uint8": "u1",
            "char": "i1",
            "int8": "i1",
            "ushort": "<u2",
            "uint16": "<u2",
            "short": "<i2",
            "int16": "<i2",
            "uint": "<u4",
            "uint32": "<u4",
            "int": "<i4",
            "int32": "<i4",
        }
        dtype = np.dtype([
            (name, type_map.get(ptype, "<f4"))
            for ptype, name in zip(prop_types, props)
        ])
        with path.open("rb") as f:
            while f.readline().strip() != b"end_header":
                pass
            arr = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
        xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
        if color_field:
            if color_field not in props:
                raise ValueError(f"PLY field not found: {color_field}; available={props}")
            rgb = scalar_colors(arr[color_field].astype(np.float32), color_field)
        elif all(name in props for name in ("red", "green", "blue")):
            rgb = np.column_stack([arr["red"], arr["green"], arr["blue"]]).astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), 210, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported PLY format: {fmt}")

    if max_points and len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
    return xyz, rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=800000)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-field", default=None,
                        help="Optional ASCII PLY scalar field to color by, e.g. cluster_status or source_abs_offset.")
    args = parser.parse_args()

    xyz, rgb = read_ply_xyzrgb(args.ply, args.max_points, args.seed, args.color_field)
    out = args.output or args.ply.with_suffix(".xy_preview.png")
    w, h, pad = args.width, args.height, 35
    xy = xyz[:, :2]
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    scale = min((w - 2 * pad) / span[0], (h - 2 * pad) / span[1])
    uv = np.empty_like(xy)
    uv[:, 0] = (xy[:, 0] - mn[0]) * scale + pad
    uv[:, 1] = h - ((xy[:, 1] - mn[1]) * scale + pad)

    img = Image.new("RGB", (w, h), (12, 12, 12))
    pix = img.load()
    for (u, v), c in zip(uv.astype(np.int32), rgb):
        if 0 <= u < w and 0 <= v < h:
            pix[int(u), int(v)] = tuple(int(x) for x in c)
    draw = ImageDraw.Draw(img)
    color_text = f" | color={args.color_field}" if args.color_field else ""
    draw.text((20, 15), f"{args.ply.name} | sampled={len(xyz)}{color_text}", fill=(235, 235, 235))
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
