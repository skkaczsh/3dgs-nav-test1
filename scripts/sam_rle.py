"""Canonical SAM mask RLE codec supporting COCO compressed and legacy counts."""

from __future__ import annotations

from typing import Any

import numpy as np


def decode_coco_counts(value: str) -> list[int]:
    counts: list[int] = []
    index = 0
    while index < len(value):
        number = 0
        shift = 0
        while True:
            byte = ord(value[index]) - 48
            index += 1
            number |= (byte & 0x1F) << shift
            shift += 5
            if not (byte & 0x20):
                if byte & 0x10:
                    number |= -1 << shift
                break
        if len(counts) > 2:
            number += counts[-2]
        counts.append(number)
    return counts


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    h, w = [int(value) for value in rle["size"]]
    raw_counts = rle["counts"]
    counts = decode_coco_counts(raw_counts) if isinstance(raw_counts, str) else [int(value) for value in raw_counts]
    flat = np.empty(h * w, dtype=bool)
    index = 0
    value = False
    for count in counts:
        next_index = min(index + count, flat.size)
        flat[index:next_index] = value
        index = next_index
        value = not value
    if index < flat.size:
        flat[index:] = False
    return flat.reshape(w, h).T


def encode_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    flat = mask.T.reshape(-1)
    counts: list[int] = []
    value = False
    run = 0
    for pixel in flat:
        pixel = bool(pixel)
        if pixel == value:
            run += 1
        else:
            counts.append(run)
            run = 1
            value = pixel
    counts.append(run)
    return {"size": [int(h), int(w)], "counts": counts}
