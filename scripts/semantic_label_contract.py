"""Shared semantic label ids and display colors for viewer-ready artifacts."""

from __future__ import annotations


SEMANTIC_TO_LABEL: dict[int, str] = {
    0: "unknown",
    1: "other",
    2: "wall",
    3: "floor",
    4: "ceiling",
    5: "grass",
    6: "tree",
    7: "person",
    8: "car",
    9: "railing",
    10: "building",
    11: "sky",
    12: "road",
    13: "water",
    14: "furniture",
    15: "pipe",
    16: "equipment",
    17: "fine_candidate",
    18: "stair",
    19: "indoor_floor",
    20: "roof",
    255: "ignore",
}

LABEL_TO_SEMANTIC: dict[str, int] = {label: semantic for semantic, label in SEMANTIC_TO_LABEL.items()}
LABEL_TO_SEMANTIC["ground"] = LABEL_TO_SEMANTIC["floor"]
LABEL_TO_SEMANTIC["ambiguous"] = LABEL_TO_SEMANTIC["unknown"]

SEMANTIC_COLORS: dict[int, tuple[int, int, int]] = {
    0: (150, 150, 150),
    1: (180, 180, 180),
    2: (120, 150, 180),
    3: (196, 168, 112),
    4: (170, 170, 210),
    5: (80, 160, 80),
    6: (50, 130, 70),
    7: (230, 80, 80),
    8: (235, 90, 80),
    9: (240, 210, 60),
    10: (145, 145, 160),
    11: (70, 150, 220),
    12: (120, 120, 120),
    13: (50, 120, 200),
    14: (180, 100, 200),
    15: (220, 160, 60),
    16: (210, 90, 210),
    17: (245, 150, 40),
    18: (245, 125, 60),
    19: (105, 180, 210),
    20: (165, 145, 210),
    255: (40, 40, 40),
}

