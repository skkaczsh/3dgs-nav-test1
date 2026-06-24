import ast
import re
from pathlib import Path

from scripts.export_frame_target_objects_for_viewer import LABEL_TO_SEMANTIC, SEMANTIC_COLORS
from scripts.semantic_label_contract import (
    LABEL_TO_SEMANTIC as CONTRACT_LABEL_TO_SEMANTIC,
    SEMANTIC_COLORS as CONTRACT_SEMANTIC_COLORS,
    SEMANTIC_TO_LABEL,
)


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "tools" / "semantic_ply_viewer.html"


def _extract_js_object(html: str, const_name: str) -> str:
    match = re.search(rf"const {const_name} = \{{(?P<body>.*?)\n    \}};", html, flags=re.S)
    assert match, f"{const_name} not found"
    return match.group("body")


def _parse_viewer_labels(html: str) -> dict[int, str]:
    body = _extract_js_object(html, "LABELS")
    labels: dict[int, str] = {}
    for key, value in re.findall(r"(\d+):\s*\"([^\"]+)\"", body):
        labels[int(key)] = value
    return labels


def _parse_viewer_colors(html: str) -> dict[str, tuple[int, int, int]]:
    body = _extract_js_object(html, "LABEL_COLORS")
    colors: dict[str, tuple[int, int, int]] = {}
    for key, value in re.findall(r"(\w+):\s*(\[[^\]]+\])", body):
        rgb = ast.literal_eval(value)
        colors[key] = tuple(int(channel) for channel in rgb)
    return colors


def test_export_viewer_label_contract_is_reexported() -> None:
    assert LABEL_TO_SEMANTIC is CONTRACT_LABEL_TO_SEMANTIC
    assert SEMANTIC_COLORS is CONTRACT_SEMANTIC_COLORS


def test_viewer_semantic_ids_match_python_contract() -> None:
    html = VIEWER.read_text(encoding="utf-8")
    viewer_labels = _parse_viewer_labels(html)

    assert viewer_labels == SEMANTIC_TO_LABEL
    assert viewer_labels[8] == "car"
    assert viewer_labels[9] == "railing"


def test_viewer_semantic_colors_match_python_contract() -> None:
    html = VIEWER.read_text(encoding="utf-8")
    viewer_colors = _parse_viewer_colors(html)

    for semantic_id, label in SEMANTIC_TO_LABEL.items():
        assert viewer_colors[label] == SEMANTIC_COLORS[semantic_id]
    assert viewer_colors["ambiguous"] == SEMANTIC_COLORS[0]

