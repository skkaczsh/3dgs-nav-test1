import ast
import re
from pathlib import Path

from scripts.export_frame_target_objects_for_viewer import LABEL_TO_SEMANTIC, SEMANTIC_COLORS
from scripts import (
    analyze_residual_absorbability,
    apply_surface_trust_guard_to_ply,
    apply_visual_promotion_geometry_guard,
    build_parking_dataset_manifest,
    build_spatial_partition_objects,
    project_semantic,
    qa_object_voxel_overlap,
    qa_viewer_candidate,
)
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


def test_qa_scripts_share_semantic_label_contract() -> None:
    assert qa_viewer_candidate.LABELS is SEMANTIC_TO_LABEL
    assert qa_viewer_candidate.LABEL_IDS is CONTRACT_LABEL_TO_SEMANTIC
    assert qa_object_voxel_overlap.LABELS is SEMANTIC_TO_LABEL
    assert build_parking_dataset_manifest.SEMANTIC_NAMES is SEMANTIC_TO_LABEL
    assert SEMANTIC_TO_LABEL[8] == "car"
    assert SEMANTIC_TO_LABEL[9] == "railing"


def test_projection_and_partition_scripts_share_semantic_label_contract() -> None:
    assert project_semantic.LABEL_NAMES is SEMANTIC_TO_LABEL
    assert project_semantic.LABEL_COLORS is CONTRACT_SEMANTIC_COLORS
    assert build_spatial_partition_objects.LABELS is SEMANTIC_TO_LABEL
    assert build_spatial_partition_objects.LABEL_TO_SEMANTIC is CONTRACT_LABEL_TO_SEMANTIC
    assert build_spatial_partition_objects.COLORS["car"] == CONTRACT_SEMANTIC_COLORS[8]
    assert build_spatial_partition_objects.COLORS["railing"] == CONTRACT_SEMANTIC_COLORS[9]
    assert build_spatial_partition_objects.COLORS["ground"] == CONTRACT_SEMANTIC_COLORS[3]
    assert analyze_residual_absorbability.SEMANTIC_NAMES is SEMANTIC_TO_LABEL
    assert analyze_residual_absorbability.SEMANTIC_IDS is CONTRACT_LABEL_TO_SEMANTIC
    assert analyze_residual_absorbability.LABEL_COLORS is CONTRACT_SEMANTIC_COLORS


def test_guard_scripts_share_semantic_label_contract() -> None:
    assert apply_surface_trust_guard_to_ply.LABEL_TO_SEMANTIC is CONTRACT_LABEL_TO_SEMANTIC
    assert apply_surface_trust_guard_to_ply.SEMANTIC_TO_LABEL is SEMANTIC_TO_LABEL
    assert apply_surface_trust_guard_to_ply.LABEL_COLORS["floor"] == CONTRACT_SEMANTIC_COLORS[3]
    assert apply_visual_promotion_geometry_guard.LABEL_TO_SEMANTIC is CONTRACT_LABEL_TO_SEMANTIC


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
