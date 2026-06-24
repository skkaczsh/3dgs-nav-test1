from scripts import build_current_dense_review_index
from scripts import validate_current_project_architecture
from scripts import validate_current_dense_patch_state
from scripts.current_mainline_contract import (
    FORBIDDEN_ARTIFACT_SUBSTRINGS,
    FORBIDDEN_PRODUCTION_INPUT_SUBSTRINGS,
    REQUIRED_ACTIVE_BASELINE_IDS,
    REQUIRED_REJECTED_ARTIFACT_IDS,
    REJECTED_ARTIFACT_SUBSTRINGS,
    forbidden_artifact_match,
    forbidden_production_input_match,
)


def test_forbidden_artifact_contract_is_shared_by_current_mainline_tools() -> None:
    assert build_current_dense_review_index.FORBIDDEN_ARTIFACT_SUBSTRINGS is REJECTED_ARTIFACT_SUBSTRINGS
    assert validate_current_dense_patch_state.REQUIRED_FORBIDDEN_PATTERNS == set(FORBIDDEN_PRODUCTION_INPUT_SUBSTRINGS)
    assert FORBIDDEN_ARTIFACT_SUBSTRINGS is REJECTED_ARTIFACT_SUBSTRINGS


def test_architecture_contract_is_shared_by_current_mainline_tools() -> None:
    assert validate_current_project_architecture.REQUIRED_ACTIVE_IDS == set(REQUIRED_ACTIVE_BASELINE_IDS)
    assert validate_current_project_architecture.REQUIRED_REJECTED_IDS == set(REQUIRED_REJECTED_ARTIFACT_IDS)


def test_forbidden_artifact_match_reports_first_matching_substring() -> None:
    forbidden = "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor"

    assert forbidden_artifact_match(f"/tmp/{forbidden}/bad.ply") == forbidden
    assert forbidden_artifact_match("/tmp/frame_object_points_stride10.ply") is None
    assert forbidden_artifact_match("/tmp/current_dense_ok.ply") is None


def test_forbidden_production_input_match_blocks_sparse_viewer_ply() -> None:
    assert forbidden_production_input_match("/tmp/frame_object_points_stride10.ply") == "frame_object_points_stride10.ply"
    assert forbidden_production_input_match("/tmp/current_dense_ok.ply") is None
