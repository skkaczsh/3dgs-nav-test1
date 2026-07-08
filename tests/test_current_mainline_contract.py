import pytest

from scripts import build_current_dense_review_index
from scripts import validate_current_project_architecture
from scripts import validate_current_dense_patch_state
from scripts import validate_geometry_input_contract_usage
from scripts import validate_production_input_guard_usage
from scripts import validate_semantic_contract_usage
from scripts.current_mainline_contract import (
    APPROVED_MAINLINE_RUNNER_PATHS,
    FORBIDDEN_ARTIFACT_SUBSTRINGS,
    FORBIDDEN_PRODUCTION_INPUT_SUBSTRINGS,
    PROTECTED_GEOMETRY_INPUT_CONTRACT_SCRIPT_PATHS,
    PROTECTED_PRODUCTION_GUARD_SCRIPT_PATHS,
    PROTECTED_SEMANTIC_CONTRACT_SCRIPT_PATHS,
    REQUIRED_ACTIVE_BASELINE_IDS,
    REQUIRED_AUTHORITATIVE_POINT_COUNT,
    REQUIRED_AUTHORITATIVE_SOURCE_ID,
    REQUIRED_DERIVED_DENSE_INPUT_ID,
    REQUIRED_DERIVED_VOXEL_COUNT,
    REQUIRED_DENSE_SOURCE_IDS,
    REQUIRED_OPERATOR_TOOL_PATHS,
    REQUIRED_REJECTED_ARTIFACT_IDS,
    REJECTED_ARTIFACT_SUBSTRINGS,
    forbidden_artifact_match,
    forbidden_production_input_match,
    qa_preview_input_match,
    reject_forbidden_production_input,
)


def test_forbidden_artifact_contract_is_shared_by_current_mainline_tools() -> None:
    assert build_current_dense_review_index.FORBIDDEN_ARTIFACT_SUBSTRINGS is REJECTED_ARTIFACT_SUBSTRINGS
    assert validate_current_dense_patch_state.REQUIRED_FORBIDDEN_PATTERNS == set(FORBIDDEN_PRODUCTION_INPUT_SUBSTRINGS)
    assert FORBIDDEN_ARTIFACT_SUBSTRINGS is REJECTED_ARTIFACT_SUBSTRINGS


def test_architecture_contract_is_shared_by_current_mainline_tools() -> None:
    assert validate_current_project_architecture.REQUIRED_ACTIVE_IDS == set(REQUIRED_ACTIVE_BASELINE_IDS)
    assert validate_current_project_architecture.REQUIRED_DENSE_SOURCE_IDS_SET == set(REQUIRED_DENSE_SOURCE_IDS)
    assert validate_current_project_architecture.REQUIRED_REJECTED_IDS == set(REQUIRED_REJECTED_ARTIFACT_IDS)
    assert validate_current_project_architecture.REQUIRED_AUTHORITATIVE_SOURCE_ID == REQUIRED_AUTHORITATIVE_SOURCE_ID
    assert validate_current_project_architecture.REQUIRED_DERIVED_DENSE_INPUT_ID == REQUIRED_DERIVED_DENSE_INPUT_ID


def test_dense_state_operator_tool_contract_is_shared() -> None:
    assert validate_current_dense_patch_state.REQUIRED_OPERATOR_TOOLS == set(REQUIRED_OPERATOR_TOOL_PATHS)
    assert validate_current_dense_patch_state.REQUIRED_AUTHORITATIVE_SOURCE_ID == REQUIRED_AUTHORITATIVE_SOURCE_ID
    assert validate_current_dense_patch_state.REQUIRED_AUTHORITATIVE_POINT_COUNT == REQUIRED_AUTHORITATIVE_POINT_COUNT
    assert validate_current_dense_patch_state.REQUIRED_DERIVED_DENSE_INPUT_ID == REQUIRED_DERIVED_DENSE_INPUT_ID
    assert validate_current_dense_patch_state.REQUIRED_DERIVED_VOXEL_COUNT == REQUIRED_DERIVED_VOXEL_COUNT


def test_protected_script_contracts_are_shared_by_usage_validators() -> None:
    assert validate_production_input_guard_usage.PROTECTED_PRODUCTION_GUARD_SCRIPT_PATHS is (
        PROTECTED_PRODUCTION_GUARD_SCRIPT_PATHS
    )
    assert validate_semantic_contract_usage.PROTECTED_SEMANTIC_CONTRACT_SCRIPT_PATHS is (
        PROTECTED_SEMANTIC_CONTRACT_SCRIPT_PATHS
    )
    assert validate_geometry_input_contract_usage.PROTECTED_GEOMETRY_INPUT_CONTRACT_SCRIPT_PATHS is (
        PROTECTED_GEOMETRY_INPUT_CONTRACT_SCRIPT_PATHS
    )
    approved_python_runners = {
        item
        for item in APPROVED_MAINLINE_RUNNER_PATHS
        if item.endswith(".py") and item != "scripts/cluster_superpoint_graph.py"
    }
    assert approved_python_runners <= set(PROTECTED_PRODUCTION_GUARD_SCRIPT_PATHS)


def test_forbidden_artifact_match_reports_first_matching_substring() -> None:
    forbidden = "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor"

    assert forbidden_artifact_match(f"/tmp/{forbidden}/bad.ply") == forbidden
    assert forbidden_artifact_match("/tmp/frame_object_points_stride10.ply") is None
    assert forbidden_artifact_match("/tmp/current_dense_ok.ply") is None


def test_forbidden_production_input_match_blocks_sparse_viewer_ply() -> None:
    assert forbidden_production_input_match("/tmp/frame_object_points_stride10.ply") == "frame_object_points_stride10.ply"
    assert forbidden_production_input_match("/tmp/geo_patches_random_color_stride3.ply") == "_stride"
    assert forbidden_production_input_match("/tmp/potree_stride10/data/metadata.json") == "_stride"
    assert forbidden_production_input_match("/tmp/current_dense_ok.ply") is None


def test_reject_forbidden_production_input_raises_stable_error() -> None:
    with pytest.raises(ValueError, match="forbidden input path contains frame_object_points_stride10.ply"):
        reject_forbidden_production_input("/tmp/frame_object_points_stride10.ply")


def test_reject_forbidden_production_input_can_explicitly_allow_qa_preview_source() -> None:
    path = "/tmp/geo_patch_objects_v7_structural_multimaterial_stride10.ply"

    assert qa_preview_input_match(path) == "_stride"
    reject_forbidden_production_input(path, allow_qa_preview=True)
