from scripts import build_current_dense_review_index
from scripts import validate_current_dense_patch_state
from scripts.current_mainline_contract import FORBIDDEN_ARTIFACT_SUBSTRINGS, forbidden_artifact_match


def test_forbidden_artifact_contract_is_shared_by_current_mainline_tools() -> None:
    assert build_current_dense_review_index.FORBIDDEN_ARTIFACT_SUBSTRINGS is FORBIDDEN_ARTIFACT_SUBSTRINGS
    assert validate_current_dense_patch_state.REQUIRED_FORBIDDEN_PATTERNS == set(FORBIDDEN_ARTIFACT_SUBSTRINGS)


def test_forbidden_artifact_match_reports_first_matching_substring() -> None:
    forbidden = "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor"

    assert forbidden_artifact_match(f"/tmp/{forbidden}/bad.ply") == forbidden
    assert forbidden_artifact_match("/tmp/current_dense_ok.ply") is None

