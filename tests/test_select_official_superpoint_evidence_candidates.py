from scripts.select_official_superpoint_evidence_candidates import evenly_spaced_indices, select_candidates


def test_evenly_spaced_indices_cover_scale_range() -> None:
    assert evenly_spaced_indices(5, 3) == [0, 2, 4]


def test_selection_is_deterministic_and_geometry_stratified() -> None:
    rows = [
        {"object_id": 1, "geometry_type": "horizontal", "count": 10},
        {"object_id": 2, "geometry_type": "horizontal", "count": 100},
        {"object_id": 3, "geometry_type": "horizontal", "count": 1000},
        {"object_id": 4, "geometry_type": "vertical", "count": 200},
    ]
    selected = select_candidates(rows, per_geometry=2, min_points=100)
    assert [row["object_id"] for row in selected] == [2, 3, 4]
    assert {row["geometry_type"] for row in selected} == {"horizontal", "vertical"}
    assert all(row["evidence_candidate_policy"] == "geometry_log_scale_stratified/v1" for row in selected)


def test_coverage_budget_adds_largest_unselected_cells_without_removing_strata() -> None:
    rows = [
        {"object_id": 1, "geometry_type": "horizontal", "count": 100},
        {"object_id": 2, "geometry_type": "horizontal", "count": 900},
        {"object_id": 3, "geometry_type": "horizontal", "count": 1_000},
        {"object_id": 4, "geometry_type": "vertical", "count": 800},
    ]

    selected = select_candidates(rows, per_geometry=2, min_points=100, coverage_budget=1)
    by_id = {row["object_id"]: row for row in selected}

    assert set(by_id) == {1, 2, 3, 4}
    assert by_id[1]["evidence_candidate_policy"] == "geometry_log_scale_stratified/v1"
    assert by_id[2]["evidence_candidate_policy"] == "point_mass_coverage/v1"
    assert by_id[3]["evidence_candidate_policy"] == "geometry_log_scale_stratified/v1"
    assert by_id[4]["evidence_candidate_policy"] == "geometry_log_scale_stratified/v1"
