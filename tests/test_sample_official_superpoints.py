from scripts.sample_official_superpoints import select_rows


def test_select_rows_stratifies_only_supported_objects() -> None:
    rows = [
        {"object_id": 1, "geometry_type": "horizontal"},
        {"object_id": 2, "geometry_type": "horizontal"},
        {"object_id": 3, "geometry_type": "vertical"},
    ]
    selected = select_rows(rows, {1, 3}, per_geometry=2, min_object_points=0, seed=17)
    assert [row["object_id"] for row in selected] == [1, 3]


def test_select_rows_can_require_a_reviewable_point_count() -> None:
    rows = [
        {"object_id": 1, "geometry_type": "horizontal", "count": 40},
        {"object_id": 2, "geometry_type": "horizontal", "count": 500},
    ]
    selected = select_rows(rows, {1, 2}, per_geometry=2, min_object_points=100, seed=17)
    assert [row["object_id"] for row in selected] == [2]
