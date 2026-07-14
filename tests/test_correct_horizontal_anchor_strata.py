from scripts.correct_horizontal_anchor_strata import correct_anchors


def horizontal(object_id: int, x0: float, x1: float, z: float) -> dict:
    return {
        "object_id": object_id,
        "geometry_type": "horizontal",
        "bbox_min": [x0, 0.0, z - 0.05],
        "bbox_max": [x1, 1.0, z + 0.05],
        "centroid": [(x0 + x1) / 2, 0.5, z],
    }


def anchor(object_id: int, label: str) -> dict:
    return {"object_id": object_id, "anchor_label": label, "propagation_eligible": True}


def test_local_ceiling_stratum_corrects_contradictory_floor() -> None:
    geometry = {1: horizontal(1, 0.0, 1.0, 5.0), 2: horizontal(2, 1.2, 2.2, 5.1)}
    rows, corrections = correct_anchors([anchor(1, "floor"), anchor(2, "ceiling")], geometry, 0.25, 0.2)
    assert rows[0]["anchor_label"] == "ceiling"
    assert rows[0]["strata_correction"]["reference_ceiling_id"] == 2
    assert corrections[0]["object_id"] == 1


def test_distant_ceiling_cannot_relabel_floor() -> None:
    geometry = {1: horizontal(1, 0.0, 1.0, 5.0), 2: horizontal(2, 10.0, 11.0, 5.0)}
    rows, corrections = correct_anchors([anchor(1, "floor"), anchor(2, "ceiling")], geometry, 0.25, 0.2)
    assert rows[0]["anchor_label"] == "floor"
    assert corrections == []
