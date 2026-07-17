from scripts.build_superpoint_contact_view_evidence import build_neighbor_requests


def test_requests_expand_only_direct_contact_neighbors_at_anchor_views() -> None:
    anchors = [
        {"object_id": 10, "frame_id": 100, "cam_id": 1},
        {"object_id": 11, "frame_id": 120, "cam_id": 2},
    ]
    contacts = [
        {"object_a": 10, "object_b": 11},
        {"object_a": 10, "object_b": 12},
        {"object_a": 11, "object_b": 13},
        {"object_a": 12, "object_b": 13},
    ]
    requests = build_neighbor_requests(anchors, contacts, {10, 11, 12, 13})

    assert requests[(11, 100, 1)] == {10}
    assert requests[(12, 100, 1)] == {10}
    assert requests[(10, 120, 2)] == {11}
    assert requests[(13, 120, 2)] == {11}
    assert (13, 100, 1) not in requests
