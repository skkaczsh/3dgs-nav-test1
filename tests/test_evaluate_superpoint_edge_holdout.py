from scripts.evaluate_superpoint_edge_holdout import consensus_labels, evaluate


def test_consensus_requires_every_review_to_agree() -> None:
    assert consensus_labels([{1: "wall", 2: "car"}, {1: "wall", 2: "floor"}]) == {1: "wall"}


def test_holdout_groups_strong_cut_and_same_mask_edges() -> None:
    edges = [
        {"object_a": 1, "object_b": 2, "sam2_affinity": 0.6, "view_count": 3},
        {"object_a": 3, "object_b": 4, "sam2_affinity": 1.0, "same_mask_lcb": 0.7, "view_count": 3},
    ]
    report = evaluate(edges, {1: "wall", 2: "car", 3: "tree_or_shrub", 4: "tree_or_shrub"}, 0.8, 0.5)
    assert report["strong_separation"]["same_label_ratio"] == 0.0
    assert report["strong_same_mask"]["same_label_ratio"] == 1.0
