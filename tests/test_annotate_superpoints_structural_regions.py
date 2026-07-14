import numpy as np

from scripts.annotate_superpoints_structural_regions import aggregate_region_votes


def test_region_votes_keep_object_ownership_exclusive() -> None:
    votes = aggregate_region_votes(
        np.array([0, 0, 1, 1, 1], dtype=np.int64),
        np.array([1, 2, 2, 2, 4], dtype=np.uint8),
    )
    assert votes.tolist() == [[0, 1, 1, 0, 0], [0, 0, 2, 0, 1]]
