import numpy as np

from scripts.build_superpoint_sam2_comask_edges import mask_support, summarize_views


def test_compact_shared_mask_supports_same_edge() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:5] = True
    same, separate = mask_support(
        [mask], np.asarray([[2.0, 2.0], [3.0, 3.0]]), np.asarray([[4.0, 2.0], [4.0, 3.0]]), 0.8,
    )
    assert same > 0.7
    assert separate == 0.0


def test_distinct_compact_masks_supply_separation_evidence() -> None:
    left = np.zeros((6, 6), dtype=bool)
    right = np.zeros((6, 6), dtype=bool)
    left[2:4, 1:3] = True
    right[2:4, 3:5] = True
    same, separate = mask_support(
        [left, right], np.asarray([[1.0, 2.0], [2.0, 3.0]]), np.asarray([[3.0, 2.0], [4.0, 3.0]]), 0.8,
    )
    assert same == 0.0
    assert separate > 0.7


def test_single_view_is_neutral_and_repeated_separation_reduces_affinity() -> None:
    assert summarize_views([(0.0, 0.9)], min_views=2)["sam2_affinity"] == 1.0
    assert summarize_views([(0.0, 0.9), (0.0, 0.8)], min_views=2)["sam2_affinity"] < 0.3
