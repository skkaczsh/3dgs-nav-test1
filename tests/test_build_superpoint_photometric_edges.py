import numpy as np

from scripts.build_superpoint_photometric_edges import sample_view_contrast, summarize_contrasts


def test_nearest_projected_samples_measure_cross_object_color_contrast() -> None:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, 6:] = (255, 255, 255)
    contrast = sample_view_contrast(
        image,
        np.asarray([[5.0, 5.0], [5.0, 6.0]], dtype=np.float32),
        np.asarray([[6.0, 5.0], [6.0, 6.0]], dtype=np.float32),
        max_pairs=4,
        max_pixel_gap=2.0,
    )
    assert contrast is not None
    assert contrast > 400.0


def test_single_view_contrast_does_not_cut_an_edge() -> None:
    summary = summarize_contrasts([200.0], sigma=45.0, min_views=2)
    assert summary["contrast_lcb"] == 0.0
    assert summary["photometric_affinity"] == 1.0


def test_repeated_high_contrast_lowers_affinity() -> None:
    summary = summarize_contrasts([180.0, 200.0, 220.0], sigma=45.0, min_views=2)
    assert summary["contrast_lcb"] > 100.0
    assert summary["photometric_affinity"] < 0.05
