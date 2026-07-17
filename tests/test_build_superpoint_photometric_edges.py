import numpy as np

from scripts.build_superpoint_photometric_edges import build_rows, sample_view_contrast, summarize_contrasts


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


def test_primary_evidence_wins_over_edge_only_duplicate(tmp_path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:] = 255
    image_path = tmp_path / "view.jpg"
    import cv2
    cv2.imwrite(str(image_path), image)
    rows, _report = build_rows(
        [
            {"object_id": 1, "frame_id": 1, "cam_id": 0, "image_path": str(image_path), "projected_points": 20,
             "projected_uv_samples": [[3.0, 3.0]], "edge_only": False},
            {"object_id": 1, "frame_id": 1, "cam_id": 0, "image_path": str(image_path), "projected_points": 99,
             "projected_uv_samples": [[0.0, 0.0]], "edge_only": True},
            {"object_id": 2, "frame_id": 1, "cam_id": 0, "image_path": str(image_path), "projected_points": 20,
             "projected_uv_samples": [[4.0, 3.0]]},
        ],
        [{"object_a": 1, "object_b": 2}],
        max_pairs=1,
        max_pixel_gap=2.0,
        min_views=1,
    )
    assert rows[0]["contrast_mean"] > 400.0
