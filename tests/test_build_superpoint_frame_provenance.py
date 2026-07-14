import numpy as np

from scripts.build_superpoint_frame_provenance import update_top_frames


def test_top_frame_support_keeps_strongest_observations() -> None:
    frames = np.full((1, 2), -1, dtype=np.int32)
    hits = np.zeros((1, 2), dtype=np.int32)
    update_top_frames(frames, hits, 0, 10, 3)
    update_top_frames(frames, hits, 0, 20, 9)
    update_top_frames(frames, hits, 0, 30, 5)
    assert sorted(zip(frames[0].tolist(), hits[0].tolist())) == [(20, 9), (30, 5)]
