import json

import numpy as np

from scripts.build_superpoint_frame_provenance import read_frame_filter, update_top_frames


def test_top_frame_support_keeps_strongest_observations() -> None:
    frames = np.full((1, 2), -1, dtype=np.int32)
    hits = np.zeros((1, 2), dtype=np.int32)
    update_top_frames(frames, hits, 0, 10, 3)
    update_top_frames(frames, hits, 0, 20, 9)
    update_top_frames(frames, hits, 0, 30, 5)
    assert sorted(zip(frames[0].tolist(), hits[0].tolist())) == [(20, 9), (30, 5)]


def test_frame_filter_reads_json_or_plain_ids(tmp_path) -> None:
    json_path = tmp_path / "frames.json"
    json_path.write_text(json.dumps([10, 20, 10]), encoding="utf-8")
    plain_path = tmp_path / "frames.txt"
    plain_path.write_text("30, 40\n50", encoding="utf-8")
    assert read_frame_filter(json_path) == {10, 20}
    assert read_frame_filter(plain_path) == {30, 40, 50}
