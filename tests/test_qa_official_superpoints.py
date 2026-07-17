from __future__ import annotations

import numpy as np
import pytest

from scripts.qa_official_superpoints import objects_agree, ownership_report, sha256_file


def test_ownership_report_requires_same_order_and_contiguous_ids() -> None:
    report = ownership_report(np.asarray([0, 0, 1, 2, 2], dtype=np.uint32), 5, 2)
    assert report["labels_contiguous"] is True
    assert report["small_superpoints"] == 1
    with pytest.raises(ValueError, match="label count"):
        ownership_report(np.asarray([0, 1], dtype=np.uint32), 3, 2)


def test_object_rows_must_match_superpoint_counts() -> None:
    labels = np.asarray([0, 0, 1], dtype=np.uint32)
    assert objects_agree([{"object_id": 0, "count": 2}, {"object_id": 1, "count": 1}], labels)["exact"] is True
    result = objects_agree([{"object_id": 0, "count": 1}], labels)
    assert result["exact"] is False
    assert result["missing_object_rows"] == 1
    assert result["count_mismatches"] == 1


def test_sha256_file_is_stable(tmp_path) -> None:
    path = tmp_path / "source.ply"
    path.write_bytes(b"dense point rows")
    assert sha256_file(path) == "8bc25e70866811653f46a534939c58f168bac9dd08e3822f4a5570dde9975df9"
