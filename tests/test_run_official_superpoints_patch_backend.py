from pathlib import Path

import pytest

from scripts.run_official_superpoints_patch import configure_official_spg_backend


def test_backend_configuration_rejects_missing_partition(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="partition directory is missing"):
        configure_official_spg_backend(tmp_path)


def test_backend_configuration_adds_partition_to_import_path(tmp_path) -> None:
    partition = tmp_path / "partition"
    partition.mkdir()
    assert configure_official_spg_backend(tmp_path) == partition
