import os
from pathlib import Path

import pytest

from scripts.run_official_superpoints_patch import (
    configure_official_spg_backend,
    configure_omp_threads,
    resolve_reg_strength,
)


def test_backend_configuration_rejects_missing_partition(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="partition directory is missing"):
        configure_official_spg_backend(tmp_path)


def test_backend_configuration_adds_partition_to_import_path(tmp_path) -> None:
    partition = tmp_path / "partition"
    partition.mkdir()
    assert configure_official_spg_backend(tmp_path) == partition


def test_production_defaults_to_single_deterministic_omp_thread(monkeypatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    configure_omp_threads(1, False)
    assert os.environ["OMP_NUM_THREADS"] == "1"
    with pytest.raises(ValueError, match="nondeterministic"):
        configure_omp_threads(2, False)


def test_label_rematerialization_requires_partition_regularization() -> None:
    assert resolve_reg_strength(None, False) == 0.1
    assert resolve_reg_strength(0.05, True) == 0.05
    with pytest.raises(ValueError, match="requires explicit"):
        resolve_reg_strength(None, True)
