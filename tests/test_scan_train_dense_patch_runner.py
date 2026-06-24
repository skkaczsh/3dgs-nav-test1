from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_scan_train_dense_patch_object_refinement_v7.sh"


def test_scan_train_dense_patch_runner_exposes_recall_and_acceptance_knobs() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'MIN_PATCH_VOXELS="${MIN_PATCH_VOXELS:-40}"' in text
    assert 'MIN_SHARED_EDGES="${MIN_SHARED_EDGES:-3}"' in text
    assert 'ACCEPT_MIN_SCORE="${ACCEPT_MIN_SCORE:-0.80}"' in text
    assert 'ATTACHMENT_MIN_SCORE="${ATTACHMENT_MIN_SCORE:-0.82}"' in text
    assert '--min-patch-voxels "${MIN_PATCH_VOXELS}"' in text
    assert '--accept-min-score "${ACCEPT_MIN_SCORE}"' in text
    assert '--attachment-min-score "${ATTACHMENT_MIN_SCORE}"' in text
