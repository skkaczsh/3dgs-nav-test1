from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_spg_risk import evaluate


def write_run(
    root: Path,
    name: str,
    *,
    accepted_edges: int,
    output_patch_count: int,
    uncertain_fragment_bridge: int = 0,
    fine50: int = 3,
    fine95: int = 0,
) -> Path:
    run = root / name
    (run / "overlap_top1000_fine005").mkdir(parents=True)
    (run / "superpoint_graph_v1_report.json").write_text(
        json.dumps(
            {
                "input_patch_count": 100,
                "output_patch_count": output_patch_count,
                "accepted_edges": accepted_edges,
                "accepted_reasons": {"uncertain_fragment_bridge": uncertain_fragment_bridge},
            }
        ),
        encoding="utf-8",
    )
    (run / "overlap_top1000_fine005" / "bbox_overlap_top1000_report.json").write_text(
        json.dumps({"fine_cell_overlap": {"fine_high_pairs_50": fine50, "fine_high_pairs_95": fine95}}),
        encoding="utf-8",
    )
    return run


def args(baseline: Path, candidate: Path) -> argparse.Namespace:
    return argparse.Namespace(
        baseline_dir=baseline,
        candidate_dir=candidate,
        max_uncertain_fragment_edges=0,
        max_accepted_edge_growth=0.5,
        max_fine50_regression=0,
        max_fine95_regression=0,
    )


def test_spg_risk_compare_passes_equivalent_candidate(tmp_path: Path) -> None:
    baseline = write_run(tmp_path, "baseline", accepted_edges=100, output_patch_count=80)
    candidate = write_run(tmp_path, "candidate", accepted_edges=120, output_patch_count=75)

    report = evaluate(args(baseline, candidate))

    assert report["passed"] is True
    assert report["errors"] == []


def test_spg_risk_compare_rejects_uncertain_bridge_and_overlap_regression(tmp_path: Path) -> None:
    baseline = write_run(tmp_path, "baseline", accepted_edges=100, output_patch_count=80, fine50=3)
    candidate = write_run(
        tmp_path,
        "candidate",
        accepted_edges=160,
        output_patch_count=70,
        uncertain_fragment_bridge=10,
        fine50=4,
    )

    report = evaluate(args(baseline, candidate))

    assert report["passed"] is False
    assert "uncertain_fragment_bridge_exceeded=10>0" in report["errors"]
    assert "accepted_edges_growth=160>150" in report["errors"]
    assert "fine_high_pairs_50_regression=4>3" in report["errors"]
